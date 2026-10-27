"""Self-healing: the LLM (.env model) repairs a test case whose failure was diagnosed as TEST_DEFECT.

Guard rails (deterministic, fed back to the model via ModelRetry):
- the healed case passes the normal IR validation;
- it tests the same thing: same case type, same requirements, same covered endpoints;
- every expected HTTP status is documented for its endpoint in the OpenAPI contract
  (healing must never adapt a test to undocumented / buggy behaviour);
- it actually changes something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.domain.ir import (
    BODY_ERROR_STATUSES,
    CaseDraft,
    ValidationContext,
    covered_endpoints,
    http_steps,
    missing_required,
    normalize,
    resolve_endpoint,
    step_statuses,
    validate_case,
)
from spec_run_graph.domain.recipes import RecipeIR, expand_recipes, recipe_brief
from spec_run_graph.llm import build_model, output_spec, run_structured

INSTRUCTIONS = """\
You repair ONE automated API test case whose failure was diagnosed as a TEST DEFECT: the API behaves
according to its contract, but the test is wrong (wrong expectation, wrong data, wrong path/header in an
assertion, missing required field, wrong step order, missing captured variable, unfounded assumption).

Return the COMPLETE corrected test case in the same format (title, type, priority, tags, actor, coverage,
preconditions, variables, parameters, steps, expected_results, cleanup) plus a short change_summary.
Rules:
- Fix the root cause described by the diagnosis; keep everything else unchanged.
- Keep testing the same behaviour: same case type, same requirement_refs, every original endpoint
  (you may add setup steps on other endpoints, e.g. to create a dedicated user or product).
- Expected status codes must be documented for the endpoint in the OpenAPI contract given below.
- Never weaken a test just to make it pass: do not delete meaningful checks, prefer correcting them.
- Header checks use lower-case names: response.headers.x-request-id
- Assert only fields documented in the response schema; never invent fields such as error.field.
- Created entities need per-run unique usernames/emails: variables {x: "qa_{{ $faker.uuid }}"} + {{ $var.x }}.
- Ids of seeded data are never env variables: look them up in an earlier step (GET the list, extract).
- Balance/stock-dependent flows: create a dedicated user with a known balance (admin API), log in as it,
  extract the token and use header Authorization: Bearer {{ $extract.token }} - shared seed accounts get drained.
- Assert/extract only the paths listed as "assertable (guaranteed) fields" (optional ones such as requestId,
  error.details or pagination may be omitted by the API); `not exists` checks are fine.
- Requests expected to succeed must send every field listed under "a successful request MUST send"; so must
  negative steps that test something other than body validation (e.g. wrong password -> 401).
- State-changing requests (orders, stock) use entities the case creates, not items picked from shared lists.
- Never assume a pristine database (balances, stock, counts, list order change during the suite): extract a
  baseline first and compare relative to it; select list items by attribute: response.body.data[model='X'].id.
- Check grammar: "<path> <op> <literal>", paths response.status | response.headers.<h> | response.body.<path>,
  list filters allowed: response.body.data[model='X'].id.
- Templates allowed: {{ $faker.x }} {{ $fixture.x }} {{ $env.X }} {{ $extract.x }} {{ $param.x }} {{ $var.x }}.
If the evidence shows the API (not the test) is wrong after all, do not change the case: set
unhealable_reason and return the case unchanged."""


class HealedCase(BaseModel):
    case: CaseDraft
    change_summary: str = Field(description="what was changed and why, 1-3 sentences")
    unhealable_reason: str | None = Field(
        None, description="set ONLY if the failure is not a test defect after all (then the case is not changed)"
    )


@dataclass
class HealDeps:
    original: CaseDraft
    ctx: ValidationContext
    documented_statuses: dict[str, set[int]]  # endpoint key -> documented response codes
    recipes: dict[str, RecipeIR] = field(default_factory=dict)
    last_errors: list[str] = field(default_factory=list)
    attempts: int = 0


def _expected_statuses(case: CaseDraft, step_index: int) -> list[int]:
    step, _ = http_steps(case)[step_index]
    return step_statuses(step, case.parameters)


def _required_field_negatives(case: CaseDraft, ctx: ValidationContext) -> set[tuple[str, frozenset[str]]]:
    """(endpoint, omitted required fields) of steps that expect a 4xx for a request missing required fields."""
    out = set()
    for i, (step, is_cleanup) in enumerate(http_steps(case)):
        if is_cleanup or step.type != "http" or not step.endpoint or not ctx.request_schemas:
            continue
        key = resolve_endpoint(step.endpoint, ctx.endpoint_paths)
        schema = ctx.request_schemas.get(key or "")
        statuses = _expected_statuses(case, i)
        if schema and statuses and all(s in BODY_ERROR_STATUSES for s in statuses):
            missing = missing_required(schema, step.request.payload)
            if missing:
                out.add((key, frozenset(missing)))
    return out


def guard_errors(healed: CaseDraft, deps: HealDeps) -> list[str]:
    errors = validate_case(healed, deps.ctx)
    orig = deps.original
    if healed.type != orig.type:
        errors.append(f"keep the case type {orig.type!r}")
    if sorted(healed.coverage.requirement_refs) != sorted(orig.coverage.requirement_refs):
        errors.append(f"keep requirement_refs {sorted(orig.coverage.requirement_refs)}")
    before = set(covered_endpoints(orig, deps.ctx.endpoint_paths))
    after = set(covered_endpoints(healed, deps.ctx.endpoint_paths))
    if not before <= after:  # setup steps may add endpoints (e.g. create a dedicated user); nothing may be dropped
        errors.append(f"keep testing all original endpoints; missing {sorted(before - after)}")
    kept = _required_field_negatives(healed, deps.ctx)
    for key, missing in sorted(_required_field_negatives(orig, deps.ctx) - kept):
        errors.append(
            f"keep the negative step that sends {key} without the contract-required {sorted(missing)} and expects "
            "a 4xx: if the API accepts it, that is an application bug the test must keep detecting"
        )
    for i, (step, is_cleanup) in enumerate(http_steps(healed)):
        if is_cleanup or step.type != "http" or not step.endpoint:
            continue
        key = resolve_endpoint(step.endpoint, deps.ctx.endpoint_paths)
        documented = deps.documented_statuses.get(key or "")
        if not documented:
            continue
        for status in _expected_statuses(healed, i):
            if status not in documented:
                errors.append(
                    f"step {i + 1} ({step.name}): expected status {status} is not documented for {key} "
                    f"(documented: {sorted(documented)}); healing must not adapt tests to undocumented behaviour"
                )
    if healed.model_dump(exclude={"title"}) == orig.model_dump(exclude={"title"}):
        errors.append("the case is unchanged; apply the fix described by the diagnosis")
    return errors


def build_agent(output: Any = None) -> Agent[HealDeps, HealedCase]:
    agent: Agent[HealDeps, HealedCase] = Agent(
        output_type=output or output_spec(HealedCase),
        instructions=INSTRUCTIONS,
        deps_type=HealDeps,
        retries=get_settings().srg_llm_retries,
        name="test-healer",
    )

    @agent.output_validator
    def check(ctx: RunContext[HealDeps], out: HealedCase) -> HealedCase:
        ctx.deps.attempts += 1
        if out.unhealable_reason:
            return out
        expanded, errors = expand_recipes(out.case, ctx.deps.recipes)
        if errors:
            ctx.deps.last_errors = errors
            raise ModelRetry("The healed case is not acceptable. Fix these problems:\n- " + "\n- ".join(errors))
        healed = normalize(expanded)
        errors = guard_errors(healed, ctx.deps)
        ctx.deps.last_errors = errors
        if errors:
            raise ModelRetry("The healed case is not acceptable. Fix these problems:\n- " + "\n- ".join(errors))
        return out.model_copy(update={"case": healed})

    return agent


def build_prompt(
    case: CaseDraft,
    diagnoses: list[dict[str, Any]],
    exchange: str,
    contracts: str,
    requirements: list[dict[str, Any]],
    previous_attempt: str | None = None,
    recipes: dict[str, RecipeIR] | None = None,
    lessons: list[str] | None = None,
) -> str:
    parts = [
        "Test case to repair (IR):\n"
        + yaml.safe_dump(case.model_dump(mode="json", exclude_defaults=True), sort_keys=False, allow_unicode=True),
        "Diagnosis of the failing rows:\n" + json.dumps(diagnoses, default=str)[:6000],
        "Recorded HTTP exchange of the failing run:\n" + exchange[:10000],
        "OpenAPI contracts (the case's endpoints first, then the others usable as setup steps):\n" + contracts,
        "Requirements:\n" + json.dumps(requirements, default=str)[:4000],
    ]
    if recipes:
        parts.append(
            "Setup recipes (verified; a step {type: recipe, recipe, inputs, alias} expands into their steps):\n"
            + json.dumps([recipe_brief(r) for r in recipes.values()])
        )
    if lessons:
        parts.append("Lessons learned from earlier test defects:\n- " + "\n- ".join(lessons))
    if previous_attempt:
        parts.append("Your previous fix was applied but the test still failed:\n" + previous_attempt[:3000])
    return "\n\n".join(parts)


async def heal(prompt: str, deps: HealDeps, model: Model | None = None) -> HealedCase:
    result = await run_structured(build_agent, prompt, HealedCase, deps=deps, model=model or build_model())
    return result.output

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
    PARAM_TEMPLATE_RE,
    CaseDraft,
    ValidationContext,
    covered_endpoints,
    http_steps,
    normalize,
    resolve_endpoint,
    validate_case,
)
from spec_run_graph.llm import build_model, output_spec, run_structured

INSTRUCTIONS = """\
You repair ONE automated API test case whose failure was diagnosed as a TEST DEFECT: the API behaves
according to its contract, but the test is wrong (wrong expectation, wrong data, wrong path/header in an
assertion, missing required field, wrong step order, missing captured variable, unfounded assumption).

Return the COMPLETE corrected test case in the same format (title, type, priority, tags, actor, coverage,
preconditions, variables, parameters, steps, expected_results, cleanup) plus a short change_summary.
Rules:
- Fix the root cause described by the diagnosis; keep everything else unchanged.
- Keep testing the same behaviour: same case type, same requirement_refs, same endpoints.
- Expected status codes must be documented for the endpoint in the OpenAPI contract given below.
- Never weaken a test just to make it pass: do not delete meaningful checks, prefer correcting them.
- Header checks use lower-case names: response.headers.x-request-id
- Check grammar: "<path> <op> <literal>", paths response.status | response.headers.<h> | response.body.<path>.
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
    last_errors: list[str] = field(default_factory=list)
    attempts: int = 0


def _expected_statuses(case: CaseDraft, step_index: int) -> list[int]:
    step, _ = http_steps(case)[step_index]
    es = step.expected_status
    if es is None:
        return []
    if isinstance(es, int) or str(es).strip().isdigit():
        return [int(es)]
    m = PARAM_TEMPLATE_RE.match(str(es).strip())
    if m:
        return [int(row[m.group(1)]) for row in case.parameters if str(row.get(m.group(1), "")).isdigit()]
    return []


def guard_errors(healed: CaseDraft, deps: HealDeps) -> list[str]:
    errors = validate_case(healed, deps.ctx)
    orig = deps.original
    if healed.type != orig.type:
        errors.append(f"keep the case type {orig.type!r}")
    if sorted(healed.coverage.requirement_refs) != sorted(orig.coverage.requirement_refs):
        errors.append(f"keep requirement_refs {sorted(orig.coverage.requirement_refs)}")
    before = set(covered_endpoints(orig, deps.ctx.endpoint_paths))
    after = set(covered_endpoints(healed, deps.ctx.endpoint_paths))
    if before != after:
        errors.append(f"keep testing the same endpoints {sorted(before)} (got {sorted(after)})")
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
        healed = normalize(out.case)
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
    contracts: dict[str, Any],
    requirements: list[dict[str, Any]],
    previous_attempt: str | None = None,
) -> str:
    parts = [
        "Test case to repair (IR):\n"
        + yaml.safe_dump(case.model_dump(mode="json", exclude_defaults=True), sort_keys=False, allow_unicode=True),
        "Diagnosis of the failing rows:\n" + json.dumps(diagnoses, default=str)[:6000],
        "Recorded HTTP exchange of the failing run:\n" + exchange[:10000],
        "OpenAPI contract of the endpoints:\n" + json.dumps(contracts, default=str)[:10000],
        "Requirements:\n" + json.dumps(requirements, default=str)[:4000],
    ]
    if previous_attempt:
        parts.append("Your previous fix was applied but the test still failed:\n" + previous_attempt[:3000])
    return "\n\n".join(parts)


async def heal(prompt: str, deps: HealDeps, model: Model | None = None) -> HealedCase:
    result = await run_structured(build_agent, prompt, HealedCase, deps=deps, model=model or build_model())
    return result.output

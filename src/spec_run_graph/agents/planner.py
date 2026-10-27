"""Test Planner: designs API test cases (IR) for one module chunk. Ids are assigned later by code."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.domain.analysis import ModuleProposal, SpecAnalysis
from spec_run_graph.domain.ir import CaseDraft, ValidationContext, declared_env_vars, normalize, validate_case
from spec_run_graph.domain.recipes import RecipeIR, expand_recipes, recipe_brief
from spec_run_graph.llm import build_model, output_spec, run_structured
from spec_run_graph.specs.models import EndpointDraft
from spec_run_graph.specs.openapi import endpoint_digest

INSTRUCTIONS = """\
You are a senior API test designer. Design test cases for ONE module in the YAML/JSON test-case
format below. Return only new cases (never repeat the "already covered" ones).

Test case fields: title, type (positive|negative|boundary|security|rbac), priority (P0|P1|P2), tags,
actor (one of the module actors), coverage.requirement_refs (requirement ids), preconditions (text),
variables, parameters, steps, expected_results, cleanup. Do NOT set an id - ids are assigned by the system.

Step fields: name, type (http|delay|recipe), actor (only when the step runs as a different module actor than the
case, e.g. a "user-role token" step in an admin case), endpoint ("METHOD /openapi/path", path parameters may be filled
inline with templates, e.g. "GET /v1/orders/{{ $extract.order_id }}", or via request.path_params),
request {headers, query, path_params, payload}, expected_status (int or "{{ $param.<key> }}"),
validate_schema (true to check the response against the OpenAPI schema), extract {var: path},
assertions [{check, message}], expected_error {code} (sugar for an error.code check), delay_ms (delay steps only).

Context variables (the ONLY allowed templates):
  {{ $faker.uuid }} {{ $faker.email }} {{ $faker.first_name }} ...  generated data
  {{ $fixture.<name> }}  a module fixture      {{ $env.<KEY> }}     environment variable
  {{ $extract.<var> }}   captured in an EARLIER step of the same case
  {{ $param.<key> }}     value of the current parameter row
  {{ $var.<name> }}      a value from the case `variables`

Assertion check grammar: "<path> <op> <literal>"
  path: response.status | response.headers.<name> | response.body.<json.path> (e.g. response.body.data.items[0].qty,
        or a list filter response.body.data[model='Galaxy S24'].price)
  op: == != > >= < <= in contains matches exists "not exists"
  literal: 'string' | number | true | false | null | ['a', 'b'] | a template
  examples: "response.body.data.status == 'DRAFT'", "response.body.data.id exists"
Extract paths use the same path syntax (response.body.data.id).

Rules:
- Data-driven cases: `parameters` is a list of rows; each row MUST have a kebab-case "name"
  (e.g. "zero-quantity") plus the keys used as {{ $param.<key> }}. Max 26 rows.
- Multi-step workflows are fine (create -> act -> verify). For setup, prefer a listed recipe:
  {"name": "Dedicated buyer", "type": "recipe", "recipe": "<name>", "inputs": {...}, "alias": "buyer"}
  Its outputs become {{ $extract.<alias>_<output> }} (or {{ $extract.<output> }} without alias) and its cleanup is
  added automatically. Otherwise setup actions are ordinary steps.
- Entities created by a test need per-run unique values for unique fields (username, email): define
  variables: {new_username: "qa_{{ $faker.uuid }}"} and use {{ $var.new_username }} - deletes may be soft,
  so a fixed name already exists on the next run.
- Never assume a pristine database: other tests create, buy and delete data, so balances, stock, counts and
  list order change. Capture a baseline in an earlier step (extract the balance/count) and assert relative to it;
  select list items by attribute with a filter path, e.g. response.body.data[model='Galaxy S24'].id, never by a
  fixed index like data[1].
- Flows that depend on a balance, stock or other per-account state must not use the shared seed accounts
  (other tests spend their money): create a dedicated user with a known balance via the admin API, log in as
  that user (extract the token) and send its requests with header Authorization: Bearer {{ $extract.token }}.
- Assert/extract only the paths listed as "assertable (guaranteed) fields" for the expected status (list items
  may be selected by attribute instead of [0]: data[model='X'].id). Optional fields such as requestId,
  error.details or pagination may be omitted by the API; `not exists` checks are fine.
- Steps expected to succeed must send every field listed under "a successful request MUST send" - also on
  setup steps against other modules' endpoints (e.g. role and balance when creating a user).
- Assert only fields that exist in the documented response schema (e.g. error.code/error.message, not
  invented fields); for lists use the documented item fields (e.g. data[0].id, not data[0].product_id).
- Put teardown of created data into `cleanup` (same step format; failures there are ignored).
- Authentication only via actors (case `actor` / step `actor`); never literal tokens.
- {{ $env.X }} is ONLY for the declared environment variables listed in the input - never for test data.
  Ids and other values created by the system (seed product ids, user ids, order ids) are looked up at run time:
  add an earlier step (e.g. GET /api/.../products) and `extract` the value, then use {{ $extract.x }}.
- Cover every requirement of the module with positive, negative/boundary, security (missing or invalid
  token -> 401) and, where several actors exist, rbac cases. Prefer few precise cases over many vague ones.
- Where specifications conflict, follow the OpenAPI contract for expected values, add the tag
  "spec-conflict" and explain the conflict in expected_results.
- Use only endpoints listed in the input."""

EXAMPLE = """\
Example case (data-driven):
{"title": "Reject order creation with invalid quantities", "type": "boundary", "priority": "P1",
 "tags": ["orders", "bva"], "actor": "customer_standard", "coverage": {"requirement_refs": ["PRD-ORD-43"]},
 "parameters": [{"name": "zero-quantity", "qty": 0, "expected_status": 422, "error_code": "MIN_QTY_EXCEEDED"},
                {"name": "negative-quantity", "qty": -1, "expected_status": 422, "error_code": "NEGATIVE_QTY"}],
 "steps": [{"name": "Attempt order creation with out-of-boundary quantity", "endpoint": "POST /v1/orders",
            "request": {"payload": {"items": [{"sku": "{{ $fixture.active_sku }}", "qty": "{{ $param.qty }}"}]}},
            "expected_status": "{{ $param.expected_status }}",
            "expected_error": {"code": "{{ $param.error_code }}"}}],
 "expected_results": ["API rejects the request with the documented status and error code"]}"""


class PlannedCases(BaseModel):
    cases: list[CaseDraft] = Field(description="new test cases for this module")


@dataclass
class PlannerDeps:
    ctx: ValidationContext
    recipes: dict[str, RecipeIR] = field(default_factory=dict)
    dropped: list[tuple[str, list[str]]] = field(default_factory=list)


def build_agent(output: Any = None) -> Agent[PlannerDeps, PlannedCases]:
    agent: Agent[PlannerDeps, PlannedCases] = Agent(
        output_type=output or output_spec(PlannedCases),
        instructions=INSTRUCTIONS + "\n\n" + EXAMPLE,
        deps_type=PlannerDeps,
        retries=get_settings().srg_llm_retries,
        name="test-planner",
    )

    @agent.output_validator
    def check(ctx: RunContext[PlannerDeps], out: PlannedCases) -> PlannedCases:
        problems = []
        valid: list[CaseDraft] = []
        for i, case in enumerate(out.cases):
            expanded, errors = expand_recipes(case, ctx.deps.recipes)
            normalized = normalize(expanded)
            errors = errors or validate_case(normalized, ctx.deps.ctx)
            if errors:
                problems.append((i, case.title, errors))
            else:
                valid.append(normalized)
        if not out.cases:
            raise ModelRetry("Return at least one new test case.")
        if problems and ctx.retry < ctx.max_retries:
            lines = [f"case {i} '{t}':\n  - " + "\n  - ".join(errs) for i, t, errs in problems]
            raise ModelRetry("Some cases are invalid. Fix them and return the FULL list again:\n" + "\n".join(lines))
        ctx.deps.dropped.extend((t, errs) for _, t, errs in problems)
        return PlannedCases(cases=valid)

    return agent


def build_prompt(
    analysis: SpecAnalysis,
    module: ModuleProposal,
    endpoints: dict[str, EndpointDraft],
    existing: list[dict],
    recipes: dict[str, RecipeIR] | None = None,
    lessons: list[str] | None = None,
) -> str:
    own = [endpoints[k] for k in module.endpoints if k in endpoints]
    others = [endpoints[k] for k in sorted(endpoints) if k not in module.endpoints]
    reqs = [
        r
        for r in analysis.requirements
        if r.req_id in module.requirement_ids or set(r.endpoints) & set(module.endpoints)
    ]
    req_ids = {r.req_id for r in reqs}
    conflicts = [c for c in analysis.conflicts if set(c.refs) & (req_ids | set(module.endpoints)) or not c.refs]
    gaps = [g for g in analysis.gaps if set(g.refs) & (req_ids | set(module.endpoints))]
    module_info = module.model_dump(
        include={"code", "domain", "base_path", "default_headers", "preconditions", "actors", "fixtures"}
    )
    parts = [
        f"Product summary:\n{analysis.summary}",
        f"Module:\n{json.dumps(module_info)}",
        f"Declared environment variables (the only allowed {{{{ $env.X }}}}): {sorted(declared_env_vars(*[a for a in module.actors if a.auth.type != 'login'], *module.fixtures)) or 'none'}",
        "Endpoints of this module:\n" + "\n".join(endpoint_digest(e) for e in own),
        "Other endpoints usable in setup/lookup steps (full contracts - send every required field):\n"
        + "\n".join(endpoint_digest(e) for e in others),
        "Requirements:\n" + json.dumps([r.model_dump() for r in reqs]),
    ]
    if recipes:
        parts.append(
            "Setup recipes (verified against the API) - use them for setup instead of writing the steps yourself:\n"
            + json.dumps([recipe_brief(r) for r in recipes.values()])
        )
    if lessons:
        parts.append("Lessons learned from earlier test defects (follow them):\n- " + "\n- ".join(lessons))
    if conflicts:
        parts.append("Specification conflicts:\n" + json.dumps([c.model_dump() for c in conflicts]))
    if gaps:
        parts.append("Known gaps:\n" + json.dumps([g.model_dump() for g in gaps]))
    if existing:
        parts.append("Already covered (do NOT repeat; add only what is missing):\n" + json.dumps(existing))
    else:
        parts.append("No test cases exist yet for this module.")
    return "\n\n".join(parts)


async def plan_module(
    analysis: SpecAnalysis,
    module: ModuleProposal,
    endpoints: dict[str, EndpointDraft],
    ctx: ValidationContext,
    existing: list[dict],
    model: Model | None = None,
    recipes: dict[str, RecipeIR] | None = None,
    lessons: list[str] | None = None,
) -> tuple[list[CaseDraft], list[tuple[str, list[str]]]]:
    deps = PlannerDeps(ctx=ctx, recipes=recipes or {})
    result = await run_structured(
        build_agent,
        build_prompt(analysis, module, endpoints, existing, recipes, lessons),
        PlannedCases,
        deps=deps,
        model=model or build_model(),
    )
    return result.output.cases, deps.dropped

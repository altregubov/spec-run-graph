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
from spec_run_graph.domain.ir import CaseDraft, ValidationContext, normalize, validate_case
from spec_run_graph.llm import build_model, output_spec, run_structured
from spec_run_graph.specs.models import EndpointDraft
from spec_run_graph.specs.openapi import endpoint_digest

INSTRUCTIONS = """\
You are a senior API test designer. Design test cases for ONE module in the YAML/JSON test-case
format below. Return only new cases (never repeat the "already covered" ones).

Test case fields: title, type (positive|negative|boundary|security|rbac), priority (P0|P1|P2), tags,
actor (one of the module actors), coverage.requirement_refs (requirement ids), preconditions (text),
variables, parameters, steps, expected_results, cleanup. Do NOT set an id - ids are assigned by the system.

Step fields: name, type (http|delay), endpoint ("METHOD /openapi/path", path parameters may be filled
inline with templates, e.g. "GET /v1/orders/{{ $extract.order_id }}", or via request.path_params),
request {headers, query, path_params, payload}, expected_status (int or "{{ $param.<key> }}"),
validate_schema (true to check the response against the OpenAPI schema), extract {var: path},
assertions [{check, message}], expected_error {code, field}, delay_ms (delay steps only).

Context variables (the ONLY allowed templates):
  {{ $faker.uuid }} {{ $faker.email }} {{ $faker.first_name }} ...  generated data
  {{ $fixture.<name> }}  a module fixture      {{ $env.<KEY> }}     environment variable
  {{ $extract.<var> }}   captured in an EARLIER step of the same case
  {{ $param.<key> }}     value of the current parameter row
  {{ $var.<name> }}      a value from the case `variables`

Assertion check grammar: "<path> <op> <literal>"
  path: response.status | response.headers.<name> | response.body.<json.path> (e.g. response.body.data.items[0].qty)
  op: == != > >= < <= in contains matches exists "not exists"
  literal: 'string' | number | true | false | null | ['a', 'b'] | a template
  examples: "response.body.data.status == 'DRAFT'", "response.body.data.id exists"
Extract paths use the same path syntax (response.body.data.id).

Rules:
- Data-driven cases: `parameters` is a list of rows; each row MUST have a kebab-case "name"
  (e.g. "zero-quantity") plus the keys used as {{ $param.<key> }}. Max 26 rows.
- Multi-step workflows are fine (create -> act -> verify). Setup actions are ordinary steps.
- Put teardown of created data into `cleanup` (same step format; failures there are ignored).
- Secrets only via actors or {{ $env.X }}; never literal tokens.
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
            "expected_error": {"code": "{{ $param.error_code }}", "field": "items[0].qty"}}],
 "expected_results": ["API rejects the request with the documented status and error code"]}"""


class PlannedCases(BaseModel):
    cases: list[CaseDraft] = Field(description="new test cases for this module")


@dataclass
class PlannerDeps:
    ctx: ValidationContext
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
            normalized = normalize(case)
            errors = validate_case(normalized, ctx.deps.ctx)
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
) -> str:
    own = [endpoints[k] for k in module.endpoints if k in endpoints]
    others = sorted(k for k in endpoints if k not in module.endpoints)
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
        f"Module:\n{json.dumps(module_info, indent=1)}",
        "Endpoints of this module:\n" + "\n".join(endpoint_digest(e) for e in own),
        f"Other endpoints usable in workflows: {others}",
        "Requirements:\n" + json.dumps([r.model_dump() for r in reqs], indent=1),
    ]
    if conflicts:
        parts.append("Specification conflicts:\n" + json.dumps([c.model_dump() for c in conflicts], indent=1))
    if gaps:
        parts.append("Known gaps:\n" + json.dumps([g.model_dump() for g in gaps], indent=1))
    if existing:
        parts.append("Already covered (do NOT repeat; add only what is missing):\n" + json.dumps(existing, indent=1))
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
) -> tuple[list[CaseDraft], list[tuple[str, list[str]]]]:
    deps = PlannerDeps(ctx=ctx)
    result = await run_structured(
        build_agent,
        build_prompt(analysis, module, endpoints, existing),
        PlannedCases,
        deps=deps,
        model=model or build_model(),
    )
    return result.output.cases, deps.dropped

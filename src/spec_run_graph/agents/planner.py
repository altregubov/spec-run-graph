"""Test Planner: designs API test cases (IR) for one module chunk. Ids are assigned later by code."""

from __future__ import annotations

import json
import re
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
actor (one of the module actors), coverage.requirement_refs (requirement ids), coverage.criteria (ids of the
acceptance criteria the case checks, e.g. "REQ-ORD-03#2", as listed with the requirements), preconditions (text),
variables, parameters, steps, expected_results, cleanup. Do NOT set an id - ids are assigned by the system.

Step fields: name, type (http|delay|recipe), actor (only when the step runs as a different module actor than the
case, e.g. a "user-role token" step in an admin case), endpoint ("METHOD /openapi/path", path parameters may be filled
inline with templates, e.g. "GET /v1/orders/{{ $extract.order_id }}", or via request.path_params),
request {headers, query, path_params, payload}, expected_status (int, "401|403" or "{{ $param.<key> }}"),
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
- A recipe creates its entities with its input values and with the values listed under its "fixed" (e.g.
  balance 5000, empty allowed_categories): a case can change only the inputs. Base expected values on them
  (or on values extracted from the recipe's responses), never on the case's own assumptions. If the scenario
  needs a different value for a fixed field (balance 100, stock 2, a non-empty whitelist), write that setup
  step yourself instead of using the recipe.
- Recipe outputs hold the value at the moment the recipe captured it (stock_before_order is the stock BEFORE the
  recipe's own order). A baseline for "unchanged"/"changed by N" checks must be read after the last step that
  changes it - a recipe output describing the state after its changes, or a GET right before the action under test.
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
  setup steps against other modules' endpoints (e.g. role and balance when creating a user). The same holds for
  negative steps that test something other than body validation (wrong password -> 401, missing token, unknown
  id): send a complete body, otherwise the API answers 400 before reaching the tested condition.
- Orders, stock changes and other state-changing requests use entities the case creates (a recipe such as
  dedicated_product, or a POST step), never an item picked from the shared catalogue ("cheapest laptop"):
  other tests leave products with zero stock or price there.
- Assert only fields that exist in the documented response schema (e.g. error.code/error.message, not
  invented fields); for lists use the documented item fields (e.g. data[0].id, not data[0].product_id).
- Expected statuses: when the requirements or the contract allow several statuses for the same condition
  ("401 or 403", "rejected with 400 or 422"), set expected_status to all of them, e.g. "401|403" (each must be
  documented for the endpoint). Otherwise use the single documented status.
- Test only behaviour the specifications define. Do not assert undefined details such as case sensitivity of
  enum/sort values, whitespace trimming or the order of ties; leave them out instead of guessing. (Duplicate
  creation is not such a detail - see the creation scenarios below.)
- Derive computed values from the case's own data (total_price = unit price x quantity of THIS order).
- Put teardown of created data into `cleanup` (same step format; failures there are ignored).
- Authentication only via actors (case `actor` / step `actor`); never literal tokens. A step without `actor`
  runs as the case actor and sends its valid token. For a request WITHOUT credentials (missing token -> 401)
  set the step actor to "anonymous"; for an invalid token set request.headers.Authorization
  ("Bearer invalid-token").
- {{ $env.X }} is ONLY for the declared environment variables listed in the input - never for test data.
  Ids and other values created by the system (seed product ids, user ids, order ids) are looked up at run time:
  add an earlier step (e.g. GET /api/.../products) and `extract` the value, then use {{ $extract.x }}.
- Cover every requirement of the module with positive, negative/boundary, security (missing or invalid
  token -> 401) and, where several actors exist, rbac cases. Every acceptance criterion is checked by at least
  one case: list the criteria a case checks in coverage.criteria.
- Test-design techniques - apply each one that fits an endpoint of this module. Put the variants of one technique
  into ONE data-driven case (a parameter row per variant) instead of many similar cases:
  * every constrained request field (minimum/maximum, length, enum, format): boundary values min-1, min, max,
    max+1 (invalid ones expect the documented error; the valid ones go into a separate positive case, since one
    case's assertions run for every row), plus wrong type ("abc" for a number,
    a number for a string), null, empty string / whitespace-only, a very long or unicode text where the field is
    free text, and a value outside the enum;
  * request bodies: an unknown extra field, and a valid body whose values are at the limits (largest accepted
    amount, smallest positive price);
  * authentication: missing token (actor "anonymous"), malformed header ("Bearer", "Basic x", "Bearer a.b"),
    a token that is well-formed but invalid, and the token of a role that must not access the endpoint;
  * state changes: every documented transition, and every forbidden one from each state (e.g. cancel an order
    that is already CANCELLED or FAILED), checking that a rejected change leaves the state unchanged;
  * lists: each filter, sort field and direction, page boundaries (first, last, beyond the last page), the empty
    result, and filter combinations;
  * ids in paths: unknown but well-formed id (404) and malformed id (400) where documented;
  * money and quantities: exact cent arithmetic (0.01 steps, totals of several units) where the specs define it.
  Expected outcomes still come only from the specs: a variant whose outcome the specs do not define is left out
  (or tagged "spec-gap" with the set of documented statuses).
- Creation scenarios: for every endpoint that creates an entity, first decide from the specs what identifies
  that entity in the business domain (a username for an account, an SKU or a manufacturer + model for a product,
  an order number ...) and cover:
  1. create, then read it back (GET by id or a list filter on the unique value) and compare the stored values
     with the sent ones;
  2. DUPLICATE creation: send the same create request twice with the same identifying values (one variable,
     used in both steps). If the specs name a uniqueness rule or document a conflict response (e.g. 409 with an
     error code), expect it on the second request and check that only one entity with that value exists. If the
     specs define no rule for this entity, expect what the contract documents for the second request (usually
     success with a DIFFERENT id), check that both entities exist, and tag the case "spec-gap" with an
     expected_results line saying that duplicate handling is undefined - the run then reveals whether the API
     silently accepts duplicates or rejects them with an undocumented status;
     To prove that each entity exists, extract both ids and look each one up by its own id (GET by id, or a
     list filter data[id={{ $extract.second_id }}].id exists): a filter on the shared identifying value
     finds only the first match and proves nothing about the second;
  3. variants of the identifying value only where the specs define normalisation (case, whitespace).
- Every case tests an endpoint listed under "Endpoints of this module"; endpoints listed only as "Other
  endpoints" are setup and verification steps, never the subject (their scenarios, e.g. the creation
  scenarios of a create endpoint, are planned together with that endpoint).
- Every case tests a distinct behaviour: never return two cases - or a case like an "already covered" one -
  that send the same kind of requests and check the same outcome with only different data or wording.
- When the specs leave an outcome open (tag "spec-gap"), do not pick one: expect the set of documented statuses that
  could apply (e.g. "200|401|403" for a soft-deleted user's still-valid token) and assert only what holds for all of
  them; the run records which one the API chose.
- Where specifications conflict, follow the OpenAPI contract for expected values, add the tag
  "spec-conflict" and explain the conflict in expected_results.
- "Observed data" shows real responses of the API (read-only samples taken before planning). Use it to pick
  existing values (categories, manufacturers, models) and to see how the API formats them (casing, number
  formats). Free-text values the API may normalise are compared case-insensitively (matches '(?i)^apple$'),
  unless the contract documents the exact format. Never assert counts, ids or balances from the samples - data
  changes between runs - and never derive an expected status from them: the contract decides.
- Use only endpoints listed in the input."""

EXAMPLE = """\
Example case (data-driven):
{"title": "Reject order creation with invalid quantities", "type": "boundary", "priority": "P1",
 "tags": ["orders", "bva"], "actor": "customer_standard",
 "coverage": {"requirement_refs": ["PRD-ORD-43"], "criteria": ["PRD-ORD-43#1", "PRD-ORD-43#2"]},
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
    rule_hits: dict[str, int] = field(default_factory=dict)  # validation rule -> times broken (all attempts)


_RULE_NOISE = [
    (re.compile(r"^(cleanup )?step \d+ \([^)]*\):\s*"), ""),
    (re.compile(r"'[^']*'|\"[^\"]*\"|\{\{[^}]*\}\}"), "<v>"),
    (re.compile(r"\[[^\]]*\]"), "[..]"),
    (re.compile(r"\b\d+\b"), "<n>"),
]


def rule_key(error: str) -> str:
    """A validation error without its case-specific details, to count which rules the LLM struggles with."""
    for pattern, repl in _RULE_NOISE:
        error = pattern.sub(repl, error)
    return error[:100]


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
                for e in errors:
                    key = rule_key(e)
                    ctx.deps.rule_hits[key] = ctx.deps.rule_hits.get(key, 0) + 1
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
    observed: str = "",
    coverage_gaps: list[str] | None = None,
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
        "Requirements (acceptance criteria keyed by their id for coverage.criteria):\n"
        + json.dumps([_requirement_view(r) for r in reqs]),
    ]
    if recipes:
        parts.append(
            "Setup recipes (verified against the API) - use them for setup instead of writing the steps yourself:\n"
            + json.dumps([recipe_brief(r) for r in recipes.values()])
        )
    if lessons:
        parts.append("Lessons learned from earlier test defects (follow them):\n- " + "\n- ".join(lessons))
    if observed:
        parts.append(
            "Observed data (real GET responses, trimmed; values and formats only - the contract decides what is"
            " correct):\n" + observed
        )
    if conflicts:
        parts.append("Specification conflicts:\n" + json.dumps([c.model_dump() for c in conflicts]))
    if gaps:
        parts.append("Known gaps:\n" + json.dumps([g.model_dump() for g in gaps]))
    if existing:
        parts.append("Already covered (do NOT repeat; add only what is missing):\n" + json.dumps(existing))
    else:
        parts.append("No test cases exist yet for this module.")
    if coverage_gaps:
        parts.append(
            "COVERAGE GAPS found after the first planning pass - design new cases that close exactly these gaps "
            "(several gaps may share one data-driven case; skip a gap only if the listed endpoints cannot test it or "
            "the specs do not define the expected outcome):\n- " + "\n- ".join(coverage_gaps)
        )
    return "\n\n".join(parts)


def _requirement_view(r: Any) -> dict[str, Any]:
    data = r.model_dump(exclude={"acceptance_criteria"})
    data["acceptance_criteria"] = {f"{r.req_id}#{n}": t for n, t in enumerate(r.acceptance_criteria, start=1)}
    return data


async def plan_module(
    analysis: SpecAnalysis,
    module: ModuleProposal,
    endpoints: dict[str, EndpointDraft],
    ctx: ValidationContext,
    existing: list[dict],
    model: Model | None = None,
    recipes: dict[str, RecipeIR] | None = None,
    lessons: list[str] | None = None,
    observed: str = "",
    rule_hits: dict[str, int] | None = None,
    coverage_gaps: list[str] | None = None,
) -> tuple[list[CaseDraft], list[tuple[str, list[str]]]]:
    deps = PlannerDeps(ctx=ctx, recipes=recipes or {}, rule_hits=rule_hits if rule_hits is not None else {})
    result = await run_structured(
        build_agent,
        build_prompt(analysis, module, endpoints, existing, recipes, lessons, observed, coverage_gaps),
        PlannedCases,
        deps=deps,
        model=model or build_model(),
    )
    return result.output.cases, deps.dropped

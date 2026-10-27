"""Failure triage: deterministic pre-classifier + ReAct diagnostics agent (the .env model).

Categories
  APPLICATION_BUG  the API behaves wrongly although the test and the contract agree (crash, wrong data/state)
  SPEC_MISMATCH    the API is internally consistent but deviates from the documented contract / requirements
  TEST_DEFECT      the generated test is wrong (bad expectation, wrong data/order/variable, invalid assumption)
  ENV_FAILURE      the environment failed (connection refused, DNS, timeouts, 502/503/504, missing secrets)
  FLAKY            the failure does not reproduce on rerun (timing, ordering, shared state)
  UNKNOWN          not enough evidence (low confidence)
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.llm import build_model, output_spec, run_structured

Category = Literal["APPLICATION_BUG", "SPEC_MISMATCH", "TEST_DEFECT", "ENV_FAILURE", "FLAKY", "UNKNOWN"]
CONFIDENCE_THRESHOLD = 0.5
MAX_TOOL_TEXT = 12000

ENV_PATTERNS = re.compile(
    r"ECONNREFUSED|ENOTFOUND|EAI_AGAIN|ETIMEDOUT|ECONNRESET|EHOSTUNREACH|socket hang up|"
    r"connect(ion)? (refused|timed out)|getaddrinfo|certificate|self[- ]signed",
    re.IGNORECASE,
)


class Diagnosis(BaseModel):
    category: Category
    root_cause: str = Field(description="one or two sentences: what exactly went wrong and where")
    evidence: list[str] = Field(default_factory=list, description="concrete facts: status codes, field values, quotes")
    recommended_action: str = Field(description="who should do what (fix API / update spec / fix test / retry env)")
    suggested_fix: str | None = Field(None, description="for TEST_DEFECT: the concrete change to the test case")
    confidence: float = Field(ge=0, le=1)


@dataclass
class FailedExecution:
    """Everything known about one failed execution (assembled by the pipeline stage)."""

    execution_id: str
    run_id: str
    test_id: str
    title: str
    param_index: int
    status: str
    http_status: int | None
    error_message: str | None
    failed_assertions: list[dict[str, Any]]
    case_ir: dict[str, Any]
    endpoints: list[str]
    requirement_ids: list[str]
    actor: str | None = None
    similar: list[str] = field(default_factory=list)  # other executions with the same failure signature
    missing_env: list[str] = field(default_factory=list)  # process.env vars the test uses that are not set
    undeclared_env: list[str] = field(default_factory=list)  # ... of which no actor/fixture declares (invented)
    # steps that omit contract-required request fields, expected a 4xx and got a 2xx (the API broke its contract)
    contract_violations: list[str] = field(default_factory=list)


_NORMALIZE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|\d+")


def signature(f: FailedExecution) -> tuple:
    """Failures with the same signature very likely share one root cause (e.g. 150 x HTTP 401)."""
    first = f.failed_assertions[0] if f.failed_assertions else {}
    what = _NORMALIZE.sub("#", str(first.get("assertion") or first.get("test") or ""))[:120]
    why = _NORMALIZE.sub("#", str(first.get("error") or f.error_message or ""))[:120]
    # an unexpected 401 across endpoints is usually one credential problem; everything else (incl. 403) is
    # endpoint-specific - grouping 403s by actor copied one diagnosis onto unrelated cases
    scope = ("actor", f.actor) if f.http_status == 401 else ("endpoints", tuple(f.endpoints))
    return (f.status, f.http_status, what, why, scope)


PLACEHOLDER = re.compile(r"\b(placeholder|tbd|todo|lorem|n/?a|unknown cause)\b", re.IGNORECASE)


def diagnosis_problems(d: Diagnosis) -> list[str]:
    """Reject empty / template answers (seen on a real run: root_cause='placeholder', confidence 0)."""
    problems = []
    if len(d.root_cause.strip()) < 25 or PLACEHOLDER.search(d.root_cause):
        problems.append("root_cause must be a concrete explanation (what failed, where, why)")
    if len(d.recommended_action.strip()) < 10 or PLACEHOLDER.search(d.recommended_action):
        problems.append("recommended_action must say who should do what")
    if d.category != "UNKNOWN" and not d.evidence:
        problems.append("give concrete evidence (status codes, field values) for the category")
    return problems


def preclassify(f: FailedExecution) -> Diagnosis | None:
    """Obvious environment failures need no LLM call."""
    if f.undeclared_env:
        names = ", ".join(f.undeclared_env)
        return Diagnosis(
            category="TEST_DEFECT",
            root_cause=f"The test reads environment variable(s) {names}, which no actor or fixture declares: "
            "the test invented them for data that only exists at run time (e.g. seeded ids).",
            evidence=[f"undeclared and unset: {names}", f"HTTP {f.http_status}"],
            recommended_action="Fix the test case: look the value up in an earlier step and extract it.",
            suggested_fix=f"Replace {{{{ $env.{f.undeclared_env[0]} }}}} with a value extracted in an earlier step "
            "(e.g. GET the list endpoint, extract the matching id) and use {{ $extract.<name> }}.",
            confidence=0.95,
        )
    if f.missing_env:
        names = ", ".join(f.missing_env)
        return Diagnosis(
            category="ENV_FAILURE",
            root_cause=f"The test uses environment variable(s) {names}, which are not set (.env / process env), "
            "so the request was sent without valid credentials or data.",
            evidence=[f"missing: {names}", f"HTTP {f.http_status}"],
            recommended_action=f"Set {names} in .env, or re-analyse the specs so documented test accounts log in "
            "automatically (srg analyze, then srg plan --from <analysis.json> --refresh-only, then srg build).",
            confidence=0.97,
        )
    if f.contract_violations:
        return Diagnosis(
            category="APPLICATION_BUG",
            root_cause="The API accepted a request that omits fields its OpenAPI contract marks as required: "
            + "; ".join(f.contract_violations),
            evidence=f.contract_violations[:5],
            recommended_action="Report to the API team: required-field validation is missing. Keep the test.",
            confidence=0.9,
        )
    text = " ".join([f.error_message or ""] + [str(a.get("error") or "") for a in f.failed_assertions])
    if ENV_PATTERNS.search(text):
        return Diagnosis(
            category="ENV_FAILURE",
            root_cause="The request did not reach a healthy service (network/connection error).",
            evidence=[text[:300]],
            recommended_action="Check that the system under test is running and reachable from the test runner, then re-run.",
            confidence=0.95,
        )
    if f.http_status in (502, 503, 504):
        return Diagnosis(
            category="ENV_FAILURE",
            root_cause=f"The service or a gateway answered HTTP {f.http_status} (unavailable).",
            evidence=[f"HTTP {f.http_status}"],
            recommended_action="Check the deployment / dependencies of the system under test, then re-run.",
            confidence=0.85,
        )
    if f.status == "TIMEOUT":
        return Diagnosis(
            category="ENV_FAILURE",
            root_cause="The test run timed out before this case produced a result.",
            evidence=["execution status TIMEOUT"],
            recommended_action="Check service latency or raise BRU_TIMEOUT_SEC, then re-run.",
            confidence=0.7,
        )
    return None


INSTRUCTIONS = """\
You are a senior QA engineer triaging ONE failed API test. Find the root cause and classify it:
- APPLICATION_BUG: the API is wrong although the test matches the contract (crash/500, wrong data, wrong state).
- SPEC_MISMATCH: the API consistently does something different from the documented contract/requirements
  (different status code, field name, error code, schema) - the spec or the API must be aligned.
- TEST_DEFECT: the generated test is wrong (wrong expectation, wrong test data, wrong step order, missing setup,
  variable not captured, assertion on the wrong path, assumption not backed by the spec).
- ENV_FAILURE: environment problem (connectivity, missing credentials/secrets, unavailable dependency).
- FLAKY: the failure does not reproduce when re-run.
- UNKNOWN: not enough evidence.

Method: start from the failed assertions, then use the tools - look at the real HTTP exchange, compare it with
the OpenAPI contract and the requirement, check the test files and the case history. Use rerun_case only when
flakiness is plausible (intermittent history, timing, shared state). Decide who is wrong: API, spec, or test.
Quote concrete evidence (status codes, field names, values). Give a calibrated confidence (0..1).
For TEST_DEFECT put the exact change to the test case into suggested_fix."""


@dataclass
class DiagDeps:
    failure: FailedExecution
    get_report: Callable[[], Awaitable[str]]
    get_contracts: Callable[[], Awaitable[dict[str, Any]]]
    get_requirements: Callable[[], Awaitable[list[dict[str, Any]]]]
    get_files: Callable[[], Awaitable[dict[str, str]]]
    get_history: Callable[[], Awaitable[list[dict[str, Any]]]]
    rerun: Callable[[], Awaitable[str]] | None = None
    rerun_result: str | None = None
    tools_used: list[str] = field(default_factory=list)


async def _safe(name: str, ctx: RunContext[DiagDeps], fetch: Callable[[], Awaitable[Any]]) -> str:
    """Tool errors are reported to the model as text instead of aborting the investigation."""
    ctx.deps.tools_used.append(name)
    try:
        return _clip(await fetch())
    except Exception as exc:  # noqa: BLE001
        return f"{name} unavailable: {exc.__class__.__name__}: {str(exc)[:200]}"


def _clip(data: Any) -> str:
    text = data if isinstance(data, str) else json.dumps(data, default=str)
    return text if len(text) <= MAX_TOOL_TEXT else text[:MAX_TOOL_TEXT] + " ...(truncated)"


def build_agent(output: Any = None) -> Agent[DiagDeps, Diagnosis]:
    agent: Agent[DiagDeps, Diagnosis] = Agent(
        output_type=output or output_spec(Diagnosis),
        instructions=INSTRUCTIONS,
        deps_type=DiagDeps,
        retries=get_settings().srg_llm_retries,
        name="failure-diagnostics",
    )

    @agent.output_validator
    def check(ctx: RunContext[DiagDeps], out: Diagnosis) -> Diagnosis:
        problems = diagnosis_problems(out)
        if problems:
            raise ModelRetry("Your diagnosis is not usable: " + "; ".join(problems) + ". Investigate and answer again.")
        return out

    @agent.tool
    async def http_exchange(ctx: RunContext[DiagDeps]) -> str:
        """The recorded requests and responses (status, headers, body) of every step of this failed row."""
        return await _safe("http_exchange", ctx, ctx.deps.get_report)

    @agent.tool
    async def api_contract(ctx: RunContext[DiagDeps]) -> str:
        """OpenAPI contract (parameters, request schema, documented responses) of the endpoints the case calls."""
        return await _safe("api_contract", ctx, ctx.deps.get_contracts)

    @agent.tool
    async def requirements(ctx: RunContext[DiagDeps]) -> str:
        """The product requirements this test case verifies."""
        return await _safe("requirements", ctx, ctx.deps.get_requirements)

    @agent.tool
    async def test_files(ctx: RunContext[DiagDeps]) -> str:
        """The generated Bruno request files (YAML) that were executed for this row."""
        return await _safe("test_files", ctx, ctx.deps.get_files)

    @agent.tool
    async def case_history(ctx: RunContext[DiagDeps]) -> str:
        """Previous results of this case/row in earlier runs (to spot flakiness or regressions)."""
        return await _safe("case_history", ctx, ctx.deps.get_history)

    @agent.tool
    async def rerun_case(ctx: RunContext[DiagDeps]) -> str:
        """Execute this case/row once more against the system under test and return the outcome (max once)."""
        if ctx.deps.rerun is None:
            return "rerun disabled"
        if ctx.deps.rerun_result is None:
            ctx.deps.rerun_result = await _safe("rerun_case", ctx, ctx.deps.rerun)
        return ctx.deps.rerun_result

    return agent


def build_prompt(f: FailedExecution) -> str:
    case = {k: v for k, v in f.case_ir.items() if k not in ("id", "status")}
    row = (case.get("parameters") or [None])[f.param_index] if case.get("parameters") else None
    parts = [
        f"Failed test: {f.test_id} '{f.title}' (row {chr(ord('a') + f.param_index)}) in run {f.run_id}",
        f"Execution status: {f.status}; last HTTP status: {f.http_status}; error: {f.error_message or '-'}",
        "Failed assertions:\n" + json.dumps(f.failed_assertions, default=str)[:4000],
        f"Endpoints: {f.endpoints}; requirements: {f.requirement_ids}",
    ]
    if row:
        parts.append(f"Parameter row: {json.dumps(row)}")
    if f.similar:
        shown = ", ".join(f.similar[:30]) + (" ..." if len(f.similar) > 30 else "")
        parts.append(
            f"The same failure signature occurs in {len(f.similar)} other executions ({shown}). "
            "Look for the common root cause; your diagnosis is applied to all of them."
        )
    parts.append("Test case (IR):\n" + yaml.safe_dump(case, sort_keys=False, allow_unicode=True)[:8000])
    return "\n\n".join(parts)


async def diagnose(deps: DiagDeps, model: Model | None = None) -> Diagnosis:
    result = await run_structured(
        build_agent, build_prompt(deps.failure), Diagnosis, deps=deps, model=model or build_model()
    )
    d: Diagnosis = result.output
    if d.confidence < CONFIDENCE_THRESHOLD and d.category != "UNKNOWN":
        d = d.model_copy(update={"category": "UNKNOWN", "root_cause": f"(low confidence {d.category}) {d.root_cause}"})
    return d

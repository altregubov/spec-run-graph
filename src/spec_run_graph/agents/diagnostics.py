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
from pydantic_ai import Agent, RunContext
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


_NORMALIZE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|\d+")


def signature(f: FailedExecution) -> tuple:
    """Failures with the same signature very likely share one root cause (e.g. 150 x HTTP 401)."""
    first = f.failed_assertions[0] if f.failed_assertions else {}
    what = _NORMALIZE.sub("#", str(first.get("assertion") or first.get("test") or ""))[:120]
    why = _NORMALIZE.sub("#", str(first.get("error") or f.error_message or ""))[:120]
    scope = ("actor", f.actor) if f.http_status in (401, 403) else ("endpoints", tuple(f.endpoints))
    return (f.status, f.http_status, what, why, scope)


def preclassify(f: FailedExecution) -> Diagnosis | None:
    """Obvious environment failures need no LLM call."""
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

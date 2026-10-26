"""Interactive QA chatbot over Gel (read-only), using the .env model."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import gel
from pydantic_ai import Agent, ModelRetry, RunContext

from spec_run_graph.config import get_settings
from spec_run_graph.knowledge import db, stats

MAX_ROWS = 200
QUERY_TIMEOUT_S = 10
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|configure|create|drop|alter|start\s+migration|commit|rollback|reset|"
    r"administer|describe\s+roles|set\s+global)\b",
    re.IGNORECASE,
)

INSTRUCTIONS = """\
You are the QA assistant of SpecRunGraph. You answer questions about the test knowledge base stored in
Gel: specifications, API endpoints, requirements, test cases, test runs and executions.
- Prefer the typed tools (overview, list_test_cases, endpoint_coverage, requirement_coverage, get_case,
  list_runs, run_results, failing_cases). Use query_gel only when no typed tool fits.
- query_gel accepts ONE read-only EdgeQL `select` (or `with ... select`) statement. The schema is below.
  Enum values are compared like `.design_status = DesignStatus.READY`; HTTP methods like `HttpMethod.POST`.
- Never invent numbers: every figure must come from a tool result. If data is missing, say so.
- You cannot modify anything; politely refuse requests to change or delete data and point to the CLI
  (`srg plan`, `srg build`, `srg reset`, `srg show <id> --approve`).
- Answer concisely in Markdown (tables for lists)."""


@dataclass
class ChatDeps:
    client: gel.AsyncIOClient
    trace: list[dict[str, Any]] = field(default_factory=list)


def guard_query(query: str) -> str:
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        raise ModelRetry("Only a single statement is allowed (no ';').")
    if not re.match(r"(?is)^(with\b.*\bselect\b|select\b)", q):
        raise ModelRetry("Only read-only `select` / `with ... select` queries are allowed.")
    if FORBIDDEN.search(q):
        raise ModelRetry("Data-modifying or administrative statements are not allowed; this assistant is read-only.")
    return q


def _dump(data: Any) -> str:
    text = json.dumps(data, default=str)
    return text if len(text) < 30000 else text[:30000] + " ...(truncated)"


def build_agent() -> Agent[ChatDeps, str]:
    schema = db.SCHEMA_FILE.read_text()
    agent: Agent[ChatDeps, str] = Agent(
        instructions=INSTRUCTIONS + "\n\nGel schema (module default):\n" + schema,
        deps_type=ChatDeps,
        retries=get_settings().srg_llm_retries,
        name="qa-chatbot",
    )

    def traced(name: str, args: dict[str, Any], ctx: RunContext[ChatDeps], query: str | None = None) -> None:
        ctx.deps.trace.append({"tool": name, "args": args, **({"edgeql": query} if query else {})})

    @agent.tool
    async def overview(ctx: RunContext[ChatDeps]) -> str:
        """Totals: test cases by status/type/priority/module, endpoint & requirement coverage, last run pass rate."""
        traced("overview", {}, ctx)
        return _dump(await stats.overview(ctx.deps.client))

    @agent.tool
    async def list_test_cases(
        ctx: RunContext[ChatDeps],
        module: str | None = None,
        status: str | None = None,
        case_type: str | None = None,
        priority: str | None = None,
        tag: str | None = None,
        id_glob: str | None = None,
        endpoint: str | None = None,
    ) -> str:
        """List test cases. status: DRAFT|READY|NEEDS_REVIEW|DEPRECATED; case_type: POSITIVE|NEGATIVE|BOUNDARY|
        SECURITY|RBAC; priority: P0|P1|P2; id_glob like 'TC-ORD-*'; endpoint like 'POST /v1/orders'."""
        f = {
            "module": module,
            "status": status,
            "type": case_type,
            "priority": priority,
            "tag": tag,
            "id": id_glob,
            "endpoint": endpoint,
        }
        traced("list_test_cases", {k: v for k, v in f.items() if v}, ctx)
        return _dump(await stats.list_cases(ctx.deps.client, f))

    @agent.tool
    async def endpoint_coverage(ctx: RunContext[ChatDeps]) -> str:
        """Every API endpoint (latest spec versions) with its active test cases and counts per case type."""
        traced("endpoint_coverage", {}, ctx)
        return _dump(await stats.endpoint_coverage(ctx.deps.client))

    @agent.tool
    async def requirement_coverage(ctx: RunContext[ChatDeps]) -> str:
        """Every requirement with its endpoints and the test cases that verify it."""
        traced("requirement_coverage", {}, ctx)
        return _dump(await stats.requirement_matrix(ctx.deps.client))

    @agent.tool
    async def get_case(ctx: RunContext[ChatDeps], test_id: str) -> str:
        """Full details of one test case (steps, coverage, generation status, last executions)."""
        traced("get_case", {"test_id": test_id}, ctx)
        raw = await ctx.deps.client.query_single_json(
            """select TestCase { test_id, title, case_type, priority, design_status, gen_status, review_reason,
                 tags, preconditions, expected_results, gen_model, last_gen_error, script_path,
                 steps: { seq, name, is_cleanup, method, path, expected_status, assertions } order by .seq,
                 executions: { run_id := .run.run_id, param_index, status, http_status, created_at }
                   order by .created_at desc limit 10 } filter .test_id = <str>$t""",
            t=test_id,
        )
        return raw or f"no test case {test_id}"

    @agent.tool
    async def list_runs(ctx: RunContext[ChatDeps], limit: int = 10) -> str:
        """Recent test runs with pass/fail counts."""
        traced("list_runs", {"limit": limit}, ctx)
        return _dump(await stats.list_runs(ctx.deps.client, min(limit, 50)))

    @agent.tool
    async def run_results(ctx: RunContext[ChatDeps], run_id: str) -> str:
        """Per-case results of one run (status, HTTP code, failed assertions)."""
        traced("run_results", {"run_id": run_id}, ctx)
        return _dump(await stats.run_results(ctx.deps.client, run_id) or f"no run {run_id}")

    @agent.tool
    async def failing_cases(ctx: RunContext[ChatDeps], last_n_runs: int = 1) -> str:
        """Failed/errored executions of the last N completed runs."""
        traced("failing_cases", {"last_n_runs": last_n_runs}, ctx)
        return _dump(await stats.failing_cases(ctx.deps.client, max(1, min(last_n_runs, 20))))

    @agent.tool
    async def query_gel(ctx: RunContext[ChatDeps], edgeql: str) -> str:
        """Run ONE read-only EdgeQL select query (fallback when no typed tool fits). Max 200 rows."""
        q = guard_query(edgeql)
        traced("query_gel", {}, ctx, q)
        try:
            raw = await asyncio.wait_for(db.readonly_query(ctx.deps.client, q), QUERY_TIMEOUT_S)
        except gel.errors.EdgeDBError as exc:
            raise ModelRetry(f"Query failed: {exc}") from exc
        except TimeoutError as exc:
            raise ModelRetry("Query timed out; make it more selective.") from exc
        rows = json.loads(raw)
        if isinstance(rows, list) and len(rows) > MAX_ROWS:
            return _dump(rows[:MAX_ROWS]) + f" ...({len(rows) - MAX_ROWS} more rows not shown)"
        return _dump(rows)

    return agent

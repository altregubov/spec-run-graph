"""Read-only dashboard queries (also the typed tools of the QA chatbot)."""

from __future__ import annotations

import json
from typing import Any

import gel

from spec_run_graph.knowledge.db import from_json

LATEST_SPECS = "(select FeatureSpec filter not exists .<previous[is FeatureSpec])"
ACTIVE = "(.design_status != DesignStatus.DEPRECATED)"


async def overview(client: gel.AsyncIOExecutor) -> dict[str, Any]:
    raw = await client.query_single_json(
        f"""
        with
            cases := (select TestCase filter {ACTIVE}),
            eps := (select ApiEndpoint filter .spec in {LATEST_SPECS}),
            reqs := (select Requirement filter .spec in {LATEST_SPECS}),
            last_run := (select TestRun filter .status = RunStatus.COMPLETED order by .started_at desc limit 1),
        select {{
            total_cases := count(cases),
            deprecated := count(TestCase filter .design_status = DesignStatus.DEPRECATED),
            by_status := (group cases by .design_status) {{ k := <str>.key.design_status, n := count(.elements) }},
            by_gen := (group cases by .gen_status) {{ k := <str>.key.gen_status, n := count(.elements) }},
            by_type := (group cases by .case_type) {{ k := <str>.key.case_type, n := count(.elements) }},
            by_priority := (group cases by .priority) {{ k := <str>.key.priority, n := count(.elements) }},
            by_module := (group cases using code := .test_module.code by code) {{ k := .key.code, n := count(.elements) }},
            endpoints_total := count(eps),
            endpoints_covered := count(eps filter exists (select .<endpoint[is TestStep].test_case filter {ACTIVE})),
            requirements_total := count(reqs),
            requirements_covered := count(reqs filter exists (select .<requirements[is TestCase] filter {ACTIVE})),
            runs := count(TestRun),
            last_run := last_run {{
                run_id, started_at,
                passed := count(.executions filter .status = ExecutionStatus.PASSED),
                total := count(.executions),
            }},
        }}
        """
    )
    data = json.loads(raw)
    for k in ("by_status", "by_gen", "by_type", "by_priority", "by_module"):
        data[k] = {row["k"]: row["n"] for row in data[k]}
    lr = data.get("last_run")
    data["last_run_pass_rate"] = round(100 * lr["passed"] / lr["total"]) if lr and lr["total"] else None
    data["endpoint_coverage_pct"] = (
        round(100 * data["endpoints_covered"] / data["endpoints_total"]) if data["endpoints_total"] else 0
    )
    return data


async def endpoint_coverage(client: gel.AsyncIOExecutor) -> list[dict[str, Any]]:
    raw = await client.query_json(
        f"""
        select ApiEndpoint {{
            key, summary, tag, spec_name := .spec.name, spec_version := .spec.version,
            cases := (select distinct .<endpoint[is TestStep].test_case filter {ACTIVE}) {{ test_id, case_type }},
        }}
        filter .spec in {LATEST_SPECS}
        order by .path then .method
        """
    )
    rows = []
    for e in json.loads(raw):
        counts: dict[str, int] = {}
        for c in e["cases"]:
            counts[c["case_type"]] = counts.get(c["case_type"], 0) + 1
        rows.append({**e, "by_type": counts, "total": len(e["cases"])})
    return rows


async def requirement_matrix(client: gel.AsyncIOExecutor) -> list[dict[str, Any]]:
    raw = await client.query_json(
        f"""
        select Requirement {{
            req_id, description, spec_name := .spec.name,
            endpoints: {{ key }},
            cases := (select .<requirements[is TestCase] filter {ACTIVE}) {{ test_id, case_type, design_status }},
        }}
        filter .spec in {LATEST_SPECS}
        order by .req_id
        """
    )
    return json.loads(raw)


async def list_cases(client: gel.AsyncIOExecutor, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    f = filters or {}
    raw = await client.query_json(
        """
        select TestCase {
            test_id, title, case_type, priority, design_status, gen_status, tags, review_reason,
            module_code := .test_module.code, endpoints: { key }, requirements: { req_id },
            last := (select .executions order by .created_at desc limit 1) { status, http_status, created_at },
        }
        filter ((.test_module.code = <optional str>$module) ?? true)
           and ((<str>.design_status = <optional str>$status) ?? true)
           and ((<str>.case_type = <optional str>$type) ?? true)
           and ((<str>.priority = <optional str>$priority) ?? true)
           and ((<optional str>$tag in array_unpack(.tags)) ?? true)
           and ((.test_id like <optional str>$id_like) ?? true)
           and (exists (select .endpoints filter .key = <optional str>$endpoint) if exists <optional str>$endpoint else true)
        order by .test_module.code then .test_id
        """,
        module=f.get("module") or None,
        status=(f.get("status") or "").upper() or None,
        type=(f.get("type") or "").upper() or None,
        priority=f.get("priority") or None,
        tag=f.get("tag") or None,
        id_like=(f.get("id") or "").replace("*", "%").replace("?", "_") or None,
        endpoint=f.get("endpoint") or None,
    )
    return json.loads(raw)


async def list_runs(client: gel.AsyncIOExecutor, limit: int = 20) -> list[dict[str, Any]]:
    raw = await client.query_json(
        """
        select TestRun {
            run_id, status, base_url, started_at, finished_at, abort_reason, log_prefix_uri,
            module_codes := array_agg(.modules.code),
            total := count(.executions),
            passed := count(.executions filter .status = ExecutionStatus.PASSED),
            failed := count(.executions filter .status in {ExecutionStatus.FAILED, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT}),
            duration_s := <float64>duration_get(.finished_at - .started_at, 'totalseconds') if exists .finished_at else <float64>{},
        }
        order by .started_at desc limit <int64>$limit
        """,
        limit=limit,
    )
    return json.loads(raw)


async def run_results(client: gel.AsyncIOExecutor, run_id: str) -> dict[str, Any] | None:
    raw = await client.query_single_json(
        """
        select TestRun {
            run_id, status, base_url, started_at, finished_at, abort_reason, log_prefix_uri,
            executions: {
                test_id := .test_case.test_id, title := .test_case.title, param_index, status, http_status,
                duration_ms, failed_assertions, error_message, log_uri,
            } order by .test_id then .param_index,
        } filter .run_id = <str>$run_id
        """,
        run_id=run_id,
    )
    data = json.loads(raw) if raw else None
    if data:
        for e in data["executions"]:
            e["failed_assertions"] = from_json(e["failed_assertions"])
    return data


async def failing_cases(client: gel.AsyncIOExecutor, last_n_runs: int = 1) -> list[dict[str, Any]]:
    raw = await client.query_json(
        """
        with runs := (select TestRun filter .status = RunStatus.COMPLETED order by .started_at desc limit <int64>$n)
        select TestExecution {
            run_id := .run.run_id, test_id := .test_case.test_id, title := .test_case.title,
            param_index, status, http_status, failed_assertions, error_message,
        }
        filter .run in runs and .status in {ExecutionStatus.FAILED, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT}
        order by .run.started_at desc then .test_id
        """,
        n=last_n_runs,
    )
    rows = json.loads(raw)
    for r in rows:
        r["failed_assertions"] = from_json(r["failed_assertions"])
    return rows

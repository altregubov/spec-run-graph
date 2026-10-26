"""End-to-end against real Gel + MinIO (docker compose), bru and the example SUT.

Uses an isolated Gel branch (`srg_test`) and S3 bucket (`spec-run-telemetry-test`), so your main data is
never touched. LLM calls use test models; production uses the .env model.  Run:  uv run pytest -m integration
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time

import gel
import httpx
import pytest

from fake_llm import synthesizer_model, tool_model
from helpers import ROOT, orders_analysis, orders_report, orders_specs, reference_module
from spec_run_graph.config import Settings
from spec_run_graph.deps import GraphDependencies
from spec_run_graph.knowledge import db, repo, stats
from spec_run_graph.knowledge.repo import CaseFilter
from spec_run_graph.pipeline import stages
from spec_run_graph.pipeline.graph import PipelineGraphState, run_pipeline
from spec_run_graph.telemetry.s3 import TelemetryStore

pytestmark = pytest.mark.integration
BRANCH = "srg_test"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def sut():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "examples.orders.app:app", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            httpx.get(url + "/health", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.2)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
async def settings(tmp_path_factory, sut):
    base = Settings()
    try:
        admin = db.create_client(base)
        await admin.query("select 1")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Gel not available: {exc}")
    if shutil.which(base.bru_bin) is None:
        pytest.skip("bru CLI not installed")
    branches = await admin.query("select sys::Branch.name")
    if BRANCH not in branches:
        await admin.execute(f"create empty branch {BRANCH}")
    await admin.aclose()
    tmp = tmp_path_factory.mktemp("srg")
    s = Settings(
        gel_dsn=base.gel_dsn.rsplit("/", 1)[0] + "/" + BRANCH,
        s3_bucket="spec-run-telemetry-test",
        test_generated_dir=tmp / "test_generated",
        reports_dir=tmp / "reports",
        sut_base_url=sut,
        sut_health_path="/health",
    )
    client = db.create_client(s)
    await db.migrate(client)
    await client.aclose()
    return s


@pytest.fixture
async def deps(settings):
    d = GraphDependencies(settings=settings)
    await db.wipe_all(d.client)
    await d.store.ensure_bucket()
    await d.store.purge_bucket()
    yield d
    await d.aclose()


async def _seed_reference_cases(d: GraphDependencies):
    specs = orders_specs()
    report = orders_report(specs)
    persisted = await repo.persist_specs(d.client, specs, report)
    await repo.insert_analysis(d.client, report, list(persisted.spec_ids.values()))
    for m in report.analysis.modules:
        await repo.upsert_module(d.client, m, report.analysis_id, persisted.spec_ids["Orders API"], None)
    ids = await repo.insert_cases(d.client, "ORD", reference_module().test_cases, persisted)
    return specs, report, persisted, ids


async def test_plan_build_test_end_to_end(deps):
    specs = orders_specs()
    report = orders_report(specs)
    ref = [c.model_dump(exclude={"id", "status"}) for c in reference_module().test_cases]
    deps.model = tool_model({"cases": ref})  # planner answer
    summary = await stages.plan_stage(deps, specs, report)
    assert summary.created["ORD"] == ["TC-ORD-001", "TC-ORD-002"]
    assert not (deps.settings.test_generated_dir).exists(), "plan must not write files"

    # planning again: same cases are recognised as duplicates, ids are not reused
    again = await stages.plan_stage(deps, specs, report)
    assert again.created["ORD"] == []
    assert {t for _, t in again.duplicates} == {c["title"] for c in ref}

    deps.model = synthesizer_model(break_first=True)
    built = await stages.build_stage(deps, CaseFilter())
    assert sorted(built.generated) == ["TC-ORD-001", "TC-ORD-002"] and not built.failed
    root = deps.settings.engine_dir("bruno")
    assert (root / "orders" / "TC-ORD-002a-zero-quantity.yml").exists()
    rows = [await repo.get_case(deps.client, t) for t in ("TC-ORD-001", "TC-ORD-002")]
    assert all(str(r.gen_status) == "GENERATED" and r.gen_model for r in rows)
    # the fake model breaks only its very first answer -> exactly one case needed a retry
    assert sorted(r.generation_attempts for r in rows) == [1, 2]

    rebuilt = await stages.build_stage(deps, CaseFilter())
    assert rebuilt.generated == [] and sorted(rebuilt.skipped_unchanged) == ["TC-ORD-001", "TC-ORD-002"]

    run = await stages.test_stage(deps, CaseFilter())
    assert run.status == "COMPLETED", run.abort_reason
    by = {(o.test_id, o.param_index): o for o in run.outcomes}
    assert by[("TC-ORD-001", 0)].status == "PASSED"
    assert by[("TC-ORD-002", 2)].status == "FAILED" and by[("TC-ORD-002", 2)].http_status == 500
    keys = await deps.store.list_keys(f"runs/{run.run_id}/")
    assert f"runs/{run.run_id}/results.json" in keys
    assert f"runs/{run.run_id}/TC-ORD-002/2/report.json" in keys
    failing = await stats.failing_cases(deps.client)
    assert [(f["test_id"], f["param_index"]) for f in failing] == [("TC-ORD-002", 2)]


async def test_health_check_aborts(deps):
    await _seed_reference_cases(deps)
    deps.model = synthesizer_model()
    await stages.build_stage(deps, CaseFilter())
    run = await stages.test_stage(deps, CaseFilter(), base_url="http://127.0.0.1:9")
    assert run.status == "ABORTED" and run.abort_reason.startswith("ENV_FAILURE")
    assert run.outcomes == []


async def test_spec_change_flags_cases(deps):
    await _seed_reference_cases(deps)
    specs2 = orders_specs("orders_v2")
    report2 = orders_report(specs2, "an-v2")
    report2.changes = await repo.spec_changes(deps.client, specs2, {})
    assert report2.changes[0].changed == ["POST /v1/orders"]
    deps.model = tool_model({"cases": [c.model_dump(exclude={"id", "status"}) for c in reference_module().test_cases]})
    summary = await stages.plan_stage(deps, specs2, report2)
    assert summary.flagged_for_review == 2
    rows = await repo.load_cases(deps.client, CaseFilter(needs_review=True))
    assert {r.test_id for r in rows} == {"TC-ORD-001", "TC-ORD-002"}
    assert all("1.0.0 -> 1.1.0" in r.review_reason for r in rows)
    # steps now point at the new spec version
    eps = await deps.client.query("select distinct TestStep.endpoint.spec.version")
    assert eps == ["1.1.0"]
    deps.model = synthesizer_model()
    built = await stages.build_stage(deps, CaseFilter())
    assert "TC-ORD-001" not in built.generated  # NEEDS_REVIEW cases are skipped


async def test_graph_routes_build_only(deps):
    await _seed_reference_cases(deps)
    deps.model = synthesizer_model()
    state = await run_pipeline(PipelineGraphState(start="build", stop="build"), deps)
    assert state.build and sorted(state.build.generated) == ["TC-ORD-001", "TC-ORD-002"]
    assert state.run is None


async def test_reset_wipes_everything(deps):
    await _seed_reference_cases(deps)
    await deps.store.put_text("runs/x/a.log", "hello")
    deleted = await db.wipe_all(deps.client)
    assert deleted["TestCase"] == 2
    counts = await db.counts(deps.client)
    assert counts["TestEngine"] == 1 and sum(counts.values()) == 1
    purge = await deps.store.purge_bucket()
    assert purge.deleted == 1 and (await deps.store.stats())[0] == 0


async def test_readonly_query_rejects_writes(deps):
    with pytest.raises(gel.errors.EdgeDBError):
        await db.readonly_query(deps.client, "delete TestCase")
    assert json.loads(await db.readonly_query(deps.client, "select count(TestEngine)")) == [1]


async def test_analysis_report_roundtrip(deps, tmp_path):
    specs = orders_specs()
    report = orders_report(specs)
    deps.settings.reports_dir = tmp_path
    path = stages.write_report(deps, report)
    loaded_specs, loaded = stages.load_report(path)
    assert loaded.analysis == orders_analysis(specs) and len(loaded_specs) == 2
    assert "Conflicts" in path.with_suffix(".md").read_text()
    assert isinstance(deps.store, TelemetryStore)


async def test_plan_survives_a_failing_chunk(deps):
    """One module's LLM answer is unusable: other modules are still stored, the failure is reported."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    specs = orders_specs()
    report = orders_report(specs)
    ref = [c.model_dump(exclude={"id", "status"}) for c in reference_module().test_cases]

    def fn(messages, info):
        prompt = str(messages[0].parts[-1].content)
        if '"code": "PROD"' in prompt:  # catalog module: always broken, in every output mode
            if info.output_tools:
                return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"cases": tr')])
            return ModelResponse(parts=[TextPart("not json")])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"cases": ref})])

    deps.model = FunctionModel(fn)
    deps.settings.srg_plan_chunk_size = 10
    summary = await stages.plan_stage(deps, specs, report)
    assert summary.created["ORD"] == ["TC-ORD-001", "TC-ORD-002"]
    assert summary.created["PROD"] == []
    assert [(code, chunk) for code, chunk, _ in summary.failed_chunks] == [("PROD", ["GET /v1/products"])]


async def test_test_then_diagnose(deps):
    """Seeded defect (qty > 100 -> 500) is executed, then triaged: one LLM investigation, note stored in Gel."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    await _seed_reference_cases(deps)
    deps.model = synthesizer_model()
    await stages.build_stage(deps, CaseFilter())
    run = await stages.test_stage(deps, CaseFilter())
    assert any(o.status == "FAILED" for o in run.outcomes)

    def fn(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("http_exchange", {}), ToolCallPart("api_contract", {})])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "category": "APPLICATION_BUG",
                        "root_cause": "qty > 100 returns 500 instead of 400",
                        "evidence": ["HTTP 500"],
                        "recommended_action": "fix validation",
                        "confidence": 0.85,
                    },
                )
            ]
        )

    deps.model = FunctionModel(fn)
    summary = await stages.diagnose_stage(deps, run.run_id, allow_rerun=False)
    assert [(r.test_id, r.param_index, r.category, r.source) for r in summary.rows] == [
        ("TC-ORD-002", 2, "APPLICATION_BUG", "LLM")
    ]
    assert summary.llm_investigations == 1
    detail = await stats.run_results(deps.client, run.run_id)
    diag = next(e["diagnosis"] for e in detail["executions"] if e["status"] == "FAILED")
    assert diag["category"] == "APPLICATION_BUG" and diag["model_name"]
    again = await stages.diagnose_stage(deps, run.run_id)
    assert again.rows == [] and again.skipped_existing == 1


def _defect_case():
    """TC-ORD-001 with a wrong expectation: a new order is DRAFT, the test claims CONFIRMED (a TEST_DEFECT)."""
    case = reference_module().test_cases[0].model_copy(deep=True)
    case.steps[0].assertions[0].check = "response.body.data.status == 'CONFIRMED'"
    return case


def _diagnose_as_defect(messages, info):
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    return ModelResponse(
        parts=[
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "category": "TEST_DEFECT",
                    "root_cause": "a new order is DRAFT; the test expects CONFIRMED",
                    "evidence": ["response.body.data.status = DRAFT"],
                    "recommended_action": "fix the expectation",
                    "suggested_fix": "expect 'DRAFT'",
                    "confidence": 0.9,
                },
            )
        ]
    )


def _heal_to(status_literal: str):
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    def fn(messages, info):
        fixed = _defect_case().model_dump(mode="json", exclude={"id", "status"})
        fixed["steps"][0]["assertions"][0]["check"] = f"response.body.data.status == '{status_literal}'"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"case": fixed, "change_summary": f"expect {status_literal} for a new order"},
                )
            ]
        )

    return fn


async def _run_defect_scenario(deps, heal_fn):
    from fake_llm import routed_model

    specs = orders_specs()
    report = orders_report(specs)
    persisted = await repo.persist_specs(deps.client, specs, report)
    await repo.insert_analysis(deps.client, report, list(persisted.spec_ids.values()))
    for m in report.analysis.modules:
        await repo.upsert_module(deps.client, m, report.analysis_id, persisted.spec_ids["Orders API"], None)
    from spec_run_graph.domain.ir import normalize

    await repo.insert_cases(deps.client, "ORD", [normalize(_defect_case())], persisted)
    deps.model = routed_model(files=synthesizer_model(), category=_diagnose_as_defect, case=heal_fn)
    state = await run_pipeline(PipelineGraphState(start="build", stop="heal", rerun_failures=False), deps)
    return state


async def test_heal_verified(deps):
    state = await _run_defect_scenario(deps, _heal_to("DRAFT"))
    assert [o.status for o in state.run.outcomes] == ["FAILED"]
    assert [r.category for r in state.diagnosis.rows] == ["TEST_DEFECT"]
    assert [(r.test_id, r.status) for r in state.healing.rows] == [("TC-ORD-001", "VERIFIED")]
    row = await repo.get_case(deps.client, "TC-ORD-001")
    assert str(row.design_status) == "READY" and "healed" in row.tags
    assert row.steps[0].assertions and "DRAFT" in row.steps[0].assertions
    heals = await repo.heal_history(deps.client, "TC-ORD-001")
    assert [h["status"] for h in heals] == ["VERIFIED"] and heals[0]["verify_run_id"]
    # verification runs never become the default run for diagnose/heal
    assert await repo.latest_run_id(deps.client) == state.run.run_id


async def test_heal_rolled_back_when_still_failing(deps):
    state = await _run_defect_scenario(deps, _heal_to("CANCELLED"))  # a "fix" that is still wrong
    assert [(r.test_id, r.status, r.attempts) for r in state.healing.rows] == [("TC-ORD-001", "ROLLED_BACK", 2)]
    row = await repo.get_case(deps.client, "TC-ORD-001")
    assert str(row.design_status) == "NEEDS_REVIEW" and "auto-heal failed" in row.review_reason
    assert "CONFIRMED" in row.steps[0].assertions  # original content restored
    assert [h["status"] for h in await repo.heal_history(deps.client, "TC-ORD-001")] == ["ROLLED_BACK"] * 2

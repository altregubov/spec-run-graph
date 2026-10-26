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
    assert again.created["ORD"] == [] and len(again.duplicates) == 2

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

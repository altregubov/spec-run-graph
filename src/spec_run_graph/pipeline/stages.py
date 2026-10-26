"""Pipeline stages: analyze -> plan (Gel) -> build (Bruno) -> test (bru run). Every LLM call uses the .env model."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from spec_run_graph.agents import analyzer, planner, synthesizer
from spec_run_graph.config import dotenv_values
from spec_run_graph.deps import GraphDependencies
from spec_run_graph.domain.analysis import AnalysisReport, SpecRef
from spec_run_graph.domain.ir import REF_RE, CaseDraft, TestCaseIR, ValidationContext, fingerprint, http_steps, ir_hash
from spec_run_graph.engines.bruno import scaffold
from spec_run_graph.engines.bruno.naming import FILE_RE, plan_files
from spec_run_graph.engines.bruno.runner import MANIFEST_NAME, aggregate, run_bru
from spec_run_graph.knowledge import repo
from spec_run_graph.knowledge.repo import CaseFilter
from spec_run_graph.llm import model_name
from spec_run_graph.specs.loader import load_specs
from spec_run_graph.specs.models import LoadedSpec
from spec_run_graph.telemetry.s3 import case_prefix, run_prefix

# ---------------------------------------------------------------- analyze


async def analyze_stage(deps: GraphDependencies, paths: list[Path]) -> tuple[list[LoadedSpec], AnalysisReport]:
    specs = load_specs(paths)
    deps.console.print(
        f"[bold]Analyzing {len(specs)} specs jointly[/] with [cyan]{model_name()}[/]: "
        + ", ".join(f"{s.name} ({s.spec_type})" for s in specs)
    )
    analysis = await analyzer.analyze(specs, model=deps.model)
    prd_reqs: dict[str, dict[str, str]] = {}
    for r in analysis.requirements:
        prd_reqs.setdefault(r.spec, {})[r.req_id] = r.description
    try:
        changes = await repo.spec_changes(deps.client, specs, prd_reqs)
    except Exception as exc:  # Gel down -> analysis still useful
        deps.console.print(f"[yellow]Skipping version diff (Gel unavailable: {exc.__class__.__name__})[/]")
        changes = []
    report = AnalysisReport(
        analysis_id=f"an-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        created_at=datetime.now(UTC),
        model=model_name(),
        specs=[
            SpecRef(
                name=s.name,
                spec_type=s.spec_type,
                version=s.version,
                source_path=s.source_path,
                content_hash=s.content_hash,
            )
            for s in specs
        ],
        analysis=analysis,
        changes=changes,
    )
    write_report(deps, report)
    return specs, report


def write_report(deps: GraphDependencies, report: AnalysisReport) -> Path:
    out = deps.settings.reports_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report.analysis_id}.json"
    path.write_text(report.model_dump_json(indent=2))
    (out / f"{report.analysis_id}.md").write_text(report_markdown(report))
    return path


def report_markdown(report: AnalysisReport) -> str:
    a = report.analysis
    lines = [
        f"# Spec analysis {report.analysis_id}",
        "",
        f"Model: `{report.model}` - {report.created_at:%Y-%m-%d %H:%M} UTC",
        "",
        "## Specifications",
        *[f"- **{s.name}** ({s.spec_type}, v{s.version}) - `{s.source_path}`" for s in report.specs],
        "",
        "## Summary",
        a.summary,
        "",
        "## Modules",
    ]
    for m in a.modules:
        lines.append(f"- **{m.code}** `{m.folder}/` {m.domain}: {', '.join(m.endpoints)}")
    lines += ["", "## Requirements", "| id | spec | endpoints | description |", "|---|---|---|---|"]
    lines += [f"| {r.req_id} | {r.spec} | {', '.join(r.endpoints) or '-'} | {r.description} |" for r in a.requirements]
    lines += ["", "## Conflicts"]
    lines += [
        f"- **[{c.severity}] {c.topic}**: {c.source_a}: _{c.statement_a}_ vs {c.source_b}: _{c.statement_b}_. {c.recommendation}"
        for c in a.conflicts
    ] or ["- none"]
    lines += ["", "## Gaps"] + (
        [f"- **{g.kind}**: {g.description} ({', '.join(g.refs)})" for g in a.gaps] or ["- none"]
    )
    lines += ["", "## Risks"] + ([f"- {r}" for r in a.risks] or ["- none"])
    if report.changes:
        lines += ["", "## Changes vs previous version"]
        for c in report.changes:
            lines.append(
                f"- **{c.spec_name}** {c.from_version} -> {c.to_version}: added {c.added}, removed {c.removed}, changed {c.changed}"
            )
    return "\n".join(lines) + "\n"


def load_report(path: Path) -> tuple[list[LoadedSpec], AnalysisReport]:
    report = AnalysisReport.model_validate_json(path.read_text())
    specs = load_specs([Path(s.source_path) for s in report.specs])
    by_name = {s.name: s for s in specs}
    for ref in report.specs:
        s = by_name.get(ref.name)
        if s is None or s.content_hash != ref.content_hash:
            raise ValueError(
                f"spec {ref.name!r} ({ref.source_path}) changed since the analysis - run `srg analyze` again"
            )
    return specs, report


# ---------------------------------------------------------------- plan


@dataclass
class PlanSummary:
    created: dict[str, list[str]] = field(default_factory=dict)
    dropped_invalid: list[tuple[str, str, list[str]]] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    flagged_for_review: int = 0
    deprecated: int = 0


async def plan_stage(
    deps: GraphDependencies, specs: list[LoadedSpec], report: AnalysisReport, replace: bool = False
) -> PlanSummary:
    summary = PlanSummary()
    client = deps.client
    persisted = await repo.persist_specs(client, specs, report)
    await repo.insert_analysis(client, report, list(persisted.spec_ids.values()))
    if persisted.new_versions:
        summary.flagged_for_review = await repo.apply_change_impact(client, persisted, report.changes)

    endpoint_owner = {e.key: s.name for s in specs for e in s.endpoints}
    prd = next((s.name for s in specs if s.spec_type == "REQUIREMENTS_DOC"), None)
    for m in report.analysis.modules:
        target = next((endpoint_owner[k] for k in m.endpoints if k in endpoint_owner), None)
        await repo.upsert_module(
            client, m, report.analysis_id, persisted.spec_ids.get(target or ""), persisted.spec_ids.get(prd or "")
        )

    endpoints = {e.key: e for s in specs for e in s.endpoints}
    sem = asyncio.Semaphore(deps.settings.srg_llm_concurrency)

    async def one(m) -> None:
        if replace:
            summary.deprecated += await repo.deprecate_cases(client, m.code)
        existing = await repo.existing_cases(client, m.code)
        ctx = ValidationContext(
            endpoint_paths=persisted.endpoint_paths,
            requirement_ids=set(persisted.requirement_ids),
            actors={a.name for a in m.actors},
            fixtures={f.name for f in m.fixtures},
        )
        async with sem:
            deps.console.print(
                f"  planning [bold]{m.code}[/] ({len(m.endpoints)} endpoints, {len(existing)} existing cases)..."
            )
            drafts, dropped = await planner.plan_module(
                report.analysis,
                m,
                endpoints,
                ctx,
                [{k: v for k, v in e.items() if k != "fingerprint"} for e in existing],
                model=deps.model,
            )
        summary.dropped_invalid += [(m.code, title, errs) for title, errs in dropped]
        seen = {e["fingerprint"] for e in existing}
        fresh: list[CaseDraft] = []
        for d in drafts:
            fp = fingerprint(m.code, d, persisted.endpoint_paths)
            if fp in seen:
                summary.duplicates.append((m.code, d.title))
                continue
            seen.add(fp)
            fresh.append(d)
        summary.created[m.code] = await repo.insert_cases(client, m.code, fresh, persisted)

    await asyncio.gather(*(one(m) for m in report.analysis.modules))
    return summary


# ---------------------------------------------------------------- build


@dataclass
class BuildSummary:
    generated: list[str] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    failed: list[tuple[str, list[str]]] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)


def _secrets_for(module: dict[str, Any], case: TestCaseIR) -> set[str]:
    names = set()
    for text in [json.dumps(module["actors"]), json.dumps(module["fixtures"]), case.model_dump_json()]:
        names |= {n for ns, n in REF_RE.findall(text) if ns == "env"}
        names |= set(re.findall(r'"env":\s*"([A-Z][A-Z0-9_]*)"', text))
    env = dotenv_values()
    return {env[n] for n in names if env.get(n) and len(env[n]) >= 4}


def _content_hash(docs: list[dict]) -> str:
    stripped = []
    for d in docs:
        d = json.loads(json.dumps(d))
        (d.get("info") or {}).pop("seq", None)
        stripped.append(d)
    return hashlib.sha256(json.dumps(stripped, sort_keys=True).encode()).hexdigest()[:32]


def _case_files(folder: Path, test_id: str) -> list[Path]:
    out = []
    for p in folder.glob(f"{test_id}*.yml"):
        m = FILE_RE.match(p.name)
        if m and m.group(1) == test_id:
            out.append(p)
    return out


def _sort_key(name: str) -> tuple:
    m = FILE_RE.match(name)
    if not m:
        return (1, name)
    num = int(m.group(1).rsplit("-", 1)[1])
    return (0, m.group(1).rsplit("-", 1)[0], num, m.group(2) or "", int(m.group(3) or 0), name)


def renumber_folder(folder: Path) -> None:
    """Deterministic info.seq across the folder: test number, param letter, step."""
    files = sorted((p for p in folder.glob("TC-*.yml")), key=lambda p: _sort_key(p.name))
    for seq, p in enumerate(files, start=1):
        doc = yaml.safe_load(p.read_text())
        if doc.get("info", {}).get("seq") != seq:
            doc.setdefault("info", {})["seq"] = seq
            p.write_text(scaffold.dump_yaml(doc))


def manifest_path(deps: GraphDependencies) -> Path:
    return deps.settings.test_generated_dir / MANIFEST_NAME


def load_manifest(deps: GraphDependencies) -> dict[str, dict[str, Any]]:
    p = manifest_path(deps)
    return json.loads(p.read_text()) if p.exists() else {}


async def build_stage(
    deps: GraphDependencies,
    flt: CaseFilter,
    force: bool = False,
    include_needs_review: bool = False,
    dry_run: bool = False,
) -> BuildSummary:
    summary = BuildSummary()
    client = deps.client
    root = deps.settings.engine_dir("bruno")
    statuses = ["DRAFT", "READY"] + (["NEEDS_REVIEW"] if include_needs_review else [])
    rows = await repo.load_cases(client, CaseFilter(flt.id_glob, flt.tags, statuses))
    codes = await repo.list_modules(client)
    modules = {c: await repo.module_context(client, c) for c in codes}
    if not dry_run:
        scaffold.write_scaffold(root, list(modules.values()), deps.settings.sut_base_url)
    manifest = load_manifest(deps)
    sem = asyncio.Semaphore(deps.settings.srg_llm_concurrency)
    model = model_name()

    async def one(row) -> None:
        case = repo.row_to_ir(row)
        module = modules[row.test_module.code]
        folder = root / module["folder"]
        planned = plan_files(case)
        ihash = ir_hash(case, {"module": module, "model": model})
        files_exist = all((folder / p.filename).exists() for p in planned)
        if not force and row.ir_hash == ihash and str(row.gen_status) == "GENERATED" and files_exist:
            summary.skipped_unchanged.append(case.id)
            return
        if any(s.poll for s, _ in http_steps(case)):
            err = "unsupported: poll (rendering of polling steps is deferred)"
            summary.failed.append((case.id, [err]))
            if not dry_run:
                await repo.record_generation(
                    client,
                    case.id,
                    ok=False,
                    script_path=None,
                    script_hash=None,
                    ir_hash=ihash,
                    attempts=0,
                    model=model,
                    error=err,
                )
            return
        endpoints = await repo.endpoint_schemas(client, list(case.coverage.endpoints))
        async with sem:
            deps.console.print(f"  implementing [bold]{case.id}[/] ({len(planned)} files)...")
            parsed, sd = await synthesizer.synthesize(
                case, module, planned, endpoints, _secrets_for(module, case), model=deps.model
            )
        if parsed is None:
            summary.failed.append((case.id, sd.last_errors))
            if not dry_run:
                await repo.record_generation(
                    client,
                    case.id,
                    ok=False,
                    script_path=None,
                    script_hash=None,
                    ir_hash=ihash,
                    attempts=sd.attempts,
                    model=model,
                    error="\n".join(sd.last_errors)[:4000],
                )
            return
        summary.generated.append(case.id)
        if dry_run:
            return
        folder.mkdir(parents=True, exist_ok=True)
        for old in _case_files(folder, case.id):
            old.unlink()
        docs = []
        for p in planned:
            doc = parsed[p.filename]
            (folder / p.filename).write_text(scaffold.dump_yaml(doc))
            docs.append(doc)
            manifest[f"{module['folder']}/{p.filename}"] = {
                "test_id": p.test_id,
                "param_index": p.param_index,
                "step_seq": p.step_seq,
                "file_step": p.file_step,
                "is_cleanup": p.is_cleanup,
            }
        await repo.record_generation(
            client,
            case.id,
            ok=True,
            script_path=f"{module['folder']}/{planned[0].filename}",
            script_hash=_content_hash(docs),
            ir_hash=ihash,
            attempts=sd.attempts,
            model=model,
            error=None,
        )

    await asyncio.gather(*(one(r) for r in rows))
    if dry_run:
        return summary

    # full build: remove files of cases that are no longer selected (deprecated / needs review / deleted)
    if not flt.id_glob and not flt.tags:
        keep = {r.test_id for r in rows}
        for path in root.glob("*/TC-*.yml"):
            m = FILE_RE.match(path.name)
            if m and m.group(1) not in keep:
                summary.removed_files.append(str(path.relative_to(root)))
                path.unlink()
                manifest.pop(str(path.relative_to(root)), None)
    for m in modules.values():
        if (root / m["folder"]).exists():
            renumber_folder(root / m["folder"])
    manifest = {k: v for k, v in manifest.items() if (root / k).exists()}
    manifest_path(deps).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return summary


# ---------------------------------------------------------------- test


@dataclass
class RunSummary:
    run_id: str
    status: str
    outcomes: list[Any] = field(default_factory=list)
    abort_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "COMPLETED" and all(o.status == "PASSED" for o in self.outcomes)


async def health_check(url: str, retries: int = 3) -> str | None:
    """None when reachable (any HTTP status < 500), else the last error."""
    last = "unknown error"
    async with httpx.AsyncClient(timeout=5) as http:
        for attempt in range(retries):
            try:
                resp = await http.get(url)
                if resp.status_code < 500:
                    return None
                last = f"HTTP {resp.status_code}"
            except httpx.HTTPError as exc:
                last = f"{exc.__class__.__name__}: {exc}"
            await asyncio.sleep(0.5 * 2**attempt)
    return last


async def test_stage(
    deps: GraphDependencies,
    flt: CaseFilter,
    env_name: str = "local",
    base_url: str | None = None,
    skip_health_check: bool = False,
) -> RunSummary:
    s = deps.settings
    client, store = deps.client, deps.store
    root = s.engine_dir("bruno")
    base_url = base_url or s.sut_base_url
    run_id = f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    rows = [
        r
        for r in await repo.load_cases(client, CaseFilter(flt.id_glob, flt.tags, None))
        if str(r.gen_status) == "GENERATED" and str(r.design_status) not in ("DEPRECATED",)
    ]
    summary = RunSummary(run_id=run_id, status="PENDING")
    codes = sorted({r.test_module.code for r in rows})
    await store.ensure_bucket()
    await repo.create_run(client, run_id, codes, env_name, base_url, store.uri(run_prefix(run_id)))
    if not rows:
        summary.status, summary.abort_reason = "ABORTED", "no generated test cases match (run `srg build` first)"
        await repo.finish_run(client, run_id, "ABORTED", summary.abort_reason)
        return summary

    if not skip_health_check:
        err = await health_check(base_url.rstrip("/") + s.sut_health_path)
        if err:
            summary.status, summary.abort_reason = "ABORTED", f"ENV_FAILURE: {base_url} unreachable ({err})"
            await repo.finish_run(client, run_id, "ABORTED", summary.abort_reason)
            return summary

    manifest = load_manifest(deps)
    expected: dict[tuple[str, int], list[str]] = {}
    for f, meta in manifest.items():
        expected.setdefault((meta["test_id"], meta["param_index"]), []).append(f)
    selected = {r.test_id for r in rows}
    expected = {
        k: sorted(v, key=lambda n: _sort_key(n.rsplit("/", 1)[-1])) for k, v in expected.items() if k[0] in selected
    }
    hashes = {r.test_id: r.script_hash for r in rows}
    folder_of = {r.test_id: r.test_module.folder for r in rows}
    for tid in selected:
        files = [root / f for (t, _), fs in expected.items() if t == tid for f in fs]
        if not files or not all(p.exists() for p in files):
            summary.warnings.append(f"{tid}: files missing - run `srg build`")
            continue
        docs = [yaml.safe_load(p.read_text()) for p in sorted(files, key=lambda p: _sort_key(p.name))]
        if _content_hash(docs) != hashes[tid]:
            summary.warnings.append(f"{tid}: files differ from the last build (edited by hand or stale)")

    if flt.id_glob or flt.tags:
        paths = sorted(
            {f for fs in expected.values() for f in fs},
            key=lambda n: (n.split("/")[0], _sort_key(n.rsplit("/", 1)[-1])),
        )
    else:
        paths = sorted({folder_of[t] for t in selected})
    out_dir = s.reports_dir / "runs" / run_id
    deps.console.print(f"[bold]Running {len(selected)} cases[/] ({len(paths)} paths) against {base_url} ...")
    result = await run_bru(
        s.bru_bin, root, paths, env_name, {"baseUrl": base_url}, dotenv_values(), out_dir, s.bru_timeout_sec
    )

    prefix = run_prefix(run_id)
    await store.put_text(prefix + "bru.stdout", result.stdout)
    await store.put_text(prefix + "bru.stderr", result.stderr)
    await store.put_text(prefix + "results.json", result.results_json, "application/json")
    if result.results_xml:
        await store.put_text(prefix + "results.xml", result.results_xml, "application/xml")

    outcomes = aggregate(result.results, manifest, expected, result.timed_out)
    for (tid, pidx), oc in sorted(outcomes.items()):
        key = case_prefix(run_id, tid, pidx)
        log_uri = await store.put_text(
            key + "report.json", json.dumps(oc.raw, indent=1, default=str), "application/json"
        )
        await repo.insert_execution(
            client,
            run_id,
            tid,
            pidx,
            status=oc.status,
            duration_ms=oc.duration_ms,
            http_status=oc.http_status,
            failed_assertions=oc.failed or None,
            error_message=oc.error_message,
            log_uri=store.uri(key) if log_uri else None,
            script_hash=hashes.get(tid),
        )
    summary.outcomes = [outcomes[k] for k in sorted(outcomes)]
    if result.timed_out:
        summary.status, summary.abort_reason = "ABORTED", f"bru run timed out after {s.bru_timeout_sec}s"
    elif not result.results and result.returncode not in (0, 1):
        summary.status, summary.abort_reason = (
            "ABORTED",
            f"bru failed (exit {result.returncode}): {result.stderr[-500:]}",
        )
    else:
        summary.status = "COMPLETED"
    await repo.finish_run(client, run_id, summary.status, summary.abort_reason)
    return summary

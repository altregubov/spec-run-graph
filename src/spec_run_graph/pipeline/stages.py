"""Pipeline stages: analyze -> plan (Gel) -> build (Bruno) -> test (bru run) -> diagnose.

Every LLM call uses the .env model."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from spec_run_graph.agents import analyzer, diagnostics, healer, planner, synthesizer
from spec_run_graph.config import dotenv_values
from spec_run_graph.deps import GraphDependencies
from spec_run_graph.domain.analysis import AnalysisReport, SpecRef
from spec_run_graph.domain.ir import (
    BODY_ERROR_STATUSES,
    REF_RE,
    CaseDraft,
    TestCaseIR,
    ValidationContext,
    declared_env_vars,
    fingerprint,
    http_steps,
    ir_hash,
    missing_required,
    normalize,
    resolve_endpoint,
    validate_case,
)
from spec_run_graph.engines.bruno import compiler, scaffold
from spec_run_graph.engines.bruno.naming import FILE_RE, plan_files
from spec_run_graph.engines.bruno.normalize import normalize_text
from spec_run_graph.engines.bruno.runner import MANIFEST_NAME, aggregate, run_bru
from spec_run_graph.engines.bruno.validate import validate_files
from spec_run_graph.knowledge import repo
from spec_run_graph.knowledge.repo import CaseFilter
from spec_run_graph.llm import model_name
from spec_run_graph.pipeline.observe import observation_digest, observe
from spec_run_graph.specs.contract import page_limits, required_body_fields, response_facts
from spec_run_graph.specs.loader import load_specs
from spec_run_graph.specs.models import LoadedSpec
from spec_run_graph.specs.openapi import endpoint_digest
from spec_run_graph.telemetry.s3 import case_prefix, run_prefix

# ---------------------------------------------------------------- analyze


async def analyze_stage(deps: GraphDependencies, paths: list[Path]) -> tuple[list[LoadedSpec], AnalysisReport]:
    skipped: list[Path] = []
    specs = load_specs(paths, skipped)
    for f in skipped:
        deps.console.print(f"[yellow]Skipping {f}: not an OpenAPI 3.x / Swagger 2.0 document[/]")
    deps.console.print(
        f"[bold]Analyzing {len(specs)} specs jointly[/] with [cyan]{model_name()}[/]: "
        + ", ".join(f"{s.name} ({s.spec_type})" for s in specs)
    )
    try:
        known_actors = await repo.existing_actors(deps.client)
    except Exception:  # noqa: BLE001 - Gel down: analysis still works
        known_actors = {}
    analysis = await analyzer.analyze(specs, model=deps.model, existing_actors=known_actors)
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
    unset_fixtures: list[str] = field(default_factory=list)  # "MODULE.fixture (VAR)" dropped: VAR is not set
    deprecated: int = 0
    failed_chunks: list[tuple[str, list[str], str]] = field(default_factory=list)  # (module, endpoints, error)
    refreshed: list[str] = field(default_factory=list)  # modules updated by refresh_only
    recipes: dict[str, str] = field(default_factory=dict)  # recipe name -> status (designed in this run)
    observed: int = 0  # GET samples taken from the SUT before planning
    rule_hits: dict[str, int] = field(default_factory=dict)  # validation rule -> times the planner broke it


# ---------------------------------------------------------------- setup recipes


def _recipe_module(modules: list[Any]) -> dict[str, Any]:
    """A synthetic module context with every actor/fixture (recipes are shared by all modules)."""
    actors: dict[str, Any] = {}
    fixtures: dict[str, Any] = {}
    for m in modules:
        for a in m.actors:
            actors.setdefault(a.name, a.auth.model_dump(exclude_none=True))
        for f in m.fixtures:
            fixtures.setdefault(f.name, {"env": f.env} if f.env else f.value)
    return {
        "code": "RCP",
        "folder": "srg-recipe-check",
        "domain": "setup recipe check",
        "base_path": next((m.base_path for m in modules if m.base_path), ""),
        "default_headers": {},
        "preconditions": [],
        "actors": actors,
        "fixtures": fixtures,
    }


async def verify_recipe(
    deps: GraphDependencies, recipe: Any, module: dict[str, Any], endpoints: dict[str, Any]
) -> tuple[bool, str]:
    """Run the recipe once against the SUT (compiled like any case, in a scratch collection)."""
    from spec_run_graph.domain.recipes import recipe_as_case

    s = deps.settings
    case = TestCaseIR.model_validate({**normalize(recipe_as_case(recipe)).model_dump(), "id": "TC-RCP-001"})
    planned = plan_files(case)
    try:
        docs = compiler.compile_case(case, module, planned, endpoints)
    except compiler.Unsupported as exc:
        return False, f"cannot compile: {exc}"
    root = s.reports_dir / "recipes" / recipe.name
    shutil.rmtree(root, ignore_errors=True)
    scaffold.write_scaffold(root, [module], s.sut_base_url)
    folder = root / module["folder"]
    manifest, files = {}, []
    for seq, p in enumerate(planned, start=1):
        docs[p.filename]["info"]["seq"] = seq
        (folder / p.filename).write_text(scaffold.dump_yaml(docs[p.filename]))
        rel = f"{module['folder']}/{p.filename}"
        files.append(rel)
        manifest[rel] = {"test_id": case.id, "param_index": 0, "step_seq": p.step_seq, "is_cleanup": p.is_cleanup}
    res = await run_bru(
        s.bru_bin,
        root,
        [module["folder"]],
        scaffold.ENV_NAME,
        {"baseUrl": s.sut_base_url},
        dotenv_values(),
        root / "out",
        300,
    )
    oc = aggregate(res.results, manifest, {(case.id, 0): files}, res.timed_out)[(case.id, 0)]
    if oc.status == "PASSED":
        return True, "passed against " + s.sut_base_url
    detail = "; ".join(
        f"{f.get('file', '').rsplit('/', 1)[-1]}: {f.get('assertion') or f.get('test') or ''} -> {f.get('error')}"
        for f in oc.failed[:5]
    )
    return False, f"{oc.status} (HTTP {oc.http_status}) {detail or oc.error_message or ''}"[:2000]


async def observe_stage(deps: GraphDependencies, report: AnalysisReport, endpoints: dict[str, Any]) -> int:
    """Store read-only GET samples of the SUT (planner context: real values, casing, formats), trimmed to the
    fields the contract guarantees so optional ones (requestId, pagination) do not invite assertions."""
    guaranteed = {
        k: paths
        for k, e in endpoints.items()
        for st, paths in response_facts(e.response_schemas).items()
        if st == "200"
    }
    if not deps.settings.srg_observe or await health_check(deps.settings.sut_base_url) is not None:
        return 0
    try:
        samples = await observe(
            deps.settings.sut_base_url, report.analysis.modules, {**os.environ, **dotenv_values()}, guaranteed
        )
    except Exception as exc:  # noqa: BLE001 - observations are optional context
        deps.console.print(f"  [yellow]observing the SUT failed[/]: {exc.__class__.__name__}")
        return 0
    for smp in samples:
        await repo.save_observation(deps.client, smp.endpoint, smp.actor, smp.http_status, smp.body)
    return len(samples)


async def recipes_stage(
    deps: GraphDependencies,
    report: AnalysisReport,
    endpoints: dict[str, Any],
    endpoint_paths: dict[str, str],
    observed: str = "",
) -> dict[str, str]:
    """Design setup recipes once per analysis and keep those that pass against the SUT."""
    from spec_run_graph.agents import recipes as recipe_agent

    client = deps.client
    if not deps.settings.srg_recipes or await repo.recipes_designed_for(client, report.analysis_id):
        return {}
    modules = report.analysis.modules
    module = _recipe_module(modules)
    ctx = ValidationContext(
        endpoint_paths=endpoint_paths,
        actors=set(module["actors"]),
        fixtures=set(module["fixtures"]),
        env_vars=declared_env_vars(
            *[a for m in modules for a in m.actors if a.auth.type != "login"], *[f for m in modules for f in m.fixtures]
        ),
        response_schemas={k: e.response_schemas for k, e in endpoints.items()},
        request_schemas={k: e.request_schema for k, e in endpoints.items()},
        page_limits=page_limits(endpoints),
    )
    schemas = {k: {"request_schema": e.request_schema, "responses": e.response_schemas} for k, e in endpoints.items()}
    reachable = await health_check(deps.settings.sut_base_url) is None
    prompt = recipe_agent.build_prompt(
        report.analysis.summary,
        module["actors"],
        module["fixtures"],
        "\n".join(endpoint_digest(e) for _, e in sorted(endpoints.items())),
        requirements=report.analysis.requirements,
        observed=observed,
    )
    result: dict[str, str] = {}
    feedback = ""
    for _round in range(2):  # one redesign round for recipes that fail against the API
        deps.console.print("  designing setup recipes...")
        try:
            designed, _ = await recipe_agent.design(prompt + feedback, ctx, model=deps.model)
        except Exception as exc:  # noqa: BLE001 - recipes are an optimisation; planning works without them
            deps.console.print(f"  [yellow]recipe design failed[/]: {exc.__class__.__name__}")
            break
        failed = []
        for r in designed:
            if result.get(r.name) == "VERIFIED":
                continue
            ok, detail = (await verify_recipe(deps, r, module, schemas)) if reachable else (False, "SUT unreachable")
            status = "VERIFIED" if ok else ("FAILED" if reachable else "UNVERIFIED")
            result[r.name] = status
            await repo.save_recipe(client, r, status, detail, report.analysis_id, model_name())
            deps.console.print(f"  recipe [bold]{r.name}[/]: {status} {'' if ok else detail[:160]}")
            if not ok:
                failed.append(f"- {r.name}: {detail}")
        if not failed or not reachable:
            break
        feedback = "\n\nThese recipes FAILED when run against the API; fix them (return all recipes):\n" + "\n".join(
            failed
        )
    return result


def add_recipe_actors(modules: list[Any], recipes: dict[str, Any]) -> list[Any]:
    """Give every module the actors its recipe steps run as (e.g. admin creating a dedicated customer for a
    customer-only module); otherwise no recipe would validate there. Returns the changed modules."""
    defs = {a.name: a for m in modules for a in m.actors}
    needed = {st.actor for r in recipes.values() for st in [*r.steps, *r.cleanup] if st.actor}
    changed = []
    for m in modules:
        missing = sorted((needed - {a.name for a in m.actors}) & set(defs))
        if missing:
            m.actors.extend(
                defs[n].model_copy(update={"description": f"{defs[n].description} (setup recipes)".strip()})
                for n in missing
            )
            changed.append(m)
    return changed


async def plan_stage(
    deps: GraphDependencies,
    specs: list[LoadedSpec],
    report: AnalysisReport,
    replace: bool = False,
    refresh_only: bool = False,
) -> PlanSummary:
    """Store specs/analysis/modules in Gel and plan new test cases (refresh_only: no new cases, e.g. to
    update actors/fixtures after a re-analysis)."""
    summary = PlanSummary()
    client = deps.client
    persisted = await repo.persist_specs(client, specs, report)
    await repo.insert_analysis(client, report, list(persisted.spec_ids.values()))
    if persisted.new_versions:
        summary.flagged_for_review = await repo.apply_change_impact(client, persisted, report.changes)

    endpoint_owner = {e.key: s.name for s in specs for e in s.endpoints}
    prd = next((s.name for s in specs if s.spec_type == "REQUIREMENTS_DOC"), None)

    async def store_module(m) -> None:
        target = next((endpoint_owner[k] for k in m.endpoints if k in endpoint_owner), None)
        await repo.upsert_module(
            client, m, report.analysis_id, persisted.spec_ids.get(target or ""), persisted.spec_ids.get(prd or "")
        )

    available = {**os.environ, **dotenv_values()}
    for m in report.analysis.modules:
        # an env-backed fixture whose variable is not set would send an empty value (ENV_FAILURE at run time);
        # without it the planner looks the value up instead
        unset = [f for f in m.fixtures if f.env and not available.get(f.env)]
        summary.unset_fixtures += [f"{m.code}.{f.name} ({f.env})" for f in unset]
        m.fixtures = [f for f in m.fixtures if f not in unset]
        await store_module(m)

    if refresh_only:
        summary.refreshed = [m.code for m in report.analysis.modules]
        return summary
    endpoints = {e.key: e for s in specs for e in s.endpoints}
    summary.observed = await observe_stage(deps, report, endpoints)
    observed = observation_digest(await repo.observations(client))
    summary.recipes = await recipes_stage(deps, report, endpoints, persisted.endpoint_paths, observed)
    recipes = await repo.load_recipes(client)
    for m in add_recipe_actors(report.analysis.modules, recipes):
        await store_module(m)
    sem = asyncio.Semaphore(deps.settings.srg_llm_concurrency)

    chunk_size = max(1, deps.settings.srg_plan_chunk_size)

    async def plan_chunk(m, chunk: list[str], ctx: ValidationContext, existing: list[dict]) -> list[CaseDraft]:
        async with sem:
            deps.console.print(f"  planning [bold]{m.code}[/] {chunk} ({len(existing)} existing cases)...")
            try:
                drafts, dropped = await planner.plan_module(
                    report.analysis,
                    m.model_copy(update={"endpoints": chunk}),
                    endpoints,
                    ctx,
                    existing,
                    model=deps.model,
                    recipes=recipes,
                    lessons=lessons,
                    observed=observed,
                    rule_hits=summary.rule_hits,
                )
            except Exception as exc:  # one bad LLM answer must not lose the other modules
                summary.failed_chunks.append((m.code, chunk, f"{exc.__class__.__name__}: {str(exc)[:300]}"))
                deps.console.print(f"  [red]planning {m.code} {chunk} failed[/]: {exc.__class__.__name__}")
                return []
        summary.dropped_invalid += [(m.code, title, errs) for title, errs in dropped]
        return drafts

    lessons = [r.text for r in await repo.lessons(client)]

    async def one(m) -> None:
        if replace:
            summary.deprecated += await repo.deprecate_cases(client, m.code)
        existing = await repo.existing_cases(client, m.code)
        ctx = ValidationContext(
            endpoint_paths=persisted.endpoint_paths,
            requirement_ids=set(persisted.requirement_ids),
            actors={a.name for a in m.actors},
            fixtures={f.name for f in m.fixtures},
            env_vars=declared_env_vars(*[a for a in m.actors if a.auth.type != "login"], *m.fixtures),
            response_schemas={k: e.response_schemas for k, e in endpoints.items()},
            request_schemas={k: e.request_schema for k, e in endpoints.items()},
            page_limits=page_limits(endpoints),
        )
        shown = [{k: v for k, v in e.items() if k != "fingerprint"} for e in existing]
        chunks = [m.endpoints[i : i + chunk_size] for i in range(0, len(m.endpoints), chunk_size)] or [[]]
        results = await asyncio.gather(*(plan_chunk(m, c, ctx, shown) for c in chunks))
        seen = {e["fingerprint"] for e in existing}
        fresh: list[CaseDraft] = []
        for d in (d for drafts in results for d in drafts):
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
    revalidated: list[str] = field(default_factory=list)  # unchanged cases regenerated because rules changed
    normalized: list[str] = field(default_factory=list)  # unchanged cases repaired by the deterministic normalizer
    compiled: list[str] = field(default_factory=list)  # generated by the deterministic compiler (no LLM)
    failed: list[tuple[str, list[str]]] = field(default_factory=list)
    removed_files: list[str] = field(default_factory=list)


def compile_case(
    case: TestCaseIR, module: dict[str, Any], planned: list[Any], endpoints: dict[str, Any]
) -> tuple[dict[str, dict] | None, str | None]:
    """Deterministic compilation, self-checked with the same validator as LLM output.
    Returns (docs, None) or (None, reason the LLM has to implement the case)."""
    try:
        docs = compiler.compile_case(case, module, planned, endpoints)
    except compiler.Unsupported as exc:
        return None, f"unsupported by the compiler: {exc}"
    synthesizer.add_schema_checks(case, planned, docs, endpoints)
    texts = {name: scaffold.dump_yaml(doc) for name, doc in docs.items()}
    errors, _ = validate_files(case, module, planned, texts, _secrets_for(module, case))
    if errors:
        return None, f"compiled files failed validation: {errors[0][:200]}"
    return docs, None


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
    rows = await repo.load_cases(client, dataclasses.replace(flt, statuses=statuses))
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
        ihash = ir_hash(case, {"module": module, "model": model, "compiler": compiler.COMPILER_VERSION})
        files_exist = all((folder / p.filename).exists() for p in planned)
        if not force and row.ir_hash == ihash and str(row.gen_status) == "GENERATED" and files_exist:
            existing = {p.filename: (folder / p.filename).read_text() for p in planned}
            fixed = {name: normalize_text(text) for name, text in existing.items()}
            normalized = {name: text for name, (text, _) in fixed.items()}
            stale, parsed_existing = validate_files(case, module, planned, normalized, _secrets_for(module, case))
            if not stale:
                if any(fixes for _, fixes in fixed.values()):  # repaired deterministically, no LLM needed
                    for name, (text, fixes) in fixed.items():
                        if fixes:
                            (folder / name).write_text(text)
                    summary.normalized.append(case.id)
                    await repo.record_generation(
                        client,
                        case.id,
                        ok=True,
                        script_path=f"{module['folder']}/{planned[0].filename}",
                        script_hash=_content_hash([parsed_existing[p.filename] for p in planned]),
                        ir_hash=ihash,
                        attempts=0,
                        model=row.gen_model or model,
                        error=None,
                    )
                else:
                    summary.skipped_unchanged.append(case.id)
                return
            deps.console.print(f"  {case.id}: files no longer pass validation ({stale[0][:90]}...) - regenerating")
            summary.revalidated.append(case.id)
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
        keys = sorted({e.key for e in row.endpoints} | set(case.coverage.endpoints))
        endpoints = await repo.endpoint_schemas(client, keys)
        parsed, why = compile_case(case, module, planned, endpoints)
        gen_model, attempts = f"compiler-v{compiler.COMPILER_VERSION}", 0
        if parsed is not None:
            summary.compiled.append(case.id)
        else:  # features the compiler does not translate: the LLM implements the case
            async with sem:
                deps.console.print(f"  implementing [bold]{case.id}[/] with the LLM ({why})...")
                parsed, sd = await synthesizer.synthesize(
                    case, module, planned, endpoints, _secrets_for(module, case), model=deps.model
                )
            gen_model, attempts = model, sd.attempts
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
            attempts=attempts,
            model=gen_model,
            error=None,
        )

    await asyncio.gather(*(one(r) for r in rows))
    if dry_run:
        return summary

    # full build: remove files of cases that are no longer selected (deprecated / needs review / deleted)
    if not flt.narrowed:
        # a file belongs to a current case only in that case's module folder (ids restart after srg reset)
        keep = {(modules[r.test_module.code]["folder"], r.test_id) for r in rows}
        for path in root.glob("*/TC-*.yml"):
            m = FILE_RE.match(path.name)
            if m and (path.parent.name, m.group(1)) not in keep:
                summary.removed_files.append(str(path.relative_to(root)))
                path.unlink()
                manifest.pop(str(path.relative_to(root)), None)
        current_folders = {m["folder"] for m in modules.values()} | {"environments"}
        for folder in root.iterdir():
            if folder.is_dir() and folder.name not in current_folders and not any(folder.glob("TC-*.yml")):
                shutil.rmtree(folder)  # orphaned module folder (only folder.yml left)
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
    kind: str = "SUITE",
) -> RunSummary:
    s = deps.settings
    client, store = deps.client, deps.store
    root = s.engine_dir("bruno")
    base_url = base_url or s.sut_base_url
    run_id = f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    rows = [
        r
        for r in await repo.load_cases(client, dataclasses.replace(flt, statuses=None))
        if str(r.gen_status) == "GENERATED" and str(r.design_status) not in ("DEPRECATED",)
    ]
    summary = RunSummary(run_id=run_id, status="PENDING")
    codes = sorted({r.test_module.code for r in rows})
    await store.ensure_bucket()
    await repo.create_run(client, run_id, codes, env_name, base_url, store.uri(run_prefix(run_id)), kind)
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
    case_folder = {r.test_id: r.test_module.folder for r in rows}
    for f, meta in manifest.items():
        if f.split("/", 1)[0] == case_folder.get(meta["test_id"]):  # ignore stale files in old folders
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

    if flt.narrowed:
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


# ---------------------------------------------------------------- diagnose


@dataclass
class DiagnosisRow:
    test_id: str
    param_index: int
    category: str
    confidence: float
    source: str
    root_cause: str
    recommended_action: str
    suggested_fix: str | None = None
    rerun: str | None = None


@dataclass
class DiagnoseSummary:
    run_id: str | None
    rows: list[DiagnosisRow] = field(default_factory=list)
    skipped_existing: int = 0
    errors: list[tuple[str, int, str]] = field(default_factory=list)
    groups: int = 0  # distinct failure signatures
    llm_investigations: int = 0
    not_diagnosed: int = 0  # over the max_llm budget


def diagnostics_declared_env(module: dict[str, Any]) -> set[str]:
    return declared_env_vars(
        {n: a for n, a in module["actors"].items() if a.get("type") != "login"}, module["fixtures"]
    )


def _failure_label(f: diagnostics.FailedExecution) -> str:
    return f"{f.test_id}{chr(ord('a') + f.param_index)}"


_STATUS_MISMATCH = re.compile(r"expected (\d{3}) to equal (\d{3})")


def contract_violations(case: TestCaseIR, failure: diagnostics.FailedExecution, drafts: dict[str, Any]) -> list[str]:
    """Failing steps that omit contract-required request fields, expected a 4xx and got a 2xx."""
    paths = {k: e.path for k, e in drafts.items()}
    by_file = {p.filename: p for p in plan_files(case)}
    steps = http_steps(case)
    out = []
    for fa in failure.failed_assertions:
        m = _STATUS_MISMATCH.search(str(fa.get("error") or ""))
        p = by_file.get(str(fa.get("file") or "").rsplit("/", 1)[-1])
        if not m or p is None or not (200 <= int(m.group(1)) < 300 and int(m.group(2)) in BODY_ERROR_STATUSES):
            continue
        step, _ = steps[p.step_seq - 1]
        key = resolve_endpoint(step.endpoint or "", paths)
        if key and drafts[key].request_schema and isinstance(step.request.payload, dict):
            missing = missing_required(drafts[key].request_schema, step.request.payload)
            if missing:
                out.append(f"step {p.step_seq} ({step.name}): {key} without {missing} answered {m.group(1)}")
    return out


def _to_failure(row: Any, run_id: str) -> diagnostics.FailedExecution:
    tc = row.test_case
    ir = repo.row_to_ir(tc)
    failed = row.failed_assertions
    return diagnostics.FailedExecution(
        execution_id=str(row.id),
        run_id=run_id,
        test_id=tc.test_id,
        title=tc.title,
        param_index=row.param_index,
        status=str(row.status),
        http_status=row.http_status,
        error_message=row.error_message,
        failed_assertions=json.loads(failed) if isinstance(failed, str) else (failed or []),
        case_ir=ir.model_dump(mode="json", exclude_defaults=True),
        endpoints=sorted(e.key for e in tc.endpoints),
        requirement_ids=sorted(q.req_id for q in tc.requirements),
        actor=tc.actor.name if tc.actor else None,
    )


async def diagnose_stage(
    deps: GraphDependencies,
    run_id: str | None = None,
    flt: CaseFilter | None = None,
    force: bool = False,
    allow_rerun: bool = True,
    max_llm: int = 30,
    group: bool = True,
) -> DiagnoseSummary:
    """Triage the failed executions of a run.

    Failures are grouped by signature; per group the rule-based pre-classifier or ONE diagnostics-agent
    investigation (the .env model) decides, and the result is stored for every member of the group."""
    client, store, s = deps.client, deps.store, deps.settings
    run_id = run_id or await repo.latest_run_id(client)
    summary = DiagnoseSummary(run_id=run_id)
    if run_id is None:
        return summary
    flt = flt or CaseFilter()
    drafts = await repo.endpoint_drafts(client)
    selected = []
    for row in await repo.failed_executions(client, run_id):
        tc = row.test_case
        if not flt.matches(tc.test_id, tc.tags):
            continue
        if row.diagnosis is not None and not force:
            summary.skipped_existing += 1
            continue
        selected.append((row, _to_failure(row, run_id)))
    groups: dict[tuple, list] = {}
    for row, f in selected:
        groups.setdefault(diagnostics.signature(f) if group else (f.execution_id,), []).append((row, f))
    summary.groups = len(groups)
    if not groups:
        return summary

    root = s.engine_dir("bruno")
    manifest = load_manifest(deps)
    sem = asyncio.Semaphore(s.srg_llm_concurrency)
    rerun_lock = asyncio.Lock()  # reruns hit the SUT; keep them sequential
    budget = {"left": max_llm}

    async def save(failure: diagnostics.FailedExecution, result: Any, source: str, rerun_result: str | None) -> None:
        await repo.save_diagnosis(
            client, failure.execution_id, result, source, model_name() if source == "LLM" else None, rerun_result
        )
        if result.category == "TEST_DEFECT":
            await repo.flag_test_defect(
                client, failure.test_id, f"diagnosis {run_id}: TEST_DEFECT - {result.root_cause}"
            )
        summary.rows.append(
            DiagnosisRow(
                failure.test_id,
                failure.param_index,
                result.category,
                result.confidence,
                source,
                result.root_cause,
                result.recommended_action,
                result.suggested_fix,
                rerun_result,
            )
        )

    async def one(members: list) -> None:
        row, failure = members[0]
        tid, pidx = failure.test_id, failure.param_index
        failure.similar = [_failure_label(f) for _, f in members[1:]]
        files = sorted(
            (
                f
                for f, m in manifest.items()
                if m["test_id"] == tid
                and m["param_index"] == pidx
                and f.split("/", 1)[0] == row.test_case.test_module.folder
            ),
            key=lambda n: _sort_key(n.rsplit("/", 1)[-1]),
        )

        async def get_report() -> str:
            try:
                return await store.get_text(case_prefix(run_id, tid, pidx) + "report.json")
            except Exception as exc:  # noqa: BLE001
                return f"report not available: {exc.__class__.__name__}"

        async def get_files() -> dict[str, str]:
            return {f: (root / f).read_text() for f in files if (root / f).exists()} or {"-": "files not found"}

        async def rerun() -> str:
            if not files:
                return "rerun impossible: files not found (run srg build)"
            async with rerun_lock:
                out_dir = s.reports_dir / "runs" / run_id / "rerun" / f"{tid}-{pidx}"
                res = await run_bru(
                    s.bru_bin,
                    root,
                    files,
                    row.run.env_name or "local",
                    {"baseUrl": row.run.base_url},
                    dotenv_values(),
                    out_dir,
                    300,
                )
            oc = aggregate(res.results, manifest, {(tid, pidx): files})[(tid, pidx)]
            detail = "; ".join(str(f.get("assertion") or f.get("test") or f.get("error")) for f in oc.failed[:5])
            return f"{oc.status} (HTTP {oc.http_status}) {detail or oc.error_message or ''}".strip()

        env = {**os.environ, **dotenv_values()}
        used: set[str] = set()
        for f in files:
            if (root / f).exists():
                used |= set(re.findall(r"process\.env\.([A-Z][A-Z0-9_]*)", (root / f).read_text()))
        failure.missing_env = sorted(v for v in used if not env.get(v))
        module_ctx = await repo.module_context(client, row.test_case.test_module.code)
        declared = diagnostics_declared_env(module_ctx)
        failure.undeclared_env = [v for v in failure.missing_env if v not in declared]
        failure.contract_violations = contract_violations(repo.row_to_ir(row.test_case), failure, drafts)
        diag_deps = diagnostics.DiagDeps(
            failure=failure,
            get_report=get_report,
            get_contracts=lambda: repo.endpoint_schemas(client, failure.endpoints),
            get_requirements=lambda: repo.requirement_texts(client, tid),
            get_files=get_files,
            get_history=lambda: repo.case_history(client, tid, pidx),
            rerun=rerun if allow_rerun else None,
        )
        result = diagnostics.preclassify(failure)
        source = "RULE"
        if result is None:
            if budget["left"] <= 0:  # checked before the first await -> largest groups win the budget
                summary.not_diagnosed += len(members)
                return
            budget["left"] -= 1
            source = "LLM"
        try:
            if source == "LLM":
                async with sem:
                    deps.console.print(
                        f"  diagnosing [bold]{_failure_label(failure)}[/] ({failure.status}, HTTP {failure.http_status})"
                        + (f" + {len(members) - 1} similar" if len(members) > 1 else "")
                        + "..."
                    )
                    summary.llm_investigations += 1
                    result = await diagnostics.diagnose(diag_deps, model=deps.model)
            await save(failure, result, source, diag_deps.rerun_result)
            for _, other in members[1:]:
                note = f"same failure signature as {_failure_label(failure)} ({len(members)} executions in this group)"
                await save(other, result.model_copy(update={"evidence": [*result.evidence, note]}), source, None)
        except Exception as exc:  # one failed triage must not stop the others
            summary.errors.append((tid, pidx, f"{exc.__class__.__name__}: {str(exc)[:300]}"))

    await asyncio.gather(*(one(m) for m in sorted(groups.values(), key=len, reverse=True)))
    summary.rows.sort(key=lambda r: (r.test_id, r.param_index))
    return summary


# ---------------------------------------------------------------- heal


@dataclass
class HealRow:
    test_id: str
    status: str  # VERIFIED | ROLLED_BACK | UNVERIFIED | REJECTED | FAILED
    attempts: int
    change_summary: str
    detail: str | None = None


@dataclass
class HealSummary:
    run_id: str | None
    rows: list[HealRow] = field(default_factory=list)
    verify_run_ids: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)  # lessons learned (new or confirmed) from verified heals


async def learn_lessons(deps: GraphDependencies, verified: list[dict[str, Any]]) -> list[str]:
    """Generalise verified heals into lessons for future planning/healing (best effort)."""
    from spec_run_graph.agents import lessons as lesson_agent

    if not verified:
        return []
    client = deps.client
    existing = [r.text for r in await repo.lessons(client, limit=200)]
    drafts_ = await repo.endpoint_drafts(client)
    fixes = [{k: v for k, v in f.items() if k != "heal_id"} for f in verified]
    try:
        required = {k: required_body_fields(e.request_schema) for k, e in drafts_.items()}
        drafts = await lesson_agent.extract(fixes, existing, set(drafts_), model=deps.model, required_fields=required)
    except Exception as exc:  # noqa: BLE001 - lessons are an optimisation
        deps.console.print(f"  [yellow]lesson extraction failed[/]: {exc.__class__.__name__}")
        return []
    out = []
    for d in drafts:
        heal_ids = [verified[i]["heal_id"] for i in d.fixes]
        if d.confirms:
            await repo.confirm_lesson(client, d.confirms, heal_ids)
            out.append(d.confirms)
        else:
            await repo.add_lesson(client, d.text, d.endpoints, heal_ids, model_name())
            out.append(d.text)
    return out


async def lint_findings(deps: GraphDependencies, flt: CaseFilter) -> dict[str, Any]:
    """Active cases that violate the current IR rules, as heal candidates with a synthetic diagnosis."""
    from types import SimpleNamespace

    client = deps.client
    ids = await repo.current_spec_ids(client)
    modules = {m: await repo.module_context(client, m) for m in await repo.list_modules(client)}
    drafts = await repo.endpoint_drafts(client)
    findings: dict[str, Any] = {}
    for row in await repo.load_cases(client, dataclasses.replace(flt, statuses=["DRAFT", "READY"])):
        m = modules[row.test_module.code]
        ctx = ValidationContext(
            endpoint_paths=ids.endpoint_paths,
            requirement_ids=set(ids.requirement_ids),
            actors=set(m["actors"]),
            fixtures=set(m["fixtures"]),
            env_vars=diagnostics_declared_env(m),
            response_schemas={k: e.response_schemas for k, e in drafts.items()},
            request_schemas={k: e.request_schema for k, e in drafts.items()},
            page_limits=page_limits(drafts),
        )
        errors = validate_case(repo.row_to_ir(row), ctx)
        if errors:
            findings[row.test_id] = SimpleNamespace(
                id=None,
                param_index=0,
                http_status=None,
                failed_assertions=[],
                run=None,
                test_case=SimpleNamespace(test_id=row.test_id, tags=row.tags, test_module=row.test_module),
                diagnosis=SimpleNamespace(
                    root_cause="The case violates framework rules: " + " | ".join(errors[:8]),
                    evidence=errors[:8],
                    suggested_fix=None,
                ),
            )
    return findings


def _documented_statuses(schemas: dict[str, Any]) -> dict[str, set[int]]:
    return {key: {int(c) for c in (info.get("responses") or {}) if str(c).isdigit()} for key, info in schemas.items()}


def _outcome_text(outcomes: list[Any]) -> str:
    lines = []
    for o in outcomes:
        fails = "; ".join(
            str(f.get("assertion") or f.get("test") or f.get("error")) + f" -> {f.get('error')}" for f in o.failed[:5]
        )
        lines.append(
            f"row {chr(ord('a') + o.param_index)}: {o.status} HTTP {o.http_status} {fails or o.error_message or ''}"
        )
    return "\n".join(lines)


async def heal_stage(
    deps: GraphDependencies,
    run_id: str | None = None,
    flt: CaseFilter | None = None,
    verify: bool = True,
    max_cases: int = 20,
    max_attempts: int = 2,
    lint: bool = False,
) -> HealSummary:
    """Repair test cases whose failures were diagnosed as TEST_DEFECT, then rebuild and re-run them.

    Verified heals are kept (READY, tag `healed`); heals that still fail after max_attempts are rolled back
    to the previous version (NEEDS_REVIEW). Every attempt is recorded as a HealAttempt with before/after IR."""
    client, store = deps.client, deps.store
    flt = flt or CaseFilter()
    by_case: dict[str, list[Any]] = {}
    if lint:  # cases that violate the current validation rules (no failing run needed)
        run_id = "lint"
        by_case = {t: [f] for t, f in (await lint_findings(deps, flt)).items()}
        already: set[str] = set()
    else:
        run_id = run_id or await repo.latest_run_id(client)
        if run_id is None:
            return HealSummary(run_id=None)
        for ex in await repo.heal_candidates(client, run_id):
            if flt.matches(ex.test_case.test_id, ex.test_case.tags):
                by_case.setdefault(ex.test_case.test_id, []).append(ex)
        already = await repo.verified_heals(client, run_id)  # do not heal the same failure twice
    summary = HealSummary(run_id=run_id)
    test_ids = sorted(t for t in by_case if t not in already)[:max_cases]
    if not test_ids:
        return summary

    persisted = await repo.current_spec_ids(client)
    recipes = await repo.load_recipes(client)
    sem = asyncio.Semaphore(deps.settings.srg_llm_concurrency)
    state: dict[str, dict[str, Any]] = {}  # test_id -> before, healed, heal_id, attempts, code, ...

    async def heal_one(test_id: str, previous_failure: str | None = None) -> bool:
        st = state.setdefault(test_id, {"attempts": 0})
        row = await repo.get_case(client, test_id)
        code = row.test_module.code
        current = CaseDraft.model_validate(repo.row_to_ir(row).model_dump(exclude={"id", "status"}))
        st.setdefault("before", current)
        st["code"] = code
        module = await repo.module_context(client, code)
        ctx = ValidationContext(
            endpoint_paths=persisted.endpoint_paths,
            requirement_ids=set(persisted.requirement_ids),
            actors=set(module["actors"]),
            fixtures=set(module["fixtures"]),
            env_vars=declared_env_vars(
                {n: a for n, a in module["actors"].items() if a.get("type") != "login"}, module["fixtures"]
            ),
        )
        endpoints = sorted(e.key for e in row.endpoints)
        st["endpoints"] = endpoints
        schemas = await repo.endpoint_schemas(client, endpoints)
        drafts = await repo.endpoint_drafts(client)
        ctx.response_schemas = {k: e.response_schemas for k, e in drafts.items()}
        ctx.request_schemas = {k: e.request_schema for k, e in drafts.items()}
        ctx.page_limits = page_limits(drafts)
        ordered = [drafts[k] for k in endpoints if k in drafts] + [
            e for k, e in sorted(drafts.items()) if k not in endpoints
        ]
        contracts = "\n".join(endpoint_digest(e) for e in ordered)
        failures = by_case[test_id]
        diagnoses = [
            {
                "row": chr(ord("a") + ex.param_index),
                "http_status": ex.http_status,
                "failed_assertions": ex.failed_assertions,
                "root_cause": ex.diagnosis.root_cause,
                "evidence": list(ex.diagnosis.evidence or []),
                "suggested_fix": ex.diagnosis.suggested_fix,
            }
            for ex in failures
        ]
        exchanges = []
        for ex in failures[:3]:
            if ex.run is None:
                exchanges.append("(lint finding: no failing run; fix the rule violations listed in the diagnosis)")
                continue
            try:
                exchanges.append(
                    await store.get_text(case_prefix(ex.run.run_id, test_id, ex.param_index) + "report.json")
                )
            except Exception as exc:  # noqa: BLE001
                exchanges.append(f"(report unavailable: {exc.__class__.__name__})")
        prompt = healer.build_prompt(
            current,
            diagnoses,
            "\n".join(exchanges),
            contracts,
            await repo.requirement_texts(client, test_id),
            previous_failure,
            recipes,
            [r.text for r in await repo.lessons(client, endpoints)],
        )
        hdeps = healer.HealDeps(
            original=current, ctx=ctx, documented_statuses=_documented_statuses(schemas), recipes=recipes
        )
        st["attempts"] += 1
        try:
            async with sem:
                deps.console.print(f"  healing [bold]{test_id}[/] (attempt {st['attempts']})...")
                result = await healer.heal(prompt, hdeps, model=deps.model)
        except Exception as exc:  # noqa: BLE001
            st["final"] = HealRow(
                test_id,
                "FAILED",
                st["attempts"],
                "-",
                f"{exc.__class__.__name__}: " + "; ".join(hdeps.last_errors[:3] or [str(exc)[:200]]),
            )
            return False
        if result.unhealable_reason:
            await repo.insert_heal_attempt(
                client,
                test_id,
                run_id,
                st["attempts"],
                "REJECTED",
                result.unhealable_reason,
                current.model_dump(mode="json"),
                current.model_dump(mode="json"),
                model_name(),
                str(failures[0].id) if failures[0].id else None,
            )
            st["final"] = HealRow(test_id, "REJECTED", st["attempts"], result.unhealable_reason)
            return False
        healed = result.case.model_copy(update={"tags": sorted({*result.case.tags, "healed"})})
        await repo.replace_case(client, test_id, code, healed, persisted, "READY", None)
        st["heal_id"] = await repo.insert_heal_attempt(
            client,
            test_id,
            run_id,
            st["attempts"],
            "UNVERIFIED",
            result.change_summary,
            st["before"].model_dump(mode="json"),
            healed.model_dump(mode="json"),
            model_name(),
            str(failures[0].id) if failures[0].id else None,
        )
        st["summary"] = result.change_summary
        return True

    healed_ok = await asyncio.gather(*(heal_one(t) for t in test_ids))
    pending = [t for t, ok in zip(test_ids, healed_ok, strict=True) if ok]

    while pending:
        if not verify:
            for t in pending:
                state[t]["final"] = HealRow(t, "UNVERIFIED", state[t]["attempts"], state[t]["summary"])
            break
        ids = set(pending)
        built = await build_stage(deps, CaseFilter(ids=ids), force=True)
        build_failed = {t for t, _ in built.failed}
        runnable = ids - build_failed
        outcomes: dict[str, list[Any]] = {t: [] for t in ids}
        verify_run = None
        if runnable:
            run = await test_stage(deps, CaseFilter(ids=runnable), kind="HEAL_VERIFY")
            verify_run = run.run_id
            summary.verify_run_ids.append(run.run_id)
            for o in run.outcomes:
                outcomes[o.test_id].append(o)
        retry = []
        for t in sorted(ids):
            st = state[t]
            ok = t in runnable and outcomes[t] and all(o.status == "PASSED" for o in outcomes[t])
            text = (
                _outcome_text(outcomes[t])
                if outcomes[t]
                else ("Bruno build failed: " + "; ".join(e for tid, errs in built.failed if tid == t for e in errs[:3]))
            )
            if ok:
                await repo.finish_heal_attempt(client, st["heal_id"], "VERIFIED", verify_run, "all rows passed")
                st["final"] = HealRow(t, "VERIFIED", st["attempts"], st["summary"])
                continue
            await repo.finish_heal_attempt(client, st["heal_id"], "ROLLED_BACK", verify_run, text[:2000])
            if st["attempts"] < max_attempts:
                retry.append((t, text))
            else:
                st["final"] = HealRow(t, "ROLLED_BACK", st["attempts"], st["summary"], text[:300])
        if retry:
            results = await asyncio.gather(*(heal_one(t, text) for t, text in retry))
            pending = [t for (t, _), ok in zip(retry, results, strict=True) if ok]
        else:
            pending = []

    # roll back every case whose heal could not be verified
    rolled = [
        t
        for t, st in state.items()
        if st.get("final") and st["final"].status in ("ROLLED_BACK", "FAILED") and "heal_id" in st
    ]
    for t in rolled:
        st = state[t]
        await repo.replace_case(
            client,
            t,
            st["code"],
            st["before"],
            persisted,
            "NEEDS_REVIEW",
            f"auto-heal failed ({st['attempts']} attempts): {st['final'].detail or ''}"[:500],
        )
    if rolled:
        await build_stage(deps, CaseFilter(ids=set(rolled)), force=True, include_needs_review=True)
    if not lint:  # lint heals fix framework-rule violations, which the rules already enforce
        verified = [
            {
                "heal_id": st["heal_id"],
                "root_cause": by_case[t][0].diagnosis.root_cause,
                "fix": st["summary"],
                "endpoints": st.get("endpoints", []),
            }
            for t, st in state.items()
            if st.get("final") and st["final"].status == "VERIFIED"
        ]
        summary.lessons = await learn_lessons(deps, verified)
    summary.rows = sorted((st["final"] for st in state.values() if st.get("final")), key=lambda r: r.test_id)
    return summary

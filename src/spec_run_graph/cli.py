"""`srg` command line: analyze -> plan -> build -> test (+ doctor, cases, show, reset, web)."""

from __future__ import annotations

import asyncio
import shutil
import time
from functools import wraps
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer
import yaml
from rich.console import Console
from rich.table import Table

from spec_run_graph.config import get_settings
from spec_run_graph.deps import GraphDependencies
from spec_run_graph.knowledge import db, repo
from spec_run_graph.knowledge.repo import CaseFilter
from spec_run_graph.llm import USAGE
from spec_run_graph.pipeline.graph import ORDER, PipelineGraphState, run_pipeline

app = typer.Typer(help="SpecRunGraph: spec-driven API tests (OpenAPI + PRD -> Gel -> Bruno).", no_args_is_help=True)
console = Console()

SpecsArg = Annotated[list[Path] | None, typer.Argument(help="spec files or folders (OpenAPI yaml/json, PRD .md)")]
IdOpt = Annotated[str | None, typer.Option("--id", help="test id glob, e.g. 'TC-ORD-*'")]
TagsOpt = Annotated[list[str] | None, typer.Option("--tags", help="only cases with one of these tags")]


def run_async(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


async def check_services(deps: GraphDependencies, need_s3: bool) -> list[str]:
    """Fail fast (before any LLM call) when Gel / MinIO are not reachable."""
    problems = []
    try:
        await asyncio.wait_for(deps.client.query_single("select 1"), 10)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Gel ({deps.settings.gel_dsn}) is not reachable: {exc.__class__.__name__}")
    if need_s3:
        try:
            await asyncio.wait_for(deps.store.ensure_bucket(), 10)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"MinIO ({deps.settings.s3_endpoint_url}) is not reachable: {exc.__class__.__name__}")
    return problems


async def _pipeline(state: PipelineGraphState) -> PipelineGraphState:
    deps = GraphDependencies(console=console)
    started = time.monotonic()
    if not (state.start == "analyze" and state.stop == "analyze"):  # analyze alone works without Gel
        need_s3 = ORDER.index(state.stop) >= ORDER.index("test")
        problems = await check_services(deps, need_s3)
        if problems:
            await deps.aclose()
            for p in problems:
                console.print(f"[red]{p}[/]")
            console.print("Start the services with [bold]docker compose up -d[/] and check with [bold]srg doctor[/].")
            raise typer.Exit(2)
    try:
        return await run_pipeline(state, deps)
    finally:
        await deps.aclose()
        print_llm_usage(time.monotonic() - started)


def print_llm_usage(wall_seconds: float) -> None:
    u = USAGE
    if not u.calls:
        return
    agents = ", ".join(f"{name} {int(n)}x/{sec:.0f}s" for name, (n, sec) in u.by_agent.items())
    console.print(
        f"[dim]LLM {get_settings().srg_model}: {u.calls} calls, {u.input_tokens:,} in / {u.output_tokens:,} out "
        f"tokens, {u.seconds:.0f}s total call time in {wall_seconds:.0f}s wall time ({agents})"
        + (f", {u.fallbacks} prompted-mode fallbacks" if u.fallbacks else "")
        + "[/]"
    )


# ---------------------------------------------------------------- printing


def print_analysis(state: PipelineGraphState) -> None:
    r = state.report
    if r is None:
        return
    a = r.analysis
    console.rule(f"[bold]Spec analysis {r.analysis_id}")
    console.print(a.summary)
    t = Table("module", "folder", "endpoints", "requirements", "actors", "fixtures", title="Modules")
    for m in a.modules:
        t.add_row(
            m.code,
            m.folder,
            "\n".join(m.endpoints),
            ", ".join(m.requirement_ids),
            ", ".join(x.name for x in m.actors),
            ", ".join(f.name for f in m.fixtures),
        )
    console.print(t)
    t = Table("id", "spec", "endpoints", "description", title="Requirements")
    for q in a.requirements:
        t.add_row(q.req_id, q.spec, "\n".join(q.endpoints) or "[red]-", q.description)
    console.print(t)
    if a.conflicts:
        t = Table("severity", "topic", "A", "B", "recommendation", title="[red]Conflicts between specs")
        for c in a.conflicts:
            t.add_row(
                c.severity,
                c.topic,
                f"{c.source_a}: {c.statement_a}",
                f"{c.source_b}: {c.statement_b}",
                c.recommendation,
            )
        console.print(t)
    if a.gaps:
        t = Table("kind", "description", "refs", title="[yellow]Gaps")
        for g in a.gaps:
            t.add_row(g.kind, g.description, ", ".join(g.refs))
        console.print(t)
    for c in r.changes:
        console.print(
            f"[magenta]Changes[/] {c.spec_name} {c.from_version} -> {c.to_version}: "
            f"added={c.added} removed={c.removed} changed={c.changed}"
        )
    out = get_settings().reports_dir / "analysis" / f"{r.analysis_id}.json"
    console.print(f"\nReport: [cyan]{out}[/] (+ .md).  Next: [bold]srg plan --from {out}[/]")


def print_plan(state: PipelineGraphState) -> None:
    p = state.plan
    if p is None:
        return
    if p.refreshed:
        console.rule("[bold]Gel refreshed")
        console.print(f"  modules, actors and fixtures updated: {', '.join(p.refreshed)} (no new test cases)")
        console.print("  Next: [bold]srg build[/] regenerates the Bruno files for the changed modules.")
        return
    if p.observed:
        console.print(f"  observed {p.observed} read-only GET samples of the SUT (planner context)")
    if p.unset_fixtures:
        console.print(
            "  [yellow]fixtures skipped, their environment variable is not set[/]: " + ", ".join(p.unset_fixtures)
        )
    if p.recipes:
        console.rule("[bold]Setup recipes")
        for name, status in sorted(p.recipes.items()):
            console.print(f"  {name}: {status}")
    console.rule("[bold]Test cases stored in Gel")
    for code, ids in p.created.items():
        console.print(f"  {code}: {len(ids)} new  {', '.join(ids)}")
    if p.duplicates:
        console.print(f"  [yellow]skipped {len(p.duplicates)} duplicates[/]: " + "; ".join(t for _, t in p.duplicates))
    for code, title, errs in p.dropped_invalid:
        console.print(f"  [red]dropped invalid[/] {code} '{title}': {errs[:3]}")
    if p.flagged_for_review:
        console.print(f"  [magenta]{p.flagged_for_review} existing cases flagged NEEDS_REVIEW by spec changes[/]")
    for code, chunk, err in p.failed_chunks:
        console.print(f"  [red]planning failed[/] {code} {chunk}: {err}")
    if p.failed_chunks:
        console.print("  Re-run the same [bold]srg plan[/] command to fill the missing coverage.")
    if p.deprecated:
        console.print(f"  deprecated {p.deprecated} old cases (--replace)")
    if p.deduped:
        console.print(f"  duplicate review: removed {len(p.deduped)} cases")
        for _, drop, keep, reason in p.deduped:
            console.print(f"    {drop} duplicates {keep}: {reason[:120]}")
    if p.gap_fill:
        found, added = sum(g for g, _ in p.gap_fill.values()), sum(a for _, a in p.gap_fill.values())
        console.print(f"  gap-fill pass: {found} coverage gaps after the first pass -> {added} new cases")
    if p.criteria:
        covered, total = sum(c for c, _ in p.criteria.values()), sum(t for _, t in p.criteria.values())
        open_gaps = sum(len(g) for g in p.open_gaps.values())
        console.print(f"  acceptance criteria covered: {covered}/{total}; open coverage gaps: {open_gaps}")
        for code, gaps in sorted(p.open_gaps.items()):
            for gap in gaps[:5]:
                console.print(f"  [dim]  {code}: {gap[:150]}[/]")
    if p.rule_hits:
        top = sorted(p.rule_hits.items(), key=lambda kv: -kv[1])[:5]
        console.print(f"  [dim]planner corrections: {sum(p.rule_hits.values())} rule violations; most frequent:[/]")
        for rule, n in top:
            console.print(f"  [dim]  {n:>3}x {rule}[/]")
    console.print("Review with [bold]srg cases[/] / [bold]srg show <id>[/]; implement with [bold]srg build[/].")


def print_build(state: PipelineGraphState) -> None:
    b = state.build
    if b is None:
        return
    console.rule("[bold]Bruno build")
    console.print(
        f"  generated: {len(b.generated)}  unchanged (skipped): {len(b.skipped_unchanged)}  failed: {len(b.failed)}"
        + (f"  (regenerated after rule changes: {len(b.revalidated)})" if b.revalidated else "")
        + (f"  (repaired without LLM: {len(b.normalized)})" if b.normalized else "")
        + (f"  (compiled without LLM: {len(b.compiled)}/{len(b.generated)})" if b.generated else "")
    )
    for tid, errs in b.failed:
        console.print(f"  [red]FAILED_GEN {tid}[/]: " + " | ".join(errs[:3]))
    if b.removed_files:
        console.print(f"  removed stale files: {b.removed_files}")
    console.print(f"Collection: [cyan]{get_settings().engine_dir('bruno')}[/]")


def print_run(state: PipelineGraphState) -> bool:
    r = state.run
    if r is None:
        return True
    console.rule(f"[bold]Run {r.run_id}: {r.status}")
    for w in r.warnings:
        console.print(f"  [yellow]warning[/] {w}")
    if r.abort_reason:
        console.print(f"  [red]{r.abort_reason}[/]")
    t = Table("case", "row", "status", "http", "ms", "failures")
    colors = {"PASSED": "green", "FAILED": "red", "ERROR": "red", "TIMEOUT": "red", "SKIPPED": "yellow"}
    for o in r.outcomes:
        fails = "; ".join(str(f.get("assertion") or f.get("test") or f.get("error")) for f in o.failed[:3])
        t.add_row(
            o.test_id,
            chr(ord("a") + o.param_index),
            f"[{colors.get(o.status, 'white')}]{o.status}",
            str(o.http_status or ""),
            str(o.duration_ms),
            (fails or o.error_message or "")[:120],
        )
    if r.outcomes:
        console.print(t)
        passed = sum(o.status == "PASSED" for o in r.outcomes)
        console.print(f"  {passed}/{len(r.outcomes)} passed. Logs: s3://{get_settings().s3_bucket}/runs/{r.run_id}/")
    return r.ok


CATEGORY_COLORS = {
    "APPLICATION_BUG": "red",
    "SPEC_MISMATCH": "magenta",
    "TEST_DEFECT": "yellow",
    "ENV_FAILURE": "cyan",
    "FLAKY": "blue",
    "UNKNOWN": "white",
}


def print_diagnosis(state: PipelineGraphState, show_empty: bool = False) -> None:
    d = state.diagnosis
    if d is None:
        return
    if d.run_id is None:
        console.print("[yellow]No completed run to diagnose.[/]")
        return
    if not d.rows and not d.errors:
        if show_empty:
            console.print(
                f"Run {d.run_id}: nothing to diagnose"
                + (f" ({d.skipped_existing} already diagnosed; use --force)" if d.skipped_existing else "")
            )
        return
    console.rule(f"[bold]Failure analysis {d.run_id}")
    t = Table("case", "row", "category", "conf", "root cause", "action / fix")
    for r in d.rows:
        color = CATEGORY_COLORS.get(r.category, "white")
        action = r.recommended_action + (f"\n[dim]fix: {r.suggested_fix}[/]" if r.suggested_fix else "")
        rerun = f"\n[dim]rerun: {r.rerun}[/]" if r.rerun else ""
        t.add_row(
            r.test_id,
            chr(ord("a") + r.param_index),
            f"[{color}]{r.category}[/]\n[dim]{r.source}[/]",
            f"{r.confidence:.2f}",
            r.root_cause + rerun,
            action,
        )
    console.print(t)
    counts: dict[str, int] = {}
    for r in d.rows:
        counts[r.category] = counts.get(r.category, 0) + 1
    console.print("  " + ", ".join(f"[{CATEGORY_COLORS.get(k, 'white')}]{k}[/]={v}" for k, v in sorted(counts.items())))
    if state.run and state.run.run_id == d.run_id and state.run.outcomes:
        ran = {o.test_id for o in state.run.outcomes}
        broken = {r.test_id for r in d.rows if r.category == "TEST_DEFECT"}
        console.print(
            f"  generation quality: {len(ran - broken)}/{len(ran)} cases without test defects"
            f" ({100 * len(ran - broken) / len(ran):.0f}%)"
        )
    if counts.get("TEST_DEFECT"):
        console.print("  TEST_DEFECT cases were set to NEEDS_REVIEW: fix them, then [bold]srg show <id> --approve[/].")
    for tid, pidx, err in d.errors:
        console.print(f"  [red]diagnosis failed[/] {tid}{chr(ord('a') + pidx)}: {err}")
    total = len(d.rows) + d.not_diagnosed + len(d.errors)
    console.print(
        f"  [dim]{len(d.rows)}/{total} failures diagnosed ({d.groups} groups, "
        f"{d.llm_investigations} LLM investigations)[/]"
    )
    if d.not_diagnosed:
        console.print(f"  [yellow]{d.not_diagnosed} failures not diagnosed (over --max-llm); re-run srg diagnose[/]")
    if d.skipped_existing:
        console.print(f"  [dim]{d.skipped_existing} already diagnosed (srg diagnose --force to redo)[/]")


HEAL_COLORS = {
    "VERIFIED": "green",
    "UNVERIFIED": "yellow",
    "ROLLED_BACK": "red",
    "REJECTED": "magenta",
    "FAILED": "red",
}


def print_healing(state: PipelineGraphState, show_empty: bool = False) -> None:
    h = state.healing
    if h is None:
        return
    if not h.rows:
        if show_empty:
            console.print(f"Run {h.run_id or '-'}: no TEST_DEFECT failures to heal (run [bold]srg diagnose[/] first)")
        return
    console.rule(f"[bold]Self-healing {h.run_id}")
    t = Table("case", "result", "attempts", "change", "detail")
    for r in h.rows:
        t.add_row(
            r.test_id,
            f"[{HEAL_COLORS.get(r.status, 'white')}]{r.status}[/]",
            str(r.attempts),
            r.change_summary,
            (r.detail or "")[:200],
        )
    console.print(t)
    counts: dict[str, int] = {}
    for r in h.rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    console.print("  " + ", ".join(f"[{HEAL_COLORS.get(k, 'white')}]{k}[/]={v}" for k, v in sorted(counts.items())))
    if h.verify_run_ids:
        console.print(f"  [dim]verification runs: {', '.join(h.verify_run_ids)}[/]")
    if counts.get("ROLLED_BACK") or counts.get("FAILED"):
        console.print("  Rolled-back cases are NEEDS_REVIEW with the reason; see [bold]srg show <id>[/].")
    for text in h.lessons:
        console.print(f"  [cyan]lesson[/]: {text}")


# ---------------------------------------------------------------- commands


@app.command()
@run_async
async def doctor() -> None:
    """Check .env, the LLM, Gel (applies the schema), MinIO and the bru CLI."""
    from spec_run_graph.engines.bruno.runner import bru_version
    from spec_run_graph.llm import build_model
    from spec_run_graph.telemetry.s3 import TelemetryStore

    s = get_settings()
    ok = True

    def line(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok &= good
        console.print(f"  {'[green]OK ' if good else '[red]ERR'}[/] {name:8} {detail}")

    line(".env", Path(".env").exists(), "found" if Path(".env").exists() else "missing - cp .env.example .env")
    version = await bru_version(s.bru_bin)
    try:
        client = db.create_client(s)
        await db.migrate(client)
        await db.seed(client, version)
        line("gel", True, f"{s.gel_dsn} (schema applied)")
        await client.aclose()
    except Exception as exc:
        line("gel", False, f"{exc.__class__.__name__}: {str(exc)[:200]} (docker compose up -d?)")
    try:
        store = TelemetryStore(s)
        await store.ensure_bucket()
        line("minio", True, f"{s.s3_endpoint_url} bucket {s.s3_bucket}")
    except Exception as exc:
        line("minio", False, f"{exc.__class__.__name__}: {str(exc)[:200]}")
    line("bru", version is not None, version or f"{s.bru_bin} not found (npm i -g @usebruno/cli)")
    try:
        from pydantic_ai import Agent

        result = await Agent(build_model()).run("Reply with the single word: pong")
        line("llm", True, f"{s.srg_model} @ {s.opencode_base_url}: {result.output.strip()[:40]!r}")
    except Exception as exc:
        line("llm", False, f"{s.srg_model} @ {s.opencode_base_url}: {exc.__class__.__name__}: {str(exc)[:200]}")
    raise typer.Exit(0 if ok else 1)


@app.command()
@run_async
async def analyze(specs: SpecsArg = None) -> None:
    """Joint analysis of all specs (LLM). Writes reports/analysis/<id>.{json,md}; nothing is stored in Gel."""
    state = await _pipeline(PipelineGraphState(start="analyze", stop="analyze", spec_paths=specs or [Path(".")]))
    print_analysis(state)


@app.command()
@run_async
async def plan(
    specs: SpecsArg = None,
    from_: Annotated[Path | None, typer.Option("--from", help="reuse a reviewed analysis report (.json)")] = None,
    replace: Annotated[bool, typer.Option(help="deprecate existing cases and plan from scratch")] = False,
    refresh_only: Annotated[
        bool, typer.Option(help="only update specs, modules, actors and fixtures in Gel; no new test cases")
    ] = False,
) -> None:
    """Generate test cases with the LLM and store them in Gel only (fills gaps when cases exist)."""
    if from_ is None and not specs:
        raise typer.BadParameter("give spec files/folders or --from <analysis.json>")
    state = PipelineGraphState(
        start="plan" if from_ else "analyze",
        stop="plan",
        spec_paths=specs or [],
        analysis_file=from_,
        replace=replace,
        refresh_only=refresh_only,
    )
    state = await _pipeline(state)
    if from_ is None:
        print_analysis(state)
    print_plan(state)


@app.command()
@run_async
async def build(
    id_: IdOpt = None,
    tags: TagsOpt = None,
    force: Annotated[bool, typer.Option(help="regenerate even when the case is unchanged")] = False,
    include_needs_review: bool = False,
    dry_run: Annotated[bool, typer.Option(help="generate and validate, but write nothing")] = False,
    engine: Annotated[str, typer.Option(help="test engine")] = "bruno",
) -> None:
    """Implement test cases from Gel as Bruno OpenCollection files (LLM) in test_generated/bruno/."""
    if engine != "bruno":
        raise typer.BadParameter("only the bruno engine is implemented")
    state = await _pipeline(
        PipelineGraphState(
            start="build",
            stop="build",
            case_filter=CaseFilter(id_, tags or []),
            force=force,
            include_needs_review=include_needs_review,
            dry_run=dry_run,
        )
    )
    print_build(state)
    if state.build and state.build.failed:
        raise typer.Exit(1)


@app.command("test")
@run_async
async def test_(
    id_: IdOpt = None,
    tags: TagsOpt = None,
    env: Annotated[str, typer.Option(help="Bruno environment")] = "local",
    base_url: Annotated[str | None, typer.Option(help="override SUT_BASE_URL")] = None,
    skip_health_check: bool = False,
    diagnose: Annotated[bool, typer.Option(help="analyse failures with the diagnostics agent")] = True,
    heal: Annotated[bool, typer.Option(help="repair TEST_DEFECT cases, rebuild and re-run them")] = False,
    rerun: Annotated[bool, typer.Option(help="allow the agent to re-run a failed case once")] = True,
    engine: Annotated[str, typer.Option(help="test engine")] = "bruno",
) -> None:
    """Run the generated suite with bru; results -> Gel, logs -> MinIO; failures are diagnosed (LLM).
    Exit code 1 on any failure."""
    if engine != "bruno":
        raise typer.BadParameter("only the bruno engine is implemented")
    state = await _pipeline(
        PipelineGraphState(
            start="test",
            stop="heal" if heal else ("diagnose" if diagnose else "test"),
            case_filter=CaseFilter(id_, tags or []),
            env_name=env,
            base_url=base_url,
            skip_health_check=skip_health_check,
            rerun_failures=rerun,
        )
    )
    ok = print_run(state)
    print_diagnosis(state)
    print_healing(state)
    if not ok:
        raise typer.Exit(1)


@app.command("heal")
@run_async
async def heal_(
    run_id: Annotated[str | None, typer.Argument(help="run id (default: latest completed run)")] = None,
    id_: IdOpt = None,
    tags: TagsOpt = None,
    verify: Annotated[bool, typer.Option(help="rebuild + re-run healed cases; roll back heals that still fail")] = True,
    max_cases: Annotated[int, typer.Option(help="max test cases to heal")] = 20,
    lint: Annotated[
        bool, typer.Option(help="heal cases that violate the current validation rules (no failed run needed)")
    ] = False,
) -> None:
    """Repair test cases whose failures were diagnosed as TEST_DEFECT (run `srg diagnose` first)."""
    state = await _pipeline(
        PipelineGraphState(
            start="heal",
            stop="heal",
            case_filter=CaseFilter(id_, tags or []),
            diagnose_run_id=run_id,
            heal_verify=verify,
            max_heal_cases=max_cases,
            heal_lint=lint,
        )
    )
    print_healing(state, show_empty=True)


@app.command()
@run_async
async def diagnose(
    run_id: Annotated[str | None, typer.Argument(help="run id (default: latest completed run)")] = None,
    id_: IdOpt = None,
    tags: TagsOpt = None,
    force: Annotated[bool, typer.Option(help="re-diagnose executions that already have a diagnosis")] = False,
    rerun: Annotated[bool, typer.Option(help="allow the agent to re-run a failed case once")] = True,
    max_llm: Annotated[int, typer.Option(help="max LLM investigations (one per failure group)")] = 30,
    group: Annotated[bool, typer.Option(help="diagnose one representative per failure signature")] = True,
) -> None:
    """Analyse the failed executions of a run: why did it fail, and is it the API, the spec or the test?"""
    state = await _pipeline(
        PipelineGraphState(
            start="diagnose",
            stop="diagnose",
            case_filter=CaseFilter(id_, tags or []),
            force=force,
            diagnose_run_id=run_id,
            rerun_failures=rerun,
            max_llm_diagnoses=max_llm,
            group_failures=group,
        )
    )
    print_diagnosis(state, show_empty=True)


@app.command()
@run_async
async def go(
    specs: SpecsArg = None,
    base_url: Annotated[str | None, typer.Option(help="override SUT_BASE_URL")] = None,
) -> None:
    """analyze -> plan -> build -> test in one shot."""
    state = await _pipeline(
        PipelineGraphState(start="analyze", stop="diagnose", spec_paths=specs or [Path(".")], base_url=base_url)
    )
    print_analysis(state)
    print_plan(state)
    print_build(state)
    ok = print_run(state)
    print_diagnosis(state)
    if not ok:
        raise typer.Exit(1)


@app.command()
@run_async
async def cases(
    id_: IdOpt = None,
    status: Annotated[list[str] | None, typer.Option(help="DRAFT|READY|NEEDS_REVIEW|DEPRECATED")] = None,
    needs_review: bool = False,
) -> None:
    """List test cases stored in Gel."""
    client = db.create_client()
    try:
        rows = await repo.load_cases(
            client, CaseFilter(id_, [], [s.upper() for s in status] if status else None, needs_review)
        )
    finally:
        await client.aclose()
    t = Table("id", "title", "type", "prio", "status", "gen", "endpoints", "reqs", title=f"{len(rows)} test cases")
    for r in rows:
        status_txt = str(r.design_status) + (f"\n[magenta]{r.review_reason}" if r.review_reason else "")
        t.add_row(
            r.test_id,
            r.title,
            str(r.case_type),
            str(r.priority),
            status_txt,
            str(r.gen_status),
            "\n".join(e.key for e in r.endpoints),
            ", ".join(q.req_id for q in r.requirements),
        )
    console.print(t)


@app.command()
@run_async
async def recipes() -> None:
    """List the setup recipes (verified ones are offered to the planner and the healer)."""
    client = db.create_client()
    try:
        rows = await repo.recipe_rows(client)
    finally:
        await client.aclose()
    t = Table("recipe", "status", "description", "verification", title=f"{len(rows)} setup recipes")
    for r in rows:
        t.add_row(r.name, str(r.status), r.description, (r.verify_result or "")[:200])
    console.print(t)


@app.command()
@run_async
async def lessons() -> None:
    """List the lessons learned from verified heals (fed into planner and healer prompts)."""
    client = db.create_client()
    try:
        rows = await repo.lessons(client, limit=500)
    finally:
        await client.aclose()
    t = Table("hits", "lesson", "endpoints", title=f"{len(rows)} lessons")
    for r in rows:
        t.add_row(str(r.hits), r.text, "\n".join(r.endpoints) or "(all)")
    console.print(t)


@app.command()
@run_async
async def show(
    test_id: str,
    approve: Annotated[bool, typer.Option(help="set design status to READY")] = False,
    deprecate: Annotated[bool, typer.Option(help="set design status to DEPRECATED")] = False,
) -> None:
    """Show one test case as IR YAML (from Gel); optionally approve or deprecate it."""
    client = db.create_client()
    try:
        new_status = "READY" if approve else "DEPRECATED" if deprecate else None
        if new_status and not await repo.set_design_status(client, test_id, new_status):
            raise typer.BadParameter(f"unknown test id {test_id}")
        row = await repo.get_case(client, test_id)
        heals = await repo.heal_history(client, test_id) if row else []
    finally:
        await client.aclose()
    if row is None:
        raise typer.BadParameter(f"unknown test id {test_id}")
    ir = repo.row_to_ir(row)
    console.print(
        yaml.safe_dump(ir.model_dump(mode="json", exclude_defaults=True), sort_keys=False, allow_unicode=True)
    )
    console.print(f"gen_status={row.gen_status} model={row.gen_model or '-'} files={row.script_path or '-'}")
    if row.last_gen_error:
        console.print(f"[red]last generation error:[/] {row.last_gen_error[:1000]}")
    for h in heals:
        color = HEAL_COLORS.get(h["status"], "white")
        console.print(
            f"heal {h['created_at'][:19]} run {h['run_id']} attempt {h['attempt']}: "
            f"[{color}]{h['status']}[/] - {h['change_summary']}"
        )


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "gel", "minio", "srg-gel", "srg-minio"}


@app.command()
@run_async
async def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="do not ask for confirmation")] = False,
    files: Annotated[bool, typer.Option(help="also delete reports/ (test_generated/ is always removed)")] = False,
    schema: Annotated[bool, typer.Option(help="also reset the Gel schema and re-apply migrations")] = False,
    knowledge: Annotated[
        bool, typer.Option(help="also forget the lessons learned about the API (kept by default)")
    ] = False,
    force_remote: bool = False,
) -> None:
    """Delete ALL data in Gel and ALL objects in the S3 bucket (schema and bucket are kept).
    Lessons learned about the API are kept unless --knowledge (or --schema) is given."""
    from spec_run_graph.telemetry.s3 import TelemetryStore

    s = get_settings()
    hosts = {urlparse(s.gel_dsn).hostname, urlparse(s.s3_endpoint_url).hostname}
    if not force_remote and not hosts <= LOCAL_HOSTS:
        console.print(f"[red]Refusing to reset non-local services {hosts} (use --force-remote)[/]")
        raise typer.Exit(2)
    deps = GraphDependencies(console=console)
    problems = await check_services(deps, need_s3=True)
    await deps.aclose()
    if problems:
        for p in problems:
            console.print(f"[red]{p}[/]")
        console.print("Start the services with [bold]docker compose up -d[/] - nothing was deleted.")
        raise typer.Exit(2)
    client, store = db.create_client(s), TelemetryStore(s)
    counts = await db.counts(client)
    keep = () if knowledge or schema else db.KNOWLEDGE
    n_obj, size = await store.stats()
    console.print("Gel objects: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v and k not in keep))
    if kept := {k: counts[k] for k in keep if counts.get(k)}:
        console.print("Kept: " + ", ".join(f"{k}={v}" for k, v in kept.items()) + " (use --knowledge to delete)")
    console.print(f"S3 s3://{s.s3_bucket}: {n_obj} objects, {size / 1024:.1f} KiB")
    console.print(f"Files: {s.test_generated_dir}/ (generated from Gel)" + (f", {s.reports_dir}/" if files else ""))
    if not yes and not typer.confirm("Delete all of this?"):
        await client.aclose()
        raise typer.Exit(1)
    failed = False
    try:
        if schema:
            await db.reset_schema(client)
            await db.migrate(client)
            await db.seed(client)
            console.print("Gel: schema reset and re-applied")
        else:
            deleted = await db.wipe_all(client, keep=keep)
            console.print("Gel: deleted " + ", ".join(f"{k}={v}" for k, v in deleted.items() if v))
    finally:
        await client.aclose()
    purge = await store.purge_bucket()
    console.print(f"S3: deleted {purge.deleted} objects")
    if purge.failed:
        failed = True
        console.print(f"[red]S3: failed to delete {len(purge.failed)} keys (re-run reset)[/]")
    for d in (s.test_generated_dir, *([s.reports_dir] if files else [])):
        shutil.rmtree(d, ignore_errors=True)
    console.print(
        "Files: removed " + ", ".join(str(d) for d in (s.test_generated_dir, *([s.reports_dir] if files else [])))
    )
    raise typer.Exit(1 if failed else 0)


@app.command()
def web(
    port: Annotated[int, typer.Option(help="port")] = 8090,
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
) -> None:
    """Local dashboard + QA chatbot over Gel (read-only)."""
    import uvicorn

    uvicorn.run("spec_run_graph.web.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()

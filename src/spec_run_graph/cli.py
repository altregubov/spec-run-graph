"""`srg` command line: analyze -> plan -> build -> test (+ doctor, cases, show, reset, web)."""

from __future__ import annotations

import asyncio
import shutil
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
from spec_run_graph.pipeline.graph import PipelineGraphState, run_pipeline

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


async def _pipeline(state: PipelineGraphState) -> PipelineGraphState:
    deps = GraphDependencies(console=console)
    try:
        return await run_pipeline(state, deps)
    finally:
        await deps.aclose()


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
    console.rule("[bold]Test cases stored in Gel")
    for code, ids in p.created.items():
        console.print(f"  {code}: {len(ids)} new  {', '.join(ids)}")
    if p.duplicates:
        console.print(f"  [yellow]skipped {len(p.duplicates)} duplicates[/]: " + "; ".join(t for _, t in p.duplicates))
    for code, title, errs in p.dropped_invalid:
        console.print(f"  [red]dropped invalid[/] {code} '{title}': {errs[:3]}")
    if p.flagged_for_review:
        console.print(f"  [magenta]{p.flagged_for_review} existing cases flagged NEEDS_REVIEW by spec changes[/]")
    if p.deprecated:
        console.print(f"  deprecated {p.deprecated} old cases (--replace)")
    console.print("Review with [bold]srg cases[/] / [bold]srg show <id>[/]; implement with [bold]srg build[/].")


def print_build(state: PipelineGraphState) -> None:
    b = state.build
    if b is None:
        return
    console.rule("[bold]Bruno build")
    console.print(
        f"  generated: {len(b.generated)}  unchanged (skipped): {len(b.skipped_unchanged)}  failed: {len(b.failed)}"
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
) -> None:
    """Generate test cases with the LLM and store them in Gel only (fills gaps when cases exist)."""
    if from_ is None and not specs:
        raise typer.BadParameter("give spec files/folders or --from <analysis.json>")
    state = PipelineGraphState(
        start="plan" if from_ else "analyze", stop="plan", spec_paths=specs or [], analysis_file=from_, replace=replace
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
    engine: Annotated[str, typer.Option(help="test engine")] = "bruno",
) -> None:
    """Run the generated suite with bru; results -> Gel, logs -> MinIO. Exit code 1 on any failure."""
    if engine != "bruno":
        raise typer.BadParameter("only the bruno engine is implemented")
    state = await _pipeline(
        PipelineGraphState(
            start="test",
            stop="test",
            case_filter=CaseFilter(id_, tags or []),
            env_name=env,
            base_url=base_url,
            skip_health_check=skip_health_check,
        )
    )
    if not print_run(state):
        raise typer.Exit(1)


@app.command()
@run_async
async def go(
    specs: SpecsArg = None,
    base_url: Annotated[str | None, typer.Option(help="override SUT_BASE_URL")] = None,
) -> None:
    """analyze -> plan -> build -> test in one shot."""
    state = await _pipeline(
        PipelineGraphState(start="analyze", stop="test", spec_paths=specs or [Path(".")], base_url=base_url)
    )
    print_analysis(state)
    print_plan(state)
    print_build(state)
    if not print_run(state):
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


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "gel", "minio", "srg-gel", "srg-minio"}


@app.command()
@run_async
async def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="do not ask for confirmation")] = False,
    files: Annotated[bool, typer.Option(help="also delete test_generated/ and reports/")] = False,
    schema: Annotated[bool, typer.Option(help="also reset the Gel schema and re-apply migrations")] = False,
    force_remote: bool = False,
) -> None:
    """Delete ALL data in Gel and ALL objects in the S3 bucket (schema and bucket are kept)."""
    from spec_run_graph.telemetry.s3 import TelemetryStore

    s = get_settings()
    hosts = {urlparse(s.gel_dsn).hostname, urlparse(s.s3_endpoint_url).hostname}
    if not force_remote and not hosts <= LOCAL_HOSTS:
        console.print(f"[red]Refusing to reset non-local services {hosts} (use --force-remote)[/]")
        raise typer.Exit(2)
    client, store = db.create_client(s), TelemetryStore(s)
    await store.ensure_bucket()
    counts = await db.counts(client)
    n_obj, size = await store.stats()
    console.print("Gel objects: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    console.print(f"S3 s3://{s.s3_bucket}: {n_obj} objects, {size / 1024:.1f} KiB")
    if files:
        console.print(f"Files: {s.test_generated_dir}/, {s.reports_dir}/")
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
            deleted = await db.wipe_all(client)
            console.print("Gel: deleted " + ", ".join(f"{k}={v}" for k, v in deleted.items() if v))
    finally:
        await client.aclose()
    purge = await store.purge_bucket()
    console.print(f"S3: deleted {purge.deleted} objects")
    if purge.failed:
        failed = True
        console.print(f"[red]S3: failed to delete {len(purge.failed)} keys (re-run reset)[/]")
    if files:
        for d in (s.test_generated_dir, s.reports_dir):
            shutil.rmtree(d, ignore_errors=True)
        console.print("Files: removed")
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

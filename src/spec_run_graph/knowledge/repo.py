"""Persistence of specs, analyses, modules, test cases and runs in Gel."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Any

import gel

from spec_run_graph.domain.analysis import AnalysisReport, ModuleProposal
from spec_run_graph.domain.ir import (
    CaseDraft,
    CoverageIR,
    StepIR,
    TestCaseIR,
    fingerprint,
    http_steps,
    resolve_endpoint,
)
from spec_run_graph.knowledge.db import from_json, to_json
from spec_run_graph.specs.contract import required_body_fields, response_facts
from spec_run_graph.specs.diff import SpecChanges, diff_hashes
from spec_run_graph.specs.models import EndpointDraft, LoadedSpec, sha256

Client = gel.AsyncIOExecutor


# ---------------------------------------------------------------- specs


async def latest_spec(client: Client, name: str) -> Any | None:
    return await client.query_single(
        """
        select FeatureSpec {
            id, name, version, content_hash,
            endpoints: { key, contract_hash },
            requirements: { req_id, description },
        }
        filter .name = <str>$name
        order by .created_at desc limit 1
        """,
        name=name,
    )


async def spec_changes(
    client: Client, specs: list[LoadedSpec], analysis_reqs: dict[str, dict[str, str]]
) -> list[SpecChanges]:
    """Diff each incoming spec against the latest stored version of the same name (read-only)."""
    changes: list[SpecChanges] = []
    for s in specs:
        prev = await latest_spec(client, s.name)
        if prev is None or prev.content_hash == s.content_hash:
            continue
        if s.spec_type == "OPENAPI":
            old = {e.key: e.contract_hash for e in prev.endpoints}
            new = {e.key: e.contract_hash for e in s.endpoints}
        else:
            old = {r.req_id: sha256(r.description) for r in prev.requirements}
            new = {rid: sha256(desc) for rid, desc in analysis_reqs.get(s.name, {}).items()}
        ch = diff_hashes(s.name, prev.version, s.version, old, new)
        if not ch.is_empty:
            changes.append(ch)
    return changes


@dataclass
class PersistedSpecs:
    spec_ids: dict[str, str] = field(default_factory=dict)  # spec name -> FeatureSpec id
    endpoint_ids: dict[str, str] = field(default_factory=dict)  # endpoint key -> ApiEndpoint id
    endpoint_paths: dict[str, str] = field(default_factory=dict)  # endpoint key -> OpenAPI path
    requirement_ids: dict[str, str] = field(default_factory=dict)  # req_id -> Requirement id
    new_versions: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (old id, new id)


async def persist_specs(client: gel.AsyncIOClient, specs: list[LoadedSpec], report: AnalysisReport) -> PersistedSpecs:
    out = PersistedSpecs()
    async for tx in client.transaction():
        async with tx:
            out = PersistedSpecs()
            for s in specs:
                existing = await tx.query_single(
                    "select FeatureSpec { id } filter .content_hash = <str>$h", h=s.content_hash
                )
                if existing is not None:
                    spec_id = str(existing.id)
                else:
                    prev = await latest_spec(tx, s.name)
                    version = s.version
                    clash = await tx.query_single(
                        "select FeatureSpec filter .name = <str>$n and .version = <str>$v", n=s.name, v=version
                    )
                    if clash is not None:
                        version = f"{version}+{s.content_hash[:8]}"
                    spec_id = str(
                        (
                            await tx.query_required_single(
                                """
                                insert FeatureSpec {
                                    name := <str>$name, spec_type := <SpecType><str>$spec_type,
                                    version := <str>$version, source_path := <str>$source_path,
                                    raw_content := <str>$raw, content_hash := <str>$hash,
                                    previous := (select detached FeatureSpec filter .id = <optional uuid>$prev),
                                }
                                """,
                                name=s.name,
                                spec_type=s.spec_type,
                                version=version,
                                source_path=s.source_path,
                                raw=s.raw_content,
                                hash=s.content_hash,
                                prev=prev.id if prev else None,
                            )
                        ).id
                    )
                    if prev is not None:
                        out.new_versions[s.name] = (str(prev.id), spec_id)
                    for e in s.endpoints:
                        await tx.query(
                            """
                            insert ApiEndpoint {
                                spec := (select FeatureSpec filter .id = <uuid>$spec),
                                method := <HttpMethod><str>$method, path := <str>$path,
                                operation_id := <optional str>$op, summary := <optional str>$summary,
                                tag := <optional str>$tag, parameters := <json>$params,
                                request_schema := <optional json>$req, response_schemas := <json>$res,
                                contract_hash := <str>$ch, requires_auth := <bool>$auth,
                                required_body_fields := <array<str>>$rbf, assertable_paths := <json>$ap,
                            }
                            """,
                            spec=spec_id,
                            method=e.method,
                            path=e.path,
                            op=e.operation_id,
                            summary=e.summary,
                            tag=e.tag,
                            params=to_json(e.parameters),
                            req=to_json(e.request_schema),
                            res=to_json(e.response_schemas),
                            ch=e.contract_hash,
                            auth=e.requires_auth,
                            rbf=required_body_fields(e.request_schema),
                            ap=to_json(response_facts(e.response_schemas)),
                        )
                out.spec_ids[s.name] = spec_id
                for e in await tx.query("select ApiEndpoint { id, key, path } filter .spec.id = <uuid>$s", s=spec_id):
                    out.endpoint_ids[e.key] = str(e.id)
                    out.endpoint_paths[e.key] = e.path

            # requirements (upsert per spec version)
            spec_by_name = {s.name: s for s in specs}
            fallback_spec = next((n for n, s in spec_by_name.items() if s.spec_type == "REQUIREMENTS_DOC"), None)
            fallback_spec = fallback_spec or next(iter(out.spec_ids))
            for r in report.analysis.requirements:
                spec_name = r.spec if r.spec in out.spec_ids else fallback_spec
                ep_ids = [out.endpoint_ids[k] for k in r.endpoints if k in out.endpoint_ids]
                row = await tx.query_required_single(
                    """
                    with spec := (select FeatureSpec filter .id = <uuid>$spec)
                    select (
                        insert Requirement {
                            spec := spec, req_id := <str>$rid, description := <str>$desc,
                            acceptance_criteria := <array<str>>$ac,
                            endpoints := (select ApiEndpoint filter .id in array_unpack(<array<uuid>>$eps)),
                        } unless conflict on (.spec, .req_id) else (
                            update Requirement set {
                                description := <str>$desc, acceptance_criteria := <array<str>>$ac,
                                endpoints := (select ApiEndpoint filter .id in array_unpack(<array<uuid>>$eps)),
                                updated_at := datetime_current(),
                            }
                        )
                    ) { id }
                    """,
                    spec=out.spec_ids[spec_name],
                    rid=r.req_id,
                    desc=r.description,
                    ac=r.acceptance_criteria,
                    eps=ep_ids,
                )
                out.requirement_ids[r.req_id] = str(row.id)
    return out


async def apply_change_impact(client: gel.AsyncIOClient, persisted: PersistedSpecs, changes: list[SpecChanges]) -> int:
    """Re-point test steps/requirements of old spec versions to the new ones; flag affected cases."""
    flagged = 0
    affected = {c.spec_name: c for c in changes}
    async for tx in client.transaction():
        async with tx:
            flagged = 0
            for name, (old_id, new_id) in persisted.new_versions.items():
                ch = affected.get(name)
                # endpoints: same key in the new version
                for key in ch.affected if ch else set():
                    reason = f"{name} {ch.from_version} -> {ch.to_version}: {key} {'removed' if key in ch.removed else 'changed'}"
                    flagged += await tx.query_required_single(
                        """
                        select count((
                            update TestCase
                            filter any(.steps.endpoint.key = <str>$key and .steps.endpoint.spec.id = <uuid>$old)
                                or any(.requirements.req_id = <str>$key and .requirements.spec.id = <uuid>$old)
                            set { design_status := DesignStatus.NEEDS_REVIEW, review_reason := <str>$reason,
                                  updated_at := datetime_current() }
                        ))
                        """,
                        key=key,
                        old=old_id,
                        reason=reason,
                    )
                await tx.query(
                    """
                    for old_ep in (select ApiEndpoint filter .spec.id = <uuid>$old) union (
                        update TestStep filter .endpoint = old_ep
                        set { endpoint := assert_single((select ApiEndpoint
                              filter .spec.id = <uuid>$new and .key = old_ep.key)) }
                    )
                    """,
                    old=old_id,
                    new=new_id,
                )
                for op in ("+=", "-="):
                    target = (
                        "(select detached Requirement filter .spec.id = <uuid>$new and .req_id = old_req.req_id)"
                        if op == "+="
                        else "old_req"
                    )
                    await tx.query(
                        f"""
                        for old_req in (select Requirement filter .spec.id = <uuid>$old) union (
                            update TestCase filter old_req in .requirements
                            set {{ requirements {op} {target} }}
                        )
                        """,
                        **({"old": old_id, "new": new_id} if op == "+=" else {"old": old_id}),
                    )
    return flagged


async def insert_analysis(client: Client, report: AnalysisReport, spec_ids: list[str]) -> None:
    a = report.analysis
    await client.query(
        """
        insert Analysis {
            analysis_id := <str>$aid, specs := (select FeatureSpec filter .id in array_unpack(<array<uuid>>$specs)),
            summary := <str>$summary, modules := <json>$modules,
            correlations := <json>$corr, conflicts := <json>$conflicts, gaps := <json>$gaps,
            changes := <json>$changes, model_name := <str>$model,
        } unless conflict on .analysis_id
        """,
        aid=report.analysis_id,
        specs=spec_ids,
        summary=a.summary,
        modules=to_json([m.model_dump() for m in a.modules]),
        corr=to_json([{"req_id": r.req_id, "endpoints": r.endpoints} for r in a.requirements]),
        conflicts=to_json([c.model_dump() for c in a.conflicts]),
        gaps=to_json([g.model_dump() for g in a.gaps]),
        changes=to_json([c.model_dump() for c in report.changes]),
        model=report.model,
    )


# ---------------------------------------------------------------- modules


async def upsert_module(
    client: Client,
    m: ModuleProposal,
    analysis_id: str,
    target_spec_id: str | None,
    product_spec_id: str | None,
) -> None:
    existing = await client.query_single("select TestModule { folder } filter .code = <str>$c", c=m.code)
    folder = existing.folder if existing else m.folder
    if existing is None:  # a new module must not take the folder of another module
        taken = set(await client.query("select TestModule.folder"))
        base, n = folder, 2
        while folder in taken:
            folder, n = f"{base}-{n}", n + 1
    await client.query(
        """
        with
            a := (select Analysis filter .analysis_id = <str>$aid),
            ts := (select FeatureSpec filter .id = <optional uuid>$ts),
            ps := (select FeatureSpec filter .id = <optional uuid>$ps),
        insert TestModule {
            code := <str>$code, folder := <str>$folder, domain := <str>$domain, analysis := a,
            target_spec := ts, product_spec := ps, base_path := <str>$base,
            default_headers := <json>$headers, default_preconditions := <array<str>>$pre,
        } unless conflict on .code else (
            update TestModule set {
                domain := <str>$domain, analysis := a,  # folder is kept: renaming would orphan its files
                target_spec := ts ?? .target_spec, product_spec := ps ?? .product_spec,
                base_path := <str>$base, default_headers := <json>$headers,
                default_preconditions := <array<str>>$pre, updated_at := datetime_current(),
            }
        )
        """,
        aid=analysis_id,
        ts=target_spec_id,
        ps=product_spec_id,
        code=m.code,
        folder=folder,
        domain=m.domain,
        base=m.base_path,
        headers=to_json(m.default_headers),
        pre=m.preconditions,
    )
    for actor in m.actors:
        await client.query(
            """
            with mod := (select TestModule filter .code = <str>$code)
            insert Actor { test_module := mod, name := <str>$name, description := <str>$desc, auth := <json>$auth }
            unless conflict on (.test_module, .name) else (
                update Actor set { description := <str>$desc, auth := <json>$auth }
            )
            """,
            code=m.code,
            name=actor.name,
            desc=actor.description,
            auth=to_json(actor.auth.model_dump(exclude_none=True)),
        )
    for fx in m.fixtures:
        value = {"env": fx.env} if fx.env else fx.value
        await client.query(
            """
            with mod := (select TestModule filter .code = <str>$code)
            insert Fixture { test_module := mod, name := <str>$name, description := <str>$desc, value := <json>$value }
            unless conflict on (.test_module, .name) else (
                update Fixture set { description := <str>$desc, value := <json>$value }
            )
            """,
            code=m.code,
            name=fx.name,
            desc=fx.description,
            value=to_json(value),
        )


async def module_context(client: Client, code: str) -> dict[str, Any]:
    m = await client.query_required_single(
        """
        select TestModule {
            code, folder, domain, base_path, default_headers, default_preconditions,
            target_spec: { name }, product_spec: { name },
            actors: { name, description, auth },
            fixtures: { name, description, value },
        } filter .code = <str>$code
        """,
        code=code,
    )
    return {
        "code": m.code,
        "folder": m.folder,
        "domain": m.domain,
        "base_path": m.base_path or "",
        "default_headers": from_json(m.default_headers) or {},
        "preconditions": list(m.default_preconditions or []),
        "target_spec": m.target_spec.name if m.target_spec else None,
        "product_spec": m.product_spec.name if m.product_spec else None,
        "actors": {a.name: from_json(a.auth) or {} for a in m.actors},
        "fixtures": {f.name: from_json(f.value) for f in m.fixtures},
    }


async def list_modules(client: Client) -> list[str]:
    return list(await client.query("select TestModule.code order by TestModule.code"))


# ---------------------------------------------------------------- test cases


async def existing_cases(client: Client, code: str) -> list[dict[str, Any]]:
    rows = await client.query(
        """
        select TestCase { test_id, title, case_type, fingerprint, design_status,
                          endpoints: { key }, requirements: { req_id } }
        filter .test_module.code = <str>$code and .design_status != DesignStatus.DEPRECATED
        order by .test_id
        """,
        code=code,
    )
    return [
        {
            "id": r.test_id,
            "title": r.title,
            "type": str(r.case_type).lower(),
            "endpoints": sorted(e.key for e in r.endpoints),
            "requirements": sorted(q.req_id for q in r.requirements),
            "fingerprint": r.fingerprint,
        }
        for r in rows
    ]


async def deprecate_cases(client: Client, code: str) -> int:
    return await client.query_required_single(
        """
        select count((update TestCase filter .test_module.code = <str>$code
            and .design_status != DesignStatus.DEPRECATED
            set { design_status := DesignStatus.DEPRECATED, updated_at := datetime_current() }))
        """,
        code=code,
    )


def _step_rows(case: CaseDraft, persisted: PersistedSpecs) -> list[dict[str, Any]]:
    rows = []
    for seq, (s, is_cleanup) in enumerate(http_steps(case), start=1):
        method = path = endpoint_id = None
        if s.type == "http" and s.endpoint:
            method, _, path = s.endpoint.strip().partition(" ")
            key = resolve_endpoint(s.endpoint, persisted.endpoint_paths)
            endpoint_id = persisted.endpoint_ids.get(key) if key else None
        rows.append(
            {
                "seq": seq,
                "is_cleanup": is_cleanup,
                "name": s.name,
                "step_type": s.type.upper(),
                "actor_name": s.actor,
                "endpoint": endpoint_id,
                "method": method.upper() if method else None,
                "path": path.strip() if path else None,
                "request": to_json(s.request.model_dump(exclude_defaults=True)),
                "expected_status": None if s.expected_status is None else str(s.expected_status),
                "validate_schema": s.validate_schema,
                "extract": to_json(s.extract) if s.extract else None,
                "assertions": to_json([a.model_dump(exclude_none=True) for a in s.assertions])
                if s.assertions
                else None,
                "poll": to_json(s.poll.model_dump()) if s.poll else None,
                "delay_ms": s.delay_ms,
                "timeout_ms": s.timeout_ms,
            }
        )
    return rows


async def insert_cases(
    client: gel.AsyncIOClient, code: str, drafts: list[CaseDraft], persisted: PersistedSpecs
) -> list[str]:
    """Assign TC-<CODE>-NNN ids from the module counter and insert cases with their steps (one tx)."""
    if not drafts:
        return []
    ids: list[str] = []
    async for tx in client.transaction():
        async with tx:
            ids = []
            start = await tx.query_required_single(
                """
                select (update TestModule filter .code = <str>$code
                        set { next_case_seq := .next_case_seq + <int32>$n }) { next_case_seq }
                """,
                code=code,
                n=len(drafts),
            )
            first = start.next_case_seq - len(drafts)
            for offset, case in enumerate(drafts):
                test_id = f"TC-{code}-{first + offset:03d}"
                ids.append(test_id)
                req_ids = [
                    persisted.requirement_ids[r]
                    for r in case.coverage.requirement_refs
                    if r in persisted.requirement_ids
                ]
                await tx.query(
                    """
                    with mod := (select TestModule filter .code = <str>$code)
                    insert TestCase {
                        test_id := <str>$tid, test_module := mod, title := <str>$title,
                        case_type := <TestCaseType><str>$type, priority := <Priority><str>$prio,
                        fingerprint := <str>$fp, tags := <array<str>>$tags,
                        actor := (select Actor filter .test_module = mod and .name = <optional str>$actor),
                        requirements := (select Requirement filter .id in array_unpack(<array<uuid>>$reqs)),
                        preconditions := <array<str>>$pre, variables := <optional json>$vars,
                        parameters := <optional json>$params, expected_results := <array<str>>$results,
                    }
                    """,
                    code=code,
                    tid=test_id,
                    title=case.title,
                    type=case.type.upper(),
                    prio=case.priority,
                    fp=fingerprint(code, case, persisted.endpoint_paths),
                    tags=case.tags,
                    actor=case.actor,
                    reqs=req_ids,
                    pre=case.preconditions,
                    vars=to_json(case.variables) if case.variables else None,
                    params=to_json(case.parameters) if case.parameters else None,
                    results=case.expected_results,
                )
                await _insert_steps(tx, test_id, case, persisted)
    return ids


async def _insert_steps(tx: Client, test_id: str, case: CaseDraft, persisted: PersistedSpecs) -> None:
    for row in _step_rows(case, persisted):
        await tx.query(
            """
            insert TestStep {
                test_case := (select TestCase filter .test_id = <str>$tid),
                seq := <int16>$seq, is_cleanup := <bool>$is_cleanup, name := <str>$name,
                step_type := <StepType><str>$step_type, actor_name := <optional str>$actor_name,
                endpoint := (select ApiEndpoint filter .id = <optional uuid>$endpoint),
                method := <HttpMethod><optional str>$method, path := <optional str>$path,
                request := <optional json>$request, expected_status := <optional str>$expected_status,
                validate_schema := <bool>$validate_schema, extract := <optional json>$extract,
                assertions := <optional json>$assertions, poll := <optional json>$poll,
                delay_ms := <optional int32>$delay_ms, timeout_ms := <optional int32>$timeout_ms,
            }
            """,
            tid=test_id,
            **row,
        )


CASE_SHAPE = """
    test_id, title, case_type, priority, design_status, gen_status, review_reason, tags,
    actor: { name }, requirements: { req_id }, endpoints: { key },
    preconditions, variables, parameters, expected_results,
    script_path, script_hash, ir_hash, generation_attempts, gen_model, last_gen_error,
    test_module: { code, folder },
    steps: { seq, is_cleanup, name, step_type, actor_name, method, path, request, expected_status,
             validate_schema, extract, assertions, poll, delay_ms, timeout_ms,
             endpoint: { key } } order by .seq,
"""


def row_to_ir(r: Any) -> TestCaseIR:
    steps, cleanup = [], []
    for s in r.steps:
        step = StepIR(
            step=s.seq,
            name=s.name,
            actor=s.actor_name,
            type=str(s.step_type).lower(),
            endpoint=f"{s.method} {s.path}" if s.method else None,
            request=from_json(s.request) or {},
            expected_status=(
                int(s.expected_status) if s.expected_status and s.expected_status.isdigit() else s.expected_status
            ),
            validate_schema=s.validate_schema,
            extract=from_json(s.extract) or {},
            assertions=from_json(s.assertions) or [],
            poll=from_json(s.poll),
            delay_ms=s.delay_ms,
            timeout_ms=s.timeout_ms,
        )
        (cleanup if s.is_cleanup else steps).append(step)
    return TestCaseIR(
        id=r.test_id,
        status=str(r.design_status).lower(),
        title=r.title,
        type=str(r.case_type).lower(),
        priority=str(r.priority),
        tags=list(r.tags or []),
        actor=r.actor.name if r.actor else None,
        coverage=CoverageIR(
            endpoints=sorted(e.key for e in r.endpoints), requirement_refs=sorted(q.req_id for q in r.requirements)
        ),
        preconditions=list(r.preconditions or []),
        variables=from_json(r.variables) or {},
        parameters=from_json(r.parameters) or [],
        steps=steps,
        expected_results=list(r.expected_results or []),
        cleanup=cleanup,
    )


@dataclass
class CaseFilter:
    id_glob: str | None = None
    tags: list[str] = field(default_factory=list)
    statuses: list[str] | None = None  # DesignStatus names; None = everything
    needs_review: bool = False
    ids: set[str] | None = None  # explicit test ids (e.g. the cases the healer just changed)

    @property
    def narrowed(self) -> bool:
        """True when only part of the suite is selected."""
        return bool(self.id_glob or self.tags or self.ids is not None)

    def matches(self, test_id: str, tags: list[str] | None) -> bool:
        if self.ids is not None and test_id not in self.ids:
            return False
        if self.id_glob and not fnmatch.fnmatch(test_id, self.id_glob):
            return False
        return not (self.tags and not set(self.tags) & set(tags or []))


async def load_cases(client: Client, flt: CaseFilter | None = None) -> list[Any]:
    """Raw case rows (CASE_SHAPE) matching the filter, ordered by id."""
    flt = flt or CaseFilter()
    rows = await client.query(f"select TestCase {{ {CASE_SHAPE} }} order by .test_module.code then .test_id")
    out = []
    for r in rows:
        if not flt.matches(r.test_id, r.tags):
            continue
        status = str(r.design_status)
        if flt.needs_review and status != "NEEDS_REVIEW":
            continue
        if flt.statuses is not None and status not in flt.statuses:
            continue
        out.append(r)
    return out


async def get_case(client: Client, test_id: str) -> Any | None:
    return await client.query_single(f"select TestCase {{ {CASE_SHAPE} }} filter .test_id = <str>$t", t=test_id)


async def set_design_status(client: Client, test_id: str, status: str) -> bool:
    row = await client.query_single(
        """
        update TestCase filter .test_id = <str>$t
        set { design_status := <DesignStatus><str>$s, review_reason := {}, updated_at := datetime_current() }
        """,
        t=test_id,
        s=status,
    )
    return row is not None


async def endpoint_drafts(client: Client) -> dict[str, EndpointDraft]:
    """All endpoints of the latest spec versions (for contract digests and validation contexts)."""
    rows = await client.query(
        """
        select ApiEndpoint { method, path, operation_id, summary, tag, parameters, request_schema,
                             response_schemas, requires_auth, spec: { created_at } }
        """
    )
    latest: dict[str, EndpointDraft] = {}
    for r in sorted(rows, key=lambda x: x.spec.created_at):
        e = EndpointDraft(
            method=str(r.method),
            path=r.path,
            operation_id=r.operation_id,
            summary=r.summary,
            tag=r.tag,
            parameters=from_json(r.parameters) or [],
            request_schema=from_json(r.request_schema),
            response_schemas=from_json(r.response_schemas) or {},
            requires_auth=bool(r.requires_auth),
        )
        latest[e.key] = e
    return latest


async def endpoint_schemas(client: Client, keys: list[str]) -> dict[str, Any]:
    """Latest-version endpoint details (for the synthesizer prompt)."""
    rows = await client.query(
        """
        select ApiEndpoint { key, summary, parameters, request_schema, response_schemas, spec: { created_at } }
        filter .key in array_unpack(<array<str>>$keys)
        """,
        keys=keys,
    )
    latest: dict[str, Any] = {}
    for r in sorted(rows, key=lambda x: x.spec.created_at):
        latest[r.key] = {
            "summary": r.summary,
            "parameters": from_json(r.parameters),
            "request_schema": from_json(r.request_schema),
            "responses": from_json(r.response_schemas),
        }
    return latest


async def record_generation(
    client: Client,
    test_id: str,
    *,
    ok: bool,
    script_path: str | None,
    script_hash: str | None,
    ir_hash: str,
    attempts: int,
    model: str,
    error: str | None,
) -> None:
    await client.query(
        """
        update TestCase filter .test_id = <str>$t set {
            gen_status := GenStatus.GENERATED if <bool>$ok else GenStatus.FAILED_GEN,
            script_path := <optional str>$path, script_hash := <optional str>$hash,
            ir_hash := <str>$ir, generation_attempts := .generation_attempts + <int16>$attempts,
            gen_model := <str>$model, last_gen_error := <optional str>$err, updated_at := datetime_current(),
        }
        """,
        t=test_id,
        ok=ok,
        path=script_path,
        hash=script_hash,
        ir=ir_hash,
        attempts=attempts,
        model=model,
        err=error,
    )


# ---------------------------------------------------------------- runs


async def create_run(
    client: Client, run_id: str, codes: list[str], env: str, base_url: str, prefix: str, kind: str = "SUITE"
) -> None:
    await client.query(
        """
        insert TestRun {
            run_id := <str>$run_id, engine := (select TestEngine filter .name = 'bruno'),
            modules := (select TestModule filter .code in array_unpack(<array<str>>$codes)),
            status := RunStatus.RUNNING, env_name := <str>$env, base_url := <str>$base,
            started_at := datetime_current(), log_prefix_uri := <str>$prefix, kind := <str>$kind,
        }
        """,
        kind=kind,
        run_id=run_id,
        codes=codes,
        env=env,
        base=base_url,
        prefix=prefix,
    )


async def finish_run(client: Client, run_id: str, status: str, abort_reason: str | None = None) -> None:
    await client.query(
        """
        update TestRun filter .run_id = <str>$r set {
            status := <RunStatus><str>$s, finished_at := datetime_current(),
            abort_reason := <optional str>$reason, updated_at := datetime_current(),
        }
        """,
        r=run_id,
        s=status,
        reason=abort_reason,
    )


async def insert_execution(client: Client, run_id: str, test_id: str, param_index: int, **fields: Any) -> None:
    await client.query(
        """
        insert TestExecution {
            run := (select TestRun filter .run_id = <str>$run_id),
            test_case := (select TestCase filter .test_id = <str>$test_id),
            param_index := <int16>$param_index, status := <ExecutionStatus><str>$status,
            duration_ms := <optional int64>$duration_ms, http_status := <optional int16>$http_status,
            failed_assertions := <optional json>$failed, error_message := <optional str>$error,
            log_uri := <optional str>$log_uri, script_hash := <optional str>$script_hash,
        }
        """,
        run_id=run_id,
        test_id=test_id,
        param_index=param_index,
        status=fields["status"],
        duration_ms=fields.get("duration_ms"),
        http_status=fields.get("http_status"),
        failed=to_json(fields.get("failed_assertions")),
        error=fields.get("error_message"),
        log_uri=fields.get("log_uri"),
        script_hash=fields.get("script_hash"),
    )


# ---------------------------------------------------------------- diagnostics


async def latest_run_id(client: Client) -> str | None:
    return await client.query_single(
        """select (select TestRun filter .status = RunStatus.COMPLETED and (.kind ?? 'SUITE') = 'SUITE'
                   order by .started_at desc limit 1).run_id"""
    )


async def failed_executions(client: Client, run_id: str) -> list[Any]:
    return list(
        await client.query(
            f"""
            select TestExecution {{
                id, param_index, status, http_status, failed_assertions, error_message, log_uri,
                run: {{ run_id, base_url, env_name }},
                diagnosis: {{ category }},
                test_case: {{ {CASE_SHAPE} }},
            }}
            filter .run.run_id = <str>$run_id
               and .status in {{ExecutionStatus.FAILED, ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT}}
            order by .test_case.test_id then .param_index
            """,
            run_id=run_id,
        )
    )


async def case_history(client: Client, test_id: str, param_index: int, limit: int = 10) -> list[dict[str, Any]]:
    raw = await client.query_json(
        """
        select TestExecution { run_id := .run.run_id, status, http_status, created_at,
                               category := .diagnosis.category }
        filter .test_case.test_id = <str>$t and .param_index = <int16>$p
        order by .created_at desc limit <int64>$limit
        """,
        t=test_id,
        p=param_index,
        limit=limit,
    )
    return json.loads(raw)


async def requirement_texts(client: Client, test_id: str) -> list[dict[str, Any]]:
    rows = await client.query(
        """select (select TestCase filter .test_id = <str>$t).requirements {
               req_id, description, acceptance_criteria, spec_name := .spec.name }""",
        t=test_id,
    )
    return [
        {
            "req_id": r.req_id,
            "spec": r.spec_name,
            "description": r.description,
            "acceptance_criteria": list(r.acceptance_criteria or []),
        }
        for r in rows
    ]


async def save_diagnosis(
    client: Client, execution_id: str, d: Any, source: str, model: str | None, rerun_status: str | None
) -> None:
    await client.query(
        """
        with ex := (select TestExecution filter .id = <uuid>$ex)
        insert DiagnosticNote {
            execution := ex, category := <DiagnosticCategory><str>$cat, source := <DiagnosticSource><str>$src,
            root_cause := <str>$root, evidence := <array<str>>$ev, recommended_action := <optional str>$action,
            suggested_fix := <optional str>$fix, confidence := <float64>$conf,
            rerun_status := <optional str>$rerun, model_name := <optional str>$model,
        } unless conflict on .execution else (
            update DiagnosticNote set {
                category := <DiagnosticCategory><str>$cat, source := <DiagnosticSource><str>$src,
                root_cause := <str>$root, evidence := <array<str>>$ev, recommended_action := <optional str>$action,
                suggested_fix := <optional str>$fix, confidence := <float64>$conf,
                rerun_status := <optional str>$rerun, model_name := <optional str>$model,
                updated_at := datetime_current(),
            }
        )
        """,
        ex=execution_id,
        cat=d.category,
        src=source,
        root=d.root_cause,
        ev=[str(e) for e in d.evidence],
        action=d.recommended_action,
        fix=d.suggested_fix,
        conf=float(d.confidence),
        rerun=rerun_status,
        model=model,
    )


async def flag_test_defect(client: Client, test_id: str, reason: str) -> None:
    await client.query(
        """update TestCase filter .test_id = <str>$t and .design_status != DesignStatus.DEPRECATED
           set { design_status := DesignStatus.NEEDS_REVIEW, review_reason := <str>$r, updated_at := datetime_current() }""",
        t=test_id,
        r=reason[:500],
    )


# ---------------------------------------------------------------- self-healing


async def current_spec_ids(client: Client) -> PersistedSpecs:
    """Endpoint / requirement ids of the latest version of every spec (to (re)write test steps)."""
    out = PersistedSpecs()
    latest = "(select FeatureSpec filter not exists .<previous[is FeatureSpec])"
    for e in await client.query(f"select ApiEndpoint {{ id, key, path }} filter .spec in {latest}"):
        out.endpoint_ids[e.key] = str(e.id)
        out.endpoint_paths[e.key] = e.path
    for r in await client.query(f"select Requirement {{ id, req_id }} filter .spec in {latest}"):
        out.requirement_ids[r.req_id] = str(r.id)
    return out


async def replace_case(
    client: gel.AsyncIOClient,
    test_id: str,
    code: str,
    case: CaseDraft,
    persisted: PersistedSpecs,
    design_status: str,
    review_reason: str | None,
) -> None:
    """Overwrite a case's content (same test id) and its steps in one transaction; it must be rebuilt after."""
    req_ids = [persisted.requirement_ids[r] for r in case.coverage.requirement_refs if r in persisted.requirement_ids]
    async for tx in client.transaction():
        async with tx:
            await tx.query("delete TestStep filter .test_case.test_id = <str>$t", t=test_id)
            await tx.query(
                """
                with tc := (select TestCase filter .test_id = <str>$t)
                update tc set {
                    title := <str>$title, case_type := <TestCaseType><str>$type, priority := <Priority><str>$prio,
                    fingerprint := <str>$fp, tags := <array<str>>$tags,
                    actor := (select Actor filter .test_module = tc.test_module and .name = <optional str>$actor),
                    requirements := (select Requirement filter .id in array_unpack(<array<uuid>>$reqs)),
                    preconditions := <array<str>>$pre, variables := <optional json>$vars,
                    parameters := <optional json>$params, expected_results := <array<str>>$results,
                    design_status := <DesignStatus><str>$status, review_reason := <optional str>$reason,
                    gen_status := GenStatus.PENDING, ir_hash := {}, updated_at := datetime_current(),
                }
                """,
                t=test_id,
                title=case.title,
                type=case.type.upper(),
                prio=case.priority,
                fp=fingerprint(code, case, persisted.endpoint_paths),
                tags=case.tags,
                actor=case.actor,
                reqs=req_ids,
                pre=case.preconditions,
                vars=to_json(case.variables) if case.variables else None,
                params=to_json(case.parameters) if case.parameters else None,
                results=case.expected_results,
                status=design_status,
                reason=review_reason,
            )
            await _insert_steps(tx, test_id, case, persisted)


async def insert_heal_attempt(
    client: Client,
    test_id: str,
    run_id: str,
    attempt: int,
    status: str,
    change_summary: str,
    before_ir: Any,
    after_ir: Any,
    model: str | None,
    diagnosis_execution_id: str | None = None,
) -> str:
    row = await client.query_required_single(
        """
        insert HealAttempt {
            test_case := (select TestCase filter .test_id = <str>$t),
            diagnosis := (select DiagnosticNote filter .execution.id = <optional uuid>$ex),
            run_id := <str>$run, attempt := <int16>$attempt, status := <HealStatus><str>$status,
            change_summary := <str>$summary, before_ir := <json>$before, after_ir := <json>$after,
            model_name := <optional str>$model,
        }
        """,
        t=test_id,
        ex=diagnosis_execution_id,
        run=run_id,
        attempt=attempt,
        status=status,
        summary=change_summary,
        before=to_json(before_ir),
        after=to_json(after_ir),
        model=model,
    )
    return str(row.id)


async def finish_heal_attempt(
    client: Client, heal_id: str, status: str, verify_run_id: str | None, verify_result: str | None
) -> None:
    await client.query(
        """update HealAttempt filter .id = <uuid>$id set {
               status := <HealStatus><str>$status, verify_run_id := <optional str>$run,
               verify_result := <optional str>$result, updated_at := datetime_current() }""",
        id=heal_id,
        status=status,
        run=verify_run_id,
        result=verify_result,
    )


HEAL_SHAPE = """
    id, param_index, http_status, failed_assertions, error_message, created_at,
    run: { run_id },
    test_case: { test_id, tags, test_module: { code } },
    diagnosis: { root_cause, evidence, suggested_fix, recommended_action, confidence },
"""


async def heal_candidates(client: Client, run_id: str) -> list[Any]:
    """TEST_DEFECT failures (confidence >= 0.5) of the run, plus cases still NEEDS_REVIEW because of a TEST_DEFECT
    diagnosis of an earlier run (they are excluded from builds and runs, so no later run would re-diagnose them)."""
    current = list(
        await client.query(
            f"""
            select TestExecution {{ {HEAL_SHAPE} }}
            filter .run.run_id = <str>$run and .diagnosis.category = DiagnosticCategory.TEST_DEFECT
               and .diagnosis.confidence >= 0.5
               and .test_case.design_status != DesignStatus.DEPRECATED
            order by .test_case.test_id then .param_index
            """,
            run=run_id,
        )
    )
    in_run = {ex.test_case.test_id for ex in current}
    stuck = await client.query(
        f"""
        select TestExecution {{ {HEAL_SHAPE} }}
        filter .diagnosis.category = DiagnosticCategory.TEST_DEFECT and .diagnosis.confidence >= 0.5
           and .test_case.design_status = DesignStatus.NEEDS_REVIEW
           and .test_case.review_reason like 'diagnosis %'
        order by .created_at desc
        """
    )
    latest_run: dict[str, str] = {}
    for ex in stuck:  # newest first: keep only the most recent failing run per case
        tid = ex.test_case.test_id
        if tid in in_run:
            continue
        latest_run.setdefault(tid, ex.run.run_id)
        if latest_run[tid] == ex.run.run_id:
            current.append(ex)
    return current


async def heal_history(client: Client, test_id: str) -> list[dict[str, Any]]:
    raw = await client.query_json(
        """select HealAttempt { run_id, attempt, status, change_summary, verify_run_id, verify_result,
                                model_name, created_at }
           filter .test_case.test_id = <str>$t order by .created_at desc""",
        t=test_id,
    )
    return json.loads(raw)


async def existing_actors(client: Client) -> dict[str, list[str]]:
    rows = await client.query("select TestModule { code, actors: { name } order by .name } order by .code")
    return {m.code: [a.name for a in m.actors] for m in rows if m.actors}


async def verified_heals(client: Client, run_id: str) -> set[str]:
    """Test ids already healed and verified for failures of this run."""
    return set(
        await client.query(
            """select distinct (select HealAttempt filter .run_id = <str>$r
                                and .status = HealStatus.VERIFIED).test_case.test_id""",
            r=run_id,
        )
    )


# ---------------------------------------------------------------- observations


async def save_observation(client: Client, endpoint: str, actor: str, http_status: int, sample: Any) -> None:
    await client.query(
        """
        insert Observation {
            endpoint := <str>$endpoint, actor := <str>$actor, http_status := <int32>$status, sample := <json>$sample
        } unless conflict on (.endpoint, .actor) else (
            update Observation set {
                http_status := <int32>$status, sample := <json>$sample, updated_at := datetime_current()
            }
        )
        """,
        endpoint=endpoint,
        actor=actor,
        status=http_status,
        sample=to_json(sample),
    )


async def observations(client: Client) -> list[Any]:
    rows = await client.query(
        "select Observation { endpoint, actor, http_status, sample, updated_at } order by .endpoint"
    )
    return list(rows)


# ---------------------------------------------------------------- setup recipes


async def save_recipe(
    client: Client, recipe: Any, status: str, verify_result: str, analysis_id: str, model: str
) -> None:
    await client.query(
        """
        insert SetupRecipe {
            name := <str>$name, description := <str>$desc, definition := <json>$defn,
            status := <RecipeStatus><str>$status, verify_result := <str>$vr,
            analysis_id := <str>$aid, model_name := <str>$model,
        }
        unless conflict on .name else (
            update SetupRecipe set {
                description := <str>$desc, definition := <json>$defn, status := <RecipeStatus><str>$status,
                verify_result := <str>$vr, analysis_id := <str>$aid, model_name := <str>$model,
                updated_at := datetime_current(),
            }
        )
        """,
        name=recipe.name,
        desc=recipe.description,
        defn=to_json(recipe.model_dump(mode="json", exclude_defaults=True)),
        status=status,
        vr=verify_result[:4000],
        aid=analysis_id,
        model=model,
    )


async def load_recipes(client: Client, statuses: tuple[str, ...] = ("VERIFIED",)) -> dict[str, Any]:
    from spec_run_graph.domain.recipes import RecipeIR

    rows = await client.query(
        "select SetupRecipe { name, definition } filter <str>.status in array_unpack(<array<str>>$s)",
        s=list(statuses),
    )
    return {r.name: RecipeIR.model_validate(from_json(r.definition)) for r in rows}


async def recipe_rows(client: Client) -> list[Any]:
    return list(
        await client.query(
            "select SetupRecipe { name, description, status, verify_result, analysis_id, created_at } order by .name"
        )
    )


async def recipes_designed_for(client: Client, analysis_id: str) -> bool:
    return bool(
        await client.query_single("select exists (select SetupRecipe filter .analysis_id = <str>$a)", a=analysis_id)
    )


# ---------------------------------------------------------------- lessons


async def add_lesson(client: Client, text: str, endpoints: list[str], heal_ids: list[str], model: str) -> None:
    await client.query(
        """
        with heals := (select HealAttempt filter .id in <uuid>array_unpack(<array<str>>$heals))
        insert Lesson { text := <str>$text, endpoints := <array<str>>$eps, heals := heals, model_name := <str>$model }
        unless conflict on .text else (
            update Lesson set {
                hits := .hits + 1, heals += heals, endpoints := <array<str>>$eps, updated_at := datetime_current()
            }
        )
        """,
        text=text,
        eps=endpoints,
        heals=heal_ids,
        model=model,
    )


async def confirm_lesson(client: Client, text: str, heal_ids: list[str]) -> None:
    await client.query(
        """
        with heals := (select HealAttempt filter .id in <uuid>array_unpack(<array<str>>$heals))
        update Lesson filter .text = <str>$text set { hits := .hits + 1, heals += heals, updated_at := datetime_current() }
        """,
        text=text,
        heals=heal_ids,
    )


async def lessons(client: Client, endpoints: list[str] | None = None, limit: int = 15) -> list[Any]:
    """Global lessons plus those about the given endpoints, most confirmed first."""
    rows = await client.query(
        "select Lesson { text, endpoints, hits, created_at } order by .hits desc then .created_at"
    )
    wanted = set(endpoints or [])
    picked = [r for r in rows if not r.endpoints or endpoints is None or set(r.endpoints) & wanted]
    return picked[:limit]

"""Joint Spec Analyzer: all specifications are analysed together in a single LLM call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.domain.analysis import SpecAnalysis
from spec_run_graph.llm import build_model, output_spec, run_structured
from spec_run_graph.specs.loader import digest
from spec_run_graph.specs.models import LoadedSpec

INSTRUCTIONS = """\
You are a senior QA architect. You receive ALL specifications of one product at once: OpenAPI
contracts and product requirement documents (PRDs). Analyse them JOINTLY - not one by one.

Produce:
1. summary: what the product does and what is in scope for API testing (5-10 sentences).
2. modules: group endpoints into test modules (normally by OpenAPI tag or path prefix).
   - code: short UPPERCASE id used in test ids (ORD, PROD, AUTH, USER ...), unique.
   - folder: kebab-case plural noun (orders, catalog, users), unique.
   - endpoints: endpoint keys exactly as given ("METHOD /path"); every endpoint belongs to exactly one module.
   - actors: roles needed for testing with their auth.
     * If the documents describe a login endpoint AND test accounts (seed data, example credentials), use
       auth {type: "login", login_endpoint: "POST /api/.../login", login_body: {"username": "...",
       "password": "..."}, token_path: "data.token"} with the documented credentials - one actor per account
       (e.g. admin, user_a, user_b). The test collection logs in at run time; never invent credentials.
     * Otherwise secrets are env references, e.g. {type: "bearer", token: "{{ $env.CUSTOMER_TOKEN }}"}.
     * If "Existing actors" are listed below, reuse those actor names for the same accounts.
   - fixtures: reusable test data mentioned in the documents (e.g. active_sku: "SKU-ACTIVE-1"), as literal
     values. Use {"env": "VAR"} only for secrets the user must provide; never for ids or other values the
     system generates (tests look those up at run time).
   - default_headers: e.g. {"Accept": "application/json"}.
   - base_path: leave empty unless the server prefix is NOT already part of the OpenAPI paths.
3. requirements: every verifiable requirement. Reuse ids from the documents (e.g. PRD-ORD-42);
   for OpenAPI-only behaviour use REQ-<CODE>-NN. Link each to the endpoint keys that implement it
   (empty list if no endpoint exists). acceptance_criteria are concrete and testable.
4. conflicts: places where the specs contradict each other (different limits, status codes, error
   codes, fields, auth rules). Quote both statements and their sources. Be precise; do not invent.
5. gaps: requirements without endpoints, endpoints without requirements, undocumented behaviour
   (e.g. an error described in the PRD but not declared in OpenAPI). For every entity an endpoint creates, check
   whether the specs say what happens on a duplicate (uniqueness rule, conflict status); if not, report a gap
   naming the endpoint and the identifying fields (e.g. product manufacturer + model).
6. risks: short bullet points on what is most likely to break.
Only use endpoint keys that appear in the input."""


@dataclass
class AnalyzerDeps:
    endpoint_keys: set[str]


def _validate(analysis: SpecAnalysis, deps: AnalyzerDeps) -> list[str]:
    errors: list[str] = []
    known = deps.endpoint_keys
    codes, folders, owned = set(), set(), []
    for m in analysis.modules:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", m.code):
            errors.append(f"module code {m.code!r} must be UPPERCASE letters/digits")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", m.folder):
            errors.append(f"module folder {m.folder!r} must be kebab-case")
        if m.code in codes:
            errors.append(f"duplicate module code {m.code}")
        if m.folder in folders:
            errors.append(f"duplicate module folder {m.folder}")
        codes.add(m.code)
        folders.add(m.folder)
        for e in m.endpoints:
            if e not in known:
                errors.append(f"module {m.code}: unknown endpoint {e!r}")
        owned.extend(m.endpoints)
        for a in m.actors:
            if a.auth.type == "login":
                if a.auth.login_endpoint not in known:
                    errors.append(f"actor {a.name}: login_endpoint {a.auth.login_endpoint!r} is not an endpoint key")
                if not a.auth.login_body:
                    errors.append(f"actor {a.name}: login actors need login_body with the documented credentials")
                if not a.auth.token_path:
                    errors.append(f"actor {a.name}: login actors need token_path (e.g. data.token)")
                continue
            for secret in (a.auth.token, a.auth.password, a.auth.value):
                if secret and not re.fullmatch(r"\{\{\s*\$env\.[A-Z][A-Z0-9_]*\s*\}\}", secret.strip()):
                    errors.append(f"actor {a.name}: secrets must be '{{{{ $env.VAR }}}}' references")
    missing = sorted(known - set(owned))
    if missing:
        errors.append(f"endpoints not assigned to any module: {missing}")
    dupes = sorted({e for e in owned if owned.count(e) > 1})
    if dupes:
        errors.append(f"endpoints assigned to several modules: {dupes}")
    req_ids = [r.req_id for r in analysis.requirements]
    if len(req_ids) != len(set(req_ids)):
        errors.append("requirement ids must be unique")
    for r in analysis.requirements:
        for e in r.endpoints:
            if e not in known:
                errors.append(f"requirement {r.req_id}: unknown endpoint {e!r}")
    for m in analysis.modules:
        for rid in m.requirement_ids:
            if rid not in req_ids:
                errors.append(f"module {m.code}: unknown requirement id {rid}")
    return errors


def build_agent(output: Any = None) -> Agent[AnalyzerDeps, SpecAnalysis]:
    agent: Agent[AnalyzerDeps, SpecAnalysis] = Agent(
        output_type=output or output_spec(SpecAnalysis),
        instructions=INSTRUCTIONS,
        deps_type=AnalyzerDeps,
        retries=get_settings().srg_llm_retries,
        name="spec-analyzer",
    )

    @agent.output_validator
    def check(ctx: RunContext[AnalyzerDeps], out: SpecAnalysis) -> SpecAnalysis:
        errors = _validate(out, ctx.deps)
        if errors:
            raise ModelRetry("Fix these problems and answer again:\n- " + "\n- ".join(errors))
        return out

    return agent


def build_prompt(specs: list[LoadedSpec], existing_actors: dict[str, list[str]] | None = None) -> str:
    keys = [e.key for s in specs for e in s.endpoints]
    parts = [f"There are {len(specs)} specifications. Endpoint keys: {keys}", *(digest(s) for s in specs)]
    if existing_actors:
        parts.append(f"Existing actors per module (reuse these names): {json.dumps(existing_actors)}")
    return "\n\n".join(parts)


async def analyze(
    specs: list[LoadedSpec], model: Model | None = None, existing_actors: dict[str, list[str]] | None = None
) -> SpecAnalysis:
    keys = {e.key for s in specs for e in s.endpoints}
    result = await run_structured(
        build_agent,
        build_prompt(specs, existing_actors),
        SpecAnalysis,
        deps=AnalyzerDeps(endpoint_keys=keys),
        model=model or build_model(),
    )
    return result.output

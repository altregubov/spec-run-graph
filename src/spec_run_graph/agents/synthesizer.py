"""Code Synthesizer: the LLM implements one test case (IR from Gel) as Bruno OpenCollection YAML files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.domain.ir import ANONYMOUS, TestCaseIR, http_steps, resolve_endpoint
from spec_run_graph.engines.bruno.naming import PlannedFile, var_prefix
from spec_run_graph.engines.bruno.normalize import normalize_text
from spec_run_graph.engines.bruno.scaffold import actor_auth, fixture_template
from spec_run_graph.engines.bruno.validate import (
    expected_status_for,
    expected_statuses_for,
    status_assertion,
    validate_files,
)
from spec_run_graph.llm import build_model, output_spec, run_structured

INSTRUCTIONS = """\
You are an expert in the Bruno API client. You implement ONE API test case as Bruno request files in
the OpenCollection YAML format (one file per HTTP step and parameter row). Files run in the listed
order inside one folder; runtime variables persist between files.

Request file format (OpenCollection 1.0.0):
info:            {name: <exact display name given>, type: http, tags: [..]}
http:
  method: POST
  url: "{{baseUrl}}<base_path><path with path params filled in>"
  headers: [{name: X-Header, value: "..."}]
  params: [{name: page, value: "1", type: query}]          # query parameters
  body: {type: json, data: "<JSON text>"}                   # omit when there is no payload
  auth: <exactly the "auth" block given for this file in the plan>   # omit when the plan gives none
runtime:
  assertions:    [{expression: res.status, operator: eq, value: "201"}, ...]
  actions:       [{type: set-variable, phase: after-response,
                   selector: {expression: res.body.data.id, method: jsonq},
                   variable: {name: <PREFIX>order_id, scope: runtime}}]
  scripts:       [{type: before-request, code: "..."}, {type: tests, code: "..."}]
settings: {timeout: 30000}                                  # only when the step has timeout_ms

Translation rules (IR -> Bruno):
- {{ $env.X }}        -> {{process.env.X}}
- {{ $fixture.name }} -> the fixture template given below
- {{ $extract.x }}    -> {{<PREFIX>x}}; capture it with a set-variable action named <PREFIX>x
- {{ $var.x }}        -> {{<PREFIX>var_x}}; set it in the FIRST file's before-request script:
                         bru.setVar("<PREFIX>var_x", bru.interpolate("<value with Bruno templates>"));
- {{ $param.k }}      -> the literal value of the current parameter row (numbers stay JSON numbers)
- {{ $faker.uuid }}   -> {{$randomUUID}}; other faker types -> the Bruno dynamic variable given below
- extract/check paths: response.status -> res.status, response.body.a.b -> res.body.a.b,
  response.headers.x -> res.headers['x']
- assertions operators: eq neq gt gte lt lte isDefined isUndefined contains matches in.
  String values are quoted inside the value: value: "'DRAFT'"; numbers/booleans are bare: value: "201".
  ALWAYS assert the expected status first, exactly as given by the file's status_assertion
  (e.g. {expression: res.status, operator: eq, value: "201"} or operator in, value "[401, 403]").
- For checks that are awkward as assertions (in, contains, matches, nested arrays) write a tests script:
  test("<message>", function () { expect(res.getBody().data.status).to.be.oneOf(["A", "B"]); });
- validate_schema: true -> do NOT write schema validation code (no ajv): the framework appends the OpenAPI
  schema check to those files automatically.
- List filters in paths (response.body.data[model='Galaxy S24'].id) select the item whose field equals the value.
  Bruno assertions/actions cannot filter, so implement them in scripts with Array.find:
    check  -> tests script: const item = (res.getBody().data || []).find(x => x.model === "Galaxy S24");
              test("<message>", function () { expect(item?.price).to.be.above(10); });
    extract -> after-response script: bru.setVar("<PREFIX>x", ((res.getBody().data || []).find(x => x.model === "Galaxy S24") || {}).id);
  A template filter value ({{ $extract.pid }}) is read in JS with bru.getVar("<PREFIX>pid").
- delay steps: put `await bru.sleep(<ms>);` in the before-request script of the NEXT file.
- cleanup files: same format, but NO assertions and no tests (their failures are ignored).
- Authentication: every planned file lists the actor of its step and that actor's "auth" block - copy it
  verbatim. Never invent token variables. Only for steps that deliberately test missing/invalid credentials
  use explicit headers (e.g. Authorization: Bearer invalid-token) or no auth.
- Never write secrets or tokens literally. Only these templates are allowed: {{baseUrl}},
  {{process.env.X}} for env variables that appear in the IR or the auth blocks, the fixture templates,
  {{auth_<actor>}} tokens from the auth blocks, {{<PREFIX>...}} runtime variables, Bruno {{$random...}} variables.
Return every planned file with exactly the planned filename and content as YAML text."""

EXAMPLE = """\
Verified example (step 1 of a 2-step case, prefix TC_ORD_001__):
info:
  name: TC-ORD-001 step1 create order
  type: http
  tags: [smoke]
http:
  method: POST
  url: "{{baseUrl}}/v1/orders"
  headers:
    - name: Content-Type
      value: application/json
    - name: Idempotency-Key
      value: "{{TC_ORD_001__var_idempotency_key}}"
  body:
    type: json
    data: |-
      {"items": [{"sku": "{{fx_ORD_active_sku}}", "qty": 2}]}
  auth:
    type: bearer
    token: "{{auth_customer}}"
runtime:
  assertions:
    - expression: res.status
      operator: eq
      value: "201"
    - expression: res.body.data.status
      operator: eq
      value: "'DRAFT'"
  actions:
    - type: set-variable
      phase: after-response
      selector:
        expression: res.body.data.id
        method: jsonq
      variable:
        name: TC_ORD_001__order_id
        scope: runtime
  scripts:
    - type: before-request
      code: |-
        bru.setVar("TC_ORD_001__var_idempotency_key", bru.interpolate("{{$randomUUID}}"));
    - type: tests
      code: |-
        test("total is price x qty", function () {
          expect(res.getBody().data.total).to.equal(25);
        });"""


class BrunoFile(BaseModel):
    filename: str = Field(description="exactly one of the planned filenames")
    content: str = Field(description="the complete YAML file content")


class SynthesizedFiles(BaseModel):
    files: list[BrunoFile]


@dataclass
class SynthDeps:
    case: TestCaseIR
    module: dict[str, Any]
    planned: list[PlannedFile]
    secrets: set[str]
    attempts: int = 0
    last_errors: list[str] = field(default_factory=list)
    parsed: dict[str, dict] = field(default_factory=dict)


def build_agent(output: Any = None) -> Agent[SynthDeps, SynthesizedFiles]:
    agent: Agent[SynthDeps, SynthesizedFiles] = Agent(
        output_type=output or output_spec(SynthesizedFiles),
        instructions=INSTRUCTIONS + "\n\n" + EXAMPLE,
        deps_type=SynthDeps,
        retries=get_settings().srg_llm_retries,
        name="bruno-synthesizer",
    )

    @agent.output_validator
    def check(ctx: RunContext[SynthDeps], out: SynthesizedFiles) -> SynthesizedFiles:
        d = ctx.deps
        d.attempts += 1
        # deterministic repairs first (query params into url, auth vs explicit header, regex quotes)
        files = {f.filename.rsplit("/", 1)[-1]: normalize_text(f.content)[0] for f in out.files}
        errors, parsed = validate_files(d.case, d.module, d.planned, files, d.secrets)
        d.last_errors = errors
        if errors:
            raise ModelRetry(
                "The files are invalid. Fix ALL problems and return all files again:\n- " + "\n- ".join(errors[:40])
            )
        d.parsed = parsed
        return out

    return agent


def build_prompt(
    case: TestCaseIR, module: dict[str, Any], planned: list[PlannedFile], endpoints: dict[str, Any]
) -> str:
    prefix = var_prefix(case.id or "")
    steps = http_steps(case)
    fixtures = {
        f"{{{{ $fixture.{n} }}}}": fixture_template(module["code"], n, v) for n, v in module["fixtures"].items()
    }
    actors = {}
    for name, auth in module["actors"].items():
        actors[name] = actor_auth({**auth, "_actor": name})
    file_plan = []
    for p in planned:
        step, _ = steps[p.step_seq - 1]
        entry: dict[str, Any] = {
            "filename": p.filename,
            "info.name": p.display_name,
            "implements": f"{'cleanup' if p.is_cleanup else 'step'} #{p.step_seq} '{step.name}'",
            "actor": step.actor or case.actor,
            "auth": "omit the auth block (no credentials)"
            if (step.actor or case.actor) == ANONYMOUS
            else actors.get(step.actor or case.actor or ""),
            "status_assertion": None
            if p.is_cleanup
            else status_assertion(expected_statuses_for(case, p.step_seq, p.param_index)),
        }
        if case.parameters:
            entry["parameter_row"] = case.parameters[p.param_index]
        file_plan.append(entry)
    schemas = {}
    for step, _ in steps:
        if step.validate_schema and step.endpoint:
            key = resolve_endpoint(step.endpoint, {k: k.split(" ", 1)[1] for k in endpoints})
            if key and key in endpoints:
                schemas[key] = endpoints[key]["responses"]
    preceding_delays = [
        f"before file of step #{i + 1}: await bru.sleep({s.delay_ms});"
        for i, (s, _) in enumerate(steps)
        if s.type == "delay" and i + 1 < len(steps)
    ]
    parts = [
        f"Runtime variable PREFIX for this case: {prefix}",
        f"Module: code={module['code']} folder={module['folder']} base_path={module['base_path']!r} "
        f"(folder.yml already sends headers {module['default_headers']})",
        f"Fixture templates: {json.dumps(fixtures)}",
        f"Actors (Bruno auth blocks): {json.dumps(actors)}",
        "Test case (IR):\n" + yaml.safe_dump(case.model_dump(mode="json", exclude_defaults=True), sort_keys=False),
        "Planned files, in run order:\n" + json.dumps(file_plan),
    ]
    if preceding_delays:
        parts.append("Delays: " + "; ".join(preceding_delays))
    if schemas:
        parts.append("OpenAPI responses (for validate_schema):\n" + json.dumps(schemas)[:12000])
    return "\n\n".join(parts)


async def synthesize(
    case: TestCaseIR,
    module: dict[str, Any],
    planned: list[PlannedFile],
    endpoints: dict[str, Any],
    secrets: set[str],
    model: Model | None = None,
) -> tuple[dict[str, dict] | None, SynthDeps]:
    """Returns (parsed files or None on failure, deps with attempts/errors)."""
    deps = SynthDeps(case=case, module=module, planned=planned, secrets=secrets)
    try:
        await run_structured(
            build_agent,
            build_prompt(case, module, planned, endpoints),
            SynthesizedFiles,
            thinking=False,  # mechanical IR -> YAML translation; validated anyway
            deps=deps,
            model=model or build_model(),
        )
    except UnexpectedModelBehavior as exc:
        if not deps.last_errors:
            deps.last_errors = [str(exc)]
        return None, deps
    add_schema_checks(case, planned, deps.parsed, endpoints)
    return deps.parsed, deps


def _strip_examples(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_examples(v) for k, v in schema.items() if k not in ("example", "examples", "description")}
    if isinstance(schema, list):
        return [_strip_examples(v) for v in schema]
    return schema


def schema_check_script(schema: Any) -> str:
    return (
        "// srg: response schema check generated from the OpenAPI contract\n{\n"
        '  const Ajv = require("ajv");\n'
        "  const ajv = new Ajv({ allErrors: true, strict: false });\n"
        f"  const validate = ajv.compile({json.dumps(_strip_examples(schema), separators=(',', ':'))});\n"
        '  test("response matches the OpenAPI schema", function () {\n'
        "    expect(validate(res.getBody()), JSON.stringify(validate.errors)).to.equal(true);\n"
        "  });\n}"
    )


def add_schema_checks(
    case: TestCaseIR, planned: list[PlannedFile], parsed: dict[str, dict], endpoints: dict[str, Any]
) -> None:
    """Deterministically append the ajv schema check to files of steps with validate_schema (the LLM used to copy
    multi-KB schemas into the script and mangle them)."""
    steps = http_steps(case)
    paths = {k: k.split(" ", 1)[1] for k in endpoints}
    for p in planned:
        step, is_cleanup = steps[p.step_seq - 1]
        if is_cleanup or not step.validate_schema or not step.endpoint or p.filename not in parsed:
            continue
        key = resolve_endpoint(step.endpoint, paths)
        status = expected_status_for(case, p.step_seq, p.param_index)
        schema = ((endpoints.get(key or "", {}).get("responses") or {}).get(str(status)) or {}).get("schema")
        if not schema:
            continue
        runtime = parsed[p.filename].setdefault("runtime", {})
        runtime.setdefault("scripts", []).append({"type": "tests", "code": schema_check_script(schema)})

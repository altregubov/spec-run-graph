"""Test-case intermediate representation (the reference test-case format, with review fixes applied).

Context variable syntax:
  {{ $faker.<type> }}    dynamic data           {{ $fixture.<name> }}  module fixture
  {{ $env.<KEY> }}       environment variable   {{ $extract.<var> }}   captured in an earlier step
  {{ $param.<key> }}     data-driven row value  {{ $var.<name> }}      case-level variable
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from spec_run_graph.domain.expr import PATH_RE, CheckSyntaxError, parse_check
from spec_run_graph.domain.faker import bruno_dynamic
from spec_run_graph.specs.models import HTTP_METHODS, stable_hash

CaseType = Literal["positive", "negative", "boundary", "security", "rbac"]
DesignStatus = Literal["draft", "ready", "needs_review", "deprecated"]
Priority = Literal["P0", "P1", "P2"]

REF_ENV_RE = re.compile(r"\{\{\s*\$env\.([A-Z][A-Z0-9_]*)\s*\}\}")
REF_RE = re.compile(r"\{\{\s*\$([a-z]+)\.([A-Za-z0-9_.\-]+)\s*\}\}")
ANY_TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FILTER_IN_PATH = re.compile(r"\[[A-Za-z_][A-Za-z0-9_]*=")
PARAM_TEMPLATE_RE = re.compile(r"^\{\{\s*\$param\.([A-Za-z0-9_]+)\s*\}\}$")
STATUS_SET_RE = re.compile(r"^\d{3}(\s*\|\s*\d{3})+$")
NAMESPACES = {"faker", "fixture", "env", "extract", "param", "var"}


class AssertionIR(BaseModel):
    check: str = Field(description="'<path> <op> <literal>', e.g. response.body.data.status == 'DRAFT'")
    message: str | None = None


class ExpectedError(BaseModel):
    code: str | None = None
    field: str | None = None
    message: str | None = None


class RequestIR(BaseModel):
    headers: dict[str, str] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    path_params: dict[str, Any] = Field(default_factory=dict, description="values for {param} path segments")
    payload: Any = None


class PollIR(BaseModel):
    interval_ms: int = 1000
    timeout_ms: int = 10000
    until: str


class StepIR(BaseModel):
    step: int | None = None
    name: str
    type: Literal["http", "delay", "recipe"] = "http"
    endpoint: str | None = Field(None, description="'METHOD /openapi/path', path segments may be templates")
    actor: str | None = Field(None, description="module actor for this step when it differs from the case actor")
    request: RequestIR = Field(default_factory=RequestIR)
    expected_status: int | str | None = Field(
        None, description="int, '401|403' (any of these, when the specs allow several), or '{{ $param.<key> }}'"
    )
    validate_schema: bool = False
    extract: dict[str, str] = Field(default_factory=dict, description="var -> response.body.<path>")
    assertions: list[AssertionIR] = Field(default_factory=list)
    expected_error: ExpectedError | None = None
    poll: PollIR | None = None
    delay_ms: int | None = None
    timeout_ms: int | None = None
    # type=recipe: a verified setup recipe, expanded into ordinary steps before validation/storage
    recipe: str | None = Field(None, description="recipe name (type=recipe only)")
    inputs: dict[str, Any] = Field(default_factory=dict, description="recipe inputs (type=recipe only)")
    alias: str | None = Field(None, description="recipe outputs become $extract.<alias>_<output> (type=recipe)")

    @model_validator(mode="before")
    @classmethod
    def _legacy_cleanup_action(cls, data: Any) -> Any:
        # Reference format cleanup items: {action: "DELETE /v1/orders/{{...}}", ignore_errors: true}
        if isinstance(data, dict) and "action" in data and "endpoint" not in data:
            data = {**data, "endpoint": data["action"], "name": data.get("name") or "cleanup"}
            data.pop("action", None)
        if isinstance(data, dict):
            data.pop("ignore_errors", None)
        return data


class CoverageIR(BaseModel):
    endpoints: list[str] = Field(default_factory=list)
    requirement_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_ref(cls, data: Any) -> Any:
        if isinstance(data, dict) and "requirement_ref" in data:
            data = dict(data)
            ref = data.pop("requirement_ref")
            refs = data.setdefault("requirement_refs", [])
            refs.extend([ref] if isinstance(ref, str) else list(ref or []))
        if isinstance(data, dict) and isinstance(data.get("requirement_refs"), str):
            data = {**data, "requirement_refs": [data["requirement_refs"]]}
        return data


class CaseDraft(BaseModel):
    """What the planner produces: a test case WITHOUT id (ids are assigned by code)."""

    title: str
    type: CaseType
    priority: Priority = "P1"
    tags: list[str] = Field(default_factory=list)
    actor: str | None = None
    coverage: CoverageIR = Field(default_factory=CoverageIR)
    preconditions: list[str] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    parameters: list[dict[str, Any]] = Field(default_factory=list, description="rows, each with a 'name' slug")
    steps: list[StepIR] = Field(min_length=1)
    expected_results: list[str] = Field(default_factory=list)
    cleanup: list[StepIR] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def _lower_type(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v


class TestCaseIR(CaseDraft):
    __test__ = False  # not a pytest class
    id: str | None = None
    status: DesignStatus = "draft"

    @field_validator("status", mode="before")
    @classmethod
    def _lower_status(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v


class ModuleHeaderIR(BaseModel):
    id: str
    domain: str
    folder: str | None = None
    target_spec: str | None = None
    product_spec: str | None = None


class ModuleDefaultsIR(BaseModel):
    base_path: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)


class ModuleIR(BaseModel):
    module: ModuleHeaderIR
    module_defaults: ModuleDefaultsIR = Field(default_factory=ModuleDefaultsIR)
    actors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fixtures: dict[str, Any] = Field(default_factory=dict)
    test_cases: list[TestCaseIR] = Field(default_factory=list)


# ---------------------------------------------------------------- validation


@dataclass
class ValidationContext:
    endpoint_paths: dict[str, str]  # endpoint key -> OpenAPI path (all specs)
    requirement_ids: set[str] = field(default_factory=set)
    actors: set[str] = field(default_factory=set)
    fixtures: set[str] = field(default_factory=set)
    env_vars: set[str] | None = None  # declared env variables; None = no restriction
    # endpoint key -> {status: JSON schema} of documented responses; None = no field checking
    response_schemas: dict[str, dict[str, Any]] | None = None
    request_schemas: dict[str, Any] | None = None  # endpoint key -> request body JSON schema
    page_limits: dict[str, tuple[str, int]] | None = None  # paginated endpoint -> (page-size param, maximum)


def missing_required(schema: Any, payload: Any, where: str = "payload") -> list[str]:
    """Required request-body fields absent from a literal payload (nested objects included)."""
    node = _merged(schema)
    if payload is None:
        payload = {}  # no body at all: every required field is missing
    if node is None or not isinstance(payload, dict):
        return []
    missing = [f"{where}.{k}" for k in node.get("required", []) if k not in payload]
    for key, value in payload.items():
        sub = (node.get("properties") or {}).get(key)
        if isinstance(value, dict) and sub is not None:
            missing += missing_required(sub, value, f"{where}.{key}")
    return missing


_PATH_PART = re.compile(r"\.([A-Za-z_][A-Za-z0-9_\-]*)|\[(\d+)\]|\[([A-Za-z_][A-Za-z0-9_]*)=(?:[^\]]+)\]")


def _merged(schema: Any) -> dict[str, Any] | None:
    """Merge allOf branches; None when the schema does not constrain properties (any/free-form)."""
    if not isinstance(schema, dict) or "$ref_cycle" in schema:
        return None
    if "allOf" in schema:
        merged: dict[str, Any] = {"type": "object", "properties": {}}
        required: set[str] = set()
        declared = False
        for part in schema["allOf"]:
            sub = _merged(part)
            if sub is None:
                return None
            merged["properties"].update(sub.get("properties", {}))
            if "required" in sub:
                declared = True
                required |= set(sub["required"])
        if declared:
            merged["required"] = sorted(required)
            if sub.get("items"):
                merged["items"] = sub["items"]
        return merged
    if "oneOf" in schema or "anyOf" in schema:
        return None
    return schema


def body_path_problem(schema: Any, path: str) -> str | None:
    """'response.body.data[0].product_id' vs the response schema -> problem text, or None if allowed/unknown."""
    node = _merged(schema)
    walked = "response.body"
    for m in _PATH_PART.finditer(path.removeprefix("response.body")):
        if node is None:
            return None
        if m.group(2) is not None or m.group(3) is not None:  # [n] or [field=value]
            items = _merged(node.get("items")) if (node.get("type") == "array" or "items" in node) else None
            if m.group(3) is not None and items and items.get("properties") and m.group(3) not in items["properties"]:
                return (
                    f"{walked}[{m.group(3)}=...] filters on a field the list items do not have "
                    f"(available: {sorted(items['properties'])})"
                )
            node = items
            walked += f"[{m.group(2) or m.group(3) + '=...'}]"
            continue
        key = m.group(1)
        if node.get("type") == "array" or "items" in node:
            if key == "length":
                return None
            return (
                f"{walked} is a list: select an item before .{key} "
                f"({walked}[0].{key} or a filter like {walked}[field='value'].{key})"
            )
        props = node.get("properties")
        if not props or node.get("additionalProperties") not in (None, False):
            return None
        if key not in props:
            return f"{walked}.{key} is not in the documented response (available: {sorted(props)})"
        if "required" in node and key not in node["required"]:
            return (
                f"{walked}.{key} is optional in the contract (required: {sorted(node['required'])}), "
                "the API may omit it - do not assert or extract it"
            )
        node = _merged(props[key])
        walked += f".{key}"
    return None


def declared_env_vars(*objs: Any) -> set[str]:
    """Env variable names declared by actors / fixtures: '{{ $env.X }}' references and {"env": "X"} values."""
    import json as _json

    text = _json.dumps([o.model_dump() if hasattr(o, "model_dump") else o for o in objs], default=str)
    return set(REF_ENV_RE.findall(text)) | set(re.findall(r'"env":\s*"([A-Z][A-Z0-9_]*)"', text))


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _strings(v)


def _segments_match(template_path: str, spec_path: str) -> bool:
    tpl = template_path.split("?")[0].rstrip("/").split("/")
    spec = spec_path.rstrip("/").split("/")
    if len(tpl) != len(spec):
        return False
    for t, s in zip(tpl, spec, strict=True):
        if s.startswith("{") and s.endswith("}"):
            continue  # any value (template or literal) fills a path parameter
        if t != s:
            return False
    return True


def resolve_endpoint(endpoint: str, endpoint_paths: dict[str, str]) -> str | None:
    """'GET /v1/orders/{{ $extract.order_id }}' -> 'GET /v1/orders/{orderId}' (spec key) or None."""
    parts = endpoint.strip().split(None, 1)
    if len(parts) != 2:
        return None
    method, path = parts[0].upper(), parts[1].strip()
    if f"{method} {path}" in endpoint_paths:
        return f"{method} {path}"
    for key, spec_path in endpoint_paths.items():
        if key.split(" ", 1)[0] == method and _segments_match(path, spec_path):
            return key
    return None


def http_steps(case: CaseDraft) -> list[tuple[StepIR, bool]]:
    """All steps in execution order with an is_cleanup flag."""
    return [(s, False) for s in case.steps] + [(s, True) for s in case.cleanup]


def validate_case(case: CaseDraft, ctx: ValidationContext) -> list[str]:
    errors: list[str] = []
    var_names = set(case.variables)
    param_keys: set[str] | None = None
    if case.parameters:
        if len(case.parameters) > 26:
            errors.append("at most 26 parameter rows")
        names = []
        for i, row in enumerate(case.parameters):
            name = row.get("name")
            if not isinstance(name, str) or not SLUG_RE.match(name):
                errors.append(f"parameters[{i}] needs a kebab-case 'name' slug, e.g. 'zero-quantity'")
            names.append(name)
            keys = set(row) - {"name"}
            param_keys = keys if param_keys is None else param_keys & keys
        if len(set(names)) != len(names):
            errors.append("parameter row names must be unique")
    param_keys = param_keys or set()

    if case.actor and case.actor not in ctx.actors:
        errors.append(f"unknown actor {case.actor!r}; module actors: {sorted(ctx.actors)}")
    for ref in case.coverage.requirement_refs:
        if ctx.requirement_ids and ref not in ctx.requirement_ids:
            errors.append(f"unknown requirement_ref {ref!r}")

    def check_refs(where: str, obj: Any, extracted: set[str], allow: set[str] = NAMESPACES) -> None:
        for s in _strings(obj):
            for raw in ANY_TEMPLATE_RE.findall(s):
                if not raw.strip().startswith("$"):
                    errors.append(f"{where}: '{{{{{raw}}}}}' must use a namespace like {{{{ $extract.x }}}}")
            for ns, name in REF_RE.findall(s):
                if ns not in allow:
                    errors.append(f"{where}: namespace ${ns} is not allowed here")
                elif ns == "faker" and bruno_dynamic(name) is None:
                    errors.append(f"{where}: unknown faker type {name!r}")
                elif ns == "fixture" and name not in ctx.fixtures:
                    errors.append(f"{where}: unknown fixture {name!r}; module fixtures: {sorted(ctx.fixtures)}")
                elif ns == "env" and not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                    errors.append(f"{where}: env names are UPPER_SNAKE ({name!r})")
                elif ns == "env" and ctx.env_vars is not None and name not in ctx.env_vars:
                    errors.append(
                        f"{where}: $env.{name} is not a declared environment variable "
                        f"(declared: {sorted(ctx.env_vars) or 'none'}). Do not use $env for test data such as ids: "
                        "look the value up in an earlier step (e.g. GET the list, then extract) or use a fixture"
                    )
                elif ns == "extract" and name not in extracted:
                    errors.append(f"{where}: $extract.{name} is not extracted by an earlier step")
                elif ns == "param" and name not in param_keys:
                    errors.append(f"{where}: $param.{name} is missing from some parameter rows")
                elif ns == "var" and name not in var_names:
                    errors.append(f"{where}: $var.{name} is not defined in variables")

    check_refs("variables", case.variables, set(), {"faker", "fixture", "env"})

    extracted: set[str] = set()
    created: set[str] = set()  # extract vars that hold ids of entities created by this case (non-login POST)
    looked_up: set[str] = set()  # extract vars read from shared lists/entities (GET)
    resolved: list[str] = []
    for idx, (step, is_cleanup) in enumerate(http_steps(case), start=1):
        where = f"{'cleanup' if is_cleanup else 'step'} {idx} ({step.name})"
        if step.actor and step.actor not in ctx.actors:
            errors.append(f"{where}: unknown actor {step.actor!r}; module actors: {sorted(ctx.actors)}")
        if step.type == "recipe":
            errors.append(f"{where}: recipe {step.recipe!r} is not available (use only the listed recipes)")
            continue
        if step.type == "delay":
            if not step.delay_ms or step.delay_ms <= 0:
                errors.append(f"{where}: delay steps need delay_ms > 0")
            continue
        if not step.endpoint:
            errors.append(f"{where}: http steps need endpoint 'METHOD /path'")
            continue
        method = step.endpoint.split(None, 1)[0].upper()
        if method not in HTTP_METHODS:
            errors.append(f"{where}: unknown HTTP method {method!r}")
        key = resolve_endpoint(step.endpoint, ctx.endpoint_paths)
        if key is None:
            errors.append(f"{where}: endpoint {step.endpoint!r} does not match any OpenAPI path")
        else:
            resolved.append(key)
        check_refs(
            where,
            [
                step.endpoint,
                step.request.model_dump(),
                step.expected_error.model_dump() if step.expected_error else None,
            ],
            extracted,
        )
        es = step.expected_status
        if es is None:
            if not is_cleanup:
                errors.append(f"{where}: expected_status is required")
        elif isinstance(es, int):
            if not 100 <= es <= 599:
                errors.append(f"{where}: expected_status {es} is not an HTTP status")
        else:
            m = PARAM_TEMPLATE_RE.match(es.strip())
            if not (es.strip().isdigit() or STATUS_SET_RE.match(es.strip()) or (m and m.group(1) in param_keys)):
                errors.append(f"{where}: expected_status must be an int, '401|403' or '{{{{ $param.<key> }}}}'")
            rows = [str(r.get(m.group(1), "")) for r in case.parameters] if m else [es]
            sets = [parse_statuses(v) for v in rows if STATUS_SET_RE.match(v.strip())]
            documented = (ctx.response_schemas or {}).get(key or "") if ctx.response_schemas is not None else None
            for alternatives in sets if documented else []:
                if undocumented := sorted(st for st in alternatives if str(st) not in documented):
                    errors.append(
                        f"{where}: a status set may only list documented statuses; {undocumented} are not "
                        f"documented for {key}"
                    )
        for a in step.assertions:
            try:
                parse_check(a.check)
            except CheckSyntaxError as exc:
                errors.append(f"{where}: {exc}")
            check_refs(where, a.check, extracted)
        for var, path in step.extract.items():
            if not IDENT_RE.match(var):
                errors.append(f"{where}: extract name {var!r} must be an identifier")
            if not PATH_RE.match(path):
                errors.append(f"{where}: extract path {path!r} must be response.body.<path> or response.headers.<h>")
        statuses = step_statuses(step, case.parameters)
        creates = any(200 <= s < 300 for s in statuses)
        body_valid = statuses and not is_cleanup and not any(s in BODY_ERROR_STATUSES for s in statuses)
        if body_valid and ctx.request_schemas is not None and key and ctx.request_schemas.get(key):
            missing = missing_required(ctx.request_schemas[key], step.request.payload)
            if missing and creates:
                errors.append(
                    f"{where}: the request is expected to succeed but misses fields the contract requires: "
                    f"{missing} - send them (with valid values)"
                )
            elif missing and step.request.payload is None:  # a partial body may be deliberate ({} -> 401)
                errors.append(
                    f"{where}: expects {statuses[0]} but sends no body, while the contract requires {missing}. The "
                    "API rejects an absent body with 400 before reaching the condition under test - send a complete, "
                    "valid body (e.g. wrong password -> real username/password fields)"
                )
        if creates and method in ("POST", "PUT", "PATCH") and "login" not in step.endpoint.lower():
            for field_name, var in _id_references(step.request.payload):
                if var in looked_up and var not in created:
                    errors.append(
                        f"{where}: {field_name}={{{{ $extract.{var} }}}} is an entity looked up from shared data; "
                        "other tests drain and change it (stock, price, balance). Use an entity this case creates: "
                        "a listed recipe that prepares it (a composite one when the actor must also be allowed to "
                        "see/buy it) or POST it in an earlier step"
                    )
        if method == "POST" and creates and "login" not in step.endpoint.lower():
            for field_name, value in _unique_literals(step.request.payload):
                errors.append(
                    f"{where}: {field_name}={value!r} is a fixed value for a field that must be unique; the entity may "
                    "already exist from an earlier run (deletes can be soft). Use a per-run unique value, e.g. "
                    "variables: {name: 'qa_{{ $faker.uuid }}'} and {{ $var.name }}"
                )
        if ctx.response_schemas is not None and key and key in ctx.response_schemas:
            statuses = [str(s) for s in step_statuses(step, case.parameters)]
            schemas = [(ctx.response_schemas[key].get(s) or {}).get("schema") for s in statuses]
            schemas = [sc for sc in schemas if sc]
            paths = list(step.extract.values())
            for a in step.assertions:
                with contextlib.suppress(CheckSyntaxError):  # syntax errors are reported above
                    chk = parse_check(a.check)
                    if chk.op != "not exists":  # asserting that an undocumented field is absent is valid
                        paths.append(chk.path)
            for path in (p for p in paths if p.startswith("response.body")):
                problems = [body_path_problem(sc, path) for sc in schemas]
                if schemas and all(problems):
                    errors.append(f"{where}: {problems[0]}; assert only documented fields")
        if ctx.page_limits and key in ctx.page_limits and creates_or_cleanup(step, is_cleanup, case.parameters):
            size_param, maximum = ctx.page_limits[key]
            checks = [*step.extract.values(), *(a.check for a in step.assertions)]
            filtered = [c for c in checks if FILTER_IN_PATH.search(c)]
            if filtered and str(step.request.query.get(size_param, "")) != str(maximum):
                errors.append(
                    f"{where}: {filtered[0].split()[0]} searches one page of a paginated list - other tests add items, "
                    f"so the wanted item can be on a later page. Send query {size_param}={maximum} and, where the "
                    "endpoint supports it, sort newest first (e.g. sort_by=created_at, order=desc), or read the "
                    "entity directly by id"
                )
        path_part = step.endpoint.split(None, 1)[1] if " " in step.endpoint else ""
        path_vars = {n for ns, n in REF_RE.findall(path_part) if ns == "extract"}
        mutates = method in ("PUT", "PATCH", "DELETE") or (method == "POST" and path_vars)
        foreign = sorted(path_vars - created)
        if mutates and foreign and creates_or_cleanup(step, is_cleanup, case.parameters):
            errors.append(
                f"{where}: modifies a pre-existing entity ({', '.join('$extract.' + v for v in foreign)} comes from a "
                "lookup/login, not from a POST that created it in this case). Tests must not change shared seed data "
                "(other tests depend on it): create a dedicated entity first (e.g. POST /api/admin/users) and modify that"
            )
        if method == "POST" and "login" not in step.endpoint.lower():
            created |= set(step.extract)
        if method == "GET":
            looked_up |= set(step.extract)
        extracted |= set(step.extract)
        if step.validate_schema and key is None:
            errors.append(f"{where}: validate_schema needs a resolvable endpoint")

    for e in case.coverage.endpoints:
        rk = resolve_endpoint(e, ctx.endpoint_paths)
        if rk is None or rk not in resolved:
            errors.append(f"coverage endpoint {e!r} is not exercised by any step")
    return errors


def creates_or_cleanup(step: StepIR, is_cleanup: bool, parameters: list[dict[str, Any]]) -> bool:
    """Only steps that are expected to succeed actually modify data (negative tests may target anything)."""
    return is_cleanup or any(200 <= s < 300 for s in step_statuses(step, parameters))


UNIQUE_FIELDS = {"username", "email", "login", "user_name"}
BODY_ERROR_STATUSES = {400, 413, 415, 422}  # statuses a test may provoke with an incomplete/invalid body
_EXTRACT_ONLY_RE = re.compile(r"^\{\{\s*\$extract\.([A-Za-z0-9_]+)\s*\}\}$")


def _id_references(payload: Any) -> list[tuple[str, str]]:
    """(field, extract var) for id fields whose value is exactly one {{ $extract.x }} template."""
    found: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            m = _EXTRACT_ONLY_RE.match(v.strip()) if isinstance(v, str) else None
            if m and (k == "id" or k.lower().endswith("_id")):
                found.append((k, m.group(1)))
            elif isinstance(v, dict | list):
                found.extend(_id_references(v))
    elif isinstance(payload, list):
        for v in payload:
            found.extend(_id_references(v))
    return found


def _unique_literals(payload: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k.lower() in UNIQUE_FIELDS and isinstance(v, str) and v.strip() and "{{" not in v:
                found.append((k, v))
            elif isinstance(v, dict | list):
                found.extend(_unique_literals(v))
    elif isinstance(payload, list):
        for v in payload:
            found.extend(_unique_literals(v))
    return found


def parse_statuses(value: Any) -> list[int]:
    """401 / '401' / '401|403' -> [401] / [401] / [401, 403]; anything else -> []."""
    if isinstance(value, int):
        return [value]
    text = str(value).strip()
    if text.isdigit() or STATUS_SET_RE.match(text):
        return [int(v) for v in text.split("|")]
    return []


def step_statuses(step: StepIR, parameters: list[dict[str, Any]]) -> list[int]:
    """Every status the step may expect (all parameter rows, all alternatives)."""
    es = step.expected_status
    if es is None:
        return []
    m = PARAM_TEMPLATE_RE.match(str(es).strip())
    if m:
        return [st for r in parameters for st in parse_statuses(r.get(m.group(1), ""))]
    return parse_statuses(es)


def normalize(case: CaseDraft) -> CaseDraft:
    """Fold expected_error sugar into assertions; fill computed coverage; number steps."""
    data = case.model_copy(deep=True)
    for i, (step, _) in enumerate(http_steps(data), start=1):
        step.step = i
        err = step.expected_error
        if err:
            for attr in ("code", "field", "message"):
                value = getattr(err, attr)
                if value is not None:
                    literal = value if PARAM_TEMPLATE_RE.match(str(value)) else f"'{value}'"
                    check = f"response.body.error.{attr} == {literal}"
                    if all(a.check != check for a in step.assertions):
                        step.assertions.append(AssertionIR(check=check))
            step.expected_error = None
    return data


def covered_endpoints(case: CaseDraft, endpoint_paths: dict[str, str]) -> list[str]:
    keys = []
    for step, _ in http_steps(case):
        if step.type == "http" and step.endpoint:
            k = resolve_endpoint(step.endpoint, endpoint_paths)
            if k and k not in keys:
                keys.append(k)
    return keys


def fingerprint(module_code: str, case: CaseDraft, endpoint_paths: dict[str, str]) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", case.title.lower()).strip()
    return stable_hash([module_code, sorted(covered_endpoints(case, endpoint_paths)), case.type, title])[:32]


def ir_hash(case: TestCaseIR, module_context: dict[str, Any]) -> str:
    """Hash of everything the synthesizer sees for one case (to skip unchanged rebuilds)."""
    return stable_hash([case.model_dump(mode="json"), module_context])[:32]

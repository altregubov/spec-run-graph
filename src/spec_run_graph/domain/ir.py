"""Test-case intermediate representation (the reference test-case format, with review fixes applied).

Context variable syntax:
  {{ $faker.<type> }}    dynamic data           {{ $fixture.<name> }}  module fixture
  {{ $env.<KEY> }}       environment variable   {{ $extract.<var> }}   captured in an earlier step
  {{ $param.<key> }}     data-driven row value  {{ $var.<name> }}      case-level variable
"""

from __future__ import annotations

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

REF_RE = re.compile(r"\{\{\s*\$([a-z]+)\.([A-Za-z0-9_.\-]+)\s*\}\}")
ANY_TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}")
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PARAM_TEMPLATE_RE = re.compile(r"^\{\{\s*\$param\.([A-Za-z0-9_]+)\s*\}\}$")
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
    type: Literal["http", "delay"] = "http"
    endpoint: str | None = Field(None, description="'METHOD /openapi/path', path segments may be templates")
    request: RequestIR = Field(default_factory=RequestIR)
    expected_status: int | str | None = Field(None, description="int, or '{{ $param.<key> }}'")
    validate_schema: bool = False
    extract: dict[str, str] = Field(default_factory=dict, description="var -> response.body.<path>")
    assertions: list[AssertionIR] = Field(default_factory=list)
    expected_error: ExpectedError | None = None
    poll: PollIR | None = None
    delay_ms: int | None = None
    timeout_ms: int | None = None

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
                elif ns == "extract" and name not in extracted:
                    errors.append(f"{where}: $extract.{name} is not extracted by an earlier step")
                elif ns == "param" and name not in param_keys:
                    errors.append(f"{where}: $param.{name} is missing from some parameter rows")
                elif ns == "var" and name not in var_names:
                    errors.append(f"{where}: $var.{name} is not defined in variables")

    check_refs("variables", case.variables, set(), {"faker", "fixture", "env"})

    extracted: set[str] = set()
    resolved: list[str] = []
    for idx, (step, is_cleanup) in enumerate(http_steps(case), start=1):
        where = f"{'cleanup' if is_cleanup else 'step'} {idx} ({step.name})"
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
            if not (es.strip().isdigit() or (m and m.group(1) in param_keys)):
                errors.append(f"{where}: expected_status must be an int or '{{{{ $param.<key> }}}}'")
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
        extracted |= set(step.extract)
        if step.validate_schema and key is None:
            errors.append(f"{where}: validate_schema needs a resolvable endpoint")

    for e in case.coverage.endpoints:
        rk = resolve_endpoint(e, ctx.endpoint_paths)
        if rk is None or rk not in resolved:
            errors.append(f"coverage endpoint {e!r} is not exercised by any step")
    return errors


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

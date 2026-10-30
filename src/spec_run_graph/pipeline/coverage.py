"""Deterministic coverage gaps of a module's cases, fed back to the planner in a gap-fill pass.

A gap is something the specs define that no case checks: a requirement, an acceptance criterion, a documented
error status of an endpoint, or a constrained request field (minimum, enum, format ...) that no negative step
sends an invalid value for."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from spec_run_graph.domain.ir import (
    PARAM_TEMPLATE_RE,
    REF_RE,
    CaseDraft,
    _merged,
    criterion_ids,
    http_steps,
    resolve_endpoint,
    step_statuses,
)

CONSTRAINTS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "enum",
    "pattern",
    "format",
)
UNTESTABLE_STATUSES = {500, 502, 503, 504}  # server faults cannot be provoked on purpose


@dataclass
class Gap:
    text: str
    endpoints: list[str]  # where it belongs (planner chunk); empty = module-wide


@dataclass
class Coverage:
    criteria_total: int
    criteria_covered: int
    gaps: list[Gap]


UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def violates(prop: dict[str, Any], value: Any) -> bool:
    """Does a literal value break the field's schema (null, wrong type, out of range, not in enum, bad format)?"""
    if value is None:
        return True
    kind = prop.get("type")
    number = isinstance(value, int | float) and not isinstance(value, bool)
    if kind in ("number", "integer") and not number:
        return True
    if kind == "integer" and number and float(value) != int(value):
        return True
    if kind == "string" and not isinstance(value, str):
        return True
    if kind == "array" and not isinstance(value, list):
        return True
    if number:
        lo, hi = prop.get("minimum"), prop.get("maximum")
        xlo, xhi = prop.get("exclusiveMinimum"), prop.get("exclusiveMaximum")
        if xlo is True:  # OpenAPI 3.0 boolean form
            xlo, lo = lo, None
        if xhi is True:
            xhi, hi = hi, None
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            return True
        if (isinstance(xlo, int | float) and value <= xlo) or (isinstance(xhi, int | float) and value >= xhi):
            return True
    if isinstance(value, str):
        if len(value) < prop.get("minLength", 0) or len(value) > prop.get("maxLength", len(value)):
            return True
        if prop.get("pattern") and not re.search(prop["pattern"], value):
            return True
        if prop.get("format") == "uuid" and not UUID_RE.match(value):
            return True
    if isinstance(value, list) and not prop.get("minItems", 0) <= len(value) <= prop.get("maxItems", len(value)):
        return True
    return "enum" in prop and value not in prop["enum"]


SECRET_FIELD_RE = re.compile(r"password|secret|token|key", re.I)
NO_TECHNIQUE_PATH_RE = re.compile(r"login|auth|token|session", re.I)


def is_free_text(name: str, prop: dict[str, Any]) -> bool:
    """A stored text field without a format of its own (or a list of them)."""
    node = prop.get("items") or {} if prop.get("type") == "array" else prop
    plain = node.get("type") == "string" and not any(c in node for c in ("enum", "format", "pattern"))
    return plain and not SECRET_FIELD_RE.search(name)


def is_decimal(prop: dict[str, Any]) -> bool:
    return prop.get("type") == "number"


def _texts(value: Any, case: CaseDraft) -> list[str]:
    """Every text a payload field takes, including the literals around templates and the case's variables."""
    out = [json.dumps(v, ensure_ascii=False) for v in _sent_values(value, case.parameters)]
    if isinstance(value, list):
        return out + [t for v in value for t in _texts(v, case)]
    if isinstance(value, str):
        out.append(value)
        out += [str(case.variables.get(name, "")) for ns, name in REF_RE.findall(value) if ns == "var"]
    return out


def _sent_values(value: Any, parameters: list[dict[str, Any]]) -> list[Any]:
    """Literal values a payload field takes: the literal itself or every parameter row's value."""
    if isinstance(value, str):
        refs = REF_RE.findall(value)
        if refs:
            m = PARAM_TEMPLATE_RE.match(value.strip())
            return [row.get(m.group(1)) for row in parameters if m.group(1) in row] if m else []
    return [value]


def requirement_owners(modules: list[Any], requirements: list[Any]) -> dict[str, str]:
    """Each requirement belongs to ONE module (so its gaps are planned once): the module that lists it, else the
    first module owning one of its endpoints."""
    owners = {}
    for r in requirements:
        owner = next((m.code for m in modules if r.req_id in m.requirement_ids), None) or next(
            (m.code for m in modules if set(r.endpoints) & set(m.endpoints)), None
        )
        if owner:
            owners[r.req_id] = owner
    return owners


def chunk_requirement_ids(module: Any, chunk: list[str], requirements: list[Any], first: bool) -> list[str]:
    """The module requirements one planner chunk answers for: those of its endpoints; requirements naming no endpoint
    of the module go to the first chunk only. Otherwise parallel chunks each plan the whole module."""
    own, here = set(module.endpoints), set(chunk)
    by_id = {r.req_id: r for r in requirements}
    keep = []
    for req_id in module.requirement_ids:
        named = set(getattr(by_id.get(req_id), "endpoints", None) or []) & own
        if named & here or (not named and first):
            keep.append(req_id)
    return keep


def module_coverage(
    module: Any,
    requirements: list[Any],
    cases: list[CaseDraft],
    endpoints: dict[str, Any],
    paths: dict[str, str],
    all_cases: list[CaseDraft] | None = None,
    owners: dict[str, str] | None = None,
) -> Coverage:
    """Gaps of one module: its own requirements (owners) checked against all cases of the suite (all_cases), its
    endpoints' statuses and fields against the module's cases."""
    own = set(module.endpoints)
    if owners is None:
        reqs = [r for r in requirements if r.req_id in module.requirement_ids or set(r.endpoints) & own]
    else:
        reqs = [r for r in requirements if owners.get(r.req_id) == module.code]
    gaps: list[Gap] = []

    suite = cases if all_cases is None else all_cases
    covered_reqs = {ref for c in suite for ref in c.coverage.requirement_refs}
    covered_crit = {ref for c in suite for ref in c.coverage.criteria}
    criteria = criterion_ids(reqs)
    for r in reqs:
        where = sorted(set(r.endpoints) & own)
        if r.req_id not in covered_reqs:
            gaps.append(Gap(f"requirement {r.req_id} ({r.title}) has no test case", where))
            continue
        for cid in (f"{r.req_id}#{n}" for n in range(1, len(r.acceptance_criteria) + 1)):
            if cid not in covered_crit:
                gaps.append(Gap(f"acceptance criterion {cid} is not checked: {criteria[cid]}", where))

    expected: dict[str, set[int]] = {}
    invalid_fields: dict[str, set[str]] = {}
    non_ascii: dict[str, set[str]] = {}  # endpoint -> text fields a successful step sends non-ASCII text for
    cents: dict[str, set[str]] = {}  # endpoint -> decimal fields a successful step sends a cent-precise value for
    over_precise: dict[str, set[str]] = {}  # endpoint -> decimal fields some step sends more than 2 decimals for
    for c in cases:
        for step, is_cleanup in http_steps(c):
            if is_cleanup or step.type != "http" or not step.endpoint:
                continue
            key = resolve_endpoint(step.endpoint, paths)
            if key not in own:
                continue
            statuses = step_statuses(step, c.parameters)
            expected.setdefault(key, set()).update(statuses)
            props = (_merged(endpoints[key].request_schema) or {}).get("properties") or {} if key in endpoints else {}
            if statuses and all(400 <= s < 500 for s in statuses) and isinstance(step.request.payload, dict):
                invalid_fields.setdefault(key, set()).update(
                    k
                    for k, v in step.request.payload.items()
                    if k in props and any(violates(props[k] or {}, x) for x in _sent_values(v, c.parameters))
                )
            if isinstance(step.request.payload, dict):
                success = any(200 <= s < 300 for s in statuses)
                for k, v in step.request.payload.items():
                    numbers = [x for x in _sent_values(v, c.parameters) if isinstance(x, float)]
                    if any(round(x, 2) != x for x in numbers):
                        over_precise.setdefault(key, set()).add(k)
                    if success and any(round(x, 2) == x and x != int(x) for x in numbers):
                        cents.setdefault(key, set()).add(k)
                    if success and any(not t.isascii() for t in _texts(v, c)):
                        non_ascii.setdefault(key, set()).add(k)

    for key in sorted(own):
        e = endpoints.get(key)
        if e is None:
            continue
        documented = {int(s) for s in (e.response_schemas or {}) if str(s).isdigit()}
        for status in sorted(documented - expected.get(key, set()) - UNTESTABLE_STATUSES):
            gaps.append(Gap(f"{key}: documented status {status} is never expected", [key]))
        schema = _merged(e.request_schema) or {}
        for name, prop in sorted((schema.get("properties") or {}).items()):
            rules = {c: (prop or {})[c] for c in CONSTRAINTS if c in (prop or {})}
            if rules and name not in invalid_fields.get(key, set()):
                text = f"{key}: field {name} {rules} - no negative step sends a value breaking it"
                gaps.append(Gap(text + " (out of range, wrong type, null, bad format)", [key]))
            if key.split()[0] in ("POST", "PUT", "PATCH") and not NO_TECHNIQUE_PATH_RE.search(key):
                prop = prop or {}
                if is_free_text(name, prop) and name not in non_ascii.get(key, set()):
                    gaps.append(
                        Gap(f"{key}: text {name} - no successful step sends non-ASCII text and reads it back", [key])
                    )
                if is_decimal(prop) and name not in cents.get(key, set()):
                    gaps.append(
                        Gap(
                            f"{key}: decimal {name} - no successful step sends a cent-precise value (e.g. 19.99)", [key]
                        )
                    )
                if is_decimal(prop) and name not in over_precise.get(key, set()):
                    gaps.append(
                        Gap(
                            f"{key}: decimal {name} - no step sends more digits than the precision allows (e.g. 19.999)",
                            [key],
                        )
                    )
    return Coverage(len(criteria), len(set(criteria) & covered_crit), gaps)

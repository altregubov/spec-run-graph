"""Deterministic OpenAPI 3.x / Swagger 2.0 parsing: endpoints, resolved schemas and a compact prompt digest."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from spec_run_graph.specs.contract import required_body_fields, response_facts
from spec_run_graph.specs.models import HTTP_METHODS, EndpointDraft, LoadedSpec

MAX_REF_DEPTH = 12


def _resolve_pointer(root: dict, ref: str) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $refs are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def resolve_refs(node: Any, root: dict, seen: tuple[str, ...] = ()) -> Any:
    """Inline local $refs. Cycles (and very deep chains) are cut with {"$ref_cycle": ref}."""
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            if ref in seen or len(seen) >= MAX_REF_DEPTH:
                return {"$ref_cycle": ref}
            target = copy.deepcopy(_resolve_pointer(root, ref))
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            resolved = resolve_refs(target, root, (*seen, ref))
            if siblings and isinstance(resolved, dict):
                resolved = {**resolved, **resolve_refs(siblings, root, seen)}
            return resolved
        return {k: resolve_refs(v, root, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_refs(v, root, seen) for v in node]
    return node


def _json_schema_of(content: dict | None) -> Any | None:
    if not content:
        return None
    for media in ("application/json", "application/problem+json", "*/*"):
        if media in content:
            return content[media].get("schema")
    first = next(iter(content.values()), {})
    return first.get("schema")


def parse_document(raw: str, suffix: str) -> Any:
    return json.loads(raw) if suffix == ".json" else yaml.safe_load(raw)


def is_api_document(doc: Any) -> bool:
    """OpenAPI 3.x (`openapi:`) or Swagger 2.0 (`swagger: "2.0"`)."""
    return isinstance(doc, dict) and ("openapi" in doc or str(doc.get("swagger", "")).startswith("2"))


def _swagger2_param_schema(prm: dict) -> dict:
    """Swagger 2.0 non-body parameters carry their type inline instead of in `schema`."""
    keys = ("type", "format", "items", "enum", "minimum", "maximum", "minLength", "maxLength", "pattern", "default")
    return {k: prm[k] for k in keys if k in prm}


def load_openapi(path: Path) -> LoadedSpec:
    raw = path.read_text()
    doc = parse_document(raw, path.suffix)
    if not is_api_document(doc):
        raise ValueError(f"{path} is not an OpenAPI 3.x or Swagger 2.0 document")
    swagger2 = "openapi" not in doc
    info = doc.get("info", {})
    global_security = doc.get("security")
    # Swagger 2.0 paths are relative to basePath; endpoint keys use the full request path.
    base_path = (doc.get("basePath") or "").rstrip("/") if swagger2 else ""
    endpoints: list[EndpointDraft] = []
    for p, item in (doc.get("paths") or {}).items():
        item = resolve_refs(item, doc)
        shared_params = item.get("parameters", [])
        for method, op in item.items():
            if method.upper() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            params = [*shared_params, *op.get("parameters", [])]
            if swagger2:
                body_params = [prm for prm in params if prm.get("in") == "body"]
                form_params = [prm for prm in params if prm.get("in") == "formData"]
                params = [prm for prm in params if prm.get("in") not in ("body", "formData")]
                request_schema = body_params[0].get("schema") if body_params else None
                if form_params and request_schema is None:
                    request_schema = {
                        "type": "object",
                        "required": [prm["name"] for prm in form_params if prm.get("required")],
                        "properties": {prm["name"]: _swagger2_param_schema(prm) for prm in form_params},
                    }
                responses = {
                    str(status): {"description": resp.get("description", ""), "schema": resp.get("schema")}
                    for status, resp in (op.get("responses") or {}).items()
                }
            else:
                request_schema = _json_schema_of((op.get("requestBody") or {}).get("content"))
                responses = {
                    str(status): {
                        "description": resp.get("description", ""),
                        "schema": _json_schema_of(resp.get("content")),
                    }
                    for status, resp in (op.get("responses") or {}).items()
                }
            security = op.get("security", global_security)
            endpoints.append(
                EndpointDraft(
                    method=method.upper(),
                    path=base_path + p,
                    operation_id=op.get("operationId"),
                    summary=op.get("summary") or op.get("description"),
                    tag=(op.get("tags") or [None])[0],
                    parameters=[
                        {
                            "name": prm.get("name"),
                            "in": prm.get("in"),
                            "required": bool(prm.get("required")),
                            "schema": prm.get("schema") or _swagger2_param_schema(prm),
                        }
                        for prm in params
                    ],
                    request_schema=request_schema,
                    response_schemas=responses,
                    requires_auth=bool(security),
                )
            )
    if swagger2:
        scheme = (doc.get("schemes") or ["http"])[0]
        servers = [f"{scheme}://{doc['host']}{base_path}"] if doc.get("host") else []
    else:
        servers = [s.get("url") for s in doc.get("servers", []) if s.get("url")]
    return LoadedSpec(
        name=info.get("title") or path.stem,
        spec_type="OPENAPI",
        version=str(info.get("version") or "0"),
        source_path=str(path),
        raw_content=raw,
        endpoints=endpoints,
        servers=servers,
    )


# ---------- compact digest for prompts ----------


def schema_brief(schema: Any, depth: int = 0) -> str:
    """Render a JSON schema as a short, LLM-friendly type expression (`*` marks required fields)."""
    if not isinstance(schema, dict):
        return "any"
    if "$ref_cycle" in schema:
        return schema["$ref_cycle"].rsplit("/", 1)[-1]
    if depth > 4:
        return "{...}"
    for combo in ("oneOf", "anyOf", "allOf"):
        if combo in schema:
            sep = " & " if combo == "allOf" else " | "
            return sep.join(schema_brief(s, depth + 1) for s in schema[combo])
    if "enum" in schema:
        return "|".join(str(v) for v in schema["enum"])
    t = schema.get("type")
    if isinstance(t, list):
        t = "|".join(t)
    if t == "array" or "items" in schema:
        return f"[{schema_brief(schema.get('items', {}), depth + 1)}]{_bounds(schema, 'minItems', 'maxItems')}"
    if t == "object" or "properties" in schema:
        req = set(schema.get("required", []))
        props = schema.get("properties", {})
        if not props:
            return "object"
        inner = ", ".join(f"{k}{'*' if k in req else ''}: {schema_brief(v, depth + 1)}" for k, v in props.items())
        return "{" + inner + "}"
    out = str(t or "any")
    if fmt := schema.get("format"):
        out += f"({fmt})"
    return out + _bounds(schema, "minimum", "maximum") + _bounds(schema, "minLength", "maxLength")


def _bounds(schema: dict, lo: str, hi: str) -> str:
    a, b = schema.get(lo), schema.get(hi)
    if a is None and b is None:
        return ""
    return f" {'' if a is None else a}..{'' if b is None else b}"


def endpoint_digest(e: EndpointDraft) -> str:
    lines = [f"{e.key}  [{e.tag or '-'}] {e.operation_id or ''} - {e.summary or ''}".rstrip()]
    if e.requires_auth:
        lines.append("  auth: bearer required")
    for p in e.parameters:
        lines.append(f"  {p['in']} param {p['name']}{'*' if p['required'] else '?'}: {schema_brief(p.get('schema'))}")
    if e.request_schema is not None:
        lines.append(f"  body: {schema_brief(e.request_schema)}")
        if required := required_body_fields(e.request_schema):
            lines.append(f"    a successful request MUST send: {', '.join(required)}")
    facts = response_facts(e.response_schemas)
    groups: dict[tuple[str, str], list[str]] = {}  # identical schemas (e.g. the error envelope) are listed once
    for status, r in e.response_schemas.items():
        schema = f" -> {schema_brief(r['schema'])}" if r.get("schema") else ""
        assertable = ", ".join(facts.get(str(status), []))
        groups.setdefault((schema, assertable), []).append(f"{status} {r.get('description', '')}".strip())
    for (schema, assertable), statuses in groups.items():
        lines.append(f"  {' | '.join(statuses)}{schema}")
        if assertable:
            lines.append(f"    assertable (guaranteed) fields: {assertable}")
    return "\n".join(lines)


def spec_digest(spec: LoadedSpec) -> str:
    head = f"# OpenAPI spec '{spec.name}' v{spec.version} ({spec.source_path})"
    return "\n".join([head, *(endpoint_digest(e) for e in spec.endpoints)])

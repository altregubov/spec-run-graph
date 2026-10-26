"""Deterministic OpenAPI 3.x parsing: endpoints, resolved schemas and a compact prompt digest."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

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


def load_openapi(path: Path) -> LoadedSpec:
    raw = path.read_text()
    doc = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    if not isinstance(doc, dict) or "openapi" not in doc:
        raise ValueError(f"{path} is not an OpenAPI 3.x document")
    info = doc.get("info", {})
    global_security = doc.get("security")
    endpoints: list[EndpointDraft] = []
    for p, item in (doc.get("paths") or {}).items():
        item = resolve_refs(item, doc)
        shared_params = item.get("parameters", [])
        for method, op in item.items():
            if method.upper() not in HTTP_METHODS or not isinstance(op, dict):
                continue
            params = [*shared_params, *op.get("parameters", [])]
            responses = {
                str(status): {
                    "description": resp.get("description", ""),
                    "schema": _json_schema_of(resp.get("content")),
                }
                for status, resp in (op.get("responses") or {}).items()
            }
            body = op.get("requestBody") or {}
            security = op.get("security", global_security)
            endpoints.append(
                EndpointDraft(
                    method=method.upper(),
                    path=p,
                    operation_id=op.get("operationId"),
                    summary=op.get("summary") or op.get("description"),
                    tag=(op.get("tags") or [None])[0],
                    parameters=[
                        {
                            "name": prm.get("name"),
                            "in": prm.get("in"),
                            "required": bool(prm.get("required")),
                            "schema": prm.get("schema"),
                        }
                        for prm in params
                    ],
                    request_schema=_json_schema_of(body.get("content")),
                    response_schemas=responses,
                    requires_auth=bool(security),
                )
            )
    return LoadedSpec(
        name=info.get("title") or path.stem,
        spec_type="OPENAPI",
        version=str(info.get("version") or "0"),
        source_path=str(path),
        raw_content=raw,
        endpoints=endpoints,
        servers=[s.get("url") for s in doc.get("servers", []) if s.get("url")],
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
    for status, r in e.response_schemas.items():
        schema = f" -> {schema_brief(r['schema'])}" if r.get("schema") else ""
        lines.append(f"  {status}: {r.get('description', '')}{schema}")
    return "\n".join(lines)


def spec_digest(spec: LoadedSpec) -> str:
    head = f"# OpenAPI spec '{spec.name}' v{spec.version} ({spec.source_path})"
    return "\n".join([head, *(endpoint_digest(e) for e in spec.endpoints)])

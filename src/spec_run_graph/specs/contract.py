"""Contract facts derived from the OpenAPI schemas: flat lists the LLM can follow literally.

- required_body_fields: request-body fields a successful request must send ("address.city", "items[].sku");
- assertable_paths: response paths the contract guarantees (required at every level), in the IR check
  grammar ("response.body.data.id", "response.body.data[0].sku"). Optional fields are left out on purpose:
  the API may omit them, so tests must not assert or extract them.

The same facts drive the prompts (planner, healer) and the deterministic IR validation."""

from __future__ import annotations

from typing import Any

from spec_run_graph.domain.ir import _merged

MAX_DEPTH = 5
MAX_PATHS = 40


def _is_array(node: dict[str, Any]) -> bool:
    return node.get("type") == "array" or "items" in node


def required_body_fields(schema: Any, prefix: str = "", depth: int = 0) -> list[str]:
    node = _merged(schema)
    if node is None or depth > MAX_DEPTH:
        return []
    if _is_array(node):
        return required_body_fields(node.get("items"), f"{prefix}[]", depth + 1)
    out: list[str] = []
    props = node.get("properties") or {}
    for key in node.get("required") or []:
        path = f"{prefix}.{key}" if prefix else key
        out.append(path)
        if key in props:
            out += required_body_fields(props[key], path, depth + 1)
    return out


def assertable_paths(schema: Any, prefix: str = "response.body", depth: int = 0) -> list[str]:
    node = _merged(schema)
    if node is None or depth > MAX_DEPTH:
        return []
    if _is_array(node):
        return assertable_paths(node.get("items"), f"{prefix}[0]", depth + 1)
    props = node.get("properties") or {}
    if not props:
        return []
    required = node.get("required")
    keys = list(props) if required is None else [k for k in props if k in required]
    out: list[str] = []
    for key in keys:
        path = f"{prefix}.{key}"
        out.append(path)
        out += assertable_paths(props[key], path, depth + 1)
    return out[:MAX_PATHS]


def response_facts(responses: dict[str, Any]) -> dict[str, list[str]]:
    """{status: assertable paths} for documented responses with a JSON schema."""
    return {
        str(status): assertable_paths(r.get("schema"))
        for status, r in (responses or {}).items()
        if isinstance(r, dict) and r.get("schema")
    }


PAGE_SIZE_PARAMS = ("page_size", "pageSize", "per_page", "limit", "size")


def page_limits(endpoints: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """Paginated list endpoints: key -> (page-size query parameter, its documented maximum)."""
    out: dict[str, tuple[str, int]] = {}
    for key, e in endpoints.items():
        for p in getattr(e, "parameters", None) or []:
            schema = p.get("schema") or {}
            if p.get("in") == "query" and p.get("name") in PAGE_SIZE_PARAMS and isinstance(schema.get("maximum"), int):
                out[key] = (p["name"], schema["maximum"])
    return out

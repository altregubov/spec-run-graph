"""Deterministic repairs of LLM-written Bruno files (applied before validation, no LLM round-trip).

Each rule fixes a mistake observed on real runs:
- query parameters in `http.params` are ignored by bru CLI -> move them into the URL query string;
- an explicit `Authorization` header together with an `auth` block -> the auth block wins and the
  negative-auth test sends a valid token -> drop the auth block;
- `matches` / `notMatches` values wrapped in quotes -> the quotes become part of the regex -> strip them.
"""

from __future__ import annotations

import copy
from typing import Any
from urllib.parse import quote

import yaml

from spec_run_graph.engines.bruno.scaffold import dump_yaml

REGEX_OPERATORS = {"matches", "notMatches"}


def _encode(value: str) -> str:
    # keep Bruno templates ({{var}}) intact, percent-encode everything else
    out, rest = [], value
    while "{{" in rest and "}}" in rest[rest.index("{{") :]:
        start = rest.index("{{")
        end = rest.index("}}", start) + 2
        out.append(quote(rest[:start], safe="-_.~"))
        out.append(rest[start:end])
        rest = rest[end:]
    out.append(quote(rest, safe="-_.~"))
    return "".join(out)


def normalize_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (normalized copy, list of applied fixes)."""
    doc = copy.deepcopy(doc)
    fixes: list[str] = []
    http = doc.get("http") or {}

    params = http.get("params") or []
    query = [p for p in params if isinstance(p, dict) and p.get("type", "query") == "query" and not p.get("disabled")]
    if query:
        url = str(http.get("url", ""))
        sep = "&" if "?" in url else "?"
        pairs = [f"{_encode(str(p.get('name', '')))}={_encode(str(p.get('value', '')))}" for p in query]
        missing = [pair for pair in pairs if pair not in url]
        if missing:
            http["url"] = url + sep + "&".join(missing)
        rest = [p for p in params if p not in query]
        if rest:
            http["params"] = rest
        else:
            http.pop("params", None)
        fixes.append("query params moved into the url")

    headers = http.get("headers") or []
    if http.get("auth") and any(
        str(h.get("name", "")).lower() == "authorization" for h in headers if isinstance(h, dict)
    ):
        http.pop("auth")
        fixes.append("auth block removed (explicit Authorization header)")

    for a in (doc.get("runtime") or {}).get("assertions") or []:
        if not isinstance(a, dict) or a.get("operator") not in REGEX_OPERATORS:
            continue
        value = str(a.get("value", "")).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            a["value"] = value[1:-1]
            fixes.append("quotes stripped from regex assertion")
    if http:
        doc["http"] = http
    return doc, fixes


def normalize_text(content: str) -> tuple[str, list[str]]:
    """Normalize one YAML file; unparsable content is returned unchanged (validation reports it)."""
    try:
        doc = yaml.safe_load(content)
    except yaml.YAMLError:
        return content, []
    if not isinstance(doc, dict):
        return content, []
    fixed, fixes = normalize_doc(doc)
    return (dump_yaml(fixed), fixes) if fixes else (content, [])

"""Observe the SUT before planning: read-only GET samples of how real data looks.

The planner otherwise guesses values and formats (is "Apple" stored as "apple"? which categories exist?).
Only GET endpoints without path parameters are called, each once, as the first actor of the owning module
that can authenticate. Samples are trimmed and stored in Gel as planner context: they help choose realistic
values, while the OpenAPI contract alone decides expected statuses and required fields."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from spec_run_graph.domain.ir import REF_ENV_RE

MAX_ITEMS = 3  # list items kept per array
MAX_STR = 80
MAX_DEPTH = 6
MAX_CHARS = 1200  # per sample in the prompt


@dataclass
class Sample:
    endpoint: str
    actor: str
    http_status: int
    body: Any


def _resolve(value: Any, env: dict[str, str]) -> Any:
    """Substitute {{ $env.X }}; None when a referenced variable is not set."""
    if isinstance(value, dict):
        out = {k: _resolve(v, env) for k, v in value.items()}
        return None if any(v is None for v in out.values()) else out
    if not isinstance(value, str):
        return value
    missing = [n for n in REF_ENV_RE.findall(value) if n not in env]
    return None if missing else REF_ENV_RE.sub(lambda m: env[m.group(1)], value)


def _dig(body: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(body, dict):
            return None
        body = body.get(part)
    return body


def trim(body: Any, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return "..."
    if isinstance(body, dict):
        return {k: trim(v, depth + 1) for k, v in body.items()}
    if isinstance(body, list):
        items = [trim(v, depth + 1) for v in body[:MAX_ITEMS]]
        return items + ([f"... {len(body) - MAX_ITEMS} more"] if len(body) > MAX_ITEMS else [])
    if isinstance(body, str) and len(body) > MAX_STR:
        return body[:MAX_STR] + "..."
    return body


def only_paths(body: Any, paths: list[str]) -> Any:
    """Keep the parts of a body named by assertable paths ("response.body.data[0].id"); drop the rest."""
    wanted = {re.sub(r"\[\d+\]", "[]", p).removeprefix("response.body") for p in paths}

    def walk(node: Any, prefix: str) -> Any:
        if isinstance(node, list):
            return [walk(v, prefix + "[]") for v in node]
        if not isinstance(node, dict):
            return node
        return {
            k: walk(v, f"{prefix}.{k}")
            for k, v in node.items()
            if any(
                w == f"{prefix}.{k}" or w.startswith(f"{prefix}.{k}.") or w.startswith(f"{prefix}.{k}[") for w in wanted
            )
        }

    return walk(body, "")


def _split(key: str) -> tuple[str, str]:
    method, _, path = key.partition(" ")
    return method.upper(), path


async def credentials(
    http: httpx.AsyncClient, base: str, auth: dict[str, Any], env: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]] | None:
    """(headers, query params) authenticating as the actor, or None when that is not possible."""
    kind = auth.get("type", "bearer")
    if kind == "none":
        return {}, {}
    if kind == "login":
        body = _resolve(auth.get("login_body") or {}, env)
        if body is None or not auth.get("login_endpoint") or not auth.get("token_path"):
            return None
        method, path = _split(auth["login_endpoint"])
        try:
            res = await http.request(method, base + path, json=body)
            token = _dig(res.json(), auth["token_path"]) if res.is_success else None
        except (httpx.HTTPError, ValueError):
            return None
        return ({"Authorization": f"Bearer {token}"}, {}) if isinstance(token, str) else None
    if kind == "bearer":
        token = _resolve(auth.get("token"), env)
        return ({"Authorization": f"Bearer {token}"}, {}) if token else None
    if kind == "basic":
        user, pwd = _resolve(auth.get("username"), env), _resolve(auth.get("password"), env)
        if user is None or pwd is None:
            return None
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pwd}".encode()).decode()}, {}
    if kind == "apikey":
        value = _resolve(auth.get("value"), env)
        if not value or not auth.get("key"):
            return None
        pair = {auth["key"]: value}
        return (pair, {}) if auth.get("placement", "header") == "header" else ({}, pair)
    return None


async def observe(
    base_url: str,
    modules: list[Any],
    env: dict[str, str],
    guaranteed: dict[str, list[str]] | None = None,
    timeout: float = 10,
) -> list[Sample]:
    """One GET per parameterless GET endpoint, as the first actor of its module that can authenticate.
    `guaranteed` (endpoint -> assertable paths of the 200 response) trims each sample to those fields."""
    samples: list[Sample] = []
    seen: set[str] = set()
    creds: dict[str, tuple[dict[str, str], dict[str, str]] | None] = {}
    async with httpx.AsyncClient(timeout=timeout) as http:
        for m in modules:
            base = base_url.rstrip("/") + (m.base_path or "")
            for key in m.endpoints:
                method, path = _split(key)
                if method != "GET" or "{" in path or key in seen:
                    continue
                seen.add(key)
                for actor in m.actors or []:
                    if actor.name not in creds:
                        creds[actor.name] = await credentials(http, base, actor.auth.model_dump(), env)
                    if creds[actor.name] is None:
                        continue
                    headers, params = creds[actor.name]
                    try:
                        res = await http.get(base + path, headers=headers, params=params)
                        body = res.json()
                    except (httpx.HTTPError, ValueError):
                        break
                    if guaranteed and guaranteed.get(key) and res.is_success:
                        body = only_paths(body, guaranteed[key])
                    samples.append(Sample(key, actor.name, res.status_code, trim(body)))
                    break
    return samples


def observation_digest(rows: list[Any]) -> str:
    """Prompt text for stored observations (rows with endpoint, actor, http_status, sample)."""
    lines = []
    for r in rows:
        sample = r.sample if not isinstance(r.sample, str) else json.loads(r.sample)
        text = json.dumps(sample, default=str)
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "...(truncated)"
        lines.append(f"{r.endpoint} as {r.actor} -> {r.http_status}: {text}")
    return "\n".join(lines)

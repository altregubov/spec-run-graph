"""Collection scaffolding written by code: opencollection.yml, environments/local.yml, folder.yml."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from spec_run_graph.config import get_settings

COLLECTION_NAME = "SpecRunGraph API tests"
ENV_NAME = "local"


class _Dumper(yaml.SafeDumper):
    pass


def _str_repr(dumper: yaml.SafeDumper, value: str):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_repr)


def dump_yaml(data: Any) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=1000)


def fixture_var(code: str, name: str) -> str:
    return f"fx_{code}_{name}"


def fixture_template(code: str, name: str, value: Any) -> str:
    """How a $fixture reference is written in Bruno files."""
    if isinstance(value, dict) and "env" in value:
        return "{{process.env." + value["env"] + "}}"
    return "{{" + fixture_var(code, name) + "}}"


def env_template(text: str) -> str:
    """'{{ $env.CUSTOMER_TOKEN }}' -> '{{process.env.CUSTOMER_TOKEN}}' (other text unchanged)."""
    return re.sub(r"\{\{\s*\$env\.([A-Z][A-Z0-9_]*)\s*\}\}", r"{{process.env.\1}}", text)


def auth_var(actor: str) -> str:
    """Runtime variable that holds the bearer token of a login actor: auth_<actor>."""
    return "auth_" + re.sub(r"\W", "_", actor)


def actor_auth(auth: dict[str, Any]) -> dict[str, Any]:
    """The Bruno auth block for an actor (login actors use the token fetched by the collection script)."""
    if auth.get("type") == "login":
        return {"type": "bearer", "token": "{{" + auth_var(auth["_actor"]) + "}}"}
    return {k: env_template(v) if isinstance(v, str) else v for k, v in auth.items() if not k.startswith("_")}


def _js_value(value: Any) -> str:
    if isinstance(value, str):
        m = re.fullmatch(r"\{\{\s*\$env\.([A-Z][A-Z0-9_]*)\s*\}\}", value.strip())
        if m:
            return f'bru.getProcessEnv("{m.group(1)}")'
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {_js_value(v)}" for k, v in value.items()) + "}"
    return json.dumps(value)


def login_script(modules: list[dict[str, Any]]) -> str | None:
    """Collection-level before-request script: log every login actor in once per run, cache the token."""
    logins: dict[str, str] = {}
    for m in sorted(modules, key=lambda x: x["code"]):
        for name, auth in sorted(m["actors"].items()):
            if auth.get("type") != "login" or not auth.get("login_endpoint"):
                continue
            method, _, path = auth["login_endpoint"].partition(" ")
            logins[auth_var(name)] = (
                f"{{ method: {json.dumps(method.upper())}, path: {json.dumps((m.get('base_path') or '') + path.strip())}, "
                f"body: {_js_value(auth.get('login_body') or {})}, tokenPath: {json.dumps(auth.get('token_path') or 'token')} }}"
            )
    if not logins:
        return None
    entries = ",\n".join(f"  {json.dumps(var)}: {spec}" for var, spec in logins.items())
    return f"""// SpecRunGraph: log every test actor in once per run (accounts from the specification).
const logins = {{
{entries}
}};
for (const [variable, l] of Object.entries(logins)) {{
  if (bru.getVar(variable)) continue;
  try {{
    const res = await bru.sendRequest({{
      method: l.method, url: bru.getEnvVar("baseUrl") + l.path,
      headers: {{ "Content-Type": "application/json" }}, data: l.body,
    }});
    const token = l.tokenPath.split(".").reduce((o, k) => (o == null ? o : o[k]), res.data);
    bru.setVar(variable, token || "LOGIN_FAILED");
    if (!token) console.error(`[srg] login for ${{variable}} returned no token (HTTP ${{res.status}})`);
  }} catch (e) {{
    bru.setVar(variable, "LOGIN_FAILED");
    console.error(`[srg] login for ${{variable}} failed: ${{e.message}}`);
  }}
}}"""


def write_scaffold(root: Path, modules: list[dict[str, Any]], base_url: str | None = None) -> list[Path]:
    """modules: module_context() dicts. Returns written paths."""
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    collection: dict[str, Any] = {"opencollection": "1.0.0", "info": {"name": COLLECTION_NAME, "version": "1.0.0"}}
    script = login_script(modules)
    if script:
        collection["request"] = {"scripts": [{"type": "before-request", "code": script}]}
    (root / "opencollection.yml").write_text(dump_yaml(collection))
    written.append(root / "opencollection.yml")

    variables = [{"name": "baseUrl", "value": base_url or get_settings().sut_base_url}]
    for m in sorted(modules, key=lambda x: x["code"]):
        for name, value in sorted(m["fixtures"].items()):
            if isinstance(value, dict) and "env" in value:
                continue  # read from process.env directly
            variables.append(
                {
                    "name": fixture_var(m["code"], name),
                    # yaml.safe_dump of a scalar appends "\n..." - JSON is exact for numbers, lists and objects
                    "value": value if isinstance(value, str) else json.dumps(value),
                }
            )
    env_dir = root / "environments"
    env_dir.mkdir(exist_ok=True)
    (env_dir / f"{ENV_NAME}.yml").write_text(dump_yaml({"name": ENV_NAME, "variables": variables}))
    written.append(env_dir / f"{ENV_NAME}.yml")

    for seq, m in enumerate(sorted(modules, key=lambda x: x["folder"]), start=1):
        folder = root / m["folder"]
        folder.mkdir(exist_ok=True)
        data: dict[str, Any] = {"info": {"name": m["folder"], "type": "folder", "seq": seq}}
        headers = [{"name": k, "value": env_template(str(v))} for k, v in (m["default_headers"] or {}).items()]
        if headers:
            data["request"] = {"headers": headers}
        docs = [f"# {m['domain']} ({m['code']})"]
        docs += [f"- {p}" for p in m.get("preconditions") or []]
        data["docs"] = "\n".join(docs)
        (folder / "folder.yml").write_text(dump_yaml(data))
        written.append(folder / "folder.yml")
    return written

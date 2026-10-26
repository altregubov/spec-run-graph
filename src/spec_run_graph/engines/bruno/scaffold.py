"""Collection scaffolding written by code: opencollection.yml, environments/local.yml, folder.yml."""

from __future__ import annotations

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


def write_scaffold(root: Path, modules: list[dict[str, Any]], base_url: str | None = None) -> list[Path]:
    """modules: module_context() dicts. Returns written paths."""
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    collection = {"opencollection": "1.0.0", "info": {"name": COLLECTION_NAME, "version": "1.0.0"}}
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
                    "value": value
                    if isinstance(value, str)
                    else yaml.safe_dump(value, default_flow_style=True).strip(),
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

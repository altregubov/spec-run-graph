"""Deterministic guard rails for LLM-written Bruno OpenCollection files."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator

from spec_run_graph.domain.faker import BRUNO_DYNAMIC
from spec_run_graph.domain.ir import PARAM_TEMPLATE_RE, TestCaseIR, http_steps
from spec_run_graph.engines.bruno.naming import PlannedFile, var_prefix
from spec_run_graph.engines.bruno.scaffold import fixture_var

SCHEMA_PATH = Path(__file__).with_name("opencollection.schema.json")
TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}")
IR_SYNTAX_RE = re.compile(r"\{\{\s*\$(faker|fixture|env|extract|param|var)\.")
STATIC_DYNAMIC = {"guid", "timestamp", "isoTimestamp"} | BRUNO_DYNAMIC


@lru_cache
def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


@lru_cache
def _validator(defname: str | None) -> Draft7Validator:
    schema = _schema()
    if defname is None:
        return Draft7Validator(schema)
    return Draft7Validator({"$ref": f"#/$defs/{defname}", "$defs": schema["$defs"]})


def schema_errors(data: Any, defname: str | None) -> list[str]:
    errors = sorted(_validator(defname).iter_errors(data), key=lambda e: list(e.absolute_path))
    out = []
    for e in errors[:6]:
        where = "/".join(str(p) for p in e.absolute_path) or "<root>"
        out.append(f"schema: {where}: {e.message[:300]}")
    return out


def expected_status_for(case: TestCaseIR, step_seq: int, param_index: int) -> int | None:
    step, _ = http_steps(case)[step_seq - 1]
    es = step.expected_status
    if es is None:
        return None
    if isinstance(es, int):
        return es
    if es.strip().isdigit():
        return int(es)
    m = PARAM_TEMPLATE_RE.match(es.strip())
    if m and case.parameters:
        value = case.parameters[param_index].get(m.group(1))
        return int(value) if value is not None else None
    return None


def _scripts(doc: dict) -> list[str]:
    return [s.get("code", "") for s in (doc.get("runtime") or {}).get("scripts") or [] if isinstance(s, dict)]


def check_js_syntax(snippets: list[str]) -> list[str]:
    """`node --check` every script, wrapped in an async function (Bruno allows top-level await)."""
    node = shutil.which("node")
    if not node or not snippets:
        return []
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, code in enumerate(snippets):
            path = Path(tmp) / f"s{i}.js"
            path.write_text(f"(async function () {{\n{code}\n}});\n")
            proc = subprocess.run([node, "--check", str(path)], capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                msg = next((ln for ln in proc.stderr.splitlines() if "Error" in ln), proc.stderr.strip()[:300])
                errors.append(f"JavaScript syntax error in script #{i + 1}: {msg}")
    return errors


def validate_files(
    case: TestCaseIR,
    module: dict[str, Any],
    planned: list[PlannedFile],
    files: dict[str, str],
    secrets: set[str],
) -> tuple[list[str], dict[str, dict]]:
    """Return (errors, parsed docs by filename)."""
    errors: list[str] = []
    parsed: dict[str, dict] = {}
    wanted = {p.filename: p for p in planned}
    missing = sorted(set(wanted) - set(files))
    extra = sorted(set(files) - set(wanted))
    if missing:
        errors.append(f"missing files: {missing}")
    if extra:
        errors.append(f"unexpected files (use exactly the planned names): {extra}")

    prefix = var_prefix(case.id or "")
    fixture_vars = {fixture_var(module["code"], n) for n in module["fixtures"]}
    steps = http_steps(case)
    js: list[str] = []
    for name, content in files.items():
        if name not in wanted:
            continue
        p = wanted[name]
        where = name
        try:
            doc = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            errors.append(f"{where}: invalid YAML: {str(exc)[:300]}")
            continue
        if not isinstance(doc, dict):
            errors.append(f"{where}: top level must be a mapping")
            continue
        parsed[name] = doc
        errors += [f"{where}: {e}" for e in schema_errors(doc, "HttpRequest")]
        info, http = doc.get("info") or {}, doc.get("http") or {}
        if info.get("type") != "http":
            errors.append(f"{where}: info.type must be 'http'")
        if not str(info.get("name", "")).startswith(p.display_name.split(" ")[0]):
            errors.append(f"{where}: info.name must be {p.display_name!r}")
        step, _ = steps[p.step_seq - 1]
        method = (step.endpoint or "").split(" ", 1)[0].upper()
        if str(http.get("method", "")).upper() != method:
            errors.append(f"{where}: http.method must be {method}")
        if not str(http.get("url", "")).startswith("{{baseUrl}}"):
            errors.append(f"{where}: http.url must start with {{{{baseUrl}}}}")

        if IR_SYNTAX_RE.search(content):
            errors.append(f"{where}: IR templates like '{{{{ $extract.x }}}}' must be translated to Bruno syntax")
        for raw in TEMPLATE_RE.findall(content):
            t = raw.strip()
            ok = (
                t == "baseUrl"
                or re.fullmatch(r"process\.env\.[A-Z][A-Z0-9_]*", t)
                or t in fixture_vars
                or (t.startswith(prefix) and re.fullmatch(r"[A-Za-z0-9_]+", t))
                or (t.startswith("$") and t[1:] in STATIC_DYNAMIC)
            )
            if not ok:
                errors.append(
                    f"{where}: template '{{{{{raw}}}}}' is not allowed (use baseUrl, process.env.X, "
                    f"{sorted(fixture_vars) or 'fixture vars'}, {prefix}<name> or a Bruno $random* variable)"
                )
        for secret in secrets:
            if secret and secret in content:
                errors.append(f"{where}: contains a literal secret value; use {{{{process.env.X}}}}")
        runtime = doc.get("runtime") or {}
        for action in runtime.get("actions") or []:
            var = ((action or {}).get("variable") or {}).get("name", "")
            if not var.startswith(prefix):
                errors.append(f"{where}: runtime variable {var!r} must start with {prefix}")
        js += _scripts(doc)
        if not p.is_cleanup:
            status = expected_status_for(case, p.step_seq, p.param_index)
            asserts = runtime.get("assertions") or []
            has_status = any(
                a.get("expression") == "res.status" and a.get("operator") == "eq" and str(a.get("value")) == str(status)
                for a in asserts
                if isinstance(a, dict)
            )
            if status is not None and not has_status:
                errors.append(f"{where}: needs assertion {{expression: res.status, operator: eq, value: '{status}'}}")
    errors += check_js_syntax(js)
    return errors, parsed

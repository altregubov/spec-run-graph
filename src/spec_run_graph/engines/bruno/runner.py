"""Run a Bruno collection with `bru run` and map per-file results back to test cases."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spec_run_graph.engines.bruno.naming import parse_filename

MANIFEST_NAME = ".srg-bruno-manifest.json"


@dataclass
class BruResult:
    returncode: int | None
    stdout: str
    stderr: str
    results: list[dict[str, Any]]  # flattened per-file results
    results_json: str
    results_xml: str
    timed_out: bool = False


async def bru_version(bru_bin: str) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            bru_bin, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
        return out.decode().strip() or None
    except (OSError, TimeoutError):
        return None


async def run_bru(
    bru_bin: str,
    root: Path,
    paths: list[str],
    env_name: str,
    env_vars: dict[str, str],
    process_env: dict[str, str],
    out_dir: Path,
    timeout_sec: int,
) -> BruResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, xml_path = out_dir / "results.json", out_dir / "results.xml"
    for p in (json_path, xml_path):
        p.unlink(missing_ok=True)
    cmd = [bru_bin, "run", *paths, "--env", env_name]
    for k, v in env_vars.items():
        cmd += ["--env-var", f"{k}={v}"]
    cmd += ["--reporter-json", str(json_path.resolve()), "--reporter-junit", str(xml_path.resolve())]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=root,
        env={**os.environ, **process_env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own process group, so a timeout kills node children too
    )
    timed_out = False
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
    except TimeoutError:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = await proc.communicate()
    results_json = json_path.read_text() if json_path.exists() else "[]"
    return BruResult(
        returncode=proc.returncode,
        stdout=out.decode(errors="replace"),
        stderr=err.decode(errors="replace"),
        results=flatten_results(json.loads(results_json or "[]")),
        results_json=results_json,
        results_xml=xml_path.read_text() if xml_path.exists() else "",
        timed_out=timed_out,
    )


def flatten_results(report: Any) -> list[dict[str, Any]]:
    """bru JSON report = [ {iterationIndex, results: [...]}, ... ]  ->  [result, ...]"""
    if isinstance(report, dict):
        report = [report]
    out = []
    for iteration in report or []:
        out.extend(iteration.get("results", []) if isinstance(iteration, dict) else [])
    return out


@dataclass
class CaseOutcome:
    test_id: str
    param_index: int
    status: str = "SKIPPED"
    duration_ms: int = 0
    http_status: int | None = None
    failed: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    raw: list[dict[str, Any]] = field(default_factory=list)


def _failures(result: dict[str, Any]) -> list[dict[str, Any]]:
    fails = []
    for a in result.get("assertionResults") or []:
        if a.get("status") != "pass":
            fails.append({"assertion": f"{a.get('lhsExpr')} {a.get('rhsExpr')}", "error": a.get("error")})
    for kind in ("testResults", "preRequestTestResults", "postResponseTestResults"):
        for t in result.get(kind) or []:
            if t.get("status") != "pass":
                fails.append({"test": t.get("description"), "error": t.get("error")})
    return fails


def aggregate(
    results: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    expected: dict[tuple[str, int], list[str]],
    timed_out: bool = False,
) -> dict[tuple[str, int], CaseOutcome]:
    """Fold per-file results into one outcome per (test_id, param_index).

    expected: (test_id, param_index) -> planned filenames (relative to the collection root).
    """
    by_file = {}
    for r in results:
        path = r.get("path") or (r.get("test") or {}).get("filename") or ""
        by_file[path.replace("\\", "/")] = r
    outcomes: dict[tuple[str, int], CaseOutcome] = {}
    for key, files in expected.items():
        oc = CaseOutcome(test_id=key[0], param_index=key[1])
        states = []
        for f in files:
            meta = manifest.get(f) or {}
            if not meta:
                parsed = parse_filename(f)
                meta = {"is_cleanup": False} if parsed else {}
            r = by_file.get(f)
            if r is None:
                states.append("missing")
                continue
            oc.raw.append(r)
            oc.duration_ms += int(round((r.get("runDuration") or 0) * 1000)) or int(
                (r.get("response") or {}).get("responseTime") or 0
            )
            if meta.get("is_cleanup"):
                continue
            resp = r.get("response") or {}
            if isinstance(resp.get("status"), int):
                oc.http_status = resp["status"]
            if r.get("error") and not resp.get("status"):
                states.append("error")
                oc.error_message = str(r.get("error"))[:2000]
            elif r.get("status") != "pass" or _failures(r):
                # bru's per-result `status` describes the request only; assertions/tests are checked separately
                states.append("fail")
                oc.failed += [{"file": f, **x} for x in _failures(r)] or [{"file": f, "error": r.get("error")}]
                if r.get("error") and not oc.error_message:
                    oc.error_message = str(r["error"])[:2000]
            else:
                states.append("pass")
        if "error" in states:
            oc.status = "ERROR"
        elif "fail" in states:
            oc.status = "FAILED"
        elif "missing" in states:
            oc.status = "TIMEOUT" if timed_out else ("SKIPPED" if "pass" not in states else "ERROR")
            if oc.status == "ERROR":
                oc.error_message = "some step files produced no result"
        elif states:
            oc.status = "PASSED"
        outcomes[key] = oc
    return outcomes

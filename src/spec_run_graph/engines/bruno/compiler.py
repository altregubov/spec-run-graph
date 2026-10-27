"""Deterministic IR -> Bruno OpenCollection compiler (no LLM).

Requests, auth, variables, extracts and checks translate one-to-one; every check becomes a `test()` with a
readable failure message ("expected response.body.data.status == 'PAID', got 'DRAFT'"), extracts become
`bru.setVar` calls in an after-response script. Features the compiler does not cover raise `Unsupported`
and the case falls back to the LLM synthesizer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from spec_run_graph.domain.expr import BODY_SEGMENT, TEMPLATE_RE, parse_check, parse_literal
from spec_run_graph.domain.faker import bruno_dynamic
from spec_run_graph.domain.ir import ANONYMOUS, REF_RE, TestCaseIR, _merged, http_steps, resolve_endpoint
from spec_run_graph.engines.bruno.naming import PlannedFile, var_prefix
from spec_run_graph.engines.bruno.normalize import _encode
from spec_run_graph.engines.bruno.scaffold import actor_auth, fixture_template
from spec_run_graph.engines.bruno.validate import expected_statuses_for, status_assertion

COMPILER_VERSION = "5"
_SEGMENT_RE = re.compile(BODY_SEGMENT)
_FILTER_RE = re.compile(r"^\[([A-Za-z_][A-Za-z0-9_]*)=(.+)\]$")

# Shared JS helpers, inlined into every script that needs them (Bruno scripts cannot share code).
HELPERS = """\
var __srg = {
  walk(o, segs) {
    for (const s of segs) {
      if (o === undefined || o === null) return undefined;
      if (s !== null && typeof s === "object") {
        o = Array.isArray(o) ? o.find((x) => x !== null && x !== undefined && __srg.same(x[s.f], s.v)) : undefined;
      } else {
        o = o[s];
      }
    }
    return o;
  },
  same(a, b) {
    if (a === b) return true;
    if (a === null || b === null || a === undefined || b === undefined) return false;
    if (typeof a === "object" || typeof b === "object") return JSON.stringify(a) === JSON.stringify(b);
    return String(a) === String(b);
  },
  show(v) {
    return v === undefined ? "undefined" : JSON.stringify(v);
  },
  check(a, op, e) {
    const num = (v) => (typeof v === "number" ? v : v !== null && v !== "" && !isNaN(Number(v)) ? Number(v) : NaN);
    let ok;
    switch (op) {
      case "==": ok = __srg.same(a, e); break;
      case "!=": ok = !__srg.same(a, e); break;
      case ">": ok = num(a) > num(e); break;
      case ">=": ok = num(a) >= num(e); break;
      case "<": ok = num(a) < num(e); break;
      case "<=": ok = num(a) <= num(e); break;
      case "in": ok = Array.isArray(e) && e.some((x) => __srg.same(a, x)); break;
      case "contains":
        ok = Array.isArray(a) ? a.some((x) => __srg.same(x, e)) : typeof a === "string" && a.includes(String(e));
        break;
      case "matches":
        ok = a !== undefined && a !== null && (e instanceof RegExp ? e : new RegExp(e)).test(String(a));
        break;
      case "exists": ok = a !== undefined; break;
      case "not exists": ok = a === undefined; break;
      default: throw new Error("unknown operator " + op);
    }
    if (!ok) throw new Error("expected " + op + " " + (op.endsWith("exists") ? "" : __srg.show(e)) + ", got " + __srg.show(a));
  },
};"""


class Unsupported(Exception):
    """The case uses a feature the deterministic compiler does not translate."""


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value)
    return str(value)


class _Ctx:
    def __init__(self, case: TestCaseIR, module: dict[str, Any], row: dict[str, Any]) -> None:
        self.case, self.module, self.row = case, module, row
        self.prefix = var_prefix(case.id or "")

    # ---- request text: IR templates -> Bruno {{...}} templates
    def tpl(self, text: str, depth: int = 0) -> str:
        def repl(m: re.Match) -> str:
            ns, name = m.group(1), m.group(2)
            if ns == "env":
                return "{{process.env." + name + "}}"
            if ns == "fixture":
                value = self.module["fixtures"].get(name)
                if isinstance(value, dict) and "env" in value:
                    return fixture_template(self.module["code"], name, value)
                return _scalar_text(value)  # literal test data is inlined (typed in JSON bodies)
            if ns == "extract":
                return "{{" + self.prefix + name + "}}"
            if ns == "var":
                return "{{" + self.prefix + "var_" + name + "}}"
            if ns == "faker":
                dyn = bruno_dynamic(name)
                if dyn is None:
                    raise Unsupported(f"faker type {name!r}")
                return "{{$" + dyn + "}}"
            if ns == "param":
                value = _scalar_text(self.row.get(name))
                return self.tpl(value, depth + 1) if depth < 3 else value
            raise Unsupported(f"template namespace ${ns}")

        return REF_RE.sub(repl, text)

    def label(self, text: str) -> str:
        """Test names: parameter values filled in, other templates shown as plain $ns.name."""

        def repl(m: re.Match) -> str:
            if m.group(1) == "param":
                return _scalar_text(self.row.get(m.group(2)))
            return f"${m.group(1)}.{m.group(2)}"

        return REF_RE.sub(repl, text)

    # ---- check literals / filter values: IR templates -> JS expressions
    def js_value(self, value: Any) -> str:
        if isinstance(value, str):
            m = REF_RE.fullmatch(value.strip())
            if m and TEMPLATE_RE.match(value.strip()):
                ns, name = m.group(1), m.group(2)
                if ns == "extract":
                    return f"bru.getVar({json.dumps(self.prefix + name)})"
                if ns == "var":
                    return f"bru.getVar({json.dumps(self.prefix + 'var_' + name)})"
                if ns == "param":
                    return self.js_value(self.row.get(name))
                if ns == "env":
                    return f"bru.getProcessEnv({json.dumps(name)})"
                if ns == "fixture":
                    fx = self.module["fixtures"].get(name)
                    if isinstance(fx, dict) and "env" in fx:
                        return f"bru.getProcessEnv({json.dumps(fx['env'])})"
                    return json.dumps(fx)
            if REF_RE.search(value):
                return f"bru.interpolate({json.dumps(self.tpl(value))})"
            return json.dumps(value)
        if isinstance(value, list):
            return "[" + ", ".join(self.js_value(v) for v in value) + "]"
        return json.dumps(value)

    def path_js(self, path: str) -> str:
        """IR path -> JS expression reading the response."""
        if path == "response.status":
            return "res.getStatus()"
        if path.startswith("response.headers."):
            return f"res.getHeader({json.dumps(path.removeprefix('response.headers.').lower())})"
        segs: list[str] = []
        for m in _SEGMENT_RE.finditer(path.removeprefix("response.body")):
            seg = m.group(0)
            if seg.startswith("."):
                segs.append(json.dumps(seg[1:]))
            elif (f := _FILTER_RE.match(seg)) is not None:
                literal, _ = parse_literal(f.group(2))
                segs.append(f"{{ f: {json.dumps(f.group(1))}, v: {self.js_value(literal)} }}")
            else:
                segs.append(seg[1:-1])
        return f"__srg.walk(res.getBody(), [{', '.join(segs)}])"


def _json_body(ctx: _Ctx, payload: Any, schema: Any) -> str:
    """Payload -> JSON text with Bruno templates. A value that is exactly one $extract/$var template is
    written unquoted when the request schema says it is a number/boolean (ids captured as numbers)."""
    raw: dict[str, str] = {}

    def conv(value: Any, node: Any) -> Any:
        node_m = _merged(node) or {}
        if isinstance(value, dict):
            props = node_m.get("properties") or {}
            return {k: conv(v, props.get(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [conv(v, node_m.get("items")) for v in value]
        if isinstance(value, str):
            m = REF_RE.fullmatch(value.strip())
            if m and m.group(1) == "param":
                return conv(ctx.row.get(m.group(2)), node)
            if m and m.group(1) == "fixture":
                fx = ctx.module["fixtures"].get(m.group(2))
                if not (isinstance(fx, dict) and "env" in fx):
                    return conv(fx, node)
            if m and m.group(1) in ("extract", "var") and node_m.get("type") in ("integer", "number", "boolean"):
                marker = f"@@srg_raw_{len(raw)}@@"
                raw[marker] = ctx.tpl(value.strip())
                return marker
            return ctx.tpl(value)
        return value

    text = json.dumps(conv(payload, schema), ensure_ascii=False)
    for marker, template in raw.items():
        text = text.replace(json.dumps(marker), template)
    return text


def _url(ctx: _Ctx, step: Any) -> str:
    path = step.endpoint.split(None, 1)[1].strip()
    for name, value in step.request.path_params.items():
        rendered = ctx.tpl(_scalar_text(value))
        path = path.replace("{" + name + "}", rendered if "{{" in rendered else _encode(rendered))
    path = ctx.tpl(path)
    if re.search(r"(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})", path):
        raise Unsupported(f"unfilled path parameter in {path!r}")
    pairs = []
    for key, value in step.request.query.items():
        for v in value if isinstance(value, list) else [value]:
            pairs.append(f"{_encode(key)}={_encode(ctx.tpl(_scalar_text(v)))}")
    url = "{{baseUrl}}" + (ctx.module.get("base_path") or "") + path
    return url + ("?" + "&".join(pairs) if pairs else "")


def compile_case(
    case: TestCaseIR, module: dict[str, Any], planned: list[PlannedFile], endpoints: dict[str, Any]
) -> dict[str, dict]:
    """Return {filename: OpenCollection doc} for every planned file, or raise Unsupported."""
    steps = http_steps(case)
    if any(s.poll for s, _ in steps):
        raise Unsupported("poll steps")
    paths = {k: k.split(" ", 1)[1] for k in endpoints}
    out: dict[str, dict] = {}
    for p in planned:
        step, is_cleanup = steps[p.step_seq - 1]
        row = case.parameters[p.param_index] if case.parameters else {}
        ctx = _Ctx(case, module, row)
        key = resolve_endpoint(step.endpoint or "", paths)
        request_schema = (endpoints.get(key or "") or {}).get("request_schema")

        http: dict[str, Any] = {"method": step.endpoint.split(None, 1)[0].upper(), "url": _url(ctx, step)}
        headers = [{"name": k, "value": ctx.tpl(_scalar_text(v))} for k, v in step.request.headers.items()]
        if headers:
            http["headers"] = headers
        if step.request.payload is not None:
            http["body"] = {"type": "json", "data": _json_body(ctx, step.request.payload, request_schema)}
        actor = step.actor or case.actor
        explicit_auth = any(h["name"].lower() == "authorization" for h in headers)
        # anonymous: no auth block - bru 4.3 sends no credentials then (not inherited from the folder either)
        if actor != ANONYMOUS and actor and not explicit_auth and actor in module["actors"]:
            http["auth"] = actor_auth({**module["actors"][actor], "_actor": actor})

        before: list[str] = []
        if p.file_step == 1:
            for name, value in case.variables.items():
                target = json.dumps(ctx.prefix + "var_" + name)
                if isinstance(value, str):
                    before.append(f"bru.setVar({target}, bru.interpolate({json.dumps(ctx.tpl(value))}));")
                else:
                    before.append(f"bru.setVar({target}, {json.dumps(value)});")
        delay, i = 0, p.step_seq - 2
        while i >= 0 and steps[i][0].type == "delay":
            delay += steps[i][0].delay_ms or 0
            i -= 1
        if delay:
            before.append(f"await bru.sleep({delay});")

        after = [
            f"bru.setVar({json.dumps(ctx.prefix + var)}, {ctx.path_js(path)});" for var, path in step.extract.items()
        ]
        tests: list[str] = []
        runtime: dict[str, Any] = {}
        if not is_cleanup:
            statuses = expected_statuses_for(case, p.step_seq, p.param_index)
            if statuses:
                runtime["assertions"] = [status_assertion(statuses)]
            for a in step.assertions:
                chk = parse_check(a.check)
                expected = "undefined" if chk.value is None and chk.op.endswith("exists") else ctx.js_value(chk.value)
                if chk.op == "matches" and isinstance(chk.value, str) and not chk.is_template:
                    flags = re.match(r"^\(\?([a-z]+)\)", chk.value)  # JS has no inline flags like (?i)
                    pattern = chk.value[flags.end() :] if flags else chk.value
                    expected = f"new RegExp({json.dumps(pattern)}, {json.dumps(flags.group(1) if flags else '')})"
                label = a.message or a.check
                tests.append(
                    f"test({json.dumps(ctx.label(label))}, function () {{\n"
                    f"  __srg.check({ctx.path_js(chk.path)}, {json.dumps(chk.op)}, {expected});\n}});"
                )
        scripts = []
        if before:
            scripts.append({"type": "before-request", "code": "\n".join(before)})
        if after:
            scripts.append({"type": "after-response", "code": HELPERS + "\n" + "\n".join(after)})
        if tests:
            scripts.append({"type": "tests", "code": HELPERS + "\n" + "\n".join(tests)})
        if scripts:
            runtime["scripts"] = scripts
        doc: dict[str, Any] = {"info": {"name": ctx.label(p.display_name), "type": "http"}, "http": http}
        if case.tags:
            doc["info"]["tags"] = list(case.tags)
        if runtime:
            doc["runtime"] = runtime
        if step.timeout_ms:
            doc["settings"] = {"timeout": step.timeout_ms}
        out[p.filename] = doc
    return out

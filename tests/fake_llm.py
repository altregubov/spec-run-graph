"""Test doubles for the LLM: FunctionModels that answer like a well-behaved model.

`synthesizer_model()` reads the synthesizer prompt (IR + planned files) and returns Bruno files, so the
build -> test chain can be exercised without an API key. Production always uses the .env model.
"""

from __future__ import annotations

import json
import re
from typing import Any

import yaml
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from spec_run_graph.domain.expr import parse_check
from spec_run_graph.domain.faker import bruno_dynamic
from spec_run_graph.domain.ir import TestCaseIR, http_steps

OPS = {"==": "eq", "!=": "neq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}


def _prompt(messages: list[ModelMessage]) -> str:
    for m in messages:
        for p in getattr(m, "parts", []):
            if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                return p.content
    return ""


def _section(prompt: str, head: str) -> str:
    start = prompt.index(head) + len(head)
    end = prompt.find("\n\n", start)
    return prompt[start:] if end == -1 else prompt[start:end]


def _respond(info: AgentInfo, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])


def tool_model(payload: dict[str, Any] | list[dict[str, Any]]) -> FunctionModel:
    """A model that answers with a fixed output (or one per call, if a list is given)."""
    answers = payload if isinstance(payload, list) else [payload]
    calls = {"n": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answer = answers[min(calls["n"], len(answers) - 1)]
        calls["n"] += 1
        return _respond(info, answer)

    return FunctionModel(fn)


class _Renderer:
    def __init__(self, prompt: str):
        self.prefix = re.search(r"Runtime variable PREFIX for this case: (\S+)", prompt).group(1)
        m = re.search(r"base_path='([^']*)'", prompt)
        self.base_path = m.group(1) if m else ""
        self.fixtures = json.loads(_section(prompt, "Fixture templates: "))
        self.actors = json.loads(_section(prompt, "Actors (Bruno auth blocks): "))
        ir_text = prompt.split("Test case (IR):\n", 1)[1].split("\n\nPlanned files, in run order:\n", 1)[0]
        self.case = TestCaseIR.model_validate(yaml.safe_load(ir_text))
        plan_text = prompt.split("Planned files, in run order:\n", 1)[1]
        self.plan = json.JSONDecoder().raw_decode(plan_text)[0]

    def tpl(self, text: str, row: dict[str, Any] | None) -> Any:
        whole = re.fullmatch(r"\{\{\s*\$param\.(\w+)\s*\}\}", text.strip())
        if whole and row is not None:
            return row[whole.group(1)]

        def sub(m: re.Match) -> str:
            ns, name = m.group(1), m.group(2)
            if ns == "env":
                return "{{process.env." + name + "}}"
            if ns == "fixture":
                return self.fixtures["{{ $fixture." + name + " }}"]
            if ns == "extract":
                return "{{" + self.prefix + name + "}}"
            if ns == "var":
                return "{{" + self.prefix + "var_" + name + "}}"
            if ns == "param":
                return str(row[name]) if row else ""
            if ns == "faker":
                return "{{$" + bruno_dynamic(name) + "}}"
            return m.group(0)

        return re.sub(r"\{\{\s*\$([a-z]+)\.([A-Za-z0-9_]+)\s*\}\}", sub, text)

    def deep(self, obj: Any, row: dict[str, Any] | None) -> Any:
        if isinstance(obj, str):
            return self.tpl(obj, row)
        if isinstance(obj, dict):
            return {k: self.deep(v, row) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.deep(v, row) for v in obj]
        return obj

    @staticmethod
    def literal_params(check: str, row: dict[str, Any] | None) -> str:
        def sub(m: re.Match) -> str:
            v = (row or {})[m.group(1)]
            return f"'{v}'" if isinstance(v, str) else json.dumps(v)

        return re.sub(r"\{\{\s*\$param\.(\w+)\s*\}\}", sub, check)

    @staticmethod
    def bexpr(path: str) -> str:
        if path == "response.status":
            return "res.status"
        if path.startswith("response.headers."):
            return f"res.headers['{path.removeprefix('response.headers.').lower()}']"
        return "res." + path.removeprefix("response.")

    def render(self, entry: dict[str, Any]) -> str:
        steps = http_steps(self.case)
        seq = int(re.search(r"#(\d+)", entry["implements"]).group(1))
        step, is_cleanup = steps[seq - 1]
        row = entry.get("parameter_row")
        method, _, path = step.endpoint.partition(" ")
        path = self.deep(path.strip(), row)
        for k, v in step.request.path_params.items():
            path = path.replace("{" + k + "}", str(self.deep(v, row)))
        http: dict[str, Any] = {"method": method.upper(), "url": "{{baseUrl}}" + self.base_path + path}
        headers = [{"name": k, "value": str(self.deep(v, row))} for k, v in step.request.headers.items()]
        if step.request.payload is not None:
            headers.insert(0, {"name": "Content-Type", "value": "application/json"})
            http["body"] = {"type": "json", "data": json.dumps(self.deep(step.request.payload, row))}
        if headers:
            http["headers"] = headers
        if step.request.query:
            http["params"] = [
                {"name": k, "value": str(self.deep(v, row)), "type": "query"} for k, v in step.request.query.items()
            ]
        actor = self.actors.get(self.case.actor or "")
        if actor and actor.get("type") == "bearer":
            http["auth"] = {"type": "bearer", "token": actor["token"]}
        doc: dict[str, Any] = {
            "info": {"name": entry["info.name"], "type": "http", "tags": self.case.tags},
            "http": http,
        }
        runtime: dict[str, Any] = {}
        scripts = []
        file_index = next(i for i, e in enumerate(self.plan) if e["filename"] == entry["filename"])
        first_of_row = file_index == 0 or self.plan[file_index - 1].get("parameter_row") != row
        if first_of_row and self.case.variables:
            code = "\n".join(
                f'bru.setVar("{self.prefix}var_{k}", bru.interpolate({json.dumps(str(self.deep(v, row)))}));'
                for k, v in self.case.variables.items()
            )
            scripts.append({"type": "before-request", "code": code})
        if not is_cleanup:
            asserts = [{"expression": "res.status", "operator": "eq", "value": str(entry["expected_status"])}]
            tests = []
            for a in step.assertions:
                c = parse_check(self.literal_params(a.check, row))
                expr = self.bexpr(c.path)
                if c.op in OPS:
                    val = f"'{c.value}'" if isinstance(c.value, str) else json.dumps(c.value)
                    asserts.append({"expression": expr, "operator": OPS[c.op], "value": val})
                elif c.op == "exists":
                    asserts.append({"expression": expr, "operator": "isDefined"})
                elif c.op == "not exists":
                    asserts.append({"expression": expr, "operator": "isUndefined"})
                else:
                    js = {"in": "to.be.oneOf", "contains": "to.include", "matches": "to.match"}[c.op]
                    arg = json.dumps(c.value) if c.op != "matches" else f"new RegExp({json.dumps(c.value)})"
                    tests.append(
                        f"test({json.dumps(a.message or a.check)}, function () {{ expect({expr.replace('res.body', 'res.getBody()')}).{js}({arg}); }});"
                    )
            runtime["assertions"] = asserts
            if tests:
                scripts.append({"type": "tests", "code": "\n".join(tests)})
        if step.extract:
            runtime["actions"] = [
                {
                    "type": "set-variable",
                    "phase": "after-response",
                    "selector": {"expression": self.bexpr(p), "method": "jsonq"},
                    "variable": {"name": self.prefix + name, "scope": "runtime"},
                }
                for name, p in step.extract.items()
            ]
        if scripts:
            runtime["scripts"] = scripts
        if runtime:
            doc["runtime"] = runtime
        return yaml.safe_dump(doc, sort_keys=False)


def synthesizer_model(break_first: bool = False) -> FunctionModel:
    """Fake synthesizer. With break_first=True the first answer is invalid (tests the ModelRetry loop)."""
    state = {"calls": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state["calls"] += 1
        r = _Renderer(_prompt(messages))
        files = [{"filename": e["filename"], "content": r.render(e)} for e in r.plan]
        if break_first and state["calls"] == 1:
            files[0]["content"] = files[0]["content"].replace("{{baseUrl}}", "http://localhost:8000")
            files = files + [{"filename": "extra.yml", "content": "info: {}"}]
        return _respond(info, {"files": files})

    fn.state = state  # type: ignore[attr-defined]
    return FunctionModel(fn)

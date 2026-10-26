"""Local dashboard + QA chatbot (FastAPI + Jinja2 + HTMX). Read-only towards Gel."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from sse_starlette.sse import EventSourceResponse

from spec_run_graph.agents.chatbot import ChatDeps, build_agent
from spec_run_graph.config import get_settings
from spec_run_graph.knowledge import db, repo, stats
from spec_run_graph.llm import LLMNotConfigured, build_model, model_name, request_settings
from spec_run_graph.telemetry.s3 import TelemetryStore, case_prefix

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["tojson_pretty"] = lambda v: json.dumps(v, indent=2, default=str)

CHAT_HISTORY: dict[str, list[ModelMessage]] = {}
MAX_SESSIONS = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = db.create_client()
    app.state.store = TelemetryStore()
    app.state.chatbot = build_agent()
    yield
    await app.state.client.aclose()


app = FastAPI(title="SpecRunGraph", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    ctx.setdefault("model", get_settings().srg_model)
    response = templates.TemplateResponse(request, name, ctx)
    if "srg_session" not in request.cookies:
        response.set_cookie("srg_session", uuid.uuid4().hex, httponly=True, samesite="lax")
    return response


@app.get("/", response_class=HTMLResponse)
async def overview_page(request: Request):
    data = await stats.overview(request.app.state.client)
    runs = await stats.list_runs(request.app.state.client, 5)
    return render(request, "overview.html", page="overview", o=data, runs=runs)


@app.get("/cases", response_class=HTMLResponse)
async def cases_page(request: Request):
    filters = {k: v for k, v in request.query_params.items() if v}
    rows = await stats.list_cases(request.app.state.client, filters)
    template = "_cases_table.html" if request.headers.get("HX-Request") else "cases.html"
    modules = await repo.list_modules(request.app.state.client)
    return render(request, template, page="cases", rows=rows, f=filters, modules=modules)


@app.get("/cases/{test_id}", response_class=HTMLResponse)
async def case_page(request: Request, test_id: str):
    row = await repo.get_case(request.app.state.client, test_id)
    if row is None:
        raise HTTPException(404, f"unknown test case {test_id}")
    ir = repo.row_to_ir(row)
    ir_yaml = yaml.safe_dump(ir.model_dump(mode="json", exclude_defaults=True), sort_keys=False, allow_unicode=True)
    raw = await request.app.state.client.query_json(
        """select TestExecution { run_id := .run.run_id, param_index, status, http_status, duration_ms,
               error_message, created_at } filter .test_case.test_id = <str>$t order by .created_at desc limit 20""",
        t=test_id,
    )
    heals = await repo.heal_history(request.app.state.client, test_id)
    return render(request, "case.html", page="cases", c=row, ir_yaml=ir_yaml, executions=json.loads(raw), heals=heals)


@app.get("/endpoints", response_class=HTMLResponse)
async def endpoints_page(request: Request):
    client = request.app.state.client
    eps = await stats.endpoint_coverage(client)
    reqs = await stats.requirement_matrix(client)
    types = ["POSITIVE", "NEGATIVE", "BOUNDARY", "SECURITY", "RBAC"]
    return render(request, "endpoints.html", page="endpoints", eps=eps, reqs=reqs, types=types)


@app.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request):
    runs = await stats.list_runs(request.app.state.client, 50)
    return render(request, "runs.html", page="runs", runs=runs)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_page(request: Request, run_id: str):
    run = await stats.run_results(request.app.state.client, run_id)
    if run is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return render(request, "run.html", page="runs", run=run)


@app.get("/runs/{run_id}/logs/{test_id}/{param_index}", response_class=PlainTextResponse)
async def run_logs(request: Request, run_id: str, test_id: str, param_index: int):
    store: TelemetryStore = request.app.state.store
    try:
        text = await store.get_text(case_prefix(run_id, test_id, param_index) + "report.json")
    except Exception as exc:  # missing object / MinIO down
        raise HTTPException(404, f"log not available: {exc.__class__.__name__}") from exc
    return PlainTextResponse(json.dumps(json.loads(text), indent=2))


@app.get("/runs/{run_id}/stdout", response_class=PlainTextResponse)
async def run_stdout(request: Request, run_id: str):
    store: TelemetryStore = request.app.state.store
    try:
        return PlainTextResponse(await store.get_text(f"runs/{run_id}/bru.stdout"))
    except Exception as exc:
        raise HTTPException(404, f"log not available: {exc.__class__.__name__}") from exc


class ChatIn(BaseModel):
    message: str


@app.post("/chat/stream")
async def chat_stream(request: Request, body: ChatIn):
    session = request.cookies.get("srg_session") or "anonymous"

    async def events():
        deps = ChatDeps(client=request.app.state.client)
        try:
            model = build_model()
        except LLMNotConfigured as exc:
            yield {"event": "error", "data": str(exc)}
            return
        history = CHAT_HISTORY.get(session, [])
        try:
            async with request.app.state.chatbot.run_stream(
                body.message,
                deps=deps,
                model=model,
                message_history=history,
                model_settings=request_settings(thinking=True, session=session),
            ) as result:
                async for delta in result.stream_text(delta=True):
                    yield {"event": "token", "data": json.dumps(delta)}
                if len(CHAT_HISTORY) > MAX_SESSIONS:
                    CHAT_HISTORY.pop(next(iter(CHAT_HISTORY)))
                CHAT_HISTORY[session] = result.all_messages()[-40:]
        except Exception as exc:  # surface model/tool errors in the chat panel
            yield {"event": "error", "data": f"{exc.__class__.__name__}: {exc}"}
        yield {"event": "trace", "data": json.dumps(deps.trace)}
        yield {"event": "done", "data": model_name()}

    return EventSourceResponse(events())


@app.post("/chat/reset")
async def chat_reset(request: Request):
    CHAT_HISTORY.pop(request.cookies.get("srg_session") or "anonymous", None)
    return {"ok": True}

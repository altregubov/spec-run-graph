"""Agents with test models (no network). Production uses the single .env model for all of them."""

import pytest
from pydantic_ai import ModelRetry

from fake_llm import synthesizer_model, tool_model
from helpers import orders_analysis, orders_specs, reference_module
from spec_run_graph.agents import analyzer, planner, synthesizer
from spec_run_graph.agents.chatbot import guard_query
from spec_run_graph.domain.ir import ValidationContext
from spec_run_graph.engines.bruno.naming import plan_files

MODULE = {
    "code": "ORD",
    "folder": "orders",
    "domain": "Orders",
    "base_path": "",
    "default_headers": {},
    "preconditions": [],
    "actors": {"customer_standard": {"type": "bearer", "token": "{{ $env.CUSTOMER_TOKEN }}"}},
    "fixtures": {"active_sku": "SKU-ACTIVE-1"},
}


async def test_analyzer_retries_until_valid():
    specs = orders_specs()
    good = orders_analysis(specs).model_dump()
    bad = {**good, "modules": [{**good["modules"][0], "endpoints": ["GET /v1/nope"]}]}
    out = await analyzer.analyze(specs, model=tool_model([bad, good]))
    assert [m.code for m in out.modules] == ["PROD", "ORD"]


async def test_analyzer_prompt_has_all_specs():
    prompt = analyzer.build_prompt(orders_specs())
    assert "OpenAPI spec 'Orders API'" in prompt and "Requirements document" in prompt
    assert "There are 2 specifications" in prompt


def _ctx():
    specs = orders_specs()
    return ValidationContext(
        {e.key: e.path for s in specs for e in s.endpoints},
        {"PRD-ORD-42", "PRD-ORD-43"},
        {"customer_standard"},
        {"active_sku"},
    )


async def test_planner_retries_then_drops_invalid():
    specs = orders_specs()
    analysis = orders_analysis(specs)
    module = analysis.modules[1]
    ref = [c.model_dump(exclude={"id", "status"}) for c in reference_module().test_cases]
    broken = [{**ref[0], "steps": [{**ref[0]["steps"][0], "endpoint": "POST /v1/unknown"}]}, ref[1]]
    endpoints = {e.key: e for s in specs for e in s.endpoints}
    # first answer invalid -> ModelRetry -> second answer valid
    cases, dropped = await planner.plan_module(
        analysis, module, endpoints, _ctx(), [], model=tool_model([{"cases": broken}, {"cases": ref}])
    )
    assert len(cases) == 2 and dropped == []
    # always invalid -> after max retries the invalid case is dropped, the valid one kept
    cases, dropped = await planner.plan_module(
        analysis, module, endpoints, _ctx(), [], model=tool_model({"cases": broken})
    )
    assert [c.title for c in cases] == [ref[1]["title"]]
    assert dropped and "does not match any OpenAPI path" in " ".join(dropped[0][1])


def test_planner_prompt_lists_existing_cases():
    specs = orders_specs()
    analysis = orders_analysis(specs)
    endpoints = {e.key: e for s in specs for e in s.endpoints}
    prompt = planner.build_prompt(analysis, analysis.modules[1], endpoints, [{"id": "TC-ORD-001", "title": "x"}])
    assert "Already covered" in prompt and "TC-ORD-001" in prompt
    assert "max qty" in prompt  # relevant conflict passed on


async def test_synthesizer_retries_and_succeeds():
    case = reference_module().test_cases[1]
    planned = plan_files(case)
    model = synthesizer_model(break_first=True)
    parsed, deps = await synthesizer.synthesize(case, MODULE, planned, {}, {"customer-token"}, model=model)
    assert parsed is not None and deps.attempts == 2
    assert parsed["TC-ORD-002a-zero-quantity.yml"]["runtime"]["assertions"][0]["value"] == "422"


async def test_synthesizer_reports_failure():
    case = reference_module().test_cases[1]
    planned = plan_files(case)
    model = tool_model({"files": [{"filename": "nope.yml", "content": "info: {}"}]})
    parsed, deps = await synthesizer.synthesize(case, MODULE, planned, {}, set(), model=model)
    assert parsed is None
    assert any("missing files" in e for e in deps.last_errors)
    assert deps.attempts >= 2


@pytest.mark.parametrize("q", ["select count(TestCase)", "with x := 1 select x", "  SELECT TestCase { title } ;"])
def test_chat_guard_allows_selects(q):
    assert guard_query(q)


@pytest.mark.parametrize(
    "q",
    [
        "delete TestCase",
        "select 1; delete TestCase",
        "update TestCase set { title := 'x' }",
        "insert TestEngine { name := 'x' }",
        "configure session set x := 1",
        "select (delete TestCase)",
    ],
)
def test_chat_guard_rejects_writes(q):
    with pytest.raises(ModelRetry):
        guard_query(q)

"""Self-healing guard rails and agent loop (test models, no network)."""

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from helpers import orders_specs, reference_module
from spec_run_graph.agents import healer
from spec_run_graph.agents.healer import HealDeps, guard_errors
from spec_run_graph.domain.ir import CaseDraft, ValidationContext, normalize


@pytest.fixture
def deps():
    paths = {e.key: e.path for s in orders_specs() for e in s.endpoints}
    module = reference_module()
    ctx = ValidationContext(paths, {"PRD-ORD-42", "PRD-ORD-43"}, set(module.actors), set(module.fixtures))
    original = normalize(CaseDraft.model_validate(module.test_cases[0].model_dump(exclude={"id", "status"})))
    return HealDeps(
        original=original,
        ctx=ctx,
        documented_statuses={"POST /v1/orders": {201, 400, 401, 422}, "DELETE /v1/orders/{orderId}": {200, 404}},
    )


def with_check(case: CaseDraft, check: str) -> CaseDraft:
    c = case.model_copy(deep=True)
    c.steps[0].assertions[0].check = check
    return c


def test_valid_fix_is_accepted(deps):
    fixed = with_check(deps.original, "response.body.data.status == 'DRAFT' ")
    fixed.steps[0].assertions.append(type(fixed.steps[0].assertions[0])(check="response.body.data.id exists"))
    assert guard_errors(fixed, deps) == []


def test_guards_reject_unsafe_heals(deps):
    assert any("unchanged" in e for e in guard_errors(deps.original, deps))
    other_type = deps.original.model_copy(update={"type": "negative"})
    assert any("keep the case type" in e for e in guard_errors(other_type, deps))
    undocumented = deps.original.model_copy(deep=True)
    undocumented.steps[0].expected_status = 500  # "heal" the test to accept a crash
    assert any("not documented" in e for e in guard_errors(undocumented, deps))
    fewer = deps.original.model_copy(update={"cleanup": []})
    assert any("same endpoints" in e for e in guard_errors(fewer, deps))
    no_reqs = deps.original.model_copy(deep=True)
    no_reqs.coverage.requirement_refs = []
    assert any("requirement_refs" in e for e in guard_errors(no_reqs, deps))


async def test_agent_retries_until_guard_rails_pass(deps):
    bad = deps.original.model_copy(deep=True)
    bad.steps[0].expected_status = 500
    good = with_check(deps.original, "response.body.data.id exists")
    answers = [
        {"case": bad.model_dump(mode="json"), "change_summary": "accept 500"},
        {"case": good.model_dump(mode="json"), "change_summary": "assert the id instead of the status"},
    ]

    def fn(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answers.pop(0))])

    out = await healer.heal("prompt", deps, model=FunctionModel(fn))
    assert out.change_summary.startswith("assert the id") and deps.attempts == 2
    assert out.case.steps[0].assertions[0].check == "response.body.data.id exists"


async def test_unhealable_is_returned_unchanged(deps):
    answer = {
        "case": deps.original.model_dump(mode="json"),
        "change_summary": "-",
        "unhealable_reason": "the API returns 500, this is an application bug",
    }

    def fn(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    out = await healer.heal("prompt", deps, model=FunctionModel(fn))
    assert out.unhealable_reason.startswith("the API returns 500")

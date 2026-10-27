"""Failure triage: rule pre-classifier, grouping and the diagnostics agent (test models, no network)."""

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from spec_run_graph.agents import diagnostics
from spec_run_graph.agents.diagnostics import DiagDeps, FailedExecution, preclassify, signature


def failure(**kw) -> FailedExecution:
    base = dict(
        execution_id="e1",
        run_id="run-1",
        test_id="TC-ORD-005",
        title="qty limit",
        param_index=0,
        status="FAILED",
        http_status=500,
        error_message=None,
        failed_assertions=[{"assertion": "res.status eq 400", "error": "expected 500 to equal 400"}],
        case_ir={"title": "qty limit", "type": "boundary"},
        endpoints=["POST /v1/orders"],
        requirement_ids=["PRD-ORD-43"],
        actor="customer_standard",
    )
    return FailedExecution(**{**base, **kw})


def test_preclassify_env_failures():
    assert (
        preclassify(failure(error_message="connect ECONNREFUSED 127.0.0.1:8080", http_status=None)).category
        == "ENV_FAILURE"
    )
    assert preclassify(failure(http_status=503)).category == "ENV_FAILURE"
    assert preclassify(failure(status="TIMEOUT", http_status=None)).category == "ENV_FAILURE"
    assert preclassify(failure()) is None  # a 500 needs a real investigation


def test_signature_groups_auth_failures_across_endpoints():
    auth = [{"assertion": "res.status eq 200", "error": "expected 401 to equal 200"}]
    a = failure(test_id="TC-A-001", http_status=401, failed_assertions=auth, endpoints=["GET /a"])
    b = failure(test_id="TC-B-007", http_status=401, failed_assertions=auth, endpoints=["POST /b/{id}"])
    c = failure(test_id="TC-C-002", http_status=401, failed_assertions=auth, actor="admin")
    assert signature(a) == signature(b) != signature(c)
    # non-auth failures stay separate per endpoint
    assert signature(failure(endpoints=["GET /x"])) != signature(failure(endpoints=["GET /y"]))


def _deps(f: FailedExecution, used: list[str]) -> DiagDeps:
    async def report():
        used.append("report")
        return '[{"response": {"status": 500, "data": "Internal Server Error"}}]'

    async def empty():
        return {}

    async def rerun():
        used.append("rerun")
        return "FAILED (HTTP 500) res.status eq 400"

    return DiagDeps(
        failure=f,
        get_report=report,
        get_contracts=empty,
        get_requirements=empty,
        get_files=empty,
        get_history=empty,
        rerun=rerun,
    )


async def test_agent_investigates_with_tools_then_diagnoses():
    used: list[str] = []
    answer = {
        "category": "APPLICATION_BUG",
        "root_cause": "qty > 100 crashes the order service with HTTP 500",
        "evidence": ["HTTP 500 instead of documented 400 MAX_ORDER_LIMIT"],
        "recommended_action": "Fix the order-limit validation in the API",
        "confidence": 0.9,
    }

    def fn(messages, info):
        if len(messages) == 1:  # first turn: call tools
            return ModelResponse(parts=[ToolCallPart("http_exchange", {}), ToolCallPart("rerun_case", {})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    f = failure(similar=["TC-ORD-006a"])
    deps = _deps(f, used)
    d = await diagnostics.diagnose(deps, model=FunctionModel(fn))
    assert d.category == "APPLICATION_BUG" and d.confidence == 0.9
    assert used == ["report", "rerun"] and deps.rerun_result.startswith("FAILED")
    assert "1 other executions" in diagnostics.build_prompt(f)


async def test_low_confidence_becomes_unknown():
    answer = {
        "category": "FLAKY",
        "root_cause": "maybe a timing issue between order creation and the read",
        "evidence": ["passed in an earlier run"],
        "recommended_action": "re-run the case",
        "confidence": 0.3,
    }

    def fn(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])

    d = await diagnostics.diagnose(_deps(failure(), []), model=FunctionModel(fn))
    assert d.category == "UNKNOWN" and "low confidence FLAKY" in d.root_cause


def test_placeholder_diagnoses_are_rejected():
    from spec_run_graph.agents.diagnostics import Diagnosis, diagnosis_problems

    junk = Diagnosis(category="UNKNOWN", root_cause="placeholder", recommended_action="placeholder", confidence=0)
    assert len(diagnosis_problems(junk)) == 2
    good = Diagnosis(
        category="TEST_DEFECT",
        root_cause="The test asserts total_count but the API returns no such field",
        evidence=["total_count undefined"],
        recommended_action="Fix the test assertion",
        confidence=0.8,
    )
    assert diagnosis_problems(good) == []

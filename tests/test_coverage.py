"""Coverage gaps (gap-fill pass) and acceptance-criterion refs."""

from types import SimpleNamespace

from spec_run_graph.domain.analysis import AnalyzedRequirement, ModuleProposal
from spec_run_graph.domain.ir import CaseDraft, StepIR, ValidationContext, criterion_ids, normalize, validate_case
from spec_run_graph.pipeline.coverage import module_coverage, violates

REQ = AnalyzedRequirement(
    req_id="REQ-PRD-01",
    spec="prd",
    title="Create product",
    description="d",
    acceptance_criteria=["price must be positive", "stock is stored"],
    endpoints=["POST /api/products"],
)
SCHEMA = {
    "type": "object",
    "required": ["price"],
    "properties": {"price": {"type": "number", "minimum": 0.01}, "name": {"type": "string"}},
}
ENDPOINT = SimpleNamespace(request_schema=SCHEMA, response_schemas={"201": {}, "400": {}, "500": {}})
MODULE = ModuleProposal(code="PRD", folder="products", domain="d", endpoints=["POST /api/products"])
PATHS = {"POST /api/products": "/api/products"}


def test_violates_detects_values_breaking_the_schema():
    prop = {"type": "number", "minimum": 0.01}
    assert violates(prop, 0) and violates(prop, "abc") and violates(prop, None)
    assert not violates(prop, 0.01)
    assert violates({"type": "string", "format": "uuid"}, "not-a-uuid")
    assert violates({"type": "string", "enum": ["a"]}, "b") and not violates({"type": "string", "enum": ["a"]}, "a")


def test_gaps_list_unchecked_criteria_statuses_and_constrained_fields():
    ok = CaseDraft(
        title="create",
        type="positive",
        coverage={"requirement_refs": ["REQ-PRD-01"], "criteria": ["REQ-PRD-01#2"]},
        steps=[StepIR(name="c", endpoint="POST /api/products", request={"payload": {"price": 5}}, expected_status=201)],
    )
    cov = module_coverage(MODULE, [REQ], [ok], {"POST /api/products": ENDPOINT}, PATHS)
    texts = [g.text for g in cov.gaps]
    assert (cov.criteria_covered, cov.criteria_total) == (1, 2)
    assert any("REQ-PRD-01#1" in t for t in texts)
    assert any("status 400 is never expected" in t for t in texts) and not any("500" in t for t in texts)
    assert any("field price" in t for t in texts)

    bad = CaseDraft(
        title="invalid prices",
        type="boundary",
        parameters=[{"name": "zero", "p": 0}, {"name": "min", "p": 0.01}],
        steps=[
            StepIR(
                name="c",
                endpoint="POST /api/products",
                request={"payload": {"price": "{{ $param.p }}"}},
                expected_status=400,
            )
        ],
    )
    texts = [g.text for g in module_coverage(MODULE, [REQ], [ok, bad], {"POST /api/products": ENDPOINT}, PATHS).gaps]
    assert not any("field price" in t or "status 400" in t for t in texts)


def test_criteria_refs_are_validated_and_imply_their_requirement():
    case = CaseDraft(
        title="t",
        type="positive",
        coverage={"criteria": ["REQ-PRD-01#2"]},
        steps=[StepIR(name="c", endpoint="POST /api/products", request={"payload": {"price": 5}}, expected_status=201)],
    )
    assert normalize(case).coverage.requirement_refs == ["REQ-PRD-01"]
    ctx = ValidationContext(endpoint_paths=PATHS, criteria_ids=set(criterion_ids([REQ])))
    bad = case.model_copy(update={"coverage": case.coverage.model_copy(update={"criteria": ["REQ-PRD-01#9"]})})
    assert any("unknown criteria ref" in e for e in validate_case(normalize(bad), ctx))
    assert not any("criteria" in e for e in validate_case(normalize(case), ctx))


def test_paginated_lookup_must_sort_newest_first_when_the_contract_allows_it():
    from spec_run_graph.specs.contract import newest_first

    params = [
        {"in": "query", "name": "sort_by", "schema": {"enum": ["price", "created_at"]}},
        {"in": "query", "name": "order", "schema": {"enum": ["asc", "desc"]}},
    ]
    sorts = newest_first({"GET /api/products": SimpleNamespace(parameters=params)})
    assert sorts == {"GET /api/products": {"sort_by": "created_at", "order": "desc"}}
    ctx = ValidationContext(
        endpoint_paths={"GET /api/products": "/api/products"},
        page_limits={"GET /api/products": ("page_size", 100)},
        newest_first=sorts,
    )
    step = StepIR(
        name="find",
        endpoint="GET /api/products",
        request={"query": {"page_size": 100}},
        expected_status=200,
        assertions=[{"check": "response.body.data[model='{{ $var.m }}'].id exists"}],
    )
    case = CaseDraft(title="t", type="positive", variables={"m": "qa_{{ $faker.uuid }}"}, steps=[step])
    assert any("sort_by=created_at, order=desc" in e for e in validate_case(case, ctx))
    seeded = StepIR.model_validate(
        {**step.model_dump(), "assertions": [{"check": "response.body.data[model='MacBook'].id exists"}]}
    )
    assert not any("paginated" in e for e in validate_case(case.model_copy(update={"steps": [seeded]}), ctx))
    step.request.query.update({"sort_by": "created_at", "order": "desc"})
    assert not any("paginated" in e for e in validate_case(case, ctx))


def _module(endpoints, requirement_ids=()):
    return SimpleNamespace(code="PRD", endpoints=endpoints, requirement_ids=list(requirement_ids))


def _create_case(payload, status=201):
    step = StepIR(name="create", endpoint="POST /api/products", request={"payload": payload}, expected_status=status)
    return normalize(CaseDraft(title="c", type="positive", steps=[step]))


def _technique_gaps(cases):
    endpoint = SimpleNamespace(response_schemas={}, request_schema=SCHEMA)
    cov = module_coverage(MODULE, [], cases, {"POST /api/products": endpoint}, PATHS)
    return [g.text for g in cov.gaps]


def test_free_text_and_decimal_fields_need_their_techniques():
    gaps = _technique_gaps([_create_case({"price": 5, "name": "plain"})])
    assert any("text name - no successful" in g for g in gaps)
    assert any("decimal price - no successful" in g for g in gaps)
    assert any("decimal price - no step sends more digits" in g for g in gaps)


def test_techniques_count_as_covered_when_sent():
    cases = [_create_case({"price": 19.99, "name": "Größe"}), _create_case({"price": 19.999, "name": "x"}, 400)]
    gaps = _technique_gaps(cases)
    assert not any("non-ASCII" in g or "cent-precise" in g or "more digits" in g for g in gaps)


def test_a_rejected_value_does_not_count_as_stored_text():
    gaps = _technique_gaps([_create_case({"price": 19.99, "name": "Größe"}, 400)])
    assert any("non-ASCII" in g for g in gaps) and any("cent-precise" in g for g in gaps)


def test_non_ascii_inside_a_variable_counts():
    step = StepIR(
        name="c", endpoint="POST /api/products", request={"payload": {"name": "{{ $var.n }}"}}, expected_status=201
    )
    case = normalize(CaseDraft(title="c", type="positive", variables={"n": "日本語_{{ $faker.uuid }}"}, steps=[step]))
    assert not any("non-ASCII" in g for g in _technique_gaps([case]))


def test_chunks_answer_only_for_their_own_endpoints():
    from spec_run_graph.pipeline.coverage import chunk_requirement_ids

    reqs = [
        SimpleNamespace(req_id="A", endpoints=["GET /a"]),
        SimpleNamespace(req_id="B", endpoints=["GET /b"]),
        SimpleNamespace(req_id="X", endpoints=["GET /elsewhere"]),
    ]
    module = _module(["GET /a", "GET /b"], ["A", "B", "X"])
    assert chunk_requirement_ids(module, ["GET /a"], reqs, first=True) == ["A", "X"]
    assert chunk_requirement_ids(module, ["GET /b"], reqs, first=False) == ["B"]


def test_non_ascii_variable_inside_a_list_counts():
    payload = {"name": ["{{ $var.n }}"]}
    step = StepIR(name="c", endpoint="POST /api/products", request={"payload": payload}, expected_status=201)
    case = normalize(CaseDraft(title="c", type="positive", variables={"n": "日本語"}, steps=[step]))
    assert not any("non-ASCII" in g for g in _technique_gaps([case]))

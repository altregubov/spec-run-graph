import pytest

from helpers import orders_specs, reference_module
from spec_run_graph.domain.expr import CheckSyntaxError, parse_check
from spec_run_graph.domain.ir import (
    CoverageIR,
    ValidationContext,
    fingerprint,
    normalize,
    resolve_endpoint,
    validate_case,
)


@pytest.fixture(scope="module")
def paths():
    return {e.key: e.path for s in orders_specs() for e in s.endpoints}


def ctx_for(module, paths):
    return ValidationContext(paths, {"PRD-ORD-42", "PRD-ORD-43"}, set(module.actors), set(module.fixtures))


@pytest.mark.parametrize(
    ("text", "op", "value"),
    [
        ("response.status == 201", "==", 201),
        ("response.body.data.status == 'DRAFT'", "==", "DRAFT"),
        ("response.body.data.items[0].qty >= 1", ">=", 1),
        ("response.body.data.status in ['CONFIRMED', 'FAILED']", "in", ["CONFIRMED", "FAILED"]),
        ("response.headers.content-type contains 'json'", "contains", "json"),
        ("response.body.data.id exists", "exists", None),
        ("response.body.data.id not exists", "not exists", None),
        ("response.body.error.code == {{ $param.error_code }}", "==", "{{ $param.error_code }}"),
    ],
)
def test_parse_check(text, op, value):
    c = parse_check(text)
    assert (c.op, c.value) == (op, value)


@pytest.mark.parametrize(
    "text",
    [
        "response.data.status == 'DRAFT'",  # ambiguous path from the reference format
        "response.status === 201",
        "response.status > 'x'",
        "response.body.status in 'x'",
        "response.body.id exists 1",
        "response.status == DRAFT",
    ],
)
def test_parse_check_rejects(text):
    with pytest.raises(CheckSyntaxError):
        parse_check(text)


def test_resolve_endpoint(paths):
    assert resolve_endpoint("GET /v1/orders/{{ $extract.order_id }}", paths) == "GET /v1/orders/{orderId}"
    assert resolve_endpoint("POST /v1/orders/abc/confirm", paths) == "POST /v1/orders/{orderId}/confirm"
    assert resolve_endpoint("PATCH /v1/orders", paths) is None


def test_original_reference_format_is_rejected(paths):
    module = reference_module("reference_original")
    tc1, tc2 = module.test_cases
    e1 = " | ".join(validate_case(tc1, ctx_for(module, paths)))
    assert "$extract.idempotency_key is not extracted" in e1
    assert "bad path 'response.data.status'" in e1
    assert "unknown actor" in e1 and "unknown fixture 'active_sku'" in e1
    e2 = " | ".join(validate_case(tc2, ctx_for(module, paths)))
    assert "needs a kebab-case 'name' slug" in e2


def test_fixed_reference_format_is_valid(paths):
    module = reference_module("reference_fixed")
    for tc in module.test_cases:
        assert validate_case(normalize(tc), ctx_for(module, paths)) == []


def test_legacy_shapes_are_coerced():
    module = reference_module("reference_original")
    tc1 = module.test_cases[0]
    assert tc1.coverage.requirement_refs == ["PRD-ORD-42"]
    assert tc1.cleanup[0].endpoint.startswith("DELETE /v1/orders/")
    assert CoverageIR.model_validate({"requirement_refs": "X-1"}).requirement_refs == ["X-1"]


def test_normalize_folds_expected_error(paths):
    tc2 = normalize(reference_module().test_cases[1])
    checks = [a.check for a in tc2.steps[0].assertions]
    assert "response.body.error.code == {{ $param.error_code }}" in checks
    assert "response.body.error.field == 'items[0].qty'" in checks
    assert tc2.steps[0].expected_error is None


def test_param_and_var_validation(paths):
    module = reference_module()
    tc = module.test_cases[1].model_copy(deep=True)
    tc.parameters[1].pop("error_code")
    errs = validate_case(tc, ctx_for(module, paths))
    assert any("$param.error_code is missing" in e for e in errs)
    tc = module.test_cases[0].model_copy(deep=True)
    tc.steps[0].request.headers["X"] = "{{ $var.nope }}"
    tc.steps[0].request.headers["Y"] = "{{ token }}"
    errs = " | ".join(validate_case(tc, ctx_for(module, paths)))
    assert "$var.nope is not defined" in errs and "must use a namespace" in errs


def test_fingerprint_is_stable(paths):
    tc = reference_module().test_cases[0]
    renamed = tc.model_copy(update={"title": "  CREATE draft order, as authorized customer!"})
    assert fingerprint("ORD", tc, paths) == fingerprint("ORD", renamed, paths)
    assert fingerprint("ORD", tc, paths) != fingerprint("PROD", tc, paths)

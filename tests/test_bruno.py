import json

import pytest
import yaml

from fake_llm import _Renderer
from helpers import FIXTURES, orders_specs, reference_module
from spec_run_graph.agents.synthesizer import build_prompt
from spec_run_graph.engines.bruno.naming import parse_filename, plan_files, slugify
from spec_run_graph.engines.bruno.runner import aggregate, flatten_results
from spec_run_graph.engines.bruno.scaffold import write_scaffold
from spec_run_graph.engines.bruno.validate import schema_errors, validate_files

MODULE = {
    "code": "ORD",
    "folder": "orders",
    "domain": "Orders",
    "base_path": "",
    "default_headers": {"Accept": "application/json"},
    "preconditions": [],
    "actors": {"customer_standard": {"type": "bearer", "token": "{{ $env.CUSTOMER_TOKEN }}"}},
    "fixtures": {"active_sku": "SKU-ACTIVE-1"},
}


@pytest.fixture
def cases():
    tc1, tc2 = reference_module().test_cases
    return tc1, tc2


def render_all(case):
    planned = plan_files(case)
    prompt = build_prompt(case, MODULE, planned, {})
    r = _Renderer(prompt)
    return planned, {e["filename"]: r.render(e) for e in r.plan}


def test_naming_shapes(cases):
    tc1, tc2 = cases
    assert [p.filename for p in plan_files(tc1)] == [
        "TC-ORD-001-step1-submit-draft-order-creation-request.yml",
        "TC-ORD-001-step2-delete-created-order.yml",
    ]
    assert [p.filename for p in plan_files(tc2)] == [
        "TC-ORD-002a-zero-quantity.yml",
        "TC-ORD-002b-negative-quantity.yml",
        "TC-ORD-002c-over-max-quantity.yml",
    ]
    single = tc2.model_copy(update={"parameters": [], "id": "TC-ORD-003"})
    assert [p.filename for p in plan_files(single)] == ["TC-ORD-003.yml"]
    multi_param = tc1.model_copy(update={"parameters": [{"name": "x"}, {"name": "y"}], "id": "TC-ORD-010"})
    assert plan_files(multi_param)[2].filename == "TC-ORD-010b-step1-submit-draft-order-creation-request.yml"
    assert plan_files(tc1)[1].is_cleanup


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("TC-ORD-003.yml", ("TC-ORD-003", 0, 1)),
        ("TC-ORD-005b-negative-quantity.yml", ("TC-ORD-005", 1, 1)),
        ("TC-ORD-009-step2-verify-balance.yml", ("TC-ORD-009", 0, 2)),
        ("TC-PROD_X-0123c-step3-browse.yml", ("TC-PROD_X-0123", 2, 3)),
        ("folder.yml", None),
    ],
)
def test_parse_filename(name, expected):
    assert parse_filename(name) == expected


def test_slugify():
    assert slugify("Verify balance -- refunded!") == "verify-balance-refunded"


def test_valid_files_pass(cases):
    for case in cases:
        planned, files = render_all(case)
        errors, parsed = validate_files(case, MODULE, planned, files, {"customer-token"})
        assert errors == []
        assert set(parsed) == {p.filename for p in planned}


def test_invalid_files_are_reported(cases):
    _, tc2 = cases
    planned, files = render_all(tc2)
    name = planned[0].filename
    files[name] = files[name].replace("{{process.env.CUSTOMER_TOKEN}}", "customer-token")
    files[name] = files[name].replace("'{{baseUrl}}/v1/orders'", "'{{baseUrl}}/v1/orders/{{ $extract.order_id }}'")
    files[planned[1].filename] = files[planned[1].filename].replace("value: '422'", "value: '200'")
    del files[planned[2].filename]
    files["TC-ORD-002-extra.yml"] = "info: {}"
    errors = " | ".join(validate_files(tc2, MODULE, planned, files, {"customer-token"})[0])
    assert "missing files" in errors and "unexpected files" in errors
    assert "literal secret" in errors
    assert "IR templates" in errors
    assert "needs assertion {expression: res.status, operator: eq, value: '422'}" in errors


def test_js_syntax_is_checked(cases):
    tc1, _ = cases
    planned, files = render_all(tc1)
    doc = yaml.safe_load(files[planned[0].filename])
    doc["runtime"]["scripts"] = [{"type": "tests", "code": "test('x', function () { expect(1).to.equal(1) "}]
    files[planned[0].filename] = yaml.safe_dump(doc)
    errors = validate_files(tc1, MODULE, planned, files, set())[0]
    assert any("JavaScript syntax error" in e for e in errors)


def test_scaffold_matches_schema(tmp_path):
    write_scaffold(tmp_path, [MODULE | {"fixtures": {"active_sku": "SKU-1", "secret_sku": {"env": "SKU"}}}])
    assert schema_errors(yaml.safe_load((tmp_path / "opencollection.yml").read_text()), None) == []
    env = yaml.safe_load((tmp_path / "environments" / "local.yml").read_text())
    assert schema_errors(env, "Environment") == []
    assert {v["name"] for v in env["variables"]} == {"baseUrl", "fx_ORD_active_sku"}
    assert schema_errors(yaml.safe_load((tmp_path / "orders" / "folder.yml").read_text()), "Folder") == []


def test_aggregate_recorded_run():
    results = flatten_results(json.loads((FIXTURES / "bru_results.json").read_text()))
    manifest = json.loads((FIXTURES / "bru_manifest.json").read_text())
    expected = {}
    for f, meta in manifest.items():
        expected.setdefault((meta["test_id"], meta["param_index"]), []).append(f)
    out = aggregate(results, manifest, expected)
    assert out[("TC-ORD-001", 0)].status == "PASSED"
    assert out[("TC-ORD-002", 0)].status == "PASSED"
    bad = out[("TC-ORD-002", 2)]
    assert bad.status == "FAILED" and bad.http_status == 500  # seeded defect: qty > 100 crashes
    assert any("res.status eq 400" in f["assertion"] for f in bad.failed)
    missing = aggregate([], manifest, expected)
    assert {o.status for o in missing.values()} == {"SKIPPED"}
    assert {o.status for o in aggregate([], manifest, expected, timed_out=True).values()} == {"TIMEOUT"}


def test_prompt_contains_translation_inputs(cases):
    _, tc2 = cases
    prompt = build_prompt(tc2, MODULE, plan_files(tc2), {})
    assert "TC_ORD_002__" in prompt and "fx_ORD_active_sku" in prompt
    assert "{{process.env.CUSTOMER_TOKEN}}" in prompt
    assert '"expected_status": 400' in prompt


def test_orders_specs_load():
    assert len(orders_specs()) == 2

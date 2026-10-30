"""Regression tests for framework defects found on a real run (warehouse API, 2026-10-26)."""

import yaml

from helpers import orders_specs, reference_module
from spec_run_graph.agents.diagnostics import FailedExecution, signature
from spec_run_graph.domain.ir import ValidationContext, body_path_problem, normalize, validate_case
from spec_run_graph.engines.bruno.normalize import normalize_doc, normalize_text


def doc(**http):
    return {"info": {"name": "x", "type": "http"}, "http": {"method": "GET", "url": "{{baseUrl}}/api/p", **http}}


# A/B/C: deterministic normalizer -------------------------------------------------------------


def test_query_params_move_into_url():  # bru CLI ignores http.params query entries
    fixed, fixes = normalize_doc(
        doc(
            params=[
                {"name": "page_size", "value": "100", "type": "query"},
                {"name": "manufacturer", "value": "{{fx_CAT_brand}}", "type": "query"},
            ]
        )
    )
    assert fixed["http"]["url"] == "{{baseUrl}}/api/p?page_size=100&manufacturer={{fx_CAT_brand}}"
    assert "params" not in fixed["http"] and fixes == ["query params moved into the url"]
    fixed, _ = normalize_doc(doc(url="{{baseUrl}}/api/p?a=1", params=[{"name": "b", "value": "x y", "type": "query"}]))
    assert fixed["http"]["url"] == "{{baseUrl}}/api/p?a=1&b=x%20y"
    again, fixes = normalize_doc(fixed)
    assert fixes == [] and again == fixed


def test_explicit_authorization_header_drops_auth_block():  # else the valid actor token wins
    d = doc(
        headers=[{"name": "Authorization", "value": "Bearer tampered"}],
        auth={"type": "bearer", "token": "{{auth_user_a}}"},
    )
    fixed, fixes = normalize_doc(d)
    assert "auth" not in fixed["http"] and fixes
    untouched, fixes = normalize_doc(doc(auth={"type": "bearer", "token": "{{auth_user_a}}"}))
    assert untouched["http"]["auth"] and fixes == []


def test_regex_quotes_are_stripped():  # quotes became part of the pattern
    d = doc()
    d["runtime"] = {
        "assertions": [
            {"expression": "res.body.data.token", "operator": "matches", "value": "'^eyJ.+$'"},
            {"expression": "res.status", "operator": "eq", "value": "'x'"},
        ]
    }
    fixed, _ = normalize_doc(d)
    assert [a["value"] for a in fixed["runtime"]["assertions"]] == ["^eyJ.+$", "'x'"]
    text, fixes = normalize_text(yaml.safe_dump(d))
    assert fixes and "^eyJ.+$" in text
    assert normalize_text("not: [valid")[1] == []


# D/E: IR validation --------------------------------------------------------------------------

ERROR = {
    "type": "object",
    "required": ["success", "error"],
    "properties": {
        "success": {"type": "boolean"},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "details": {"type": "array", "items": {"type": "object", "properties": {"field": {}, "issue": {}}}},
            },
        },
    },
}
PRODUCTS = {
    "allOf": [
        {"type": "object", "properties": {"success": {}, "data": {}}},
        {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {}, "model": {}, "price": {}}},
                }
            },
        },
    ]
}


def test_body_paths_are_checked_against_the_contract():
    assert body_path_problem(ERROR, "response.body.error.code") is None
    assert body_path_problem(ERROR, "response.body.error.details[0].field") is None
    assert "error.field is not in the documented response" in body_path_problem(ERROR, "response.body.error.field")
    assert body_path_problem(PRODUCTS, "response.body.data[0].id") is None
    assert "available: ['id', 'model', 'price']" in body_path_problem(PRODUCTS, "response.body.data[0].product_id")
    assert body_path_problem(PRODUCTS, "response.body.data.length") is None
    assert body_path_problem({"type": "object"}, "response.body.anything") is None  # unconstrained


def _ctx(**kw):
    paths = {e.key: e.path for s in orders_specs() for e in s.endpoints}
    return ValidationContext(paths, {"PRD-ORD-42", "PRD-ORD-43"}, {"customer_standard"}, {"active_sku"}, **kw)


def test_ir_rejects_undocumented_fields():
    case = normalize(reference_module().test_cases[1])  # asserts response.body.error.field
    schemas = {"POST /v1/orders": {"422": {"schema": ERROR}, "400": {"schema": ERROR}}}
    errs = " ".join(validate_case(case, _ctx(response_schemas=schemas)))
    assert "error.field is not in the documented response" in errs
    assert validate_case(case, _ctx()) == []  # no schemas -> no field check


def test_ir_requires_unique_values_for_created_entities():
    case = reference_module().test_cases[0].model_copy(deep=True)
    case.steps[0].request.payload["username"] = "qa_catalog_wl_user"  # expected 201: creates an entity
    errs = " ".join(validate_case(case, _ctx()))
    assert "username='qa_catalog_wl_user' is a fixed value" in errs
    case.steps[0].request.payload["username"] = "qa_{{ $var.idempotency_key }}"
    assert validate_case(case, _ctx()) == []
    case.steps[0].request.payload["username"] = "ab"
    case.steps[0].expected_status = 422  # a validation test may use literals
    assert validate_case(case, _ctx()) == []


# G: grouping ---------------------------------------------------------------------------------


def test_403_failures_are_not_grouped_across_endpoints():
    def f(endpoint, status):
        return FailedExecution(
            execution_id="e",
            run_id="r",
            test_id="t",
            title="t",
            param_index=0,
            status="FAILED",
            http_status=status,
            error_message=None,
            failed_assertions=[{"assertion": "res.status eq 200", "error": "expected 403"}],
            case_ir={},
            endpoints=[endpoint],
            requirement_ids=[],
            actor="user_a",
        )

    assert signature(f("GET /a", 403)) != signature(f("POST /b", 403))
    assert signature(f("GET /a", 401)) == signature(f("POST /b", 401))


# list filters ----------------------------------------------------------------------------------


def test_filter_paths_select_list_items_by_attribute():
    from spec_run_graph.domain.expr import parse_check

    c = parse_check("response.body.data[model='Galaxy S24'].price > 10")
    assert (c.path, c.op, c.value) == ("response.body.data[model='Galaxy S24'].price", ">", 10)
    assert parse_check("response.body.data[id={{ $extract.pid }}].stock exists").op == "exists"
    assert body_path_problem(PRODUCTS, "response.body.data[model='X'].id") is None
    assert "filters on a field the list items do not have" in body_path_problem(
        PRODUCTS, "response.body.data[name='X'].id"
    )


def test_bruno_files_must_not_use_filters_in_assertions():
    from spec_run_graph.engines.bruno.validate import validate_files
    from test_bruno import MODULE, render_all

    case = reference_module().test_cases[0]
    planned, files = render_all(case)
    d = yaml.safe_load(files[planned[0].filename])
    d["runtime"]["assertions"].append({"expression": "res.body.data[model='X'].id", "operator": "isDefined"})
    files[planned[0].filename] = yaml.safe_dump(d)
    errors = validate_files(case, MODULE, planned, files, set())[0]
    assert any("Array.find" in e for e in errors)


def test_absence_checks_on_undocumented_fields_are_allowed():
    case = normalize(reference_module().test_cases[1]).model_copy(deep=True)
    case.steps[0].assertions = [type(case.steps[0].assertions[0])(check="response.body.error.password_hash not exists")]
    schemas = {"POST /v1/orders": {"422": {"schema": ERROR}, "400": {"schema": ERROR}}}
    assert validate_case(case, _ctx(response_schemas=schemas)) == []


def test_isolation_rule_rejects_changes_to_pre_existing_entities():
    from spec_run_graph.domain.ir import StepIR

    case = reference_module().test_cases[0].model_copy(deep=True)
    lookup = StepIR(
        name="find order",
        endpoint="GET /v1/orders/{{ $var.idempotency_key }}",
        expected_status=200,
        extract={"seed_order": "response.body.data.id"},
    )
    change = StepIR(
        name="confirm seed order", endpoint="POST /v1/orders/{{ $extract.seed_order }}/confirm", expected_status=200
    )
    case.steps = [case.steps[0], lookup, change]
    errs = " ".join(validate_case(case, _ctx()))
    assert "modifies a pre-existing entity ($extract.seed_order" in errs
    # the cleanup DELETE of the order created by step 1 (POST) is fine
    assert "$extract.order_id" not in errs


def test_schema_checks_are_generated_not_written_by_the_llm():
    """Regression: the LLM copied multi-KB schemas into ajv scripts and broke the JS syntax."""
    from spec_run_graph.agents.synthesizer import add_schema_checks
    from spec_run_graph.engines.bruno.naming import plan_files
    from spec_run_graph.engines.bruno.validate import check_js_syntax, validate_files
    from test_bruno import MODULE, render_all

    case = reference_module().test_cases[0]  # step 1 has validate_schema: true, expects 201
    planned, files = render_all(case)
    parsed = {name: yaml.safe_load(text) for name, text in files.items()}
    endpoints = {
        "POST /v1/orders": {
            "responses": {
                "201": {
                    "schema": {"type": "object", "properties": {"data": {"type": "object", "example": {"id": "x"}}}}
                }
            }
        }
    }
    add_schema_checks(case, plan_files(case), parsed, endpoints)
    scripts = parsed[planned[0].filename]["runtime"]["scripts"]
    code = scripts[-1]["code"]
    assert code.startswith("// srg: response schema check") and '"example"' not in code
    assert check_js_syntax([code]) == []
    assert "scripts" not in parsed[planned[1].filename].get("runtime", {})  # cleanup step untouched
    # the generated check passes validation, LLM-written ajv does not
    files[planned[0].filename] = yaml.safe_dump(parsed[planned[0].filename])
    assert validate_files(case, MODULE, planned, files, set())[0] == []
    d = parsed[planned[0].filename]
    d["runtime"]["scripts"] = [{"type": "tests", "code": 'const Ajv = require("ajv");'}]
    files[planned[0].filename] = yaml.safe_dump(d)
    assert any("schema validation (ajv)" in e for e in validate_files(case, MODULE, planned, files, set())[0])


def test_missing_required_request_fields():
    from spec_run_graph.domain.ir import missing_required

    schema = {
        "type": "object",
        "required": ["username", "role", "address"],
        "properties": {"address": {"type": "object", "required": ["city"], "properties": {"city": {}}}},
    }
    assert missing_required(schema, {"username": "x", "address": {}}) == ["payload.role", "payload.address.city"]
    assert missing_required(schema, {"username": "x", "role": "a", "address": {"city": "c"}}) == []


def test_optional_response_field_rejected():
    from spec_run_graph.domain.ir import body_path_problem

    schema = {"type": "object", "required": ["id"], "properties": {"id": {}, "requestId": {}}}
    assert body_path_problem(schema, "response.body.id") is None
    assert "optional" in body_path_problem(schema, "response.body.requestId")
    merged = {
        "allOf": [
            {"type": "object", "required": ["a"], "properties": {"a": {}}},
            {"type": "object", "properties": {"b": {}}},
        ]
    }
    assert "optional" in body_path_problem(merged, "response.body.b")


def _described_step(name, query=None):
    from spec_run_graph.domain.ir import StepIR

    return StepIR(name=name, endpoint="GET /api/items", request={"query": query or {}}, expected_status=200)


def test_step_announcing_a_query_parameter_must_send_it():
    from spec_run_graph.domain.ir import _described_but_unsent

    documented = {"page_size", "page"}
    assert _described_but_unsent(_described_step("Read a page with page_size 2"), documented) == ["page_size"]
    assert _described_but_unsent(_described_step("Read a page with page_size 2", {"page_size": 2}), documented) == []
    assert _described_but_unsent(_described_step("Read without page_size"), documented) == []
    assert _described_but_unsent(_described_step("Default page_size applies"), documented) == []


def _dedup_case(test_id, endpoint="GET /a"):
    from spec_run_graph.domain.ir import StepIR, TestCaseIR

    step = StepIR(name="s", endpoint=endpoint, expected_status=200)
    return TestCaseIR(id=test_id, title=test_id, type="positive", steps=[step])


def _accepted(found, cases, candidates):
    from spec_run_graph.agents.dedup import Duplicate, accepted

    paths = {"GET /a": "/a", "GET /b": "/b"}
    return accepted([Duplicate(drop=d, keep=k, reason="r") for d, k in found], cases, candidates, set(paths), paths)


def test_dedup_only_drops_new_cases_for_a_case_on_the_same_endpoint():
    ids = [f"TC-A-00{i}" for i in range(1, 7)]
    cases = [_dedup_case(i) for i in ids[:5]] + [_dedup_case(ids[5], "GET /b")]
    assert [d.drop for d in _accepted([(ids[1], ids[0])], cases, set(ids))] == [ids[1]]
    assert _accepted([(ids[0], ids[1])], cases, {ids[1]}) == []  # an older case is never dropped
    assert _accepted([(ids[5], ids[0])], cases, set(ids)) == []  # no shared endpoint
    assert _accepted([(ids[1], "TC-X-999")], cases, set(ids)) == []  # kept case must exist


def test_dedup_is_capped_and_never_drops_the_kept_case():
    ids = [f"TC-A-00{i}" for i in range(1, 6)]
    cases = [_dedup_case(i) for i in ids]
    capped = _accepted([(ids[1], ids[0]), (ids[2], ids[0]), (ids[3], ids[0])], cases, set(ids))
    assert len(capped) == 2
    chained = _accepted([(ids[1], ids[2]), (ids[2], ids[0])], cases, set(ids))
    assert ids[2] not in {d.keep for d in chained}


async def test_dedup_review_applies_the_guards_to_the_model_answer():
    from fake_llm import tool_model
    from spec_run_graph.agents.dedup import review

    ids = [f"TC-A-00{i}" for i in range(1, 6)]
    answer = {
        "duplicates": [
            {"drop": ids[1], "keep": ids[0], "reason": "same"},
            {"drop": "TC-X-1", "keep": ids[0], "reason": "x"},
        ]
    }
    found = await review("A", [_dedup_case(i) for i in ids], set(ids), {"GET /a"}, {"GET /a": "/a"}, tool_model(answer))
    assert [d.drop for d in found] == [ids[1]]


def test_a_generated_recipe_input_is_one_value_across_the_recipe():
    from spec_run_graph.domain.ir import CaseDraft, StepIR
    from spec_run_graph.domain.recipes import RecipeIR, expand_recipes

    recipe = RecipeIR.model_validate(
        {
            "name": "r",
            "description": "d",
            "inputs": {"m": {"description": "maker"}},
            "outputs": {},
            "steps": [
                {
                    "name": "a",
                    "endpoint": "POST /p",
                    "request": {"payload": {"m": "{{ $input.m }}"}},
                    "expected_status": 201,
                },
                {
                    "name": "b",
                    "endpoint": "GET /p",
                    "request": {"query": {"m": "{{ $input.m }}"}},
                    "expected_status": 200,
                },
            ],
        }
    )
    use = StepIR(name="u", type="recipe", recipe="r", inputs={"m": "qa-{{ $faker.uuid }}"})
    case, errors = expand_recipes(CaseDraft(title="c", type="positive", steps=[use]), {"r": recipe})
    assert not errors
    assert case.steps[0].request.payload["m"] == case.steps[1].request.query["m"]
    assert "$var." in case.steps[1].request.query["m"]
    assert "qa-{{ $faker.uuid }}" in case.variables.values()


def test_a_lookup_cannot_repeat_a_faker_template():
    from spec_run_graph.domain.ir import CaseDraft, StepIR, ValidationContext, validate_case

    steps = [
        StepIR(name="a", endpoint="POST /p", request={"payload": {"m": "qa-{{ $faker.uuid }}"}}, expected_status=201),
        StepIR(name="b", endpoint="GET /p", request={"query": {"m": "qa-{{ $faker.uuid }}"}}, expected_status=200),
    ]
    errors = validate_case(
        CaseDraft(title="c", type="positive", steps=steps), ValidationContext({"POST /p": "/p", "GET /p": "/p"})
    )
    assert any("generates a NEW value" in e for e in errors)


def test_text_fixture_values_become_numbers_in_numeric_fields():
    from spec_run_graph.engines.bruno.compiler import _as_schema_type

    assert _as_schema_type("500.00", {"type": "number"}) == 500.0
    assert _as_schema_type("25", {"type": "integer"}) == 25
    assert _as_schema_type("abc", {"type": "number"}) == "abc"
    assert _as_schema_type("25", {"type": "string"}) == "25"

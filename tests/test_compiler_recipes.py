"""Deterministic compiler, contract facts, setup recipes and lessons (framework improvements 2026-10-27)."""

import shutil
import subprocess

import pytest

from spec_run_graph.agents.lessons import LessonDeps, LessonDraft, lesson_problems
from spec_run_graph.agents.recipes import recipe_errors
from spec_run_graph.domain.ir import (
    CaseDraft,
    RequestIR,
    StepIR,
    TestCaseIR,
    ValidationContext,
    normalize,
    validate_case,
)
from spec_run_graph.domain.recipes import RecipeIR, expand_recipes
from spec_run_graph.engines.bruno.compiler import HELPERS, compile_case
from spec_run_graph.engines.bruno.naming import plan_files
from spec_run_graph.engines.bruno.scaffold import dump_yaml
from spec_run_graph.engines.bruno.validate import validate_files
from spec_run_graph.specs.contract import assertable_paths, required_body_fields

LOGIN = {"type": "login", "login_endpoint": "POST /api/admin/login", "login_body": {}, "token_path": "data.token"}
MODULE = {
    "code": "ORD",
    "folder": "orders",
    "base_path": "",
    "default_headers": {},
    "actors": {"admin": LOGIN},
    "fixtures": {"brand": "dell", "amount": 500.0, "cats": ["laptop"]},
}
USER_SCHEMA = {
    "type": "object",
    "required": ["username", "password", "role", "balance"],
    "properties": {
        "username": {"type": "string"},
        "password": {"type": "string"},
        "role": {"enum": ["admin", "user"]},
        "balance": {"type": "number"},
    },
}
ERROR = {
    "type": "object",
    "required": ["success", "error"],
    "properties": {
        "success": {"type": "boolean"},
        "requestId": {"type": "string"},
        "error": {
            "type": "object",
            "required": ["code"],
            "properties": {"code": {"type": "string"}, "details": {"type": "array", "items": {"type": "object"}}},
        },
    },
}
CREATED = {
    "type": "object",
    "required": ["data"],
    "properties": {"data": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}},
}
ITEM = {"type": "object", "required": ["id", "quantity"], "properties": {"id": {}, "quantity": {"type": "integer"}}}
ENDPOINTS = {
    "POST /api/admin/users": {"request_schema": USER_SCHEMA, "responses": {"201": {"schema": CREATED}}},
    "POST /api/user/login": {"request_schema": None, "responses": {"200": {"schema": {}}}},
    "POST /api/user/orders": {
        "request_schema": {"type": "object", "properties": {"product_id": {"type": "integer"}, "qty": {}}},
        "responses": {"201": {"schema": CREATED}, "400": {"schema": ERROR}},
    },
    "GET /api/user/products": {
        "request_schema": None,
        "responses": {
            "200": {
                "schema": {
                    "type": "object",
                    "required": ["data"],
                    "properties": {"data": {"type": "array", "items": ITEM}},
                }
            }
        },
    },
    "DELETE /api/admin/users/{id}": {"request_schema": None, "responses": {"204": {}}},
}
PATHS = {k: k.split(" ", 1)[1] for k in ENDPOINTS}


def ctx(**kw):
    return ValidationContext(
        endpoint_paths=PATHS,
        actors={"admin"},
        fixtures={"brand", "amount", "cats"},
        env_vars=set(),
        response_schemas={k: {s: r for s, r in v["responses"].items()} for k, v in ENDPOINTS.items()},
        request_schemas={k: v["request_schema"] for k, v in ENDPOINTS.items()},
        **kw,
    )


# contract facts -------------------------------------------------------------------------------


def test_contract_facts():
    assert required_body_fields(USER_SCHEMA) == ["username", "password", "role", "balance"]
    assert assertable_paths(ERROR) == ["response.body.success", "response.body.error", "response.body.error.code"]
    item_list = ENDPOINTS["GET /api/user/products"]["responses"]["200"]["schema"]
    assert assertable_paths(item_list) == [
        "response.body.data",
        "response.body.data[0].id",
        "response.body.data[0].quantity",
    ]


def test_object_key_on_list_is_rejected():
    case = CaseDraft(
        title="t",
        type="positive",
        actor="admin",
        steps=[
            StepIR(
                name="list",
                endpoint="GET /api/user/products",
                expected_status=200,
                extract={"pid": "response.body.data.id"},
            )
        ],
    )
    errors = validate_case(normalize(case), ctx())
    assert any("is a list" in e for e in errors)


# compiler -------------------------------------------------------------------------------------


def compiled_case() -> TestCaseIR:
    return TestCaseIR(
        id="TC-ORD-001",
        title="order flow",
        type="positive",
        actor="admin",
        tags=["orders"],
        variables={"username": "qa_{{ $faker.uuid }}"},
        parameters=[{"name": "one", "qty": 1, "code": "OK"}, {"name": "two", "qty": 2, "code": "OK"}],
        steps=[
            StepIR(
                name="create user",
                endpoint="POST /api/admin/users",
                request={
                    "payload": {"username": "{{ $var.username }}", "password": "p", "role": "user", "balance": 10}
                },
                expected_status=201,
                extract={"uid": "response.body.data.id"},
            ),
            StepIR(
                name="find product",
                endpoint="GET /api/user/products",
                request={"query": {"manufacturer": "{{ $fixture.brand }}", "page_size": 100}},
                expected_status=200,
                extract={"pid": "response.body.data[quantity=5].id"},
                assertions=[{"check": "response.body.data[0].quantity >= 0"}],
            ),
            StepIR(name="wait", type="delay", delay_ms=50),
            StepIR(
                name="order",
                endpoint="POST /api/user/orders",
                request={
                    "headers": {"Authorization": "Bearer {{ $extract.uid }}"},
                    "payload": {
                        "product_id": "{{ $extract.pid }}",
                        "qty": "{{ $param.qty }}",
                        "amount": "{{ $fixture.amount }}",
                        "cats": "{{ $fixture.cats }}",
                    },
                },
                expected_status=201,
                assertions=[
                    {"check": "response.body.data.id exists"},
                    {"check": "response.body.data.id matches '(?i)^[a-f0-9-]+$'"},
                    {"check": "response.body.data.id != {{ $extract.uid }}", "message": "new id for {{ $param.qty }}"},
                ],
            ),
        ],
        cleanup=[StepIR(name="delete user", endpoint="DELETE /api/admin/users/{{ $extract.uid }}")],
    )


def test_compiler_output_is_valid_and_typed():
    case = compiled_case()
    planned = plan_files(case)
    docs = compile_case(case, MODULE, planned, ENDPOINTS)
    texts = {n: dump_yaml(d) for n, d in docs.items()}
    errors, _ = validate_files(case, MODULE, planned, texts, set())
    assert errors == []
    first = docs[planned[0].filename]
    assert first["http"]["auth"] == {"type": "bearer", "token": "{{auth_admin}}"}
    assert "TC_ORD_001__var_username" in first["runtime"]["scripts"][0]["code"]
    order_b = docs["TC-ORD-001b-step3-order.yml"]
    assert (  # integer extract unquoted, fixtures inlined with their JSON type
        order_b["http"]["body"]["data"]
        == '{"product_id": {{TC_ORD_001__pid}}, "qty": 2, "amount": 500.0, "cats": ["laptop"]}'
    )
    assert 'new RegExp("^[a-f0-9-]+$", "i")' in order_b["runtime"]["scripts"][-1]["code"]
    assert "auth" not in order_b["http"]  # explicit Authorization header wins
    assert "await bru.sleep(50);" in order_b["runtime"]["scripts"][0]["code"]
    assert '"new id for 2"' in order_b["runtime"]["scripts"][-1]["code"]
    products = docs["TC-ORD-001a-step2-find-product.yml"]
    assert products["http"]["url"] == "{{baseUrl}}/api/user/products?manufacturer=dell&page_size=100"
    assert '{ f: "quantity", v: 5 }' in products["runtime"]["scripts"][0]["code"]
    cleanup = docs["TC-ORD-001a-step4-delete-user.yml"]
    assert "runtime" not in cleanup and cleanup["http"]["url"].endswith("/users/{{TC_ORD_001__uid}}")


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_compiler_helpers_behave():
    script = (
        HELPERS
        + """
const body = {data: [{id: 1, model: "A", n: 3}, {id: 2, model: "B", n: 5}]};
const out = [];
const t = (fn) => { try { fn(); out.push("ok"); } catch (e) { out.push(e.message); } };
t(() => __srg.check(__srg.walk(body, ["data", {f: "model", v: "B"}, "id"]), "==", "2"));
t(() => __srg.check(__srg.walk(body, ["data", 0, "n"]), ">", 4));
t(() => __srg.check(__srg.walk(body, ["data", "length"]), "==", 2));
t(() => __srg.check(__srg.walk(body, ["missing", "x"]), "not exists", undefined));
t(() => __srg.check("PAID", "in", ["PAID", "FAILED"]));
console.log(JSON.stringify(out));
"""
    )
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == '["ok","expected > 4, got 3","ok","ok","ok"]'


# recipes --------------------------------------------------------------------------------------

RECIPE = RecipeIR(
    name="dedicated_customer",
    description="fresh user with a balance; outputs user_id",
    inputs={"balance": {"description": "start balance", "default": 100}},
    variables={"username": "qa_{{ $faker.uuid }}"},
    steps=[
        StepIR(
            name="create",
            actor="admin",
            endpoint="POST /api/admin/users",
            request={
                "payload": {
                    "username": "{{ $var.username }}",
                    "password": "p",
                    "role": "user",
                    "balance": "{{ $input.balance }}",
                }
            },
            expected_status=201,
            extract={"user_id": "response.body.data.id", "tmp": "response.body.data.id"},
        )
    ],
    cleanup=[StepIR(name="delete", actor="admin", endpoint="DELETE /api/admin/users/{{ $extract.user_id }}")],
    outputs={"user_id": "id of the new user"},
)


def test_recipe_expansion_renames_and_keeps_types():
    case = CaseDraft(
        title="two buyers",
        type="positive",
        actor="admin",
        steps=[
            StepIR(name="Buyer", type="recipe", recipe="dedicated_customer", inputs={"balance": 5000}, alias="buyer"),
            StepIR(name="Other", type="recipe", recipe="dedicated_customer"),
            StepIR(
                name="list",
                endpoint="GET /api/user/products",
                expected_status=200,
                request={"headers": {"X-A": "{{ $extract.buyer_user_id }}", "X-B": "{{ $extract.user_id }}"}},
            ),
        ],
    )
    expanded, errors = expand_recipes(case, {"dedicated_customer": RECIPE})
    assert errors == []
    s1, s2 = expanded.steps[0], expanded.steps[1]
    assert s1.request.payload["balance"] == 5000 and s2.request.payload["balance"] == 100
    assert set(s1.extract) == {"buyer_user_id", "buyer_1_tmp"} and set(s2.extract) == {
        "user_id",
        "dedicated_customer_2_tmp",
    }
    assert s1.request.payload["username"] == "{{ $var.buyer_1_username }}"
    assert len(expanded.cleanup) == 2 and expanded.cleanup[0].endpoint.endswith("{{ $extract.buyer_user_id }}")
    assert validate_case(normalize(expanded), ctx()) == []


def test_recipe_errors():
    assert recipe_errors(RECIPE, ctx()) == []
    broken = RECIPE.model_copy(
        update={"steps": [RECIPE.steps[0].model_copy(update={"request": RequestIR(payload={"username": "x"})})]}
    )
    assert any("misses fields the contract requires" in e for e in recipe_errors(broken, ctx()))
    _, errors = expand_recipes(
        CaseDraft(title="t", type="positive", steps=[StepIR(name="x", type="recipe", recipe="nope")]), {}
    )
    assert "unknown recipe" in errors[0]


# lessons --------------------------------------------------------------------------------------


def test_lesson_validation():
    deps = LessonDeps(existing=["old lesson"], endpoints={"GET /a"}, n_fixes=2)
    assert lesson_problems(LessonDraft(text="Never assert requestId.", fixes=[0, 1]), deps) == []
    bad = LessonDraft(text="TC-ORD-001 broke", endpoints=["GET /b"], fixes=[5], confirms="nope")
    assert len(lesson_problems(bad, deps)) == 4


def test_lesson_contradicting_the_contract_is_rejected():
    deps = LessonDeps(existing=[], endpoints={"POST /p"}, n_fixes=1, required_fields={"POST /p": ["price", "model"]})
    bad = LessonDraft(text="On POST /p only model is required; price is optional and defaults to 0.", fixes=[0])
    assert any("marks ['price'" in p for p in lesson_problems(bad, deps))
    ok = LessonDraft(text="POST /p needs price and model.", endpoints=["POST /p"], fixes=[0])
    assert lesson_problems(ok, deps) == []


def test_healer_keeps_required_field_negative_steps():
    from spec_run_graph.agents.healer import HealDeps, guard_errors

    def case(payload):
        return CaseDraft(
            title="t",
            type="negative",
            actor="admin",
            steps=[
                StepIR(
                    name="create", endpoint="POST /api/admin/users", request={"payload": payload}, expected_status=400
                )
            ],
        )

    orig = case({"username": "x", "password": "p", "role": "user"})  # omits required balance
    healed = case({"username": "x", "password": "p", "balance": 1})  # now omits role instead
    errors = guard_errors(normalize(healed), HealDeps(original=normalize(orig), ctx=ctx(), documented_statuses={}))
    assert any("keep the negative step" in e and "balance" in e for e in errors)


def test_contract_violation_is_diagnosed_as_application_bug():
    from spec_run_graph.agents.diagnostics import FailedExecution, preclassify
    from spec_run_graph.pipeline.stages import contract_violations
    from spec_run_graph.specs.models import EndpointDraft

    c = TestCaseIR(
        id="TC-ORD-009",
        title="missing balance",
        type="negative",
        actor="admin",
        steps=[
            StepIR(
                name="create",
                endpoint="POST /api/admin/users",
                request={"payload": {"username": "x", "password": "p", "role": "user"}},
                expected_status=400,
            )
        ],
    )
    f = FailedExecution(
        "e",
        "r",
        c.id,
        c.title,
        0,
        "FAILED",
        201,
        None,
        [{"file": "orders/TC-ORD-009.yml", "assertion": "res.status eq 400", "error": "expected 201 to equal 400"}],
        {},
        ["POST /api/admin/users"],
        [],
    )
    drafts = {
        "POST /api/admin/users": EndpointDraft(method="POST", path="/api/admin/users", request_schema=USER_SCHEMA)
    }
    f.contract_violations = contract_violations(c, f, drafts)
    assert f.contract_violations and "balance" in f.contract_violations[0]
    assert preclassify(f).category == "APPLICATION_BUG"


def test_list_filter_on_paginated_endpoint_needs_max_page_size():
    def case(query):
        return CaseDraft(
            title="t",
            type="positive",
            actor="admin",
            steps=[
                StepIR(
                    name="find",
                    endpoint="GET /api/user/products",
                    request={"query": query},
                    expected_status=200,
                    extract={"pid": "response.body.data[quantity=5].id"},
                )
            ],
        )

    limits = {"GET /api/user/products": ("page_size", 100)}
    assert any("paginated list" in e for e in validate_case(normalize(case({})), ctx(page_limits=limits)))
    assert validate_case(normalize(case({"page_size": 100})), ctx(page_limits=limits)) == []


def test_non_validation_negative_step_needs_a_complete_body():
    def case(payload, status):
        return CaseDraft(
            title="t",
            type="negative",
            actor="admin",
            steps=[
                StepIR(name="s", endpoint="POST /api/admin/users", request={"payload": payload}, expected_status=status)
            ],
        )

    errors = validate_case(normalize(case(None, 401)), ctx())
    assert any("sends no body" in e for e in errors)
    assert validate_case(normalize(case({}, 401)), ctx()) == []  # deliberate empty object: allowed
    assert validate_case(normalize(case({"username": "x"}, 400)), ctx()) == []  # body validation test: fine


def test_state_change_must_not_use_looked_up_shared_entity():
    case = CaseDraft(
        title="t",
        type="positive",
        actor="admin",
        steps=[
            StepIR(
                name="pick",
                endpoint="GET /api/user/products",
                request={"query": {"page_size": 100}},
                expected_status=200,
                extract={"pid": "response.body.data[0].id"},
            ),
            StepIR(
                name="order",
                endpoint="POST /api/user/orders",
                request={"payload": {"product_id": "{{ $extract.pid }}", "qty": 1}},
                expected_status=201,
            ),
        ],
    )
    assert any("looked up from shared data" in e for e in validate_case(normalize(case), ctx()))


def test_status_sets_compile_to_an_in_assertion_and_must_be_documented():
    def case(status):
        return CaseDraft(
            title="t",
            type="security",
            actor="admin",
            steps=[
                StepIR(
                    name="order",
                    endpoint="POST /api/user/orders",
                    expected_status=status,
                    request=RequestIR(payload={"product_id": 1}),
                )
            ],
        )

    assert not validate_case(normalize(case("201|400")), ctx())
    assert any("not documented" in e for e in validate_case(normalize(case("400|403")), ctx()))
    tc = TestCaseIR.model_validate({**normalize(case("201|400")).model_dump(), "id": "TC-ORD-001"})
    planned = plan_files(tc)
    docs = compile_case(tc, MODULE, planned, ENDPOINTS)
    asserts = docs[planned[0].filename]["runtime"]["assertions"]
    assert asserts == [{"expression": "res.status", "operator": "in", "value": "[201, 400]"}]
    errors, _ = validate_files(tc, MODULE, planned, {p.filename: dump_yaml(docs[p.filename]) for p in planned}, set())
    assert not errors

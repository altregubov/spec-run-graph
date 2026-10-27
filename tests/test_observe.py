"""Generation-time quality: SUT observations, planner rule statistics, workflow-driven recipe prompt."""

import httpx

from spec_run_graph.agents.planner import rule_key
from spec_run_graph.agents.recipes import build_prompt as recipe_prompt
from spec_run_graph.domain.analysis import ActorDef, AnalyzedRequirement, AuthDef, ModuleProposal
from spec_run_graph.knowledge.db import KNOWLEDGE, WIPE_ORDER
from spec_run_graph.pipeline import observe as obs


def test_trim_keeps_samples_small():
    body = {"data": [{"model": "x" * 200, "n": i} for i in range(10)]}
    t = obs.trim(body)
    assert len(t["data"]) == obs.MAX_ITEMS + 1 and t["data"][-1] == "... 7 more"
    assert t["data"][0]["model"].endswith("...") and len(t["data"][0]["model"]) == obs.MAX_STR + 3


def test_resolve_env_templates():
    assert obs._resolve({"password": "{{ $env.PW }}"}, {"PW": "s"}) == {"password": "s"}
    assert obs._resolve({"password": "{{ $env.PW }}"}, {}) is None


async def test_observe_only_parameterless_gets_as_first_working_actor():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.url.path == "/api/user/login":
            return httpx.Response(200, json={"data": {"token": "tok"}})
        return httpx.Response(200, json={"data": [{"manufacturer": "apple"}]})

    login = AuthDef(
        type="login", login_endpoint="POST /api/user/login", login_body={"username": "a"}, token_path="data.token"
    )
    broken = AuthDef(type="bearer", token="{{ $env.MISSING }}")
    module = ModuleProposal(
        code="CAT",
        folder="catalog",
        domain="Catalog",
        endpoints=["GET /api/user/products", "GET /api/user/products/{id}", "POST /api/user/orders"],
        actors=[ActorDef(name="nobody", auth=broken), ActorDef(name="user_a", auth=login)],
    )
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    httpx.AsyncClient = lambda **kw: real(transport=transport, **kw)  # type: ignore[misc]
    try:
        samples = await obs.observe("http://sut", [module], {})
    finally:
        httpx.AsyncClient = real  # type: ignore[misc]
    assert [(s.endpoint, s.actor, s.http_status) for s in samples] == [("GET /api/user/products", "user_a", 200)]
    assert ("GET", "/api/user/products", "Bearer tok") in calls
    assert not any(m == "POST" and p == "/api/user/orders" for m, p, _ in calls)


def test_rule_key_groups_errors_of_the_same_rule():
    a = rule_key("step 2 (Buy): product_id={{ $extract.cheapest }} is an entity looked up from shared data; x")
    b = rule_key("step 5 (Order): product_id={{ $extract.laptop }} is an entity looked up from shared data; x")
    assert a == b and "step" not in a


def test_reset_keeps_lessons_by_default():
    assert set(KNOWLEDGE) <= set(WIPE_ORDER) and "Observation" in WIPE_ORDER and "Observation" not in KNOWLEDGE


def test_recipe_prompt_lists_workflows_and_observations():
    req = AnalyzedRequirement(req_id="PRD-ORD-1", spec="prd", title="Place order", description="buy visible product")
    text = recipe_prompt("shop", {}, {}, "contracts", requirements=[req], observed="GET /x as a -> 200: {}")
    assert "PRD-ORD-1" in text and "Observed data" in text


def test_samples_keep_only_guaranteed_fields():
    body = {"success": True, "requestId": "r", "pagination": {"page": 1}, "data": [{"id": 1, "model": "m", "x": 2}]}
    paths = ["response.body.success", "response.body.data", "response.body.data[0].id", "response.body.data[0].model"]
    assert obs.only_paths(body, paths) == {"success": True, "data": [{"id": 1, "model": "m"}]}


def test_modules_get_the_actors_their_recipes_need():
    from spec_run_graph.domain.ir import StepIR
    from spec_run_graph.domain.recipes import RecipeIR
    from spec_run_graph.pipeline.stages import add_recipe_actors

    admin = ActorDef(name="admin", auth=AuthDef(type="none"))
    user = ActorDef(name="user_a", auth=AuthDef(type="none"))
    staff = ModuleProposal(code="ADM", folder="adm", domain="Admin", endpoints=[], actors=[admin])
    shop = ModuleProposal(code="CUS", folder="cus", domain="Customer", endpoints=[], actors=[user])
    recipe = RecipeIR(
        name="buyer",
        description="d",
        steps=[StepIR(name="create", endpoint="POST /u", actor="admin"), StepIR(name="login", endpoint="POST /l")],
        outputs={},
    )
    changed = add_recipe_actors([staff, shop], {"buyer": recipe})
    assert changed == [shop] and [a.name for a in shop.actors] == ["user_a", "admin"]
    assert add_recipe_actors([staff, shop], {"buyer": recipe}) == []

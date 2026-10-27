"""Login actors: credentials from the spec, token fetched by the collection at run time."""

import yaml

from spec_run_graph.agents.analyzer import AnalyzerDeps, _validate
from spec_run_graph.agents.diagnostics import FailedExecution, preclassify
from spec_run_graph.domain.analysis import ActorDef, AuthDef, ModuleProposal, SpecAnalysis
from spec_run_graph.engines.bruno.scaffold import actor_auth, login_script, write_scaffold
from spec_run_graph.engines.bruno.validate import check_js_syntax, schema_errors

LOGIN = {
    "type": "login",
    "login_endpoint": "POST /api/admin/login",
    "login_body": {"username": "admin", "password": "admin123"},
    "token_path": "data.token",
}
MODULE = {
    "code": "ADM",
    "folder": "admin",
    "domain": "Admin",
    "base_path": "",
    "default_headers": {},
    "preconditions": [],
    "fixtures": {},
    "actors": {"admin": LOGIN, "legacy": {"type": "bearer", "token": "{{ $env.LEGACY_TOKEN }}"}},
}


def test_actor_auth_mapping():
    assert actor_auth({**LOGIN, "_actor": "admin"}) == {"type": "bearer", "token": "{{auth_admin}}"}
    assert actor_auth({"type": "bearer", "token": "{{ $env.LEGACY_TOKEN }}", "_actor": "legacy"}) == {
        "type": "bearer",
        "token": "{{process.env.LEGACY_TOKEN}}",
    }


def test_login_script_is_valid_js_and_schema(tmp_path):
    script = login_script([MODULE])
    assert '"auth_admin"' in script and '"password": "admin123"' in script and "legacy" not in script
    assert check_js_syntax([script]) == []
    env_script = login_script([{**MODULE, "actors": {"u": {**LOGIN, "login_body": {"password": "{{ $env.PW }}"}}}}])
    assert 'bru.getProcessEnv("PW")' in env_script
    write_scaffold(tmp_path, [MODULE], "http://x")
    collection = yaml.safe_load((tmp_path / "opencollection.yml").read_text())
    assert collection["request"]["scripts"][0]["type"] == "before-request"
    assert schema_errors(collection, None) == []
    no_login = tmp_path / "plain"
    write_scaffold(no_login, [{**MODULE, "actors": {}}], "http://x")
    assert "request" not in yaml.safe_load((no_login / "opencollection.yml").read_text())


def test_analyzer_validates_login_actors():
    def analysis(auth):
        return SpecAnalysis(
            summary="s",
            requirements=[],
            modules=[
                ModuleProposal(
                    code="ADM",
                    folder="admin",
                    domain="Admin",
                    endpoints=["POST /api/admin/login"],
                    actors=[ActorDef(name="admin", auth=auth)],
                )
            ],
        )

    deps = AnalyzerDeps(endpoint_keys={"POST /api/admin/login"})
    assert _validate(analysis(AuthDef(**LOGIN)), deps) == []
    errs = " ".join(_validate(analysis(AuthDef(type="login", login_endpoint="POST /nope")), deps))
    assert "not an endpoint key" in errs and "login_body" in errs and "token_path" in errs
    errs = " ".join(_validate(analysis(AuthDef(type="bearer", token="literal-secret")), deps))
    assert "$env.VAR" in errs


def test_missing_env_var_is_diagnosed_by_rule():
    f = FailedExecution(
        execution_id="e",
        run_id="r",
        test_id="TC-A-001",
        title="t",
        param_index=0,
        status="FAILED",
        http_status=401,
        error_message=None,
        failed_assertions=[{"assertion": "res.status eq 200", "error": "expected 401 to equal 200"}],
        case_ir={},
        endpoints=["GET /a"],
        requirement_ids=[],
        missing_env=["USER_A_TOKEN"],
    )
    d = preclassify(f)
    assert d.category == "ENV_FAILURE" and "USER_A_TOKEN" in d.root_cause and d.confidence > 0.9


def test_invented_env_tokens_are_rejected():
    """Regression: the synthesizer wrote {{process.env.USER_A_TOKEN}} although the actor logs in."""
    from fake_llm import _Renderer
    from helpers import reference_module
    from spec_run_graph.agents.synthesizer import build_prompt
    from spec_run_graph.engines.bruno.naming import plan_files
    from spec_run_graph.engines.bruno.validate import allowed_env_vars, validate_files

    case = reference_module().test_cases[0]
    module = {
        "code": "ORD",
        "folder": "orders",
        "domain": "Orders",
        "base_path": "",
        "default_headers": {},
        "preconditions": [],
        "fixtures": {"active_sku": "SKU-ACTIVE-1"},
        "actors": {"customer_standard": {**LOGIN, "login_endpoint": "POST /v1/login"}},
    }
    planned = plan_files(case)
    prompt = build_prompt(case, module, planned, {})
    assert '"auth": {"type": "bearer", "token": "{{auth_customer_standard}}"}' in prompt
    files = {e["filename"]: _Renderer(prompt).render(e) for e in _Renderer(prompt).plan}
    assert validate_files(case, module, planned, files, set())[0] == []
    name = planned[0].filename
    files[name] = files[name].replace("{{auth_customer_standard}}", "{{process.env.USER_A_TOKEN}}")
    errors = validate_files(case, module, planned, files, set())[0]
    assert any("process.env.USER_A_TOKEN" in e for e in errors)
    assert allowed_env_vars(case, {**module, "actors": {"x": {"type": "bearer", "token": "{{ $env.X_TOKEN }}"}}}) == {
        "X_TOKEN"
    }


def test_step_actor_is_validated():
    from helpers import orders_specs, reference_module
    from spec_run_graph.domain.ir import ValidationContext, validate_case

    case = reference_module().test_cases[0].model_copy(deep=True)
    paths = {e.key: e.path for s in orders_specs() for e in s.endpoints}
    ctx = ValidationContext(paths, {"PRD-ORD-42"}, {"customer_standard", "admin"}, {"active_sku"})
    case.steps[0].actor = "admin"
    assert validate_case(case, ctx) == []
    case.steps[0].actor = "ghost"
    assert any("unknown actor 'ghost'" in e for e in validate_case(case, ctx))


def test_undeclared_env_in_ir_is_rejected_and_diagnosed_as_test_defect():
    """Regression: the planner invented {{ $env.SEED_SAMSUNG_GALAXY_S24_ID }} for a seeded product id."""
    from helpers import orders_specs, reference_module
    from spec_run_graph.domain.ir import ValidationContext, declared_env_vars, validate_case

    case = reference_module().test_cases[0].model_copy(deep=True)
    case.steps[0].request.payload["items"][0]["sku"] = "{{ $env.SEED_SAMSUNG_GALAXY_S24_ID }}"
    paths = {e.key: e.path for s in orders_specs() for e in s.endpoints}
    ctx = ValidationContext(
        paths,
        {"PRD-ORD-42"},
        {"customer_standard"},
        {"active_sku"},
        env_vars=declared_env_vars(
            {"legacy": {"type": "bearer", "token": "{{ $env.LEGACY_TOKEN }}"}}, {"sku": {"env": "SKU_FROM_ENV"}}
        ),
    )
    assert ctx.env_vars == {"LEGACY_TOKEN", "SKU_FROM_ENV"}
    errs = " ".join(validate_case(case, ctx))
    assert "$env.SEED_SAMSUNG_GALAXY_S24_ID is not a declared environment variable" in errs
    case.steps[0].request.payload["items"][0]["sku"] = "{{ $env.SKU_FROM_ENV }}"
    assert validate_case(case, ctx) == []

    base = dict(
        execution_id="e",
        run_id="r",
        test_id="TC-O-023",
        title="t",
        param_index=0,
        status="FAILED",
        http_status=400,
        error_message=None,
        failed_assertions=[],
        case_ir={},
        endpoints=[],
        requirement_ids=[],
    )
    invented = preclassify(FailedExecution(**base, missing_env=["SEED_X_ID"], undeclared_env=["SEED_X_ID"]))
    assert invented.category == "TEST_DEFECT" and "SEED_X_ID" in invented.suggested_fix
    declared = preclassify(FailedExecution(**base, missing_env=["LEGACY_TOKEN"], undeclared_env=[]))
    assert declared.category == "ENV_FAILURE"

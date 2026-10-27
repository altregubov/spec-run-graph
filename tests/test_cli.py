"""CLI options must reach the pipeline state (regression: --max-llm/--group were parsed but ignored)."""

from typer.testing import CliRunner

from spec_run_graph import cli


def capture(monkeypatch):
    seen = {}

    async def fake_pipeline(state):
        seen["state"] = state
        return state

    monkeypatch.setattr(cli, "_pipeline", fake_pipeline)
    return seen


def test_diagnose_options_reach_state(monkeypatch):
    seen = capture(monkeypatch)
    result = CliRunner().invoke(
        cli.app, ["diagnose", "run-1", "--max-llm", "80", "--no-group", "--force", "--no-rerun"]
    )
    assert result.exit_code == 0, result.output
    s = seen["state"]
    assert (s.start, s.stop, s.diagnose_run_id) == ("diagnose", "diagnose", "run-1")
    assert (s.max_llm_diagnoses, s.group_failures, s.force, s.rerun_failures) == (80, False, True, False)


def test_heal_and_plan_options_reach_state(monkeypatch):
    seen = capture(monkeypatch)
    assert CliRunner().invoke(cli.app, ["heal", "--max-cases", "200", "--no-verify"]).exit_code == 0
    assert (seen["state"].max_heal_cases, seen["state"].heal_verify) == (200, False)
    assert CliRunner().invoke(cli.app, ["plan", "--from", "a.json", "--refresh-only"]).exit_code == 0
    assert seen["state"].refresh_only is True and seen["state"].start == "plan"
    assert CliRunner().invoke(cli.app, ["test", "--heal"]).exit_code == 0
    assert seen["state"].stop == "heal"


def test_pipeline_fails_fast_when_gel_is_down(monkeypatch):
    """Regression: Gel/MinIO were down after a Docker restart; srg go spent an LLM call, then crashed."""
    from spec_run_graph.config import Settings

    down = Settings(gel_dsn="gel://admin@127.0.0.1:1/main", s3_endpoint_url="http://127.0.0.1:1")
    orig = cli.GraphDependencies
    monkeypatch.setattr(cli, "GraphDependencies", lambda console: orig(settings=down, console=console))
    called = {"pipeline": False}

    async def no_pipeline(state, deps):
        called["pipeline"] = True

    monkeypatch.setattr(cli, "run_pipeline", no_pipeline)
    result = CliRunner().invoke(cli.app, ["build"])
    assert result.exit_code == 2 and not called["pipeline"]
    assert "Gel" in result.output and "docker compose up -d" in result.output

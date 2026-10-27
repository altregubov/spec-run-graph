import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os  # noqa: E402

# Tests must not depend on the developer's .env tuning (environment variables win over .env).
os.environ["SRG_OUTPUT_MODE"] = "tool"
os.environ["SRG_RECIPES"] = "false"  # recipe design would consume the scripted test-model answers
os.environ["SRG_OBSERVE"] = "false"  # no live GET samples in tests (the prompt stays deterministic)
os.environ["SRG_PLAN_GAP_FILL"] = "false"  # one planner pass per chunk (fake models answer every call the same)

import pytest  # noqa: E402

from spec_run_graph import llm  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_llm_fallback():
    """The prompted-output fallback is sticky per process; isolate it between tests."""
    llm._fallback_to_prompted = False
    yield
    llm._fallback_to_prompted = False

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from spec_run_graph import llm  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_llm_fallback():
    """The prompted-output fallback is sticky per process; isolate it between tests."""
    llm._fallback_to_prompted = False
    yield
    llm._fallback_to_prompted = False

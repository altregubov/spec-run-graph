"""The single LLM used by every agent: SRG_MODEL on OpenCode Go (OpenAI-compatible chat completions)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai import Agent, AgentRunResult, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from spec_run_graph import __version__
from spec_run_graph.config import Settings, get_settings

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
_SESSION_ID = str(uuid.uuid4())
log = logging.getLogger(__name__)


class LLMNotConfigured(RuntimeError):
    pass


@lru_cache
def build_model() -> OpenAIChatModel:
    settings = get_settings()
    if not settings.opencode_api_key:
        raise LLMNotConfigured("OPENCODE_API_KEY is not set - copy .env.example to .env and fill it in")
    client = AsyncOpenAI(
        base_url=settings.opencode_base_url,
        api_key=settings.opencode_api_key,
        timeout=600,
        max_retries=2,
        default_headers={
            # OpenCode recommends an identifying User-Agent and a stable session id per conversation.
            "User-Agent": f"spec-run-graph/{__version__}",
            "x-opencode-session": _SESSION_ID,
        },
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIChatModel(settings.srg_model, provider=provider)


def model_name() -> str:
    return get_settings().srg_model


_fallback_to_prompted = False


@dataclass
class LLMUsage:
    """Process-wide LLM accounting, printed after each CLI command."""

    calls: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    fallbacks: int = 0
    by_agent: dict[str, list[float]] = field(default_factory=dict)  # name -> [calls, seconds]

    def add(self, agent: str, seconds: float, result: AgentRunResult | None) -> None:
        self.calls += 1
        self.seconds += seconds
        if result is not None:
            u = result.usage() if callable(result.usage) else result.usage
            self.input_tokens += u.input_tokens or 0
            self.output_tokens += u.output_tokens or 0
        row = self.by_agent.setdefault(agent, [0, 0.0])
        row[0] += 1
        row[1] += seconds


USAGE = LLMUsage()


def output_spec(output_type: Any, settings: Settings | None = None) -> Any:
    """Wrap an output type according to SRG_OUTPUT_MODE (or the automatic prompted fallback)."""
    mode = (settings or get_settings()).srg_output_mode
    if mode == "prompted" or _fallback_to_prompted:
        return PromptedOutput(output_type)
    if mode == "native":
        return NativeOutput(output_type)
    return ToolOutput(output_type)


def request_settings(thinking: bool = True, session: str | None = None) -> dict[str, Any]:
    """Per-conversation model settings.

    - Own `x-opencode-session` per conversation: OpenCode routes by session, so parallel calls that share
      one session are processed one after another.
    - `thinking: disabled` skips hidden reasoning tokens (honoured by reasoning models on OpenCode Go).
    """
    s = get_settings()
    mode = s.srg_thinking
    use_thinking = thinking if mode == "auto" else mode == "on"
    settings: dict[str, Any] = {"extra_headers": {"x-opencode-session": session or str(uuid.uuid4())}}
    if not use_thinking:
        settings["extra_body"] = {"thinking": {"type": "disabled"}}
    if s.srg_max_tokens:
        settings["max_tokens"] = s.srg_max_tokens
    return settings


def _unsupported_structured_output(exc: ModelHTTPError) -> bool:
    text = str(exc.body).lower()
    return exc.status_code == 400 and any(k in text for k in ("tool_choice", "response_format", "json_schema"))


async def _timed_run(make_agent: Callable[[Any], Agent], spec: Any, prompt: str, **kwargs: Any) -> AgentRunResult:
    agent = make_agent(spec)
    start = time.monotonic()
    result = None
    try:
        result = await agent.run(prompt, **kwargs)
        return result
    finally:
        USAGE.add(agent.name or "agent", time.monotonic() - start, result)


async def run_structured(
    make_agent: Callable[[Any], Agent], prompt: str, output_type: Any, thinking: bool = True, **kwargs: Any
) -> AgentRunResult:
    """agent.run() with the configured output mode and per-conversation settings. Switches to prompted output
    (for the rest of the process) if the provider rejects forced tool calls / JSON schema, or if the model keeps
    producing invalid structured output.

    make_agent(output_spec) builds the agent (output validators forbid a per-run output_type override)."""
    global _fallback_to_prompted
    kwargs.setdefault("model_settings", request_settings(thinking))
    spec = output_spec(output_type)
    try:
        return await _timed_run(make_agent, spec, prompt, **kwargs)
    except ModelHTTPError as exc:
        if isinstance(spec, PromptedOutput) or not _unsupported_structured_output(exc):
            raise
        reason = f"rejected structured output mode {get_settings().srg_output_mode!r}"
    except UnexpectedModelBehavior as exc:
        # e.g. malformed or truncated tool-call JSON on every retry
        if isinstance(spec, PromptedOutput):
            raise
        reason = str(exc)
    if not _fallback_to_prompted:
        _fallback_to_prompted = True
        log.warning("%s: %s; using prompted output from now on", model_name(), reason)
    USAGE.fallbacks += 1
    return await _timed_run(make_agent, PromptedOutput(output_type), prompt, **kwargs)

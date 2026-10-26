"""The single LLM used by every agent: SRG_MODEL on OpenCode Go (OpenAI-compatible chat completions)."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai import Agent, AgentRunResult, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.exceptions import ModelHTTPError
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


def output_spec(output_type: Any, settings: Settings | None = None) -> Any:
    """Wrap an output type according to SRG_OUTPUT_MODE (or the automatic prompted fallback)."""
    mode = (settings or get_settings()).srg_output_mode
    if mode == "prompted" or _fallback_to_prompted:
        return PromptedOutput(output_type)
    if mode == "native":
        return NativeOutput(output_type)
    return ToolOutput(output_type)


def _unsupported_structured_output(exc: ModelHTTPError) -> bool:
    text = str(exc.body).lower()
    return exc.status_code == 400 and any(k in text for k in ("tool_choice", "response_format", "json_schema"))


async def run_structured(
    make_agent: Callable[[Any], Agent], prompt: str, output_type: Any, **kwargs: Any
) -> AgentRunResult:
    """agent.run() with the configured output mode; switches to prompted output once if the provider
    rejects forced tool calls / JSON schema (e.g. reasoning models in thinking mode).

    make_agent(output_spec) builds the agent (output validators forbid a per-run output_type override)."""
    global _fallback_to_prompted
    spec = output_spec(output_type)
    try:
        return await make_agent(spec).run(prompt, **kwargs)
    except ModelHTTPError as exc:
        if isinstance(spec, PromptedOutput) or not _unsupported_structured_output(exc):
            raise
        if not _fallback_to_prompted:
            _fallback_to_prompted = True
            log.warning(
                "%s rejected structured output mode %r; falling back to prompted output",
                model_name(),
                get_settings().srg_output_mode,
            )
        return await make_agent(PromptedOutput(output_type)).run(prompt, **kwargs)

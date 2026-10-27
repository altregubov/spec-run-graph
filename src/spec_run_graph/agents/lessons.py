"""Lesson Extractor: turns verified heals into generic rules for future planning and healing.

A heal fixes one case; a lesson ("this API returns the request id only in the x-request-id header") keeps
the planner from generating the same defect again. Only VERIFIED heals are used as evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.llm import build_model, output_spec, run_structured

INSTRUCTIONS = """\
You maintain a short list of LESSONS for an AI that designs API tests. You receive test defects that were
fixed and verified against the real API (root cause + the fix), plus the lessons that already exist.
Write a lesson only for a GENERIC, reusable insight about this API or about writing its tests, phrased as an
instruction, e.g. "The API returns the request id only in the x-request-id header; never assert
response.body.requestId." or "POST /api/admin/users needs role and balance; created users start inactive
until ...". Skip fixes that are specific to one test (typos, a wrong step order). A lesson must never contradict the
OpenAPI contract: if the API deviates from it (e.g. accepts a request without a required field), that is an
application bug, not a lesson.
For each lesson give: text (one or two sentences, max 300 characters, no test ids), endpoints (endpoint keys
it is about; empty when it applies everywhere), fixes (indexes of the fixes that support it), and confirms
(the exact text of an existing lesson when the fixes only confirm it - then repeat that text)."""

TEST_ID_RE = re.compile(r"\bTC-[A-Z0-9_]+-\d+")


class LessonDraft(BaseModel):
    text: str
    endpoints: list[str] = Field(default_factory=list)
    fixes: list[int] = Field(default_factory=list)
    confirms: str | None = None


class LessonSet(BaseModel):
    lessons: list[LessonDraft] = Field(default_factory=list)


@dataclass
class LessonDeps:
    existing: list[str]
    endpoints: set[str]
    n_fixes: int
    required_fields: dict[str, list[str]] = field(default_factory=dict)  # endpoint -> contract-required fields
    problems: list[str] = field(default_factory=list)


OPTIONAL_CLAIM_RE = re.compile(r"\boptional\b|not (strictly )?required|\bdefaults? to\b|may be omitted", re.I)


def contradicts_contract(d: LessonDraft, deps: LessonDeps) -> list[str]:
    """Contract-required fields the lesson calls optional (that would teach tests to accept a bug)."""
    claims = [c for c in re.split(r"[;.]\s|, (?:but|while) ", d.text) if OPTIONAL_CLAIM_RE.search(c)]
    scope = d.endpoints or list(deps.required_fields)
    fields = {f for e in scope for f in deps.required_fields.get(e, [])}
    return sorted(f for f in fields if any(re.search(rf"\b{re.escape(f)}\b", c) for c in claims))


def lesson_problems(d: LessonDraft, deps: LessonDeps) -> list[str]:
    errors = []
    if d.confirms is not None and d.confirms not in deps.existing:
        errors.append(f"confirms {d.confirms[:60]!r}... is not an existing lesson text")
    if len(d.text) > 300:
        errors.append(f"lesson too long ({len(d.text)} > 300 chars): {d.text[:60]}...")
    if TEST_ID_RE.search(d.text):
        errors.append(f"lessons must be generic - no test ids: {d.text[:60]}...")
    unknown = sorted(set(d.endpoints) - deps.endpoints)
    if unknown:
        errors.append(f"unknown endpoint keys {unknown}")
    if bad := contradicts_contract(d, deps):
        errors.append(
            f"the contract marks {bad} as required - an API that accepts them missing has a bug: {d.text[:60]}"
        )
    if not d.fixes or any(not 0 <= i < deps.n_fixes for i in d.fixes):
        errors.append(f"fixes must be indexes 0..{deps.n_fixes - 1} of the supporting fixes")
    return errors


def build_agent(output: Any = None) -> Agent[LessonDeps, LessonSet]:
    agent: Agent[LessonDeps, LessonSet] = Agent(
        output_type=output or output_spec(LessonSet),
        instructions=INSTRUCTIONS,
        deps_type=LessonDeps,
        retries=get_settings().srg_llm_retries,
        name="lesson-extractor",
    )

    @agent.output_validator
    def check(ctx: RunContext[LessonDeps], out: LessonSet) -> LessonSet:
        problems = [p for d in out.lessons for p in lesson_problems(d, ctx.deps)]
        if problems and ctx.retry < ctx.max_retries:
            raise ModelRetry("Fix these problems and return all lessons again:\n- " + "\n- ".join(problems))
        return LessonSet(lessons=[d for d in out.lessons if not lesson_problems(d, ctx.deps)])

    return agent


def build_prompt(fixes: list[dict[str, Any]], existing: list[str]) -> str:
    return "\n\n".join(
        [
            "Existing lessons:\n" + ("\n".join(f"- {t}" for t in existing) or "(none)"),
            "Verified fixes:\n" + "\n".join(f"[{i}] {json.dumps(f, default=str)[:1500]}" for i, f in enumerate(fixes)),
        ]
    )


async def extract(
    fixes: list[dict[str, Any]],
    existing: list[str],
    endpoints: set[str],
    model: Model | None = None,
    required_fields: dict[str, list[str]] | None = None,
) -> list[LessonDraft]:
    deps = LessonDeps(existing=existing, endpoints=endpoints, n_fixes=len(fixes), required_fields=required_fields or {})
    result = await run_structured(
        build_agent, build_prompt(fixes, existing), LessonSet, thinking=False, deps=deps, model=model or build_model()
    )
    return result.output.lessons

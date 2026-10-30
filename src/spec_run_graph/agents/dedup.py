"""Duplicate review: finds cases of one module that check the same behaviour. Parallel planner chunks cannot see each
other's answers, and their titles differ too much for a deterministic check."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.domain.ir import TestCaseIR, http_steps, resolve_endpoint, step_statuses
from spec_run_graph.llm import build_model, output_spec, run_structured

MAX_DROP_SHARE = 0.4  # more than this from one module means the review, not the planner, went wrong

INSTRUCTIONS = """\
You review API test cases of one module for duplicates. Two cases are duplicates only if they send the same kind of
requests to the same endpoint and check the same outcome, so that a bug failing one would also fail the other.

NOT duplicates: different invalid fields or kinds of invalid value, different roles or actors, different error codes,
different states, or a case that verifies a side effect the other does not check.

Report a case as "drop" only if it is listed under candidates, and name the case to keep: the more thorough one. If
in doubt, keep both. An empty list is a good answer."""


class Duplicate(BaseModel):
    drop: str = Field(description="id of the redundant case")
    keep: str = Field(description="id of the case that checks the same behaviour")
    reason: str = Field(description="one sentence: the behaviour both cases check")


class DuplicateReview(BaseModel):
    duplicates: list[Duplicate] = Field(default_factory=list)


def build_agent(output: Any = None) -> Agent[None, DuplicateReview]:
    return Agent(
        output_type=output or output_spec(DuplicateReview),
        instructions=INSTRUCTIONS,
        retries=get_settings().srg_llm_retries,
        name="case-dedup",
    )


def _own_steps(case: TestCaseIR, own: set[str], paths: dict[str, str]) -> list[tuple[str, list[int]]]:
    """(endpoint key, expected statuses) of the steps against the module's own endpoints."""
    out = []
    for step, is_cleanup in http_steps(case):
        key = (
            resolve_endpoint(step.endpoint, paths) if step.type == "http" and step.endpoint and not is_cleanup else None
        )
        if key in own:
            out.append((key, sorted(set(step_statuses(step, case.parameters)))))
    return out


def case_view(case: TestCaseIR, own: set[str], paths: dict[str, str]) -> dict[str, Any]:
    return {
        "id": case.id,
        "title": case.title,
        "type": str(case.type),
        "criteria": case.coverage.criteria,
        "requests": [f"{key} -> {'|'.join(map(str, statuses))}" for key, statuses in _own_steps(case, own, paths)],
        "rows": [r.get("name") for r in case.parameters],
    }


def accepted(
    found: list[Duplicate], cases: list[TestCaseIR], candidates: set[str], own: set[str], paths: dict[str, str]
) -> list[Duplicate]:
    """The findings the code can stand behind: a new case dropped for a case that stays and that tests one of the
    same endpoints, within a limit per module."""
    by_id = {c.id: c for c in cases}
    limit = int(len(candidates) * MAX_DROP_SHARE)
    drops: set[str] = set()
    out: list[Duplicate] = []
    for d in found:
        if d.drop not in candidates or d.keep not in by_id or d.drop == d.keep:
            continue
        if d.drop in drops or d.keep in drops or len(out) >= limit:
            continue
        shared = {k for k, _ in _own_steps(by_id[d.drop], own, paths)} & {
            k for k, _ in _own_steps(by_id[d.keep], own, paths)
        }
        if shared:
            drops.add(d.drop)
            out.append(d)
    return [d for d in out if d.keep not in drops]


async def review(
    code: str,
    cases: list[TestCaseIR],
    candidates: set[str],
    own: set[str],
    paths: dict[str, str],
    model: Model | None = None,
) -> list[Duplicate]:
    prompt = f"Module {code}.\nCandidates (may be dropped): {sorted(candidates)}\n\nCases:\n" + "\n".join(
        json.dumps(case_view(c, own, paths)) for c in cases
    )
    result = await run_structured(build_agent, prompt, DuplicateReview, model=model or build_model())
    return accepted(result.output.duplicates, cases, candidates, own, paths)

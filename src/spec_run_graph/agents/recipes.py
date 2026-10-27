"""Recipe Designer: the LLM (.env model) proposes reusable setup recipes once per analysis.

Guard rails: every recipe passes the normal IR validation (contract facts included) as a stand-alone case;
the pipeline then runs each recipe once against the SUT and only VERIFIED recipes reach the planner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from spec_run_graph.config import get_settings
from spec_run_graph.domain.ir import IDENT_RE, ValidationContext, normalize, validate_case
from spec_run_graph.domain.recipes import RecipeIR, recipe_as_case
from spec_run_graph.llm import build_model, output_spec, run_structured

INSTRUCTIONS = """\
You design reusable SETUP RECIPES for automated API tests of one product. A recipe is a short, reliable
step sequence that test cases call instead of re-inventing their setup, e.g.
- dedicated_customer: create a fresh user account with a given balance (admin API), log in as it,
  outputs: user_id, token (send as header Authorization: Bearer {{ $extract.token }}), username;
- dedicated_product: create a product with a given stock and price, outputs: product_id, model.
Propose 2-6 recipes for the setups that many tests need (dedicated accounts with a known balance, entities
whose state tests change). Do not write tests: recipes only prepare data and capture outputs.

Read the requirements: for every workflow that needs several prepared entities which must fit together, add
one COMPOSITE recipe that prepares all of them in a single call, so tests never fall back to shared seed data.
Example: if customers can only see or buy products matching their allowed categories/manufacturers, a
purchasable_product_for_customer recipe creates a product (stock, price, category, manufacturer), creates a
customer whose balance covers it and whose filters allow it, logs in as that customer, and outputs
product_id, unit price, user_id and token. Inputs let tests vary the important values (stock, price, balance).
Every business value a test may depend on - amounts, balances, prices, quantities, stock, restrictions such as
allowed categories/manufacturers, statuses - must be an input with a default, never a literal in the payload: a
test that needs balance 100 or stock 2 can only pass it through an input. Literals are only for constant
technical values (role, flags that every caller needs the same).
Use the observed data to pick values that exist in this API (category/manufacturer names and their casing).

Recipe fields: name (snake_case), description (what it sets up, what each output holds),
inputs {name: {description, default}} (EVERY input needs a sensible default; use them as {{ $input.name }}),
variables (per-run unique values: {"username": "qa_{{ $faker.uuid }}"} used as {{ $var.username }}),
steps (the normal test-step format: name, endpoint, actor, request {headers, query, path_params, payload},
expected_status, extract {var: response.body.<path>}, assertions), cleanup (steps that delete what the recipe
created, e.g. DELETE the user), outputs {extract name: description}.
Rules:
- Every step must succeed (2xx expected_status) and send EVERY field listed under "a successful request MUST
  send" for its endpoint; extract only "assertable (guaranteed) fields".
- Steps that need privileges name the actor (one of the listed actors, e.g. admin); a login step for the new
  account needs no actor.
- Unique fields (username, email, model names) always come from variables with {{ $faker.uuid }}.
- Outputs must be extracted by the steps. Templates allowed: {{ $input.x }} {{ $var.x }} {{ $extract.x }}
  {{ $faker.x }} {{ $fixture.x }}."""


class RecipeSet(BaseModel):
    recipes: list[RecipeIR] = Field(description="2-6 reusable setup recipes")


@dataclass
class RecipeDeps:
    ctx: ValidationContext
    rejected: list[tuple[str, list[str]]] = field(default_factory=list)


def recipe_errors(r: RecipeIR, ctx: ValidationContext) -> list[str]:
    errors = []
    if not IDENT_RE.match(r.name):
        errors.append(f"name {r.name!r} must be snake_case")
    for name, spec in r.inputs.items():
        if spec.default is None:
            errors.append(f"input {name!r} needs a default")
    extracts = {v for s in r.steps for v in s.extract}
    for out in r.outputs:
        if out not in extracts:
            errors.append(f"output {out!r} is not extracted by any step")
    if errors:
        return errors
    try:
        case = recipe_as_case(r)
    except ValueError as exc:
        return [str(exc)]
    errors += validate_case(normalize(case), ctx)
    for i, s in enumerate(case.steps, start=1):
        es = s.expected_status
        if s.type == "http" and not (isinstance(es, int) and 200 <= es < 300):
            errors.append(f"step {i} ({s.name}): recipe steps must expect a 2xx status")
    return errors


def build_agent(output: Any = None) -> Agent[RecipeDeps, RecipeSet]:
    agent: Agent[RecipeDeps, RecipeSet] = Agent(
        output_type=output or output_spec(RecipeSet),
        instructions=INSTRUCTIONS,
        deps_type=RecipeDeps,
        retries=get_settings().srg_llm_retries,
        name="recipe-designer",
    )

    @agent.output_validator
    def check(ctx: RunContext[RecipeDeps], out: RecipeSet) -> RecipeSet:
        problems = {r.name: recipe_errors(r, ctx.deps.ctx) for r in out.recipes}
        problems = {k: v for k, v in problems.items() if v}
        names = [r.name for r in out.recipes]
        if len(set(names)) != len(names):
            raise ModelRetry("recipe names must be unique")
        if problems and ctx.retry < ctx.max_retries:
            lines = [f"recipe {n}:\n  - " + "\n  - ".join(e) for n, e in problems.items()]
            raise ModelRetry("Some recipes are invalid. Fix them and return ALL recipes again:\n" + "\n".join(lines))
        ctx.deps.rejected.extend(problems.items())
        return RecipeSet(recipes=[r for r in out.recipes if r.name not in problems])

    return agent


def build_prompt(
    summary: str,
    actors: dict[str, Any],
    fixtures: dict[str, Any],
    contracts: str,
    requirements: list[Any] | None = None,
    observed: str = "",
) -> str:
    parts = [
        f"Product summary:\n{summary}",
        f"Actors (authentication is automatic for these names): {json.dumps(actors)}",
        f"Fixtures: {json.dumps(fixtures, default=str)}",
        "OpenAPI contracts:\n" + contracts,
    ]
    if requirements:
        brief = [
            {"id": r.req_id, "title": r.title, "description": r.description, "criteria": r.acceptance_criteria}
            for r in requirements
        ]
        parts.append("Requirements (workflows the tests must cover):\n" + json.dumps(brief))
    if observed:
        parts.append("Observed data (real GET responses, trimmed):\n" + observed)
    return "\n\n".join(parts)


async def design(prompt: str, ctx: ValidationContext, model: Model | None = None) -> tuple[list[RecipeIR], RecipeDeps]:
    deps = RecipeDeps(ctx=ctx)
    result = await run_structured(build_agent, prompt, RecipeSet, deps=deps, model=model or build_model())
    return result.output.recipes, deps

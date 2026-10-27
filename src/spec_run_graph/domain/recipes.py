"""Setup recipes: verified, reusable step sequences ("create a dedicated customer with a balance and log in").

The planner references a recipe with one step instead of re-inventing the setup:
    {"name": "Dedicated buyer", "type": "recipe", "recipe": "dedicated_customer",
     "inputs": {"balance": 5000}, "alias": "buyer"}
`expand_recipes` replaces it by the recipe's steps before validation and storage, so Gel, the compiler and
the healer only ever see ordinary steps. Outputs become extracts of the case ({{ $extract.buyer_token }}, or
{{ $extract.token }} without alias); the recipe's internal variables/extracts are renamed so two uses of the
same recipe never collide. Recipe cleanup steps are prepended to the case cleanup.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from spec_run_graph.domain.ir import IDENT_RE, REF_RE, CaseDraft, StepIR

INPUT_RE = re.compile(r"\{\{\s*\$input\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class RecipeInput(BaseModel):
    description: str = ""
    default: Any = Field(None, description="used when the case does not pass the input; null = required")


class RecipeIR(BaseModel):
    name: str = Field(description="snake_case, e.g. dedicated_customer")
    description: str = Field(description="what the recipe sets up and what its outputs hold")
    inputs: dict[str, RecipeInput] = Field(default_factory=dict, description="used as {{ $input.<name> }}")
    variables: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepIR] = Field(min_length=1)
    cleanup: list[StepIR] = Field(default_factory=list)
    outputs: dict[str, str] = Field(description="extract name -> what it holds (exposed to the case)")


def recipe_brief(r: RecipeIR) -> dict[str, Any]:
    """What the planner needs to call a recipe."""
    return {
        "recipe": r.name,
        "description": r.description,
        "inputs": {k: {"description": v.description, "default": v.default} for k, v in r.inputs.items()},
        "outputs": r.outputs,
        "creates": [s.endpoint for s in r.steps if s.endpoint],
    }


def _substitute(obj: Any, inputs: dict[str, Any], rename: dict[str, dict[str, str]]) -> Any:
    if isinstance(obj, dict):
        return {k: _substitute(v, inputs, rename) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, inputs, rename) for v in obj]
    if not isinstance(obj, str):
        return obj
    whole = INPUT_RE.fullmatch(obj.strip())
    if whole:  # keep the input's type (numbers stay numbers)
        return inputs[whole.group(1)]
    text = INPUT_RE.sub(lambda m: str(inputs[m.group(1)]), obj)

    def ref(m: re.Match) -> str:
        ns, name = m.group(1), m.group(2)
        if ns in rename and name in rename[ns]:
            return "{{ $" + ns + "." + rename[ns][name] + " }}"
        return m.group(0)

    return REF_RE.sub(ref, text)


def expand_recipes(case: CaseDraft, recipes: dict[str, RecipeIR]) -> tuple[CaseDraft, list[str]]:
    """Return (case with recipe steps replaced by ordinary steps, errors)."""
    if not any(s.type == "recipe" for s in case.steps):
        return case, []
    errors: list[str] = []
    steps: list[StepIR] = []
    cleanup: list[StepIR] = []
    variables = dict(case.variables)
    for idx, step in enumerate(case.steps, start=1):
        if step.type != "recipe":
            steps.append(step)
            continue
        r = recipes.get(step.recipe or "")
        if r is None:
            errors.append(f"step {idx} ({step.name}): unknown recipe {step.recipe!r}; available: {sorted(recipes)}")
            continue
        if step.alias and not IDENT_RE.match(step.alias):
            errors.append(f"step {idx} ({step.name}): alias {step.alias!r} must be an identifier")
            continue
        unknown = sorted(set(step.inputs) - set(r.inputs))
        if unknown:
            errors.append(
                f"step {idx} ({step.name}): recipe {r.name} has no inputs {unknown}; inputs: {sorted(r.inputs)}"
            )
        inputs = {k: step.inputs.get(k, v.default) for k, v in r.inputs.items()}
        missing = sorted(k for k, v in inputs.items() if v is None)
        if missing:
            errors.append(f"step {idx} ({step.name}): recipe {r.name} needs inputs {missing}")
            continue
        own = f"{step.alias or r.name}_{idx}_"  # internal names of this use
        out_prefix = f"{step.alias}_" if step.alias else ""
        extracts = {v for s in r.steps for v in s.extract}
        rename = {
            "var": {v: own + v for v in r.variables},
            "extract": {v: (out_prefix + v) if v in r.outputs else own + v for v in extracts},
        }
        for name, value in r.variables.items():
            variables[own + name] = _substitute(value, inputs, rename)
        for s in r.steps:
            data = _substitute(s.model_dump(), inputs, rename)
            data["extract"] = {rename["extract"][k]: v for k, v in s.extract.items()}
            data["name"] = f"{step.name}: {s.name}"
            steps.append(StepIR.model_validate(data))
        for s in r.cleanup:
            data = _substitute(s.model_dump(), inputs, rename)
            data["name"] = f"{step.name}: {s.name}"
            cleanup.append(StepIR.model_validate(data))
    if errors:
        return case, errors
    return case.model_copy(update={"steps": steps, "cleanup": cleanup + list(case.cleanup), "variables": variables}), []


def recipe_as_case(r: RecipeIR) -> CaseDraft:
    """A recipe with its default/sample inputs as a stand-alone case (for validation and the live check)."""
    sample = {k: (v.default if v.default is not None else 1) for k, v in r.inputs.items()}
    step = StepIR(name=r.name, type="recipe", recipe=r.name, inputs=sample)
    case = CaseDraft(title=f"recipe {r.name}", type="positive", steps=[step])
    expanded, errors = expand_recipes(case, {r.name: r})
    if errors:
        raise ValueError("; ".join(errors))
    return expanded

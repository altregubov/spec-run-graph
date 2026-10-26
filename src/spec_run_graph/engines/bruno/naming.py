"""Deterministic Bruno file naming for a test case (flat layout inside the module folder).

TC-ORD-003.yml                          one HTTP step, no params
TC-ORD-005a-zero-quantity.yml           param row a
TC-ORD-009-step1-cancel-order.yml       multi-step workflow
TC-ORD-010a-step1-create-order.yml      params x multi-step
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from spec_run_graph.domain.ir import TestCaseIR, http_steps

FILE_RE = re.compile(r"^(TC-[A-Z0-9_]+-\d{3,})([a-z])?(?:-step(\d+))?(?:-.+)?\.yml$")


def slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "step"


def var_prefix(test_id: str) -> str:
    """Runtime-variable namespace per case: TC-ORD-009 -> TC_ORD_009__"""
    return test_id.replace("-", "_") + "__"


@dataclass(frozen=True)
class PlannedFile:
    filename: str
    test_id: str
    param_index: int  # 0 when the case has no parameter rows
    param_name: str | None
    step_seq: int  # 1-based position among the case's steps (IR order)
    file_step: int  # 1-based position among rendered HTTP files of this row
    step_name: str
    is_cleanup: bool
    display_name: str


def plan_files(case: TestCaseIR) -> list[PlannedFile]:
    assert case.id, "case needs an id"
    steps = [(i, s, c) for i, (s, c) in enumerate(http_steps(case), start=1) if s.type == "http"]
    rows = case.parameters or [None]
    multi = len(steps) > 1
    out: list[PlannedFile] = []
    used: set[str] = set()
    for p_idx, row in enumerate(rows):
        letter = string.ascii_lowercase[p_idx] if case.parameters else ""
        pname = row.get("name") if row else None
        for n, (seq, step, is_cleanup) in enumerate(steps, start=1):
            base = f"{case.id}{letter}"
            if multi:
                slug = slugify(step.name)
                name = f"{base}-step{n}-{slug}"
                display = f"{case.id}{letter} step{n} {step.name}"
            elif pname:
                name = f"{base}-{slugify(pname)}"
                display = f"{case.id}{letter} {pname}"
            else:
                name = base
                display = f"{case.id} {case.title}"
            while name in used:
                name += "-x"
            used.add(name)
            out.append(
                PlannedFile(
                    filename=f"{name}.yml",
                    test_id=case.id,
                    param_index=p_idx,
                    param_name=pname,
                    step_seq=seq,
                    file_step=n,
                    step_name=step.name,
                    is_cleanup=is_cleanup,
                    display_name=display,
                )
            )
    return out


def parse_filename(filename: str) -> tuple[str, int, int] | None:
    """Fallback mapping file -> (test_id, param_index, file_step)."""
    m = FILE_RE.match(filename.rsplit("/", 1)[-1])
    if not m:
        return None
    letter = m.group(2)
    return m.group(1), (ord(letter) - ord("a")) if letter else 0, int(m.group(3) or 1)

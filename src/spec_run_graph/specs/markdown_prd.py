"""PRD markdown loading: sectioning only. Requirement extraction is done by the analyzer agent."""

from __future__ import annotations

import re
from pathlib import Path

from spec_run_graph.specs.models import LoadedSpec, Section, sha256

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
REQ_ID = re.compile(r"\b([A-Z][A-Z0-9]*-[A-Z][A-Z0-9]*-\d+|[A-Z]{2,}-\d+)\b")


def load_prd(path: Path) -> LoadedSpec:
    raw = path.read_text()
    sections: list[Section] = []
    title: str | None = None
    current: Section | None = None
    body: list[str] = []
    in_code = False

    def flush() -> None:
        if current is not None:
            current.body = "\n".join(body).strip()
            sections.append(current)

    for line in raw.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2)
            if level == 1 and title is None:
                title = heading
            rid = REQ_ID.search(heading)
            current, body = Section(heading=heading, level=level, body="", req_id=rid and rid.group(1)), []
        else:
            body.append(line)
    if current is None:
        current = Section(heading=title or path.stem, level=1, body="")
    flush()

    return LoadedSpec(
        name=title or path.stem,
        spec_type="REQUIREMENTS_DOC",
        version=f"sha-{sha256(raw)[:10]}",
        source_path=str(path),
        raw_content=raw,
        sections=sections,
    )


def prd_digest(spec: LoadedSpec) -> str:
    parts = [f"# Requirements document '{spec.name}' ({spec.source_path})"]
    for s in spec.sections:
        rid = f" [id: {s.req_id}]" if s.req_id else ""
        parts.append(f"{'#' * max(s.level, 2)} {s.heading}{rid}\n{s.body}".rstrip())
    return "\n\n".join(parts)

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SpecType = Literal["OPENAPI", "REQUIREMENTS_DOC"]
HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


class EndpointDraft(BaseModel):
    method: str
    path: str
    operation_id: str | None = None
    summary: str | None = None
    tag: str | None = None
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    request_schema: Any | None = None
    response_schemas: dict[str, Any] = Field(default_factory=dict)  # status -> {description, schema}
    requires_auth: bool = False

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"

    @property
    def contract_hash(self) -> str:
        return stable_hash(
            {"p": self.parameters, "req": self.request_schema, "res": self.response_schemas, "a": self.requires_auth}
        )


class Section(BaseModel):
    heading: str
    level: int
    body: str
    req_id: str | None = None


class LoadedSpec(BaseModel):
    name: str
    spec_type: SpecType
    version: str
    source_path: str
    raw_content: str
    endpoints: list[EndpointDraft] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    servers: list[str] = Field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return sha256(self.raw_content)

    def endpoint_index(self) -> dict[str, EndpointDraft]:
        return {e.key: e for e in self.endpoints}


def requirement_specs(citation: str, specs: list[LoadedSpec]) -> list[str]:
    """Names of the specs a requirement's free-text citation refers to, in citation order. Names and file names
    count, longest first so a name inside a longer one does not match; without a match, the first requirements doc."""

    def norm(text: str) -> str:
        return re.sub(r"[\u2010-\u2015]", "-", text).casefold()

    aliases = sorted(
        {
            (norm(alias), s.name)
            for s in specs
            for alias in (s.name, Path(s.source_path).name, Path(s.source_path).stem)
        },
        key=lambda a: -len(a[0]),
    )
    text = norm(citation)
    found: dict[str, int] = {}
    for alias, name in aliases:
        for m in re.finditer(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", text):
            found[name] = min(found.get(name, m.start()), m.start())
            text = text[: m.start()] + " " * len(alias) + text[m.end() :]
    if found:
        return sorted(found, key=lambda n: found[n])
    fallback = next((s for s in specs if s.spec_type == "REQUIREMENTS_DOC"), specs[0])
    return [fallback.name]

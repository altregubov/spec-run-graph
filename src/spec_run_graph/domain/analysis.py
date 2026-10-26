"""Output of the joint Spec Analyzer and the persisted analysis report."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from spec_run_graph.specs.diff import SpecChanges


class AuthDef(BaseModel):
    type: Literal["bearer", "basic", "apikey", "none"] = "bearer"
    token: str | None = Field(None, description="bearer token, as '{{ $env.VAR }}' - never a literal secret")
    username: str | None = None
    password: str | None = Field(None, description="as '{{ $env.VAR }}'")
    key: str | None = Field(None, description="apikey header/query name")
    value: str | None = Field(None, description="apikey value, as '{{ $env.VAR }}'")
    placement: Literal["header", "query"] = "header"


class ActorDef(BaseModel):
    name: str = Field(description="snake_case role name, e.g. customer_standard")
    description: str = ""
    auth: AuthDef = Field(default_factory=AuthDef)


class FixtureDef(BaseModel):
    name: str = Field(description="snake_case fixture name, e.g. active_sku")
    value: Any = Field(None, description="literal test-data value (omit when env is used)")
    env: str | None = Field(None, description="environment variable that holds the value instead")
    description: str = ""


class ModuleProposal(BaseModel):
    code: str = Field(description="UPPERCASE module code used in test ids, e.g. ORD")
    folder: str = Field(description="kebab-case Bruno folder name, e.g. orders")
    domain: str = Field(description="display name, e.g. 'Orders'")
    endpoints: list[str] = Field(description="endpoint keys 'METHOD /path' owned by this module")
    requirement_ids: list[str] = Field(default_factory=list)
    base_path: str = Field("", description="server prefix NOT already part of the OpenAPI paths (usually '')")
    default_headers: dict[str, str] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    actors: list[ActorDef] = Field(default_factory=list)
    fixtures: list[FixtureDef] = Field(default_factory=list)


class AnalyzedRequirement(BaseModel):
    req_id: str = Field(description="id from the document, or REQ-<CODE>-NN when the document has none")
    spec: str = Field(description="name of the spec the requirement comes from")
    title: str = ""
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list, description="endpoint keys implementing it")


class Conflict(BaseModel):
    topic: str
    statement_a: str
    source_a: str
    statement_b: str
    source_b: str
    severity: Literal["low", "medium", "high"] = "medium"
    recommendation: str = ""
    refs: list[str] = Field(default_factory=list, description="requirement ids / endpoint keys")


class Gap(BaseModel):
    kind: Literal["requirement_without_endpoint", "endpoint_without_requirement", "undocumented_behaviour", "other"]
    description: str
    refs: list[str] = Field(default_factory=list)


class SpecAnalysis(BaseModel):
    summary: str
    modules: list[ModuleProposal]
    requirements: list[AnalyzedRequirement]
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class SpecRef(BaseModel):
    name: str
    spec_type: str
    version: str
    source_path: str
    content_hash: str


class AnalysisReport(BaseModel):
    analysis_id: str
    created_at: datetime
    model: str
    specs: list[SpecRef]
    analysis: SpecAnalysis
    changes: list[SpecChanges] = Field(default_factory=list)

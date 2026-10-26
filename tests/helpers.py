"""Shared test helpers: a hand-written analysis of examples/orders (no LLM)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from spec_run_graph.domain.analysis import (
    ActorDef,
    AnalysisReport,
    AnalyzedRequirement,
    AuthDef,
    Conflict,
    FixtureDef,
    Gap,
    ModuleProposal,
    SpecAnalysis,
    SpecRef,
)
from spec_run_graph.domain.ir import ModuleIR
from spec_run_graph.specs.loader import load_specs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def orders_specs(folder: str = "orders"):
    return load_specs([ROOT / "examples" / folder])


def orders_analysis(specs) -> SpecAnalysis:
    prd = next(s for s in specs if s.spec_type == "REQUIREMENTS_DOC").name
    keys = [e.key for s in specs for e in s.endpoints]
    order_keys = [k for k in keys if "/orders" in k]
    return SpecAnalysis(
        summary="Orders API for a demo shop.",
        modules=[
            ModuleProposal(
                code="PROD",
                folder="catalog",
                domain="Catalog",
                endpoints=["GET /v1/products"],
                requirement_ids=["PRD-ORD-40"],
                default_headers={"Accept": "application/json"},
                actors=[ActorDef(name="customer_standard", auth=AuthDef(token="{{ $env.CUSTOMER_TOKEN }}"))],
            ),
            ModuleProposal(
                code="ORD",
                folder="orders",
                domain="Orders",
                endpoints=order_keys,
                requirement_ids=["PRD-ORD-42", "PRD-ORD-43"],
                default_headers={"Accept": "application/json"},
                actors=[
                    ActorDef(name="customer_standard", auth=AuthDef(token="{{ $env.CUSTOMER_TOKEN }}")),
                    ActorDef(name="admin", auth=AuthDef(token="{{ $env.ADMIN_TOKEN }}")),
                ],
                fixtures=[FixtureDef(name="active_sku", value="SKU-ACTIVE-1")],
            ),
        ],
        requirements=[
            AnalyzedRequirement(
                req_id="PRD-ORD-40", spec=prd, description="List products", endpoints=["GET /v1/products"]
            ),
            AnalyzedRequirement(
                req_id="PRD-ORD-42", spec=prd, description="Create draft order", endpoints=["POST /v1/orders"]
            ),
            AnalyzedRequirement(
                req_id="PRD-ORD-43", spec=prd, description="Quantity validation", endpoints=["POST /v1/orders"]
            ),
        ],
        conflicts=[
            Conflict(
                topic="max qty",
                statement_a="qty > 50 rejected",
                source_a=prd,
                statement_b="maximum 100",
                source_b="Orders API",
                refs=["PRD-ORD-43"],
            )
        ],
        gaps=[
            Gap(kind="requirement_without_endpoint", description="Order history has no endpoint", refs=["PRD-ORD-48"])
        ],
    )


def orders_report(specs, analysis_id: str = "test-analysis") -> AnalysisReport:
    return AnalysisReport(
        analysis_id=analysis_id,
        created_at=datetime.now(UTC),
        model="test",
        specs=[
            SpecRef(
                name=s.name,
                spec_type=s.spec_type,
                version=s.version,
                source_path=s.source_path,
                content_hash=s.content_hash,
            )
            for s in specs
        ],
        analysis=orders_analysis(specs),
    )


def reference_module(name: str = "reference_fixed") -> ModuleIR:
    return ModuleIR.model_validate(yaml.safe_load((FIXTURES / f"{name}.yaml").read_text()))

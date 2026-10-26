"""Control plane: the pydantic-graph state machine that drives every CLI verb.

RouteNode -> AnalyzeNode -(stop=analyze)-> End
          -> PlanNode    -(stop=plan)----> End        (stores test cases in Gel)
          -> BuildNode   -(stop=build)---> End        (LLM implements them as Bruno files)
          -> TestNode    ----------------> End        (health check + bru run + results)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph_builder import GraphBuilder

from spec_run_graph.deps import GraphDependencies
from spec_run_graph.domain.analysis import AnalysisReport
from spec_run_graph.knowledge.repo import CaseFilter
from spec_run_graph.pipeline import stages
from spec_run_graph.specs.models import LoadedSpec

Stage = Literal["analyze", "plan", "build", "test"]
ORDER: list[Stage] = ["analyze", "plan", "build", "test"]


@dataclass
class PipelineGraphState:
    """Volatile run state; everything durable lives in Gel, MinIO and the files on disk."""

    start: Stage
    stop: Stage
    spec_paths: list[Path] = field(default_factory=list)
    analysis_file: Path | None = None
    replace: bool = False
    case_filter: CaseFilter = field(default_factory=CaseFilter)
    force: bool = False
    include_needs_review: bool = False
    dry_run: bool = False
    env_name: str = "local"
    base_url: str | None = None
    skip_health_check: bool = False
    # results
    specs: list[LoadedSpec] = field(default_factory=list)
    report: AnalysisReport | None = None
    plan: stages.PlanSummary | None = None
    build: stages.BuildSummary | None = None
    run: stages.RunSummary | None = None

    def stops_after(self, stage: Stage) -> bool:
        return ORDER.index(stage) >= ORDER.index(self.stop)


Ctx = GraphRunContext[PipelineGraphState, GraphDependencies]


@dataclass
class TestNode(BaseNode[PipelineGraphState, GraphDependencies, PipelineGraphState]):
    __test__ = False

    async def run(self, ctx: Ctx) -> End[PipelineGraphState]:
        s = ctx.state
        s.run = await stages.test_stage(ctx.deps, s.case_filter, s.env_name, s.base_url, s.skip_health_check)
        return End(s)


@dataclass
class BuildNode(BaseNode[PipelineGraphState, GraphDependencies, PipelineGraphState]):
    async def run(self, ctx: Ctx) -> TestNode | End[PipelineGraphState]:
        s = ctx.state
        s.build = await stages.build_stage(ctx.deps, s.case_filter, s.force, s.include_needs_review, s.dry_run)
        return End(s) if s.stops_after("build") else TestNode()


@dataclass
class PlanNode(BaseNode[PipelineGraphState, GraphDependencies, PipelineGraphState]):
    async def run(self, ctx: Ctx) -> BuildNode | End[PipelineGraphState]:
        s = ctx.state
        if s.report is None:
            assert s.analysis_file is not None
            s.specs, s.report = stages.load_report(s.analysis_file)
        s.plan = await stages.plan_stage(ctx.deps, s.specs, s.report, s.replace)
        return End(s) if s.stops_after("plan") else BuildNode()


@dataclass
class AnalyzeNode(BaseNode[PipelineGraphState, GraphDependencies, PipelineGraphState]):
    async def run(self, ctx: Ctx) -> PlanNode | End[PipelineGraphState]:
        s = ctx.state
        s.specs, s.report = await stages.analyze_stage(ctx.deps, s.spec_paths)
        return End(s) if s.stops_after("analyze") else PlanNode()


@dataclass
class RouteNode(BaseNode[PipelineGraphState, GraphDependencies, PipelineGraphState]):
    async def run(self, ctx: Ctx) -> AnalyzeNode | PlanNode | BuildNode | TestNode:
        start = ctx.state.start
        if start == "analyze":
            return AnalyzeNode()
        if start == "plan":
            return PlanNode()
        if start == "build":
            return BuildNode()
        return TestNode()


def build_graph():
    g = GraphBuilder(
        name="spec-run-graph",
        state_type=PipelineGraphState,
        deps_type=GraphDependencies,
        input_type=RouteNode,
        output_type=PipelineGraphState,
        auto_instrument=False,
    )
    g.add(
        g.edge_from(g.start_node).to(RouteNode),
        g.node(RouteNode),
        g.node(AnalyzeNode),
        g.node(PlanNode),
        g.node(BuildNode),
        g.node(TestNode),
    )
    return g.build(validate_graph_structure=False)


async def run_pipeline(state: PipelineGraphState, deps: GraphDependencies) -> PipelineGraphState:
    graph = build_graph()
    result: Any = await graph.run(state=state, deps=deps, inputs=RouteNode())
    return result

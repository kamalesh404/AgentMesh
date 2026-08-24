"""Multi-agent coordination: graphs, supervision, debate, and pipelines."""

from src.orchestration.debate import DebateOrchestrator, DebateResult, DebateTurn
from src.orchestration.graph import AgentGraph, RouteNotFoundError
from src.orchestration.pipeline import Pipeline, PipelineError, StageResult
from src.orchestration.supervisor import Supervisor, SupervisorError

__all__ = [
    "DebateOrchestrator",
    "DebateResult",
    "DebateTurn",
    "AgentGraph",
    "RouteNotFoundError",
    "Pipeline",
    "PipelineError",
    "StageResult",
    "Supervisor",
    "SupervisorError",
]

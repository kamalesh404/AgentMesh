"""AgentMesh agents: base lifecycle plus five specialized implementations."""

from src.agents.base import (
    AgentError,
    AgentLifecycleError,
    AgentState,
    BaseAgent,
)
from src.agents.conversational import ChatAgent
from src.agents.critic import CriticAgent, ReviewIssue, ReviewResult
from src.agents.planner import GoalTree, PlannerAgent, TaskNode

__all__ = [
    "AgentError",
    "AgentLifecycleError",
    "AgentState",
    "BaseAgent",
    "ChatAgent",
    "CriticAgent",
    "GoalTree",
    "PlannerAgent",
    "ReviewIssue",
    "ReviewResult",
    "TaskNode",
]


def __getattr__(name: str):
    """Lazily expose tool-dependent agents to keep imports lightweight."""
    if name == "CodingAgent":
        from src.agents.coding import CodingAgent

        return CodingAgent
    if name == "ResearchAgent":
        from src.agents.research import ResearchAgent

        return ResearchAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

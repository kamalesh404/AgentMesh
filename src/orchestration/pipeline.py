"""Sequential agent pipelines with per-stage transforms."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agents.base import AgentState, BaseAgent
from src.protocols.message import Message, MessageType

Transform = Callable[[Any, Dict[str, Any]], Any]


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails or produces nothing."""


@dataclass
class StageResult:
    stage: str
    output: Any
    duration_ms: float
    skipped: bool = False


class Pipeline:
    """Runs payloads through an ordered chain of agents.

    Each stage receives the previous stage's output as message content. An
    optional transform reshapes the output before it flows to the next stage;
    transforms also receive the running context for cross-stage access.
    """

    def __init__(self, name: str = "pipeline", initial_context: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.context: Dict[str, Any] = dict(initial_context or {})
        self.context.setdefault("steps", {})
        self._stages: List[Tuple[BaseAgent, Optional[Transform]]] = []

    def add_stage(self, agent: BaseAgent, transform: Optional[Transform] = None) -> "Pipeline":
        self._stages.append((agent, transform))
        return self

    @property
    def stage_names(self) -> List[str]:
        return [agent.agent_id for agent, _ in self._stages]

    async def run(self, payload: Any) -> Dict[str, Any]:
        """Execute every stage in order; returns the accumulated context."""
        if not self._stages:
            raise PipelineError("pipeline has no stages")
        current = payload
        results: List[StageResult] = []
        for index, (agent, transform) in enumerate(self._stages):
            if getattr(agent, "state", None) is AgentState.CREATED:
                await agent.initialize()
            message = Message(
                sender=f"pipeline:{self.name}",
                recipient=agent.agent_id,
                content=current,
                metadata={"intent": str(agent.role), "stage_index": index},
            )
            started = time.perf_counter()
            reply = await agent.deliver(message)
            duration = round((time.perf_counter() - started) * 1000, 2)
            if reply is None:
                raise PipelineError(f"stage '{agent.agent_id}' returned no reply")
            if reply.type is MessageType.ERROR:
                raise PipelineError(f"stage '{agent.agent_id}' failed: {reply.content}")
            current = reply.content
            if transform is not None:
                current = transform(current, self.context)
            self.context["steps"][agent.agent_id] = current
            results.append(StageResult(stage=agent.agent_id, output=current, duration_ms=duration))
        self.context["output"] = current
        self.context["stage_results"] = [
            {"stage": r.stage, "duration_ms": r.duration_ms} for r in results
        ]
        return self.context

    def reset(self) -> None:
        """Clear accumulated context between runs."""
        self.context.clear()
        self.context["steps"] = {}

"""Supervisor agent: capability-aware task delegation to workers."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Set

from src.agents.base import AgentError, BaseAgent
from src.protocols.message import Message, MessageType


class SupervisorError(AgentError):
    """Raised when delegation cannot be completed."""


class Supervisor(BaseAgent):
    """Routes tasks to registered workers chosen by capability and load."""

    role = "supervisor"

    def __init__(self, name: str, provider: Any = None, dispatch_timeout: float = 30.0) -> None:
        super().__init__(name=name, provider=provider)
        self.dispatch_timeout = dispatch_timeout
        self._workers: Dict[str, BaseAgent] = {}
        self._skills: Dict[str, Set[str]] = {}
        self._inflight: Dict[str, int] = {}
        self.completed_dispatches = 0
        self.failed_dispatches = 0

    def register(self, worker: BaseAgent, capabilities: Iterable[str] = ()) -> None:
        if worker.agent_id in self._workers:
            raise SupervisorError(f"worker '{worker.agent_id}' already registered")
        self._workers[worker.agent_id] = worker
        self._skills[worker.agent_id] = {cap.strip().lower() for cap in capabilities}
        self._inflight[worker.agent_id] = 0

    def unregister(self, agent_id: str) -> bool:
        removed = self._workers.pop(agent_id, None) is not None
        self._skills.pop(agent_id, None)
        self._inflight.pop(agent_id, None)
        return removed

    @property
    def workers(self) -> List[str]:
        return sorted(self._workers)

    def _select_worker(
        self, required: Optional[Iterable[str]] = None
    ) -> Optional[BaseAgent]:
        wanted = {cap.strip().lower() for cap in (required or [])}
        candidates: List[tuple] = []
        for agent_id, worker in self._workers.items():
            if wanted and not wanted.issubset(self._skills[agent_id]):
                continue
            overlap = len(wanted & self._skills[agent_id])
            load = self._inflight[agent_id]
            candidates.append((-overlap, load, agent_id, worker))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    async def delegate(
        self,
        task_description: str,
        required_capabilities: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Dispatch a task and return the worker's reply content."""
        worker = self._select_worker(required_capabilities)
        if worker is None:
            wanted = sorted(cap.strip() for cap in (required_capabilities or []))
            raise SupervisorError(f"no registered worker satisfies capabilities {wanted}")
        message = Message(
            sender=self.agent_id,
            recipient=worker.agent_id,
            content=task_description,
            metadata={"intent": "task", "capabilities": sorted(self._skills[worker.agent_id])},
        )
        self._inflight[worker.agent_id] += 1
        try:
            reply = await asyncio.wait_for(worker.deliver(message), timeout or self.dispatch_timeout)
        except asyncio.TimeoutError as exc:
            self.failed_dispatches += 1
            raise SupervisorError(f"worker '{worker.agent_id}' timed out") from exc
        finally:
            self._inflight[worker.agent_id] -= 1
        if reply is None:
            self.failed_dispatches += 1
            raise SupervisorError(f"worker '{worker.agent_id}' returned no reply")
        if reply.type is MessageType.ERROR:
            self.failed_dispatches += 1
            raise SupervisorError(f"worker failure: {reply.content}")
        self.completed_dispatches += 1
        return reply.content

    async def process_message(self, message: Message) -> Message:
        """Accept inbound status updates; unknown intents are delegated."""
        if message.metadata.get("intent") == "status":
            self.logger.info("status from %s: %s", message.sender, message.content)
            return message.reply({"acknowledged": True})
        result = await self.delegate(str(message.content))
        return message.reply(result)

    def info(self) -> Dict[str, Any]:
        payload = super().info()
        payload.update(
            {
                "workers": self.workers,
                "skills": {wid: sorted(skills) for wid, skills in self._skills.items()},
                "completed": self.completed_dispatches,
                "failed": self.failed_dispatches,
            }
        )
        return payload

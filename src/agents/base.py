"""Agent base class: lifecycle state machine and inbox processing."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from src.protocols.message import Message, MessageType
from src.tools.base import ToolRegistry
from src.utils.logging import get_logger


class AgentState(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


_VALID_TRANSITIONS: Dict[AgentState, set] = {
    AgentState.CREATED: {AgentState.INITIALIZING},
    AgentState.INITIALIZING: {AgentState.READY, AgentState.ERROR},
    AgentState.READY: {AgentState.RUNNING, AgentState.PAUSED, AgentState.STOPPED, AgentState.ERROR},
    AgentState.RUNNING: {AgentState.PAUSED, AgentState.READY, AgentState.STOPPED, AgentState.ERROR},
    AgentState.PAUSED: {AgentState.RUNNING, AgentState.STOPPED, AgentState.ERROR},
    AgentState.STOPPED: set(),
    AgentState.ERROR: {AgentState.INITIALIZING},
}


class AgentError(RuntimeError):
    """Generic agent failure."""


class AgentLifecycleError(AgentError):
    """Raised on an invalid lifecycle transition."""


class BaseAgent(ABC):
    """Abstract agent with a validated state machine and async inbox.

    Subclasses implement :meth:`process_message`, which receives one inbound
    :class:`Message` and may return a reply message. Messages can be handled
    synchronously via :meth:`deliver` or queued for background consumption by
    starting the agent with :meth:`start`.
    """

    role: str = "generic"

    def __init__(
        self,
        name: str,
        provider: Any = None,
        memory: Any = None,
        max_inbox: int = 256,
    ) -> None:
        if not name.strip():
            raise ValueError("agent name must be non-empty")
        self.name = name
        self.provider = provider
        self.memory = memory
        self.agent_id = f"{self.role}.{name}"
        self.state: AgentState = AgentState.CREATED
        self.inbox: asyncio.Queue = asyncio.Queue(maxsize=max_inbox)
        self.tools = ToolRegistry()
        self.processed = 0
        self.failed = 0
        self.last_error: Optional[str] = None
        self.logger = get_logger(f"agent.{self.agent_id}")
        self._worker_task: Optional[asyncio.Task] = None
        self._resumed = asyncio.Event()
        self._resumed.set()

    def transition(self, next_state: AgentState) -> None:
        """Move to ``next_state`` when the transition is legal."""
        allowed = _VALID_TRANSITIONS[self.state]
        if next_state not in allowed:
            raise AgentLifecycleError(
                f"invalid transition {self.state.value} -> {next_state.value} "
                f"(allowed: {sorted(s.value for s in allowed)})"
            )
        self.logger.debug("state %s -> %s", self.state.value, next_state.value)
        self.state = next_state

    async def setup(self) -> None:  # noqa: B027 - optional subclass hook
        """Hook for subclass initialization work."""

    async def initialize(self) -> None:
        """Run setup hooks and move CREATED -> INITIALIZING -> READY."""
        self.transition(AgentState.INITIALIZING)
        try:
            await self.setup()
        except Exception as exc:
            self.transition(AgentState.ERROR)
            raise AgentError(f"initialization failed for {self.agent_id}: {exc}") from exc
        self.transition(AgentState.READY)

    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """Handle one inbound message and optionally return a reply."""

    async def deliver(self, message: Message) -> Optional[Message]:
        """Synchronously handle a validated message."""
        message.validate()
        if message.expired:
            self.logger.warning("dropping expired message %s", message.id)
            return None
        try:
            reply = await self.process_message(message)
            self.processed += 1
            return reply
        except Exception as exc:
            self.failed += 1
            self.last_error = str(exc)
            self.logger.error("processing failed: %s", exc)
            return message.reply(
                {"error": str(exc), "agent": self.agent_id}, type=MessageType.ERROR
            )

    async def send(self, message: Message) -> None:
        """Queue a message for background consumption after start()."""
        await self.inbox.put(message)

    async def start(self) -> None:
        """Enter RUNNING and spawn the background consumer loop."""
        if self.state is AgentState.CREATED:
            await self.initialize()
        if self.state is not AgentState.READY:
            raise AgentLifecycleError(f"cannot start from state '{self.state.value}'")
        self.transition(AgentState.RUNNING)
        self._worker_task = asyncio.create_task(self._consume(), name=f"{self.agent_id}-loop")

    async def _consume(self) -> None:
        while True:
            message = await self.inbox.get()
            if message is None:
                break
            await self._resumed.wait()
            if self.state is not AgentState.RUNNING:
                break
            await self.deliver(message)

    async def pause(self) -> None:
        self.transition(AgentState.PAUSED)
        self._resumed.clear()

    async def resume(self) -> None:
        self.transition(AgentState.RUNNING)
        self._resumed.set()

    async def stop(self) -> None:
        """Drain the loop and mark the agent STOPPED (idempotent)."""
        if self.state is AgentState.STOPPED:
            return
        if self.state in {AgentState.READY, AgentState.RUNNING, AgentState.PAUSED}:
            self.transition(AgentState.STOPPED)
        self._resumed.set()
        self.inbox.put_nowait(None)
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def info(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "name": self.name,
            "state": self.state.value,
            "processed": self.processed,
            "failed": self.failed,
            "tools": self.tools.names(),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.agent_id!r} state={self.state.value}>"

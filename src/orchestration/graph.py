"""Agent communication graph with BFS path resolution."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set

from src.agents.base import BaseAgent
from src.protocols.message import Message


class RouteNotFoundError(RuntimeError):
    """Raised when no path connects a sender to its recipient."""


class AgentGraph:
    """Topology of agents connected by directed communication edges."""

    def __init__(self) -> None:
        self.agents: Dict[str, BaseAgent] = {}
        self._adjacency: Dict[str, Set[str]] = {}

    def add(self, agent: BaseAgent) -> None:
        if agent.agent_id in self.agents:
            raise ValueError(f"agent '{agent.agent_id}' is already in the graph")
        self.agents[agent.agent_id] = agent
        self._adjacency.setdefault(agent.agent_id, set())

    def connect(self, source_id: str, target_id: str, bidirectional: bool = True) -> None:
        self._require(source_id)
        self._require(target_id)
        self._adjacency[source_id].add(target_id)
        if bidirectional:
            self._adjacency[target_id].add(source_id)

    def neighbors(self, agent_id: str) -> List[str]:
        self._require(agent_id)
        return sorted(self._adjacency[agent_id])

    def resolve_path(self, sender_id: str, recipient_id: str) -> List[str]:
        """Shortest hop sequence between two registered agents."""
        self._require(sender_id)
        self._require(recipient_id)
        if sender_id == recipient_id:
            return [sender_id]
        visited = {sender_id}
        queue: deque = deque([[sender_id]])
        while queue:
            trail = queue.popleft()
            for neighbor in sorted(self._adjacency[trail[-1]]):
                if neighbor in visited:
                    continue
                extended = trail + [neighbor]
                if neighbor == recipient_id:
                    return extended
                visited.add(neighbor)
                queue.append(extended)
        raise RouteNotFoundError(f"no route from '{sender_id}' to '{recipient_id}'")

    async def route(self, message: Message) -> Optional[Message]:
        """Deliver a message along the shortest path, recording the route."""
        path = self.resolve_path(message.sender, message.recipient)
        message.metadata["route"] = list(path)
        target = self.agents[path[-1]]
        return await target.deliver(message)

    async def broadcast(self, sender_id: str, content: Any) -> List[Message]:
        """Fan a request out to the sender's direct neighbors."""
        self._require(sender_id)
        replies: List[Message] = []
        for neighbor in sorted(self._adjacency[sender_id]):
            fanout = Message(sender=sender_id, recipient=neighbor, content=content)
            reply = await self.agents[neighbor].deliver(fanout)
            if reply is not None:
                replies.append(reply)
        return replies

    def draw(self) -> List[str]:
        """Render edges as sorted 'a <-> b' lines for quick debugging."""
        lines: List[str] = []
        for source in sorted(self._adjacency):
            for target in sorted(self._adjacency[source]):
                pair = f"{source} <-> {target}"
                if pair in lines or f"{target} <-> {source}" in lines:
                    continue
                symmetric = source in self._adjacency.get(target, set())
                arrow = "<->" if symmetric else "-->"
                lines.append(f"{source} {arrow} {target}")
        return lines

    def _require(self, agent_id: str) -> None:
        if agent_id not in self.agents:
            raise KeyError(f"unknown agent '{agent_id}'")

    def __len__(self) -> int:
        return len(self.agents)

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self.agents

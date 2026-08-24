"""Episodic memory: goal-scoped event sequences with replay."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.memory.store import MemoryEntry, MemoryStore


class EpisodeError(RuntimeError):
    """Raised for invalid episode lifecycle operations."""


class Episode:
    """A bounded sequence of events tied to one goal or task."""

    def __init__(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.id = uuid.uuid4().hex
        self.title = title
        self.metadata = dict(metadata or {})
        self.events: List[Dict[str, Any]] = []
        self.outcome: Optional[str] = None
        self.success: bool = True
        self.started_at = time.time()
        self.ended_at: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return round(max(0.0, end - self.started_at), 3)

    def record(self, description: str, **data: Any) -> Dict[str, Any]:
        if not self.active:
            raise EpisodeError("cannot record events on a closed episode")
        event = {"description": description, "timestamp": time.time(), **data}
        self.events.append(event)
        return event

    def close(self, outcome: str, success: bool = True) -> None:
        if not self.active:
            raise EpisodeError("episode already closed")
        self.outcome = outcome
        self.success = success
        self.ended_at = time.time()

    def transcript(self) -> str:
        lines = [f"Episode '{self.title}' ({'open' if self.active else 'closed'})"]
        for index, event in enumerate(self.events, start=1):
            lines.append(f"  {index}. {event['description']}")
        if self.outcome is not None:
            status = "success" if self.success else "failure"
            lines.append(f"  outcome: {self.outcome} ({status})")
        return "\n".join(lines)


class EpisodicMemory(MemoryStore):
    """Ordered collection of episodes with keyword search across events."""

    name = "episodic"

    def __init__(self) -> None:
        self._episodes: "OrderedDict[str, Episode]" = OrderedDict()
        self._active: Optional[Episode] = None

    def begin(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        if self._active is not None:
            raise EpisodeError(f"episode '{self._active.title}' is still open; close it first")
        episode = Episode(title, metadata)
        self._episodes[episode.id] = episode
        self._active = episode
        return episode

    def record(self, description: str, **data: Any) -> Dict[str, Any]:
        if self._active is None:
            raise EpisodeError("no active episode; call begin() first")
        return self._active.record(description, **data)

    def end(self, outcome: str, success: bool = True) -> Episode:
        if self._active is None:
            raise EpisodeError("no active episode to end")
        self._active.close(outcome, success)
        finished = self._active
        self._active = None
        return finished

    def get(self, entry_id: str) -> Optional[Episode]:
        return self._episodes.get(entry_id)

    def list_episodes(self, successful_only: bool = False, limit: int = 50) -> List[Episode]:
        episodes = reversed(list(self._episodes.values()))
        if successful_only:
            episodes = (e for e in episodes if e.success and e.ended_at is not None)
        return list(episodes)[:limit]

    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        results: List[MemoryEntry] = []
        for episode in reversed(list(self._episodes.values())):
            haystack = " ".join([episode.title] + [e["description"] for e in episode.events])
            if query.strip().lower() in haystack.lower():
                results.append(
                    MemoryEntry(
                        content=episode.transcript(),
                        metadata={"episode_id": episode.id, "title": episode.title},
                        importance=0.8,
                    )
                )
            if len(results) >= k:
                break
        return results

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> MemoryEntry:
        episode = Episode(content.splitlines()[0][:120] if content else "(untitled)", metadata)
        for line in content.splitlines()[1:]:
            if line.strip():
                episode.record(line.strip())
        episode.close(kwargs.pop("outcome", "archived"), kwargs.pop("success", True))
        self._episodes[episode.id] = episode
        return MemoryEntry(
            id=episode.id,
            content=episode.transcript(),
            metadata={"title": episode.title},
            timestamp=episode.started_at,
        )

    def delete(self, entry_id: str) -> bool:
        if entry_id in self._episodes:
            if self._active and self._active.id == entry_id:
                self._active = None
            del self._episodes[entry_id]
            return True
        return False

    def clear(self) -> None:
        self._episodes.clear()
        self._active = None

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def has_active(self) -> bool:
        return self._active is not None

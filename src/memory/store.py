"""Memory store abstractions shared by all memory implementations."""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single stored memory item."""

    content: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    embedding: Optional[List[float]] = None

    def matches(self, keyword: str) -> bool:
        """Case-insensitive substring test against the content."""
        return keyword.strip().lower() in self.content.lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
            "importance": self.importance,
        }


class MemoryStore(ABC):
    """Common interface implemented by every memory backend."""

    name: str = "memory"

    @abstractmethod
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> MemoryEntry:
        """Store new content and return the resulting entry."""

    @abstractmethod
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Fetch an entry by identifier."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """Return up to ``k`` entries relevant to ``query``."""

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Remove an entry; returns True when something was deleted."""

    @abstractmethod
    def clear(self) -> None:
        """Drop every stored entry."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of entries currently held."""

    def stats(self) -> Dict[str, Any]:
        return {"store": self.name, "size": self.size}

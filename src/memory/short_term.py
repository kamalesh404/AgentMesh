"""Short-term conversation memory with context-window budgeting."""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from src.memory.store import MemoryEntry, MemoryStore


class ConversationBuffer(MemoryStore):
    """Bounded FIFO buffer of dialogue turns for the current session."""

    name = "short_term"

    def __init__(self, max_messages: int = 200, max_context_chars: int = 6000) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        self.max_context_chars = max_context_chars
        self._buffer: Deque[MemoryEntry] = deque(maxlen=max_messages)

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> MemoryEntry:
        role = kwargs.pop("role", "user")
        entry = MemoryEntry(content=content, metadata={"role": role, **(metadata or {})}, **kwargs)
        self._buffer.append(entry)
        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return next((entry for entry in self._buffer if entry.id == entry_id), None)

    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        hits = [entry for entry in reversed(self._buffer) if entry.matches(query)]
        return hits[:k]

    def delete(self, entry_id: str) -> bool:
        for index, entry in enumerate(self._buffer):
            if entry.id == entry_id:
                del self._buffer[index]
                return True
        return False

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)

    def messages(self) -> List[Dict[str, str]]:
        """Return the buffer as an OpenAI-style transcript."""
        return [
            {"role": str(entry.metadata.get("role", "user")), "content": entry.content}
            for entry in self._buffer
        ]

    def build_context(self, budget_chars: Optional[int] = None) -> str:
        """Render recent turns newest-first within a character budget.

        The oldest lines are dropped first so prompts stay inside the model's
        usable window while always preserving the latest exchange.
        """
        budget = self.max_context_chars if budget_chars is None else max(1, budget_chars)
        lines: List[str] = []
        total = 0
        for entry in reversed(self._buffer):
            line = f"{entry.metadata.get('role', 'user')}: {entry.content}"
            if lines and total + len(line) + 1 > budget:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(reversed(lines))

    def summarize(
        self,
        summarizer: Callable[[str], str],
        keep_last: int = 8,
        role: str = "system",
    ) -> Optional[MemoryEntry]:
        """Compress older turns into a single summary entry in place."""
        if len(self._buffer) <= keep_last:
            return None
        older = list(self._buffer)[:-keep_last]
        recent = list(self._buffer)[-keep_last:]
        transcript = "\n".join(f"{e.metadata.get('role', 'user')}: {e.content}" for e in older)
        summary = summarizer(transcript).strip() or "(empty summary)"
        replacement = MemoryEntry(
            content=f"[summary of {len(older)} earlier turns] {summary}",
            metadata={"role": role, "summarized_from": len(older)},
            importance=0.9,
        )
        self._buffer.clear()
        self._buffer.append(replacement)
        for entry in recent:
            self._buffer.append(entry)
        return replacement

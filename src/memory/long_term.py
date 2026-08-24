"""Long-term vector memory with cosine retrieval and JSONL persistence."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.memory.store import MemoryEntry, MemoryStore
from src.providers.base import deterministic_embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorMemory(MemoryStore):
    """Embedding-backed store supporting semantic top-k search.

    Args:
        embed_fn: Callable turning text into a vector; defaults to the
            framework's deterministic hashing embedder so the store works
            offline without any provider credentials.
        persist_path: Optional JSONL file for :meth:`save`/:meth:`load`.
    """

    name = "long_term"

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        dim: int = 64,
        persist_path: Optional[str | Path] = None,
    ) -> None:
        self._embed_fn = embed_fn or (lambda text: deterministic_embedding(text, dim=dim))
        self.persist_path = Path(persist_path) if persist_path else None
        self._entries: Dict[str, MemoryEntry] = {}
        self._order: List[str] = []
        self._fingerprints: set[str] = set()

    def _embed(self, text: str) -> List[float]:
        return self._embed_fn(text)

    @staticmethod
    def _fingerprint(content: str) -> str:
        return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> MemoryEntry:
        fingerprint = self._fingerprint(content)
        existing_id = kwargs.pop("existing_id", None)
        if existing_id is None and fingerprint in self._fingerprints:
            return self._find_by_fingerprint(fingerprint)
        entry = MemoryEntry(
            content=content,
            metadata=dict(metadata or {}),
            embedding=self._embed(content),
            **kwargs,
        )
        self._entries[entry.id] = entry
        self._order.append(entry.id)
        self._fingerprints.add(fingerprint)
        return entry

    def _find_by_fingerprint(self, fingerprint: str) -> MemoryEntry:
        for entry_id in reversed(self._order):
            entry = self._entries[entry_id]
            if self._fingerprint(entry.content) == fingerprint:
                return entry
        raise KeyError(fingerprint)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        self._order.remove(entry_id)
        return True

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()
        self._fingerprints.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        """Rank entries by cosine similarity to the embedded query."""
        if not self._order:
            return []
        query_vector = self._embed(query)
        scored = [
            (cosine_similarity(query_vector, entry.embedding or []), entry)
            for entry_id in self._order
            for entry in [self._entries[entry_id]]
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: List[MemoryEntry] = []
        for score, entry in scored[:k]:
            annotated = replace(entry, metadata={**entry.metadata, "score": round(score, 4)})
            results.append(annotated)
        return results

    def save(self, path: Optional[str | Path] = None) -> Path:
        """Persist all entries as JSON lines; returns the written path."""
        target = Path(path) if path else self.persist_path
        if target is None:
            raise ValueError("no persist_path configured")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for entry_id in self._order:
                entry = self._entries[entry_id]
                record = {
                    "id": entry.id,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp,
                    "importance": entry.importance,
                    "embedding": entry.embedding,
                }
                handle.write(json.dumps(record) + "\n")
        return target

    def load(self, path: Optional[str | Path] = None) -> int:
        """Merge previously saved entries; returns the number loaded."""
        target = Path(path) if path else self.persist_path
        if target is None or not target.exists():
            return 0
        loaded = 0
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                entry = MemoryEntry(
                    id=record["id"],
                    content=record["content"],
                    metadata=record.get("metadata", {}),
                    timestamp=record.get("timestamp", 0.0),
                    importance=record.get("importance", 0.5),
                    embedding=record.get("embedding"),
                )
                if entry.id not in self._entries:
                    self._entries[entry.id] = entry
                    self._order.append(entry.id)
                    loaded += 1
        return loaded

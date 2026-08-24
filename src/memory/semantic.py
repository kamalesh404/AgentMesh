"""Semantic memory: a lightweight knowledge graph of triples."""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Hashable, List, Optional, Set, Tuple

from src.memory.store import MemoryEntry, MemoryStore


@dataclass(frozen=True)
class Triple:
    """A directed subject-predicate-object fact."""

    subject: str
    predicate: str
    objekt: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def render(self) -> str:
        return f"{self.subject} -[{self.predicate}]-> {self.objekt}"


class KnowledgeGraphMemory(MemoryStore):
    """In-memory triple store with neighborhood and path queries."""

    name = "semantic"

    def __init__(self) -> None:
        self._triples: Dict[str, Triple] = {}
        self._adjacency: Dict[str, Set[Tuple[str, str]]] = {}

    def add_triple(
        self,
        subject: str,
        predicate: str,
        objekt: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Triple:
        for existing in self._triples.values():
            if (existing.subject, existing.predicate, existing.objekt) == (
                subject.strip().lower(),
                predicate.strip().lower(),
                objekt.strip().lower(),
            ):
                return existing
        triple = Triple(subject=subject.strip(), predicate=predicate.strip(), objekt=objekt.strip())
        self._triples[triple.id] = triple
        self._adjacency.setdefault(triple.subject.lower(), set()).add(
            (triple.predicate.lower(), triple.objekt.lower())
        )
        return triple

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> MemoryEntry:
        parts = [part.strip() for part in content.split("|")]
        if len(parts) != 3 or not all(parts):
            raise ValueError("expected content formatted as 'subject | predicate | object'")
        triple = self.add_triple(*parts)
        return MemoryEntry(
            id=triple.id,
            content=triple.render(),
            metadata={"kind": "triple", **(metadata or {})},
        )

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        triple = self._triples.get(entry_id)
        if triple is None:
            return None
        return MemoryEntry(id=triple.id, content=triple.render(), metadata={"kind": "triple"})

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Triple]:
        """Return triples matching the given non-None positions."""
        wanted_subject = subject.strip().lower() if subject else None
        wanted_predicate = predicate.strip().lower() if predicate else None
        wanted_object = obj.strip().lower() if obj else None
        results: List[Triple] = []
        for triple in self._triples.values():
            if wanted_subject and triple.subject.lower() != wanted_subject:
                continue
            if wanted_predicate and triple.predicate.lower() != wanted_predicate:
                continue
            if wanted_object and triple.objekt.lower() != wanted_object:
                continue
            results.append(triple)
        return results

    def neighbors(self, entity: str, depth: int = 1) -> Set[str]:
        """Collect entities reachable within ``depth`` undirected hops."""
        frontier: Set[str] = {entity.strip().lower()}
        seen: Set[str] = set(frontier)
        for _ in range(max(0, depth)):
            next_frontier: Set[str] = set()
            for node in frontier:
                next_frontier.update(self._adjacent_entities(node))
            next_frontier -= seen
            seen |= next_frontier
            frontier = next_frontier
        seen.discard(entity.strip().lower())
        return seen

    def _adjacent_entities(self, node_lower: str) -> Set[str]:
        adjacent: Set[str] = set()
        for _predicate, target in self._adjacency.get(node_lower, set()):
            adjacent.add(target)
        for source_lower, edges in self._adjacency.items():
            if source_lower == node_lower:
                continue
            for _, target in edges:
                if target == node_lower:
                    adjacent.add(source_lower)
        return adjacent

    def path(self, start: str, goal: str) -> Optional[List[str]]:
        """Shortest entity chain between two nodes via BFS, or None."""
        start_key, goal_key = start.strip().lower(), goal.strip().lower()
        if start_key == goal_key:
            return [start]
        visited: Set[Hashable] = {start_key}
        queue: Deque[List[str]] = deque([[start]])
        while queue:
            trail = queue.popleft()
            current = trail[-1].lower()
            for neighbor in sorted(self._adjacent_entities(current)):
                original = self._display_name(neighbor)
                if neighbor in visited:
                    continue
                extended = trail + [original]
                if neighbor == goal_key:
                    return extended
                visited.add(neighbor)
                queue.append(extended)
        return None

    def _display_name(self, lowered: str) -> str:
        for triple in self._triples.values():
            if triple.subject.lower() == lowered:
                return triple.subject
        return lowered.title()

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self._triples:
            return False
        del self._triples[entry_id]
        self._reindex()
        return True

    def clear(self) -> None:
        self._triples.clear()
        self._adjacency.clear()

    def _reindex(self) -> None:
        self._adjacency.clear()
        for triple in self._triples.values():
            self._adjacency.setdefault(triple.subject.lower(), set()).add(
                (triple.predicate.lower(), triple.objekt.lower())
            )

    def search(self, query: str, k: int = 5) -> List[MemoryEntry]:
        terms = query.lower().split()
        scored: List[Tuple[int, MemoryEntry]] = []
        for triple in self._triples.values():
            haystack = triple.render().lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, MemoryEntry(id=triple.id, content=triple.render())))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    def clear_all(self) -> None:
        self.clear()

    @property
    def size(self) -> int:
        return len(self._triples)

    @property
    def facts(self) -> List[str]:
        return [triple.render() for triple in self._triples.values()]

"""Tests for short-term, long-term, episodic, and semantic memory."""
from __future__ import annotations

import pytest

from src.memory.episodic import EpisodeError, EpisodicMemory
from src.memory.long_term import VectorMemory, cosine_similarity
from src.memory.semantic import KnowledgeGraphMemory
from src.memory.short_term import ConversationBuffer


def test_short_term_trim_and_context() -> None:
    buffer = ConversationBuffer(max_messages=3)
    for index in range(5):
        buffer.add(f"message {index}", role="user")
    assert buffer.size == 3
    context = buffer.build_context()
    assert "message 4" in context and "message 3" in context
    assert "message 0" not in context and "message 1" not in context


def test_short_term_budget_keeps_latest() -> None:
    buffer = ConversationBuffer(max_messages=50, max_context_chars=40)
    buffer.add("x" * 30, role="user")
    buffer.add("y" * 30, role="assistant")
    context = buffer.build_context(budget_chars=35)
    assert context.count("y") == 30
    assert "xxxx" not in context


def test_vector_memory_ranking() -> None:
    memory = VectorMemory(embed_fn=lambda text: [1.0, 0.0] if "apple" in text else [0.0, 1.0])
    memory.add("apple pie recipe", metadata={"tag": "food"})
    memory.add("quantum physics paper")
    hits = memory.search("apple crumble dessert", k=1)
    assert len(hits) == 1
    assert "apple" in hits[0].content
    assert hits[0].metadata["score"] == pytest.approx(1.0)


def test_vector_memory_dedupe_and_persistence(tmp_path) -> None:
    memory = VectorMemory(persist_path=tmp_path / "mem.jsonl")
    first = memory.add("unique content here")
    duplicate = memory.add("unique content here")
    assert first.id == duplicate.id and memory.size == 1
    memory.add("second entry")
    path = memory.save()
    reloaded = VectorMemory()
    loaded = reloaded.load(path)
    assert loaded == 2
    assert reloaded.size == 2


def test_cosine_similarity_basics() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], []) == 0.0


def test_episodic_lifecycle_and_search() -> None:
    memory = EpisodicMemory()
    with pytest.raises(EpisodeError):
        memory.record("orphan event")
    episode = memory.begin("deploy service", metadata={"env": "prod"})
    memory.record("ran tests")
    memory.record("shipped build")
    memory.end("deployed successfully")
    assert not episode.active
    assert episode.duration_seconds >= 0
    hits = memory.search("tests")
    assert len(hits) == 1
    assert "ran tests" in hits[0].content
    assert memory.list_episodes()[0].success


def test_semantic_triples_queries_and_paths() -> None:
    graph = KnowledgeGraphMemory()
    graph.add("alice | works_at | acme")
    graph.add_triple("acme", "located_in", "berlin")
    graph.add_triple("bob", "works_at", "acme")
    assert len(graph.query(subject="alice")) == 1
    assert len(graph.query(predicate="works_at")) == 2
    assert {"acme", "berlin"} <= graph.neighbors("alice", depth=2)
    path = graph.path("alice", "berlin")
    assert path is not None and path[0].lower() == "alice" and path[-1].lower() == "berlin"
    assert graph.path("alice", "nowhere") is None
    assert graph.delete(graph.query(subject="bob")[0].id)
    assert graph.query(subject="bob") == []

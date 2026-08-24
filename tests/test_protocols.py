"""Tests for the message protocol, handshakes, and consensus."""
from __future__ import annotations

import pytest

from src.protocols.consensus import Cluster, Role
from src.protocols.handshake import (
    CapabilityDescriptor,
    HandshakeError,
    HandshakeManager,
)
from src.protocols.message import Message, MessageType, content_text


def test_message_json_roundtrip() -> None:
    original = Message(
        sender="agent.a",
        recipient="agent.b",
        content={"text": "hello", "nested": {"k": 1}},
        metadata={"intent": "task"},
        ttl=60,
    )
    clone = Message.from_json(original.to_json())
    assert clone.id == original.id
    assert clone.sender == original.sender
    assert clone.recipient == original.recipient
    assert clone.content == original.content
    assert clone.metadata == original.metadata
    assert clone.type is MessageType.REQUEST


def test_reply_links_correlation() -> None:
    request = Message(sender="a", recipient="b", content="question")
    response = request.reply("answer")
    assert response.type is MessageType.RESPONSE
    assert response.correlation_id == request.correlation_id == request.id
    assert response.reply_to == request.id


def test_forwarding_records_hops() -> None:
    first = Message(sender="client", recipient="router", content="payload")
    relayed = first.forwarded("worker")
    assert relayed.recipient == "worker"
    assert relayed.metadata["hops"] == ["worker"]
    assert relayed.correlation_id == first.correlation_id


def test_validation_and_expiry() -> None:
    with pytest.raises(ValueError):
        Message(sender="", recipient="b", content="x").validate()
    with pytest.raises(ValueError):
        Message(sender="a", recipient="", content="x").validate()
    stale = Message(sender="a", recipient="b", content="x", ttl=1, created_at=0.0)
    assert stale.expired and stale.age > 0


def test_checksum_is_deterministic() -> None:
    message = Message(sender="a", recipient="b", content="same")
    assert message.checksum() == Message.from_json(message.to_json()).checksum()


def test_content_text_extraction() -> None:
    assert content_text("plain") == "plain"
    assert content_text({"text": "chosen"}) == "chosen"
    assert content_text({"other": 1}) == '{"other": 1}'
    assert content_text(42) == "42"


def _descriptor(agent_id: str, caps, version: str = "1.0") -> CapabilityDescriptor:
    return CapabilityDescriptor(agent_id=agent_id, capabilities=frozenset(caps), version=version)


def test_handshake_negotiation_and_discovery() -> None:
    local = HandshakeManager(_descriptor("orchestrator", ["search", "summarize"]))
    peer_one = _descriptor("searcher", ["search", "crawl"])
    peer_two = _descriptor("writer", ["summarize", "draft"])
    local.register(peer_one)
    local.register(peer_two)
    assert local.discover("search") == ["searcher"]
    agreement = local.negotiate("searcher", required=["search"])
    assert agreement.agreed_capabilities == frozenset({"search"})
    with pytest.raises(HandshakeError):
        local.negotiate("writer", required=["crawl"])
    with pytest.raises(HandshakeError):
        local.register(_descriptor("orchestrator", []))


def test_handshake_rejects_major_version_mismatch() -> None:
    manager = HandshakeManager(_descriptor("local", ["chat"], version="2.0"))
    manager.register(_descriptor("peer", ["chat"], version="1.5"))
    with pytest.raises(HandshakeError):
        manager.negotiate("peer")


def test_cluster_elects_leader_and_replicates() -> None:
    cluster = Cluster(["node-1", "node-2", "node-3"])
    assert cluster.hold_election("node-1") is True
    leader = cluster.leader()
    assert leader is not None and leader.node_id == "node-1"
    assert leader.role is Role.LEADER
    index = cluster.replicate("set x = 1")
    assert index is not None
    follower = cluster.nodes["node-2"]
    assert follower.log[-1].command == "set x = 1"
    assert leader.committed() == ["set x = 1"]


def test_cluster_requires_majority_nodes() -> None:
    with pytest.raises(ValueError):
        Cluster(["solo"])

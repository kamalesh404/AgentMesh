"""Tests for agent lifecycle, chat behavior, and coding utilities."""
from __future__ import annotations

import pytest

from src.agents.base import AgentLifecycleError, AgentState
from src.agents.coding import apply_patch, extract_code
from src.agents.conversational import ChatAgent
from src.protocols.message import Message, MessageType

from .conftest import run


def test_lifecycle_happy_path(provider) -> None:
    agent = ChatAgent("alpha", provider=provider)
    assert agent.state is AgentState.CREATED
    run(agent.initialize())
    assert agent.state is AgentState.READY
    run(agent.stop())
    assert agent.state is AgentState.STOPPED


def test_invalid_transition_rejected(make_chat) -> None:
    agent = make_chat("beta")
    with pytest.raises(AgentLifecycleError):
        agent.transition(AgentState.RUNNING)
    run(agent.initialize())
    with pytest.raises(AgentLifecycleError):
        agent.transition(AgentState.CREATED)
    run(agent.stop())


def test_chat_reply_and_transcript(provider) -> None:
    agent = ChatAgent("gamma", provider=provider)
    run(agent.initialize())
    message = Message(sender="user", recipient=agent.agent_id, content="hello world")
    reply = run(agent.deliver(message))
    assert reply is not None and reply.type is MessageType.RESPONSE
    assert reply.recipient == "user"
    assert "[mock/" in reply.content
    roles = [entry["role"] for entry in agent.transcript]
    assert roles == ["system", "user", "assistant"]
    assert agent.processed == 1
    run(agent.stop())


def test_reset_control_message(provider) -> None:
    agent = ChatAgent("delta", provider=provider)
    run(agent.initialize())
    first = Message(sender="user", recipient=agent.agent_id, content="hi")
    run(agent.deliver(first))
    control = Message(
        sender="user",
        recipient=agent.agent_id,
        content="reset",
        type=MessageType.CONTROL,
    )
    ack = run(agent.deliver(control))
    assert ack is not None and ack.type is MessageType.CONTROL
    assert len(agent.transcript) == 1
    run(agent.stop())


def test_deliver_captures_provider_failure() -> None:
    class ExplodingProvider:
        async def complete(self, *args, **kwargs):
            raise RuntimeError("boom")

        name = "exploding"

    agent = ChatAgent("epsilon", provider=ExplodingProvider())
    message = Message(sender="user", recipient=agent.agent_id, content="anything")
    reply = run(agent.deliver(message))
    assert reply is not None and reply.type is MessageType.ERROR
    assert "boom" in reply.content["error"]
    assert agent.failed == 1


def test_apply_patch_and_extract_code() -> None:
    original = "def add(a, b):\n    return a - b\n"
    patch = (
        "--- a/math.py\n+++ b/math.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    patched = apply_patch(original, patch)
    assert "return a + b" in patched
    assert "return a - b" not in patched
    assert patched.startswith("def add(a, b):")

    fenced = "Here you go:\n```python\nprint('hi')\n```\nDone."
    assert extract_code(fenced) == "print('hi')"

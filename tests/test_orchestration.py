"""Tests for pipelines, supervision, graphs, and debate orchestration."""
from __future__ import annotations

import pytest

from src.agents.base import BaseAgent
from src.orchestration.debate import DebateOrchestrator
from src.orchestration.graph import AgentGraph, RouteNotFoundError
from src.orchestration.pipeline import Pipeline, PipelineError
from src.orchestration.supervisor import Supervisor, SupervisorError
from src.protocols.message import Message

from .conftest import run


class EchoAgent(BaseAgent):
    role = "echo"

    async def process_message(self, message: Message) -> Message:
        return message.reply(f"echo:{message.content}")


def test_pipeline_sequential_stages() -> None:
    stage_one = EchoAgent("one")
    stage_two = EchoAgent("two")
    pipeline = Pipeline(name="demo")
    pipeline.add_stage(stage_one)
    pipeline.add_stage(stage_two, transform=lambda out, ctx: f"{out}!")
    context = run(pipeline.run("hello"))
    assert context["steps"]["echo.one"] == "echo:hello"
    assert context["output"] == "echo:echo:hello!"


def test_pipeline_requires_stages() -> None:
    pipeline = Pipeline()
    with pytest.raises(PipelineError):
        run(pipeline.run("payload"))


def test_supervisor_delegates_by_capability() -> None:
    supervisor = Supervisor("boss")
    math_worker = EchoAgent("mathbot")
    generic_worker = EchoAgent("generic")
    supervisor.register(math_worker, capabilities={"math", "calc"})
    supervisor.register(generic_worker, capabilities={"text"})
    result = run(supervisor.delegate("compute 2+2", required_capabilities=["math"]))
    assert result == "echo:compute 2+2"
    with pytest.raises(SupervisorError):
        run(supervisor.delegate("translate this", required_capabilities=["french"]))


def test_graph_routes_through_intermediary() -> None:
    graph = AgentGraph()
    agent_a = EchoAgent("A")
    agent_b = EchoAgent("B")
    agent_c = EchoAgent("C")
    for node in (agent_a, agent_b, agent_c):
        graph.add(node)
    graph.connect(agent_a.agent_id, agent_b.agent_id)
    graph.connect(agent_b.agent_id, agent_c.agent_id)
    message = Message(sender=agent_a.agent_id, recipient=agent_c.agent_id, content="ping")
    reply = run(graph.route(message))
    assert reply is not None and reply.content == "echo:ping"
    assert reply.metadata["route"] == [agent_a.agent_id, agent_b.agent_id, agent_c.agent_id]
    with pytest.raises(RouteNotFoundError):
        isolated = EchoAgent("D")
        graph.add(isolated)
        run(
            graph.route(
                Message(sender=agent_a.agent_id, recipient=isolated.agent_id, content="x")
            )
        )


def test_debate_produces_turns_and_result(make_chat) -> None:
    debater_one = make_chat("debater-one")
    debater_two = make_chat("debater-two")
    orchestrator = DebateOrchestrator([debater_one, debater_two], max_rounds=1)
    result = run(orchestrator.run("Should we adopt microservices?"))
    assert result.rounds_used == 1
    assert len(result.turns) == 2
    assert result.winner in {debater_one.agent_id, debater_two.agent_id}


def test_supervisor_status_intent_acknowledged() -> None:
    supervisor = Supervisor("chief")
    status = Message(
        sender="worker.x",
        recipient=supervisor.agent_id,
        content="all good",
        metadata={"intent": "status"},
    )
    ack = run(supervisor.deliver(status))
    assert ack is not None and ack.content == {"acknowledged": True}

# Orchestration Guide

AgentMesh ships four coordination patterns. All of them operate on ordinary
agents and communicate through the standard `Message` protocol, so any
`BaseAgent` subclass works without modification.

## Pipeline — sequential stages

```python
from src.orchestration.pipeline import Pipeline

pipeline = Pipeline(name="author-review")
pipeline.add_stage(writer_agent)
pipeline.add_stage(critic_agent)
pipeline.add_stage(editor_agent, transform=lambda out, ctx: out.upper())

context = await pipeline.run("Explain eventual consistency")
final_output = context["output"]
per_stage_timings = context["stage_results"]
```

Each stage receives the previous output as message content; transforms reshape
values between stages (they also receive the shared context dict). A stage
that errors produces a `PipelineError` naming the failing agent.

## Supervisor — capability-based delegation

```python
from src.orchestration.supervisor import Supervisor

supervisor = Supervisor("chief-of-staff", dispatch_timeout=15)
supervisor.register(math_agent, capabilities={"math", "calc"})
supervisor.register(writer_agent, capabilities={"prose"})

answer = await supervisor.delegate("compute 17 * 23", required_capabilities=["math"])
```

Worker selection scores candidates by capability overlap, then by current
in-flight load, then deterministically by ID. Timeouts and worker failures
raise `SupervisorError` with the offending worker named.

## AgentGraph — routed communication

```python
from src.orchestration.graph import AgentGraph

graph = AgentGraph()
for node in (client, router, planner, executor):
    graph.add(node)

graph.connect(client.agent_id, router.agent_id)
graph.connect(router.agent_id, planner.agent_id)
graph.connect(planner.agent_id, executor.agent_id)

reply = await graph.route(Message(sender=client.agent_id,
                                  recipient=executor.agent_id,
                                  content="ship it"))
print(reply.metadata["route"])   # full hop path recorded on delivery
```

Routing performs a BFS over the topology and stamps the resolved path into
message metadata before delivering to the final recipient. `broadcast()`
fans a request out to an agent's direct neighbors.

## DebateOrchestrator — consensus through argument

```python
from src.orchestration.debate import DebateOrchestrator

debate = DebateOrchestrator(
    participants=[agent_a, agent_b, agent_c],
    judge=None,                 # optional judge agent picks winners
    max_rounds=3,
    agreement_threshold=0.55,
)
result = await debate.run("Should we migrate to event sourcing?")
print(result.consensus, result.winner, result.rounds_used)
print(result.transcript_text())
```

Each round collects one position per participant (the transcript so far is
passed in message metadata). Without a judge, positions are compared via
Jaccard word overlap: the speaker whose position best matches the field wins,
and consensus is declared when agreement clears the threshold.

## Choosing a pattern

| Need | Pattern |
| --- | --- |
| Fixed multi-step workflow | `Pipeline` |
| Route tasks to specialists | `Supervisor` |
| Peer-to-peer messaging topologies | `AgentGraph` |
| Decisions requiring deliberation | `DebateOrchestrator` |

All patterns are async-first and compose: a pipeline stage can itself be a
supervisor, and debate participants may be pipelines.

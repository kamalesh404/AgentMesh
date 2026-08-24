# AgentMesh Architecture

AgentMesh is organized as seven cooperating layers. Dependencies flow strictly
downward: orchestration depends on agents, agents depend on tools/memory/
protocols/providers, and everything rests on shared utilities.

```
                ┌─────────────────────────────┐
                │           cli/              │  click commands
                └─────────────┬───────────────┘
                ┌─────────────▼───────────────┐
                │      orchestration/         │  graph · supervisor · debate · pipeline
                └─────────────┬───────────────┘
                ┌─────────────▼───────────────┐
                │          agents/            │  lifecycle state machine + roles
                ├──┬──────────┬────────┬───────┤
                │tools│memory│providers│protocols│  capabilities, recall, models, wire format
                └──┴──────────┴────────┴───────┤
                ┌─────────────▼───────────────┐
                │           utils/            │  logging · retry · config · serialization
                └─────────────────────────────┘
```

## The agent core

`BaseAgent` (src/agents/base.py) implements:

1. **A validated state machine** — `created → initializing → ready → running → paused → stopped`, with `error` reachable from any live state and recovery via re-initialization. Illegal transitions raise `AgentLifecycleError`.
2. **Two delivery modes** — `deliver()` handles a message synchronously (used by tests and orchestrators), while `start()` spawns a background consumer draining an `asyncio.Queue` inbox.
3. **Failure containment** — exceptions inside `process_message` are converted into `ERROR`-type reply messages rather than crashing the caller.

## Message flow

Messages (`src/protocols/message.py`) are dataclasses with sender, recipient,
correlation IDs, TTLs, and metadata. Replies preserve the correlation chain,
and `Message.forwarded()` records hop history for graph relaying.

```
sender ──▶ AgentGraph.route() ──BFS path──▶ recipient.deliver()
                    │
                    └─ metadata["route"] = [hops...]
```

## Memory layering

| Store | Purpose | Retrieval |
| --- | --- | --- |
| `ConversationBuffer` | current dialogue window | newest-first, char budget |
| `VectorMemory` | durable knowledge | cosine top-k over embeddings |
| `EpisodicMemory` | goal-scoped event trails | keyword search + replay |
| `KnowledgeGraphMemory` | entity/relation facts | triple queries, BFS paths |

Stores implement a common `MemoryStore` interface, so agents accept any of
them interchangeably via their `memory` constructor argument.

## Providers

All model access funnels through `BaseProvider.complete()` / `.embed()`.
Concrete providers only implement the raw backend call; cross-cutting concerns
live in the base class:

- token-bucket rate limiting (`TokenBucketRateLimiter`)
- retry with exponential backoff and jitter (`utils.retry`)
- usage accounting and latency stats

The offline `MockProvider` keeps the entire stack testable without credentials.

## Consensus primitives

`src/protocols/consensus.py` provides a Raft-inspired cluster: nodes hold
terms, vote for candidates whose logs are at least as up-to-date, replicate
append-only entries, and commit once a majority acknowledges. The in-memory
transport makes it suitable for coordinating local agent swarms or as a
reference implementation before wiring real RPC.

## Extension points

- **New agent**: subclass `BaseAgent`, set `role`, implement `process_message`.
- **New tool**: subclass `Tool`, declare `parameters`, implement `_run`; validation, timing, and OpenAI-schema export come free.
- **New provider**: subclass `BaseProvider`, implement `_complete` and `embed`.
- **New memory store**: subclass `MemoryStore` and implement five methods.

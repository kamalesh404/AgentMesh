# AgentMesh

[![CI](https://github.com/agentmesh/agentmesh/actions/workflows/ci.yml/badge.svg)](https://github.com/agentmesh/agentmesh/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-ruff-261230)
![Status](https://img.shields.io/badge/status-beta-orange)

**AgentMesh** is a batteries-included Python framework for building, connecting, and
orchestrating LLM-powered agents. It ships first-class abstractions for agents,
tools, layered memory, wire protocols, model providers, and multi-agent
coordination patterns (supervision, debate, pipelines, and message graphs).

## Features

- **Agent lifecycle state machine** - created → initializing → ready → running → paused → stopped, with validated transitions and background inbox processing.
- **Five specialized agents** - conversational chat, code generation with patching, research with search synthesis, planning with goal trees, and rubric-based critique.
- **Tool system** - schema-validated tools with coercion, timing, error capture, and OpenAI-compatible function definitions: web search, sandboxed code execution, files, shell, HTTP APIs, and SQL databases.
- **Layered memory** - short-term conversation buffers, vector long-term store with persistence, episodic event replay, and a semantic knowledge graph.
- **Wire protocols** - typed messages with correlation IDs and TTLs, capability handshakes, and a Raft-inspired consensus module for replicated logs.
- **Pluggable providers** - OpenAI, Anthropic, and local Ollama behind one interface, with token-bucket rate limiting and exponential-backoff retries.
- **Orchestration patterns** - communication graphs with BFS routing, a capability-aware supervisor, multi-agent debate with consensus voting, and sequential pipelines.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                            CLI (click)                           │
├──────────────────────────────┬───────────────────────────────────┤
│        Orchestration         │           Protocols               │
│  graph · supervisor ·        │  message · handshake ·            │
│  debate · pipeline           │  consensus                        │
├──────────────────────────────┴───────────────────────────────────┤
│                          Agents                                  │
│  base · conversational · coding · research · planner · critic    │
├───────────────┬─────────────────────┬────────────────────────────┤
│     Tools     │       Memory        │          Providers         │
│ web_search    │ short_term          │ openai                     │
│ code_exec     │ long_term (vector)  │ anthropic                  │
│ file_ops      │ episodic            │ ollama                     │
│ shell / api   │ semantic (KG)       │ base + rate limiter        │
│ database      │                     │                            │
├───────────────┴─────────────────────┴────────────────────────────┤
│                    Utils: logging · retry · config · serialization│
└──────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
```

```python
import asyncio

from src.agents.conversational import ChatAgent
from src.agents.critic import CriticAgent
from src.orchestration.pipeline import Pipeline
from src.providers.base import MockProvider


async def main() -> None:
    writer = ChatAgent("writer", provider=MockProvider())
    reviewer = CriticAgent("reviewer", provider=MockProvider())
    await writer.initialize()
    await reviewer.initialize()

    pipeline = Pipeline(name="write-review")
    pipeline.add_stage(writer)
    pipeline.add_stage(reviewer)
    result = await pipeline.run("Draft a haiku about distributed systems")
    print(result["output"])


asyncio.run(main())
```

### CLI

```bash
agentmesh agent create --name scout --role research --provider mock
agentmesh agent list
agentmesh agent run --name scout --prompt "Summarize AgentMesh"
agentmesh session start --name demo
agentmesh session history
```

## Agents

| Agent              | Role        | Highlights                                              |
| ------------------ | ----------- | ------------------------------------------------------- |
| `ChatAgent`        | `chat`      | Context-window trimming, transcript, reset control      |
| `CodingAgent`      | `coder`     | Code generation, unified-diff patching, test execution  |
| `ResearchAgent`    | `researcher`| Search-tool integration, per-source summarization       |
| `PlannerAgent`     | `planner`   | Goal decomposition, dependency trees, topological plans |
| `CriticAgent`      | `critic`    | Heuristic + LLM review, severity scoring, approval gate |

## Configuration

Configuration merges defaults ← `agentmesh.yaml` ← environment variables using
the `AGENTMESH__SECTION__KEY` convention:

```bash
export AGENTMESH__LOG_LEVEL=DEBUG
export AGENTMESH__PROVIDERS__OPENAI__MODEL=gpt-4o-mini
```

## Testing

```bash
pytest tests -q
pytest tests --cov=src --cov-report=term-missing
```

## Roadmap

- [x] Message protocol with TTLs and correlation IDs
- [x] Raft-inspired consensus primitives
- [x] Debate-based consensus orchestration
- [ ] Persistent process supervision across restarts
- [ ] gRPC transport for remote agents
- [ ] Agent marketplace and signed capability manifests

## Contributing

Issues and pull requests are welcome. Run `make lint test` before submitting.

## License

MIT — see the project header in `setup.py`.

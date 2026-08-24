# Getting Started with AgentMesh

AgentMesh is a Python framework for building multi-agent LLM systems. This
guide takes you from a clean checkout to your first orchestrated agent run.

## Requirements

- Python 3.10, 3.11, or 3.12
- pip 23+
- Optional: Docker (for containerized code execution), Ollama (for local models)

## Installation

```bash
git clone https://github.com/agentmesh/agentmesh.git
cd agentmesh

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Verify the CLI works:

```bash
agentmesh --version
agentmesh --help
```

## Your first agent

Every agent needs a *provider* that generates text. The built-in
`MockProvider` requires no API key and echoes deterministic responses — ideal
for development:

```python
import asyncio

from src.agents.conversational import ChatAgent
from src.providers.base import MockProvider
from src.protocols.message import Message


async def main() -> None:
    agent = ChatAgent("scout", provider=MockProvider())
    await agent.initialize()

    request = Message(sender="you", recipient=agent.agent_id, content="Who are you?")
    reply = await agent.deliver(request)
    print(reply.content)

    await agent.stop()


asyncio.run(main())
```

## Switching to a real model

Set an environment variable and swap the provider class:

```bash
export OPENAI_API_KEY=sk-...      # or ANTHROPIC_API_KEY=...
```

```python
from src.providers.openai import OpenAIProvider          # GPT models + embeddings
from src.providers.anthropic import AnthropicProvider    # Claude models
from src.providers.ollama import OllamaProvider           # local llama3.1 etc.

provider = OpenAIProvider(model="gpt-4o-mini")
agent = ChatAgent("scout", provider=provider)
```

Rate limiting (token bucket) and retry-with-backoff are applied automatically.

## Adding tools

Tools are schema-validated callables. Attach them to agents that accept them:

```python
from src.agents.research import ResearchAgent
from src.tools.web_search import WebSearchTool
from src.providers.base import MockProvider

researcher = ResearchAgent(
    "scholar",
    provider=MockProvider(),
    search_tool=WebSearchTool(),   # needs TAVILY_API_KEY or SERPAPI_API_KEY
)
report = asyncio.run(researcher.research("vector databases"))
print(report["summary"])
```

## Memory

Give an agent persistent recall by passing a memory store:

```python
from src.memory.long_term import VectorMemory

memory = VectorMemory(persist_path="data/memory.jsonl")
agent = ChatAgent("scout", provider=provider, memory=memory)
```

## Configuration

Create `agentmesh.yaml` in your working directory or use env vars:

```yaml
log_level: INFO
providers:
  default: openai
  openai:
    model: gpt-4o-mini
orchestration:
  debate_rounds: 3
```

```bash
export AGENTMESH__LOG_LEVEL=DEBUG
export AGENTMESH__PROVIDERS__OPENAI__MODEL=gpt-4o-mini
```

Precedence: **defaults < agentmesh.yaml < environment variables**.

## Next steps

- Read [architecture.md](architecture.md) for how the pieces fit together.
- Explore [agents.md](agents.md) and [tools.md](tools.md) for deep dives.
- Run `pytest tests -q` to see the framework exercised end-to-end.

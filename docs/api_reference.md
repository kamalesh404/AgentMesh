# API Reference

Condensed reference for the public surface of AgentMesh. All async methods
return awaitables; constructors are synchronous.

## src.agents.base

```python
class AgentState(Enum): CREATED INITIALIZING READY RUNNING PAUSED STOPPED ERROR

class BaseAgent(ABC):
    role: str = "generic"
    def __init__(name: str, provider=None, memory=None, max_inbox: int = 256)
    def transition(next_state: AgentState) -> None
    async def initialize() -> None                 # CREATED -> READY
    async def start() / stop() / pause() / resume() -> None
    async def deliver(message: Message) -> Message | None
    async def send(message: Message) -> None       # enqueue for start()-loop
    @abstractmethod
    async def process_message(message: Message) -> Message | None
    def info() -> dict
```

Errors: `AgentError`, `AgentLifecycleError`.

## src.agents (concrete agents)

```python
ChatAgent(name, provider, memory=None, system_prompt=None,
          window_chars=4000, max_turns=200)
    .transcript -> list[dict]     .turns -> int     .reset()

CodingAgent(name, provider, executor=None)
    .generate_code(spec, language="python") -> str
    .explain_code(code) -> str
    .fix_with_patch(original, patch) -> str
    .run_tests(code=None, timeout=10.0) -> dict

ResearchAgent(name, provider, search_tool=None, max_sources=5)
    .research(topic) -> dict   # {topic, summary, sources, generated_at}

PlannerAgent(name, provider)
    .decompose(goal, n_subtasks=4) -> GoalTree
    .plan(goal, n_subtasks=4) -> list[str]

CriticAgent(name, provider, approval_threshold=75)
    .review(code) -> ReviewResult  # .score .approved .issues .summary
```

## src.tools.base

```python
@dataclass ToolResult: success output error duration_ms metadata
@dataclass Parameter(name, type="string", description="", required=True,
                     default=None, enum=None)

class Tool(ABC):
    name description parameters
    def validate(args: dict) -> dict
    async def run(**kwargs) -> ToolResult
    def schema() -> dict
    def openai_definition() -> dict

class ToolRegistry:
    register(tool) get(name) unregister(name) names()
    openai_definitions() describe()
```

## src.memory

```python
ConversationBuffer(max_messages=200, max_context_chars=6000)
    .add(content, role="user") .messages() .build_context(budget_chars=None)
    .summarize(summarizer, keep_last=8)

VectorMemory(embed_fn=None, dim=64, persist_path=None)
    .search(query, k=5)          # cosine top-k, score in metadata
    .save(path=None) .load(path=None)

EpisodicMemory()
    .begin(title) .record(description) .end(outcome, success=True)
    .list_episodes() .search(query, k=5)

KnowledgeGraphMemory()
    .add_triple(s, p, o) .query(subject=None, predicate=None, obj=None)
    .neighbors(entity, depth=1) .path(start, goal)
```

## src.orchestration

```python
Pipeline(name).add_stage(agent, transform=None)
Pipeline.run(payload) -> dict           # context with steps/output

Supervisor(name, dispatch_timeout=30.0)
    .register(worker, capabilities) .unregister(agent_id)
    .delegate(task, required_capabilities=None, timeout=None)

AgentGraph()
    .add(agent) .connect(a, b, bidirectional=True)
    .route(message) .broadcast(sender_id, content) .resolve_path(a, b)

DebateOrchestrator(participants, judge=None, max_rounds=3,
                   agreement_threshold=0.55)
    .run(topic) -> DebateResult
```

## src.protocols

```python
Message(sender, recipient, content, type=REQUEST, metadata={}, ttl=30)
    .reply(content) .forwarded(new_recipient) .checksum()
    .to_json()/from_json() .to_dict()/from_dict() .validate()

HandshakeManager(local: CapabilityDescriptor)
    .register(descriptor) .discover(capability) .peers
    .negotiate(peer_id, required) / await .handshake(...)

Cluster(node_ids)                       # >= 3 nodes
    .hold_election(candidate_id) .replicate(command) .leader()
```

## src.providers

```python
class BaseProvider(ABC):
    async complete(prompt, system=None, temperature=0.7, max_tokens=1024)
        -> Completion(text model usage latency_ms)
    async chat(messages, **kw) -> Completion
    async embed(texts: list[str]) -> list[list[float]]

MockProvider(model="mock-medium")               # offline deterministic
OpenAIProvider(api_key=None, model="gpt-4o-mini")
AnthropicProvider(api_key=None, model="claude-3-5-sonnet-latest")
OllamaProvider(model="llama3.1", base_url="http://localhost:11434")
```

## src.utils

```python
Config.load(path=None).get("providers.openai.model", default)
get_logger(name); new_correlation_id(); configure_logging(level, json_output=False)
RetryPolicy(max_attempts=3, base_delay=0.2); retry_async(fn, policy=...);
with_retry(policy)                      # decorator for async callables
dumps(obj)/loads(raw); dump_msgpack/load_msgpack; fingerprint(obj)
```

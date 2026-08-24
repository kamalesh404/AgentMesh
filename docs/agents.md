# Agents Reference

Every AgentMesh agent inherits from `BaseAgent` and participates in the same
lifecycle: `created → initializing → ready → running → paused → stopped`.

## Lifecycle in practice

```python
agent = ChatAgent("helper", provider=provider)
await agent.initialize()          # -> READY (runs setup hooks)
reply = await agent.deliver(msg)  # synchronous handling
await agent.start()               # -> RUNNING, spawns inbox consumer
await agent.send(other_msg)       # queued for background processing
await agent.pause()               # consumer parks until resume()
await agent.stop()                # drains and cancels the loop
```

Illegal transitions raise `AgentLifecycleError`; for example, you cannot jump
from `created` straight to `running`.

## ChatAgent — `src/agents/conversational.py`

Conversational agent with a memory-backed context window.

- Keeps a `ConversationBuffer` transcript including a system prompt entry.
- Trims prompts to `window_chars` characters, newest turns first.
- Responds to a `CONTROL` message containing `"reset"` by clearing history.

```python
from src.agents.conversational import ChatAgent

chat = ChatAgent(
    "support",
    provider=provider,
    system_prompt="You are a support engineer for ACME rockets.",
    window_chars=6000,
)
```

## CodingAgent — `src/agents/coding.py`

Code generation with patch application and sandboxed smoke tests.

- `generate_code(spec)` asks for a fenced block and extracts it.
- `fix_with_patch(original, patch)` applies tolerant unified diffs.
- `run_tests()` executes code via the `code_exec` tool (subprocess or Docker).
- Message intents: `generate` (default), `explain`, `patch`, `test`.

```python
from src.agents.coding import CodingAgent

coder = CodingAgent("builder", provider=provider)   # executor auto-attached
code = await coder.generate_code("CLI that sums numbers from stdin")
report = await coder.run_tests(code)
assert report["passed"]
```

## ResearchAgent — `src/agents/research.py`

Search-and-summarize pipeline over web sources.

- With a search tool, normalizes results and summarizes each source.
- Without one, degrades gracefully to an LLM-generated outline.
- Synthesizes per-source notes into an executive brief; stores it in memory.

```python
researcher = ResearchAgent("scholar", provider=provider,
                           search_tool=WebSearchTool(), max_sources=5)
report = await researcher.research("Raft vs Paxos")
print(report["summary"], report["sources"])
```

## PlannerAgent — `src/agents/planner.py`

Goal decomposition into dependency-ordered tasks.

- `decompose(goal, n_subtasks)` parses JSON task arrays from the LLM with
  robust fallbacks (bullet lines → generic template).
- `GoalTree.topo_order()` yields a valid execution order and detects cycles.
- `ready()` exposes tasks whose dependencies are satisfied.

```python
planner = PlannerAgent("strategist", provider=provider)
tree = await planner.decompose("Launch v2 of the docs site", n_subtasks=4)
for task_id in tree.topo_order():
    node = tree.tasks[task_id]
    print(node.id, node.title)
    tree.mark_complete(task_id)
```

## CriticAgent — `src/agents/critic.py`

Quality gate combining static heuristics with an LLM review note.

Heuristic penalties include bare `except:`, `eval(`, hardcoded credentials,
TODO/FIXME markers, debug prints, and long lines. The final score starts at
100 minus penalties; approval requires score ≥ threshold and zero criticals.

```python
critic = CriticAgent("gatekeeper", provider=provider, approval_threshold=80)
verdict = await critic.review(source_code)
print(verdict.score, verdict.approved, [i.message for i in verdict.blocking_issues])
```

## Composing agents

Agents are plain objects — pass them into orchestration primitives:

```python
pipeline = Pipeline(name="author-review")
pipeline.add_stage(writer).add_stage(reviewer)
context = await pipeline.run("Explain vector search")
```

See [orchestration.md](orchestration.md) for graphs, supervisors, and debate.

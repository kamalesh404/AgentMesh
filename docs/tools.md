# Tools Reference

Tools are how agents touch the outside world. Each tool declares typed
parameters, validates and coerces inputs, times execution, and returns a
`ToolResult` instead of raising for expected failures.

## Anatomy of a tool

```python
from src.tools.base import Parameter, Tool


class DiceTool(Tool):
    name = "dice"
    description = "Roll an N-sided die."
    parameters = [Parameter(name="sides", type="integer", required=False, default=6)]

    async def _run(self, sides: int = 6):
        import random
        return random.randint(1, max(2, sides))


result = await DiceTool().run(sides="20")   # "20" coerced to int
print(result.success, result.output)        # True 17
```

Every tool also exposes `schema()` (JSON-schema style) and
`openai_definition()` for function-calling integrations.

## Built-in tools

| Tool | Module | Notes |
| --- | --- | --- |
| `web_search` | tools/web_search.py | Tavily or SerpAPI; needs API key env vars |
| `code_exec` | tools/code_exec.py | subprocess (`-I` isolated) or Docker sandbox |
| `file_read` / `file_write` / `list_directory` / `grep` | tools/file_ops.py | rooted at a sandbox base directory |
| `shell` | tools/shell.py | timeout + destructive-command blocklist |
| `api_call` | tools/api_call.py | HTTPS-only policy, bearer auth, retries |
| `database` | tools/database.py | SQLite; optional read-only guard |

## Sandboxed code execution

```python
from src.tools.code_exec import CodeExecutionTool

executor = CodeExecutionTool()
result = await executor.run(code="print(sum(range(10)))", backend="subprocess")
assert result.output["stdout"].strip() == "45"
```

The Docker backend adds `--network none`, memory/CPU/PID caps, and an
ephemeral workspace mount. It requires the docker CLI on PATH.

## File operations with a sandbox root

All filesystem tools resolve paths against a root and reject escapes:

```python
from src.tools.file_ops import FileReadTool, FileWriteTool, build_default_file_tools

tools = build_default_file_tools(root="./workspace")
await tools[1].run(path="reports/q3.md", content="# Q3")     # write
await tools[0].run(path="reports/q3.md")                     # read
```

`FileWriteTool(path="../etc/passwd")` fails with a ToolError — path escapes
are blocked by construction.

## Shell execution

```python
from src.tools.shell import ShellTool

shell = ShellTool(default_timeout=20)
outcome = await shell.run(command="pytest -q tests/test_memory.py")
print(outcome.output["exit_code"], outcome.output["duration_ms"])
```

A small blocklist rejects obviously destructive commands such as
`rm -rf /` or disk-formatting invocations. Treat this tool as trusted-input
only.

## HTTP APIs

```python
from src.tools.api_call import APICallTool

github = APICallTool(token_env_var="GITHUB_TOKEN")
response = await github.run(
    url="https://api.github.com/repos/agentmesh/agentmesh",
    headers={"Accept": "application/vnd.github+json"},
)
print(response.output["status"], response.output["body"]["full_name"])
```

Transport errors retry automatically with exponential backoff; only `https://`
URLs are permitted by default.

## Databases

```python
from src.tools.database import DatabaseTool

with DatabaseTool(path="app.db", read_only=True) as db:
    rows = await db.run(query="SELECT * FROM events LIMIT ?", params=[10])
print(rows.output["columns"], rows.output["rows"])
```

In `read_only` mode only `SELECT`, `WITH`, `EXPLAIN`, and `PRAGMA` statements
are allowed. Write statements commit automatically otherwise.

## Registries

Agents hold tools in a `ToolRegistry`, which rejects duplicate names, renders
prompt-friendly descriptions via `describe()`, and exports OpenAI-compatible
function definitions in one call.

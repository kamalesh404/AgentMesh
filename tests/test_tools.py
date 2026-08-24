"""Tests for tool execution: files, shell, code sandbox, database, registry."""
from __future__ import annotations

import pytest

from src.tools.api_call import APICallTool
from src.tools.base import Parameter, Tool, ToolError, ToolRegistry, ToolResult
from src.tools.code_exec import CodeExecutionTool
from src.tools.database import DatabaseTool
from src.tools.file_ops import FileReadTool, FileWriteTool, GrepTool
from src.tools.shell import ShellTool

from .conftest import run


def test_file_write_then_read(tmp_path) -> None:
    write_tool = FileWriteTool(root=tmp_path)
    read_tool = FileReadTool(root=tmp_path)
    result = run(write_tool.run(path="notes/hello.txt", content="line one\nline two\n"))
    assert result.success and result.output["bytes_written"] > 0
    read_back = run(read_tool.run(path="notes/hello.txt"))
    assert "line two" in read_back.output["content"]
    assert read_back.output["total_lines"] == 2


def test_file_tools_block_path_escape(tmp_path) -> None:
    write_tool = FileWriteTool(root=tmp_path)
    result = run(write_tool.run(path="../outside.txt", content="nope"))
    assert not result.success
    assert "escapes the sandbox" in result.error


def test_grep_finds_matches(tmp_path) -> None:
    (tmp_path / "app.py").write_text("def handler():\n    return 42\n", encoding="utf-8")
    grep = GrepTool(root=tmp_path)
    result = run(grep.run(pattern=r"return \d+"))
    assert result.success, result.error
    matches = result.output
    assert len(matches) == 1
    assert matches[0]["file"] == "app.py"
    assert matches[0]["line"] == 2


@pytest.mark.skipif(__import__("os").name == "nt" and False, reason="shell works on both platforms")
def test_shell_echo(tmp_path) -> None:
    shell = ShellTool()
    result = run(shell.run(command="echo agentmesh-test", timeout=15))
    assert result.success, result.error
    assert "agentmesh-test" in result.output["stdout"]


def test_shell_blocks_destructive_command() -> None:
    shell = ShellTool()
    result = run(shell.run(command="rm -rf / --no-preserve-root", timeout=5))
    assert not result.success
    assert "blocked by safety policy" in result.error


def test_code_exec_computes_expression() -> None:
    tool = CodeExecutionTool()
    result = run(tool.run(code="print(21 * 2)"))
    assert result.success, result.error
    assert result.output["stdout"].strip() == "42"
    assert result.output["returncode"] == 0


def test_database_roundtrip(tmp_path) -> None:
    db = DatabaseTool(path=tmp_path / "test.db")
    created = run(db.run(query="CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"))
    assert created.success and created.output["committed"]
    inserted = run(db.run(query="INSERT INTO items (name) VALUES (?)", params=["widget"]))
    assert inserted.output["rowcount"] == 1
    selected = run(db.run(query="SELECT id, name FROM items ORDER BY id"))
    assert selected.output["rows"] == [{"id": 1, "name": "widget"}]
    db.close()


def test_database_read_only_guard(tmp_path) -> None:
    db = DatabaseTool(path=tmp_path / "ro.db", read_only=True)
    result = run(db.run(query="DROP TABLE anything"))
    assert not result.success
    assert "read_only mode" in result.error


def test_registry_and_schema(tmp_path) -> None:
    class DemoTool(Tool):
        name = "demo"
        description = "demo tool"
        parameters = [Parameter(name="count", type="integer", required=False, default=1)]

        async def _run(self, count: int = 1):
            return count * 2

    registry = ToolRegistry()
    demo = DemoTool()
    registry.register(demo)
    with pytest.raises(ToolError):
        registry.register(DemoTool())
    validated = demo.validate({"count": "5"})
    assert validated == {"count": 5}
    definition = registry.openai_definitions()[0]
    assert definition["function"]["name"] == "demo"


def test_api_call_rejects_http_urls() -> None:
    tool = APICallTool()
    result = run(tool.run(url="http://insecure.example.com"))
    assert not result.success
    assert "https" in result.error


def test_tool_result_helpers() -> None:
    ok = ToolResult.ok({"value": 1})
    failed = ToolResult.fail("bad input")
    assert ok.success and ok.output == {"value": 1}
    assert not failed.success and failed.error == "bad input"

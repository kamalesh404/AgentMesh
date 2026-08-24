"""AgentMesh tools: schema-validated capabilities agents can invoke."""

from src.tools.base import (
    Parameter,
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
)

__all__ = [
    "Parameter",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
]


def __getattr__(name: str):
    """Lazily expose concrete tools so importing the package stays light."""
    mapping = {
        "WebSearchTool": ("src.tools.web_search", "WebSearchTool"),
        "CodeExecutionTool": ("src.tools.code_exec", "CodeExecutionTool"),
        "SubprocessSandbox": ("src.tools.code_exec", "SubprocessSandbox"),
        "DockerSandbox": ("src.tools.code_exec", "DockerSandbox"),
        "FileReadTool": ("src.tools.file_ops", "FileReadTool"),
        "FileWriteTool": ("src.tools.file_ops", "FileWriteTool"),
        "ListDirectoryTool": ("src.tools.file_ops", "ListDirectoryTool"),
        "GrepTool": ("src.tools.file_ops", "GrepTool"),
        "ShellTool": ("src.tools.shell", "ShellTool"),
        "APICallTool": ("src.tools.api_call", "APICallTool"),
        "DatabaseTool": ("src.tools.database", "DatabaseTool"),
    }
    if name in mapping:
        module_name, attribute = mapping[name]
        import importlib

        return getattr(importlib.import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

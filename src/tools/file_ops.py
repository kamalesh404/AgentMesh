"""File operation tools rooted at a sandboxed base directory."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.base import Parameter, Tool, ToolError

_TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".csv", ".log"}
_MAX_LISTED_ENTRIES = 500
_MAX_GREP_FILE_BYTES = 1_000_000


def _resolve(root: Path, user_path: str) -> Path:
    candidate = (root / user_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError(f"path '{user_path}' escapes the sandbox root '{root}'")
    return candidate


class FileReadTool(Tool):
    """Read a text file, optionally returning a window of lines."""

    name = "file_read"
    description = "Read a text file inside the sandbox and return its contents."
    parameters = [
        Parameter(name="path", type="string", description="File path relative to the sandbox root"),
        Parameter(name="offset", type="integer", description="0-based first line to read", required=False, default=0),
        Parameter(name="limit", type="integer", description="Maximum number of lines", required=False, default=1000),
    ]

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    async def _run(self, path: str, offset: int = 0, limit: int = 1000) -> Dict[str, Any]:
        target = _resolve(self.root, path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        window = lines[offset : offset + limit]
        return {"path": str(target), "total_lines": len(lines), "content": "\n".join(window)}


class FileWriteTool(Tool):
    """Write or append text to a file inside the sandbox."""

    name = "file_write"
    description = "Write content to a file inside the sandbox (creates parents)."
    parameters = [
        Parameter(name="path", type="string", description="File path relative to the sandbox root"),
        Parameter(name="content", type="string", description="Text content to write"),
        Parameter(name="append", type="boolean", description="Append instead of overwrite", required=False, default=False),
    ]

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    async def _run(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        target = _resolve(self.root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return {"path": str(target), "bytes_written": len(content.encode("utf-8")), "appended": append}


class ListDirectoryTool(Tool):
    """Enumerate files and directories under the sandbox root."""

    name = "list_directory"
    description = "List entries below a sandbox directory."
    parameters = [
        Parameter(name="path", type="string", description="Directory relative to the sandbox root", required=False, default="."),
        Parameter(name="recursive", type="boolean", description="Walk nested directories", required=False, default=False),
    ]

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    async def _run(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        base = _resolve(self.root, path)
        if not base.is_dir():
            raise ToolError(f"not a directory: {path}")
        pattern = "**/*" if recursive else "*"
        entries = sorted(str(item.relative_to(self.root)) for item in base.glob(pattern))[:_MAX_LISTED_ENTRIES]
        return {"root": str(self.root), "count": len(entries), "entries": entries}


class GrepTool(Tool):
    """Regex search across text files within the sandbox."""

    name = "grep"
    description = "Search text files for a regex pattern and return matching lines."
    parameters = [
        Parameter(name="pattern", type="string", description="Python regular expression"),
        Parameter(name="path", type="string", description="Directory to scan", required=False, default="."),
        Parameter(name="max_results", type="integer", description="Cap on matches returned", required=False, default=50),
    ]

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    async def _run(self, pattern: str, path: str = ".", max_results: int = 50) -> List[Dict[str, Any]]:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc
        base = _resolve(self.root, path)
        if not base.is_dir():
            raise ToolError(f"not a directory: {path}")
        matches: List[Dict[str, Any]] = []
        for file_path in sorted(base.rglob("*")):
            if len(matches) >= max_results:
                break
            if not file_path.is_file() or file_path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if file_path.stat().st_size > _MAX_GREP_FILE_BYTES:
                continue
            try:
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if regex.search(line):
                            matches.append(
                                {
                                    "file": str(file_path.relative_to(self.root)),
                                    "line": line_number,
                                    "text": line.strip()[:200],
                                }
                            )
                            if len(matches) >= max_results:
                                break
            except OSError:
                continue
        return matches


def build_default_file_tools(root: Optional[str | Path] = None) -> List[Tool]:
    """Convenience factory returning the standard file tool set."""
    base_root = root or Path.cwd()
    return [
        FileReadTool(base_root),
        FileWriteTool(base_root),
        ListDirectoryTool(base_root),
        GrepTool(base_root),
    ]

"""Shell command execution tool with timeouts and basic safety checks."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.tools.base import Parameter, Tool, ToolError

_BLOCKED_FRAGMENTS: Set[str] = {
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
    "shutdown",
    "format c:",
    "diskutil erase",
}

_DEFAULT_TIMEOUT = 30.0


class ShellTool(Tool):
    """Execute a shell command and capture its streams.

    Intended for trusted environments only; commands run with the privileges
    of the host process. A small blocklist rejects obviously destructive
    patterns, and every invocation is bounded by ``timeout`` seconds.
    """

    name = "shell"
    description = "Run a shell command with a timeout and return exit code plus stdout/stderr."
    parameters = [
        Parameter(name="command", type="string", description="The shell command line to execute"),
        Parameter(name="timeout", type="number", description="Seconds before the command is killed", required=False, default=_DEFAULT_TIMEOUT),
        Parameter(name="cwd", type="string", description="Working directory for the command", required=False),
    ]

    def __init__(self, root: Optional[str | Path] = None, default_timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.root = Path(root).resolve() if root else None
        self.default_timeout = default_timeout
        self.executions = 0

    def _check_safety(self, command: str) -> None:
        lowered = " ".join(command.lower().split())
        for fragment in _BLOCKED_FRAGMENTS:
            if fragment in lowered:
                raise ToolError(f"command blocked by safety policy: contains {fragment!r}")

    async def _run(self, command: str, timeout: float = _DEFAULT_TIMEOUT, cwd: Optional[str] = None) -> Dict[str, Any]:
        self._check_safety(command)
        working_dir = self._resolve_cwd(cwd)
        started_at = __import__("time").perf_counter()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(working_dir) if working_dir else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=max(1.0, timeout))
        except asyncio.TimeoutError:
            timed_out = True
            process.kill()
            await process.wait()
            stdout, stderr = b"", b""
        self.executions += 1
        import time as _time

        result = {
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[:20000],
            "stderr": stderr.decode("utf-8", errors="replace")[:20000],
            "timed_out": timed_out,
            "duration_ms": round((_time.perf_counter() - started_at) * 1000, 2),
        }
        if timed_out:
            raise ToolError(f"command timed out after {timeout}s: {command[:120]!r}")
        return result

    def _resolve_cwd(self, cwd: Optional[str]) -> Optional[Path]:
        if not cwd:
            return self.root
        base = self.root or Path.cwd()
        candidate = (base / cwd).resolve()
        if self.root is not None and candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"cwd '{cwd}' escapes the sandbox root '{self.root}'")
        return candidate

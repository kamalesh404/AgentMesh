"""Sandboxed Python code execution via subprocess or Docker."""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional

from src.tools.base import Parameter, Tool, ToolError


class SandboxTimeout(ToolError):
    """Raised when sandboxed execution exceeds the allowed duration."""


class SubprocessSandbox:
    """Runs Python snippets in an isolated interpreter process."""

    def __init__(self, python_executable: Optional[str] = None) -> None:
        self.executable = python_executable or sys.executable

    async def run(self, code: str, timeout: float = 10.0) -> Dict[str, Any]:
        handle = tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8", prefix="agentmesh_exec_"
        )
        path = handle.name
        try:
            with handle:
                handle.write(code)
            started = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                self.executable, "-I", path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.wait()
                raise SandboxTimeout(f"execution exceeded {timeout}s and was killed") from exc
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "backend": "subprocess",
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class DockerSandbox:
    """Runs Python snippets inside an ephemeral locked-down container."""

    def __init__(self, image: str = "python:3.12-slim", memory_limit: str = "256m") -> None:
        self.image = image
        self.memory_limit = memory_limit

    async def run(self, code: str, timeout: float = 10.0) -> Dict[str, Any]:
        if shutil.which("docker") is None:
            raise ToolError("docker executable not found on PATH")
        workdir = tempfile.mkdtemp(prefix="agentmesh_docker_")
        script_path = os.path.join(workdir, "main.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        command = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", self.memory_limit,
            "--cpus", "0.5",
            "--pids-limit", "64",
            "-v", f"{workdir}:/sandbox",
            "-w", "/sandbox",
            self.image, "python", "/sandbox/main.py",
        ]
        started = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout + 20.0)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise SandboxTimeout(f"docker run exceeded {timeout + 20.0}s") from exc
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "backend": "docker",
        }


class CodeExecutionTool(Tool):
    """Execute generated Python code in a sandbox and capture its output."""

    name = "code_exec"
    description = "Run Python source code in a sandbox and return stdout/stderr/exit code."
    parameters = [
        Parameter(name="code", type="string", description="Python source to execute"),
        Parameter(name="timeout", type="number", description="Seconds before the run is killed", required=False, default=10.0),
        Parameter(
            name="backend",
            type="string",
            description="Sandbox implementation",
            required=False,
            default="subprocess",
            enum=["subprocess", "docker"],
        ),
    ]

    def __init__(self) -> None:
        self._subprocess_sandbox = SubprocessSandbox()
        self._docker_sandbox: Optional[DockerSandbox] = None

    async def _run(self, code: str, timeout: float = 10.0, backend: str = "subprocess") -> Dict[str, Any]:
        if backend == "subprocess":
            return await self._subprocess_sandbox.run(code, timeout=timeout)
        if backend == "docker":
            if self._docker_sandbox is None:
                self._docker_sandbox = DockerSandbox()
            return await self._docker_sandbox.run(code, timeout=timeout)
        raise ToolError(f"unknown backend '{backend}'")

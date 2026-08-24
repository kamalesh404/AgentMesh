"""Tool abstraction: schemas, validation, execution, and registries."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_COERCIONS = {"string": str, "integer": int, "number": float}


class ToolError(Exception):
    """Raised when a tool cannot complete its work."""


@dataclass
class ToolResult:
    """Outcome of a single tool invocation."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any, duration_ms: float = 0.0, **metadata: Any) -> "ToolResult":
        return cls(success=True, output=output, duration_ms=duration_ms, metadata=metadata)

    @classmethod
    def fail(cls, error: str, duration_ms: float = 0.0) -> "ToolResult":
        return cls(success=False, error=error, duration_ms=duration_ms)


@dataclass
class Parameter:
    """Declaration of a single tool argument with coercion rules."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None

    def coerce(self, value: Any) -> Any:
        if self.enum is not None and value not in self.enum:
            raise ToolError(f"parameter '{self.name}' must be one of {self.enum}")
        try:
            if self.type == "boolean":
                if isinstance(value, bool):
                    return value
                lowered = str(value).strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True
                if lowered in {"false", "0", "no", "off"}:
                    return False
                raise ValueError(f"cannot interpret {value!r} as boolean")
            if self.type == "array":
                if isinstance(value, (list, tuple)):
                    return list(value)
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("expected a JSON array")
                return parsed
            if self.type == "object":
                if isinstance(value, dict):
                    return value
                parsed = json.loads(value)
                if not isinstance(parsed, dict):
                    raise ValueError("expected a JSON object")
                return parsed
            return _COERCIONS[self.type](value)
        except (TypeError, ValueError, KeyError) as exc:
            raise ToolError(f"parameter '{self.name}' ({self.type}): {exc}") from exc


class Tool(ABC):
    """Base class for every AgentMesh tool."""

    name: str = "tool"
    description: str = ""
    parameters: List[Parameter] = []

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce raw arguments into typed values."""
        known = {param.name for param in self.parameters}
        for key in args:
            if key not in known:
                raise ToolError(f"unknown parameter '{key}' for tool '{self.name}'")
        cleaned: Dict[str, Any] = {}
        for param in self.parameters:
            if args.get(param.name) is not None:
                cleaned[param.name] = param.coerce(args[param.name])
            elif param.required:
                raise ToolError(f"missing required parameter '{param.name}' for tool '{self.name}'")
            else:
                cleaned[param.name] = param.default
        return cleaned

    @abstractmethod
    async def _run(self, **kwargs: Any) -> Any:
        """Execute the tool with validated keyword arguments."""

    async def run(self, **kwargs: Any) -> ToolResult:
        """Validate arguments, execute, and capture timing or errors."""
        started = time.perf_counter()
        try:
            validated = self.validate(kwargs)
            output = await self._run(**validated)
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return ToolResult.ok(output, duration_ms=elapsed)
        except ToolError as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return ToolResult.fail(str(exc), duration_ms=elapsed)
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            return ToolResult.fail(f"unexpected error in '{self.name}': {exc}", duration_ms=elapsed)

    def schema(self) -> Dict[str, Any]:
        properties = {}
        for param in self.parameters:
            entry: Dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum is not None:
                entry["enum"] = param.enum
            properties[param.name] = entry
        required = [p.name for p in self.parameters if p.required]
        return {"type": "object", "properties": properties, "required": required}

    def openai_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema(),
            },
        }


class ToolRegistry:
    """Named collection of tools attached to an agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"a tool named '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"tool '{name}' is not registered")
        return self._tools[name]

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def names(self) -> List[str]:
        return sorted(self._tools)

    def openai_definitions(self) -> List[Dict[str, Any]]:
        return [tool.openai_definition() for tool in self._tools.values()]

    def describe(self) -> str:
        lines = [f"- {t.name}: {t.description}" for t in self._tools.values()]
        return "\n".join(lines) or "(no tools registered)"

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

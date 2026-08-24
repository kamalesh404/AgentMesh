"""Layered configuration loading: defaults, YAML files, and environment."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None

ENV_PREFIX = "AGENTMESH__"

DEFAULTS: Dict[str, Any] = {
    "log_level": "INFO",
    "log_format": "console",
    "providers": {
        "default": "mock",
        "openai": {"model": "gpt-4o-mini"},
        "anthropic": {"model": "claude-3-5-sonnet-latest"},
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.1"},
    },
    "memory": {"short_term_size": 200, "vector_dim": 64},
    "orchestration": {"debate_rounds": 3, "supervisor_timeout": 30.0},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    for caster in (int, float):
        try:
            return caster(raw)
        except ValueError:
            continue
    if raw.startswith(("{", "[")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _set_dotted(data: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = data
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


class Config:
    """Attribute-free configuration object with dotted-path access."""

    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        self._data = _deep_merge(copy.deepcopy(DEFAULTS), data or {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Build a Config from defaults, an optional YAML file, and env vars."""
        file_data: Dict[str, Any] = {}
        candidate = Path(path) if path else Path.cwd() / "agentmesh.yaml"
        if candidate.exists():
            if yaml is None:
                raise RuntimeError("PyYAML is required to read config files")
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config root in {candidate} must be a mapping")
            file_data = loaded
        for key, value in os.environ.items():
            if not key.startswith(ENV_PREFIX):
                continue
            dotted = ".".join(seg.lower() for seg in key[len(ENV_PREFIX):].split("__") if seg)
            if dotted:
                _set_dotted(file_data, dotted, _coerce_scalar(value))
        return cls(file_data)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Resolve a dotted path such as ``providers.openai.model``."""
        cursor: Any = self._data
        for part in dotted.split("."):
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                return default
        return cursor

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the underlying mapping."""
        return copy.deepcopy(self._data)

    def save(self, path: str | Path) -> None:
        """Persist the current values to YAML or JSON based on suffix."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix in {".yaml", ".yml"} and yaml is not None:
            with target.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self._data, handle, default_flow_style=False)
        else:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2)

    def __contains__(self, dotted: str) -> bool:
        sentinel = object()
        return self.get(dotted, sentinel) is not sentinel

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

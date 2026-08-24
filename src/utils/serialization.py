"""Robust serialization for messages, entries, and arbitrary payloads."""
from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import pathlib
from enum import Enum
from typing import Any


def to_compatible(obj: Any) -> Any:
    """Recursively convert objects into JSON-safe primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_compatible(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(to_compatible(item) for item in obj)
    if isinstance(obj, (list, tuple)):
        return [to_compatible(item) for item in obj]
    return str(obj)


def dumps(obj: Any, indent: int | None = 2) -> str:
    """Serialize an object to a canonical JSON string."""
    return json.dumps(to_compatible(obj), indent=indent, default=str)


def loads(raw: str) -> Any:
    """Deserialize a JSON string produced by :func:`dumps`."""
    return json.loads(raw)


def dump_msgpack(obj: Any) -> bytes:
    """Serialize an object with msgpack; requires the optional extra."""
    try:
        import msgpack
    except ImportError as exc:
        raise RuntimeError("msgpack is required: pip install 'agentmesh[msgpack]'") from exc
    return msgpack.packb(to_compatible(obj), use_bin_type=True)


def load_msgpack(payload: bytes) -> Any:
    """Deserialize msgpack bytes back into Python structures."""
    try:
        import msgpack
    except ImportError as exc:
        raise RuntimeError("msgpack is required: pip install 'agentmesh[msgpack]'") from exc
    return msgpack.unpackb(payload, raw=False)


def fingerprint(obj: Any) -> str:
    """Return a stable SHA-256 fingerprint of any serializable object."""
    canonical = json.dumps(to_compatible(obj), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

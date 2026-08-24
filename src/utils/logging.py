"""Structured logging with correlation IDs for tracing multi-agent flows."""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict

_CORRELATION_ID: ContextVar[str] = ContextVar("agentmesh_correlation_id", default="")


def new_correlation_id() -> str:
    """Generate and install a fresh correlation ID, returning it."""
    correlation_id = uuid.uuid4().hex[:16]
    _CORRELATION_ID.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Explicitly bind the current context to an existing correlation ID."""
    _CORRELATION_ID.set(correlation_id)


def get_correlation_id() -> str:
    """Return the correlation ID bound to the current execution context."""
    return _CORRELATION_ID.get()


class CorrelationIdFilter(logging.Filter):
    """Injects the ambient correlation ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class JsonFormatter(logging.Formatter):
    """Renders log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", ""),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the root logger with console or JSON formatting."""
    handler = logging.StreamHandler(stream=sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(correlation_id)s] %(name)s: %(message)s")
        )
    handler.addFilter(CorrelationIdFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger carrying the correlation-ID filter."""
    logger = logging.getLogger(f"agentmesh.{name}")
    if not any(isinstance(f, CorrelationIdFilter) for f in logger.filters):
        logger.addFilter(CorrelationIdFilter())
    return logger

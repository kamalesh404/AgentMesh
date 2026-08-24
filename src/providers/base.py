"""Model provider abstraction shared by OpenAI, Anthropic, and Ollama."""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.utils.logging import get_logger


class ProviderError(RuntimeError):
    """Generic failure raised by provider backends."""


class RateLimitError(ProviderError):
    """Raised when the upstream API signals rate limiting (HTTP 429)."""


def deterministic_embedding(text: str, dim: int = 64) -> List[float]:
    """Hash tokens into a normalized fixed-size vector (no network needed).

    Useful for tests, offline demos, and as a fallback embedding function.
    """
    vector = [0.0] * dim
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest, 16) % dim
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Completion:
    text: str
    model: str
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class TokenBucketRateLimiter:
    """Classic token-bucket limiter for smoothing outbound API traffic."""

    def __init__(self, rate_per_second: float = 5.0, capacity: float = 10.0) -> None:
        if rate_per_second <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = float(rate_per_second)
        self.capacity = float(capacity)
        self._tokens = capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        """Wait until ``amount`` tokens are available, then consume them."""
        if amount > self.capacity:
            raise ValueError("requested amount exceeds bucket capacity")
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._updated_at = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                await asyncio.sleep(max(deficit / self.rate, 0.005))


class BaseProvider(ABC):
    """Common interface for chat completion and embedding backends."""

    name: str = "base"
    supports_embeddings: bool = True

    def __init__(self, model: Optional[str] = None, requests_per_second: float = 5.0) -> None:
        self.model = model or "default"
        self.limiter = TokenBucketRateLimiter(rate_per_second=requests_per_second)
        self.logger = get_logger(f"provider.{self.name}")
        self.stats: Dict[str, Any] = {"completions": 0, "embeddings": 0, "errors": 0}

    @abstractmethod
    async def _complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        """Backend-specific completion; called from :meth:`complete`."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into fixed-size vectors."""

    async def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        """Public completion entry point applying rate limiting and stats."""
        await self.limiter.acquire()
        started = time.perf_counter()
        try:
            completion = await self._complete(prompt, system, temperature, max_tokens)
        except ProviderError:
            self.stats["errors"] += 1
            raise
        except Exception as exc:
            self.stats["errors"] += 1
            raise ProviderError(f"{self.name} backend failed: {exc}") from exc
        completion.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.stats["completions"] += 1
        return completion

    async def chat(self, messages: Iterable[Dict[str, str]], **kwargs: Any) -> Completion:
        """Flatten a role/content transcript into a single prompt."""
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prefix = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, role.title())
            lines.append(f"{prefix}: {content}")
        lines.append("Assistant:")
        return await self.complete("\n".join(lines), **kwargs)


class MockProvider(BaseProvider):
    """Deterministic offline provider used by tests, demos, and the CLI."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-medium",
        default_response: str = "Understood. Here is my response.",
        requests_per_second: float = 50.0,
    ) -> None:
        super().__init__(model=model, requests_per_second=requests_per_second)
        self.default_response = default_response
        self.calls: List[str] = []

    async def _complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        self.calls.append(prompt)
        tail = " ".join(prompt.split()[-24:])
        text = f"[{self.name}/{self.model}] {self.default_response} (echo: {tail})"
        usage = Usage(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4)
        return Completion(text=text, model=self.model, usage=usage)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        self.stats["embeddings"] += len(texts)
        return [deterministic_embedding(text) for text in texts]

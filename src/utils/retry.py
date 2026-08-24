"""Retry helpers with exponential backoff and jitter."""
from __future__ import annotations

import asyncio
import functools
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Tuple, TypeVar

T = TypeVar("T")

ExceptionTypes = Tuple[type, ...]


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts have failed."""

    def __init__(self, attempts: int, last_error: BaseException | None) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"gave up after {attempts} attempt(s): {last_error!r}")


@dataclass(frozen=True)
class RetryPolicy:
    """Declarative policy describing how failures should be retried."""

    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 5.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on: ExceptionTypes = field(default=(Exception,), kw_only=True)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")

    def delay_for(self, attempt: int) -> float:
        """Compute the sleep duration before the given retry attempt."""
        raw = min(self.max_delay, self.base_delay * (self.exponential_base ** (attempt - 1)))
        if self.jitter:
            return max(0.0, raw * (1 + random.uniform(-self.jitter, self.jitter)))
        return raw


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> T:
    """Await ``fn(*args, **kwargs)``, retrying transient failures per policy.

    Args:
        fn: Async callable to invoke.
        policy: Retry behaviour; defaults to three attempts on ``Exception``.
        args/kwargs: Positional and keyword arguments forwarded to ``fn``.

    Raises:
        RetryExhaustedError: When every permitted attempt fails.
    """
    active_policy = policy or RetryPolicy()
    last_error: BaseException | None = None
    for attempt in range(1, active_policy.max_attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except active_policy.retry_on as exc:
            last_error = exc
            if attempt == active_policy.max_attempts:
                break
            await asyncio.sleep(active_policy.delay_for(attempt))
    raise RetryExhaustedError(active_policy.max_attempts, last_error) from last_error


def with_retry(policy: RetryPolicy | None = None) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator applying :func:`retry_async` to an async function."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_async(func, *args, policy=policy, **kwargs)

        return wrapper

    return decorator

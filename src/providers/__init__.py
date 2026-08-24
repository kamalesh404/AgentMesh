"""LLM provider backends: OpenAI, Anthropic, Ollama, and an offline mock."""

from src.providers.base import (
    BaseProvider,
    Completion,
    MockProvider,
    ProviderError,
    RateLimitError,
    TokenBucketRateLimiter,
    Usage,
    deterministic_embedding,
)

__all__ = [
    "BaseProvider",
    "Completion",
    "MockProvider",
    "ProviderError",
    "RateLimitError",
    "TokenBucketRateLimiter",
    "Usage",
    "deterministic_embedding",
]


def __getattr__(name: str):
    """Lazily expose network-backed providers so imports stay dependency-light."""
    if name == "OpenAIProvider":
        from src.providers.openai import OpenAIProvider

        return OpenAIProvider
    if name == "AnthropicProvider":
        from src.providers.anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OllamaProvider":
        from src.providers.ollama import OllamaProvider

        return OllamaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""OpenAI chat-completions and embeddings provider."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from src.providers.base import (
    BaseProvider,
    Completion,
    ProviderError,
    RateLimitError,
    Usage,
)
from src.utils.retry import RetryPolicy, retry_async

API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    """Concrete provider talking to the OpenAI REST API."""

    name = "openai"
    embedding_model = "text-embedding-3-small"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        organization: Optional[str] = None,
        base_url: str = API_BASE,
        timeout: float = 60.0,
        max_retries: int = 3,
        requests_per_second: float = 4.0,
    ) -> None:
        super().__init__(model=model, requests_per_second=requests_per_second)
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ProviderError("OpenAI API key required (pass api_key or set OPENAI_API_KEY)")
        if httpx is None:
            raise ProviderError("httpx is required for OpenAIProvider: pip install httpx")
        self.api_key = resolved_key
        self.organization = organization or os.environ.get("OPENAI_ORG_ID")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        transport_errors = (httpx.TransportError,) if httpx else ()
        policy = RetryPolicy(
            max_attempts=self.max_retries,
            base_delay=0.5,
            retry_on=(RateLimitError, *transport_errors),
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await retry_async(client.post, url, json=payload, headers=self._headers(), policy=policy)
        if response.status_code == 429:
            raise RateLimitError(f"OpenAI rate limited: {response.text[:200]}")
        if response.status_code >= 400:
            raise ProviderError(f"OpenAI HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    async def _complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Completion:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await self._post("/chat/completions", payload)
        choice = data["choices"][0]
        raw_usage = data.get("usage", {})
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
        )
        return Completion(
            text=choice["message"]["content"],
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=usage,
            raw=data,
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"model": self.embedding_model, "input": texts}
        data = await self._post("/embeddings", payload)
        ordered = sorted(data["data"], key=lambda item: item["index"])
        self.stats["embeddings"] += len(texts)
        return [item["embedding"] for item in ordered]

"""Local Ollama provider for self-hosted open models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

from src.providers.base import (
    BaseProvider,
    Completion,
    ProviderError,
    Usage,
)

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """Provider wrapping a local Ollama daemon (chat and embeddings)."""

    name = "ollama"
    embedding_model = "nomic-embed-text"

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        requests_per_second: float = 10.0,
    ) -> None:
        super().__init__(model=model, requests_per_second=requests_per_second)
        if httpx is None:
            raise ProviderError("httpx is required for OllamaProvider: pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise ProviderError(f"Ollama unreachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(f"Ollama HTTP {response.status_code}: {response.text[:300]}")
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
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = await self._post("/api/chat", payload)
        message = data.get("message", {})
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )
        return Completion(
            text=message.get("content", ""),
            model=data.get("model", self.model),
            finish_reason=data.get("done_reason", "stop"),
            usage=usage,
            raw=data,
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            payload = {"model": self.embedding_model, "prompt": text}
            data = await self._post("/api/embeddings", payload)
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise ProviderError(f"Ollama returned no embedding for: {text[:60]!r}")
            vectors.append(vector)
        self.stats["embeddings"] += len(texts)
        return vectors

"""Web search tool backed by Tavily or SerpAPI."""
from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    import httpx
except ImportError:
    httpx = None

from src.tools.base import Parameter, Tool, ToolError

_TIMEOUT_SECONDS = 20.0


class WebSearchTool(Tool):
    """Search the web and return normalized organic results.

    Requires one of the following environment variables depending on backend:
    ``TAVILY_API_KEY`` or ``SERPAPI_API_KEY``.
    """

    name = "web_search"
    description = "Search the web and return a list of {title, url, snippet} results."
    parameters = [
        Parameter(name="query", type="string", description="The search query"),
        Parameter(name="num_results", type="integer", description="Maximum results to return", required=False, default=5),
        Parameter(
            name="backend",
            type="string",
            description="Search provider",
            required=False,
            default="tavily",
            enum=["tavily", "serpapi"],
        ),
    ]

    async def _run(self, query: str, num_results: int = 5, backend: str = "tavily") -> List[Dict[str, Any]]:
        if httpx is None:
            raise ToolError("httpx is required for web search: pip install httpx")
        num_results = max(1, min(int(num_results), 20))
        if backend == "tavily":
            results = await self._search_tavily(query, num_results)
        elif backend == "serpapi":
            results = await self._search_serpapi(query, num_results)
        else:
            raise ToolError(f"unsupported search backend '{backend}'")
        if not results:
            raise ToolError(f"no results returned for query: {query!r}")
        return results

    async def _search_tavily(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            raise ToolError("TAVILY_API_KEY is not set")
        payload = {"api_key": api_key, "query": query, "max_results": num_results}
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
        if response.status_code >= 400:
            raise ToolError(f"Tavily HTTP {response.status_code}: {response.text[:200]}")
        raw = response.json().get("results", []) or []
        return [
            {
                "title": item.get("title", "(untitled)"),
                "url": item.get("url", ""),
                "snippet": (item.get("content") or "")[:500],
            }
            for item in raw[:num_results]
        ]

    async def _search_serpapi(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        api_key = os.environ.get("SERPAPI_API_KEY", "")
        if not api_key:
            raise ToolError("SERPAPI_API_KEY is not set")
        params = {"engine": "google", "q": query, "api_key": api_key, "num": str(num_results)}
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get("https://serpapi.com/search", params=params)
        if response.status_code >= 400:
            raise ToolError(f"SerpAPI HTTP {response.status_code}: {response.text[:200]}")
        raw = response.json().get("organic_results", []) or []
        return [
            {
                "title": item.get("title", "(untitled)"),
                "url": item.get("link", ""),
                "snippet": (item.get("snippet") or "")[:500],
            }
            for item in raw[:num_results]
        ]

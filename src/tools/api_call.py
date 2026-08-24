"""HTTP API calling tool with retries, auth, and response normalization."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:
    httpx = None

from src.tools.base import Parameter, Tool, ToolError
from src.utils.retry import RetryPolicy, retry_async

_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD")
_MAX_BODY_CHARS = 4000


class APICallTool(Tool):
    """Call an HTTP endpoint with bearer auth and exponential-backoff retries."""

    name = "api_call"
    description = "Perform an HTTP request against an API and return status plus parsed body."
    parameters = [
        Parameter(name="url", type="string", description="Absolute http(s) URL to call"),
        Parameter(name="method", type="string", description="HTTP method", required=False, default="GET"),
        Parameter(name="headers", type="object", description="Extra request headers", required=False),
        Parameter(name="query", type="object", description="Query parameters", required=False),
        Parameter(name="body", type="object", description="JSON body for write methods", required=False),
        Parameter(name="timeout", type="number", description="Per-attempt timeout in seconds", required=False, default=30.0),
        Parameter(name="retries", type="integer", description="Additional attempts after the first", required=False, default=2),
    ]

    def __init__(
        self,
        default_headers: Optional[Dict[str, str]] = None,
        token_env_var: Optional[str] = None,
        base_headers_only_https: bool = True,
    ) -> None:
        self.default_headers = dict(default_headers or {})
        self.token_env_var = token_env_var
        self.https_only = base_headers_only_https

    def _build_headers(self, headers: Optional[Dict[str, Any]]) -> Dict[str, str]:
        merged: Dict[str, str] = {k: str(v) for k, v in self.default_headers.items()}
        for key, value in (headers or {}).items():
            merged[str(key)] = str(value)
        if self.token_env_var:
            token = os.environ.get(self.token_env_var, "")
            if token:
                merged.setdefault("Authorization", f"Bearer {token}")
        return merged

    async def _attempt(
        self,
        client: "httpx.AsyncClient",
        url: str,
        method: str,
        headers: Dict[str, str],
        query: Optional[Dict[str, Any]],
        body: Optional[Dict[str, Any]],
        timeout: float,
    ) -> Dict[str, Any]:
        response = await client.request(
            method=method,
            url=url,
            params=query,
            json=body if body is not None else None,
            headers=headers,
            timeout=timeout,
        )
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                parsed_body: Any = response.json()
            except ValueError:
                parsed_body = response.text[:_MAX_BODY_CHARS]
        else:
            parsed_body = response.text[:_MAX_BODY_CHARS]
        return {
            "status": response.status_code,
            "ok": response.is_success,
            "content_type": content_type.split(";")[0],
            "body": parsed_body,
        }

    async def _run(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        retries: int = 2,
    ) -> Dict[str, Any]:
        if httpx is None:
            raise ToolError("httpx is required for api_call: pip install httpx")
        verb = method.upper()
        if verb not in _METHODS:
            raise ToolError(f"unsupported HTTP method '{method}'")
        if self.https_only and not url.lower().startswith("https://"):
            raise ToolError("only https:// URLs are permitted by policy")
        final_headers = self._build_headers(headers)
        transport_errors = (httpx.TransportError,) if httpx else ()
        policy = RetryPolicy(max_attempts=max(1, int(retries) + 1), base_delay=0.4, retry_on=tuple(transport_errors))
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await retry_async(
                self._attempt, client, url, verb, final_headers, query, body, float(timeout), policy=policy
            )

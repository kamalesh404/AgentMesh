"""Shared pytest fixtures and path bootstrap for the AgentMesh tests."""
from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.agents.conversational import ChatAgent  # noqa: E402
from src.providers.base import MockProvider  # noqa: E402


def run(coro):
    """Run a coroutine to completion from synchronous test functions."""
    return asyncio.run(coro)


class ScriptedProvider(MockProvider):
    """Mock provider that returns queued responses before falling back."""

    def __init__(self, responses, **kwargs) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses)

    async def _complete(self, prompt, system=None, temperature=0.7, max_tokens=1024):
        from src.providers.base import Completion, Usage

        if self._responses:
            text = self._responses.pop(0)
        else:
            text = f"{self.default_response} (echo: {' '.join(prompt.split()[-12:])})"
        self.calls.append(prompt)
        return Completion(
            text=text,
            model=self.model,
            usage=Usage(prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4),
        )


@pytest.fixture()
def provider() -> MockProvider:
    return MockProvider()


@pytest.fixture()
def make_chat():
    def _factory(name: str = "bot", prov=None) -> ChatAgent:
        return ChatAgent(name, provider=prov or MockProvider())

    return _factory

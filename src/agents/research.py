"""Research agent combining search tools with LLM summarization."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from src.agents.base import AgentError, BaseAgent
from src.protocols.message import Message
from src.tools.base import Tool


class ResearchAgent(BaseAgent):
    """Searches the web (optionally), summarizes sources, and synthesizes."""

    role = "researcher"
    _MAX_SYNTHESIS_SOURCES = 6

    def __init__(
        self,
        name: str,
        provider: Any,
        memory: Any = None,
        search_tool: Optional[Tool] = None,
        max_sources: int = 5,
    ) -> None:
        super().__init__(name=name, provider=provider, memory=memory)
        self.search_tool = search_tool
        if search_tool is not None:
            self.tools.register(search_tool)
        self.max_sources = max(1, min(max_sources, 10))

    async def research(self, topic: str) -> Dict[str, Any]:
        sources = await self._collect_sources(topic)
        summarized: List[Dict[str, Any]] = []
        for source in sources[: self.max_sources]:
            summary = await self._summarize_source(topic, source)
            summarized.append(
                {"title": source["title"], "url": source["url"], "summary": summary}
            )
        synthesis = await self._synthesize(topic, summarized)
        if self.memory is not None and hasattr(self.memory, "add"):
            self.memory.add(synthesis, metadata={"kind": "research", "topic": topic})
        return {
            "topic": topic,
            "summary": synthesis,
            "sources": summarized,
            "generated_at": time.time(),
        }

    @staticmethod
    def _normalize(source: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": str(source.get("title", "(untitled)"))[:200],
            "url": str(source.get("url") or source.get("link", "")),
            "text": str(source.get("content") or source.get("snippet") or "")[:1200],
        }

    async def _collect_sources(self, topic: str) -> List[Dict[str, Any]]:
        if self.search_tool is None:
            completion = await self.provider.complete(
                f"Outline the key research questions and subtopics for: {topic}"
            )
            return [{"title": "Model outline", "url": "", "content": completion.text}]
        result = await self.search_tool.run(query=topic, num_results=self.max_sources)
        if not result.success:
            raise AgentError(f"search failed: {result.error}")
        return [self._normalize(item) for item in (result.output or [])]

    async def _summarize_source(self, topic: str, source: Dict[str, Any]) -> str:
        prompt = (
            f"Research topic: {topic}\n\n"
            f"Source '{source['title']}' says:\n{source['text']}\n\n"
            "Summarize in 2-3 sentences what this contributes to the topic."
        )
        completion = await self.provider.complete(prompt)
        return completion.text.strip()

    async def _synthesize(self, topic: str, summarized: List[Dict[str, Any]]) -> str:
        if not summarized:
            return f"No material found for topic: {topic}"
        blocks = []
        for index, item in enumerate(summarized[: self._MAX_SYNTHESIS_SOURCES], start=1):
            attribution = item["title"] + (f" ({item['url']})" if item["url"] else "")
            blocks.append(f"[{index}] {attribution}\n{item['summary'][:700]}")
        joined = "\n\n".join(blocks)
        completion = await self.provider.complete(
            f"Write an executive brief on '{topic}' synthesizing these notes:\n\n{joined}"
        )
        return completion.text.strip()

    async def process_message(self, message: Message) -> Message:
        intent = str(message.metadata.get("intent", "research"))
        if intent != "research":
            raise AgentError(f"ResearchAgent cannot handle intent '{intent}'")
        report = await self.research(str(message.content))
        return message.reply(json.dumps(report, indent=2))

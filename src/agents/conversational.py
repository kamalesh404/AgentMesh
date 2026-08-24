"""Conversational agent with memory-backed context windows."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base import AgentError, BaseAgent
from src.memory.short_term import ConversationBuffer
from src.protocols.message import Message, MessageType, content_text

DEFAULT_SYSTEM_PROMPT = "You are {name}, a helpful and concise conversational assistant."
RESET_KEYWORD = "reset"


class ChatAgent(BaseAgent):
    """Chat agent that trims history to fit a character budget per turn."""

    role = "chat"

    def __init__(
        self,
        name: str,
        provider: Any = None,
        memory: Any = None,
        system_prompt: Optional[str] = None,
        window_chars: int = 4000,
        max_turns: int = 200,
    ) -> None:
        super().__init__(name=name, provider=provider, memory=memory)
        if provider is None:
            raise AgentError("ChatAgent requires an LLM provider")
        self.history = ConversationBuffer(max_messages=max_turns, max_context_chars=window_chars)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT.format(name=name)
        self.window_chars = window_chars

    async def setup(self) -> None:
        self.history.add(self.system_prompt, role="system", importance=1.0)

    async def process_message(self, message: Message) -> Optional[Message]:
        if message.type is MessageType.CONTROL and str(message.content).strip() == RESET_KEYWORD:
            self.reset()
            return message.reply({"status": RESET_KEYWORD}, type=MessageType.CONTROL)
        user_text = content_text(message.content)
        self.history.add(user_text, role="user", metadata={"sender": message.sender})
        completion = await self.provider.complete(self._build_prompt())
        reply_text = completion.text.strip()
        self.history.add(reply_text, role="assistant", metadata={"model": completion.model})
        return message.reply(reply_text)

    def _build_prompt(self) -> str:
        lines = [f"System: {self.system_prompt}", "", "Transcript:"]
        context = self.history.build_context()
        if context:
            lines.append(context)
        lines.extend(["", "Respond to the latest user message."])
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear the transcript back to the bare system prompt."""
        self.history.clear()
        self.history.add(self.system_prompt, role="system", importance=1.0)

    @property
    def transcript(self) -> List[Dict[str, str]]:
        """Return the full ordered transcript of roles and content."""
        return self.history.messages()

    @property
    def turns(self) -> int:
        return sum(1 for m in self.transcript if m["role"] == "user")

    def info(self) -> Dict[str, Any]:
        payload = super().info()
        payload.update({"turns": self.turns, "window_chars": self.window_chars})
        return payload

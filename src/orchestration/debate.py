"""Multi-agent debate orchestration with consensus voting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.agents.base import BaseAgent
from src.protocols.message import Message


@dataclass
class DebateTurn:
    speaker: str
    round: int
    content: str

    @property
    def words(self) -> set:
        return set(self.content.lower().split())


@dataclass
class DebateResult:
    topic: str
    turns: List[DebateTurn] = field(default_factory=list)
    winner: Optional[str] = None
    consensus: bool = False
    rounds_used: int = 0

    def transcript_text(self) -> str:
        return "\n".join(f"[r{turn.round}] {turn.speaker}: {turn.content}" for turn in self.turns)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DebateOrchestrator:
    """Runs structured debate rounds until agents converge or rounds expire."""

    def __init__(
        self,
        participants: List[BaseAgent],
        judge: Optional[BaseAgent] = None,
        max_rounds: int = 3,
        agreement_threshold: float = 0.55,
    ) -> None:
        if len(participants) < 2:
            raise ValueError("debate requires at least two participants")
        self.participants = list(participants)
        self.judge = judge
        self.max_rounds = max(1, max_rounds)
        self.agreement_threshold = agreement_threshold

    async def run(self, topic: str) -> DebateResult:
        result = DebateResult(topic=topic)
        for current_round in range(1, self.max_rounds + 1):
            positions = await self._collect_positions(topic, current_round, result)
            result.turns.extend(positions.values())
            result.rounds_used = current_round
            winner, consensus = await self._evaluate(positions, topic)
            if consensus:
                result.winner = winner
                result.consensus = True
                return result
            result.winner = winner
        return result

    async def _collect_positions(
        self, topic: str, round_number: int, result: DebateResult
    ) -> Dict[str, DebateTurn]:
        history = [(turn.speaker, turn.content) for turn in result.turns]
        turns: Dict[str, DebateTurn] = {}
        for participant in sorted(self.participants, key=lambda agent: agent.agent_id):
            message = Message(
                sender="moderator",
                recipient=participant.agent_id,
                content=topic,
                metadata={
                    "intent": "debate",
                    "round": round_number,
                    "history": history,
                },
            )
            reply = await participant.deliver(message)
            text = reply.content if isinstance(reply.content, str) else str(reply.content)
            turns[participant.agent_id] = DebateTurn(
                speaker=participant.agent_id, round=round_number, content=text
            )
        return turns

    async def _evaluate(
        self, positions: Dict[str, DebateTurn], topic: str
    ) -> Tuple[Optional[str], bool]:
        """Pick a winner via the judge when available, else Jaccard overlap."""
        speakers = sorted(positions)
        if self.judge is not None and len(speakers) >= 2:
            return await self._judge_vote(positions, topic), True
        scores: Dict[str, float] = {speaker: 0.0 for speaker in speakers}
        for index, speaker in enumerate(speakers):
            others = speakers[:index] + speakers[index + 1 :]
            if not others:
                continue
            total = sum(_jaccard(positions[speaker].words, positions[o].words) for o in others)
            scores[speaker] = total / len(others)
        best_speaker = max(sorted(scores), key=lambda s: scores[s])
        best_score = scores[best_speaker]
        return (best_speaker if best_score > 0 else None), best_score >= self.agreement_threshold

    async def _judge_vote(self, positions: Dict[str, DebateTurn], topic: str) -> Optional[str]:
        assert self.judge is not None
        options = "\n".join(
            f"{index}. {speaker}: {positions[speaker].content[:400]}"
            for index, speaker in enumerate(sorted(positions), start=1)
        )
        prompt = (
            f"Topic: {topic}\n\nCandidate answers:\n{options}\n\n"
            "Reply with the number of the strongest answer only."
        )
        completion = await self.judge.provider.complete(prompt, temperature=0.0)
        digits = "".join(ch for ch in completion.text if ch.isdigit())
        if not digits:
            return None
        choice = int(digits[0])
        ordered = sorted(positions)
        if 1 <= choice <= len(ordered):
            return ordered[choice - 1]
        return None

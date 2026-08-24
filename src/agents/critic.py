"""Critic agent: heuristic + LLM code review with severity scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.agents.base import AgentError, BaseAgent
from src.protocols.message import Message

_MAX_LINE_LENGTH = 120


@dataclass
class ReviewIssue:
    severity: str
    line: int
    message: str
    excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity, "line": self.line, "message": self.message}


@dataclass
class ReviewResult:
    score: int
    issues: List[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    approved: bool = False

    @property
    def blocking_issues(self) -> List[ReviewIssue]:
        return [issue for issue in self.issues if issue.severity in {"critical", "major"}]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "approved": self.approved,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
        }


_HEURISTICS = (
    (re.compile(r"^\s*except\s*:"), "major", 15, "Bare except clause swallows all exceptions"),
    (re.compile(r"\beval\s*\("), "critical", 25, "Use of eval() is dangerous"),
    (
        re.compile(r"(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        "critical",
        30,
        "Possible hardcoded credential",
    ),
    (re.compile(r"\b(TODO|FIXME)\b"), "minor", 5, "Unresolved TODO/FIXME marker"),
    (re.compile(r"^\s*print\s*\("), "minor", 3, "Debug print statement left in code"),
)


class CriticAgent(BaseAgent):
    """Scores code quality using static heuristics plus an LLM review pass."""

    role = "critic"

    def __init__(self, name: str, provider: Any, approval_threshold: int = 75) -> None:
        super().__init__(name=name, provider=provider)
        self.approval_threshold = max(0, min(100, approval_threshold))

    async def review(self, code: str) -> ReviewResult:
        issues, penalty = self._heuristic_scan(code)
        score = max(0, min(100, 100 - penalty))
        note = await self._llm_note(code)
        summary = f"Heuristic score {score}/100 from {len(issues)} finding(s). {note}"
        has_critical = any(issue.severity == "critical" for issue in issues)
        approved = score >= self.approval_threshold and not has_critical
        result = ReviewResult(score=score, issues=issues, summary=summary[:600], approved=approved)
        if self.memory is not None and hasattr(self.memory, "add"):
            self.memory.add(result.summary, metadata={"kind": "review", "score": score})
        return result

    @staticmethod
    def _heuristic_scan(code: str) -> tuple:
        issues: List[ReviewIssue] = []
        penalty = 0
        for line_number, line in enumerate(code.splitlines(), start=1):
            for pattern, severity, weight, message in _HEURISTICS:
                if pattern.search(line):
                    issues.append(
                        ReviewIssue(
                            severity=severity,
                            line=line_number,
                            message=message,
                            excerpt=line.strip()[:120],
                        )
                    )
                    penalty += weight
            if len(line) > _MAX_LINE_LENGTH:
                issues.append(ReviewIssue("minor", line_number, f"Line exceeds {_MAX_LINE_LENGTH} chars"))
                penalty += 2
        return issues, penalty

    async def _llm_note(self, code: str) -> str:
        prompt = (
            "You are a strict code reviewer. In at most three sentences, name the "
            f"single biggest risk in this code:\n\n{code[:1800]}"
        )
        completion = await self.provider.complete(prompt, temperature=0.1)
        return " ".join(completion.text.strip().split())[:280]

    async def process_message(self, message: Message) -> Message:
        intent = str(message.metadata.get("intent", "review"))
        if intent != "review":
            raise AgentError(f"CriticAgent cannot handle intent '{intent}'")
        result = await self.review(str(message.content))
        verdict = "APPROVED" if result.approved else "CHANGES REQUESTED"
        return message.reply({"verdict": verdict, **result.to_dict()})

"""Planning agent with goal decomposition and dependency-ordered tasks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.agents.base import AgentError, BaseAgent
from src.protocols.message import Message

_FALLBACK_TASKS = [
    ("Clarify the goal and success criteria", "Restate the objective and define measurable outcomes."),
    ("Identify required resources", "List information, tools, and people needed."),
    ("Execute the core work", "Carry out the main steps of the goal."),
    ("Verify results and iterate", "Check outputs against the criteria and refine."),
]


@dataclass
class TaskNode:
    """One node in a goal tree with dependencies and completion state."""

    id: str
    title: str
    description: str = ""
    dependencies: Set[str] = field(default_factory=set)
    status: str = "pending"
    result: Optional[Any] = None

    @property
    def done(self) -> bool:
        return self.status == "done"


class GoalTree:
    """Directed acyclic graph of tasks supporting ready-set computation."""

    def __init__(self) -> None:
        self.tasks: Dict[str, TaskNode] = {}

    def add_task(
        self, task_id: str, title: str, description: str = "", dependencies: Optional[List[str]] = None
    ) -> TaskNode:
        if task_id in self.tasks:
            raise ValueError(f"duplicate task id '{task_id}'")
        for dep in dependencies or []:
            if dep not in self.tasks:
                raise ValueError(f"unknown dependency '{dep}' for task '{task_id}'")
        node = TaskNode(id=task_id, title=title, description=description, dependencies=set(dependencies or []))
        self.tasks[task_id] = node
        return node

    def mark_complete(self, task_id: str, result: Any = None) -> None:
        node = self.tasks.get(task_id)
        if node is None:
            raise KeyError(task_id)
        node.status = "done"
        node.result = result

    def ready(self) -> List[str]:
        return sorted(
            node.id
            for node in self.tasks.values()
            if not node.done and all(self.tasks[d].done for d in node.dependencies)
        )

    def topo_order(self) -> List[str]:
        """Kahn topological ordering; raises on dependency cycles."""
        remaining_deps = {tid: set(node.dependencies) for tid, node in self.tasks.items()}
        dependents: Dict[str, List[str]] = {tid: [] for tid in self.tasks}
        for tid, deps in remaining_deps.items():
            for dep in deps:
                dependents[dep].append(tid)
        queue = sorted(tid for tid, deps in remaining_deps.items() if not deps)
        order: List[str] = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for child in dependents[tid]:
                remaining_deps[child].discard(tid)
                if not remaining_deps[child] and child not in order and child not in queue:
                    queue.append(child)
        if len(order) != len(self.tasks):
            raise ValueError("goal tree contains a dependency cycle")
        return order

    @property
    def progress(self) -> str:
        done = sum(1 for node in self.tasks.values() if node.done)
        return f"{done}/{len(self.tasks)}"


class PlannerAgent(BaseAgent):
    """Decomposes goals into ordered subtasks using the LLM with fallbacks."""

    role = "planner"

    async def decompose(self, goal: str, n_subtasks: int = 4) -> GoalTree:
        n_subtasks = max(2, min(n_subtasks, 8))
        prompt = (
            "You are a planning assistant. Decompose the goal into "
            f"{n_subtasks} concrete ordered subtasks.\nGoal: {goal}\n\n"
            'Reply ONLY with a JSON array like [{"title": "...", "description": "..."}]'
        )
        completion = await self.provider.complete(prompt, temperature=0.3)
        items = self._parse_items(completion.text, n_subtasks)
        tree = GoalTree()
        previous: Optional[str] = None
        for index, (title, description) in enumerate(items):
            task_id = f"task-{index}"
            deps = [previous] if previous else []
            tree.add_task(task_id, title, description, dependencies=deps)
            previous = task_id
        return tree

    @staticmethod
    def _parse_items(raw: str, n_subtasks: int) -> List[tuple]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                items = [
                    (str(item.get("title", f"Step {i + 1}")), str(item.get("description", "")))
                    for i, item in enumerate(parsed)
                    if isinstance(item, dict) and item.get("title")
                ]
                if items:
                    return items[:n_subtasks]
            except (json.JSONDecodeError, AttributeError):
                pass
        lines = [
            line.strip().lstrip("-*0123456789. ")
            for line in raw.splitlines()
            if line.strip().lstrip("-*0123456789. ")
        ]
        if lines:
            return [(line[:120], "") for line in lines[:n_subtasks]]
        return _FALLBACK_TASKS[:n_subtasks]

    async def plan(self, goal: str, n_subtasks: int = 4) -> List[str]:
        tree = await self.decompose(goal, n_subtasks)
        return [f"{tree.tasks[tid].title} :: {tree.tasks[tid].description}".rstrip(" :") for tid in tree.topo_order()]

    async def process_message(self, message: Message) -> Message:
        intent = str(message.metadata.get("intent", "plan"))
        if intent != "plan":
            raise AgentError(f"PlannerAgent cannot handle intent '{intent}'")
        count = int(message.metadata.get("subtasks", 4))
        steps = await self.plan(str(message.content), n_subtasks=count)
        body = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))
        return message.reply(f"Plan:\n{body}")

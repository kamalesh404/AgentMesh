"""Raft-inspired consensus primitives for replicated agent state."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """A single replicated command."""

    term: int
    index: int
    command: Any


@dataclass
class VoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


class ConsensusNode:
    """One member of a consensus cluster holding a replicated log."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.role: Role = Role.FOLLOWER
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index: int = -1

    @property
    def last_log_index(self) -> int:
        return len(self.log) - 1

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def become_candidate(self) -> int:
        """Increment the term and self-vote for a new election."""
        self.current_term += 1
        self.voted_for = self.node_id
        self.role = Role.CANDIDATE
        return self.current_term

    def become_leader(self) -> None:
        self.role = Role.LEADER

    def step_down(self) -> None:
        if self.role is not Role.FOLLOWER:
            self.role = Role.FOLLOWER

    def request_vote(self, request: VoteRequest) -> tuple[int, bool]:
        """Handle an incoming vote request; returns (current_term, granted)."""
        if request.term < self.current_term:
            return self.current_term, False
        if request.term > self.current_term:
            self.current_term = request.term
            self.voted_for = None
            self.step_down()
        up_to_date = request.last_log_term > self.last_log_term or (
            request.last_log_term == self.last_log_term
            and request.last_log_index >= self.last_log_index
        )
        grant = self.voted_for in (None, request.candidate_id) and up_to_date
        if grant:
            self.voted_for = request.candidate_id
        return self.current_term, grant

    def append_entries(
        self, entries: List[LogEntry], leader_term: int, leader_commit: int
    ) -> bool:
        """Append or overwrite entries from a leader; returns acceptance."""
        if leader_term < self.current_term:
            return False
        self.current_term = leader_term
        self.step_down()
        for entry in entries:
            if entry.index < len(self.log):
                self.log[entry.index] = entry
            else:
                self.log.append(entry)
        self.commit_index = max(self.commit_index, min(leader_commit, self.last_log_index))
        return True

    def committed(self) -> List[Any]:
        return [entry.command for entry in self.log[: self.commit_index + 1]]


class Cluster:
    """In-memory cluster coordinating elections and log replication."""

    def __init__(self, node_ids: List[str]) -> None:
        if len(node_ids) < 3:
            raise ValueError("a Raft-style cluster needs at least three nodes")
        self.nodes: Dict[str, ConsensusNode] = {nid: ConsensusNode(nid) for nid in node_ids}

    @property
    def majority(self) -> int:
        return len(self.nodes) // 2 + 1

    def leader(self) -> Optional[ConsensusNode]:
        for node in sorted(self.nodes.values(), key=lambda n: n.node_id):
            if node.role is Role.LEADER:
                return node
        return None

    def hold_election(self, candidate_id: str) -> bool:
        """Run one election round; returns True when the candidate wins."""
        candidate = self.nodes[candidate_id]
        term = candidate.become_candidate()
        votes = 1
        for peer in sorted(self.nodes.values(), key=lambda node: node.node_id):
            if peer.node_id == candidate_id:
                continue
            request = VoteRequest(term, candidate_id, candidate.last_log_index, candidate.last_log_term)
            current_term, granted = peer.request_vote(request)
            if granted:
                votes += 1
        if votes >= self.majority:
            candidate.become_leader()
            return True
        candidate.step_down()
        return False

    def ensure_leader(self, preferred: Optional[str] = None) -> ConsensusNode:
        leader = self.leader()
        if leader is not None:
            return leader
        order = ([preferred] if preferred else []) + sorted(self.nodes)
        for candidate_id in order:
            if candidate_id and self.hold_election(candidate_id):
                return self.nodes[candidate_id]
        raise RuntimeError("failed to elect a leader")

    def replicate(self, command: Any, preferred_leader: Optional[str] = None) -> Optional[int]:
        """Replicate a command; returns its committed index or None."""
        leader = self.ensure_leader(preferred_leader)
        entry = LogEntry(term=leader.current_term, index=len(leader.log), command=command)
        leader.log.append(entry)
        acknowledgements = 1
        for peer in sorted(self.nodes.values(), key=lambda node: node.node_id):
            if peer.node_id == leader.node_id:
                continue
            if peer.append_entries([entry], leader.current_term, leader.commit_index):
                acknowledgements += 1
        if acknowledgements >= self.majority:
            leader.commit_index = entry.index
            return entry.index
        leader.log.pop()
        return None

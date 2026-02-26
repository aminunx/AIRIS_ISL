"""
procedural_memory.py
====================
AIRIS-ISL Core Module — Phase 5: Procedural Memory Update

Procedural Memory stores what the system has EXPERIENCED — not what
it has been told.

Key properties:
  - NOT a log of text descriptions
  - IS a weighted registry of (state_hash, action) → outcome patterns
  - Generalises: patterns from level 1 inform decisions in level 8
    if the relevant abstractions match
  - Deduplication: same experience is not stored twice — confidence
    of the existing pattern is updated instead

Author: A.M. Almurish
Project: AIRIS-ISL
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, copy
from typing import Any

from dream_engine import AgentState
from reality_verifier import VerificationResult


# ─────────────────────────────────────────────
#  Procedural Pattern — one unit of experience
# ─────────────────────────────────────────────
@dataclass
class ProceduralPattern:
    """
    A single verified experience unit.

    Fields:
        state_hash       : compact hash of the grid state
        action           : action that was taken
        outcome_type     : what happened (goal_reached, moved, blocked...)
        actual_pos       : where the agent ended up
        confidence       : reliability of this pattern [0,1]
        generation       : abstraction generation this came from
        verified_real    : was this confirmed by reality?
        prediction_error : how far off the simulation was
        level            : which puzzle level this came from
        step             : step number within the episode
        times_seen       : how many times this exact pattern occurred
    """
    state_hash       : str
    action           : str
    outcome_type     : str
    actual_pos       : tuple[int, int]
    confidence       : float = 1.0
    generation       : int   = 0
    verified_real    : bool  = True
    prediction_error : float = 0.0
    level            : int   = 0
    step             : int   = 0
    times_seen       : int   = 1

    @property
    def key(self) -> str:
        """Unique identity of this pattern — used for deduplication."""
        return f"{self.state_hash}:{self.action}"

    def reinforce(self, new_error: float) -> None:
        """Called when the same pattern is seen again successfully."""
        self.times_seen += 1
        self.confidence  = min(1.0, self.confidence + 0.05)
        # running average of prediction error
        self.prediction_error = (
            (self.prediction_error * (self.times_seen - 1) + new_error)
            / self.times_seen
        )

    def to_dict(self) -> dict:
        return {
            "state_hash"       : self.state_hash,
            "action"           : self.action,
            "outcome_type"     : self.outcome_type,
            "actual_pos"       : list(self.actual_pos),
            "confidence"       : round(self.confidence, 4),
            "generation"       : self.generation,
            "verified_real"    : self.verified_real,
            "prediction_error" : round(self.prediction_error, 4),
            "level"            : self.level,
            "step"             : self.step,
            "times_seen"       : self.times_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProceduralPattern":
        d = dict(d)
        d["actual_pos"] = tuple(d["actual_pos"])
        return cls(**d)

    def __repr__(self):
        return (
            f"ProceduralPattern(action={self.action!r} | "
            f"outcome={self.outcome_type!r} | "
            f"conf={self.confidence:.2f} | "
            f"seen={self.times_seen}x)"
        )


# ─────────────────────────────────────────────
#  State Hasher — compact grid fingerprint
# ─────────────────────────────────────────────
def hash_state(grid: list[list[int]], agent: AgentState) -> str:
    """
    Produce a short deterministic hash for (grid, agent_position).
    Two identical states always produce the same hash.
    Used as the key for ProceduralMemory lookup.
    """
    flat   = [cell for row in grid for cell in row]
    digest = hashlib.md5(
        json.dumps([flat, agent.row, agent.col,
                    sorted(agent.inventory)]).encode()
    ).hexdigest()[:12]
    return digest


# ─────────────────────────────────────────────
#  Procedural Memory
# ─────────────────────────────────────────────
class ProceduralMemory:
    """
    The agent's experiential knowledge base.

    Stores ProceduralPatterns indexed by (state_hash, action).
    Supports:
        store(verification_result, grid, agent_state) → stores/updates pattern
        recall(grid, agent_state)                     → returns known outcomes
        best_recalled_action(grid, agent_state)       → highest confidence action
        size()                                        → total patterns stored
        save(path) / load(path)                       → persistence
    """

    def __init__(self):
        self._patterns: dict[str, ProceduralPattern] = {}

    # ── Core Operations ───────────────────────
    def store(
        self,
        result      : VerificationResult,
        grid        : list[list[int]],
        agent_state : AgentState,
    ) -> ProceduralPattern:
        """
        Store or update a pattern from a VerificationResult.

        Deduplication: if the same (state, action) was seen before,
        reinforce the existing pattern instead of creating a duplicate.
        """
        state_hash = hash_state(grid, agent_state)
        pattern_key = f"{state_hash}:{result.action}"

        if pattern_key in self._patterns:
            # Pattern already known — reinforce it
            existing = self._patterns[pattern_key]
            existing.reinforce(result.prediction_error)
            return existing

        # New pattern — create and store
        generation = self._compute_generation(result)
        pattern = ProceduralPattern(
            state_hash       = state_hash,
            action           = result.action,
            outcome_type     = result.actual.outcome_type,
            actual_pos       = (result.actual.new_state.row,
                                result.actual.new_state.col),
            confidence       = 1.0 if result.verified else 0.85,
            generation       = generation,
            verified_real    = result.verified,
            prediction_error = result.prediction_error,
            level            = result.level,
            step             = result.step,
            times_seen       = 1,
        )
        self._patterns[pattern_key] = pattern
        return pattern

    def recall(
        self,
        grid        : list[list[int]],
        agent_state : AgentState,
    ) -> list[ProceduralPattern]:
        """
        Return all known patterns for the current state.
        Sorted by confidence descending.
        """
        state_hash = hash_state(grid, agent_state)
        matches = [
            p for key, p in self._patterns.items()
            if key.startswith(state_hash + ":")
        ]
        return sorted(matches, key=lambda p: p.confidence, reverse=True)

    def best_recalled_action(
        self,
        grid        : list[list[int]],
        agent_state : AgentState,
    ) -> ProceduralPattern | None:
        """
        Return the highest-confidence pattern for this state,
        but only if it led to a positive outcome (not danger/blocked).

        Returns None if no useful pattern exists → fall back to Dream Engine.
        """
        patterns = self.recall(grid, agent_state)
        positive_outcomes = {"goal_reached", "item_collected", "moved"}
        for p in patterns:
            if p.outcome_type in positive_outcomes and p.confidence >= 0.70:
                return p
        return None

    def has_seen(
        self,
        grid        : list[list[int]],
        agent_state : AgentState,
        action      : str,
    ) -> bool:
        """Return True if this exact (state, action) pair was seen before."""
        state_hash  = hash_state(grid, agent_state)
        pattern_key = f"{state_hash}:{action}"
        return pattern_key in self._patterns

    def repeated_mistake_count(self) -> int:
        """
        Count patterns where the agent repeated a dangerous/blocked action.
        Used as a key metric: lower = better learning.
        """
        count = 0
        for p in self._patterns.values():
            if p.outcome_type in ("danger", "blocked") and p.times_seen > 1:
                count += p.times_seen - 1
        return count

    # ── Statistics ────────────────────────────
    def size(self) -> int:
        return len(self._patterns)

    def stats(self) -> dict:
        if not self._patterns:
            return {"size": 0, "avg_confidence": 0.0,
                    "verified_count": 0, "repeated_mistakes": 0}
        confs = [p.confidence for p in self._patterns.values()]
        verified = sum(1 for p in self._patterns.values() if p.verified_real)
        return {
            "size"              : self.size(),
            "avg_confidence"    : round(sum(confs) / len(confs), 4),
            "verified_count"    : verified,
            "repeated_mistakes" : self.repeated_mistake_count(),
        }

    # ── Persistence ───────────────────────────
    def save(self, path: str) -> None:
        data = {k: v.to_dict() for k, v in self._patterns.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProceduralMemory":
        mem = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            mem._patterns[k] = ProceduralPattern.from_dict(v)
        return mem

    # ── Internal ──────────────────────────────
    def _compute_generation(self, result: VerificationResult) -> int:
        """
        Generation of the pattern = max generation of involved abstractions.
        Higher generation → lower initial confidence.
        """
        return 0  # Phase 1: all patterns are gen 0 (from seed abstractions)
                  # Will be updated in Phase 2 when derived abstractions exist

    def __repr__(self):
        s = self.stats()
        return (
            f"ProceduralMemory("
            f"size={s['size']} | "
            f"avg_conf={s['avg_confidence']:.2f} | "
            f"verified={s['verified_count']} | "
            f"repeated_mistakes={s['repeated_mistakes']})"
        )

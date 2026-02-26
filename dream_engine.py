"""
dream_engine.py
===============
AIRIS-ISL Core Module — Phase 2 & 3: Dream Composition + Virtual Experiment

The Dream Engine:
  1. Takes the current real state (grid + position + inventory)
  2. Composes a Virtual Environment from known Abstractions
  3. Runs Virtual Experiments for every candidate action
  4. Returns ranked actions with predicted outcomes + confidence scores

Key principle: The engine does NOT invent — it assembles from what is known.
               It does NOT act in reality — it simulates only.

Author: A.M. Almurish
Project: AIRIS-ISL
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import copy

from abstractions import Abstraction, AbstractionStore, MIN_CONFIDENCE


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
ACTIONS            = ["up", "down", "left", "right", "nothing"]
GRID_ROWS          = 15
GRID_COLS          = 20
UNKNOWN_ENTITY_ID  = -1
VERIFICATION_THRESHOLD = 0.20   # max prediction error to count as verified


# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────
@dataclass
class AgentState:
    """Snapshot of the agent's real position and inventory."""
    row       : int
    col       : int
    inventory : list[str] = field(default_factory=list)

    def clone(self) -> "AgentState":
        return AgentState(self.row, self.col, list(self.inventory))

    def __repr__(self):
        return f"AgentState(row={self.row}, col={self.col}, inv={self.inventory})"


@dataclass
class ExperimentResult:
    """
    The outcome of a single Virtual Experiment.

    Fields:
        action           : the action that was simulated
        predicted_state  : what the agent state will look like after action
        outcome_type     : "goal_reached" | "blocked" | "danger" |
                           "item_collected" | "moved" | "no_change"
        confidence       : product of all abstraction confidences involved
        reasoning        : human-readable trace of why this outcome was predicted
        involved_ids     : entity IDs that participated in this prediction
    """
    action          : str
    predicted_state : AgentState
    outcome_type    : str
    confidence      : float
    reasoning       : str
    involved_ids    : list[int] = field(default_factory=list)

    @property
    def score(self) -> float:
        """
        Composite score used to rank actions.
        Rewards goals, penalises danger, scales by confidence.
        """
        base = {
            "goal_reached"   : 1.0,
            "item_collected" : 0.7,
            "moved"          : 0.3,
            "no_change"      : 0.1,
            "blocked"        : 0.05,
            "danger"         : -1.0,
        }.get(self.outcome_type, 0.1)
        return base * self.confidence

    def __repr__(self):
        return (
            f"ExperimentResult(action={self.action!r} | "
            f"outcome={self.outcome_type!r} | "
            f"score={self.score:.3f} | conf={self.confidence:.2f})"
        )


# ─────────────────────────────────────────────
#  Virtual Environment
# ─────────────────────────────────────────────
class VirtualEnvironment:
    """
    An internal simulation of the AIRIS grid world.
    Built entirely from Abstractions — never from raw AIRIS code.

    The VirtualEnvironment does NOT share state with the real environment.
    It is a disposable clone used for one dream cycle, then discarded.
    """

    def __init__(
        self,
        grid        : list[list[int]],
        agent_state : AgentState,
        store       : AbstractionStore,
    ):
        # deep copy — we must NEVER modify the real grid
        self.grid        = copy.deepcopy(grid)
        self.agent       = agent_state.clone()
        self.store       = store
        self.rows        = len(grid)
        self.cols        = len(grid[0]) if grid else 0
        self._terminated = False

    # ── movement helpers ──────────────────────
    def _next_position(self, action: str) -> tuple[int, int]:
        """Compute next (row, col) for a given action."""
        deltas = {
            "up"     : (-1,  0),
            "down"   : ( 1,  0),
            "left"   : ( 0, -1),
            "right"  : ( 0,  1),
            "nothing": ( 0,  0),
        }
        dr, dc = deltas.get(action, (0, 0))
        return self.agent.row + dr, self.agent.col + dc

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _entity_at(self, row: int, col: int) -> int:
        """Return entity ID at position, or UNKNOWN_ENTITY_ID if OOB."""
        if not self._in_bounds(row, col):
            return 2  # treat out-of-bounds as wall
        return self.grid[row][col]

    # ── core simulation step ──────────────────
    def step(self, action: str) -> ExperimentResult:
        """
        Simulate one action inside the virtual environment.
        Uses Abstraction properties to determine outcome.
        Returns an ExperimentResult — never modifies the real world.
        """
        if action == "nothing":
            return ExperimentResult(
                action="nothing",
                predicted_state=self.agent.clone(),
                outcome_type="no_change",
                confidence=1.0,
                reasoning="Agent chose to stay. No interaction.",
                involved_ids=[],
            )

        next_row, next_col = self._next_position(action)
        entity_id          = self._entity_at(next_row, next_col)
        abstraction        = self.store.get_active(entity_id)

        # ── unknown entity ────────────────────
        if abstraction is None:
            return ExperimentResult(
                action=action,
                predicted_state=self.agent.clone(),
                outcome_type="blocked",
                confidence=MIN_CONFIDENCE,
                reasoning=(
                    f"Entity {entity_id} at ({next_row},{next_col}) "
                    f"is unknown or quarantined. Treating as obstacle."
                ),
                involved_ids=[entity_id],
            )

        confidence    = abstraction.confidence
        involved_ids  = [entity_id]
        new_agent     = self.agent.clone()

        # ── DANGER — fatal entity ─────────────
        if abstraction.is_dangerous:
            return ExperimentResult(
                action=action,
                predicted_state=self.agent.clone(),  # stays — episode ends
                outcome_type="danger",
                confidence=confidence,
                reasoning=(
                    f"'{abstraction.name}' at ({next_row},{next_col}) "
                    f"is dangerous (fatal={abstraction.properties.get('fatal',False)}). "
                    f"Moving here terminates the episode. AVOID."
                ),
                involved_ids=involved_ids,
            )

        # ── BLOCKED — non-passable entity ─────
        if not abstraction.is_passable:
            # special case: door + key in inventory
            if abstraction.name == "door" and "key" in self.agent.inventory:
                new_agent.row = next_row
                new_agent.col = next_col
                new_agent.inventory.remove("key")
                return ExperimentResult(
                    action=action,
                    predicted_state=new_agent,
                    outcome_type="moved",
                    confidence=confidence * self.store.get_active(5).confidence
                               if self.store.get_active(5) else confidence * 0.85,
                    reasoning=(
                        f"Door at ({next_row},{next_col}) opened using key from inventory. "
                        f"Agent moves through."
                    ),
                    involved_ids=[entity_id, 5],
                )
            return ExperimentResult(
                action=action,
                predicted_state=self.agent.clone(),
                outcome_type="blocked",
                confidence=confidence,
                reasoning=(
                    f"'{abstraction.name}' at ({next_row},{next_col}) "
                    f"is not passable. Movement blocked."
                ),
                involved_ids=involved_ids,
            )

        # ── COLLECTABLE item ──────────────────
        new_agent.row = next_row
        new_agent.col = next_col

        if abstraction.is_collectable:
            new_agent.inventory.append(abstraction.name)
            outcome = "goal_reached" if abstraction.is_goal else "item_collected"
            return ExperimentResult(
                action=action,
                predicted_state=new_agent,
                outcome_type=outcome,
                confidence=confidence,
                reasoning=(
                    f"'{abstraction.name}' at ({next_row},{next_col}) "
                    f"is collectable. Added to inventory. "
                    f"{'PRIMARY GOAL REACHED.' if abstraction.is_goal else ''}"
                ),
                involved_ids=involved_ids,
            )

        # ── SIMPLE MOVE ───────────────────────
        return ExperimentResult(
            action=action,
            predicted_state=new_agent,
            outcome_type="moved",
            confidence=confidence,
            reasoning=(
                f"'{abstraction.name}' at ({next_row},{next_col}) "
                f"is passable. Agent moves freely."
            ),
            involved_ids=involved_ids,
        )


# ─────────────────────────────────────────────
#  Dream Engine
# ─────────────────────────────────────────────
class DreamEngine:
    """
    Orchestrates Phase 2 (Dream Composition) and Phase 3 (Virtual Experiments).

    Usage:
        engine  = DreamEngine(store)
        results = engine.run(grid, agent_state)
        best    = results[0]   # highest scoring action
    """

    def __init__(self, store: AbstractionStore):
        self.store          = store
        self.total_dreams   = 0   # lifetime counter

    def run(
        self,
        grid        : list[list[int]],
        agent_state : AgentState,
        actions     : list[str] | None = None,
    ) -> list[ExperimentResult]:
        """
        Phase 2 + Phase 3 combined.

        1. Compose a VirtualEnvironment from Abstractions (Phase 2)
        2. Run one VirtualExperiment per action (Phase 3)
        3. Return results sorted by score descending

        Args:
            grid        : current real grid (will be deep-copied, never modified)
            agent_state : current real agent state (will be cloned)
            actions     : candidate actions (default: all 5)

        Returns:
            list[ExperimentResult] sorted by score descending
        """
        if actions is None:
            actions = ACTIONS

        # Phase 2 — Compose Virtual Environment
        venv = VirtualEnvironment(grid, agent_state, self.store)

        # Phase 3 — Run Virtual Experiments
        results: list[ExperimentResult] = []
        for action in actions:
            venv_clone = VirtualEnvironment(grid, agent_state, self.store)
            result     = venv_clone.step(action)
            results.append(result)

        self.total_dreams += 1

        # Sort by score descending — best action first
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def best_action(
        self,
        grid        : list[list[int]],
        agent_state : AgentState,
    ) -> ExperimentResult:
        """Convenience method — returns only the top-ranked action."""
        return self.run(grid, agent_state)[0]

    def summarise(
        self,
        results: list[ExperimentResult],
    ) -> str:
        """Human-readable summary of a dream cycle — for logging."""
        lines = ["── Dream Cycle Summary ──────────────────"]
        for r in results:
            lines.append(
                f"  {r.action:<8} | {r.outcome_type:<16} | "
                f"score={r.score:+.3f} | conf={r.confidence:.2f}"
            )
        lines.append(f"  → Best: {results[0].action!r} ({results[0].outcome_type})")
        return "\n".join(lines)

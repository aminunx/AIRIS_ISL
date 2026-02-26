"""
isl_agent.py
============
AIRIS-ISL Core Module — The Complete ISL Agent

Orchestrates all 5 phases in a single decision loop:

  Phase 1: ABSTRACTION LOADING  — what do I know about this state?
  Phase 2: DREAM COMPOSITION    — build virtual environment
  Phase 3: VIRTUAL EXPERIMENT   — test all actions in simulation
  Phase 4: REALITY VERIFICATION — compare prediction vs reality
  Phase 5: PROCEDURAL MEMORY    — store what was experienced

Decision Priority:
  1. Check ProceduralMemory first — if a high-confidence pattern
     exists for this state, use it directly (no dreaming needed)
  2. If no known pattern → run Dream Engine
  3. Execute best action in reality
  4. Verify and update

Author: A.M. Almurish
Project: AIRIS-ISL
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from collections import deque
import random
import time

from abstractions  import AbstractionStore, SEED_ABSTRACTIONS
from dream_engine  import DreamEngine, AgentState, ExperimentResult, ACTIONS
from reality_verifier import RealityVerifier, ActualOutcome, VerificationResult
from procedural_memory import ProceduralMemory, hash_state


# ─────────────────────────────────────────────
#  Decision Record — what happened at each step
# ─────────────────────────────────────────────
@dataclass
class StepRecord:
    """Full audit trail for one agent step."""
    step             : int
    level            : int
    action           : str
    decision_source  : str   # "memory" | "dream" | "fallback"
    predicted_outcome: str
    actual_outcome   : str
    prediction_error : float
    verified         : bool
    dream_rank       : int   # rank of chosen action in dream results (1=best)
    memory_hit       : bool  # was there a matching memory pattern?
    elapsed_ms       : float

    def to_dict(self) -> dict:
        return {
            "step"             : self.step,
            "level"            : self.level,
            "action"           : self.action,
            "decision_source"  : self.decision_source,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome"   : self.actual_outcome,
            "prediction_error" : round(self.prediction_error, 4),
            "verified"         : self.verified,
            "dream_rank"       : self.dream_rank,
            "memory_hit"       : self.memory_hit,
            "elapsed_ms"       : round(self.elapsed_ms, 2),
        }


# ─────────────────────────────────────────────
#  Episode Summary — one complete puzzle run
# ─────────────────────────────────────────────
@dataclass
class EpisodeSummary:
    """Summary of one complete puzzle episode."""
    level               : int
    total_steps         : int
    success             : bool
    repeated_mistakes   : int
    avg_prediction_error: float
    prediction_accuracy : float
    memory_hits         : int
    dream_decisions     : int
    patterns_stored     : int
    duration_s          : float
    step_records        : list[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level"               : self.level,
            "total_steps"         : self.total_steps,
            "success"             : self.success,
            "repeated_mistakes"   : self.repeated_mistakes,
            "avg_prediction_error": round(self.avg_prediction_error, 4),
            "prediction_accuracy" : round(self.prediction_accuracy, 4),
            "memory_hits"         : self.memory_hits,
            "dream_decisions"     : self.dream_decisions,
            "patterns_stored"     : self.patterns_stored,
            "duration_s"          : round(self.duration_s, 3),
        }


# ─────────────────────────────────────────────
#  ISL Agent
# ─────────────────────────────────────────────
class ISLAgent:
    """
    The complete Internal Simulation Learning agent.

    Wires together all 5 ISL phases into a single decision loop.
    Connects to any environment that implements the ISLEnvironment protocol.

    Usage:
        env   = AirisEnv(level=1)
        agent = ISLAgent()
        summary = agent.run_episode(env, level=1, max_steps=500)
    """

    def __init__(self, max_steps: int = 500, verbose: bool = False):
        # Phase 1 — Abstraction Store
        self.store   = AbstractionStore()
        self.store.load_seeds()

        # Phase 2+3 — Dream Engine
        self.dream   = DreamEngine(self.store)

        # Phase 4 — Reality Verifier
        self.verifier = RealityVerifier(self.store)

        # Phase 5 — Procedural Memory
        self.memory  = ProceduralMemory()

        self.max_steps = max_steps
        self.verbose   = verbose

        # Lifetime counters
        self._total_steps    = 0
        self._total_episodes = 0

    # ── Main API ──────────────────────────────
    def run_episode(
        self,
        env,
        level    : int = 0,
        max_steps: int | None = None,
    ) -> EpisodeSummary:
        """
        Run one complete episode using the 5-phase ISL loop.

        Args:
            env      : environment implementing ISLEnvironment protocol
            level    : puzzle level number (for logging)
            max_steps: override default max_steps

        Returns:
            EpisodeSummary with full metrics
        """
        steps_limit = max_steps or self.max_steps
        episode_start = time.time()
        step_records: list[StepRecord] = []
        errors: list[float] = []
        patterns_before = self.memory.size()
        self.memory._repeat_count = {}

        grid, agent_state = env.reset()
        done = False
        step = 0
        position_history: deque[tuple[int, int]] = deque(maxlen=5)

        while not done and step < steps_limit:
            step_start = time.time()
            step += 1
            self._total_steps += 1
            position_history.append((agent_state.row, agent_state.col))

            # ── Phase 1: Abstraction Loading ──
            # (implicit — store is always loaded; we activate relevant ones)
            active_abstractions = self._load_active(grid)

            # ── Decision Priority ─────────────
            # Check memory first — if high-confidence pattern exists, use it
            memory_pattern = self.memory.best_recalled_action(grid, agent_state)
            memory_hit     = memory_pattern is not None
            dream_results: list[ExperimentResult] = []
            selected_prediction: ExperimentResult | None = None

            if memory_hit:
                # Memory path: skip dreaming — use known pattern
                chosen_action      = memory_pattern.action
                decision_source    = "memory"
                predicted_outcome  = memory_pattern.outcome_type
                dream_rank         = 0
            else:
                # Dream path: Phase 2+3 — build virtual env + run experiments
                dream_results     = self.dream.run(grid, agent_state)
                selected_prediction = dream_results[0]
                chosen_action     = selected_prediction.action
                predicted_outcome = selected_prediction.outcome_type
                decision_source   = "dream"
                dream_rank        = 1  # best action = rank 1

                # Fix A: avoid final "nothing" choice unless truly stuck.
                if chosen_action == "nothing":
                    non_nothing = [r for r in dream_results if r.action != "nothing"]
                    if non_nothing:
                        selected_prediction = non_nothing[0]
                        chosen_action = selected_prediction.action
                        predicted_outcome = selected_prediction.outcome_type
                        dream_rank = dream_results.index(selected_prediction) + 1

            # Fix B: stuck detector + epsilon-free exploration among non-blocked actions.
            if self._is_stuck(position_history):
                if not dream_results:
                    dream_results = self.dream.run(grid, agent_state)
                non_blocked = [
                    r for r in dream_results
                    if r.action != "nothing" and r.outcome_type not in ("blocked", "danger")
                ]
                if non_blocked:
                    selected_prediction = random.choice(non_blocked)
                    chosen_action = selected_prediction.action
                    predicted_outcome = selected_prediction.outcome_type
                    dream_rank = dream_results.index(selected_prediction) + 1
                    decision_source = "fallback"
                    memory_hit = False

            if self.verbose:
                print(f"  Step {step:03d} | src={decision_source:<7} | "
                      f"action={chosen_action:<6} | predicted={predicted_outcome}")

            # ── Phase 4: Execute + Verify ─────
            new_grid, new_agent_state, done, actual_outcome = env.step(chosen_action)

            predicted_exp = (
                selected_prediction
                if selected_prediction is not None
                else self._make_synthetic_prediction(
                    chosen_action, memory_pattern, agent_state
                )
            )

            vresult = self.verifier.verify(
                predicted  = predicted_exp,
                actual     = actual_outcome,
                step       = step,
                level      = level,
            )
            errors.append(vresult.prediction_error)

            # ── Phase 5: Procedural Memory Update ──
            self.memory.store(vresult, grid, agent_state)

            # ── Step Record ───────────────────
            elapsed_ms = (time.time() - step_start) * 1000
            record = StepRecord(
                step             = step,
                level            = level,
                action           = chosen_action,
                decision_source  = decision_source,
                predicted_outcome= predicted_outcome,
                actual_outcome   = actual_outcome.outcome_type,
                prediction_error = vresult.prediction_error,
                verified         = vresult.verified,
                dream_rank       = dream_rank,
                memory_hit       = memory_hit,
                elapsed_ms       = elapsed_ms,
            )
            step_records.append(record)

            # Advance state
            grid        = new_grid
            agent_state = new_agent_state

        # ── Episode Summary ───────────────────
        self._total_episodes += 1
        duration_s          = time.time() - episode_start
        avg_error           = sum(errors) / len(errors) if errors else 0.0
        patterns_stored     = self.memory.size() - patterns_before
        memory_hits_count   = sum(1 for r in step_records if r.memory_hit)
        dream_decisions     = sum(1 for r in step_records if r.decision_source == "dream")
        repeated_mistakes   = self.memory.repeated_mistake_count()

        summary = EpisodeSummary(
            level               = level,
            total_steps         = step,
            success             = done,
            repeated_mistakes   = repeated_mistakes,
            avg_prediction_error= avg_error,
            prediction_accuracy = 1.0 - avg_error,
            memory_hits         = memory_hits_count,
            dream_decisions     = dream_decisions,
            patterns_stored     = patterns_stored,
            duration_s          = duration_s,
            step_records        = step_records,
        )

        if self.verbose:
            self._print_summary(summary)

        return summary

    # ── Helpers ───────────────────────────────
    def _load_active(self, grid: list[list[int]]) -> list:
        """Return active abstractions for entities present in grid."""
        entity_ids = {cell for row in grid for cell in row}
        return [
            self.store.get_active(eid)
            for eid in entity_ids
            if self.store.get_active(eid)
        ]

    def _is_stuck(self, history: deque[tuple[int, int]]) -> bool:
        """
        Agent is stuck if the last 5 recorded positions are identical.
        """
        if len(history) < history.maxlen:
            return False
        first = history[0]
        return all(pos == first for pos in history)

    def _make_synthetic_prediction(
        self,
        action        : str,
        memory_pattern,
        agent_state   : AgentState,
    ) -> ExperimentResult:
        """
        When using memory (no dream run), create a synthetic ExperimentResult
        from the stored pattern so the verifier has something to compare.
        """
        row, col = memory_pattern.actual_pos
        return ExperimentResult(
            action          = action,
            predicted_state = AgentState(row=row, col=col,
                                         inventory=list(agent_state.inventory)),
            outcome_type    = memory_pattern.outcome_type,
            confidence      = memory_pattern.confidence,
            reasoning       = f"From procedural memory (seen {memory_pattern.times_seen}x)",
            involved_ids    = [],
        )

    def _print_summary(self, s: EpisodeSummary) -> None:
        print(f"\n{'─'*52}")
        print(f"  Episode Summary — Level {s.level}")
        print(f"{'─'*52}")
        print(f"  Steps         : {s.total_steps}")
        print(f"  Success       : {'✅' if s.success else '❌'}")
        print(f"  Pred Accuracy : {s.prediction_accuracy:.1%}")
        print(f"  Memory Hits   : {s.memory_hits}")
        print(f"  Dream Runs    : {s.dream_decisions}")
        print(f"  Patterns New  : {s.patterns_stored}")
        print(f"  Repeat Errors : {s.repeated_mistakes}")
        print(f"  Duration      : {s.duration_s:.2f}s")
        print(f"{'─'*52}\n")

    # ── Agent Stats ───────────────────────────
    def stats(self) -> dict:
        return {
            "total_steps"       : self._total_steps,
            "total_episodes"    : self._total_episodes,
            "total_dreams"      : self.dream.total_dreams,
            "memory_size"       : self.memory.size(),
            "verifier_accuracy" : self.verifier.accuracy,
            "active_abstractions": len(self.store.active_all()),
        }

    def __repr__(self):
        s = self.stats()
        return (
            f"ISLAgent("
            f"episodes={s['total_episodes']} | "
            f"steps={s['total_steps']} | "
            f"memory={s['memory_size']} | "
            f"accuracy={s['verifier_accuracy']:.1%})"
        )

"""
reality_verifier.py
===================
AIRIS-ISL Core Module — Phase 4: Reality Verification

After the agent executes the best action in the real environment,
the Reality Verifier:
  1. Compares the predicted outcome (from Dream Engine) with
     the actual outcome (from the real environment)
  2. Computes a prediction error score
  3. Updates abstraction confidences accordingly
  4. Returns a VerificationResult for logging

Rule 5 — Reality is the Final Arbiter:
  No matter how confident the simulation, the real outcome always
  takes precedence in updating the Procedural Memory.

Author: A.M. Almurish
Project: AIRIS-ISL
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from abstractions import AbstractionStore, Abstraction
from dream_engine import ExperimentResult, AgentState


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
VERIFICATION_THRESHOLD = 0.20   # max error to count as verified
POSITION_WEIGHT        = 0.50   # weight of position match in error
OUTCOME_WEIGHT         = 0.50   # weight of outcome type match in error


# ─────────────────────────────────────────────
#  Actual Outcome — what really happened
# ─────────────────────────────────────────────
@dataclass
class ActualOutcome:
    """
    The ground truth result after executing an action in the real environment.

    Fields:
        action        : the action that was executed
        new_state     : the real agent state after the action
        outcome_type  : what actually happened
        entities_seen : entity IDs the agent interacted with
    """
    action        : str
    new_state     : AgentState
    outcome_type  : str
    entities_seen : list[int] = field(default_factory=list)

    def __repr__(self):
        return (
            f"ActualOutcome(action={self.action!r} | "
            f"outcome={self.outcome_type!r} | "
            f"pos=({self.new_state.row},{self.new_state.col}))"
        )


# ─────────────────────────────────────────────
#  Verification Result
# ─────────────────────────────────────────────
@dataclass
class VerificationResult:
    """
    The complete record of one verification cycle.

    Fields:
        action             : the action that was executed
        predicted          : what the Dream Engine expected
        actual             : what the real environment returned
        prediction_error   : float [0.0 = perfect, 1.0 = completely wrong]
        verified           : True if error < VERIFICATION_THRESHOLD
        abstractions_boosted  : names of abstractions whose confidence increased
        abstractions_corrected: names of abstractions that were corrected
        step               : step number in the episode
        level              : puzzle level number
    """
    action                : str
    predicted             : ExperimentResult
    actual                : ActualOutcome
    prediction_error      : float
    verified              : bool
    abstractions_boosted  : list[str] = field(default_factory=list)
    abstractions_corrected: list[str] = field(default_factory=list)
    step                  : int = 0
    level                 : int = 0

    @property
    def prediction_accuracy(self) -> float:
        """1.0 = perfect prediction, 0.0 = completely wrong."""
        return 1.0 - self.prediction_error

    def __repr__(self):
        status = "✅ VERIFIED" if self.verified else "❌ CORRECTED"
        return (
            f"VerificationResult({status} | "
            f"error={self.prediction_error:.3f} | "
            f"action={self.action!r} | "
            f"predicted={self.predicted.outcome_type!r} | "
            f"actual={self.actual.outcome_type!r})"
        )

    def to_dict(self) -> dict:
        return {
            "action"               : self.action,
            "predicted_outcome"    : self.predicted.outcome_type,
            "actual_outcome"       : self.actual.outcome_type,
            "predicted_pos"        : (self.predicted.predicted_state.row,
                                      self.predicted.predicted_state.col),
            "actual_pos"           : (self.actual.new_state.row,
                                      self.actual.new_state.col),
            "prediction_error"     : round(self.prediction_error, 4),
            "verified"             : self.verified,
            "abstractions_boosted" : self.abstractions_boosted,
            "abstractions_corrected": self.abstractions_corrected,
            "step"                 : self.step,
            "level"                : self.level,
        }


# ─────────────────────────────────────────────
#  Reality Verifier
# ─────────────────────────────────────────────
class RealityVerifier:
    """
    Compares Dream Engine predictions against real environment outcomes.
    Updates AbstractionStore confidence based on results.

    Usage:
        verifier = RealityVerifier(store)
        result   = verifier.verify(predicted, actual, step=1, level=1)
    """

    def __init__(self, store: AbstractionStore):
        self.store              = store
        self.total_verifications = 0
        self.total_verified      = 0   # predictions that were correct
        self.total_corrected     = 0   # predictions that were wrong

    # ── Main API ──────────────────────────────
    def verify(
        self,
        predicted : ExperimentResult,
        actual    : ActualOutcome,
        step      : int = 0,
        level     : int = 0,
    ) -> VerificationResult:
        """
        Phase 4 — Reality Verification.

        1. Compute prediction error
        2. Update abstraction confidences
        3. Return full VerificationResult for logging
        """
        error    = self._compute_error(predicted, actual)
        verified = error < VERIFICATION_THRESHOLD

        boosted   = []
        corrected = []

        if verified:
            # Prediction was correct — strengthen involved abstractions
            boosted   = self._boost(predicted.involved_ids)
            self.total_verified += 1
        else:
            # Prediction was wrong — correct involved abstractions
            corrected = self._correct(predicted, actual)
            self.total_corrected += 1

        self.total_verifications += 1

        return VerificationResult(
            action=predicted.action,
            predicted=predicted,
            actual=actual,
            prediction_error=error,
            verified=verified,
            abstractions_boosted=boosted,
            abstractions_corrected=corrected,
            step=step,
            level=level,
        )

    # ── Error Computation ─────────────────────
    def _compute_error(
        self,
        predicted : ExperimentResult,
        actual    : ActualOutcome,
    ) -> float:
        """
        Prediction error in [0.0, 1.0].

        Components:
          position_error : did the agent end up where expected?
          outcome_error  : did the outcome type match?

        Final error = POSITION_WEIGHT * pos_err + OUTCOME_WEIGHT * out_err
        """
        position_error = self._position_error(
            predicted.predicted_state, actual.new_state
        )
        outcome_error = 0.0 if (
            predicted.outcome_type == actual.outcome_type
        ) else 1.0

        return (POSITION_WEIGHT * position_error +
                OUTCOME_WEIGHT  * outcome_error)

    def _position_error(
        self,
        predicted_state : AgentState,
        actual_state    : AgentState,
    ) -> float:
        """
        0.0 = exact position match
        1.0 = position completely wrong

        Uses normalised Manhattan distance capped at 1.0.
        Max expected distance in AIRIS grid ≈ 33 (20+15-2).
        """
        manhattan = (
            abs(predicted_state.row - actual_state.row) +
            abs(predicted_state.col - actual_state.col)
        )
        MAX_DIST = 33.0
        return min(1.0, manhattan / MAX_DIST)

    # ── Abstraction Updates ───────────────────
    def _boost(self, entity_ids: list[int]) -> list[str]:
        """
        Strengthen confidence of all abstractions involved
        in a successful prediction.
        Returns list of boosted abstraction names.
        """
        boosted = []
        for eid in entity_ids:
            abs_ = self.store.get(eid)
            if abs_:
                abs_.verify_success()
                boosted.append(abs_.name)
        return boosted

    def _correct(
        self,
        predicted : ExperimentResult,
        actual    : ActualOutcome,
    ) -> list[str]:
        """
        Correct abstractions when prediction was wrong.

        Strategy:
          - Penalise all involved abstractions
          - If outcome_type differs: update the primary entity's
            properties to reflect what actually happened
          - Rule 3: NEVER delete — only correct and penalise

        Returns list of corrected abstraction names.
        """
        corrected = []

        # Penalise all involved abstractions
        for eid in predicted.involved_ids:
            abs_ = self.store.get(eid)
            if abs_:
                abs_.verify_failure()
                corrected.append(abs_.name)

        # Deep correction: update properties based on actual outcome
        if predicted.outcome_type != actual.outcome_type:
            self._deep_correct(predicted, actual)

        return corrected

    def _deep_correct(
        self,
        predicted : ExperimentResult,
        actual    : ActualOutcome,
    ) -> None:
        """
        Update abstraction properties to reflect the actual outcome.

        Examples:
          predicted=blocked, actual=moved   → entity is actually passable
          predicted=moved,   actual=danger  → entity is actually dangerous
          predicted=blocked, actual=danger  → entity is dangerous not just solid
        """
        if not predicted.involved_ids:
            return

        primary_id  = predicted.involved_ids[0]
        abs_        = self.store.get(primary_id)
        if not abs_:
            return

        correction_map = {
            # (predicted, actual) → property corrections
            ("blocked", "moved")        : {"passable": True},
            ("blocked", "danger")       : {"dangerous": True, "fatal": True},
            ("moved",   "blocked")      : {"passable": False},
            ("moved",   "danger")       : {"dangerous": True, "passable": False},
            ("moved",   "goal_reached") : {"goal": True, "collectable": True},
            ("moved",   "item_collected"): {"collectable": True},
            ("danger",  "moved")        : {"dangerous": False, "passable": True},
            ("danger",  "blocked")      : {"dangerous": False, "passable": False},
        }

        key = (predicted.outcome_type, actual.outcome_type)
        corrections = correction_map.get(key, {})
        if corrections:
            abs_.correct(corrections)

    # ── Statistics ────────────────────────────
    @property
    def accuracy(self) -> float:
        """Overall prediction accuracy so far."""
        if self.total_verifications == 0:
            return 0.0
        return self.total_verified / self.total_verifications

    def stats(self) -> dict:
        return {
            "total_verifications" : self.total_verifications,
            "total_verified"      : self.total_verified,
            "total_corrected"     : self.total_corrected,
            "accuracy"            : round(self.accuracy, 4),
        }

    def __repr__(self):
        return (
            f"RealityVerifier("
            f"accuracy={self.accuracy:.2%} | "
            f"verified={self.total_verified} | "
            f"corrected={self.total_corrected})"
        )

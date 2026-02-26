"""
test_verifier_and_memory.py
===========================
AIRIS-ISL â€” Tests for reality_verifier.py + procedural_memory.py

Author: A.M. Almurish | Project: AIRIS-ISL
"""

import pytest, sys, copy
sys.path.insert(0, "/home/user")

from abstractions import AbstractionStore, MIN_CONFIDENCE, VERIFY_BOOST, VERIFY_PENALTY
from dream_engine import DreamEngine, AgentState, ExperimentResult
from reality_verifier import (RealityVerifier, ActualOutcome,
                               VerificationResult, VERIFICATION_THRESHOLD)
from procedural_memory import ProceduralMemory, ProceduralPattern, hash_state


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Shared Fixtures
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@pytest.fixture
def store():
    s = AbstractionStore(); s.load_seeds(); return s

@pytest.fixture
def verifier(store):
    return RealityVerifier(store)

@pytest.fixture
def memory():
    return ProceduralMemory()

@pytest.fixture
def ac():
    return AgentState(row=2, col=2)

def mg(r=5, c=5, fill=0):
    return [[fill]*c for _ in range(r)]

def make_predicted(outcome_type, row=1, col=2, conf=1.0, action="up", eids=None):
    return ExperimentResult(
        action=action,
        predicted_state=AgentState(row=row, col=col),
        outcome_type=outcome_type,
        confidence=conf,
        reasoning="test",
        involved_ids=eids or [],
    )

def make_actual(outcome_type, row=1, col=2, action="up", eids=None):
    return ActualOutcome(
        action=action,
        new_state=AgentState(row=row, col=col),
        outcome_type=outcome_type,
        entities_seen=eids or [],
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 1 â€” Error Computation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestErrorComputation:

    def test_perfect_prediction_zero_error(self, verifier):
        """Identical predicted and actual â†’ error = 0."""
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=1, col=2)
        error  = verifier._compute_error(pred, actual)
        assert error == pytest.approx(0.0)

    def test_wrong_outcome_adds_error(self, verifier):
        """Different outcome_type â†’ error â‰¥ OUTCOME_WEIGHT (0.5)."""
        pred   = make_predicted("moved",   row=1, col=2)
        actual = make_actual("blocked",    row=2, col=2)
        error  = verifier._compute_error(pred, actual)
        assert error >= 0.5

    def test_wrong_position_adds_error(self, verifier):
        """Different position â†’ error > 0."""
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=3, col=2)
        error  = verifier._compute_error(pred, actual)
        assert error > 0.0

    def test_perfect_error_below_threshold(self, verifier):
        """Zero error must be below verification threshold."""
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=1, col=2)
        error  = verifier._compute_error(pred, actual)
        assert error < VERIFICATION_THRESHOLD

    def test_wrong_outcome_above_threshold(self, verifier):
        """Completely wrong prediction must exceed threshold."""
        pred   = make_predicted("moved",  row=1, col=2)
        actual = make_actual("danger",    row=2, col=2)
        error  = verifier._compute_error(pred, actual)
        assert error >= VERIFICATION_THRESHOLD

    def test_error_capped_at_one(self, verifier):
        """Error must never exceed 1.0."""
        pred   = make_predicted("moved",   row=0,  col=0)
        actual = make_actual("danger",     row=14, col=19)
        error  = verifier._compute_error(pred, actual)
        assert error <= 1.0


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 2 â€” Abstraction Updates via Verify
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestAbstractionUpdates:

    def test_correct_prediction_boosts_abstraction(self, verifier, store):
        store.get(0).confidence = 0.90   # set below 1.0 so boost is visible
        before = store.get(0).confidence
        verifier.verify(
            make_predicted("moved", eids=[0]),
            make_actual("moved")
        )
        assert store.get(0).confidence == pytest.approx(0.90 + VERIFY_BOOST)

    def test_wrong_prediction_penalises_abstraction(self, verifier, store):
        """Failed prediction â†’ involved abstractions get penalised."""
        wall_before = store.get(2).confidence
        pred   = make_predicted("blocked", row=2, col=2, eids=[2])
        actual = make_actual("moved",      row=1, col=2)
        result = verifier.verify(pred, actual)
        assert result.verified is False
        assert store.get(2).confidence < wall_before

    def test_correct_prediction_never_exceeds_one(self, verifier, store):
        """Confidence must never exceed 1.0 after boost."""
        store.get(0).confidence = 0.99
        pred   = make_predicted("moved", row=1, col=2, eids=[0])
        actual = make_actual("moved",    row=1, col=2)
        verifier.verify(pred, actual)
        assert store.get(0).confidence <= 1.0

    def test_deep_correct_blocked_to_moved(self, verifier, store):
        """
        Predicted blocked but actually moved â†’
        primary entity should be corrected to passable=True.
        """
        store.get(2).confidence = 1.0  # wall starts with conf=1.0
        pred   = make_predicted("blocked", row=2, col=2, eids=[2])
        actual = make_actual("moved",      row=1, col=2)
        verifier.verify(pred, actual)
        assert store.get(2).properties.get("passable") is True

    def test_deep_correct_moved_to_danger(self, verifier, store):
        """
        Predicted moved but actually danger â†’
        primary entity updated to dangerous=True, passable=False.
        """
        store.get(0).confidence = 1.0
        pred   = make_predicted("moved",  row=1, col=2, eids=[0])
        actual = make_actual("danger",    row=2, col=2)
        verifier.verify(pred, actual)
        assert store.get(0).properties.get("dangerous") is True

    def test_abstraction_never_deleted_after_failure(self, verifier, store):
        """Rule 3: abstraction must still exist after failed verification."""
        pred   = make_predicted("moved", row=1, col=2, eids=[2])
        actual = make_actual("blocked",  row=2, col=2)
        verifier.verify(pred, actual)
        assert store.get(2) is not None

    def test_no_involved_ids_safe(self, verifier):
        """Verification with empty involved_ids must not crash."""
        pred   = make_predicted("moved",  row=1, col=2, eids=[])
        actual = make_actual("blocked",   row=2, col=2)
        result = verifier.verify(pred, actual)
        assert result is not None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 3 â€” VerificationResult
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestVerificationResult:

    def test_verified_flag_set_correctly(self, verifier):
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=1, col=2)
        result = verifier.verify(pred, actual)
        assert result.verified is True

    def test_not_verified_when_wrong(self, verifier):
        pred   = make_predicted("moved",  row=1, col=2)
        actual = make_actual("danger",    row=2, col=2)
        result = verifier.verify(pred, actual)
        assert result.verified is False

    def test_prediction_accuracy_is_complement_of_error(self, verifier):
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=1, col=2)
        result = verifier.verify(pred, actual)
        assert result.prediction_accuracy == pytest.approx(1.0 - result.prediction_error)

    def test_to_dict_has_required_fields(self, verifier):
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=1, col=2)
        result = verifier.verify(pred, actual)
        d      = result.to_dict()
        for key in ["action","predicted_outcome","actual_outcome",
                    "prediction_error","verified","step","level"]:
            assert key in d

    def test_verifier_stats_accumulate(self, verifier):
        """Stats must track total verifications accurately."""
        pred   = make_predicted("moved", row=1, col=2)
        actual = make_actual("moved",    row=1, col=2)
        verifier.verify(pred, actual)
        verifier.verify(pred, actual)
        assert verifier.total_verifications == 2
        assert verifier.total_verified == 2

    def test_accuracy_tracks_correct_ratio(self, verifier):
        """Accuracy = verified / total."""
        pred_ok  = make_predicted("moved",  row=1, col=2)
        act_ok   = make_actual("moved",     row=1, col=2)
        pred_bad = make_predicted("moved",  row=1, col=2)
        act_bad  = make_actual("danger",    row=2, col=2)
        verifier.verify(pred_ok,  act_ok)   # correct
        verifier.verify(pred_bad, act_bad)  # wrong
        assert verifier.accuracy == pytest.approx(0.5)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 4 â€” hash_state
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestHashState:

    def test_same_state_same_hash(self, ac):
        g1 = mg(); g2 = mg()
        assert hash_state(g1, ac) == hash_state(g2, ac)

    def test_different_position_different_hash(self):
        g = mg()
        a1 = AgentState(2, 2); a2 = AgentState(3, 3)
        assert hash_state(g, a1) != hash_state(g, a2)

    def test_different_grid_different_hash(self, ac):
        g1 = mg(); g2 = mg(); g2[0][0] = 2
        assert hash_state(g1, ac) != hash_state(g2, ac)

    def test_different_inventory_different_hash(self):
        g = mg()
        a1 = AgentState(2, 2, []); a2 = AgentState(2, 2, ["key"])
        assert hash_state(g, a1) != hash_state(g, a2)

    def test_hash_is_12_chars(self, ac):
        g = mg()
        assert len(hash_state(g, ac)) == 12


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 5 â€” ProceduralPattern
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestProceduralPattern:

    def _make_pattern(self, outcome="moved", action="up", conf=1.0):
        return ProceduralPattern(
            state_hash="abc123def456", action=action,
            outcome_type=outcome, actual_pos=(1, 2),
            confidence=conf, generation=0, verified_real=True,
            prediction_error=0.0, level=1, step=1, times_seen=1
        )

    def test_key_format(self):
        p = self._make_pattern()
        assert p.key == "abc123def456:up"

    def test_reinforce_increments_seen(self):
        p = self._make_pattern(conf=0.90)
        p.reinforce(0.0)
        assert p.times_seen == 2

    def test_reinforce_boosts_confidence(self):
        p = self._make_pattern(conf=0.90)
        p.reinforce(0.0)
        assert p.confidence > 0.90

    def test_reinforce_conf_capped(self):
        p = self._make_pattern(conf=0.99)
        p.reinforce(0.0)
        assert p.confidence <= 1.0

    def test_roundtrip_serialization(self):
        p = self._make_pattern(conf=0.85)
        restored = ProceduralPattern.from_dict(p.to_dict())
        assert restored.action       == p.action
        assert restored.outcome_type == p.outcome_type
        assert restored.confidence   == p.confidence
        assert restored.actual_pos   == p.actual_pos


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 6 â€” ProceduralMemory
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestProceduralMemory:

    def _make_vresult(self, outcome="moved", action="up",
                      row=1, col=2, verified=True, error=0.0):
        pred   = make_predicted(outcome, row=row, col=col, action=action)
        actual = make_actual(outcome,    row=row, col=col, action=action)
        # build a minimal VerificationResult
        from reality_verifier import VerificationResult
        return VerificationResult(
            action=action, predicted=pred, actual=actual,
            prediction_error=error, verified=verified,
            step=1, level=1,
        )

    def test_store_adds_pattern(self, memory, ac):
        g = mg()
        r = self._make_vresult("moved")
        memory.store(r, g, ac)
        assert memory.size() == 1

    def test_store_deduplicates(self, memory, ac):
        g = mg()
        r = self._make_vresult("moved")
        memory.store(r, g, ac)
        memory.store(r, g, ac)
        assert memory.size() == 1   # not 2

    def test_dedup_increments_times_seen(self, memory, ac):
        g = mg()
        r = self._make_vresult("moved")
        p1 = memory.store(r, g, ac)
        p2 = memory.store(r, g, ac)
        assert p2.times_seen == 2

    def test_recall_returns_known_patterns(self, memory, ac):
        g = mg()
        r = self._make_vresult("moved", action="up")
        memory.store(r, g, ac)
        recalled = memory.recall(g, ac)
        assert len(recalled) >= 1
        assert recalled[0].action == "up"

    def test_recall_empty_for_unseen_state(self, memory):
        g = mg(); g[0][0] = 9  # unseen state
        recalled = memory.recall(g, AgentState(0, 0))
        assert recalled == []

    def test_has_seen_true_after_store(self, memory, ac):
        g = mg()
        r = self._make_vresult("moved", action="up")
        memory.store(r, g, ac)
        assert memory.has_seen(g, ac, "up") is True

    def test_has_seen_false_for_unseen(self, memory, ac):
        g = mg()
        assert memory.has_seen(g, ac, "up") is False

    def test_best_recalled_skips_danger(self, memory, ac):
        """best_recalled_action must not recommend dangerous patterns."""
        g = mg()
        r = self._make_vresult("danger", action="up", verified=False)
        memory.store(r, g, ac)
        best = memory.best_recalled_action(g, ac)
        assert best is None

    def test_best_recalled_returns_high_conf_positive(self, memory, ac):
        """Returns highest-confidence positive outcome pattern."""
        g = mg()
        r = self._make_vresult("moved", action="right", verified=True)
        memory.store(r, g, ac)
        best = memory.best_recalled_action(g, ac)
        assert best is not None
        assert best.outcome_type in {"goal_reached","item_collected","moved"}

    def test_repeated_mistake_count(self, memory, ac):
        """Repeated dangerous actions must be counted."""
        g = mg()
        r = self._make_vresult("danger", action="up", verified=False)
        memory.store(r, g, ac)
        memory.store(r, g, ac)  # second time = mistake repeated
        assert memory.repeated_mistake_count() >= 1

    def test_stats_returns_correct_size(self, memory, ac):
        g = mg()
        r = self._make_vresult("moved")
        memory.store(r, g, ac)
        assert memory.stats()["size"] == 1

    def test_save_load_roundtrip(self, memory, ac, tmp_path):
        g = mg()
        r = self._make_vresult("moved", action="up")
        memory.store(r, g, ac)
        path = str(tmp_path / "memory.json")
        memory.save(path)
        loaded = ProceduralMemory.load(path)
        assert loaded.size() == memory.size()
        recalled = loaded.recall(g, ac)
        assert len(recalled) == 1
        assert recalled[0].action == "up"


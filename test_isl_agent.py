
import pytest, sys, os; sys.path.insert(0, os.getcwd())
from abstractions import AbstractionStore
from dream_engine import DreamEngine, AgentState
from reality_verifier import RealityVerifier, ActualOutcome
from procedural_memory import ProceduralMemory

def mg(r=6, c=6, fill=0): return [[fill]*c for _ in range(r)]

@pytest.fixture
def components():
    store = AbstractionStore(); store.load_seeds()
    engine = DreamEngine(store)
    verifier = RealityVerifier(store)
    memory = ProceduralMemory()
    return store, engine, verifier, memory

class TestISLCycle:
    def test_dream_avoids_fire(self, components):
        store, engine, _, _ = components
        g = mg(); g[1][2] = 7  # fire above agent
        best = engine.best_action(g, AgentState(2, 2))
        assert best.outcome_type != "danger"

    def test_dream_seeks_battery(self, components):
        store, engine, _, _ = components
        g = mg(); g[1][2] = 3  # battery above
        best = engine.best_action(g, AgentState(2, 2))
        assert best.outcome_type == "goal_reached"

    def test_verifier_correct(self, components):
        from dream_engine import ExperimentResult
        _, _, verifier, _ = components
        pred = ExperimentResult("up", AgentState(1, 2), "moved", 1.0, "t", [0])
        actual = ActualOutcome("up", AgentState(1, 2), "moved", [0])
        vr = verifier.verify(pred, actual, step=1, level=1)
        assert vr.verified and vr.prediction_error == pytest.approx(0.0)

    def test_verifier_wrong(self, components):
        from dream_engine import ExperimentResult
        _, _, verifier, _ = components
        pred = ExperimentResult("up", AgentState(1, 2), "moved", 1.0, "t", [0])
        actual = ActualOutcome("up", AgentState(2, 2), "danger", [7])
        vr = verifier.verify(pred, actual, step=1, level=1)
        assert not vr.verified

    def test_memory_stores(self, components):
        from dream_engine import ExperimentResult
        from reality_verifier import VerificationResult
        _, _, verifier, memory = components
        pred = ExperimentResult("right", AgentState(2, 3), "moved", 1.0, "t", [0])
        actual = ActualOutcome("right", AgentState(2, 3), "moved", [0])
        vr = verifier.verify(pred, actual, step=1, level=1)
        memory.store(vr, mg(), AgentState(2, 2))
        assert memory.size() == 1

    def test_memory_recall(self, components):
        from dream_engine import ExperimentResult
        from reality_verifier import VerificationResult
        _, _, verifier, memory = components
        pred = ExperimentResult("right", AgentState(2, 3), "moved", 1.0, "t", [0])
        actual = ActualOutcome("right", AgentState(2, 3), "moved", [0])
        vr = verifier.verify(pred, actual, step=1, level=1)
        memory.store(vr, mg(), AgentState(2, 2))
        best = memory.best_recalled_action(mg(), AgentState(2, 2))
        assert best is not None and best.action == "right"

    def test_memory_no_repeated_mistakes(self, components):
        _, _, _, memory = components
        assert memory.repeated_mistake_count() == 0

    def test_full_cycle_integration(self, components):
        """Wake→Dream→Verify→Store cycle, no errors thrown"""
        store, engine, verifier, memory = components
        from dream_engine import ExperimentResult
        g = mg(); g[1][2] = 3  # battery
        agent = AgentState(2, 2)

        # Dream
        results = engine.run(g, agent)
        assert results

        # Simulate real outcome matching best prediction
        best = results[0]
        actual = ActualOutcome(
            best.action,
            best.predicted_state,
            best.outcome_type,
            best.involved_ids
        )

        # Verify
        vr = verifier.verify(best, actual, step=1, level=1)
        assert vr.verified

        # Store
        memory.store(vr, g, agent)
        assert memory.size() == 1
        assert memory.repeated_mistake_count() == 0

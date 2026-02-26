"""
test_dream_engine.py
====================
AIRIS-ISL â€” Test Suite for dream_engine.py

Tests cover:
  - VirtualEnvironment: movement, blocking, collecting, danger
  - DreamEngine: ranking, best_action, door+key interaction
  - ExperimentResult: scoring logic
  - Isolation: real grid must never be modified

Author: A.M. Almurish
Project: AIRIS-ISL
"""

import pytest
import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstractions import AbstractionStore, MIN_CONFIDENCE
from dream_engine import (
    DreamEngine,
    VirtualEnvironment,
    ExperimentResult,
    AgentState,
    ACTIONS,
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Shared Fixtures
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@pytest.fixture
def store():
    s = AbstractionStore()
    s.load_seeds()
    return s


@pytest.fixture
def flat_grid():
    """5Ã—5 grid â€” all floor (0), agent at center."""
    g = [[0]*5 for _ in range(5)]
    g[2][2] = 1   # agent position marker (visual only)
    return g


@pytest.fixture
def agent_center():
    return AgentState(row=2, col=2)


def make_grid(rows=5, cols=5, fill=0):
    return [[fill]*cols for _ in range(rows)]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 1 â€” AgentState
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestAgentState:

    def test_clone_is_independent(self):
        """Cloned state must not share reference with original."""
        original = AgentState(row=3, col=4, inventory=["key"])
        cloned   = original.clone()
        cloned.inventory.append("battery")
        assert "battery" not in original.inventory

    def test_clone_preserves_values(self):
        original = AgentState(row=3, col=4, inventory=["key"])
        cloned   = original.clone()
        assert cloned.row == 3
        assert cloned.col == 4
        assert cloned.inventory == ["key"]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 2 â€” ExperimentResult Scoring
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestExperimentResultScoring:

    def _make_result(self, outcome_type, confidence=1.0, action="up"):
        return ExperimentResult(
            action=action,
            predicted_state=AgentState(0, 0),
            outcome_type=outcome_type,
            confidence=confidence,
            reasoning="test",
        )

    def test_goal_reached_is_highest_score(self):
        goal    = self._make_result("goal_reached",    confidence=1.0)
        blocked = self._make_result("blocked",         confidence=1.0)
        assert goal.score > blocked.score

    def test_danger_has_negative_score(self):
        danger = self._make_result("danger", confidence=1.0)
        assert danger.score < 0

    def test_confidence_scales_score(self):
        high = self._make_result("moved", confidence=1.0)
        low  = self._make_result("moved", confidence=0.5)
        assert high.score > low.score

    def test_goal_beats_item_collected(self):
        goal = self._make_result("goal_reached",   confidence=1.0)
        item = self._make_result("item_collected", confidence=1.0)
        assert goal.score > item.score

    def test_item_beats_simple_move(self):
        item = self._make_result("item_collected", confidence=1.0)
        move = self._make_result("moved",          confidence=1.0)
        assert item.score > move.score


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 3 â€” VirtualEnvironment
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestVirtualEnvironment:

    def test_move_onto_floor_returns_moved(self, store, flat_grid, agent_center):
        """Moving onto floor (passable, not collectable) â†’ moved."""
        venv   = VirtualEnvironment(flat_grid, agent_center, store)
        result = venv.step("up")
        assert result.outcome_type == "moved"
        assert result.predicted_state.row == 1
        assert result.predicted_state.col == 2

    def test_move_into_wall_returns_blocked(self, store, agent_center):
        """Moving into wall â†’ blocked, agent stays."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 2  # wall above agent
        venv       = VirtualEnvironment(grid, agent_center, store)
        result     = venv.step("up")
        assert result.outcome_type == "blocked"
        assert result.predicted_state.row == 2  # did not move
        assert result.predicted_state.col == 2

    def test_move_into_fire_returns_danger(self, store, agent_center):
        store.get(7).confidence = 1.0
        grid = make_grid(5, 5, fill=0)
        grid[2][3] = 7  # fire is one step right of agent at (2,2)
        result = VirtualEnvironment(grid, agent_center, store).step("right")
        assert result.outcome_type == "danger"

    def test_collect_battery_returns_goal_reached(self, store, agent_center):
        """Moving onto battery â†’ goal_reached, battery in inventory."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 3  # battery above agent
        venv       = VirtualEnvironment(grid, agent_center, store)
        result     = venv.step("up")
        assert result.outcome_type == "goal_reached"
        assert "battery" in result.predicted_state.inventory

    def test_collect_key_returns_item_collected(self, store, agent_center):
        """Moving onto key â†’ item_collected, key in inventory."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 5  # key above agent
        venv       = VirtualEnvironment(grid, agent_center, store)
        result     = venv.step("up")
        assert result.outcome_type == "item_collected"
        assert "key" in result.predicted_state.inventory

    def test_door_without_key_is_blocked(self, store, agent_center):
        """Door without key in inventory â†’ blocked."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 4  # door above agent
        agent      = AgentState(row=2, col=2, inventory=[])
        venv       = VirtualEnvironment(grid, agent, store)
        result     = venv.step("up")
        assert result.outcome_type == "blocked"

    def test_door_with_key_allows_passage(self, store, agent_center):
        """Door with key in inventory â†’ agent moves through."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 4  # door above agent
        agent      = AgentState(row=2, col=2, inventory=["key"])
        venv       = VirtualEnvironment(grid, agent, store)
        result     = venv.step("up")
        assert result.outcome_type == "moved"
        assert result.predicted_state.row == 1
        assert "key" not in result.predicted_state.inventory  # key consumed

    def test_nothing_action_returns_no_change(self, store, flat_grid, agent_center):
        """Action 'nothing' â†’ no_change, agent stays."""
        venv   = VirtualEnvironment(flat_grid, agent_center, store)
        result = venv.step("nothing")
        assert result.outcome_type == "no_change"
        assert result.predicted_state.row == 2
        assert result.predicted_state.col == 2

    def test_out_of_bounds_treated_as_wall(self, store):
        """Moving out of grid bounds â†’ blocked (treated as wall)."""
        grid  = make_grid(5, 5, fill=0)
        agent = AgentState(row=0, col=0)  # top-left corner
        venv  = VirtualEnvironment(grid, agent, store)
        result_up   = venv.step("up")
        result_left = venv.step("left")
        assert result_up.outcome_type   == "blocked"
        assert result_left.outcome_type == "blocked"

    def test_unknown_entity_returns_blocked_with_low_confidence(self, store, agent_center):
        """Unknown entity (no abstraction) â†’ blocked with MIN_CONFIDENCE."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 99  # no abstraction exists for 99
        venv       = VirtualEnvironment(grid, agent_center, store)
        result     = venv.step("up")
        assert result.outcome_type == "blocked"
        assert result.confidence   == MIN_CONFIDENCE

    def test_real_grid_not_modified(self, store, agent_center):
        """
        CRITICAL: VirtualEnvironment must NEVER modify the real grid.
        """
        grid         = make_grid(5, 5, fill=0)
        grid[1][2]   = 3  # battery
        grid_before  = copy.deepcopy(grid)
        venv         = VirtualEnvironment(grid, agent_center, store)
        venv.step("up")
        assert grid == grid_before  # real grid unchanged


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  SECTION 4 â€” DreamEngine
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class TestDreamEngine:

    def test_run_returns_all_actions(self, store, flat_grid, agent_center):
        """DreamEngine.run() must return one result per candidate action."""
        engine  = DreamEngine(store)
        results = engine.run(flat_grid, agent_center)
        returned_actions = {r.action for r in results}
        assert returned_actions == set(ACTIONS)

    def test_results_sorted_by_score_descending(self, store, flat_grid, agent_center):
        """Results must be sorted highest score first."""
        engine  = DreamEngine(store)
        results = engine.run(flat_grid, agent_center)
        scores  = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_action_avoids_fire(self, store, agent_center):
        """When fire is in one direction, best_action must not choose it."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 7  # fire above
        engine     = DreamEngine(store)
        best       = engine.best_action(grid, agent_center)
        assert best.action != "up"
        assert best.outcome_type != "danger"

    def test_best_action_prefers_battery(self, store, agent_center):
        """When battery is reachable, best_action must pick it."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 3  # battery directly above
        engine     = DreamEngine(store)
        best       = engine.best_action(grid, agent_center)
        assert best.action == "up"
        assert best.outcome_type == "goal_reached"

    def test_best_action_prefers_key_over_floor(self, store, agent_center):
        """Key collection ranks higher than plain movement."""
        grid       = make_grid(5, 5, fill=0)
        grid[1][2] = 5  # key above â€” all others floor
        engine     = DreamEngine(store)
        best       = engine.best_action(grid, agent_center)
        assert best.action == "up"
        assert best.outcome_type == "item_collected"

    def test_dream_counter_increments(self, store, flat_grid, agent_center):
        """total_dreams counter must increment with each run."""
        engine = DreamEngine(store)
        assert engine.total_dreams == 0
        engine.run(flat_grid, agent_center)
        assert engine.total_dreams == 1
        engine.run(flat_grid, agent_center)
        assert engine.total_dreams == 2

    def test_summarise_contains_all_actions(self, store, flat_grid, agent_center):
        """summarise() output must mention every action."""
        engine  = DreamEngine(store)
        results = engine.run(flat_grid, agent_center)
        summary = engine.summarise(results)
        for action in ACTIONS:
            assert action in summary

    def test_real_grid_unchanged_after_engine_run(self, store, agent_center):
        """DreamEngine.run() must never modify the real grid."""
        grid        = make_grid(5, 5, fill=0)
        grid[1][2]  = 3  # battery
        grid_before = copy.deepcopy(grid)
        engine      = DreamEngine(store)
        engine.run(grid, agent_center)
        assert grid == grid_before

    def test_danger_always_ranked_last(self, store, agent_center):
        """
        Action leading to danger must always be the lowest ranked.
        All other options are floor â€” danger must be last.
        """
        grid = make_grid(5, 5, fill=0)
        # surround agent: up=fire, rest=floor
        grid[1][2] = 7
        engine  = DreamEngine(store)
        results = engine.run(grid, agent_center)
        # find danger result
        danger_results = [r for r in results if r.outcome_type == "danger"]
        if danger_results:
            danger_score = danger_results[0].score
            other_scores = [r.score for r in results if r.outcome_type != "danger"]
            assert all(danger_score < s for s in other_scores)


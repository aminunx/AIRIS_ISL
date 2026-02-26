"""
airis_env.py
============
AIRIS-ISL environment adapter.

Provides:
1. `ISLEnvironment` protocol used by `ISLAgent`
2. `MockEnv` that requires no AIRIS dependency
3. `AirisEnv` wrapper that tries real AIRIS first and falls back to `MockEnv`

Author: A.M. Almurish
Project: AIRIS-ISL
Version: 1.0.0
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import copy
import importlib
import importlib.util
import os
from pathlib import Path
import sys

from dream_engine import AgentState
from reality_verifier import ActualOutcome


_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "nothing": (0, 0),
}

_ALLOWED_OUTCOMES = {
    "moved",
    "blocked",
    "danger",
    "goal_reached",
    "item_collected",
    "no_change",
}

_ENTITY_NAMES = {
    0: "floor",
    1: "agent",
    2: "wall",
    3: "battery",
    4: "door",
    5: "key",
    7: "fire",
    9: "extinguisher",
}


@runtime_checkable
class ISLEnvironment(Protocol):
    """Protocol required by ISLAgent."""

    def reset(self) -> tuple[list[list[int]], AgentState]:
        ...

    def step(self, action: str) -> tuple[list[list[int]], AgentState, bool, ActualOutcome]:
        ...


def _clone_grid(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in grid]


def _value_from(source: Any, names: tuple[str, ...]) -> Any:
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _extract_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (tuple, list, set)):
        output: list[int] = []
        for item in value:
            try:
                output.append(int(item))
            except (TypeError, ValueError):
                continue
        return output
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _coerce_grid(raw_grid: Any) -> list[list[int]]:
    if hasattr(raw_grid, "tolist"):
        raw_grid = raw_grid.tolist()
    if not isinstance(raw_grid, list):
        raise ValueError("Grid must be a list of lists.")
    output: list[list[int]] = []
    for row in raw_grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list):
            row = list(row)
        output.append([int(cell) for cell in row])
    return output


def _coerce_agent_state(raw_agent: Any) -> AgentState:
    if isinstance(raw_agent, AgentState):
        return raw_agent.clone()

    if isinstance(raw_agent, dict):
        row = int(raw_agent.get("row", raw_agent.get("agent_row", raw_agent.get("y", 0))))
        col = int(raw_agent.get("col", raw_agent.get("agent_col", raw_agent.get("x", 0))))
        inventory = list(raw_agent.get("inventory", raw_agent.get("items", [])))
        return AgentState(row=row, col=col, inventory=inventory)

    if isinstance(raw_agent, (tuple, list)) and len(raw_agent) >= 2:
        row = int(raw_agent[0])
        col = int(raw_agent[1])
        inventory = list(raw_agent[2]) if len(raw_agent) >= 3 else []
        return AgentState(row=row, col=col, inventory=inventory)

    row = _value_from(raw_agent, ("row", "agent_row", "y"))
    col = _value_from(raw_agent, ("col", "agent_col", "x"))
    inventory = _value_from(raw_agent, ("inventory", "items", "bag"))
    if row is None or col is None:
        raise ValueError("Could not determine agent row/col from AIRIS state.")
    return AgentState(
        row=int(row),
        col=int(col),
        inventory=list(inventory or []),
    )


def _normalise_outcome_type(
    raw_outcome: Any,
    action: str,
    prev_state: AgentState,
    new_state: AgentState,
    done: bool,
    entities_seen: list[int],
) -> str:
    mapping = {
        "moved": "moved",
        "move": "moved",
        "walked": "moved",
        "blocked": "blocked",
        "wall": "blocked",
        "danger": "danger",
        "dead": "danger",
        "death": "danger",
        "burned": "danger",
        "goal_reached": "goal_reached",
        "goal": "goal_reached",
        "success": "goal_reached",
        "win": "goal_reached",
        "item_collected": "item_collected",
        "collected": "item_collected",
        "pickup": "item_collected",
        "picked_up": "item_collected",
        "no_change": "no_change",
        "nothing": "no_change",
        "idle": "no_change",
        "stay": "no_change",
    }

    if isinstance(raw_outcome, str):
        key = raw_outcome.strip().lower()
        if key in mapping:
            return mapping[key]

    if done and 3 in entities_seen:
        return "goal_reached"
    if 7 in entities_seen:
        return "danger"
    if any(eid in (5, 9) for eid in entities_seen):
        return "item_collected"

    same_pos = (prev_state.row == new_state.row and prev_state.col == new_state.col)
    if same_pos:
        if action == "nothing":
            return "no_change"
        if any(eid in (2, 4) for eid in entities_seen):
            return "blocked"
        return "no_change"

    return "moved"


class MockEnv:
    """
    Minimal AIRIS-compatible environment used in tests and local benchmarking.

    Defaults to a simple deterministic grid. When `level` is provided (1..8),
    a benchmark-specific layout is generated with deterministic forced jumps to
    create measurable learning dynamics across repeated episodes.
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        battery_pos: tuple[int, int] = (3, 3),
        fire_pos: tuple[int, int] | None = None,
        key_pos: tuple[int, int] | None = None,
        door_pos: tuple[int, int] | None = None,
        agent_start: tuple[int, int] = (2, 2),
        max_steps: int = 200,
        level: int | None = None,
    ):
        self.level = level
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps

        self.battery_pos = battery_pos
        self.fire_pos = fire_pos
        self.key_pos = key_pos
        self.door_pos = door_pos
        self.agent_start = agent_start

        self._base_grid: list[list[int]] = []
        self._forced_transitions: dict[tuple[int, int, str], tuple[int, int]] = {}

        self._steps = 0
        self._done = False
        self._agent = AgentState(row=agent_start[0], col=agent_start[1], inventory=[])

    def _configure_default_layout(self) -> None:
        self._base_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._forced_transitions = {}

        self._place_if_valid(self.battery_pos, 3)
        self._place_if_valid(self.fire_pos, 7)
        self._place_if_valid(self.key_pos, 5)
        self._place_if_valid(self.door_pos, 4)

    def _configure_benchmark_level(self, level: int) -> None:
        self.rows = 15
        self.cols = 20
        self.max_steps = max(self.max_steps, 180)
        self._base_grid = [[2 for _ in range(self.cols)] for _ in range(self.rows)]
        self._forced_transitions = {}

        left_col = 1
        right_col = 18
        high_rows = [13, 12, 11, 10]
        low_rows = [2, 3, 4]

        path: list[tuple[int, int]] = []
        for idx in range(7):
            if idx % 2 == 0:
                row = high_rows[idx // 2]
                col = left_col
            else:
                row = low_rows[idx // 2]
                col = right_col
            path.append((row, col))

        for idx, (row, col) in enumerate(path):
            self._base_grid[row][col] = 0
            self._base_grid[row - 1][col] = 0
            if idx < len(path) - 1:
                self._forced_transitions[(row, col, "up")] = path[idx + 1]

        self.agent_start = path[0]
        final_row, final_col = path[-1]
        self.battery_pos = (final_row - 1, final_col)
        self._base_grid[self.battery_pos[0]][self.battery_pos[1]] = 3
        self.fire_pos = None
        self.key_pos = None
        self.door_pos = None

    def _place_if_valid(self, pos: tuple[int, int] | None, value: int) -> None:
        if pos is None:
            return
        row, col = pos
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._base_grid[row][col] = value

    def _grid_with_agent(self) -> list[list[int]]:
        grid = _clone_grid(self._base_grid)
        if 0 <= self._agent.row < self.rows and 0 <= self._agent.col < self.cols:
            grid[self._agent.row][self._agent.col] = 1
        return grid

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def reset(self) -> tuple[list[list[int]], AgentState]:
        if self.level is not None and 1 <= self.level <= 8:
            self._configure_benchmark_level(self.level)
        else:
            self._configure_default_layout()

        self._steps = 0
        self._done = False
        self._agent = AgentState(
            row=self.agent_start[0],
            col=self.agent_start[1],
            inventory=[],
        )
        return self._grid_with_agent(), self._agent.clone()

    def step(self, action: str) -> tuple[list[list[int]], AgentState, bool, ActualOutcome]:
        action = action if action in _ACTION_DELTAS else "nothing"

        if self._done:
            outcome = ActualOutcome(
                action=action,
                new_state=self._agent.clone(),
                outcome_type="no_change",
                entities_seen=[],
            )
            return self._grid_with_agent(), self._agent.clone(), self._done, outcome

        previous_state = self._agent.clone()
        entities_seen: list[int] = []
        done = False
        outcome_type = "no_change"

        if action != "nothing":
            forced_key = (self._agent.row, self._agent.col, action)
            forced_pos = self._forced_transitions.get(forced_key)
            if forced_pos is not None:
                self._agent.row, self._agent.col = forced_pos
                entities_seen = [0]
                outcome_type = "moved"
            else:
                dr, dc = _ACTION_DELTAS[action]
                next_row, next_col = self._agent.row + dr, self._agent.col + dc

                if not self._in_bounds(next_row, next_col):
                    entities_seen = [2]
                    outcome_type = "blocked"
                else:
                    entity_id = int(self._base_grid[next_row][next_col])
                    entities_seen = [entity_id]

                    if entity_id == 2:
                        outcome_type = "blocked"
                    elif entity_id == 4:
                        if "key" in self._agent.inventory:
                            self._agent.inventory.remove("key")
                            self._agent.row = next_row
                            self._agent.col = next_col
                            outcome_type = "moved"
                        else:
                            outcome_type = "blocked"
                    elif entity_id == 7:
                        outcome_type = "danger"
                        done = True
                    elif entity_id in (0, 1):
                        self._agent.row = next_row
                        self._agent.col = next_col
                        outcome_type = "moved"
                    elif entity_id == 3:
                        self._agent.row = next_row
                        self._agent.col = next_col
                        self._agent.inventory.append("battery")
                        self._base_grid[next_row][next_col] = 0
                        outcome_type = "goal_reached"
                        done = True
                    elif entity_id == 5:
                        self._agent.row = next_row
                        self._agent.col = next_col
                        if "key" not in self._agent.inventory:
                            self._agent.inventory.append("key")
                        self._base_grid[next_row][next_col] = 0
                        outcome_type = "item_collected"
                    elif entity_id == 9:
                        self._agent.row = next_row
                        self._agent.col = next_col
                        if "extinguisher" not in self._agent.inventory:
                            self._agent.inventory.append("extinguisher")
                        self._base_grid[next_row][next_col] = 0
                        outcome_type = "item_collected"
                    else:
                        outcome_type = "blocked"

        self._steps += 1
        if self._steps >= self.max_steps:
            done = True
        self._done = done

        if outcome_type not in _ALLOWED_OUTCOMES:
            outcome_type = _normalise_outcome_type(
                raw_outcome=outcome_type,
                action=action,
                prev_state=previous_state,
                new_state=self._agent,
                done=done,
                entities_seen=entities_seen,
            )

        outcome = ActualOutcome(
            action=action,
            new_state=self._agent.clone(),
            outcome_type=outcome_type,
            entities_seen=entities_seen,
        )
        return self._grid_with_agent(), self._agent.clone(), self._done, outcome

    def render(self) -> str:
        symbols = {
            0: ".",
            1: "A",
            2: "#",
            3: "B",
            4: "D",
            5: "K",
            7: "F",
            9: "E",
        }
        grid = self._grid_with_agent()
        lines: list[str] = []
        for row in grid:
            lines.append(" ".join(symbols.get(cell, "?") for cell in row))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MockEnv(level={self.level}, size={self.rows}x{self.cols}, "
            f"agent=({self._agent.row},{self._agent.col}), steps={self._steps})"
        )


class AirisEnv:
    """
    Adapter around the real AIRIS engine.

    Connects directly to the real AIRIS_Public puzzle-game engine.
    No silent fallback is allowed.
    """

    # AIRIS_Public ids -> ISL ids
    # AIRIS_Public: 6=extinguisher, 8-11 arrows, 12=open_door, 13-17 character variants
    _ID_MAP = {
        0: 0,   # floor
        1: 1,   # agent
        2: 2,   # wall
        3: 3,   # battery
        4: 4,   # door
        5: 5,   # key
        6: 9,   # extinguisher -> ISL extinguisher id
        7: 7,   # fire
        8: 0,   # arrows -> treat as floor in ISL abstraction space
        9: 0,
        10: 0,
        11: 0,
        12: 0,  # open door -> floor-like/passable in ISL space
        13: 1,  # character on arrow/open-door variants
        14: 1,
        15: 1,
        16: 1,
        17: 1,
    }

    def __init__(self, level: int = 1, max_steps: int = 500, engine: Any | None = None):
        self.level = level
        self.max_steps = max_steps
        self._engine = engine if engine is not None else self._try_create_real_engine(level)

        self._grid: list[list[int]] = []
        self._agent_state = AgentState(row=0, col=0, inventory=[])
        self._done = False
        self._steps = 0

    @property
    def using_mock(self) -> bool:
        return False

    def _error_with_context(self, reasons: list[str]) -> ImportError:
        msg = (
            "Real AIRIS API not found. Missing required AIRIS_Public files/imports.\n"
            "Required from AIRIS_Public:\n"
            "  - airis_stable.py\n"
            "  - puzzle_game_driver_universal.py (Model + PyGameKeyboardController)\n"
            "  - game_objects.py\n"
            "  - images/ assets\n"
            "Details:\n  - "
            + "\n  - ".join(reasons)
        )
        print(msg)
        return ImportError(msg)

    def _try_create_real_engine(self, level: int) -> Any:
        reasons: list[str] = []
        public_root = os.getenv("AIRIS_PUBLIC_PATH", "").strip()
        if public_root:
            root = Path(public_root).expanduser().resolve()
        else:
            root = (Path(__file__).resolve().parent / "AIRIS_Public").resolve()

        required = [
            root / "airis_stable.py",
            root / "puzzle_game_driver_universal.py",
            root / "game_objects.py",
            root / "images",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            reasons.append("Missing AIRIS_Public required paths:")
            reasons.extend(missing)
            raise self._error_with_context(reasons)

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        try:
            import pygame  # type: ignore
        except Exception as exc:
            reasons.append(f"import pygame failed: {exc}")
            reasons.append("Install it with: pip install pygame")
            raise self._error_with_context(reasons)

        try:
            pygame.init()
        except Exception as exc:
            reasons.append(f"pygame.init() failed: {exc}")
            raise self._error_with_context(reasons)

        try:
            cwd_before = Path.cwd()
            os.chdir(root)
            try:
                driver = importlib.import_module("puzzle_game_driver_universal")
            finally:
                os.chdir(cwd_before)
        except Exception as exc:
            reasons.append(f"import puzzle_game_driver_universal failed: {exc}")
            raise self._error_with_context(reasons)

        model_cls = getattr(driver, "Model", None)
        controller_cls = getattr(driver, "PyGameKeyboardController", None)
        if model_cls is None or controller_cls is None:
            reasons.append(
                "puzzle_game_driver_universal is missing Model or PyGameKeyboardController"
            )
            raise self._error_with_context(reasons)

        try:
            cwd_before = Path.cwd()
            os.chdir(root)
            try:
                controller = controller_cls()
                model = model_cls(controller, True)  # ai_controlled=True
            finally:
                os.chdir(cwd_before)
        except Exception as exc:
            reasons.append(f"Model/controller init failed: {exc}")
            raise self._error_with_context(reasons)

        self._airis_public_root = root
        self._controller = controller
        self._model = model
        self._set_level(level)
        return model

    def _with_airis_cwd(self):
        class _CwdCtx:
            def __init__(self, path: Path):
                self.path = path
                self.prev = None

            def __enter__(self):
                self.prev = Path.cwd()
                os.chdir(self.path)

            def __exit__(self, exc_type, exc, tb):
                if self.prev is not None:
                    os.chdir(self.prev)
                return False

        return _CwdCtx(self._airis_public_root)

    def _set_level(self, level: int) -> None:
        level_clamped = max(1, min(13, int(level)))
        with self._with_airis_cwd():
            self._model.current_maze = level_clamped - 1
            self._model.get_next_maze()
            self._model.time_counter = 0
            self._model.current_environment()

    def _inventory_from_model(self) -> list[str]:
        inv: list[str] = []
        inv.extend(["key"] * int(getattr(self._model, "keys_collected", 0)))
        inv.extend(["extinguisher"] * int(getattr(self._model, "extinguishers_collected", 0)))
        return inv

    def _grid_from_model(self) -> list[list[int]]:
        # AIRIS game_map shape is [x][y]; ISL expects [row][col] => [y][x]
        width = int(len(self._model.game_map))
        height = int(len(self._model.game_map[0])) if width > 0 else 0
        grid: list[list[int]] = [[0 for _ in range(width)] for _ in range(height)]

        for x in range(width):
            for y in range(height):
                raw_id = int(getattr(self._model.game_map[x][y], "id", 0))
                mapped = self._ID_MAP.get(raw_id, 0)
                grid[y][x] = mapped

        ax, ay = self._model.character_current_pos
        if 0 <= ay < height and 0 <= ax < width:
            grid[ay][ax] = 1
        return grid

    def _agent_state_from_model(self) -> AgentState:
        ax, ay = self._model.character_current_pos
        return AgentState(row=int(ay), col=int(ax), inventory=self._inventory_from_model())

    def _entities_seen_from_target(
        self, action: str, prev_agent: AgentState, prev_grid: list[list[int]]
    ) -> list[int]:
        dr, dc = _ACTION_DELTAS.get(action, (0, 0))
        tr, tc = prev_agent.row + dr, prev_agent.col + dc
        if 0 <= tr < len(prev_grid) and 0 <= tc < len(prev_grid[0]):
            return [int(prev_grid[tr][tc])]
        if action != "nothing":
            return [2]
        return [int(prev_grid[prev_agent.row][prev_agent.col])]

    def reset(self) -> tuple[list[list[int]], AgentState]:
        self._set_level(self.level)
        self._done = False
        self._steps = 0

        grid = self._grid_from_model()
        agent_state = self._agent_state_from_model()
        self._grid = _clone_grid(grid)
        self._agent_state = agent_state.clone()
        return _clone_grid(self._grid), self._agent_state.clone()

    def step(self, action: str) -> tuple[list[list[int]], AgentState, bool, ActualOutcome]:
        action = action if action in _ACTION_DELTAS else "nothing"
        if self._done:
            outcome = ActualOutcome(
                action=action,
                new_state=self._agent_state.clone(),
                outcome_type="no_change",
                entities_seen=[],
            )
            return _clone_grid(self._grid), self._agent_state.clone(), True, outcome

        prev_grid = _clone_grid(self._grid)
        prev_agent = self._agent_state.clone()
        prev_keys = int(getattr(self._model, "keys_collected", 0))
        prev_ext = int(getattr(self._model, "extinguishers_collected", 0))
        prev_batteries = int(getattr(self._model, "batteries_collected", 0))

        with self._with_airis_cwd():
            self._model.game_logic(action)
            self._model.current_environment()

        self._steps += 1
        collected_battery = int(self._model.batteries_collected) > prev_batteries
        done_goal = int(self._model.batteries_collected) == int(self._model.num_batteries)
        if collected_battery:
            done_goal = True
        done_danger = bool(self._model.maze_reset)
        done_limit = self._steps >= self.max_steps
        done = done_goal or done_danger or done_limit

        grid = self._grid_from_model()
        agent_state = self._agent_state_from_model()

        entities_seen = self._entities_seen_from_target(action, prev_agent, prev_grid)
        if done_danger:
            entities_seen = [7]

        if done_danger:
            outcome_type = "danger"
        elif collected_battery:
            outcome_type = "goal_reached"
        elif int(self._model.keys_collected) > prev_keys or int(self._model.extinguishers_collected) > prev_ext:
            outcome_type = "item_collected"
        elif agent_state.row != prev_agent.row or agent_state.col != prev_agent.col:
            outcome_type = "moved"
        elif action == "nothing":
            outcome_type = "no_change"
        else:
            outcome_type = "blocked"

        if outcome_type not in _ALLOWED_OUTCOMES:
            outcome_type = _normalise_outcome_type(
                raw_outcome=outcome_type,
                action=action,
                prev_state=prev_agent,
                new_state=agent_state,
                done=done,
                entities_seen=entities_seen,
            )

        outcome = ActualOutcome(
            action=action,
            new_state=agent_state.clone(),
            outcome_type=outcome_type,
            entities_seen=entities_seen,
        )

        self._grid = _clone_grid(grid)
        self._agent_state = agent_state.clone()
        self._done = bool(done)
        return _clone_grid(self._grid), self._agent_state.clone(), self._done, outcome

    def __repr__(self) -> str:
        return f"AirisEnv(level={self.level}, backend=airis)"

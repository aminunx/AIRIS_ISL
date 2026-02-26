"""
isl_visualizer.py
=================
Real AIRIS comparison and watch utility.

Modes:
  python isl_visualizer.py watch
  python isl_visualizer.py compare
  python isl_visualizer.py results

This script uses:
  - ISLAgent + AirisEnv (real AIRIS backend)
  - A baseline policy without ISL memory/verification
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from airis_env import AirisEnv
from dream_engine import AgentState
from isl_agent import ISLAgent, StepRecord


ACTIONS = ("up", "down", "left", "right", "nothing")
RESULTS_PATH = Path("isl_real_comparison.json")


@dataclass
class BaselineEpisode:
    level: int
    episode: int
    steps: int
    success: bool
    blocked_steps: int
    danger_steps: int

    def to_dict(self) -> dict:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _win_steps(values: list[dict], step_key: str, win_key: str) -> float:
    wins = [float(v[step_key]) for v in values if bool(v[win_key])]
    return _mean(wins)


def render_grid(grid: list[list[int]]) -> str:
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
    return "\n".join(" ".join(symbols.get(cell, "?") for cell in row) for row in grid)


def _first_entity(grid: list[list[int]], entity_id: int) -> tuple[int, int] | None:
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == entity_id:
                return (r, c)
    return None


def _baseline_passable(cell: int, has_key: bool) -> bool:
    if cell == 2:
        return False
    if cell == 4:
        return has_key
    # No ISL safety memory: fire is not treated as blocked here.
    return True


def baseline_action(grid: list[list[int]], state: AgentState) -> str:
    target = _first_entity(grid, 3)
    if target is None:
        return "nothing"

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return "nothing"

    start = (state.row, state.col)
    has_key = "key" in state.inventory
    queue: deque[tuple[int, int, list[str]]] = deque([(start[0], start[1], [])])
    visited = {start}
    deltas = (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1))

    while queue:
        row, col, path = queue.popleft()
        if (row, col) == target:
            return path[0] if path else "nothing"

        for action, dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if (nr, nc) in visited:
                continue
            cell = int(grid[nr][nc])
            if not _baseline_passable(cell, has_key):
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, path + [action]))

    for action, dr, dc in deltas:
        nr, nc = state.row + dr, state.col + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
            continue
        if _baseline_passable(int(grid[nr][nc]), has_key):
            return action
    return "nothing"


def run_baseline_episode(level: int, episode: int, max_steps: int) -> BaselineEpisode:
    env = AirisEnv(level=level, max_steps=max_steps)
    grid, state = env.reset()
    done = False
    steps = 0
    blocked = 0
    danger = 0
    success = False

    while not done and steps < max_steps:
        action = baseline_action(grid, state)
        grid, state, done, actual = env.step(action)
        steps += 1

        if actual.outcome_type == "blocked":
            blocked += 1
        elif actual.outcome_type == "danger":
            danger += 1
        elif actual.outcome_type == "goal_reached":
            success = True

    return BaselineEpisode(
        level=level,
        episode=episode,
        steps=steps,
        success=success,
        blocked_steps=blocked,
        danger_steps=danger,
    )


def _has_goal(step_records: list[StepRecord]) -> bool:
    return any(r.actual_outcome == "goal_reached" for r in step_records)


def run_compare(levels: int, episodes: int, max_steps: int, output: Path) -> None:
    agent = ISLAgent(max_steps=max_steps, verbose=False)
    rows: list[dict] = []
    global_ep = 0

    print("\nISL vs Baseline (real AIRIS)")
    print(
        " Ep  Lvl | ISL_steps ISL_win ISL_acc ISL_mem% | "
        "Base_steps Base_win Base_block% Base_danger%"
    )
    print("-" * 93)

    for level in range(1, levels + 1):
        for ep in range(1, episodes + 1):
            global_ep += 1

            env_isl = AirisEnv(level=level, max_steps=max_steps)
            isl_summary = agent.run_episode(env_isl, level=level, max_steps=max_steps)
            isl_goal = _has_goal(isl_summary.step_records)
            isl_mem_rate = isl_summary.memory_hits / max(isl_summary.total_steps, 1)

            baseline = run_baseline_episode(level=level, episode=ep, max_steps=max_steps)
            base_block_rate = baseline.blocked_steps / max(baseline.steps, 1)
            base_danger_rate = baseline.danger_steps / max(baseline.steps, 1)

            row = {
                "episode_global": global_ep,
                "episode_in_level": ep,
                "level": level,
                "isl_steps": isl_summary.total_steps,
                "isl_win": isl_goal,
                "isl_prediction_accuracy": round(isl_summary.prediction_accuracy, 4),
                "isl_memory_hits": isl_summary.memory_hits,
                "isl_memory_hit_rate": round(isl_mem_rate, 4),
                "isl_repeated_mistakes": isl_summary.repeated_mistakes,
                "isl_avg_prediction_error": round(isl_summary.avg_prediction_error, 4),
                "base_steps": baseline.steps,
                "base_win": baseline.success,
                "base_blocked_steps": baseline.blocked_steps,
                "base_danger_steps": baseline.danger_steps,
                "base_blocked_rate": round(base_block_rate, 4),
                "base_danger_rate": round(base_danger_rate, 4),
            }
            rows.append(row)

            print(
                f"{global_ep:>3} {level:>4} | "
                f"{row['isl_steps']:>9} {str(row['isl_win']):>7} "
                f"{row['isl_prediction_accuracy']:.3f} {row['isl_memory_hit_rate']*100:>7.1f} | "
                f"{row['base_steps']:>10} {str(row['base_win']):>8} "
                f"{row['base_blocked_rate']*100:>11.1f} {row['base_danger_rate']*100:>11.1f}"
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "levels": levels,
            "episodes_per_level": episodes,
            "max_steps": max_steps,
            "environment": "AirisEnv(real AIRIS)",
        },
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    isl_wins = sum(1 for r in rows if r["isl_win"])
    base_wins = sum(1 for r in rows if r["base_win"])
    isl_acc_first10 = _mean([r["isl_prediction_accuracy"] for r in rows[:10]])
    isl_acc_last10 = _mean([r["isl_prediction_accuracy"] for r in rows[-10:]])
    isl_mem_first10 = _mean([r["isl_memory_hit_rate"] for r in rows[:10]])
    isl_mem_last10 = _mean([r["isl_memory_hit_rate"] for r in rows[-10:]])

    print("\nSummary")
    print(f"  Episodes: {len(rows)}")
    print(f"  ISL win rate:  {isl_wins}/{len(rows)} ({isl_wins/max(len(rows),1):.1%})")
    print(f"  Base win rate: {base_wins}/{len(rows)} ({base_wins/max(len(rows),1):.1%})")
    print(f"  ISL avg win steps:  {_win_steps(rows, 'isl_steps', 'isl_win'):.2f}")
    print(f"  Base avg win steps: {_win_steps(rows, 'base_steps', 'base_win'):.2f}")
    print(f"  ISL accuracy first10 -> last10: {isl_acc_first10:.3f} -> {isl_acc_last10:.3f}")
    print(f"  ISL memory hit first10 -> last10: {isl_mem_first10:.3f} -> {isl_mem_last10:.3f}")
    print(f"\nSaved results to: {output.resolve()}")


def _replay_episode(level: int, records: list[StepRecord], delay: float, max_steps: int) -> None:
    env = AirisEnv(level=level, max_steps=max_steps)
    grid, _state = env.reset()
    print("\nReplay start:")
    print(render_grid(grid))

    for idx, record in enumerate(records, start=1):
        grid, _state, done, actual = env.step(record.action)
        print(
            f"\nStep {idx:03d} | action={record.action:<7} "
            f"pred={record.predicted_outcome:<12} actual={actual.outcome_type:<12} "
            f"mem_hit={record.memory_hit}"
        )
        print(render_grid(grid))
        if delay > 0:
            time.sleep(delay)
        if done:
            break


def run_watch(level: int, episodes: int, max_steps: int, delay: float, replay_steps: int) -> None:
    agent = ISLAgent(max_steps=max_steps, verbose=False)
    print("\nWatch mode (real AIRIS)")
    print(f"Level={level} Episodes={episodes} MaxSteps={max_steps}")

    for ep in range(1, episodes + 1):
        env = AirisEnv(level=level, max_steps=max_steps)
        summary = agent.run_episode(env, level=level, max_steps=max_steps)
        isl_goal = _has_goal(summary.step_records)
        mem_rate = summary.memory_hits / max(summary.total_steps, 1)

        print(f"\nEpisode {ep}")
        print(f"  goal_reached: {isl_goal}")
        print(f"  steps: {summary.total_steps}")
        print(f"  prediction_accuracy: {summary.prediction_accuracy:.4f}")
        print(f"  avg_prediction_error: {summary.avg_prediction_error:.4f}")
        print(f"  memory_hits: {summary.memory_hits} ({mem_rate:.2%})")
        print(f"  repeated_mistakes: {summary.repeated_mistakes}")

        records = summary.step_records[:replay_steps]
        _replay_episode(level, records, delay=delay, max_steps=max_steps)


def run_results(path: Path) -> None:
    if not path.exists():
        print(f"No comparison file found at: {path.resolve()}")
        print("Run: python isl_visualizer.py compare")
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        print("Comparison file exists but has no rows.")
        return

    isl_wins = sum(1 for r in rows if r["isl_win"])
    base_wins = sum(1 for r in rows if r["base_win"])
    isl_acc = _mean([r["isl_prediction_accuracy"] for r in rows])
    isl_mem = _mean([r["isl_memory_hit_rate"] for r in rows])
    base_block = _mean([r["base_blocked_rate"] for r in rows])
    base_danger = _mean([r["base_danger_rate"] for r in rows])

    print("\nResults from saved real comparison")
    print(f"Generated: {payload.get('generated_at_utc', 'unknown')}")
    print(f"Episodes: {len(rows)}")
    print(f"ISL win rate:  {isl_wins}/{len(rows)} ({isl_wins/max(len(rows),1):.1%})")
    print(f"Base win rate: {base_wins}/{len(rows)} ({base_wins/max(len(rows),1):.1%})")
    print(f"ISL avg prediction accuracy: {isl_acc:.4f}")
    print(f"ISL avg memory hit rate: {isl_mem:.4f}")
    print(f"Base avg blocked rate: {base_block:.4f}")
    print(f"Base avg danger rate: {base_danger:.4f}")

    first10 = rows[:10]
    last10 = rows[-10:]
    print(
        f"ISL first10 accuracy -> last10 accuracy: "
        f"{_mean([r['isl_prediction_accuracy'] for r in first10]):.4f} -> "
        f"{_mean([r['isl_prediction_accuracy'] for r in last10]):.4f}"
    )
    print(
        f"ISL first10 mem_hit -> last10 mem_hit: "
        f"{_mean([r['isl_memory_hit_rate'] for r in first10]):.4f} -> "
        f"{_mean([r['isl_memory_hit_rate'] for r in last10]):.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real AIRIS ISL visualizer")
    sub = parser.add_subparsers(dest="mode", required=False)

    watch = sub.add_parser("watch", help="Run ISL on real AIRIS with replay")
    watch.add_argument("--level", type=int, default=1)
    watch.add_argument("--episodes", type=int, default=2)
    watch.add_argument("--max-steps", type=int, default=200)
    watch.add_argument("--delay", type=float, default=0.03)
    watch.add_argument("--replay-steps", type=int, default=60)

    compare = sub.add_parser("compare", help="Real AIRIS: ISL vs baseline")
    compare.add_argument("--levels", type=int, default=8)
    compare.add_argument("--episodes", type=int, default=5)
    compare.add_argument("--max-steps", type=int, default=500)
    compare.add_argument("--output", type=Path, default=RESULTS_PATH)

    results = sub.add_parser("results", help="Show saved comparison summary")
    results.add_argument("--input", type=Path, default=RESULTS_PATH)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mode = args.mode or "compare"

    if mode == "watch":
        run_watch(
            level=args.level,
            episodes=args.episodes,
            max_steps=args.max_steps,
            delay=args.delay,
            replay_steps=args.replay_steps,
        )
    elif mode == "compare":
        run_compare(
            levels=args.levels,
            episodes=args.episodes,
            max_steps=args.max_steps,
            output=args.output,
        )
    elif mode == "results":
        run_results(path=args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
benchmark.py
============
Run AIRIS-ISL benchmark:
- 8 levels
- 5 episodes per level
- 40 episodes total

Outputs:
- benchmark_results.json
- live summary per episode
"""

from __future__ import annotations

from pathlib import Path
import json
import time

from airis_env import AirisEnv
from isl_agent import ISLAgent


LEVELS = range(1, 9)
EPISODES_PER_LEVEL = 5
MAX_STEPS = 500
OUTPUT_PATH = Path("benchmark_results.json")


def evaluate_proof_targets(results: list[dict]) -> dict:
    if not results:
        return {"all_passed": False, "checks": {"results_exist": False}}

    ep1 = results[0]
    ep20_plus = results[19:] if len(results) >= 20 else []
    after_ep3 = results[3:] if len(results) >= 4 else []

    def _avg(rows: list[dict], key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

    def _success_rate(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        successes = sum(1 for row in rows if bool(row.get("success", False)))
        return successes / len(rows)

    checks = {
        "prediction_accuracy_ep1_lt_0_70": float(ep1.get("prediction_accuracy", 1.0)) < 0.70,
        "prediction_accuracy_ep20_plus_ge_0_90": _avg(ep20_plus, "prediction_accuracy") >= 0.90,
        "memory_hits_ep1_is_0": int(ep1.get("memory_hits", -1)) == 0,
        "memory_hit_rate_ep20_plus_ge_0_30": _avg(ep20_plus, "memory_hit_rate") >= 0.30,
        "repeated_mistakes_after_ep3_is_0": all(int(row.get("repeated_mistakes", 1)) == 0 for row in after_ep3),
        "avg_prediction_error_ep1_gt_0_30": float(ep1.get("avg_prediction_error", 0.0)) > 0.30,
        "avg_prediction_error_ep20_plus_lt_0_10": _avg(ep20_plus, "avg_prediction_error") < 0.10,
        "success_rate_ep20_plus_is_100pct": _success_rate(ep20_plus) == 1.0 if ep20_plus else False,
    }
    return {"all_passed": all(checks.values()), "checks": checks}


def run_benchmark() -> tuple[list[dict], dict]:
    agent = ISLAgent(max_steps=MAX_STEPS, verbose=False)
    results: list[dict] = []
    start_ts = time.time()
    episode_index = 0

    for level in LEVELS:
        for episode_in_level in range(1, EPISODES_PER_LEVEL + 1):
            episode_index += 1
            env = AirisEnv(level=level, max_steps=MAX_STEPS)
            summary = agent.run_episode(env, level=level, max_steps=MAX_STEPS)

            row = summary.to_dict()
            row["episode"] = episode_index
            row["episode_in_level"] = episode_in_level
            row["memory_hit_rate"] = round(summary.memory_hits / max(summary.total_steps, 1), 4)
            row["env_backend"] = "mock" if getattr(env, "using_mock", False) else "airis"
            results.append(row)

            print(
                f"Ep {episode_index:02d}/40 | L{level} E{episode_in_level} | "
                f"backend={row['env_backend']:<5} | steps={summary.total_steps:3d} | "
                f"success={summary.success} | acc={summary.prediction_accuracy:.1%} | "
                f"mem_hits={summary.memory_hits:3d} ({row['memory_hit_rate']:.0%}) | "
                f"repeat={summary.repeated_mistakes:2d}"
            )

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    elapsed = time.time() - start_ts
    print(f"\nSaved {len(results)} episodes to {OUTPUT_PATH.resolve()}")
    print(f"Elapsed: {elapsed:.2f}s")

    proof = evaluate_proof_targets(results)
    print("\nProof Targets:")
    for name, passed in proof["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  - {name}: {status}")
    print(f"Overall: {'PASS' if proof['all_passed'] else 'FAIL'}")

    return results, proof


if __name__ == "__main__":
    run_benchmark()


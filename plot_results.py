"""
plot_results.py
===============
Load benchmark_results.json and draw learning curves:
- prediction_accuracy over episodes
- memory_hits over episodes
"""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_PATH = Path("benchmark_results.json")
OUTPUT_PATH = Path("learning_curves.png")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH.resolve()}")

    results = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not results:
        raise ValueError("benchmark_results.json is empty.")

    episodes = [int(row.get("episode", idx + 1)) for idx, row in enumerate(results)]
    prediction_accuracy = [float(row.get("prediction_accuracy", 0.0)) for row in results]
    memory_hits = [int(row.get("memory_hits", 0)) for row in results]

    fig, ax_left = plt.subplots(figsize=(12, 6))

    line_acc = ax_left.plot(
        episodes,
        prediction_accuracy,
        color="#0f766e",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="prediction_accuracy",
    )[0]
    ax_left.set_xlabel("Episode")
    ax_left.set_ylabel("Prediction Accuracy", color="#0f766e")
    ax_left.set_ylim(0.0, 1.05)
    ax_left.tick_params(axis="y", labelcolor="#0f766e")
    ax_left.grid(True, linestyle="--", alpha=0.3)

    ax_right = ax_left.twinx()
    line_mem = ax_right.plot(
        episodes,
        memory_hits,
        color="#b45309",
        linewidth=2.0,
        marker="s",
        markersize=3.0,
        label="memory_hits",
    )[0]
    ax_right.set_ylabel("Memory Hits", color="#b45309")
    ax_right.tick_params(axis="y", labelcolor="#b45309")
    ax_right.set_ylim(0, max(memory_hits) * 1.1 if memory_hits else 1)

    fig.suptitle("AIRIS-ISL Learning Curves")
    fig.legend(
        handles=[line_acc, line_mem],
        labels=["prediction_accuracy", "memory_hits"],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()


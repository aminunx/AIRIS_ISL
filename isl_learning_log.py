"""
isl_learning_log.py
===================
Episode-level learning telemetry for AIRIS-ISL.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


class LearningTracker:
    """
    Measures real ISL learning progress.

    Metrics:
    - prediction_error per episode
    - concept confidence over time
    - goal reached rate
    - steps to goal
    - danger avoidance
    - dream/memory usage
    """

    def __init__(
        self,
        path: str = "isl_learning_log.json",
        discoveries_path: str = "isl_discoveries.json",
    ):
        self.path = path
        self.discoveries_path = discoveries_path
        self.episodes: list[dict] = []
        self.discoveries: list[dict] = []
        p = Path(self.path)
        if p.exists():
            try:
                self.episodes = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self.episodes = []
        d = Path(self.discoveries_path)
        if d.exists():
            try:
                self.discoveries = json.loads(d.read_text(encoding="utf-8"))
            except Exception:
                self.discoveries = []

    def log_episode(
        self,
        ep: int,
        level: int,
        steps: int,
        won: bool,
        avg_prediction_error: float,
        avg_confidence: float,
        danger_encountered: int,
        danger_avoided: int,
        dream_count: int,
        memory_hit_count: int,
        abstractions_updated: int,
        patterns_stored: int = 0,
    ) -> dict:
        record = {
            "episode": ep,
            "level": level,
            "steps": steps,
            "won": won,
            "avg_prediction_error": avg_prediction_error,
            "avg_confidence": avg_confidence,
            "danger_encountered": danger_encountered,
            "danger_avoided": danger_avoided,
            "dream_count": dream_count,
            "memory_hit_count": memory_hit_count,
            "abstractions_updated": abstractions_updated,
            "patterns_stored": patterns_stored,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "supabase_synced": False,
        }
        self.episodes.append(record)
        if patterns_stored > 0 or abstractions_updated > 0:
            self.discoveries.append(
                {
                    "episode": ep,
                    "level": level,
                    "patterns_stored": patterns_stored,
                    "abstractions_updated": abstractions_updated,
                    "memory_hit_count": memory_hit_count,
                    "timestamp": record["timestamp"],
                }
            )
        self._save()
        return record

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, indent=2)
        with open(self.discoveries_path, "w", encoding="utf-8") as f:
            json.dump(self.discoveries, f, indent=2)

    def print_summary(self):
        """Print learning curve summary."""
        print("\n  ISL LEARNING CURVE")
        print(
            f"  {'Ep':>4} {'Lvl':>4} {'Steps':>6} {'Win':>5} "
            f"{'PredErr':>8} {'Conf':>6} {'DangerAvoid':>12} {'Dreams':>7}"
        )
        print("  " + "-" * 65)
        for r in self.episodes:
            da = (
                f"{r['danger_avoided']}/{r['danger_encountered']}"
                if r["danger_encountered"] > 0
                else "N/A"
            )
            print(
                f"  {r['episode']:>4} {r['level']:>4} {r['steps']:>6} "
                f"{'WIN' if r['won'] else 'FAIL':>5} "
                f"{r['avg_prediction_error']:>8.4f} "
                f"{r['avg_confidence']:>6.4f} "
                f"{da:>12} {r['dream_count']:>7}"
            )

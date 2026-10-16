"""Incremental prediction persistence and resume.

One ``predictions.jsonl`` per bench doubles as the result log and the resume
checkpoint: on restart, already-recorded ``uid``s are skipped. ``summary.json``
is written on finalize and enriched with aggregator metrics + failure counts.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..types import Prediction, TrajectoryEvent


class ResumeState:
    """Append-only JSONL writer keyed by uid, with resume + summary."""

    def __init__(self, out_dir: str | Path, flush_every: int = 20) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.out_dir / "predictions.jsonl"
        self.summary_path = self.out_dir / "summary.json"
        self.trajectory_dir = self.out_dir / "artifacts" / "trajectories"
        self._flush_every = flush_every
        self._started_at = time.monotonic()
        self.records: list[Prediction] = []
        self._pending: list[Prediction] = []
        self.done_uids: set[str] = set()

        if self.predictions_path.exists():
            with self.predictions_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    pred = Prediction.from_row(row)
                    if pred.uid in self.done_uids:
                        raise ValueError(f"duplicate uid {pred.uid!r} in {self.predictions_path}")
                    self.records.append(pred)
                    self.done_uids.add(pred.uid)

    def is_done(self, uid: str) -> bool:
        return str(uid) in self.done_uids

    def record(self, pred: Prediction) -> None:
        if pred.uid in self.done_uids or any(item.uid == pred.uid for item in self._pending):
            raise ValueError(f"prediction uid {pred.uid!r} was already recorded")
        self._pending.append(pred)
        self.records.append(pred)

    def write_trajectory(self, uid: str, events: tuple[TrajectoryEvent, ...]) -> str | None:
        """Atomically persist a non-empty agent trajectory and return its relative path."""
        if not events:
            return None
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(uid).encode("utf-8")).hexdigest()[:24]
        path = self.trajectory_dir / f"{digest}.json"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {"uid": str(uid), "events": [event.to_dict() for event in events]}
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return str(path.relative_to(self.out_dir))

    def should_flush(self) -> bool:
        return len(self._pending) >= self._flush_every

    def flush(self) -> None:
        if not self._pending:
            return
        with self.predictions_path.open("a", encoding="utf-8") as handle:
            for pred in self._pending:
                handle.write(json.dumps(pred.to_row(), ensure_ascii=False) + "\n")
                self.done_uids.add(pred.uid)
            handle.flush()
        self._pending.clear()

    def write_summary(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """Flush, compute base counts, merge aggregator ``metrics``, persist."""
        self.flush()
        n_total = len(self.records)
        n_failed = sum(1 for r in self.records if r.status in ("failed", "cancelled"))
        n_unscored = sum(1 for r in self.records if r.score is None and r.status not in ("failed", "cancelled"))
        summary: dict[str, Any] = {
            "n_total": n_total,
            "n_failed": n_failed,
            "n_unscored": n_unscored,
            "wall_seconds": round(time.monotonic() - self._started_at, 3),
        }
        if metrics:
            summary["metrics"] = metrics
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


__all__ = ["ResumeState"]

"""Convert JointAVBench with the official deterministic option shuffle."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .base import (
    BaseConverter,
    make_mcq_row,
    normalize_answer_letter,
    require_options,
    require_str,
)

_DEPRECATED = {
    "TKGeXyYiLBo_task1_0",
    "z9ncrJM4gd4_task1_0",
    "STU9x7OxUy4_task2_4",
    "T21p-RHFT3U_task2_1",
    "STU9x7OxUy4_task4_4",
    "TKGeXyYiLBo_task4_2",
    "StCB7h6PjFw_task5_1",
    "T21p-RHFT3U_task5_0",
    "Rfos56t9HB4_task9_2",
}
_SCENE_TASKS = {
    "Single": {"STL", "VSSR", "SPL", "SOOG", "SOER", "SPER", "MPTI"},
    "Multiple": {"CSA", "MPO", "PDP", "AFA", "PTG"},
    "Full": {"AVDM", "MESI", "CRI"},
}
_COGNITIVE_TASKS = {
    "Temporal": {"STL", "VSSR", "PTG"},
    "Spatial": {"SPL", "SOOG", "SOER"},
    "Emotion": {"SPER", "MPTI", "MESI"},
    "Long-form": {"CSA", "AVDM"},
    "Plot": {"MPO", "PDP", "AFA", "CRI"},
}


class JointAVBenchConverter(BaseConverter):
    name = "jointavbench_full"
    source_relative = Path("benchmarks/JointAVBench/jointavbench.json")
    output_relative = Path("annotations/jointavbench_full.json")
    video_dir = "benchmarks/JointAVBench"
    validation_unit_label = "clips"
    scoring_protocol = "full_mc_accuracy_shuffle42_exclude9_deprecated"
    report: ClassVar[Mapping[str, Any]] = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "scene_type", "key": "facets.scene_type"},
            {"name": "cognitive_type", "key": "facets.cognitive_type"},
        ]
    }

    def iter_source_rows(self, raw: Any) -> list[Any]:
        rows = super().iter_source_rows(raw)
        rng = random.Random(42)
        converted: list[dict[str, Any]] = []
        for index, raw_row in enumerate(rows, 1):
            if not isinstance(raw_row, Mapping):
                raise TypeError(f"JointAVBench row {index} must be an object")
            row = dict(raw_row)
            options = require_options(row, index, dataset="JointAVBench")
            correct = require_str(row, "correct_answer", index, dataset="JointAVBench")
            if correct not in options:
                raise ValueError(f"JointAVBench row {index} correct_answer is not an option")
            indexed = list(enumerate(options))
            rng.shuffle(indexed)
            row["_options"] = [f"{chr(65 + pos)}. {text}" for pos, (_, text) in enumerate(indexed)]
            row["_answer"] = chr(65 + next(pos for pos, (old, _) in enumerate(indexed) if options[old] == correct))
            timestamps = row.get("segment_timestamp")
            if not isinstance(timestamps, list) or len(timestamps) != 2:
                raise ValueError(f"JointAVBench row {index} has invalid segment_timestamp")
            row["_duration"] = float(timestamps[1]) - float(timestamps[0])
            if row.get("qid") not in _DEPRECATED:
                converted.append(row)
        return converted

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError(f"JointAVBench row {index} must be an object")
        task = require_str(row, "task", index, dataset="JointAVBench")
        facets = {
            "task": task,
            "scene_type": _taxonomy_value(task, _SCENE_TASKS),
            "cognitive_type": _taxonomy_value(task, _COGNITIVE_TASKS),
            "duration": row.get("_duration"),
        }
        qid = require_str(row, "qid", index, dataset="JointAVBench")
        options = require_options(row, index, key="_options", dataset="JointAVBench")
        answer = normalize_answer_letter(row.get("_answer"), len(options), index=index, dataset="JointAVBench")
        return make_mcq_row(
            problem_id=qid,
            problem=require_str(row, "question", index, dataset="JointAVBench"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./videos/{qid}.mp4",
            facets=facets,
            extra={
                "video_id": qid,
                "source_video_id": row.get("video_name"),
                "segment_timestamp": row.get("segment_timestamp"),
                "video_url": row.get("video_url"),
                "explanation": row.get("explanation"),
            },
        )

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        super().validate_row(index, row)
        if row.get("data_type") != "video" or not str(row.get("path", "")).startswith("./videos/"):
            raise ValueError(f"annotation row {index} must point to a video under ./videos")

    def registry_entry(self) -> dict[str, Any]:
        entry = super().registry_entry()
        entry["source_counts"] = {"raw_questions": 2853, "evaluation_questions": 2844}
        return entry


def _taxonomy_value(task: str, taxonomy: Mapping[str, set[str]]) -> str | None:
    return next((name for name, tasks in taxonomy.items() if task in tasks), None)


CONVERTER = JointAVBenchConverter()

__all__ = ["CONVERTER", "JointAVBenchConverter"]

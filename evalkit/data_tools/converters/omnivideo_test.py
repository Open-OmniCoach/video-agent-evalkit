"""Convert OmniVideo-Test metadata into evalkit annotations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseConverter, dot_relative, make_mcq_row, normalize_answer_letter, require_options, require_str


class OmniVideoTestConverter(BaseConverter):
    name = "omnivideo_test_full"
    source_relative = Path("benchmarks/OmniVideo-Test/test_505.jsonl")
    output_relative = Path("annotations/omnivideo_test_full.json")
    video_dir = "benchmarks/OmniVideo-Test"
    source_format = "jsonl"
    validation_unit_label = "source videos"
    scoring_protocol = "full_mc_accuracy"
    facet_map: ClassVar[Mapping[str, str]] = {
        "task": "task",
        "subtask": "subtask",
        "language": "language",
    }
    report: ClassVar[Mapping[str, Any]] = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "subtask", "key": "facets.subtask"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError(f"OmniVideo-Test row {index} must be an object")
        options = require_options(row, index, dataset="OmniVideo-Test")
        answer = normalize_answer_letter(
            require_str(row, "answer", index, dataset="OmniVideo-Test"),
            len(options),
            index=index,
            dataset="OmniVideo-Test",
        )
        question_id = require_str(row, "question_id", index, dataset="OmniVideo-Test")
        video_path = require_str(row, "video_path", index, dataset="OmniVideo-Test")
        return make_mcq_row(
            problem_id=question_id,
            problem=require_str(row, "question", index, dataset="OmniVideo-Test"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=dot_relative(video_path),
            facets={**self.build_facets(row), "duration": _clip_duration(row)},
            extra={
                # ``video_id`` identifies the source video, while one source
                # may produce multiple task-specific clip files.  Cache
                # identity must therefore follow the concrete clip path.
                "media_id": f"omnivideo_test_{Path(video_path).stem}",
                "video_id": row.get("video_id"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "source_duration": row.get("duration"),
                "resolution": row.get("resolution"),
                "search_tag": row.get("search_tag"),
            },
        )

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        super().validate_row(index, row)
        if row.get("data_type") != "video" or not str(row.get("path", "")).startswith("./videos/"):
            raise ValueError(f"annotation row {index} must point to a video under ./videos")

    def registry_entry(self) -> dict[str, Any]:
        entry = super().registry_entry()
        entry["source_counts"] = {"questions": 505, "source_videos": 264}
        return entry


def _clip_duration(row: Mapping[str, Any]) -> float:
    return float(row.get("end_time", 0)) - float(row.get("start_time", 0))


CONVERTER = OmniVideoTestConverter()

__all__ = ["CONVERTER", "OmniVideoTestConverter"]

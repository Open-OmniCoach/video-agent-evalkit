"""Convert raw LVOmniBench metadata into evalkit annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import BaseConverter, make_mcq_row, normalize_answer_letter, require_options, require_str


class LVOmniBenchConverter(BaseConverter):
    name = "lvomnibench"
    source_relative = Path("benchmarks/LVOmniBench/data.json")
    output_relative = Path("annotations/lvomnibench.json")
    video_dir = "benchmarks/LVOmniBench"
    validation_unit_label = "videos"
    scoring_protocol = "official_mc_accuracy"
    facet_map = {
        "question_type": "task",
        "audio_type": "audio_type",
        "video_category": "video_category",
        "sub_category": "sub_category",
        "difficulty": "difficulty",
        "duration": "duration",
    }
    report = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "difficulty", "key": "facets.difficulty"},
            {"name": "audio_type", "key": "facets.audio_type"},
            {"name": "video_category", "key": "facets.video_category"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise ValueError(f"LVOmniBench row {index} must be an object")

        options = require_options(row, index, dataset="LVOmniBench")
        answer = normalize_answer_letter(
            require_str(row, "correct_option", index, dataset="LVOmniBench"),
            len(options),
            index=index,
            dataset="LVOmniBench",
        )
        video_id = require_str(row, "video_id", index, dataset="LVOmniBench")
        extra = {
            "video_id": video_id,
            "video_link": row.get("video_link"),
            "original_question_id": row.get("question_id"),
            "original_question_type": row.get("question_type"),
            "duration": row.get("duration"),
            "video_category": row.get("video_category"),
            "sub_category": row.get("sub_category"),
            "audio_type": row.get("audio_type"),
            "difficulty": row.get("difficulty"),
            "answer": row.get("answer"),
        }

        return make_mcq_row(
            problem_id=index,
            problem=require_str(row, "question", index, dataset="LVOmniBench"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./videos/{video_id}.mp4",
            facets=self.build_facets(row),
            extra={k: v for k, v in extra.items() if v is not None},
        )

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        super().validate_row(index, row)
        if row.get("data_type") != "video":
            raise ValueError(f"annotation row {index} must be video")
        if not str(row.get("path", "")).startswith("./videos/"):
            raise ValueError(f"annotation row {index} path must point under ./videos")


CONVERTER = LVOmniBenchConverter()

__all__ = ["CONVERTER", "LVOmniBenchConverter"]

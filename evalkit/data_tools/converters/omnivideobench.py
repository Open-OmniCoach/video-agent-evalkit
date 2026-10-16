"""Convert raw OmniVideoBench metadata into evalkit annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import BaseConverter, dot_relative, make_mcq_row, normalize_answer_letter, require_options, require_str


class OmniVideoBenchConverter(BaseConverter):
    name = "omnivideobench"
    source_relative = Path("benchmarks/OmniVideoBench/data.parquet")
    output_relative = Path("annotations/omnivideobench.json")
    video_dir = "benchmarks/OmniVideoBench"
    source_format = "parquet"
    validation_unit_label = "videos"
    scoring_protocol = "official_mc_accuracy"
    facet_map = {
        "question_type": "task",
        "audio_type": "audio_type",
        "video_type": "video_type",
        "duration": "duration",
    }
    report = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "audio_type", "key": "facets.audio_type"},
            {"name": "video_type", "key": "facets.video_type"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise ValueError(f"OmniVideoBench row {index} must be an object")

        options = require_options(row, index, dataset="OmniVideoBench")
        answer = normalize_answer_letter(
            require_str(row, "correct_option", index, dataset="OmniVideoBench"),
            len(options),
            index=index,
            dataset="OmniVideoBench",
        )
        video_path = require_str(row, "video", index, dataset="OmniVideoBench")
        video_id = Path(video_path).stem

        extra: dict[str, Any] = {
            "video_id": video_id,
            "original_question_type": row.get("question_type"),
            "audio_type": row.get("audio_type"),
            "video_type": row.get("video_type"),
            "duration": row.get("duration"),
            "answer": row.get("answer"),
        }
        if row.get("reasoning_steps") is not None:
            extra["reasoning_steps"] = row.get("reasoning_steps")

        return make_mcq_row(
            problem_id=index,
            problem=require_str(row, "question", index, dataset="OmniVideoBench"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=dot_relative(video_path),
            facets=self.build_facets(row),
            extra={k: v for k, v in extra.items() if v is not None},
        )

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        super().validate_row(index, row)
        if row.get("data_type") != "video":
            raise ValueError(f"annotation row {index} must be video")
        if not str(row.get("path", "")).startswith("./videos/"):
            raise ValueError(f"annotation row {index} path must point under ./videos")


CONVERTER = OmniVideoBenchConverter()

__all__ = ["CONVERTER", "OmniVideoBenchConverter"]

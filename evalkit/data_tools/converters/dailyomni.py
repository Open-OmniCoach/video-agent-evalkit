"""Convert Daily-Omni QA metadata into evalkit annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import BaseConverter, make_mcq_row, normalize_answer_letter, require_options, require_str


class DailyOmniConverter(BaseConverter):
    name = "dailyomni"
    source_relative = Path("benchmarks/Daily-Omni/qa.json")
    output_relative = Path("annotations/dailyomni.json")
    video_dir = "benchmarks/Daily-Omni/Videos"
    validation_unit_label = "videos"
    scoring_protocol = "non_official_simple_mc_accuracy"
    facet_map = {
        "Type": "task",
        "video_duration": "duration",
        "content_parent_category": "domain",
        "content_fine_category": "subdomain",
        "video_category": "video_category",
    }
    report = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "duration", "key": "facets.duration"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise ValueError(f"Daily-Omni row {index} must be an object")

        question = require_str(row, "Question", index, dataset="Daily-Omni")
        options = require_options(row, index, key="Choice", dataset="Daily-Omni")

        answer = normalize_answer_letter(
            require_str(row, "Answer", index, dataset="Daily-Omni"),
            len(options),
            index=index,
            dataset="Daily-Omni",
        )
        video_id = require_str(row, "video_id", index, dataset="Daily-Omni")
        extra: dict[str, Any] = {
            "video_id": video_id,
            "audio_path": f"./{video_id}/{video_id}_audio.wav",
        }
        passthrough = {
            "Type": "original_question_type",
            "content_parent_category": "content_parent_category",
            "content_fine_category": "content_fine_category",
            "video_category": "video_category",
            "video_duration": "video_duration",
            "Explaination": "explanation",
        }
        for raw_key, out_key in passthrough.items():
            value = row.get(raw_key)
            if value is not None:
                extra[out_key] = value

        return make_mcq_row(
            problem_id=index,
            problem=question,
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./{video_id}/{video_id}_video.mp4",
            facets=self.build_facets(row),
            extra=extra,
        )

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        super().validate_row(index, row)
        required = {"data_type", "path", "video_id", "audio_path"}
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"annotation row {index} missing keys: {missing}")
        if row["problem_type"] != "multiple choice":
            raise ValueError(f"annotation row {index} must be multiple choice")
        if row["data_type"] != "video":
            raise ValueError(f"annotation row {index} must be video")
        if row["data_source"] != self.name:
            raise ValueError(f"annotation row {index} has wrong data_source")
        if not str(row["path"]).endswith("_video.mp4"):
            raise ValueError(f"annotation row {index} path must point to an mp4")
        if not str(row["audio_path"]).endswith("_audio.wav"):
            raise ValueError(f"annotation row {index} audio_path must point to a wav")


CONVERTER = DailyOmniConverter()

__all__ = ["CONVERTER", "DailyOmniConverter"]

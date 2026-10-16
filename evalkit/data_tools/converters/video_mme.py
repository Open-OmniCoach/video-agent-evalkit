"""Convert the full Video-MME test split."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseConverter, make_mcq_row, normalize_answer_letter, require_options, require_str


class VideoMMEConverter(BaseConverter):
    name = "video_mme_full"
    source_relative = Path("benchmarks/Video-MME/videomme/test-00000-of-00001.parquet")
    output_relative = Path("annotations/video_mme_full.json")
    video_dir = "benchmarks/Video-MME"
    source_format = "parquet"
    validation_unit_label = "videos"
    scoring_protocol = "full_mc_accuracy_no_external_subtitles"
    facet_map: ClassVar[Mapping[str, str]] = {
        "duration": "duration",
        "domain": "domain",
        "sub_category": "sub_category",
        "task_type": "task",
    }
    report: ClassVar[Mapping[str, Any]] = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "domain", "key": "facets.domain"},
            {"name": "sub_category", "key": "facets.sub_category"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError(f"Video-MME row {index} must be an object")
        options = require_options(row, index, dataset="Video-MME")
        answer = normalize_answer_letter(row.get("answer"), len(options), index=index, dataset="Video-MME")
        video_id = require_str(row, "videoID", index, dataset="Video-MME")
        return make_mcq_row(
            problem_id=require_str(row, "question_id", index, dataset="Video-MME"),
            problem=require_str(row, "question", index, dataset="Video-MME"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./video/{video_id}.mp4",
            facets=self.build_facets(row),
            extra={"video_id": video_id, "source_video_id": row.get("video_id"), "url": row.get("url")},
        )

    def registry_entry(self) -> dict[str, Any]:
        entry = super().registry_entry()
        entry["source_counts"] = {"questions": 2700, "videos": 900}
        return entry


CONVERTER = VideoMMEConverter()

__all__ = ["CONVERTER", "VideoMMEConverter"]

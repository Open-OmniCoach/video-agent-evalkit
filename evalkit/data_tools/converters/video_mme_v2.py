"""Convert the full Video-MME-v2 test split."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseConverter, make_mcq_row, normalize_answer_letter, require_str


class VideoMMEV2Converter(BaseConverter):
    name = "video_mme_v2_full"
    source_relative = Path("benchmarks/Video-MME-v2/test.parquet")
    output_relative = Path("annotations/video_mme_v2_full.json")
    video_dir = "benchmarks/Video-MME-v2"
    source_format = "parquet"
    validation_unit_label = "videos"
    scoring_protocol = "full_official_grouped_rating_no_external_subtitles"
    report: ClassVar[Mapping[str, Any]] = {
        "breakdowns": [
            {"name": "group_type", "key": "facets.group_type"},
            {"name": "level", "key": "facets.level"},
            {"name": "second_head", "key": "facets.second_head"},
            {"name": "third_head", "key": "facets.third_head"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError(f"Video-MME-v2 row {index} must be an object")
        options = _parse_options(row.get("options"), index)
        answer = normalize_answer_letter(row.get("answer"), len(options), index=index, dataset="Video-MME-v2")
        question_id = require_str(row, "question_id", index, dataset="Video-MME-v2")
        video_id = require_str(row, "video_id", index, dataset="Video-MME-v2")
        group_position = _group_position(question_id, index)
        facets = {
            "group_id": video_id,
            "group_position": group_position,
            "group_type": row.get("group_type"),
            "group_structure": row.get("group_structure"),
            "level": row.get("level"),
            "second_head": row.get("second_head"),
            "third_head": row.get("third_head"),
        }
        return make_mcq_row(
            problem_id=question_id,
            problem=require_str(row, "question", index, dataset="Video-MME-v2"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./videos/{video_id}.mp4",
            facets=facets,
            extra={"video_id": video_id, "url": row.get("url")},
        )

    def registry_entry(self) -> dict[str, Any]:
        entry = super().registry_entry()
        entry["source_counts"] = {"questions": 3200, "videos": 800}
        return entry


def _parse_options(value: Any, index: int) -> list[str]:
    if not isinstance(value, str):
        raise TypeError(f"Video-MME-v2 row {index} has invalid options")
    options = [line.strip() for line in value.splitlines() if line.strip()]
    if not options:
        raise ValueError(f"Video-MME-v2 row {index} has empty options")
    return options


def _group_position(question_id: str, index: int) -> int:
    try:
        position = int(question_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Video-MME-v2 row {index} has invalid question_id {question_id!r}") from exc
    if position not in {1, 2, 3, 4}:
        raise ValueError(f"Video-MME-v2 row {index} group position must be 1..4")
    return position


CONVERTER = VideoMMEV2Converter()

__all__ = ["CONVERTER", "VideoMMEV2Converter"]

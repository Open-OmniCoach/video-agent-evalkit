"""Convert raw WorldSense metadata into evalkit annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .base import BaseConverter, make_mcq_row, normalize_answer_letter, require_options, require_str


class WorldSenseConverter(BaseConverter):
    name = "worldsense"
    source_relative = Path("benchmarks/WorldSense/worldsense_qa.json")
    output_relative = Path("annotations/worldsense.json")
    video_dir = "benchmarks/WorldSense"
    validation_unit_label = "videos"
    scoring_protocol = "official_mc_accuracy"
    report = {
        "breakdowns": [
            {"name": "task", "key": "facets.task"},
            {"name": "task_domain", "key": "facets.task_domain"},
            {"name": "domain", "key": "facets.domain"},
            {"name": "sub_category", "key": "facets.sub_category"},
            {"name": "duration", "key": "facets.duration"},
        ]
    }

    def iter_source_rows(self, raw: Any) -> list[Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("WorldSense source must be a mapping")
        rows: list[dict[str, Any]] = []
        for video_key, video in raw.items():
            if not isinstance(video, Mapping):
                raise ValueError(f"WorldSense video {video_key!r} must be an object")
            for task_key in sorted(
                (key for key in video if str(key).startswith("task")),
                key=_task_sort_key,
            ):
                task = video.get(task_key)
                if not isinstance(task, Mapping):
                    continue
                rows.append({"video": video, "task_key": str(task_key), "task": task})
        return rows

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise ValueError(f"WorldSense row {index} must be an object")
        video = row.get("video")
        task = row.get("task")
        if not isinstance(video, Mapping) or not isinstance(task, Mapping):
            raise ValueError(f"WorldSense row {index} missing video/task")

        options = require_options(task, index, key="candidates", dataset="WorldSense")

        answer = normalize_answer_letter(
            require_str(task, "answer", index, dataset="WorldSense"),
            len(options),
            index=index,
            dataset="WorldSense",
        )
        video_id = require_str(video, "video_id", index, dataset="WorldSense")
        facets = {
            "task": task.get("task_type"),
            "task_domain": task.get("task_domain"),
            "domain": video.get("domain"),
            "sub_category": video.get("sub_category"),
            "duration": video.get("duration"),
            "video_duration": video.get("video_duration"),
            "audio_class": video.get("audio_class"),
        }
        extra = {
            "video_id": video_id,
            "task_id": row.get("task_key"),
            "task_domain": task.get("task_domain"),
            "original_question_type": task.get("task_type"),
            "video_duration": video.get("video_duration"),
            "duration": video.get("duration"),
            "domain": video.get("domain"),
            "sub_category": video.get("sub_category"),
            "audio_class": video.get("audio_class"),
            "video_caption": video.get("video_caption"),
        }

        return make_mcq_row(
            problem_id=index,
            problem=require_str(task, "question", index, dataset="WorldSense"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./{video_id}.mp4",
            facets={k: v for k, v in facets.items() if v is not None and str(v).strip() != ""},
            extra={k: v for k, v in extra.items() if v is not None},
        )

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        super().validate_row(index, row)
        if row.get("data_type") != "video":
            raise ValueError(f"annotation row {index} must be video")
        if not str(row.get("path", "")).endswith(".mp4"):
            raise ValueError(f"annotation row {index} path must point to an mp4")


def _task_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    suffix = text.removeprefix("task")
    try:
        return int(suffix), text
    except ValueError:
        return 10_000, text


CONVERTER = WorldSenseConverter()

__all__ = ["CONVERTER", "WorldSenseConverter"]

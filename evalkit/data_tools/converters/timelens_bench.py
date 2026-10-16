"""Convert raw TimeLens-Bench metadata into evalkit annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .base import BaseConverter, validate_standard_row

_SOURCES = ("activitynet", "charades", "qvhighlights")


class TimeLensBenchConverter(BaseConverter):
    name = "timelens_bench"
    source_relative = Path("benchmarks/TimeLens-Bench")
    output_relative = Path("annotations/timelens_bench.json")
    video_dir = "benchmarks/TimeLens-Bench"
    source_format = "json_directory"
    validation_unit_label = "queries"
    scoring_protocol = "temporal_grounding_miou_r1"
    report = {
        "breakdowns": [
            {"name": "source_dataset", "key": "facets.source_dataset"},
        ]
    }

    def convert(self, source_path: Path, output_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source_dataset in _SOURCES:
            raw = _load_source_file(source_path, source_dataset)
            rows.extend(self._convert_source(source_path, source_dataset, raw, len(rows)))

        for index, row in enumerate(rows, 1):
            self.validate_row(index, row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_annotations(output_path, rows)
        return rows

    def _convert_source(
        self,
        source_path: Path,
        source_dataset: str,
        raw: Mapping[str, Any],
        offset: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for video_id, item in raw.items():
            if not isinstance(item, Mapping):
                raise ValueError(f"TimeLens {source_dataset} video {video_id!r} must be an object")
            queries = item.get("queries")
            spans = item.get("spans")
            if not isinstance(queries, list) or not isinstance(spans, list):
                raise ValueError(f"TimeLens {source_dataset} video {video_id!r} missing queries/spans")
            if len(queries) != len(spans):
                raise ValueError(f"TimeLens {source_dataset} video {video_id!r} has mismatched queries/spans")

            media_path = Path("videos") / source_dataset / f"{video_id}.mp4"
            if not (source_path / media_path).exists():
                continue

            for query_index, (query, span) in enumerate(zip(queries, spans)):
                start, end = _span_pair(span, source_dataset, video_id, query_index)
                rows.append(
                    {
                        "problem_id": offset + len(rows) + 1,
                        "problem": str(query).strip(),
                        "problem_type": "temporal grounding",
                        "options": [],
                        "solution": f'<answer>{{"time": [{start}, {end}], "boxes": {{}}}}</answer>',
                        "data_source": self.name,
                        "data_type": "video",
                        "path": f"./{media_path.as_posix()}",
                        "video_id": str(video_id),
                        "source_dataset": source_dataset,
                        "query_index": query_index,
                        "duration": item.get("duration"),
                        "facets": {
                            "source_dataset": source_dataset,
                            "duration": item.get("duration"),
                        },
                    }
                )
        return rows

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        validate_standard_row(index, row)
        if row.get("problem_type") != "temporal grounding":
            raise ValueError(f"annotation row {index} must be temporal grounding")
        if row.get("data_type") != "video":
            raise ValueError(f"annotation row {index} must be video")
        if not str(row.get("path", "")).startswith("./videos/"):
            raise ValueError(f"annotation row {index} path must point under ./videos")

    def unit_key(self, row: Mapping[str, Any]) -> Any:
        return row.get("problem_id")


def _load_source_file(source_path: Path, source_dataset: str) -> Mapping[str, Any]:
    path = source_path / f"{source_dataset}-timelens.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"TimeLens source must be a mapping: {path}")
    return raw


def _span_pair(span: Any, source_dataset: str, video_id: str, query_index: int) -> tuple[float, float]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError(f"TimeLens {source_dataset} video {video_id!r} query {query_index} has invalid span")
    try:
        start = float(span[0])
        end = float(span[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TimeLens {source_dataset} video {video_id!r} query {query_index} has invalid span") from exc
    if end < start:
        start, end = end, start
    return start, end


CONVERTER = TimeLensBenchConverter()

__all__ = ["CONVERTER", "TimeLensBenchConverter"]

"""Convert the full FutureOmni test split."""

from __future__ import annotations

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


class FutureOmniConverter(BaseConverter):
    name = "futureomni_full"
    source_relative = Path("benchmarks/FutureOmni/futureomni_test.json")
    output_relative = Path("annotations/futureomni_full.json")
    video_dir = "benchmarks/FutureOmni"
    validation_unit_label = "videos"
    scoring_protocol = "full_mc_accuracy"
    facet_map: ClassVar[Mapping[str, str]] = {
        "video_domain": "domain",
        "audio_type": "audio_type",
        "forecasting_pattern": "forecasting_pattern",
    }
    report: ClassVar[Mapping[str, Any]] = {
        "breakdowns": [
            {"name": "domain", "key": "facets.domain"},
            {"name": "audio_type", "key": "facets.audio_type"},
            {"name": "forecasting_pattern", "key": "facets.forecasting_pattern"},
            {"name": "duration", "key": "facets.duration"},
        ]
    }

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        if not isinstance(row, Mapping):
            raise TypeError(f"FutureOmni row {index} must be an object")
        options = require_options(row, index, dataset="FutureOmni")
        answer = normalize_answer_letter(row.get("answer"), len(options), index=index, dataset="FutureOmni")
        qid = row.get("qid")
        if qid is None:
            raise ValueError(f"FutureOmni row {index} missing qid")
        facets = self.build_facets(row)
        facets["duration"] = _duration_bucket(float(row.get("seconds", 0)))
        return make_mcq_row(
            problem_id=qid,
            problem=require_str(row, "question", index, dataset="FutureOmni"),
            options=options,
            answer=answer,
            data_source=self.name,
            data_type="video",
            path=f"./videos/{qid}.mp4",
            facets=facets,
            extra={"video_id": str(qid), "original_video": row.get("video"), "seconds": row.get("seconds")},
        )

    def registry_entry(self) -> dict[str, Any]:
        entry = super().registry_entry()
        entry["source_counts"] = {"questions": 1034}
        return entry


def _duration_bucket(seconds: float) -> str:
    if seconds <= 60.0:
        return "(0,60]s"
    if seconds <= 300.0:
        return "(60,300]s"
    return "(300,inf)s"


CONVERTER = FutureOmniConverter()

__all__ = ["CONVERTER", "FutureOmniConverter"]

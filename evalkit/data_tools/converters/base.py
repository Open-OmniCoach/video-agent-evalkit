"""Shared converter primitives for benchmark data preparation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


class BaseConverter:
    """Template-method base class for raw benchmark -> evalkit annotations."""

    name: str
    source_relative: Path
    output_relative: Path
    video_dir: str
    image_dir: str | None = None
    source_format: str = "json"
    annotation_format: str = "json"
    scoring_protocol: str = ""
    validation_unit_label: str = "items"
    facet_map: Mapping[str, str] = {}
    report: Mapping[str, Any] = {}

    def load_source(self, source_path: Path) -> Any:
        if self.source_format == "json":
            return json.loads(source_path.read_text(encoding="utf-8"))
        if self.source_format == "jsonl":
            rows: list[Any] = []
            with source_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if self.source_format == "parquet":
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("parquet conversion requires pandas and a parquet engine") from exc
            frame = pd.read_parquet(source_path)
            return [parquet_value(row) for row in frame.to_dict(orient="records")]
        raise ValueError(f"unsupported source_format {self.source_format!r} for {self.name}")

    def iter_source_rows(self, raw: Any) -> list[Any]:
        if not isinstance(raw, list):
            raise ValueError(f"{self.name} source must be a list")
        return raw

    def convert_row(self, index: int, row: Any) -> dict[str, Any]:
        raise NotImplementedError

    def validate_row(self, index: int, row: Mapping[str, Any]) -> None:
        validate_standard_row(index, row)

    def unit_key(self, row: Mapping[str, Any]) -> Any:
        return row.get("video_id") or row.get("image_id") or row.get("problem_id")

    def build_facets(self, row: Mapping[str, Any]) -> dict[str, Any]:
        facets: dict[str, Any] = {}
        for raw_key, facet_key in self.facet_map.items():
            value = row.get(raw_key)
            if value is not None and str(value).strip() != "":
                facets[facet_key] = value
        return facets

    def convert(self, source_path: Path, output_path: Path) -> list[dict[str, Any]]:
        raw = self.load_source(source_path)
        source_rows = self.iter_source_rows(raw)
        converted = [self.convert_row(index, row) for index, row in enumerate(source_rows, 1)]
        for index, row in enumerate(converted, 1):
            self.validate_row(index, row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_annotations(output_path, converted)
        return converted

    def write_annotations(self, output_path: Path, rows: list[dict[str, Any]]) -> None:
        if self.annotation_format == "json":
            output_path.write_text(
                json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if self.annotation_format == "jsonl":
            with output_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            return
        raise ValueError(f"unsupported annotation_format {self.annotation_format!r} for {self.name}")

    def validate_annotation(self, path: Path) -> tuple[int, int]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"annotation must be a list: {path}")
        unit_ids: set[str] = set()
        for index, row in enumerate(data, 1):
            if not isinstance(row, Mapping):
                raise ValueError(f"annotation row {index} must be an object")
            self.validate_row(index, row)
            unit_ids.add(str(self.unit_key(row)))
        return len(data), len(unit_ids)

    def registry_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "annotation": str(self.output_relative),
            "video_dir": self.video_dir,
            "status": "ready",
            "source_format": self.source_format,
            "source_path": str(self.source_relative),
        }
        if self.image_dir:
            entry["image_dir"] = self.image_dir
        if self.scoring_protocol:
            entry["scoring_protocol"] = self.scoring_protocol
        if self.report:
            entry["report"] = dict(self.report)
        return entry


def parquet_value(value: Any) -> Any:
    """Normalize a pandas/parquet value into JSON-serializable form.

    ``float('nan')`` becomes ``None`` because standard JSON cannot represent NaN.
    """
    if hasattr(value, "tolist"):
        return parquet_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): parquet_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [parquet_value(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def dot_relative(path: str) -> str:
    """Ensure a video path starts with ``./`` for portable annotations."""
    text = str(path).strip()
    return text if text.startswith("./") else f"./{text}"


def require_str(row: Mapping[str, Any], key: str, index: int, *, dataset: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{dataset} row {index} missing {key}")
    return value.strip()


def require_options(row: Mapping[str, Any], index: int, *, key: str = "options", dataset: str) -> list[str]:
    raw_options = row.get(key)
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError(f"{dataset} row {index} has invalid {key}")
    options = [str(option).strip() for option in raw_options]
    if any(not option for option in options):
        raise ValueError(f"{dataset} row {index} has empty {key}")
    return options


def normalize_answer_letter(answer: Any, n_options: int, *, index: int, dataset: str) -> str:
    letter = str(answer).strip().upper()
    valid_letters = {chr(ord("A") + offset) for offset in range(n_options)}
    if letter not in valid_letters:
        raise ValueError(f"{dataset} row {index} answer {letter!r} is not in {sorted(valid_letters)}")
    return letter


def make_mcq_row(
    *,
    problem_id: Any,
    problem: str,
    options: list[str],
    answer: str,
    data_source: str,
    data_type: str | None = None,
    path: str | None = None,
    facets: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "problem_id": problem_id,
        "problem": problem,
        "problem_type": "multiple choice",
        "options": options,
        "solution": f"<answer>{answer}</answer>",
        "data_source": data_source,
    }
    if data_type:
        row["data_type"] = data_type
    if path:
        row["path"] = path
    clean_facets = {k: v for k, v in dict(facets or {}).items() if v is not None and str(v).strip() != ""}
    if clean_facets:
        row["facets"] = clean_facets
    row.update(dict(extra or {}))
    return row


def validate_standard_row(index: int, row: Mapping[str, Any]) -> None:
    required = {"problem_id", "problem", "problem_type", "solution", "data_source"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"annotation row {index} missing keys: {missing}")
    if row.get("problem_type") == "multiple choice":
        options = row.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"annotation row {index} multiple choice requires non-empty options")
    if "facets" in row and not isinstance(row["facets"], Mapping):
        raise ValueError(f"annotation row {index} facets must be an object")


__all__ = [
    "BaseConverter",
    "dot_relative",
    "make_mcq_row",
    "normalize_answer_letter",
    "parquet_value",
    "require_options",
    "require_str",
    "validate_standard_row",
]

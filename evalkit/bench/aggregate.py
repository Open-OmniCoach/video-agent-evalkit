"""Corpus-level metric aggregation.

This layer turns scored :class:`~evalkit.types.Prediction` rows into reported
per-benchmark metrics.

Most problem types just average the per-sample score (``mean_score``). Only
metrics that genuinely need the whole set (e.g. AP / ranking) register their
own aggregator via :func:`register`. Failed or cancelled samples contribute zero
to numeric metric denominators and are also reported by :class:`ResumeState`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping

from ..types import Prediction, ProblemType

Aggregator = Callable[[list[Prediction]], dict[str, Any]]

AGGREGATORS: dict[ProblemType, Aggregator] = {}
# Per-dataset corpus aggregators (keyed by dataset / config name). A benchmark
# whose headline metric needs the whole set (grouped/weighted accuracy, mIoU +
# R@IoU, AP, per-phase breakdown, ...) registers here; it wins over the
# ProblemType default.
DATASET_AGGREGATORS: dict[str, Aggregator] = {}


def register(problem_type: ProblemType) -> Callable[[Aggregator], Aggregator]:
    def _decorator(fn: Aggregator) -> Aggregator:
        AGGREGATORS[problem_type] = fn
        return fn

    return _decorator


def register_dataset(name: str) -> Callable[[Aggregator], Aggregator]:
    def _decorator(fn: Aggregator) -> Aggregator:
        DATASET_AGGREGATORS[name] = fn
        return fn

    return _decorator


def _numeric_score(pred: Prediction) -> float | None:
    score = pred.score
    if isinstance(score, bool):
        return float(score)
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict) and isinstance(score.get("score"), (int, float)):
        return float(score["score"])
    return None


def _effective_numeric_score(pred: Prediction) -> float | None:
    """Return the numeric value used for aggregation, including failed rows."""
    if pred.status in ("failed", "cancelled"):
        return 0.0
    return _numeric_score(pred)


def mean_score(records: list[Prediction]) -> dict[str, Any]:
    """Default aggregator: mean numeric score with failed rows scored as zero."""
    scored = [_effective_numeric_score(record) for record in records]
    values = [v for v in scored if v is not None]
    return {
        "score": (sum(values) / len(values)) if values else 0.0,
        "n_scored": len(values),
    }


def _field_value(record: Prediction, key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if isinstance(current, Prediction):
            current = getattr(current, part, None)
        elif isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _breakdown_specs(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    raw_specs = (report or {}).get("breakdowns") or []
    specs: list[dict[str, Any]] = []
    for item in raw_specs:
        if isinstance(item, str):
            specs.append({"name": item.split(".")[-1], "key": item})
        elif isinstance(item, Mapping):
            keys = item.get("keys")
            key = item.get("key")
            if keys:
                resolved_keys = [str(k) for k in keys]
                default_name = "_by_".join(k.split(".")[-1] for k in resolved_keys)
                specs.append(
                    {
                        "name": str(item.get("name") or default_name),
                        "keys": resolved_keys,
                    }
                )
            elif key:
                key_text = str(key)
                specs.append({"name": str(item.get("name") or key_text.split(".")[-1]), "key": key_text})
    return specs


def _group_label(values: list[Any]) -> str:
    normalized = ["__missing__" if value is None or str(value).strip() == "" else str(value) for value in values]
    return " | ".join(normalized)


def _numeric_breakdowns(records: list[Prediction], report: Mapping[str, Any] | None) -> dict[str, Any]:
    breakdowns: dict[str, Any] = {}
    for spec in _breakdown_specs(report):
        keys = list(spec.get("keys") or [spec["key"]])
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            score = _effective_numeric_score(record)
            if score is None:
                continue
            label = _group_label([_field_value(record, key) for key in keys])
            grouped[label].append(score)
        breakdowns[str(spec["name"])] = {
            label: {"score": sum(values) / len(values), "n_scored": len(values)}
            for label, values in sorted(grouped.items())
        }
    return breakdowns


def _with_report_breakdowns(
    metrics: dict[str, Any],
    records: list[Prediction],
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    breakdowns = _numeric_breakdowns(records, report)
    if not breakdowns:
        return metrics
    out = dict(metrics)
    existing = dict(out.get("breakdowns") or {})
    existing.update(breakdowns)
    out["breakdowns"] = existing
    return out


def aggregate(
    problem_type: ProblemType,
    records: list[Prediction],
    *,
    aggregator_name: str | None = None,
    aggregator_fn: Aggregator | None = None,
    dataset: str | None = None,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick the aggregator: explicit name > dataset name > ProblemType default.

    Mirrors :func:`evalkit.bench.scoring.resolve_scorer`: an explicit
    ``aggregator_name`` (from bench config) must exist or we raise; an implicit
    ``dataset`` (bench name) silently falls back to the ProblemType default.
    """
    if aggregator_fn is not None:
        metrics = aggregator_fn(records)
        if not isinstance(metrics, dict):
            raise TypeError(f"custom aggregator returned {type(metrics).__name__}, expected dict")
    elif aggregator_name:
        if aggregator_name not in DATASET_AGGREGATORS:
            raise KeyError(f"unknown dataset aggregator {aggregator_name!r}; registered: {sorted(DATASET_AGGREGATORS)}")
        metrics = DATASET_AGGREGATORS[aggregator_name](records)
    elif dataset and dataset in DATASET_AGGREGATORS:
        metrics = DATASET_AGGREGATORS[dataset](records)
    else:
        metrics = AGGREGATORS.get(problem_type, mean_score)(records)
    return _with_report_breakdowns(metrics, records, report)


__all__ = [
    "Aggregator",
    "AGGREGATORS",
    "DATASET_AGGREGATORS",
    "register",
    "register_dataset",
    "mean_score",
    "aggregate",
]

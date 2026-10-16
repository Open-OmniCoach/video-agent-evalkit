"""Benchmark-specific scorers and aggregators.

Registering here connects a benchmark's bespoke metric:
* a per-sample :class:`Scorer` via ``register_dataset_scorer`` (overrides the
  generic ``SCORERS[problem_type]``), and/or
* a corpus aggregator via ``register_dataset`` (overrides ``mean_score``).

Selected by configured scorer / dataset name and aggregator / dataset name.

Judge-based benches (activitynetqa, nextqa, omnicloze, dream1k, omnigaia) use the
per-sample judge path instead, so they need no entry here.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from ..types import Prediction
from .aggregate import _effective_numeric_score, register_dataset
from .scoring import register_dataset_scorer

# =====================================================================
# Diving48 F-16: four-phase tuple. Per-sample score = fraction of phases
# matched; corpus detail = per-phase accuracy + exact-tuple accuracy.
# =====================================================================

_PHASES = ("takeoff", "somersault", "twist", "flight")
_ALLOWED_TOKENS: dict[str, tuple[str, ...]] = {
    "takeoff": ("Back", "Forward", "Inward", "Reverse"),
    "somersault": ("Dive", "1som", "15som", "2som", "25som", "3som", "35som", "45som"),
    "twist": ("NoTwis", "05Twis", "1Twis", "15Twis", "2Twis", "25Twis", "3Twis", "35Twis"),
    "flight": ("STR", "PIKE", "TUCK", "FREE"),
}


def _try_parse_mapping(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    return value if isinstance(value, dict) else None


def _extract_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    candidates: list[str] = []
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        candidates.append(answer_match.group(1).strip())
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    start = text.find("{")
    while start >= 0:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
        start = text.find("{", start + 1)
    candidates.append(text)
    for candidate in candidates:
        parsed = _try_parse_mapping(candidate)
        if parsed is not None:
            return parsed
    return {}


def _canonical_token(phase: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for token in _ALLOWED_TOKENS[phase]:
        if text == token:
            return token
    lowered = text.lower()
    for token in _ALLOWED_TOKENS[phase]:
        if lowered == token.lower():
            return token
    return ""


def _parse_phases(value: Any) -> dict[str, str]:
    payload = _extract_object(value)
    return {phase: _canonical_token(phase, payload.get(phase)) for phase in _PHASES}


@register_dataset_scorer("diving48_f16")
class Diving48F16Scorer:
    """Per-sample phase-match score; prediction carries phase details."""

    def extract(self, raw: str, sample: Mapping[str, Any]) -> dict[str, Any]:
        predicted = _parse_phases(raw)
        expected = _parse_phases(sample.get("solution", ""))
        matches = {p: bool(expected[p]) and predicted[p] == expected[p] for p in _PHASES}
        return {"phases": predicted, "matches": matches}

    def score(self, pred: Any, sample: Mapping[str, Any]) -> float:
        del sample
        matches = (pred or {}).get("matches", {}) if isinstance(pred, dict) else {}
        return sum(1 for hit in matches.values() if hit) / len(_PHASES)


@register_dataset("diving48_f16")
def _aggregate_diving48(records: list[Prediction]) -> dict[str, Any]:
    scored = [
        record for record in records if record.status in ("failed", "cancelled") or isinstance(record.prediction, dict)
    ]
    n = len(scored)
    phase_correct = {p: 0 for p in _PHASES}
    exact = 0
    for r in scored:
        if not isinstance(r.prediction, dict):
            continue
        matches = r.prediction.get("matches", {})
        for p in _PHASES:
            if matches.get(p):
                phase_correct[p] += 1
        if all(matches.get(p) for p in _PHASES):
            exact += 1
    phase_acc = {f"{p}_acc": (phase_correct[p] / n if n else 0.0) for p in _PHASES}
    mean_phase_acc = sum(phase_acc.values()) / len(_PHASES) if n else 0.0
    return {
        "score": mean_phase_acc,
        "n_scored": n,
        "protocol": "f16_phase_mean_accuracy",
        "exact_tuple_acc": (exact / n if n else 0.0),
        **phase_acc,
    }


# =====================================================================
# LongVALE temporal grounding: per-sample score is temporal IoU (from the
# generic TEMPORAL_GROUNDING scorer); corpus metric = mIoU + R1@IoU thresholds.
# =====================================================================

_LONGVALE_THRESHOLDS = (0.3, 0.5, 0.7)


@register_dataset("timelens_bench")
@register_dataset("longvale_grounding")
def _aggregate_longvale(records: list[Prediction]) -> dict[str, Any]:
    ious = [v for r in records if (v := _effective_numeric_score(r)) is not None]
    n = len(ious)
    mean_iou = sum(ious) / n if n else 0.0
    detail: dict[str, Any] = {
        "score": mean_iou,
        "n_scored": n,
        "protocol": "temporal_grounding_miou_r1",
        "mIoU": mean_iou,
    }
    for threshold in _LONGVALE_THRESHOLDS:
        detail[f"R1@{threshold:g}"] = (sum(1 for iou in ious if iou >= threshold) / n) if n else 0.0
    return detail


# =====================================================================
# Video-MME-v2: four questions form one relevance/logic group. The
# official headline is the mean nonlinear group rating, not sample accuracy.
# =====================================================================


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _video_mme_v2_logic(scores: list[int], structure: str) -> float:
    parsed = ast.literal_eval(structure)
    last_correct = -1
    for index, score in enumerate(scores):
        if score:
            last_correct = index
        else:
            break
    if parsed == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 25.0, 3: 56.25, 4: 100.0}
    elif parsed == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 / 3, 3: 700.0 / 12, 4: 100.0}
        if last_correct == 0 and scores[2]:
            last_correct += 1
    elif parsed == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 10.0, 2: 20.0, 3: 50.0, 4: 100.0}
        if last_correct == -1 and scores[1]:
            last_correct += 1
    else:
        raise ValueError(f"unknown Video-MME-v2 group structure: {structure!r}")
    return score_map.get(last_correct + 1, 0.0)


@register_dataset("video_mme_v2")
@register_dataset("video_mme_v2_full")
def _aggregate_video_mme_v2(records: list[Prediction]) -> dict[str, Any]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for record in records:
        group_id = record.facets.get("group_id")
        if group_id is not None:
            grouped[str(group_id)].append(record)

    ratings: dict[str, list[float]] = defaultdict(list)
    second_heads: dict[str, list[float]] = defaultdict(list)
    third_heads: dict[str, list[float]] = defaultdict(list)
    complete_groups = 0
    incomplete_groups = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda record: int(record.facets.get("group_position", 0)))
        positions = [int(record.facets.get("group_position", 0)) for record in ordered]
        if positions != [1, 2, 3, 4]:
            incomplete_groups += 1
            continue
        complete_groups += 1
        scores = [int((_effective_numeric_score(record) or 0.0) > 0.0) for record in ordered]
        group_type = str(ordered[-1].facets.get("group_type"))
        if group_type == "relevance":
            count = sum(scores)
            rating = {0: 0.0, 1: 6.25, 2: 25.0, 3: 56.25, 4: 100.0}[count]
            ratings["relevance_score"].append(rating)
            ratings["relevance_linear_score"].append(count * 25.0)
        elif group_type == "logic":
            rating = _video_mme_v2_logic(scores, str(ordered[-1].facets.get("group_structure")))
            ratings["logic_score"].append(rating)
        else:
            raise ValueError(f"unknown Video-MME-v2 group type: {group_type!r}")
        ratings["total"].append(rating)
        level = ordered[-1].facets.get("level")
        if level is not None:
            ratings[f"level_{int(float(level))}"].append(rating)
        second_head = ordered[-1].facets.get("second_head")
        third_head = ordered[-1].facets.get("third_head")
        if second_head is not None:
            second_heads[str(second_head)].append(rating)
        if third_head is not None:
            third_heads[str(third_head)].append(rating)

    total_percent = _mean(ratings["total"])
    sample_scores = [value for record in records if (value := _effective_numeric_score(record)) is not None]
    return {
        "score": total_percent / 100.0,
        "n_scored": len(sample_scores),
        "n_groups": complete_groups,
        "n_incomplete_groups": incomplete_groups,
        "protocol": "official_video_mme_v2_grouped_rating",
        "total_percent": total_percent,
        "sample_accuracy": _mean(sample_scores),
        **{f"{key}_percent": _mean(values) for key, values in sorted(ratings.items()) if key != "total"},
        "second_head_percent": {key: _mean(values) for key, values in sorted(second_heads.items())},
        "third_head_percent": {key: _mean(values) for key, values in sorted(third_heads.items())},
    }


__all__ = ["Diving48F16Scorer"]

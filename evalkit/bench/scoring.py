"""Rule-based scoring for each :class:`ProblemType`.

Each scorer takes a raw model response plus the case scoring context and returns
a parsed prediction and score. OPEN_ENDED returns ``score=None`` so an LLM judge
can score it asynchronously in the runtime scoring stage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Protocol

from ..types import ProblemType


class Scorer(Protocol):
    def extract(self, raw: str, sample: Mapping[str, Any]) -> Any: ...

    def score(self, pred: Any, sample: Mapping[str, Any]) -> float | None: ...


_OPTION_SET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_FINAL_ANSWER_RE = re.compile(
    r"(?:Therefore|So|Thus|Hence|Finally)[,\s]*(?:the\s+)?"
    r"(?:final\s+|best\s+|correct\s+)?answer\s+is[:\s]+(.+)",
    re.IGNORECASE,
)
_JSON_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PARTIAL_JSON_TIME_RE = re.compile(
    rf'"time"\s*:\s*\[\s*({_JSON_NUMBER_RE})\s*,\s*({_JSON_NUMBER_RE})',
    flags=re.IGNORECASE,
)
_TEXT_TIME_VALUE_RE = r"(?:\d+:\d{2}(?::\d{2}(?:\.\d+)?)?|\d+(?:\.\d+)?)"


def _normalize_problem_type(value: Any) -> ProblemType:
    if isinstance(value, ProblemType):
        return value
    return ProblemType(str(value))


def _extract_between(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_final_answer_scope(text: str) -> str | None:
    openings = list(re.finditer(r"<answer>\s*", text, flags=re.IGNORECASE))
    if not openings:
        return None
    start = openings[-1].end()
    closing = re.search(r"</answer>", text[start:], flags=re.IGNORECASE)
    end = start + closing.start() if closing else len(text)
    return text[start:end].strip()


def _clean_text(text: str) -> str:
    cleaned = _strip_think_blocks(text)
    cleaned = re.sub(r"</?answer>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_think_blocks(text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"</?think>", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _extract_mc_letter(text: str, valid_options: frozenset[str] = _OPTION_SET) -> str:
    stripped = text.strip().rstrip(".")
    if len(stripped) == 1 and stripped.upper() in valid_options:
        return stripped.upper()

    patterns = (
        r"answer\s+is[:\s]+([A-Z])\b",
        r"(?:choose|select|pick)\s+(?:option\s+)?([A-Z])\b",
        r"(?:^|\b)([A-Z])(?:\s*[.)。：:,，]|\s|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            letter = match.group(1).upper()
            if letter in valid_options:
                return letter
    return ""


def _extract_mc_letters(text: str, valid_options: frozenset[str] = _OPTION_SET) -> str:
    cleaned = text.strip().upper()
    if not cleaned:
        return ""

    letters: list[str] = []
    patterns = (
        r"(?<![A-Z])([A-Z])(?=\s*[.)。：:,，]|\s+(?:AND|OR)\b|\s*&|\s|$)",
        r"\bOPTION\s+([A-Z])\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned):
            letter = match.group(1).upper()
            if letter in valid_options and letter not in letters:
                letters.append(letter)
        if letters:
            break
    return ",".join(letters)


def _extract_mc_letter_from_json(text: str, valid_options: frozenset[str] = _OPTION_SET) -> str:
    if not text:
        return ""

    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        candidates.append(match.group(0))
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])

    for raw in candidates:
        for candidate in (raw, _strip_trailing_commas(raw)):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            answer_value = data.get("answer")
            if answer_value is None:
                continue
            letter = _extract_mc_letter(str(answer_value), valid_options)
            if letter:
                return letter
    return ""


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def extract_answer(text: str, problem_type: ProblemType | str) -> str:
    """Extract the final answer text using the legacy priority order."""
    resolved_type = _normalize_problem_type(problem_type)

    if resolved_type == ProblemType.MULTIPLE_CHOICE:
        json_letter = _extract_mc_letter_from_json(text)
        if json_letter:
            return json_letter

    answer = _extract_between(text, "answer")
    if answer:
        if resolved_type == ProblemType.MULTIPLE_CHOICE:
            letter = _extract_mc_letter(answer)
            return letter or answer.strip()
        return answer.strip()

    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        answer = final_match.group(1).strip()
        if resolved_type == ProblemType.MULTIPLE_CHOICE:
            letter = _extract_mc_letter(answer)
            return letter or answer
        first_line = answer.splitlines()[0].strip()
        return first_line or answer

    think_match = re.search(r"</think>\s*(.*?)$", text, flags=re.IGNORECASE | re.DOTALL)
    if think_match:
        answer = think_match.group(1).strip()
        if answer:
            if resolved_type == ProblemType.MULTIPLE_CHOICE:
                letter = _extract_mc_letter(answer)
                return letter or answer
            return answer

    if resolved_type == ProblemType.MULTIPLE_CHOICE:
        for line in reversed(text.strip().splitlines()[-5:]):
            letter = _extract_mc_letter(line.strip())
            if letter:
                return letter

    return _clean_text(text)


def _normalize_number(text: str) -> float | None:
    match = re.search(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)", text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _normalize_yes_no(text: str) -> str:
    cleaned = text.strip().upper()
    if cleaned in {"YES", "Y", "1", "TRUE"}:
        return "YES"
    if cleaned in {"NO", "N", "0", "FALSE"}:
        return "NO"
    if re.search(r"\bYES\b", cleaned):
        return "YES"
    if re.search(r"\bNO\b", cleaned):
        return "NO"
    return ""


def _extract_grounding_payload(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return {"time": list(value)}
    if value is None:
        return None

    raw = str(value)
    answer = _extract_final_answer_scope(raw)
    text = answer if answer is not None else extract_answer(raw, ProblemType.TEMPORAL_GROUNDING)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(data, dict):
            return data
    # Guided decoding may hit the token budget after both values but before
    # the JSON array/object closes.
    partial_time = _PARTIAL_JSON_TIME_RE.search(text)
    if partial_time is not None:
        return {"time": [float(group) for group in partial_time.groups()]}
    span = _extract_text_time_span(text)
    if span is not None:
        return {"time": span}
    return None


def _extract_text_time_span(text: str) -> list[float] | None:
    normalized = re.sub(r"</?t>", "", text, flags=re.IGNORECASE)
    value = _TEXT_TIME_VALUE_RE
    patterns = (
        rf"\[\s*({value})\s*,\s*({value})\s*\]",
        rf"\bfrom\s+({value})\s*(?:s|sec|secs|second|seconds)?\s+"
        rf"(?:to|and|[-–—])\s+({value})\s*(?:s|sec|secs|second|seconds)?\b",
        rf"\b({value})\s*(?:s|sec|secs|second|seconds)?\s*"
        rf"(?:to|and|[-–—])\s*({value})\s*(?:s|sec|secs|second|seconds)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        start = _parse_time_value(match.group(1))
        end = _parse_time_value(match.group(2))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        return [start, end]
    return None


def _parse_time_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    token = value.strip()
    if not token:
        return None

    if ":" in token:
        parts = token.split(":")
        if len(parts) not in {2, 3} or any(part.strip() == "" for part in parts):
            return None
        try:
            values = [float(part.strip()) for part in parts]
        except ValueError:
            return None
        if any(part < 0 for part in values):
            return None
        total = 0.0
        for part in values:
            total = (total * 60.0) + part
        return total

    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _normalize_time_span(span: Any) -> list[float] | None:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return None
    start = _parse_time_value(span[0])
    end = _parse_time_value(span[1])
    if start is None or end is None:
        return None
    return [start, end]


def _iou_1d(pred: Any, gt: Any) -> float:
    pred_span = _normalize_time_span(pred)
    gt_span = _normalize_time_span(gt)
    if pred_span is None or gt_span is None:
        return 0.0
    pred_start, pred_end = pred_span
    gt_start, gt_end = gt_span
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 1e-12 else 0.0


def _normalize_box_format(value: Any, default: str = "xyxy") -> str:
    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower()
    token = token.strip("[]()")
    for char in (" ", "_", "-", ","):
        token = token.replace(char, "")
    if token == "xywh":
        return "xywh"
    if token in {"xyxy", "x1y1x2y2"}:
        return "xyxy"
    return default


def _first_present(mapping: Any, keys: tuple[str, ...]) -> Any:
    if not hasattr(mapping, "get"):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _box_to_xyxy(box: Any, box_format: str) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        first, second, third, fourth = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if box_format == "xywh":
        return [first, second, first + third, second + fourth]
    return [first, second, third, fourth]


def _iou_2d(pred: Any, gt: Any, *, pred_format: str = "xyxy", gt_format: str = "xyxy") -> float:
    pred_box = _box_to_xyxy(pred, pred_format)
    gt_box = _box_to_xyxy(gt, gt_format)
    if pred_box is None or gt_box is None:
        return 0.0
    x1, y1, x2, y2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    inter_x1 = max(x1, gx1)
    inter_y1 = max(y1, gy1)
    inter_x2 = min(x2, gx2)
    inter_y2 = min(y2, gy2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    pred_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter_area
    return inter_area / union if union > 1e-12 else 0.0


def _mean_iou_over_intersection(
    pred_boxes: Any,
    gt_boxes: Any,
    *,
    pred_format: str = "xyxy",
    gt_format: str = "xyxy",
) -> float:
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common_keys = [key for key in pred_boxes if key in gt_boxes]
    if not common_keys:
        return 0.0
    values = [
        _iou_2d(pred_boxes[key], gt_boxes[key], pred_format=pred_format, gt_format=gt_format) for key in common_keys
    ]
    return sum(values) / len(values) if values else 0.0


def _mean_relative_accuracy(pred: float, target: float) -> float:
    eps = 1e-8
    rel_error = abs(pred - target) / (abs(target) + eps)
    thresholds = [0.5 + (0.05 * index) for index in range(10)]
    hits = [1.0 for threshold in thresholds if rel_error < (1.0 - threshold)]
    return sum(hits) / len(thresholds)


@dataclass(frozen=True)
class _FunctionScorer:
    extractor: Callable[[str, Mapping[str, Any]], Any]
    score_fn: Callable[[Any, Mapping[str, Any]], float | None]

    def extract(self, raw: str, sample: Mapping[str, Any]) -> Any:
        return self.extractor(raw, sample)

    def score(self, pred: Any, sample: Mapping[str, Any]) -> float | None:
        return self.score_fn(pred, sample)


def _extract_multiple_choice(raw: str, sample: Mapping[str, Any]) -> str:
    valid_options = _valid_mc_options(sample)
    cleaned_raw = _drop_echoed_option_lines(raw, sample)
    if _is_multi_select_sample(sample):
        return _extract_multiple_choice_letters(cleaned_raw, valid_options)

    candidates = [_strip_think_blocks(cleaned_raw), cleaned_raw]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        answer = extract_answer(candidate, ProblemType.MULTIPLE_CHOICE).strip().upper()
        if len(answer) == 1:
            if answer in valid_options:
                return answer
            continue
        letter = _extract_mc_letter(answer, valid_options)
        if letter:
            return letter
    return ""


def _extract_multiple_choice_letters(raw: str, valid_options: frozenset[str]) -> str:
    final_match = _FINAL_ANSWER_RE.search(raw)
    candidates = [
        _extract_between(raw, "answer"),
        final_match.group(1).strip() if final_match else "",
        _strip_think_blocks(raw),
        raw,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        letters = _extract_mc_letters(candidate, valid_options)
        if letters:
            return letters
    return ""


def _valid_mc_options(sample: Mapping[str, Any]) -> frozenset[str]:
    options = sample.get("options") or ()
    letters: set[str] = set()
    for index, option in enumerate(options):
        fallback = chr(ord("A") + index)
        text = str(option).strip()
        match = re.match(r"^\s*([A-Z])\s*[.)。：:]", text, flags=re.IGNORECASE)
        letters.add((match.group(1) if match else fallback).upper())
    return frozenset(letters) or _OPTION_SET


def _drop_echoed_option_lines(raw: str, sample: Mapping[str, Any]) -> str:
    option_lines = {str(option).strip() for option in (sample.get("options") or ())}
    if not option_lines:
        return raw
    echoed_count = sum(1 for line in raw.splitlines() if line.strip() in option_lines)
    if echoed_count < 2:
        return raw
    lines = [line for line in raw.splitlines() if line.strip() not in option_lines]
    return "\n".join(lines)


def _is_multi_select_sample(sample: Mapping[str, Any]) -> bool:
    if str(sample.get("answer_mode", "")).strip().lower() == "multi_select":
        return True
    expected = _expected_mc_letters(sample)
    return "," in expected


def _expected_mc_letters(sample: Mapping[str, Any]) -> str:
    valid_options = _valid_mc_options(sample)
    solution = str(sample.get("solution", ""))
    answer = (_extract_between(solution, "answer") or solution).strip().upper()
    if "," in answer or "，" in answer:
        return _extract_mc_letters(answer, valid_options)
    letter = _extract_mc_letter(answer, valid_options)
    if letter:
        return letter
    return extract_answer(solution, ProblemType.MULTIPLE_CHOICE).strip().upper()


def _score_multiple_choice(pred: Any, sample: Mapping[str, Any]) -> float:
    expected = _expected_mc_letters(sample)
    predicted = (
        _extract_mc_letters(str(pred), _valid_mc_options(sample)) if "," in expected else str(pred).strip().upper()
    )
    return 1.0 if predicted == expected and expected else 0.0


def _extract_number(problem_type: ProblemType) -> Callable[[str, Mapping[str, Any]], float | None]:
    def _inner(raw: str, sample: Mapping[str, Any]) -> float | None:
        del sample
        return _normalize_number(extract_answer(raw, problem_type))

    return _inner


def _score_numerical(pred: Any, sample: Mapping[str, Any]) -> float:
    predicted = float(pred) if isinstance(pred, (int, float)) else None
    expected = _normalize_number(extract_answer(str(sample.get("solution", "")), ProblemType.NUMERICAL))
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if abs(predicted - expected) <= 0.01 else 0.0


def _score_regression(pred: Any, sample: Mapping[str, Any]) -> float:
    predicted = float(pred) if isinstance(pred, (int, float)) else None
    expected = _normalize_number(extract_answer(str(sample.get("solution", "")), ProblemType.REGRESSION))
    if predicted is None or expected is None:
        return 0.0
    return _mean_relative_accuracy(predicted, expected)


def _extract_rec(raw: str, sample: Mapping[str, Any]) -> str:
    del sample
    return "".join(re.findall(r"\d+", extract_answer(raw, ProblemType.REC)))


def _score_rec(pred: Any, sample: Mapping[str, Any]) -> float:
    predicted = str(pred)
    expected = "".join(re.findall(r"\d+", extract_answer(str(sample.get("solution", "")), ProblemType.REC)))
    return 1.0 if expected and predicted == expected else 0.0


def _extract_yes_no(raw: str, sample: Mapping[str, Any]) -> str:
    problem_type = _normalize_problem_type(sample.get("problem_type"))
    return _normalize_yes_no(extract_answer(raw, problem_type))


def _score_yes_no(pred: Any, sample: Mapping[str, Any]) -> float:
    predicted = str(pred).strip().upper()
    if not predicted:
        return 0.0
    if "ovo_type" in sample:
        expected = "YES" if int(sample["ovo_type"]) == 1 else "NO"
    else:
        problem_type = _normalize_problem_type(sample.get("problem_type"))
        expected = _normalize_yes_no(extract_answer(str(sample.get("solution", "")), problem_type))
    return 1.0 if predicted == expected and expected else 0.0


def _extract_grounding(raw: str, sample: Mapping[str, Any]) -> Dict[str, Any] | None:
    del sample
    return _extract_grounding_payload(raw)


def _score_temporal_grounding(pred: Any, sample: Mapping[str, Any]) -> float:
    expected = _extract_grounding_payload(sample.get("solution", ""))
    if pred is None or expected is None:
        return 0.0
    return _iou_1d(pred.get("time"), expected.get("time"))


def _score_spatial_temporal_grounding(pred: Any, sample: Mapping[str, Any]) -> float:
    expected = _extract_grounding_payload(sample.get("solution", ""))
    if pred is None or expected is None or not hasattr(pred, "get"):
        return 0.0
    temporal_iou = _iou_1d(pred.get("time"), expected.get("time"))
    expected_boxes = expected.get("boxes")
    if not isinstance(expected_boxes, dict) or not expected_boxes:
        return temporal_iou
    sample_box_format = _normalize_box_format(_first_present(sample, ("box_format", "bbox_format")))
    expected_format = _normalize_box_format(
        _first_present(expected, ("box_format", "bbox_format", "solution_box_format")),
        sample_box_format,
    )
    pred_format = _normalize_box_format(
        _first_present(pred, ("box_format", "bbox_format", "prediction_box_format", "pred_box_format")),
        _normalize_box_format(
            _first_present(sample, ("prediction_box_format", "pred_box_format")),
            sample_box_format,
        ),
    )
    spatial_iou = _mean_iou_over_intersection(
        pred.get("boxes"),
        expected_boxes,
        pred_format=pred_format,
        gt_format=expected_format,
    )
    return 0.5 * temporal_iou + 0.5 * spatial_iou


def _extract_open_ended(raw: str, sample: Mapping[str, Any]) -> str:
    del sample
    return extract_answer(raw, ProblemType.OPEN_ENDED)


def _score_open_ended(pred: Any, sample: Mapping[str, Any]) -> None:
    del pred, sample
    return None


SCORERS: dict[ProblemType, Scorer] = {
    ProblemType.MULTIPLE_CHOICE: _FunctionScorer(_extract_multiple_choice, _score_multiple_choice),
    ProblemType.NUMERICAL: _FunctionScorer(_extract_number(ProblemType.NUMERICAL), _score_numerical),
    ProblemType.REGRESSION: _FunctionScorer(_extract_number(ProblemType.REGRESSION), _score_regression),
    ProblemType.REC: _FunctionScorer(_extract_rec, _score_rec),
    ProblemType.SSR: _FunctionScorer(_extract_yes_no, _score_yes_no),
    ProblemType.CRR: _FunctionScorer(_extract_yes_no, _score_yes_no),
    ProblemType.TEMPORAL_GROUNDING: _FunctionScorer(_extract_grounding, _score_temporal_grounding),
    ProblemType.SPATIAL_TEMPORAL_GROUNDING: _FunctionScorer(_extract_grounding, _score_spatial_temporal_grounding),
    ProblemType.OPEN_ENDED: _FunctionScorer(_extract_open_ended, _score_open_ended),
}


# ---- per-dataset scorer override (keyed by dataset / config name) ----
# A benchmark with a bespoke per-sample metric (beyond a generic ProblemType)
# registers its own Scorer here. Dispatch prefers, in order: an explicit
# configured scorer name, then the dataset name, then the problem-type default.
DATASET_SCORERS: dict[str, Scorer] = {}


def register_dataset_scorer(name: str):
    """Register a per-dataset Scorer (decorator; accepts a class or instance)."""

    def _decorator(scorer):
        DATASET_SCORERS[name] = scorer() if isinstance(scorer, type) else scorer
        return scorer

    return _decorator


def resolve_scorer(
    problem_type: ProblemType,
    *,
    dataset: str | None = None,
    scorer_name: str | None = None,
) -> Scorer:
    """Pick the scorer: explicit name > dataset name > ProblemType default."""
    if scorer_name:
        if scorer_name in DATASET_SCORERS:
            return DATASET_SCORERS[scorer_name]
        raise KeyError(f"unknown dataset scorer {scorer_name!r}; registered: {sorted(DATASET_SCORERS)}")
    if dataset and dataset in DATASET_SCORERS:
        return DATASET_SCORERS[dataset]
    return SCORERS[problem_type]


__all__ = [
    "SCORERS",
    "Scorer",
    "extract_answer",
    "DATASET_SCORERS",
    "register_dataset_scorer",
    "resolve_scorer",
]

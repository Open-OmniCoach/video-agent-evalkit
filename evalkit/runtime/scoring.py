"""Evaluation-native rule and judge scoring."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..bench.scoring import resolve_scorer
from ..types import EvalCase, GenerationResult, GenerationStatus, ScoreResult
from .plugins import load_callable

ScoreFn = Callable[[Any, EvalCase, GenerationResult], Awaitable[ScoreResult]]

_JUDGE_TEMPLATE = (
    "You are grading a model's answer to a question.\n"
    "Question:\n{question}\n\n"
    "Reference answer:\n{reference}\n\n"
    "Model answer:\n{answer}\n\n"
    "If the model answer is correct or semantically equivalent to the reference, "
    "output 1. Otherwise output 0. Output ONLY the single digit."
)


def _parse_judge_score(text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return 0.0
    try:
        return float(match.group(0))
    except ValueError:
        return 0.0


async def score(
    runtime: Any,
    case: EvalCase,
    generation: GenerationResult,
    *,
    config: Mapping[str, Any] | None = None,
) -> ScoreResult:
    """Built-in scorer dispatching to a rule scorer or an LLM judge."""
    if generation.status not in {GenerationStatus.COMPLETED, GenerationStatus.TRUNCATED}:
        return ScoreResult()

    scoring = dict(config or {"type": "rule"})
    context = case.scoring_context()
    scoring_type = str(scoring.get("type", "rule")).lower()
    if scoring_type not in {"rule", "judge"}:
        raise ValueError(f"unsupported scoring type {scoring_type!r}")
    if scoring_type == "rule":
        scorer = resolve_scorer(
            case.problem_type,
            dataset=case.dataset,
            scorer_name=scoring.get("scorer"),
        )
        prediction = scorer.extract(generation.response, context)
        return ScoreResult(prediction=prediction, score=scorer.score(prediction, context))

    judge_name = str(scoring.get("model", "judge"))
    question = str(context.get("problem") or context.get("question") or "")
    rubric = str(scoring.get("rubric") or _JUDGE_TEMPLATE)
    prompt = rubric.format(question=question, reference=case.target, answer=generation.response)
    messages = ({"role": "user", "content": [{"type": "text", "text": prompt}]},)
    sampling = dict(scoring.get("sampling") or {"temperature": 0, "max_tokens": 8})
    judged = await runtime.chat(judge_name, messages, sampling=sampling, route_key=case.uid)
    return ScoreResult(
        prediction=generation.response,
        score=_parse_judge_score(judged.text),
        details={"judge_model": judge_name, "judge_response": judged.text},
    )


def build_score_fn(path: str, config: Mapping[str, Any]) -> ScoreFn:
    """Load a scorer and bind its per-benchmark configuration."""
    fn = load_callable(path)
    accepts_config = "config" in inspect.signature(fn).parameters

    async def _bound(runtime: Any, case: EvalCase, generation: GenerationResult) -> ScoreResult:
        if accepts_config:
            result = fn(runtime, case, generation, config=config)
        else:
            result = fn(runtime, case, generation)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ScoreResult):
            raise TypeError(f"score plugin {path!r} returned {type(result).__name__}, expected ScoreResult")
        return result

    return _bound


__all__ = ["ScoreFn", "build_score_fn", "score"]

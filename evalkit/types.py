"""Evaluation-native domain types.

The benchmark input, generation output, score, and persisted prediction are
separate values.  In particular, generation plugins never mutate benchmark
inputs and scoring never owns model-serving state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias


class ProblemType(str, Enum):
    """Discriminates parsing, scoring, and aggregation behavior."""

    MULTIPLE_CHOICE = "multiple choice"
    NUMERICAL = "numerical"
    REGRESSION = "regression"
    REC = "REC"
    SSR = "SSR"
    CRR = "CRR"
    TEMPORAL_GROUNDING = "temporal grounding"
    SPATIAL_TEMPORAL_GROUNDING = "spatial-temporal grounding"
    OPEN_ENDED = "open-ended"


class GenerationStatus(str, Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    FAILED = "failed"
    CANCELLED = "cancelled"


SamplingParams: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True)
class EvalCase:
    """One immutable-by-contract benchmark input."""

    uid: str
    dataset: str
    index: int
    messages: tuple[dict[str, Any], ...]
    target: Any
    problem_type: ProblemType
    metadata: Mapping[str, Any] = field(default_factory=dict)
    facets: Mapping[str, Any] = field(default_factory=dict)

    def scoring_context(self) -> dict[str, Any]:
        """Return the mapping consumed by existing benchmark scorers."""
        context = dict(self.metadata)
        context.update(
            {
                "uid": self.uid,
                "dataset": self.dataset,
                "problem_type": self.problem_type.value,
                "solution": self.target,
                "facets": dict(self.facets),
            }
        )
        return context


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "GenerationUsage") -> "GenerationUsage":
        return GenerationUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass(frozen=True)
class GenerationTiming:
    latency_ms: float = 0.0
    non_generation_ms: float = 0.0

    def __add__(self, other: "GenerationTiming") -> "GenerationTiming":
        return GenerationTiming(
            latency_ms=self.latency_ms + other.latency_ms,
            non_generation_ms=self.non_generation_ms + other.non_generation_ms,
        )


@dataclass(frozen=True)
class TrajectoryEvent:
    """One model/tool/error event in an agent trajectory."""

    kind: str
    turn: int
    data: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ms: float | None = None
    attempt: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    status: GenerationStatus
    response: str = ""
    assistant_message: Mapping[str, Any] | None = None
    trajectory: tuple[TrajectoryEvent, ...] = ()
    usage: GenerationUsage = field(default_factory=GenerationUsage)
    timing: GenerationTiming = field(default_factory=GenerationTiming)
    endpoint: str | None = None
    finish_reason: str | None = None
    error: str | None = None

    @classmethod
    def completed(
        cls,
        response: str,
        *,
        assistant_message: Mapping[str, Any] | None = None,
        trajectory: tuple[TrajectoryEvent, ...] = (),
        usage: GenerationUsage | None = None,
        timing: GenerationTiming | None = None,
        endpoint: str | None = None,
        finish_reason: str | None = None,
    ) -> "GenerationResult":
        return cls(
            status=GenerationStatus.COMPLETED,
            response=response,
            assistant_message=assistant_message,
            trajectory=trajectory,
            usage=usage or GenerationUsage(),
            timing=timing or GenerationTiming(),
            endpoint=endpoint,
            finish_reason=finish_reason,
        )


@dataclass(frozen=True)
class ScoreResult:
    prediction: Any = None
    score: Any = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResult:
    message: Mapping[str, Any]
    text: str
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    finish_reason: str | None = None
    usage: GenerationUsage = field(default_factory=GenerationUsage)
    timing: GenerationTiming = field(default_factory=GenerationTiming)
    endpoint: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """One scored result serialized to ``predictions.jsonl``."""

    uid: str
    status: str = GenerationStatus.COMPLETED.value
    response: str = ""
    prediction: Any = None
    score: Any = None
    score_details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    facets: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    trajectory_artifact: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "uid": str(self.uid),
            "status": self.status,
            "response": self.response,
            "prediction": self.prediction,
            "score": self.score,
            "score_details": self.score_details,
            "error": self.error or None,
            "facets": self.facets,
            "generation": self.generation,
            "trajectory_artifact": self.trajectory_artifact,
        }

    @staticmethod
    def from_row(row: Mapping[str, Any]) -> "Prediction":
        return Prediction(
            uid=str(row["uid"]),
            status=str(row.get("status", GenerationStatus.COMPLETED.value)),
            response=str(row.get("response", "")),
            prediction=row.get("prediction"),
            score=row.get("score"),
            score_details=dict(row.get("score_details") or {}),
            error=row.get("error"),
            facets=dict(row.get("facets") or {}),
            generation=dict(row.get("generation") or {}),
            trajectory_artifact=row.get("trajectory_artifact"),
        )


__all__ = [
    "ChatResult",
    "EvalCase",
    "GenerationResult",
    "GenerationStatus",
    "GenerationTiming",
    "GenerationUsage",
    "Prediction",
    "ProblemType",
    "SamplingParams",
    "ScoreResult",
    "TrajectoryEvent",
]

"""Benchmark annotation -> evaluation-native :class:`EvalCase`.

The loader is model-agnostic and content-level: the rendered prompt is
plain instruction text (no chat-template markers), and media are carried as
generic content parts (``{"type":"video","video": path}`` etc.). The generate
function maps those to a concrete API payload.

Prompt rendering and media assembly remain model-agnostic. Runtime sampling,
generation, and scoring policy deliberately do not enter this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..types import EvalCase, ProblemType
from .registry import DatasetSpec, resolve_sample_path

# Model-agnostic instruction suffixes that keep outputs parseable by SCORERS.
_INSTRUCTION_BY_TYPE: dict[ProblemType, str] = {
    ProblemType.MULTIPLE_CHOICE: ("\nAnswer with only the option letter (A, B, C, ...) inside <answer></answer> tags."),
    ProblemType.NUMERICAL: "\nProvide the numerical value inside <answer></answer> tags.",
    ProblemType.REGRESSION: "\nProvide the numerical value inside <answer></answer> tags.",
    ProblemType.REC: ("\nProvide the total count as a single number inside <answer></answer> tags."),
    ProblemType.SSR: '\nAnswer only "Yes" or "No" inside <answer></answer> tags.',
    ProblemType.CRR: '\nAnswer only "Yes" or "No" inside <answer></answer> tags.',
    ProblemType.TEMPORAL_GROUNDING: (
        "\nProvide the time span in seconds as JSON inside <answer></answer> tags, "
        'e.g. <answer>{"time": [12.3, 25.7]}</answer>.'
    ),
    ProblemType.SPATIAL_TEMPORAL_GROUNDING: (
        "\nProvide the time span and one bounding box per integer second as JSON inside "
        '<answer></answer> tags, e.g. <answer>{"time": [8, 13], "boxes": {"8": [x1,y1,x2,y2]}}</answer>.'
    ),
    ProblemType.OPEN_ENDED: "\nProvide your answer inside <answer></answer> tags.",
}

_MEDIA_KEYS = ("image", "video", "audio")


@dataclass
class BuildContext:
    """Case-building settings resolved for one benchmark."""

    spec: DatasetSpec
    prompt: dict[str, Any] = field(default_factory=dict)  # {variant, template, audio}
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def audio_enabled(self) -> bool:
        value = self.prompt.get("audio", "on")
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() != "off"


def _normalize_problem_type(value: Any) -> ProblemType:
    if isinstance(value, ProblemType):
        return value
    return ProblemType(str(value))


def _looks_like_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:", "file:"))


def _case_uid(annotation: Mapping[str, Any], index: int) -> str:
    for key in ("problem_id", "uid", "id"):
        value = annotation.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(index)


def resolve_media(spec: DatasetSpec, kind: str, raw_path: str) -> str:
    """Resolve a media reference to an absolute path (or pass URIs through)."""
    value = str(raw_path).strip()
    if _looks_like_uri(value):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str(resolve_sample_path(spec, {"path": value, "data_type": kind}))


def render_prompt(sample: Mapping[str, Any], prompt_cfg: Mapping[str, Any] | None = None) -> str:
    """Build the content-level instruction text for one sample."""
    prompt_cfg = dict(prompt_cfg or {})
    problem_type = _normalize_problem_type(sample.get("problem_type"))
    question = str(sample.get("problem") or sample.get("question") or "").strip()

    options = sample.get("options") or ()
    if problem_type == ProblemType.MULTIPLE_CHOICE and options:
        option_lines = [str(option) for option in options]
        if not all(option in question for option in option_lines):
            question = f"{question}\n" + "\n".join(option_lines)

    template_file = prompt_cfg.get("template")
    if template_file:
        text = Path(str(template_file)).expanduser().read_text(encoding="utf-8")
        return text.format(
            question=question,
            problem=str(sample.get("problem", "")).strip(),
            problem_type=problem_type.value,
        )

    suffix = _INSTRUCTION_BY_TYPE.get(problem_type, _INSTRUCTION_BY_TYPE[ProblemType.OPEN_ENDED])
    return question + suffix


def build_content(
    sample: Mapping[str, Any],
    spec: DatasetSpec,
    prompt_text: str,
    *,
    audio_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Assemble ordered content-level parts: media first, then instruction text.

    Handles the common annotation field variants: single ``data_type``+``path``,
    ``image_path``, sidecar ``images``/``videos``/``audios`` lists, OmniGAIA
    ``omni_modal_input``, and OmniClean ``media_paths`` dict.
    """
    parts: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, raw_path: Any) -> None:
        if kind == "audio" and not audio_enabled:
            return
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        resolved = resolve_media(spec, kind, raw_path)
        key = (kind, resolved)
        if key in seen:
            return
        seen.add(key)
        parts.append({"type": kind, kind: resolved})

    # OmniGAIA: omni_modal_input = [{type, path}, ...]
    for item in sample.get("omni_modal_input") or []:
        if isinstance(item, Mapping):
            kind = str(item.get("type", "")).strip().lower()
            if kind in _MEDIA_KEYS:
                add(kind, item.get("path"))

    # OmniClean: media_paths = {image: [...], audio: [...], video: [...]}
    media_paths = sample.get("media_paths")
    if isinstance(media_paths, Mapping):
        for kind in _MEDIA_KEYS:
            for raw in media_paths.get(kind) or []:
                add(kind, raw)

    # Standard single-media: data_type + path
    data_type = str(sample.get("data_type", "")).strip().lower()
    if data_type in _MEDIA_KEYS and sample.get("path"):
        add(data_type, str(sample["path"]))
    if sample.get("image_path"):
        add("image", str(sample["image_path"]))

    # Generic sidecar lists
    for kind in _MEDIA_KEYS:
        for raw in sample.get(f"{kind}s") or []:
            if isinstance(raw, str):
                add(kind, raw)
            elif isinstance(raw, Mapping):
                add(kind, raw.get(kind) or raw.get("path") or raw.get("url"))

    # Daily-Omni style separate audio sidecar
    if sample.get("audio_path"):
        add("audio", str(sample["audio_path"]))

    parts.append({"type": "text", "text": prompt_text})
    return parts


class BenchLoader:
    """Default loader: annotation mapping -> :class:`EvalCase`."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    def build_case(self, annotation: Mapping[str, Any], ctx: BuildContext, index: int) -> EvalCase:
        problem_type = _normalize_problem_type(annotation.get("problem_type"))
        uid = _case_uid(annotation, index)
        solution = annotation.get("solution", "")
        options = tuple(str(o) for o in (annotation.get("options") or ()))

        prompt_text = render_prompt(annotation, ctx.prompt)
        content = build_content(annotation, self.spec, prompt_text, audio_enabled=ctx.audio_enabled)
        messages: list[dict[str, Any]] = []
        system_prompt = str(annotation.get("system_prompt") or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        facets = dict(annotation.get("facets") or {})
        # Preserve the raw annotation for benchmark-specific scorers and agents.
        metadata: dict[str, Any] = dict(annotation)
        metadata.update(
            {
                "options": list(options),
                "budget": dict(ctx.budget),
                "audio_enabled": ctx.audio_enabled,
            }
        )
        return EvalCase(
            uid=uid,
            dataset=self.spec.name,
            index=index,
            messages=tuple(messages),
            target=solution,
            problem_type=problem_type,
            metadata=metadata,
            facets=facets,
        )


# Special formats are handled inline by build_content via field detection, so a
# single loader covers dailyomni / omnigaia / omniclean. The map exists so a
# dataset can register a dedicated subclass if it ever needs distinct logic.
_LOADER_BY_DATASET: dict[str, type[BenchLoader]] = {}


def get_loader(spec: DatasetSpec) -> BenchLoader:
    cls = _LOADER_BY_DATASET.get(spec.name, BenchLoader)
    return cls(spec)


__all__ = [
    "BenchLoader",
    "BuildContext",
    "get_loader",
    "render_prompt",
    "build_content",
    "resolve_media",
]

"""Qwen3-Omni native-media generation with explicit local staging."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from evalkit.types import (
    EvalCase,
    GenerationResult,
    GenerationStatus,
    GenerationTiming,
    SamplingParams,
    TrajectoryEvent,
)
from examples.qwen3_omni_instruct.media import (
    local_media_path,
    media_cache_dir,
    stage_source,
)

_MEDIA_FIELDS = {"image": "image", "video": "video", "audio": "audio"}


async def _stage_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[TrajectoryEvent, ...]]:
    media_cache_dir()
    transformed: list[dict[str, Any]] = []
    events: list[TrajectoryEvent] = []
    for message in messages:
        copied = dict(message)
        content = message.get("content")
        if isinstance(content, str) or content is None:
            transformed.append(copied)
            continue
        parts: list[dict[str, Any]] = []
        for raw_part in content:
            part = dict(raw_part)
            kind = str(part.get("type", "")).lower()
            field = _MEDIA_FIELDS.get(kind)
            if field is not None:
                raw = str(part[field])
                if not raw.startswith(("http://", "https://", "data:")):
                    source = local_media_path(raw)
                    staged = await stage_source(source)
                    part[field] = str(staged)
                    if staged != source:
                        events.append(
                            TrajectoryEvent(
                                "media_staging",
                                0,
                                {
                                    "kind": kind,
                                    "source": str(source),
                                    "staged_path": str(staged),
                                    "source_bytes": source.stat().st_size,
                                },
                            )
                        )
            parts.append(part)
        copied["content"] = parts
        transformed.append(copied)
    return transformed, tuple(events)


async def generate(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    started = time.monotonic()
    messages, events = await _stage_messages(case.messages)
    preprocessing_ms = (time.monotonic() - started) * 1000.0
    chat = await runtime.chat(
        "candidate",
        messages,
        sampling=sampling,
        route_key=case.uid,
    )
    status = (
        GenerationStatus.TRUNCATED
        if str(chat.finish_reason).lower() in {"length", "max_tokens"}
        else GenerationStatus.COMPLETED
    )
    return GenerationResult(
        status=status,
        response=chat.text,
        assistant_message=chat.message,
        trajectory=events,
        usage=chat.usage,
        timing=GenerationTiming(
            latency_ms=chat.timing.latency_ms,
            non_generation_ms=preprocessing_ms,
        ),
        endpoint=chat.endpoint,
        finish_reason=chat.finish_reason,
    )


__all__ = ["generate"]

"""Resource limits for the video-QA agent."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_CACHE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class AgentPolicy:
    media_cache_id: str = ""
    initial_fps: float = 0.5
    initial_max_frames: int = 256
    initial_frames: int = 256
    initial_max_pixels: int = 224 * 224
    focused_max_side: int = 384
    max_frames_per_call: int = 256
    initial_audio_max_seconds: int = 1_800
    max_turns: int = 32
    max_invalid_retries: int = 3
    max_context_tokens: int = 65_536
    default_max_output_tokens: int = 4_096
    decoder_concurrency: int = 16
    decode_batch_size: int = 16
    jpeg_quality: int = 90

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> AgentPolicy:
        budget = dict(metadata.get("budget") or {})
        cache_id = str(budget.get("media_cache_id") or "").strip()
        if not cache_id:
            raise ValueError("defaults.budget.media_cache_id is required for the video-QA agent")
        if _CACHE_ID_RE.fullmatch(cache_id) is None:
            raise ValueError("media_cache_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        initial_frames = budget.get("initial_frames", cls.initial_frames)
        if type(initial_frames) is not int or not 1 <= initial_frames <= cls.initial_max_frames:
            raise ValueError(f"initial_frames must be an integer between 1 and {cls.initial_max_frames}")

        def positive_integer(name: str, default: int) -> int:
            value = budget.get(name, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"agent policy {name} must be a positive integer")
            return value

        return cls(
            media_cache_id=cache_id,
            initial_frames=initial_frames,
            max_turns=positive_integer("max_turns", cls.max_turns),
            max_invalid_retries=positive_integer("max_invalid_retries", cls.max_invalid_retries),
            max_context_tokens=positive_integer("max_context_tokens", cls.max_context_tokens),
            default_max_output_tokens=positive_integer("max_output_tokens", cls.default_max_output_tokens),
            initial_audio_max_seconds=positive_integer("initial_audio_max_seconds", cls.initial_audio_max_seconds),
        )


__all__ = ["AgentPolicy"]

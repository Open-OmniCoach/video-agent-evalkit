"""Qwen3-Omni long-video request policy.

Run: python tests/test_qwen3_omni_long_video.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.types import (
    ChatResult,
    EvalCase,
    GenerationTiming,
    GenerationUsage,
    ProblemType,
)
from examples.qwen3_omni_instruct import long_video, media


class FakeRuntime:
    def __init__(self) -> None:
        self.request_kwargs: dict[str, Any] = {}

    async def chat(
        self,
        model_name: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> ChatResult:
        assert model_name == "candidate"
        self.request_kwargs = dict(kwargs)
        return ChatResult(
            message={"role": "assistant", "content": "A"},
            text="A",
            finish_reason="stop",
            usage=GenerationUsage(prompt_tokens=10, completion_tokens=1),
            timing=GenerationTiming(latency_ms=1),
            endpoint="http://fake/v1/chat/completions",
        )


async def _exercise(video: Path) -> None:
    case = EvalCase(
        uid="long-video-1",
        dataset="fixture",
        index=0,
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video)},
                    {"type": "text", "text": "Choose the answer."},
                ],
            },
        ),
        target="A",
        problem_type=ProblemType.MULTIPLE_CHOICE,
        metadata={"video_duration": "00:10"},
    )
    runtime = FakeRuntime()
    original_video_info = long_video._video_info

    async def fake_video_info(path: Path) -> long_video.VideoInfo:
        assert path.is_file(), path
        return long_video.VideoInfo(
            duration=10,
            height=720,
            width=1280,
            audio_channels=2,
        )

    long_video._video_info = fake_video_info
    try:
        await long_video.generate(runtime, case, {"temperature": 0})
    finally:
        long_video._video_info = original_video_info

    body = runtime.request_kwargs["request_body"]
    assert body["media_io_kwargs"]["video"] == {
        "fps": 1,
        "num_frames": -1,
        "max_frames": 256,
        "backend": "pyav",
    }
    assert body["mm_processor_kwargs"]["use_audio_in_video"] is True


def _check_resize_policy() -> None:
    assert media.smart_resize(1214, 2160) == (256, 512)
    assert media.smart_resize(1080, 1920) == (288, 512)
    assert media.smart_resize(720, 1280) == (288, 512)
    output_height, output_width = media.smart_resize(1080, 1920)
    assert output_height % 32 == 0
    assert output_width % 32 == 0
    assert output_height * output_width <= 384 * 384
    assert long_video._audio_filter(10, 2) == (
        "atrim=duration=10.000,asetpts=PTS-STARTPTS,pan=mono|c0=0.500000000000*c0+0.500000000000*c1,aresample=16000"
    )
    capped = long_video._request_body(use_audio_in_video=True, fps=0.5)
    assert capped["media_io_kwargs"]["video"]["fps"] == 0.5
    assert capped["mm_processor_kwargs"]["fps"] == 0.5


def main() -> int:
    _check_resize_policy()
    previous_cache = os.environ.get(media.CACHE_ENV)
    try:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ[media.CACHE_ENV] = str(root / "cache")
            video = root / "fixture.mp4"
            video.touch()
            asyncio.run(_exercise(video))
    finally:
        if previous_cache is None:
            os.environ.pop(media.CACHE_ENV, None)
        else:
            os.environ[media.CACHE_ENV] = previous_cache
    print("Qwen3-Omni long-video request policy OK")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Qwen3-Omni generation for bounded, audio-bearing long videos."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from weakref import WeakKeyDictionary

from evalkit.environment import host_tool_command
from evalkit.types import (
    EvalCase,
    GenerationResult,
    GenerationStatus,
    GenerationTiming,
    SamplingParams,
    TrajectoryEvent,
)
from examples.qwen3_omni_instruct.media import (
    MAX_PIXELS,
    MIN_PIXELS,
    RESIZE_FACTOR,
    media_cache_dir,
    smart_resize,
    stage_source,
)

SAMPLE_FPS = 1
MAX_FRAMES = 256
AUDIO_SAMPLE_RATE = 16_000
DEFAULT_MAX_MEDIA_SECONDS = 3_500.0

_PROXY_GATES: WeakKeyDictionary[Any, asyncio.Semaphore] = WeakKeyDictionary()
_PROXY_TASKS: WeakKeyDictionary[Any, dict[str, asyncio.Task[Path]]] = WeakKeyDictionary()
_VIDEO_INFO_TASKS: WeakKeyDictionary[Any, dict[str, asyncio.Task[VideoInfo]]] = WeakKeyDictionary()


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    height: int
    width: int
    audio_channels: int | None


def _video_path(messages: Sequence[Mapping[str, Any]]) -> Path:
    paths: list[Path] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) or content is None:
            continue
        for part in content:
            if str(part.get("type", "")).lower() == "video":
                raw = str(part["video"])
                if raw.startswith(("http://", "https://", "data:", "file:")):
                    raise ValueError("bounded long-video generation requires a local video path")
                paths.append(Path(raw).expanduser().resolve())
    if len(paths) != 1:
        raise ValueError(f"bounded long-video generation requires one video, found {len(paths)}")
    return paths[0]


def _duration_seconds(case: EvalCase) -> float:
    for raw in (
        case.metadata.get("video_duration"),
        case.facets.get("video_duration"),
        case.metadata.get("duration"),
        case.facets.get("duration"),
    ):
        value = str(raw or "").strip().lower()
        if value.endswith("s"):
            try:
                return float(value[:-1])
            except ValueError:
                continue
        fields = value.split(":")
        if len(fields) in {2, 3}:
            try:
                numbers = [float(field) for field in fields]
            except ValueError:
                continue
            if len(numbers) == 2:
                return numbers[0] * 60 + numbers[1]
            return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    raise ValueError(f"case {case.uid} has no parseable video duration")


def _proxy_gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _PROXY_GATES.setdefault(loop, asyncio.Semaphore(8))


async def _video_info_uncached(source: Path) -> VideoInfo:
    process = await asyncio.create_subprocess_exec(
        host_tool_command("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,channels:format=duration",
        "-of",
        "json",
        str(source),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-1200:]
        raise RuntimeError(f"ffprobe failed with {process.returncode}: {detail}")
    data = json.loads(stdout)
    streams = data.get("streams") or ()
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"ffprobe found no video stream in {source}")
    width = int(video_streams[0].get("width") or 0)
    height = int(video_streams[0].get("height") or 0)
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    audio_channels = int(audio_streams[0].get("channels") or 0) if audio_streams else None
    if audio_channels == 0:
        audio_channels = None
    duration = float((data.get("format") or {}).get("duration") or 0)
    if width <= 0 or height <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"ffprobe returned invalid video metadata for {source}: duration={duration}, size={width}x{height}"
        )
    return VideoInfo(
        duration=duration,
        height=height,
        width=width,
        audio_channels=audio_channels,
    )


async def _video_info(source: Path) -> VideoInfo:
    loop = asyncio.get_running_loop()
    tasks = _VIDEO_INFO_TASKS.setdefault(loop, {})
    key = str(source)
    task = tasks.get(key)
    if task is None:
        task = asyncio.create_task(_video_info_uncached(source))
        tasks[key] = task
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except BaseException:
        if tasks.get(key) is task:
            tasks.pop(key, None)
        raise


async def _stop_subprocess(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            return
        await process.wait()


def _audio_filter(duration: float, channels: int | None) -> str:
    filters = [f"atrim=duration={duration:.3f}", "asetpts=PTS-STARTPTS"]
    if channels and channels > 1:
        weight = 1.0 / channels
        terms = "+".join(f"{weight:.12f}*c{index}" for index in range(channels))
        filters.append(f"pan=mono|c0={terms}")
    filters.append(f"aresample={AUDIO_SAMPLE_RATE}")
    return ",".join(filters)


async def _proxy_uncached(source: Path, duration: float, max_audio_seconds: float) -> Path:
    identity = (
        f"v9:{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}:"
        f"{duration:.3f}:{max_audio_seconds:.3f}:{MAX_FRAMES}:"
        f"{RESIZE_FACTOR}:{MIN_PIXELS}:{MAX_PIXELS}:vorbis-q10-16kmono-mean"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    target = media_cache_dir() / f"{source.stem}-{digest}.mkv"
    if target.is_file() and target.stat().st_size > 0:
        return target

    temporary = target.with_name(f".{target.stem}-{os.getpid()}.partial.mkv")
    sample_rate = MAX_FRAMES / duration
    source_info = await _video_info(source)
    output_height, output_width = smart_resize(source_info.height, source_info.width)
    command = (
        host_tool_command("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        (f"fps={sample_rate:.12f},scale={output_width}:{output_height}:flags=bicubic"),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-frames:v",
        str(MAX_FRAMES),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-af",
        _audio_filter(max_audio_seconds, source_info.audio_channels),
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        "libvorbis",
        "-q:a",
        "10",
        "-threads",
        "8",
        str(temporary),
    )
    process: asyncio.subprocess.Process | None = None
    try:
        async with _proxy_gate():
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
    except BaseException:
        if process is not None:
            await asyncio.shield(_stop_subprocess(process))
        temporary.unlink(missing_ok=True)
        raise
    assert process is not None
    if process.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = stderr.decode("utf-8", errors="replace")[-1200:]
        raise RuntimeError(f"ffmpeg proxy failed with {process.returncode}: {detail}")
    os.replace(temporary, target)
    return target


async def _bounded_av_proxy(source: Path, duration: float, max_audio_seconds: float) -> Path:
    loop = asyncio.get_running_loop()
    tasks = _PROXY_TASKS.setdefault(loop, {})
    key = f"{source}:{duration:.3f}:{max_audio_seconds:.3f}"
    task = tasks.get(key)
    if task is None:
        task = asyncio.create_task(_proxy_uncached(source, duration, max_audio_seconds))
        tasks[key] = task
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except BaseException:
        if tasks.get(key) is task:
            tasks.pop(key, None)
        raise


def _replace_video(
    messages: Sequence[Mapping[str, Any]],
    source: Path,
    replacement: Path,
    *,
    add_audio_sidecar: bool = False,
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = message.get("content")
        if isinstance(content, str) or content is None:
            transformed.append(copied)
            continue
        parts: list[dict[str, Any]] = []
        for raw_part in content:
            part = dict(raw_part)
            if (
                str(part.get("type", "")).lower() == "video"
                and Path(str(part["video"])).expanduser().resolve() == source
            ):
                part["video"] = str(replacement)
            parts.append(part)
            if add_audio_sidecar and part.get("video") == str(replacement):
                parts.append({"type": "audio", "audio": str(replacement)})
        copied["content"] = parts
        transformed.append(copied)
    return transformed


def _request_body(*, use_audio_in_video: bool, fps: float = SAMPLE_FPS) -> dict[str, Any]:
    video_io: dict[str, Any] = {
        "fps": fps,
        "num_frames": -1,
        "max_frames": MAX_FRAMES,
        "backend": "pyav",
    }
    return {
        "media_io_kwargs": {"video": video_io},
        "mm_processor_kwargs": {
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "fps": fps,
            "max_frames": MAX_FRAMES,
            "use_audio_in_video": use_audio_in_video,
        },
    }


async def generate(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    started = time.monotonic()
    media_cache_dir()
    source = _video_path(case.messages)
    reported_duration = _duration_seconds(case)
    budget = dict(case.metadata.get("budget") or {})
    max_media_seconds = float(budget.get("max_media_seconds", DEFAULT_MAX_MEDIA_SECONDS))

    messages = list(case.messages)
    events: list[TrajectoryEvent] = []
    media_source = await stage_source(source)
    if media_source != source:
        messages = _replace_video(messages, source, media_source)
        events.append(
            TrajectoryEvent(
                "media_staging",
                0,
                {
                    "source": str(source),
                    "staged_path": str(media_source),
                    "source_bytes": source.stat().st_size,
                },
            )
        )

    source_info = await _video_info(media_source)
    duration = source_info.duration

    uses_proxy = duration * SAMPLE_FPS > MAX_FRAMES
    truncates_audio = duration > max_media_seconds
    processor_fps = MAX_FRAMES / duration if uses_proxy else SAMPLE_FPS
    if uses_proxy:
        audio_seconds = min(duration, max_media_seconds)
        proxy = await _bounded_av_proxy(media_source, duration, audio_seconds)
        proxy_info = await _video_info(proxy)
        messages = _replace_video(
            messages,
            media_source,
            proxy,
            add_audio_sidecar=truncates_audio,
        )
        events.append(
            TrajectoryEvent(
                "audio_truncation" if truncates_audio else "video_frame_cap",
                0,
                {
                    "source": str(source),
                    "reported_duration_seconds": reported_duration,
                    "original_duration_seconds": duration,
                    "visual_range_seconds": [0.0, duration],
                    "visual_frame_count": MAX_FRAMES,
                    "visual_effective_fps": processor_fps,
                    "visual_size": [proxy_info.width, proxy_info.height],
                    "audio_range_seconds": [0.0, audio_seconds],
                    "audio_codec": "vorbis",
                    "audio_sample_rate": AUDIO_SAMPLE_RATE,
                    "audio_channels": 1,
                    "audio_quality": 10,
                    "audio_input_mode": ("separate" if truncates_audio else "embedded"),
                    "interleaving_fallback_reason": (
                        "embedded truncated audio is shorter than the full visual timeline" if truncates_audio else None
                    ),
                    "cache_path": str(proxy),
                },
            )
        )

    preprocessing_ms = (time.monotonic() - started) * 1000.0
    chat = await runtime.chat(
        "candidate",
        messages,
        sampling=sampling,
        request_body=_request_body(
            use_audio_in_video=not truncates_audio,
            fps=processor_fps,
        ),
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
        trajectory=tuple(events),
        usage=chat.usage,
        timing=GenerationTiming(
            latency_ms=chat.timing.latency_ms,
            non_generation_ms=preprocessing_ms,
        ),
        endpoint=chat.endpoint,
        finish_reason=chat.finish_reason,
    )


__all__ = ["generate"]

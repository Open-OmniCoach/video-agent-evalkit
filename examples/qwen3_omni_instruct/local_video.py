"""Qwen generation with client-controlled, representation-matched video frames."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
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
    local_media_path,
    media_cache_dir,
    smart_resize,
    split_mjpeg,
    stage_source,
)

SAMPLE_FPS = 1.0
MAX_FRAMES = 256
MIN_FRAMES = 4
TEMPORAL_PATCH_SIZE = 2
JPEG_QUALITY = 3
FFMPEG_THREADS = 4
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_CONCURRENT_DECODERS = 32

_GATES: WeakKeyDictionary[Any, asyncio.Semaphore] = WeakKeyDictionary()


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int


@dataclass(frozen=True)
class PreparedVideo:
    info: VideoInfo
    frames: tuple[bytes, ...]
    timestamps: tuple[float, ...]
    output_height: int
    output_width: int
    nominal_fps: float
    effective_fps: float
    frames_sha256: str


_PREPARE_TASKS: WeakKeyDictionary[Any, dict[str, asyncio.Task[PreparedVideo]]] = WeakKeyDictionary()


def _decoder_gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _GATES.setdefault(loop, asyncio.Semaphore(MAX_CONCURRENT_DECODERS))


async def _run(command: Sequence[str]) -> bytes:
    async with _decoder_gate():
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-1200:]
        raise RuntimeError(f"{Path(command[0]).name} exited with {process.returncode}: {detail}")
    return stdout


async def probe_video(raw_path: str) -> VideoInfo:
    path = local_media_path(raw_path)
    stdout = await _run(
        (
            host_tool_command("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        )
    )
    data = json.loads(stdout)
    streams = data.get("streams") or ()
    if not streams:
        raise ValueError(f"ffprobe found no video stream in {path}")
    duration = float((data.get("format") or {}).get("duration") or 0)
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"ffprobe returned invalid metadata for {path}: duration={duration}, width={width}, height={height}"
        )
    return VideoInfo(path=path, duration=duration, width=width, height=height)


def frame_count(duration: float, sample_fps: float = SAMPLE_FPS) -> int:
    requested = math.floor(duration * sample_fps)
    bounded = min(max(requested, MIN_FRAMES), MAX_FRAMES)
    aligned = math.floor(bounded / TEMPORAL_PATCH_SIZE) * TEMPORAL_PATCH_SIZE
    return max(TEMPORAL_PATCH_SIZE, aligned)


def effective_fps(duration: float, count: int, sample_fps: float = SAMPLE_FPS) -> float:
    capped = math.floor(duration * sample_fps) > MAX_FRAMES
    expanded = count > duration * sample_fps
    return count / duration if capped or expanded else sample_fps


async def extract_video(info: VideoInfo, sample_fps: float = SAMPLE_FPS) -> tuple[tuple[bytes, ...], int, int]:
    count = frame_count(info.duration, sample_fps)
    output_height, output_width = smart_resize(info.height, info.width)
    extraction_fps = effective_fps(info.duration, count, sample_fps)
    payload = await _run(
        (
            host_tool_command("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(FFMPEG_THREADS),
            "-i",
            str(info.path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-filter_threads",
            str(FFMPEG_THREADS),
            "-vf",
            f"fps={extraction_fps:.9f},scale={output_width}:{output_height}",
            "-frames:v",
            str(count),
            "-c:v",
            "mjpeg",
            "-threads",
            str(FFMPEG_THREADS),
            "-q:v",
            str(JPEG_QUALITY),
            "-f",
            "image2pipe",
            "pipe:1",
        )
    )
    frames = split_mjpeg(payload)
    if len(frames) != count:
        raise ValueError(f"FFmpeg returned {len(frames)} frames, expected {count}")
    oversized = [len(frame) for frame in frames if len(frame) > MAX_FRAME_BYTES]
    if oversized:
        raise ValueError(f"encoded frame exceeds {MAX_FRAME_BYTES} bytes: {max(oversized)}")
    return frames, output_height, output_width


def jpeg_data_uri(frame: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")


def jpeg_video_data_uri(frames: Sequence[bytes]) -> str:
    payload = ",".join(base64.b64encode(frame).decode("ascii") for frame in frames)
    return "data:video/jpeg;base64," + payload


def frame_timestamps(duration: float, count: int, sample_fps: float = SAMPLE_FPS) -> tuple[float, ...]:
    capped = math.floor(duration * sample_fps) > MAX_FRAMES
    if capped or count > duration * sample_fps:
        step = duration / count
        return tuple(index * step for index in range(count))
    return tuple(index / sample_fps for index in range(count))


async def _prepare_video_uncached(raw_path: str, sample_fps: float) -> PreparedVideo:
    info = await probe_video(raw_path)
    frames, output_height, output_width = await extract_video(info, sample_fps)
    timestamps = frame_timestamps(info.duration, len(frames), sample_fps)
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame)
    return PreparedVideo(
        info=info,
        frames=frames,
        timestamps=timestamps,
        output_height=output_height,
        output_width=output_width,
        nominal_fps=sample_fps,
        effective_fps=effective_fps(info.duration, len(frames), sample_fps),
        frames_sha256=digest.hexdigest(),
    )


async def prepare_video(raw_path: str, sample_fps: float = SAMPLE_FPS) -> PreparedVideo:
    """Coalesce concurrent extraction of the same video within one event loop."""
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps!r}")
    path = str(local_media_path(raw_path))
    loop = asyncio.get_running_loop()
    tasks = _PREPARE_TASKS.setdefault(loop, {})
    key = f"{path}:{sample_fps:.9f}"
    task = tasks.get(key)
    if task is None:
        task = asyncio.create_task(_prepare_video_uncached(path, sample_fps))
        tasks[key] = task

        def discard(done: asyncio.Task[PreparedVideo]) -> None:
            if tasks.get(key) is done:
                tasks.pop(key, None)

        task.add_done_callback(discard)
    return await asyncio.shield(task)


def _metadata(prepared: PreparedVideo, transport: str) -> dict[str, Any]:
    return {
        "source": str(prepared.info.path),
        "duration_seconds": round(prepared.info.duration, 6),
        "source_size": [prepared.info.width, prepared.info.height],
        "output_size": [prepared.output_width, prepared.output_height],
        "nominal_sample_fps": prepared.nominal_fps,
        "effective_sample_fps": round(prepared.effective_fps, 9),
        "frame_count": len(prepared.frames),
        "max_frames": MAX_FRAMES,
        "uniform_due_to_cap": (math.floor(prepared.info.duration * prepared.nominal_fps) > MAX_FRAMES),
        "max_pixels": MAX_PIXELS,
        "jpeg_bytes": sum(len(frame) for frame in prepared.frames),
        "frames_sha256": prepared.frames_sha256,
        "transport": transport,
        "first_timestamp_seconds": round(prepared.timestamps[0], 6),
        "last_timestamp_seconds": round(prepared.timestamps[-1], 6),
    }


async def _replace_video(
    messages: Sequence[Mapping[str, Any]],
    *,
    representation: str,
    sample_fps: float,
) -> tuple[list[dict[str, Any]], PreparedVideo, dict[str, Any]]:
    if representation not in {"video_sequence", "timestamped_images"}:
        raise ValueError(f"unknown video representation: {representation!r}")
    transformed: list[dict[str, Any]] = []
    prepared_video: PreparedVideo | None = None
    has_audio = any(
        str(part.get("type", "")).lower() == "audio"
        for message in messages
        for part in (
            message.get("content")
            if isinstance(message.get("content"), Sequence) and not isinstance(message.get("content"), str)
            else ()
        )
    )
    audio_source: Path | None = None
    for message in messages:
        copied = dict(message)
        content = message.get("content")
        if isinstance(content, str) or content is None:
            transformed.append(copied)
            continue
        parts: list[dict[str, Any]] = []
        for raw_part in content:
            part = dict(raw_part)
            if str(part.get("type", "")).lower() != "video":
                parts.append(part)
                continue
            if prepared_video is not None:
                raise ValueError("Qwen local-video example supports exactly one video per case")
            prepared_video = await prepare_video(str(part["video"]), sample_fps)
            if representation == "video_sequence":
                parts.append(
                    {
                        "type": "video",
                        "video": jpeg_video_data_uri(prepared_video.frames),
                    }
                )
            else:
                for timestamp, frame in zip(prepared_video.timestamps, prepared_video.frames, strict=True):
                    parts.append({"type": "text", "text": f"<{timestamp:.2f} seconds>"})
                    parts.append({"type": "image", "image": jpeg_data_uri(frame)})
            if not has_audio:
                audio_source = await stage_source(prepared_video.info.path)
                parts.append({"type": "audio", "audio": str(audio_source)})
        copied["content"] = parts
        transformed.append(copied)
    if prepared_video is None:
        raise ValueError("Qwen local-video generation requires one video part")
    transport = (
        "data:video/jpeg frame sequence" if representation == "video_sequence" else "timestamped data:image/jpeg parts"
    )
    metadata = _metadata(prepared_video, transport)
    metadata.update(
        {
            "audio_input_mode": "separate",
            "audio_source": str(audio_source) if audio_source is not None else "annotation",
            "audio_derived_from_video": not has_audio,
        }
    )
    return transformed, prepared_video, metadata


def _request_body(representation: str, prepared: PreparedVideo) -> dict[str, Any]:
    processor = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "use_audio_in_video": False,
    }
    if representation == "timestamped_images":
        return {"mm_processor_kwargs": processor}
    processor["fps"] = prepared.effective_fps
    return {
        "media_io_kwargs": {
            "video": {
                "num_frames": -1,
                "fps": prepared.effective_fps,
                "duration": prepared.info.duration,
                "do_sample_frames": False,
            }
        },
        "mm_processor_kwargs": processor,
    }


async def _generate(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
    *,
    representation: str,
    sample_fps: float,
) -> GenerationResult:
    started = time.monotonic()
    media_cache_dir()
    messages, prepared, metadata = await _replace_video(
        case.messages,
        representation=representation,
        sample_fps=sample_fps,
    )
    preprocessing_ms = (time.monotonic() - started) * 1000.0
    chat = await runtime.chat(
        "candidate",
        messages,
        sampling=sampling,
        request_body=_request_body(representation, prepared),
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
        trajectory=(TrajectoryEvent("media_preprocess", 0, metadata),),
        usage=chat.usage,
        timing=GenerationTiming(
            latency_ms=chat.timing.latency_ms,
            non_generation_ms=preprocessing_ms,
        ),
        endpoint=chat.endpoint,
        finish_reason=chat.finish_reason,
    )


async def generate_video_sequence(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    return await _generate(
        runtime,
        case,
        sampling,
        representation="video_sequence",
        sample_fps=SAMPLE_FPS,
    )


async def generate_timestamped_images(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    return await _generate(
        runtime,
        case,
        sampling,
        representation="timestamped_images",
        sample_fps=SAMPLE_FPS,
    )


async def generate_timestamped_images_2fps(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    return await _generate(
        runtime,
        case,
        sampling,
        representation="timestamped_images",
        sample_fps=2.0,
    )


__all__ = [
    "MAX_FRAMES",
    "MAX_PIXELS",
    "PreparedVideo",
    "SAMPLE_FPS",
    "effective_fps",
    "frame_count",
    "frame_timestamps",
    "generate_timestamped_images",
    "generate_timestamped_images_2fps",
    "generate_video_sequence",
    "jpeg_data_uri",
    "jpeg_video_data_uri",
    "prepare_video",
    "smart_resize",
]

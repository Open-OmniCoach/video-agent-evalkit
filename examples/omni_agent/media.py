"""Immutable decoded media cache and per-case tool views."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse
from weakref import WeakKeyDictionary

from evalkit.environment import host_tool_command
from evalkit.types import EvalCase

from .policy import AgentPolicy
from .protocol import ToolRequest, natural_seconds

CACHE_SCHEMA = "qwen3-omni-videoqa-fast-cache-v1"
MANIFEST_SCHEMA = "qwen3-omni-videoqa-fast-media-v1"
VISUAL_FPS = 2.0
LEVEL_FPS = {"coarse": 1.0, "detail": 2.0}
WHISPER_MIN_AUDIO_FRAMES = 400
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_T = TypeVar("_T")
_TASKS: WeakKeyDictionary[Any, dict[str, asyncio.Task[Any]]] = WeakKeyDictionary()
_BUILD_GATES: WeakKeyDictionary[Any, dict[int, asyncio.Semaphore]] = WeakKeyDictionary()


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    audio_path: Path | None
    duration: float
    width: int
    height: int
    average_fps: float
    frame_count: int
    has_audio: bool


@dataclass(frozen=True)
class MediaBundle:
    kind: str
    header: str
    parts: tuple[dict[str, Any], ...]
    manifest: Mapping[str, Any]


class MediaExecutionError(RuntimeError):
    pass


class CacheConflictError(MediaExecutionError):
    pass


def media_cache_dir() -> Path:
    raw = os.environ.get("OMNI_AGENT_MEDIA_CACHE_ROOT", "").strip()
    raw = raw or str(Path(tempfile.gettempdir()) / "evalkit-media" / "omni-agent")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_media_path(value: str) -> Path:
    if value.startswith("file:"):
        parsed = urlparse(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"non-local file URI is not supported: {value}")
        path = Path(unquote(parsed.path))
    elif "://" in value or value.startswith("data:"):
        raise ValueError(f"agent media must be a local path: {value[:80]}")
    else:
        path = Path(value).expanduser()
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"media does not exist: {resolved}")
    return resolved


def find_media_parts(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path | None]:
    videos: list[Path] = []
    audios: list[Path] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) or content is None:
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = str(part.get("type") or "").lower()
            if kind == "video":
                videos.append(local_media_path(str(part["video"])))
            elif kind == "audio":
                audios.append(local_media_path(str(part["audio"])))
    if len(videos) != 1:
        raise ValueError(f"video-QA agent requires exactly one video, found {len(videos)}")
    if len(audios) > 1:
        raise ValueError(f"video-QA agent supports at most one audio sidecar, found {len(audios)}")
    return videos[0], audios[0] if audios else None


def select_initial_timestamps(
    duration: float,
    *,
    fps: float,
    max_frames: int,
) -> tuple[float, ...]:
    """Match training: uniformly cap the global 0.5fps candidate sequence."""

    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive")
    last = max(0.0, duration - 0.2)
    step = 1.0 / fps
    count = max(1, int(math.floor(last / step)) + 1)
    candidates = [min(last, index * step) for index in range(count)]
    if len(candidates) <= max_frames:
        return tuple(candidates)
    last_index = len(candidates) - 1
    return tuple(candidates[round(index * last_index / (max_frames - 1))] for index in range(max_frames))


def _fit_max_pixels(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / float(width * height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def _fit_max_side(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max(width, height) <= max_side:
        return width, height
    scale = max_side / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _safe_component(value: str, *, label: str, max_length: int = 128) -> str:
    value = value.strip()
    if value and len(value) <= max_length and _COMPONENT_RE.fullmatch(value):
        return value
    if not value:
        raise ValueError(f"{label} must not be empty")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or label
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{readable[: max_length - 11]}-{suffix}"


def resolve_media_id(case: EvalCase, video: Path) -> str:
    raw = case.metadata.get("media_id") or case.metadata.get("video_id") or video.stem
    return _safe_component(str(raw), label="media_id")


def cache_contract(policy: AgentPolicy) -> dict[str, Any]:
    preprocess = {
        "visual": {
            "fps": VISUAL_FPS,
            "max_side": policy.focused_max_side,
            "format": "concatenated-jpeg",
            "jpeg_quality": policy.jpeg_quality,
            "end_policy": "last-fixed-grid-point-at-or-before-duration",
        },
        "initial": {
            "fps": policy.initial_fps,
            "max_frames": policy.initial_max_frames,
            "max_pixels": policy.initial_max_pixels,
            "format": "concatenated-jpeg",
        },
        "audio": {
            "codec": "pcm_s16le",
            "channels": 1,
            "sample_rate": 16_000,
            "container": "wav",
        },
        "selection": "eligible-fixed-grid-uniform-inclusive-endpoints-v1",
    }
    encoded = json.dumps(preprocess, sort_keys=True, separators=(",", ":"))
    return {
        "schema": CACHE_SCHEMA,
        "cache_id": policy.media_cache_id,
        "contract_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "preprocess": preprocess,
    }


def cache_namespace(case: EvalCase, policy: AgentPolicy) -> Path:
    bench = _safe_component(case.dataset, label="bench", max_length=96)
    return media_cache_dir() / bench / policy.media_cache_id


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheConflictError(f"invalid cache metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheConflictError(f"cache metadata must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}-{os.getpid()}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            unsupported = {errno.ENOSYS, errno.EOPNOTSUPP}
            if exc.errno not in unsupported:
                raise
            with _directory_lock(path):
                yield
            return
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _directory_lock(path: Path) -> Iterator[None]:
    """Portable fallback for shared filesystems that do not implement flock."""

    lock_dir = path.with_name(f"{path.name}.d")
    deadline = time.monotonic() + 3_600
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for media cache lock: {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        lock_dir.rmdir()


def _ensure_cache_contract(namespace: Path, contract: Mapping[str, Any]) -> None:
    namespace.mkdir(parents=True, exist_ok=True)
    (namespace / "_tmp").mkdir(exist_ok=True)
    (namespace / "_locks").mkdir(exist_ok=True)
    cache_json = namespace / "cache.json"
    with _file_lock(namespace / "_locks/cache.lock"):
        if cache_json.is_file():
            existing = _json_load(cache_json)
            if existing != dict(contract):
                raise CacheConflictError(
                    f"cache_id {contract['cache_id']!r} already has a different "
                    "preprocessing contract; choose a new media_cache_id"
                )
        else:
            _write_json_atomic(cache_json, contract)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    before = path.stat()
    sha256 = _hash_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MediaExecutionError(f"media changed while caching: {path}")
    return {
        "path": str(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": sha256,
    }


def _assert_source_unchanged(path: Path, expected: Mapping[str, Any]) -> None:
    stat = path.stat()
    if stat.st_size != int(expected["size"]) or stat.st_mtime_ns != int(expected["mtime_ns"]):
        raise MediaExecutionError(f"media changed while caching: {path}")


def _source_is_same(path: Path, expected: Mapping[str, Any]) -> bool:
    stat = path.stat()
    if (
        str(path) == str(expected.get("path"))
        and stat.st_size == int(expected.get("size") or -1)
        and stat.st_mtime_ns == int(expected.get("mtime_ns") or -1)
    ):
        return True
    return _hash_file(path) == str(expected.get("sha256") or "")


def _validate_source(manifest: Mapping[str, Any], video: Path, audio_sidecar: Path | None) -> None:
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(sources.get("video"), Mapping):
        raise CacheConflictError("cached manifest has no valid source identity")
    if not _source_is_same(video, sources["video"]):
        raise CacheConflictError(
            "media_id resolves to different video bytes; choose a different media_id or media_cache_id"
        )
    expected_sidecar = sources.get("audio_sidecar")
    if (audio_sidecar is None) != (expected_sidecar is None):
        raise CacheConflictError("media_id was cached with different audio-sidecar usage; choose a new media_cache_id")
    if audio_sidecar is not None:
        if not isinstance(expected_sidecar, Mapping) or not _source_is_same(audio_sidecar, expected_sidecar):
            raise CacheConflictError("media_id resolves to different audio bytes; choose a new media_cache_id")


def _validate_index(path: Path, value: Any, *, label: str) -> list[list[float | int]]:
    if not isinstance(value, list) or not value:
        raise CacheConflictError(f"cached {label} index is empty or invalid")
    size = path.stat().st_size if path.is_file() else -1
    if size <= 0:
        raise CacheConflictError(f"cached {label} payload is missing or empty")
    previous = -1
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 3:
            raise CacheConflictError(f"cached {label} index entry is invalid")
        offset, length = int(entry[1]), int(entry[2])
        if offset < previous or length <= 0 or offset + length > size:
            raise CacheConflictError(f"cached {label} index points outside its payload")
        previous = offset
    return value


def _load_reusable_cache(
    media_dir: Path,
    *,
    contract: Mapping[str, Any],
    media_id: str,
    video: Path,
    audio_sidecar: Path | None,
) -> dict[str, Any] | None:
    manifest_path = media_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _json_load(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise CacheConflictError(f"unsupported media manifest schema in {media_dir}")
    if manifest.get("cache_id") != contract["cache_id"]:
        raise CacheConflictError(f"cached media has the wrong cache_id: {media_dir}")
    if manifest.get("contract_hash") != contract["contract_hash"]:
        raise CacheConflictError(f"cached media contract differs for {media_id}; choose a new media_cache_id")
    if manifest.get("media_id") != media_id:
        raise CacheConflictError(f"cached media_id differs in {media_dir}")
    visual = manifest.get("visual")
    initial = manifest.get("initial")
    if not isinstance(visual, Mapping) or not isinstance(initial, Mapping):
        raise CacheConflictError(f"cached visual metadata is invalid in {media_dir}")
    _validate_index(media_dir / "frames.mjpeg", visual.get("index"), label="visual")
    _validate_index(media_dir / "initial.mjpeg", initial.get("index"), label="initial")
    audio = manifest.get("audio")
    if audio is not None:
        if not isinstance(audio, Mapping):
            raise CacheConflictError(f"cached audio metadata is invalid in {media_dir}")
        audio_path = media_dir / "audio.wav"
        if not audio_path.is_file() or audio_path.stat().st_size <= 44:
            raise CacheConflictError(f"cached audio is missing or empty in {media_dir}")
    _validate_source(manifest, video, audio_sidecar)
    return manifest


def _probe_video_sync(path: Path, audio_sidecar: Path | None) -> VideoInfo:
    command = [
        host_tool_command("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise MediaExecutionError(f"could not execute ffprobe: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1200:]
        raise MediaExecutionError(f"ffprobe failed with {completed.returncode}: {detail}")
    data = json.loads(completed.stdout)
    streams = list(data.get("streams") or ())
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not videos:
        raise MediaExecutionError(f"ffprobe found no video stream in {path}")
    stream = videos[0]
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    rate = str(stream.get("avg_frame_rate") or "0")
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        average_fps = float(numerator) / float(denominator) if float(denominator) else 0
    else:
        average_fps = float(rate or 0)
    duration = float((data.get("format") or {}).get("duration") or 0)
    frame_count = int(stream.get("nb_frames") or 0)
    if frame_count <= 0 and average_fps > 0 and duration > 0:
        frame_count = max(1, int(round(duration * average_fps)))
    embedded_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    if (
        width <= 0
        or height <= 0
        or average_fps <= 0
        or frame_count <= 0
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise MediaExecutionError(
            f"invalid video metadata: duration={duration}, fps={average_fps}, "
            f"size={width}x{height}, frames={frame_count}"
        )
    return VideoInfo(
        path=path,
        audio_path=audio_sidecar or (path if embedded_audio else None),
        duration=duration,
        width=width,
        height=height,
        average_fps=average_fps,
        frame_count=frame_count,
        has_audio=bool(audio_sidecar or embedded_audio),
    )


def _visual_timestamps(duration: float) -> tuple[float, ...]:
    count = max(1, int(math.floor(max(0.0, duration) * VISUAL_FPS)) + 1)
    return tuple(index / VISUAL_FPS for index in range(count))


def _append_jpeg(handle, image, *, quality: int) -> list[int]:
    offset = handle.tell()
    image.save(handle, format="JPEG", quality=quality)
    return [offset, handle.tell() - offset]


def _build_visual_cache_decord(
    info: VideoInfo, output_dir: Path, policy: AgentPolicy
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from decord import VideoReader, cpu
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("agent cache building requires `uv sync --extra media`") from exc

    visual_width, visual_height = _fit_max_side(info.width, info.height, policy.focused_max_side)
    initial_width, initial_height = _fit_max_pixels(info.width, info.height, policy.initial_max_pixels)
    visual_timestamps = _visual_timestamps(info.duration)
    initial_timestamps = select_initial_timestamps(
        info.duration,
        fps=policy.initial_fps,
        max_frames=policy.initial_max_frames,
    )
    initial_indices = {int(round(value * VISUAL_FPS)) for value in initial_timestamps}
    reader = VideoReader(
        str(info.path),
        ctx=cpu(0),
        num_threads=2,
        width=visual_width,
        height=visual_height,
    )
    frame_indices = [
        min(info.frame_count - 1, max(0, int(round(value * info.average_fps)))) for value in visual_timestamps
    ]
    visual_index: list[list[float | int]] = []
    initial_index: list[list[float | int]] = []
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with (
        (output_dir / "frames.mjpeg").open("wb") as visual_handle,
        (output_dir / "initial.mjpeg").open("wb") as initial_handle,
    ):
        for batch_start in range(0, len(frame_indices), policy.decode_batch_size):
            batch = frame_indices[batch_start : batch_start + policy.decode_batch_size]
            arrays = reader.get_batch(batch).asnumpy()
            for offset, array in enumerate(arrays, start=batch_start):
                timestamp = visual_timestamps[offset]
                image = Image.fromarray(array).convert("RGB")
                location = _append_jpeg(visual_handle, image, quality=policy.jpeg_quality)
                visual_index.append([round(timestamp, 6), *location])
                if offset in initial_indices:
                    initial_image = image.resize((initial_width, initial_height), resampling)
                    initial_location = _append_jpeg(
                        initial_handle,
                        initial_image,
                        quality=policy.jpeg_quality,
                    )
                    initial_index.append([round(timestamp, 6), *initial_location])
    if len(initial_index) != len(initial_timestamps):
        raise MediaExecutionError("initial cache selection did not map exactly onto the fixed 2fps grid")
    return (
        {
            "file": "frames.mjpeg",
            "fps": VISUAL_FPS,
            "width": visual_width,
            "height": visual_height,
            "index": visual_index,
        },
        {
            "file": "initial.mjpeg",
            "fps": policy.initial_fps,
            "max_frames": policy.initial_max_frames,
            "max_pixels": policy.initial_max_pixels,
            "width": initial_width,
            "height": initial_height,
            "index": initial_index,
        },
    )


def _read_exact(stream, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if not chunks:
        return None
    payload = b"".join(chunks)
    if len(payload) != size:
        raise MediaExecutionError(f"ffmpeg produced a partial RGB frame: {len(payload)}/{size} bytes")
    return payload


def _build_visual_cache_ffmpeg(
    info: VideoInfo, output_dir: Path, policy: AgentPolicy
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode the fixed 2fps grid when Decord lacks the source codec (for example AV1)."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("agent cache building requires `uv sync --extra media`") from exc

    visual_width, visual_height = _fit_max_side(info.width, info.height, policy.focused_max_side)
    initial_width, initial_height = _fit_max_pixels(info.width, info.height, policy.initial_max_pixels)
    visual_timestamps = _visual_timestamps(info.duration)
    initial_timestamps = select_initial_timestamps(
        info.duration,
        fps=policy.initial_fps,
        max_frames=policy.initial_max_frames,
    )
    initial_indices = {int(round(value * VISUAL_FPS)) for value in initial_timestamps}
    command = [
        host_tool_command("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(info.path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        (f"fps=fps={VISUAL_FPS}:start_time=0:round=near,scale={visual_width}:{visual_height}:flags=lanczos"),
        "-frames:v",
        str(len(visual_timestamps)),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    visual_index: list[list[float | int]] = []
    initial_index: list[list[float | int]] = []
    frame_bytes = visual_width * visual_height * 3
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    process: subprocess.Popen[bytes] | None = None
    last_payload: bytes | None = None
    with (
        tempfile.TemporaryFile() as stderr,
        (output_dir / "frames.mjpeg").open("wb") as visual_handle,
        (output_dir / "initial.mjpeg").open("wb") as initial_handle,
    ):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr,
            )
            assert process.stdout is not None
            for offset, timestamp in enumerate(visual_timestamps):
                payload = _read_exact(process.stdout, frame_bytes)
                if payload is None:
                    if last_payload is None:
                        raise MediaExecutionError("ffmpeg produced no visual frames")
                    # The probed container duration may follow a slightly longer
                    # audio stream. Match the Decord path, which clamps such final
                    # grid points to the last video frame.
                    payload = last_payload
                else:
                    last_payload = payload
                image = Image.frombytes("RGB", (visual_width, visual_height), payload)
                location = _append_jpeg(visual_handle, image, quality=policy.jpeg_quality)
                visual_index.append([round(timestamp, 6), *location])
                if offset in initial_indices:
                    initial_image = image.resize((initial_width, initial_height), resampling)
                    initial_location = _append_jpeg(
                        initial_handle,
                        initial_image,
                        quality=policy.jpeg_quality,
                    )
                    initial_index.append([round(timestamp, 6), *initial_location])
            process.stdout.close()
            returncode = process.wait()
            stderr.seek(0)
            detail = stderr.read().decode("utf-8", errors="replace").strip()
            if returncode != 0:
                raise MediaExecutionError(
                    f"ffmpeg visual decode failed with {returncode}: {detail[-2000:] or 'no stderr'}"
                )
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            raise
    if len(initial_index) != len(initial_timestamps):
        raise MediaExecutionError("initial cache selection did not map exactly onto the fixed 2fps grid")
    return (
        {
            "file": "frames.mjpeg",
            "fps": VISUAL_FPS,
            "width": visual_width,
            "height": visual_height,
            "index": visual_index,
        },
        {
            "file": "initial.mjpeg",
            "fps": policy.initial_fps,
            "max_frames": policy.initial_max_frames,
            "max_pixels": policy.initial_max_pixels,
            "width": initial_width,
            "height": initial_height,
            "index": initial_index,
        },
    )


def _build_visual_cache(
    info: VideoInfo, output_dir: Path, policy: AgentPolicy
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _build_visual_cache_decord(info, output_dir, policy)
    except Exception as decord_error:
        try:
            return _build_visual_cache_ffmpeg(info, output_dir, policy)
        except Exception as ffmpeg_error:
            raise MediaExecutionError(
                "both visual decoders failed: "
                f"decord={type(decord_error).__name__}: {decord_error}; "
                f"ffmpeg={type(ffmpeg_error).__name__}: {ffmpeg_error}"
            ) from ffmpeg_error


def _build_audio_cache(source: Path, output_dir: Path) -> dict[str, Any]:
    target = output_dir / "audio.wav"
    command = [
        host_tool_command("ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise MediaExecutionError(f"could not execute ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1200:]
        raise MediaExecutionError(f"ffmpeg audio extraction failed with {completed.returncode}: {detail}")
    if not target.is_file() or target.stat().st_size <= 44:
        raise MediaExecutionError("ffmpeg produced an empty audio cache")
    with wave.open(str(target), "rb") as handle:
        frames = handle.getnframes()
        if handle.getnchannels() != 1 or handle.getframerate() != 16_000 or handle.getsampwidth() != 2:
            raise MediaExecutionError("ffmpeg produced an unexpected WAV format")
    return {
        "file": "audio.wav",
        "codec": "pcm_s16le",
        "channels": 1,
        "sample_rate": 16_000,
        "sample_width": 2,
        "frames": frames,
    }


def _build_media_cache(
    target: Path,
    *,
    contract: Mapping[str, Any],
    policy: AgentPolicy,
    media_id: str,
    video: Path,
    audio_sidecar: Path | None,
) -> dict[str, Any]:
    parent = target.parent
    temporary = parent / "_tmp" / f".{media_id}-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        video_source = _source_record(video)
        sidecar_source = _source_record(audio_sidecar) if audio_sidecar is not None else None
        info = _probe_video_sync(video, audio_sidecar)
        visual, initial = _build_visual_cache(
            info,
            temporary,
            policy,
        )
        audio = _build_audio_cache(info.audio_path, temporary) if info.audio_path is not None else None
        _assert_source_unchanged(video, video_source)
        if audio_sidecar is not None and sidecar_source is not None:
            _assert_source_unchanged(audio_sidecar, sidecar_source)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "cache_id": contract["cache_id"],
            "contract_hash": contract["contract_hash"],
            "media_id": media_id,
            "sources": {
                "video": video_source,
                "audio_sidecar": sidecar_source,
            },
            "video": {
                "duration": info.duration,
                "width": info.width,
                "height": info.height,
                "average_fps": info.average_fps,
                "frame_count": info.frame_count,
                "has_audio": info.has_audio,
            },
            "visual": visual,
            "initial": initial,
            "audio": audio,
        }
        _write_json_atomic(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _ensure_media_cache_sync(
    namespace: Path,
    *,
    contract: Mapping[str, Any],
    policy: AgentPolicy,
    media_id: str,
    video: Path,
    audio_sidecar: Path | None,
) -> tuple[dict[str, Any], bool]:
    _ensure_cache_contract(namespace, contract)
    target = namespace / media_id
    with _file_lock(namespace / "_locks" / f"{media_id}.lock"):
        cached = _load_reusable_cache(
            target,
            contract=contract,
            media_id=media_id,
            video=video,
            audio_sidecar=audio_sidecar,
        )
        if cached is not None:
            return cached, True
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        built = _build_media_cache(
            target,
            contract=contract,
            policy=policy,
            media_id=media_id,
            video=video,
            audio_sidecar=audio_sidecar,
        )
        return built, False


def _gate(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _BUILD_GATES.setdefault(loop, {}).setdefault(limit, asyncio.Semaphore(limit))


def _quick_source_key(path: Path | None) -> tuple[str, int, int] | None:
    if path is None:
        return None
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


async def _coalesced(key: str, factory: Callable[[], Awaitable[_T]]) -> _T:
    loop = asyncio.get_running_loop()
    tasks = _TASKS.setdefault(loop, {})
    task = tasks.get(key)
    if task is None:
        task = asyncio.create_task(factory())
        tasks[key] = task

        def discard(done: asyncio.Task[Any]) -> None:
            if tasks.get(key) is done:
                tasks.pop(key, None)

        task.add_done_callback(discard)
    return await asyncio.shield(task)


def _info_from_manifest(manifest: Mapping[str, Any], video: Path, media_dir: Path) -> VideoInfo:
    value = manifest["video"]
    assert isinstance(value, Mapping)
    audio_path = media_dir / "audio.wav" if manifest.get("audio") is not None else None
    return VideoInfo(
        path=video,
        audio_path=audio_path,
        duration=float(value["duration"]),
        width=int(value["width"]),
        height=int(value["height"]),
        average_fps=float(value["average_fps"]),
        frame_count=int(value["frame_count"]),
        has_audio=audio_path is not None,
    )


def _uniform_subset(values: Sequence[_T], count: int) -> tuple[_T, ...]:
    if len(values) <= count:
        return tuple(values)
    if count <= 1:
        return (values[0],)
    last = len(values) - 1
    return tuple(values[round(index * last / (count - 1))] for index in range(count))


def _select_entries(
    entries: Sequence[Sequence[float | int]], request: ToolRequest
) -> tuple[Sequence[float | int], ...]:
    assert request.level is not None
    eligible = [
        entry
        for entry in entries
        if request.start - 1e-6 <= float(entry[0]) <= request.end + 1e-6
        and (request.level == "detail" or int(round(float(entry[0]) * VISUAL_FPS)) % 2 == 0)
    ]
    if not eligible:
        raise MediaExecutionError(f"no cached frames overlap [{request.start_code},{request.end_code}]")
    desired = max(
        1,
        math.ceil((request.end - request.start) * LEVEL_FPS[request.level] - 1e-9),
    )
    return _uniform_subset(eligible, desired)


def _materialize_jpegs(
    source: Path,
    entries: Sequence[Sequence[float | int]],
    output_dir: Path,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with source.open("rb") as handle:
        for index, entry in enumerate(entries, start=1):
            timestamp, offset, length = float(entry[0]), int(entry[1]), int(entry[2])
            handle.seek(offset)
            payload = handle.read(length)
            if len(payload) != length:
                raise MediaExecutionError("short read from cached MJPEG payload")
            target = output_dir / f"frame_{index:04d}_{timestamp:.3f}.jpg"
            target.write_bytes(payload)
            paths.append(target)
    return tuple(paths)


def _slice_wav(source: Path, start: float, end: float, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        start_frame = min(reader.getnframes(), max(0, round(start * rate)))
        end_frame = min(reader.getnframes(), max(start_frame, round(end * rate)))
        reader.setpos(start_frame)
        payload = reader.readframes(end_frame - start_frame)
        with wave.open(str(target), "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(payload)
    if target.stat().st_size <= 44:
        raise MediaExecutionError("requested audio cache slice is empty")
    return target


def _timestamped_image_parts(
    entries: Sequence[Sequence[float | int]], paths: Sequence[Path]
) -> tuple[dict[str, Any], ...]:
    parts: list[dict[str, Any]] = []
    for entry, path in zip(entries, paths, strict=True):
        parts.append({"type": "text", "text": f"<{float(entry[0]):.1f} seconds>"})
        parts.append({"type": "image", "image": str(path)})
    return tuple(parts)


def ensure_whisper_audio_minimum(path: str | Path) -> Path:
    """Zero-pad a tiny valid WAV so Whisper's reflect-padding STFT can read it."""

    target = Path(path)
    with wave.open(str(target), "rb") as reader:
        params = reader.getparams()
        frame_count = reader.getnframes()
        payload = reader.readframes(frame_count)
    if frame_count >= WHISPER_MIN_AUDIO_FRAMES:
        return target
    missing_bytes = (WHISPER_MIN_AUDIO_FRAMES - frame_count) * params.nchannels * params.sampwidth
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with wave.open(str(temporary), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(payload + bytes(missing_bytes))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _concat_wavs_sync(paths: list[Path], target: Path) -> Path:
    """Concatenate same-source PCM spans for SGLang's one-audio mRoPE path."""
    if not paths:
        raise ValueError("at least one WAV is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    params = None
    payloads: list[bytes] = []
    for path in paths:
        with wave.open(str(path), "rb") as reader:
            current = reader.getparams()
            comparable = (
                current.nchannels,
                current.sampwidth,
                current.framerate,
                current.comptype,
            )
            if params is None:
                params = current
                expected = comparable
            elif comparable != expected:
                raise MediaExecutionError("ordered multi-span audio segments have incompatible WAV parameters")
            payloads.append(reader.readframes(current.nframes))
    assert params is not None
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        for payload in payloads:
            writer.writeframes(payload)
    ensure_whisper_audio_minimum(target)
    return target


def _decode_high_sync(info: VideoInfo, timestamps: list[float], output: Path) -> list[Path]:
    from decord import VideoReader, cpu
    from PIL import Image

    width, height = _fit_max_side(info.width, info.height, 1280)
    reader = VideoReader(str(info.path), ctx=cpu(0), num_threads=2, width=width, height=height)
    indices = [min(len(reader) - 1, max(0, int(round(ts * info.average_fps)))) for ts in timestamps]
    arrays = reader.get_batch(indices).asnumpy()
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (timestamp, array) in enumerate(zip(timestamps, arrays, strict=True)):
        path = output / f"frame_{index + 1:04d}_{timestamp:.3f}.jpg"
        Image.fromarray(array).convert("RGB").save(path, format="JPEG", quality=90)
        paths.append(path)
    return paths


class MediaSession:
    def __init__(
        self,
        *,
        info: VideoInfo,
        policy: AgentPolicy,
        namespace: Path,
        media_id: str,
        manifest: Mapping[str, Any],
        cache_hit: bool,
    ) -> None:
        self.info = info
        self.policy = policy
        self.namespace = namespace
        self.media_id = media_id
        self.media_dir = namespace / media_id
        self.manifest = manifest
        self.cache_hit = cache_hit
        self._view_root: Path | None = None
        self._view_count = 0

    @classmethod
    async def open(
        cls,
        case: EvalCase,
        policy: AgentPolicy,
    ) -> MediaSession:
        video, audio = find_media_parts(case.messages)
        media_id = resolve_media_id(case, video)
        namespace = cache_namespace(case, policy)
        contract = cache_contract(policy)

        async def ensure() -> tuple[dict[str, Any], bool]:
            async with _gate(policy.decoder_concurrency):
                return await asyncio.to_thread(
                    _ensure_media_cache_sync,
                    namespace,
                    contract=contract,
                    policy=policy,
                    media_id=media_id,
                    video=video,
                    audio_sidecar=audio,
                )

        source_key = (_quick_source_key(video), _quick_source_key(audio))
        manifest, cache_hit = await _coalesced(f"media-cache:{namespace / media_id}:{source_key!r}", ensure)
        media_dir = namespace / media_id
        info = _info_from_manifest(manifest, video, media_dir)
        if not bool(case.metadata.get("audio_enabled", True)):
            # Keep the immutable cache reusable across audio-on/off runs, but
            # hide cached audio from both the initial bundle and every tool.
            info = replace(info, audio_path=None, has_audio=False)
        return cls(
            info=info,
            policy=policy,
            namespace=namespace,
            media_id=media_id,
            manifest=manifest,
            cache_hit=cache_hit,
        )

    def _next_view(self, label: str) -> Path:
        if self._view_root is None:
            self._view_root = self.namespace / "_tmp/cases" / f"{self.media_id}-{os.getpid()}-{uuid.uuid4().hex}"
            self._view_root.mkdir(parents=True)
        self._view_count += 1
        return self._view_root / f"{self._view_count:03d}-{label}"

    async def close(self) -> None:
        root, self._view_root = self._view_root, None
        if root is not None:
            await asyncio.to_thread(shutil.rmtree, root, True)

    async def initial_bundle(self) -> MediaBundle:
        initial = self.manifest["initial"]
        assert isinstance(initial, Mapping)
        entries = initial["index"]
        assert isinstance(entries, Sequence)
        parent_frame_count = len(entries)
        # Select the model's view without changing the shared preprocessing cache.
        indices = _uniform_subset(range(parent_frame_count), self.policy.initial_frames)
        entries = [entries[index] for index in indices]
        output = self._next_view("initial")
        paths = await asyncio.to_thread(
            _materialize_jpegs,
            self.media_dir / "initial.mjpeg",
            entries,
            output,
        )
        parts = list(_timestamped_image_parts(entries, paths))
        audio_path = self.info.audio_path
        include_initial_audio = bool(
            audio_path is not None and self.info.duration <= self.policy.initial_audio_max_seconds
        )
        if include_initial_audio:
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": f"Audio: 0 seconds-{math.ceil(self.info.duration)} seconds (continuous full audio)",
                    },
                    {"type": "audio", "audio": str(audio_path)},
                ]
            )
            audio_summary = "full"
        elif audio_path is not None:
            audio_summary = (
                "available-via-get_audio (full audio omitted from the initial request to preserve context budget)"
            )
        else:
            audio_summary = "not available"
        sampling = (
            "uniform-over-existing-initial-frames"
            if self.policy.initial_frames < self.policy.initial_max_frames
            else "uniform-over-fps-candidates"
        )
        summary = (
            f"Initial frame sampling: fps={self.policy.initial_fps}; "
            f"max_frames={self.policy.initial_frames}; selected_frames={len(paths)}; "
            f"max_pixels={self.policy.initial_max_pixels} (~224x224 area); "
            f"sampling={sampling}; "
            f"audio={audio_summary}."
        )
        return MediaBundle(
            kind="initial",
            header=summary,
            parts=tuple(parts),
            manifest={
                "kind": "initial",
                "cache_id": self.policy.media_cache_id,
                "media_id": self.media_id,
                "cache_hit": self.cache_hit,
                "frame_count": len(paths),
                "parent_frame_count": parent_frame_count,
                "selection_indices": list(indices),
                "timestamps": [round(float(entry[0]), 6) for entry in entries],
                "output_size": [int(initial["width"]), int(initial["height"])],
                "audio": include_initial_audio,
                "audio_available": audio_path is not None,
            },
        )

    async def execute(self, request: ToolRequest) -> MediaBundle:
        if request.name == "get_multi_span":
            return await self._multi_span_bundle(request)
        if request.resolution == "high" and request.name in {"get_frames", "get_clip"}:
            return await self._high_bundle(request, clip=request.name == "get_clip")
        if request.name == "get_audio":
            return await self._audio_bundle(request)
        if request.name == "get_frames":
            return await self._frames_bundle(request, clip=False)
        if request.name == "get_clip":
            return await self._frames_bundle(request, clip=True)
        raise AssertionError(f"unhandled tool {request.name}")

    async def _audio_bundle(self, request: ToolRequest) -> MediaBundle:
        if self.info.audio_path is None:
            raise MediaExecutionError("audio is unavailable")
        target = self._next_view("audio") / "audio.wav"
        path = await asyncio.to_thread(_slice_wav, self.info.audio_path, request.start, request.end, target)
        ensure_whisper_audio_minimum(path)
        header = f"Audio {natural_seconds(request.start)} to {natural_seconds(request.end)} (continuous)."
        return MediaBundle(
            kind="audio",
            header=header,
            parts=(
                {"type": "text", "text": header},
                {"type": "audio", "audio": str(path)},
            ),
            manifest={
                "kind": "audio",
                "cache_id": self.policy.media_cache_id,
                "media_id": self.media_id,
                "start": request.start,
                "end": request.end,
                "duration": request.end - request.start,
            },
        )

    async def _frames_bundle(self, request: ToolRequest, *, clip: bool) -> MediaBundle:
        assert request.level is not None
        visual = self.manifest["visual"]
        assert isinstance(visual, Mapping)
        raw_entries = visual["index"]
        assert isinstance(raw_entries, Sequence)
        entries = _select_entries(raw_entries, request)
        if len(entries) > self.policy.max_frames_per_call:
            raise MediaExecutionError(
                f"focused call needs {len(entries)} frames, maximum is {self.policy.max_frames_per_call}"
            )
        view = self._next_view("clip" if clip else "frames")
        frame_task = asyncio.create_task(
            asyncio.to_thread(
                _materialize_jpegs,
                self.media_dir / "frames.mjpeg",
                entries,
                view / "frames",
            )
        )
        audio_task: asyncio.Task[Path] | None = None
        if clip and self.info.audio_path is not None:
            audio_task = asyncio.create_task(
                asyncio.to_thread(
                    _slice_wav,
                    self.info.audio_path,
                    request.start,
                    request.end,
                    view / "audio.wav",
                )
            )
        try:
            paths = await frame_task
            audio_path = await audio_task if audio_task is not None else None
        except BaseException:
            frame_task.cancel()
            if audio_task is not None:
                audio_task.cancel()
            await asyncio.gather(
                frame_task,
                *((audio_task,) if audio_task is not None else ()),
                return_exceptions=True,
            )
            raise
        if audio_path is not None:
            ensure_whisper_audio_minimum(audio_path)

        fps = LEVEL_FPS[request.level]
        start, end = natural_seconds(request.start), natural_seconds(request.end)
        if clip:
            audio_text = "continuous same-span audio included" if audio_path is not None else "audio unavailable"
            header = f"Clip {start} to {end} (level={request.level}, decoder=decord, max_side=384; {audio_text})."
        else:
            header = (
                f"Frames {start} to {end} (level={request.level}, fps={fps:.0f}, "
                f"num={len(paths)}, decoder=decord, max_side=384)."
            )
        parts = [{"type": "text", "text": header}]
        parts.extend(_timestamped_image_parts(entries, paths))
        if audio_path is not None:
            parts.extend(
                [
                    {"type": "text", "text": f"Audio span: {start} to {end}, continuous."},
                    {"type": "audio", "audio": str(audio_path)},
                ]
            )
        return MediaBundle(
            kind="clip" if clip else "frames",
            header=header,
            parts=tuple(parts),
            manifest={
                "kind": "clip" if clip else "frames",
                "cache_id": self.policy.media_cache_id,
                "media_id": self.media_id,
                "start": request.start,
                "end": request.end,
                "level": request.level,
                "fps": fps,
                "frame_count": len(paths),
                "timestamps": [round(float(entry[0]), 6) for entry in entries],
                "output_size": [int(visual["width"]), int(visual["height"])],
                "audio": audio_path is not None,
            },
        )

    async def _high_bundle(self, request: ToolRequest, *, clip: bool) -> MediaBundle:
        visual = self.manifest["visual"]
        entries = _select_entries(visual["index"], request)
        if len(entries) > self.policy.max_frames_per_call:
            raise MediaExecutionError(f"high-resolution call requires {len(entries)} frames")
        timestamps = [float(entry[0]) for entry in entries]
        view = self._next_view("high-clip" if clip else "high-frames")
        frame_task = asyncio.create_task(asyncio.to_thread(_decode_high_sync, self.info, timestamps, view / "frames"))
        audio_task = None
        if clip and self.info.audio_path is not None:
            audio_task = asyncio.create_task(
                asyncio.to_thread(_slice_wav, self.info.audio_path, request.start, request.end, view / "audio.wav")
            )
        try:
            paths = await frame_task
            audio_path = await audio_task if audio_task is not None else None
        except BaseException:
            frame_task.cancel()
            if audio_task is not None:
                audio_task.cancel()
            await asyncio.gather(frame_task, *([audio_task] if audio_task else []), return_exceptions=True)
            raise
        start, end = natural_seconds(request.start), natural_seconds(request.end)
        header = (
            f"{'Clip' if clip else 'Frames'} {start} to {end} "
            f"(level={request.level}, resolution=high, max_side=1280"
            + ("; continuous same-span audio included)." if audio_path else ").")
        )
        parts: list[dict[str, Any]] = [{"type": "text", "text": header}]
        parts.extend(_timestamped_image_parts(entries, paths))
        if audio_path is not None:
            ensure_whisper_audio_minimum(audio_path)
            parts.extend(
                [
                    {"type": "text", "text": f"Audio span: {start} to {end}, continuous."},
                    {"type": "audio", "audio": str(audio_path)},
                ]
            )
        return MediaBundle(
            kind="clip" if clip else "frames",
            header=header,
            parts=tuple(parts),
            manifest={
                "kind": "clip" if clip else "frames",
                "cache_id": self.policy.media_cache_id,
                "media_id": self.media_id,
                "start": request.start,
                "end": request.end,
                "level": request.level,
                "resolution": "high",
                "frame_count": len(paths),
                "timestamps": timestamps,
                "output_size": list(_fit_max_side(self.info.width, self.info.height, 1280)),
                "audio": audio_path is not None,
            },
        )

    async def _multi_span_bundle(self, request: ToolRequest) -> MediaBundle:
        bundles = [await self.execute(span) for span in request.spans]
        summaries: list[str] = []
        parts: list[dict[str, Any]] = []
        audio_paths: list[Path] = []
        total_frames = 0
        total_audio = 0.0
        for index, (span, bundle) in enumerate(zip(request.spans, bundles, strict=True), start=1):
            modality = {"get_frames": "frames", "get_audio": "audio", "get_clip": "clip"}[span.name]
            summaries.append(
                f"span_{index} | {modality} | {natural_seconds(span.start)}–{natural_seconds(span.end)}"
                + (f" | {span.level} | {span.resolution}" if span.level else "")
            )
            payload = list(bundle.parts)
            if payload and payload[0].get("type") == "text":
                payload = payload[1:]
            for part in payload:
                if part.get("type") == "audio":
                    audio_paths.append(Path(str(part["audio"])))
                else:
                    parts.append(dict(part))
            total_frames += int(bundle.manifest.get("frame_count") or 0)
            total_audio += (span.end - span.start) if modality in {"audio", "clip"} and self.info.has_audio else 0
        header = "\n".join(summaries)
        if audio_paths:
            if len(audio_paths) == 1:
                combined_audio = audio_paths[0]
            else:
                combined_audio = await asyncio.to_thread(
                    _concat_wavs_sync,
                    audio_paths,
                    self._next_view("multi-audio") / "audio.wav",
                )
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Ordered audio for the audio-bearing spans above, concatenated "
                            "in listed span order without gaps."
                        ),
                    },
                    {"type": "audio", "audio": str(combined_audio)},
                ]
            )
        return MediaBundle(
            kind="multi_span",
            header=header,
            parts=({"type": "text", "text": header}, *parts),
            manifest={
                "kind": "multi_span",
                "mode": "ordered_multi_span_comparison",
                "span_count": len(request.spans),
                "total_frames": total_frames,
                "total_audio_seconds": total_audio,
                "audio_payloads": 1 if audio_paths else 0,
                "audio_segments": len(audio_paths),
                "spans": [span.identity for span in request.spans],
            },
        )


__all__ = [
    "CACHE_SCHEMA",
    "LEVEL_FPS",
    "MANIFEST_SCHEMA",
    "VISUAL_FPS",
    "WHISPER_MIN_AUDIO_FRAMES",
    "CacheConflictError",
    "MediaBundle",
    "MediaExecutionError",
    "MediaSession",
    "VideoInfo",
    "cache_contract",
    "cache_namespace",
    "ensure_whisper_audio_minimum",
    "find_media_parts",
    "local_media_path",
    "media_cache_dir",
    "resolve_media_id",
    "select_initial_timestamps",
]

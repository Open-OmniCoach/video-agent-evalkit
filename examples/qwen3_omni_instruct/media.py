"""Shared local-media primitives for the Qwen3-Omni integration."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from weakref import WeakKeyDictionary

CACHE_ENV = "QWEN3_OMNI_MEDIA_CACHE_DIR"
RESIZE_FACTOR = 32
MIN_PIXELS = 128 * 32 * 32
MAX_PIXELS = 384 * 384

_JPEG_START = b"\xff\xd8"
_JPEG_END = b"\xff\xd9"
_STAGE_GATES: WeakKeyDictionary[Any, asyncio.Semaphore] = WeakKeyDictionary()
_STAGE_TASKS: WeakKeyDictionary[Any, dict[str, asyncio.Task[Path]]] = WeakKeyDictionary()


def local_media_path(value: str) -> Path:
    if value.startswith("file:"):
        parsed = urlparse(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"non-local file URI is not supported: {value}")
        path = Path(unquote(parsed.path))
    elif "://" in value or value.startswith("data:"):
        raise ValueError(f"Qwen3-Omni local media requires a filesystem path: {value[:80]}")
    else:
        path = Path(value).expanduser()
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"media does not exist: {resolved}")
    return resolved


def split_mjpeg(payload: bytes) -> tuple[bytes, ...]:
    frames: list[bytes] = []
    cursor = 0
    while True:
        start = payload.find(_JPEG_START, cursor)
        if start < 0:
            break
        end = payload.find(_JPEG_END, start + len(_JPEG_START))
        if end < 0:
            raise ValueError("FFmpeg returned a truncated MJPEG stream")
        end += len(_JPEG_END)
        frames.append(payload[start:end])
        cursor = end
    if not frames:
        raise ValueError("FFmpeg returned no JPEG frames")
    return tuple(frames)


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Match Qwen's factor-32 area resize for video frames."""
    if height < RESIZE_FACTOR or width < RESIZE_FACTOR:
        raise ValueError(f"video dimensions must be at least {RESIZE_FACTOR}: {width}x{height}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"video aspect ratio is too extreme: {width}x{height}")

    resized_height = round(height / RESIZE_FACTOR) * RESIZE_FACTOR
    resized_width = round(width / RESIZE_FACTOR) * RESIZE_FACTOR
    if resized_height * resized_width > MAX_PIXELS:
        scale = math.sqrt((height * width) / MAX_PIXELS)
        resized_height = max(
            RESIZE_FACTOR,
            math.floor(height / scale / RESIZE_FACTOR) * RESIZE_FACTOR,
        )
        resized_width = max(
            RESIZE_FACTOR,
            math.floor(width / scale / RESIZE_FACTOR) * RESIZE_FACTOR,
        )
    elif resized_height * resized_width < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (height * width))
        resized_height = math.ceil(height * scale / RESIZE_FACTOR) * RESIZE_FACTOR
        resized_width = math.ceil(width * scale / RESIZE_FACTOR) * RESIZE_FACTOR
    if resized_height * resized_width > MAX_PIXELS:
        raise ValueError(f"factor-aligned resize exceeds max pixels: {resized_width}x{resized_height}")
    return resized_height, resized_width


def media_cache_dir() -> Path:
    raw = os.environ.get(CACHE_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{CACHE_ENV} is required when media must be cached")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _STAGE_GATES.setdefault(loop, asyncio.Semaphore(16))


def _copy_source(source: Path, target: Path, temporary: Path) -> None:
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise OSError(f"staged media size mismatch for {source}")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


async def _stage_uncached(source: Path) -> Path:
    stat = source.stat()
    identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    stage_dir = media_cache_dir() / "sources"
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / f"{source.stem}-{digest}{source.suffix.lower()}"
    if target.is_file() and target.stat().st_size == stat.st_size:
        return target

    temporary = target.with_name(f".{target.name}-{os.getpid()}.partial")
    async with _stage_gate():
        await asyncio.to_thread(_copy_source, source, target, temporary)
    return target


async def stage_source(source: Path) -> Path:
    """Optionally copy a source once into the integration-owned cache."""
    if not os.environ.get(CACHE_ENV, "").strip():
        return source
    cache_root = media_cache_dir()
    if source == cache_root or cache_root in source.parents:
        return source
    loop = asyncio.get_running_loop()
    tasks = _STAGE_TASKS.setdefault(loop, {})
    key = str(source)
    task = tasks.get(key)
    if task is None:
        task = asyncio.create_task(_stage_uncached(source))
        tasks[key] = task
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except BaseException:
        if tasks.get(key) is task:
            tasks.pop(key, None)
        raise


__all__ = [
    "CACHE_ENV",
    "MAX_PIXELS",
    "MIN_PIXELS",
    "RESIZE_FACTOR",
    "local_media_path",
    "media_cache_dir",
    "smart_resize",
    "split_mjpeg",
    "stage_source",
]

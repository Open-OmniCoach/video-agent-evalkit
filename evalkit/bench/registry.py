"""Benchmark registry and annotation loading.

Reads a ``registry.yaml`` and exposes:

* :class:`DatasetSpec` -- frozen description of one benchmark.
* :func:`load_registry` -- parse the yaml, resolve relative paths, return
  ``{name: DatasetSpec}``.
* :func:`load_annotations` -- read the JSON / JSONL annotation for one spec.
* :func:`resolve_video_path` / :func:`resolve_sample_path` -- media path joins.

Deals only in annotations and paths; turning annotations into evaluation cases
is the responsibility of ``loaders.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

STATUS_READY = "ready"
STATUS_PENDING_CONVERSION = "pending_conversion"
STATUS_PATH_MISMATCH = "path_mismatch"
STATUS_DISABLED = "disabled"

_KNOWN_STATUSES = frozenset({STATUS_READY, STATUS_PENDING_CONVERSION, STATUS_PATH_MISMATCH, STATUS_DISABLED})


@dataclass(frozen=True)
class DatasetSpec:
    """One benchmark dataset, fully resolved to absolute paths."""

    name: str
    annotation_path: Path
    video_dir: Path
    status: str
    image_dir: Optional[Path] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def load_registry(
    registry_path: str | Path,
    data_root: str | Path | None = None,
    *,
    include_status: Iterable[str] = (STATUS_READY,),
) -> dict[str, DatasetSpec]:
    """Parse ``registry.yaml`` into ``{name: DatasetSpec}``.

    ``data_root`` is what relative entries resolve against (defaults to the
    directory containing the registry file).
    """
    import yaml

    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"registry not found: {registry_path}")
    data_root = Path(data_root) if data_root else registry_path.parent

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"registry top-level must be a mapping, got {type(raw).__name__}")

    wanted = set(include_status)
    unknown = wanted - _KNOWN_STATUSES
    if unknown:
        raise ValueError(f"unknown statuses requested: {sorted(unknown)}")

    out: dict[str, DatasetSpec] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"entry {name!r} must be a mapping, got {type(entry).__name__}")

        status = entry.get("status", STATUS_READY)
        if status not in _KNOWN_STATUSES:
            raise ValueError(f"entry {name!r} has unknown status {status!r}; valid: {sorted(_KNOWN_STATUSES)}")
        if status not in wanted:
            continue

        annotation_rel = entry.get("annotation")
        video_rel = entry.get("video_dir")
        if not annotation_rel or not video_rel:
            raise ValueError(f"entry {name!r} missing required 'annotation' or 'video_dir'")

        annotation_path = (data_root / annotation_rel).resolve()
        video_dir = (data_root / video_rel).resolve()
        image_dir = (data_root / entry["image_dir"]).resolve() if entry.get("image_dir") else None

        extra = {
            k: v for k, v in entry.items() if k not in {"annotation", "video_dir", "image_dir", "status", "reason"}
        }

        out[name] = DatasetSpec(
            name=name,
            annotation_path=annotation_path,
            video_dir=video_dir,
            image_dir=image_dir,
            status=status,
            extra=extra,
        )

    return out


def load_annotations(spec: DatasetSpec) -> list[dict[str, Any]]:
    """Load ``spec``'s annotation file as a list of sample dicts (.json / .jsonl)."""
    path = spec.annotation_path
    if not path.exists():
        raise FileNotFoundError(f"annotation file missing for {spec.name!r}: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"annotation {path} must contain a top-level list, got {type(data).__name__}")
    elif suffix == ".jsonl":
        data = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno} JSON decode error: {e}") from e
    else:
        raise ValueError(f"unsupported annotation extension: {path.suffix}")

    for sample in data:
        if isinstance(sample, dict):
            sample.setdefault("data_source", spec.name)
    return data


def resolve_video_path(spec: DatasetSpec, sample_path: str) -> Path:
    rel = sample_path.lstrip("./") if sample_path.startswith("./") else sample_path
    return (spec.video_dir / rel).resolve()


def resolve_sample_path(spec: DatasetSpec, sample: Mapping[str, Any]) -> Path:
    sample_path = str(sample.get("path", ""))
    rel = sample_path.lstrip("./") if sample_path.startswith("./") else sample_path
    if sample.get("data_type") == "image" and spec.image_dir is not None:
        return (spec.image_dir / rel).resolve()
    return (spec.video_dir / rel).resolve()


__all__ = [
    "DatasetSpec",
    "STATUS_READY",
    "STATUS_PENDING_CONVERSION",
    "STATUS_PATH_MISMATCH",
    "STATUS_DISABLED",
    "load_registry",
    "load_annotations",
    "resolve_video_path",
    "resolve_sample_path",
]

"""Media symlink helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def apply_manifest(
    *,
    data_root: Path,
    manifest_path: Path,
    force: bool = False,
) -> list[tuple[str, str]]:
    """Create or verify symlinks declared in ``manifest_path``."""
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"link manifest must be a mapping: {manifest_path}")

    common_root = _expand_path(manifest.get("common_root", ""))
    raw_links = manifest.get("links") or {}
    if not isinstance(raw_links, dict):
        raise ValueError("link manifest field 'links' must be a mapping")

    out: list[tuple[str, str]] = []
    data_root.mkdir(parents=True, exist_ok=True)
    for rel_link, rel_target in raw_links.items():
        link_path = data_root / str(rel_link)
        target_path = _resolve_target(common_root, rel_target)
        if not target_path.exists():
            raise FileNotFoundError(f"link target does not exist: {target_path}")
        _ensure_symlink(link_path, target_path, force=force)
        out.append((str(rel_link), str(target_path)))
    return out


def ensure_overlay_benchmarks_symlink(overlay_root: Path, target: str = "../../benchmarks") -> None:
    """Create ``overlay_root/benchmarks`` as a symlink back to shared media."""
    link = overlay_root / "benchmarks"
    expected = str(Path(target))
    if link.is_symlink():
        if os.readlink(link) != expected:
            raise ValueError(f"{link} points to {os.readlink(link)!r}, expected {expected!r}")
        return
    if link.exists():
        raise ValueError(f"{link} exists and is not a symlink")
    overlay_root.mkdir(parents=True, exist_ok=True)
    os.symlink(expected, link)


def _expand_path(value: Any) -> Path:
    text = os.path.expandvars(str(value)).strip()
    if "$" in text:
        raise ValueError(f"unresolved environment variable in path: {value!r}")
    if not text:
        return Path("")
    return Path(text).expanduser()


def _resolve_target(common_root: Path, value: Any) -> Path:
    target = _expand_path(value)
    if target.is_absolute():
        return target
    if not str(common_root):
        return target
    return common_root / target


def _ensure_symlink(link_path: Path, target_path: Path, *, force: bool) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_text = str(target_path)
    if link_path.is_symlink():
        current = os.readlink(link_path)
        if current == target_text:
            return
        if not force:
            raise ValueError(f"{link_path} points to {current!r}, expected {target_text!r}")
        link_path.unlink()
    elif link_path.exists():
        raise ValueError(f"{link_path} exists and is not a symlink")
    os.symlink(target_text, link_path)


__all__ = ["apply_manifest", "ensure_overlay_benchmarks_symlink"]

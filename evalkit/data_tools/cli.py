"""Data preparation CLI for evalkit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from ..bench.registry import STATUS_READY, load_annotations, load_registry, resolve_sample_path
from . import links
from .converters import CONVERTERS


def add_data_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("data", help="Prepare and validate benchmark data.")
    sub = parser.add_subparsers(dest="data_cmd", required=True)

    convert = sub.add_parser("convert", help="Convert raw benchmark metadata to evalkit annotations.")
    convert.add_argument("datasets", nargs="*", help="Dataset converter names.")
    convert.add_argument("--all", action="store_true", help="Run all registered converters.")
    convert.add_argument("--data-root", default="data", help="Benchmark data root.")
    convert.add_argument("--source", help="Raw source file for a single dataset conversion.")
    convert.add_argument("--registry-path", help="Registry YAML to update. Defaults to configs/registry.yaml.")
    convert.add_argument("--validate", action="store_true", help="Validate converted annotations after writing.")
    convert.set_defaults(func=_cmd_convert)

    validate = sub.add_parser("validate", help="Validate registry paths and a sample of media paths.")
    validate.add_argument("--data-root", default="data", help="Benchmark data root.")
    validate.add_argument("--registry-path", help="Registry YAML. Defaults to configs/registry.yaml.")
    validate.add_argument("--include-status", nargs="*", default=[STATUS_READY])
    validate.add_argument("--media-sample", type=int, default=5, help="Rows per dataset to check for media existence.")
    validate.add_argument(
        "--check-source",
        action="store_true",
        help="Also require optional source_path files used by converters.",
    )
    validate.set_defaults(func=_cmd_validate)

    link = sub.add_parser("link", help="Create/verify benchmark media symlinks from a manifest.")
    link.add_argument("--data-root", default="data", help="Benchmark data root.")
    link.add_argument("--manifest", required=True, help="YAML manifest describing symlinks.")
    link.add_argument("--force", action="store_true", help="Replace existing symlink with a different target.")
    link.set_defaults(func=_cmd_link)


def _cmd_convert(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser()
    registry_path = _resolve_registry_path(args.registry_path)
    names = _resolve_converter_names(args.datasets, all_requested=args.all)
    if args.source and len(names) != 1:
        raise SystemExit("--source can only be used with exactly one dataset")

    for name in names:
        converter = CONVERTERS[name]
        source = Path(args.source).expanduser() if args.source else data_root / converter.source_relative
        output = data_root / converter.output_relative
        if not source.exists():
            raise SystemExit(f"missing source for {name}: {source}")
        rows = converter.convert(source, output)
        _upsert_registry_entry(registry_path, name, converter.registry_entry())
        print(f"{name}: wrote {len(rows)} rows to {output}")
        if args.validate:
            n_rows, n_units = converter.validate_annotation(output)
            unit_label = converter.validation_unit_label
            print(f"{name}: validated {n_rows} rows over {n_units} {unit_label}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser()
    registry_path = _resolve_registry_path(args.registry_path)
    specs = load_registry(registry_path, data_root, include_status=args.include_status)
    if not specs:
        print("no registry entries matched")
        return 0

    failures: list[str] = []
    for name, spec in specs.items():
        if not spec.annotation_path.exists():
            failures.append(f"{name}: missing annotation {spec.annotation_path}")
            continue
        try:
            rows = load_annotations(spec)
        except Exception as exc:  # noqa: BLE001 - validation report
            failures.append(f"{name}: failed to load annotation: {type(exc).__name__}: {exc}")
            continue

        if not spec.video_dir.exists():
            failures.append(f"{name}: missing video_dir {spec.video_dir}")
        if spec.image_dir is not None and not spec.image_dir.exists():
            failures.append(f"{name}: missing image_dir {spec.image_dir}")

        source_path = spec.extra.get("source_path") if spec.extra else None
        if args.check_source and source_path:
            source = (data_root / str(source_path)).resolve()
            if not source.exists():
                failures.append(f"{name}: missing source_path {source}")

        missing_media = _missing_media_rows(spec, rows, max_rows=max(args.media_sample, 0))
        for uid, path in missing_media:
            failures.append(f"{name}: media missing for {uid}: {path}")
        print(f"{name}: annotation={len(rows)} checked_media={min(len(rows), max(args.media_sample, 0))}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("data validation OK")
    return 0


def _cmd_link(args: argparse.Namespace) -> int:
    created = links.apply_manifest(
        data_root=Path(args.data_root).expanduser(),
        manifest_path=Path(args.manifest).expanduser(),
        force=args.force,
    )
    for rel, target in created:
        print(f"{rel} -> {target}")
    print(f"linked {len(created)} entries")
    return 0


def _resolve_converter_names(names: list[str], *, all_requested: bool) -> list[str]:
    if all_requested:
        return sorted(CONVERTERS)
    if not names:
        raise SystemExit("Specify at least one dataset or --all")
    unknown = sorted(set(names) - set(CONVERTERS))
    if unknown:
        raise SystemExit(f"Unknown converter(s): {', '.join(unknown)}")
    return names


def _resolve_registry_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    for root in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        candidate = root / "configs" / "registry.yaml"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[2] / "configs" / "registry.yaml"


class _RegistryDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def _upsert_registry_entry(registry_path: Path, name: str, entry: dict[str, Any]) -> None:
    """Replace or append one top-level entry, leaving comments and other entries untouched."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    text = registry_path.read_text(encoding="utf-8") if registry_path.exists() else ""
    raw = yaml.safe_load(text) if text.strip() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"registry top-level must be a mapping, got {type(raw).__name__}")
    block = yaml.dump({name: entry}, Dumper=_RegistryDumper, sort_keys=False, allow_unicode=True)
    if text and not text.endswith("\n"):
        text += "\n"
    match = re.search(rf"(?m)^{re.escape(name)}:\n(?:[ \t]+\S.*\n)*", text)
    if match:
        updated = text[: match.start()] + block + text[match.end() :]
    else:
        updated = f"{text}\n{block}" if text.strip() else block
    if (yaml.safe_load(updated) or {}).get(name) != entry:
        raise ValueError(f"could not update registry entry {name!r} in {registry_path}")
    registry_path.write_text(updated, encoding="utf-8")


def _missing_media_rows(spec: Any, rows: list[dict[str, Any]], *, max_rows: int) -> list[tuple[str, Path]]:
    missing: list[tuple[str, Path]] = []
    for row in rows[:max_rows]:
        data_type = str(row.get("data_type", "")).strip().lower()
        if data_type not in {"video", "image"}:
            continue
        path = resolve_sample_path(spec, row)
        if not path.exists():
            uid = str(row.get("problem_id") or row.get("uid") or row.get("id") or "?")
            missing.append((uid, path))
    return missing


__all__ = ["add_data_subparser"]

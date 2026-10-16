"""Benchmark registry and loaders exposed as a lazy stream of EvalCase values."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..bench.loaders import BuildContext, get_loader
from ..bench.registry import DatasetSpec, load_annotations, load_registry
from ..config import BenchCfg, RunConfig
from ..types import EvalCase


def resolve_registry_path(run_cfg: RunConfig) -> Path:
    return Path(run_cfg.registry_path or "configs/registry.yaml")


def load_specs(run_cfg: RunConfig) -> dict[str, DatasetSpec]:
    return load_registry(resolve_registry_path(run_cfg), run_cfg.data_root)


def iter_bench_cases(run_cfg: RunConfig, bench: BenchCfg, spec: DatasetSpec) -> Iterator[EvalCase]:
    """Yield cases lazily for one benchmark, respecting its optional limit."""
    loader = get_loader(spec)
    ctx = BuildContext(
        spec=spec,
        prompt=run_cfg.bench_prompt(bench),
        budget=run_cfg.bench_budget(bench),
    )
    for index, annotation in enumerate(load_annotations(spec)):
        if bench.limit is not None and index >= bench.limit:
            break
        yield loader.build_case(annotation, ctx, index=index)


__all__ = ["iter_bench_cases", "load_specs", "resolve_registry_path"]

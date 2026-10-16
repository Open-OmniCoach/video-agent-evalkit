"""Benchmark registry, loaders, scoring, aggregation, and persistence."""

# Import benchmark-specific scorers/aggregators for their registration side effects.
from . import benchmarks as _benchmarks  # noqa: F401

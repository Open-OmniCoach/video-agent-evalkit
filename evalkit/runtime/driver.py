"""Finite, bounded, fully-asynchronous evaluation driver."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..bench import aggregate as aggregate_module
from ..bench.resume import ResumeState
from ..config import BenchCfg, RunConfig
from ..types import (
    EvalCase,
    GenerationResult,
    GenerationStatus,
    Prediction,
    ProblemType,
    ScoreResult,
    TrajectoryEvent,
)
from .context import EvalRuntime
from .datasource import iter_bench_cases, load_specs
from .plugins import load_callable
from .scoring import ScoreFn, build_score_fn
from .transport import EndpointPoolUnavailable

logger = logging.getLogger("evalkit.driver")


async def _flush_resume(resume: ResumeState) -> None:
    """Keep the single writer ordered without blocking other cases on storage."""
    writer = asyncio.create_task(asyncio.to_thread(resume.flush))
    try:
        await asyncio.shield(writer)
    except asyncio.CancelledError:
        await writer
        raise


@dataclass(frozen=True)
class _CaseExecution:
    case: EvalCase
    generation: GenerationResult
    score: ScoreResult
    status: GenerationStatus
    attempts: int
    attempt_errors: tuple[str, ...] = ()
    error: str | None = None


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _attempt_events(events: tuple[TrajectoryEvent, ...], attempt: int) -> tuple[TrajectoryEvent, ...]:
    """Place each turn's model_exchange between its request-side and result-side
    events; trainable consumers split steps at exchanges."""
    tagged = tuple(replace(event, attempt=attempt) for event in events)
    before = {"case_input", "media_initial", "model_request"}
    return tuple(
        sorted(
            tagged,
            key=lambda event: (
                event.turn,
                (0 if event.kind in before else 1 if event.kind == "model_exchange" else 2),
            ),
        )
    )


async def _call_generate(
    fn: Any,
    runtime: EvalRuntime,
    case: EvalCase,
    sampling: dict[str, Any],
) -> GenerationResult:
    result = fn(runtime, case, sampling)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, GenerationResult):
        raise TypeError(f"generate plugin returned {type(result).__name__}, expected GenerationResult")
    return result


async def _execute_case(
    runtime: EvalRuntime,
    case: EvalCase,
    *,
    generate_fn: Any,
    score_fn: ScoreFn,
    sampling: dict[str, Any],
    idempotent: bool,
    case_max_attempts: int,
    capture_model_io: bool,
) -> _CaseExecution:
    max_attempts = case_max_attempts if idempotent else 1
    errors: list[str] = []
    prior_events: list[TrajectoryEvent] = []
    generation: GenerationResult | None = None
    attempts_used = 0

    try:
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            trace_token = runtime.begin_model_trace() if capture_model_io else None
            model_events: tuple[TrajectoryEvent, ...] = ()
            try:
                try:
                    generation = await _call_generate(generate_fn, runtime, case, sampling)
                finally:
                    if capture_model_io:
                        model_events = runtime.end_model_trace(trace_token)
            except asyncio.CancelledError:
                raise
            except EndpointPoolUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - plugin/infra failure becomes a result row
                error = _error_text(exc)
                errors.append(error)
                prior_events.extend(_attempt_events(model_events, attempt - 1))
                prior_events.append(TrajectoryEvent("error", attempt - 1, {"error": error}, attempt=attempt - 1))
                if attempt < max_attempts:
                    continue
                generation = GenerationResult(status=GenerationStatus.FAILED, error=error)
                break

            generation = replace(
                generation,
                trajectory=_attempt_events(generation.trajectory + model_events, attempt - 1),
            )

            if generation.status == GenerationStatus.FAILED:
                error = generation.error or "generation returned failed status"
                errors.append(error)
                if generation.error is None:
                    generation = replace(generation, error=error)
                if attempt < max_attempts:
                    prior_events.extend(generation.trajectory)
                    prior_events.append(TrajectoryEvent("error", attempt - 1, {"error": error}, attempt=attempt - 1))
                    continue
            break

        assert generation is not None
        if prior_events:
            generation = replace(generation, trajectory=tuple(prior_events) + generation.trajectory)

        if generation.status not in {
            GenerationStatus.COMPLETED,
            GenerationStatus.TRUNCATED,
        }:
            return _CaseExecution(
                case=case,
                generation=generation,
                score=ScoreResult(),
                status=generation.status,
                attempts=attempts_used,
                attempt_errors=tuple(errors),
                error=generation.error,
            )

        try:
            scored = await score_fn(runtime, case, generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - retain generated answer with scoring failure
            error = _error_text(exc)
            events = generation.trajectory + (
                TrajectoryEvent(
                    "error",
                    len(generation.trajectory),
                    {"stage": "score", "error": error},
                    attempt=attempts_used - 1,
                ),
            )
            generation = replace(generation, trajectory=events)
            errors.append(error)
            return _CaseExecution(
                case=case,
                generation=generation,
                score=ScoreResult(details={"failure_stage": "score"}),
                status=GenerationStatus.FAILED,
                attempts=attempts_used,
                attempt_errors=tuple(errors),
                error=error,
            )
        return _CaseExecution(
            case=case,
            generation=generation,
            score=scored,
            status=generation.status,
            attempts=attempts_used,
            attempt_errors=tuple(errors),
            error=generation.error,
        )
    finally:
        runtime.release_route_key(case.uid)


async def _prediction(execution: _CaseExecution, resume: ResumeState) -> Prediction:
    generation = execution.generation
    artifact = await asyncio.to_thread(resume.write_trajectory, execution.case.uid, generation.trajectory)
    return Prediction(
        uid=execution.case.uid,
        status=execution.status.value,
        response=generation.response,
        prediction=execution.score.prediction,
        score=execution.score.score,
        score_details=dict(execution.score.details),
        error=execution.error,
        facets=dict(execution.case.facets),
        generation={
            "attempts": execution.attempts,
            "attempt_errors": list(execution.attempt_errors),
            "status": generation.status.value,
            "endpoint": generation.endpoint,
            "finish_reason": generation.finish_reason,
            "usage": {
                "prompt_tokens": generation.usage.prompt_tokens,
                "completion_tokens": generation.usage.completion_tokens,
                "cached_tokens": generation.usage.cached_tokens,
            },
            "timing": {
                "latency_ms": generation.timing.latency_ms,
                "non_generation_ms": generation.timing.non_generation_ms,
            },
        },
        trajectory_artifact=artifact,
    )


def _bench_report_cfg(bench: BenchCfg, spec: Any) -> dict[str, Any]:
    report = dict((getattr(spec, "extra", None) or {}).get("report") or {})
    report.update(bench.report)
    return report


async def run_bench(
    runtime: EvalRuntime,
    run_cfg: RunConfig,
    bench: BenchCfg,
    spec: Any,
) -> dict[str, Any]:
    out_dir = Path(run_cfg.out_dir) / bench.name
    resume = ResumeState(out_dir, flush_every=run_cfg.flush_every)
    initial_done = len(resume.done_uids)
    generate_fn = load_callable(run_cfg.bench_generate_fn(bench))
    score_fn = build_score_fn(run_cfg.bench_score_fn(bench), run_cfg.bench_scoring(bench))
    sampling = run_cfg.bench_sampling(bench)
    idempotent = run_cfg.bench_idempotent(bench)

    cases = iter(iter_bench_cases(run_cfg, bench, spec))
    active: set[asyncio.Task[_CaseExecution]] = set()
    source_exhausted = False
    problem_type: ProblemType | None = None
    scheduled = 0
    seen_uids: set[str] = set()

    def _next_pending() -> EvalCase | None:
        nonlocal source_exhausted, problem_type
        while not source_exhausted:
            try:
                case = next(cases)
            except StopIteration:
                source_exhausted = True
                return None
            if case.uid in seen_uids:
                raise ValueError(f"benchmark {bench.name!r} contains duplicate uid {case.uid!r}")
            seen_uids.add(case.uid)
            problem_type = problem_type or case.problem_type
            if not resume.is_done(case.uid):
                return case
        return None

    try:
        while not source_exhausted or active:
            while not source_exhausted and len(active) < run_cfg.max_active_cases:
                case = _next_pending()
                if case is None:
                    break
                active.add(
                    asyncio.create_task(
                        _execute_case(
                            runtime,
                            case,
                            generate_fn=generate_fn,
                            score_fn=score_fn,
                            sampling=sampling,
                            idempotent=idempotent,
                            case_max_attempts=run_cfg.case_max_attempts,
                            capture_model_io=run_cfg.trajectory == "trainable",
                        )
                    )
                )
                scheduled += 1

            if not active:
                continue
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                execution = await task
                resume.record(await _prediction(execution, resume))
                if resume.should_flush():
                    await _flush_resume(resume)
    finally:
        try:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
        finally:
            resume.flush()

    logger.info(
        "bench %s: scheduled=%d resumed=%d total=%d",
        bench.name,
        scheduled,
        initial_done,
        len(resume.records),
    )
    aggregator_name = bench.aggregator
    aggregator_fn = None
    if aggregator_name and ":" in aggregator_name:
        aggregator_fn = load_callable(aggregator_name)
        aggregator_name = None
    metrics = aggregate_module.aggregate(
        problem_type or ProblemType.OPEN_ENDED,
        resume.records,
        aggregator_name=aggregator_name,
        aggregator_fn=aggregator_fn,
        dataset=bench.name,
        report=_bench_report_cfg(bench, spec),
    )
    return resume.write_summary(metrics)


async def run(run_cfg: RunConfig) -> dict[str, Any]:
    specs = load_specs(run_cfg)
    summaries: dict[str, Any] = {}
    async with EvalRuntime.from_run_config(run_cfg) as runtime:
        for bench in run_cfg.eval_datasets:
            spec = specs.get(bench.name)
            if spec is None:
                raise KeyError(f"benchmark {bench.name!r} is not ready in the registry; available: {sorted(specs)}")
            summaries[bench.name] = await run_bench(runtime, run_cfg, bench, spec)
        result = {"benchmarks": summaries, "runtime": runtime.metrics()}

    out_root = Path(run_cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "summary.json"
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return result


def run_sync(run_cfg: RunConfig) -> dict[str, Any]:
    return asyncio.run(run(run_cfg))


__all__ = ["run", "run_bench", "run_sync"]

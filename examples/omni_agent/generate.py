"""Native tool-calling video-QA agent loop."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Mapping
from typing import Any

from evalkit.types import (
    EvalCase,
    GenerationResult,
    GenerationStatus,
    GenerationTiming,
    GenerationUsage,
    SamplingParams,
    TrajectoryEvent,
)

from .media import MediaBundle, MediaExecutionError, MediaSession
from .policy import AgentPolicy
from .protocol import TOOLS, ProtocolError, require_single_tool_call, validate_assistant_text
from .session import AgentSession


def _turn_sampling(sampling: SamplingParams, policy: AgentPolicy) -> dict[str, Any]:
    values = dict(sampling)
    if "max_new_tokens" not in values and "max_tokens" not in values:
        values["max_tokens"] = policy.default_max_output_tokens
    return values


def _tool_identity(call: Mapping[str, Any] | None) -> tuple[str, str]:
    if not isinstance(call, Mapping):
        return "", ""
    function = call.get("function")
    name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
    return str(call.get("id") or ""), name


def _result(
    *,
    status: GenerationStatus,
    response: str,
    assistant_message: Mapping[str, Any] | None,
    events: list[TrajectoryEvent],
    usage: GenerationUsage,
    model_latency_ms: float,
    non_generation_ms: float,
    endpoint: str | None,
    finish_reason: str | None,
    error: str | None = None,
) -> GenerationResult:
    return GenerationResult(
        status=status,
        response=response,
        assistant_message=assistant_message,
        trajectory=tuple(events),
        usage=usage,
        timing=GenerationTiming(
            latency_ms=model_latency_ms,
            non_generation_ms=non_generation_ms,
        ),
        endpoint=endpoint,
        finish_reason=finish_reason,
        error=error,
    )


async def generate(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    policy = AgentPolicy.from_metadata(case.metadata)
    media_started = time.monotonic()
    media = await MediaSession.open(case, policy)
    try:
        initial = await media.initial_bundle()
        preparation_ms = (time.monotonic() - media_started) * 1000.0
        return await _generate_with_media(
            runtime,
            case,
            sampling,
            policy=policy,
            media=media,
            initial=initial,
            preparation_ms=preparation_ms,
            trace_complete=os.environ.get("OMNI_AGENT_TRACE_COMPLETE", "1").strip() != "0",
        )
    finally:
        if os.environ.get("OMNI_AGENT_PRESERVE_TRACE_MEDIA", "").strip() != "1":
            await media.close()


async def _generate_with_media(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
    *,
    policy: AgentPolicy,
    media: MediaSession,
    initial: MediaBundle,
    preparation_ms: float,
    trace_complete: bool,
) -> GenerationResult:
    events: list[TrajectoryEvent] = [
        TrajectoryEvent(
            "case_input",
            0,
            {
                "uid": case.uid,
                "dataset": case.dataset,
                "index": case.index,
                "problem_type": str(case.problem_type.value),
                "target": case.target,
                "metadata": copy.deepcopy(dict(case.metadata)),
                "source_messages": copy.deepcopy([dict(item) for item in case.messages]),
                "video_path": str(media.info.path),
                "audio_path": (str(media.info.audio_path) if media.info.audio_path is not None else None),
                "duration": media.info.duration,
                "has_audio": media.info.has_audio,
            },
        ),
        TrajectoryEvent(
            "media_initial",
            0,
            {
                **dict(initial.manifest),
                **({"parts": copy.deepcopy([dict(part) for part in initial.parts])} if trace_complete else {}),
            },
        ),
    ]
    usage = GenerationUsage()
    model_latency_ms = 0.0
    non_generation_ms = preparation_ms
    max_prompt_tokens = 0
    last = None

    session = AgentSession.create(case, media.info, initial)
    request_body = {"tools": [dict(tool) for tool in TOOLS], "tool_choice": "auto"}
    turn_sampling = _turn_sampling(sampling, policy)

    for turn in range(policy.max_turns):
        session.append_control_state(step=turn, max_steps=policy.max_turns)
        events.append(
            TrajectoryEvent(
                "model_request",
                turn,
                {
                    "summary": session.request_summary(),
                    **(
                        {
                            "messages": copy.deepcopy(session.messages),
                            "sampling": copy.deepcopy(turn_sampling),
                            "request_body": copy.deepcopy(request_body),
                        }
                        if trace_complete
                        else {}
                    ),
                },
            )
        )
        last = await runtime.chat(
            "candidate",
            session.messages,
            sampling=turn_sampling,
            request_body=request_body,
            route_key=case.uid,
        )
        usage = usage + last.usage
        model_latency_ms += last.timing.latency_ms
        max_prompt_tokens = max(max_prompt_tokens, last.usage.prompt_tokens)
        session.append_assistant(last.message)
        calls = tuple(last.tool_calls)
        events.append(
            TrajectoryEvent(
                "model_response",
                turn,
                {
                    "text": last.text,
                    "message": copy.deepcopy(dict(last.message)),
                    "tool_calls": copy.deepcopy([dict(call) for call in calls]),
                    "finish_reason": last.finish_reason,
                    "endpoint": last.endpoint,
                    "usage": {
                        "prompt_tokens": last.usage.prompt_tokens,
                        "completion_tokens": last.usage.completion_tokens,
                        "cached_tokens": last.usage.cached_tokens,
                    },
                },
            )
        )

        if not calls:
            try:
                validate_assistant_text(last.text, has_tool_call=False)
            except ProtocolError as exc:
                session.append_protocol_error(
                    exc.code,
                    str(exc),
                    max_invalid_retries=policy.max_invalid_retries,
                )
                events.append(
                    TrajectoryEvent(
                        "protocol_error",
                        turn,
                        {"code": exc.code, "error": str(exc)},
                    )
                )
                continue

            status = (
                GenerationStatus.TRUNCATED
                if str(last.finish_reason).lower() in {"length", "max_tokens"}
                else GenerationStatus.COMPLETED
            )
            events.append(
                TrajectoryEvent(
                    "agent_summary",
                    turn,
                    {
                        "turns": turn + 1,
                        "tool_calls": session.tool_calls_used,
                        "invalid_actions": session.invalid_actions,
                        "prompt_tokens_sum": usage.prompt_tokens,
                        "prompt_tokens_max": max_prompt_tokens,
                        "completion_tokens_sum": usage.completion_tokens,
                    },
                )
            )
            return _result(
                status=status,
                response=last.text,
                assistant_message=last.message,
                events=events,
                usage=usage,
                model_latency_ms=model_latency_ms,
                non_generation_ms=non_generation_ms,
                endpoint=last.endpoint,
                finish_reason=last.finish_reason,
            )

        first_call = calls[0]
        call_id, tool_name = _tool_identity(first_call)
        try:
            validate_assistant_text(last.text, has_tool_call=True)
            request = require_single_tool_call(
                calls,
                duration=media.info.duration,
                has_audio=media.info.has_audio,
            )
        except ProtocolError as exc:
            if len(calls) > 1:
                session.append_protocol_error_for_calls(
                    calls,
                    exc.code,
                    str(exc),
                    max_invalid_retries=policy.max_invalid_retries,
                )
            else:
                session.append_protocol_error(
                    exc.code,
                    str(exc),
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    max_invalid_retries=policy.max_invalid_retries,
                )
            events.append(
                TrajectoryEvent(
                    "protocol_error",
                    turn,
                    {"code": exc.code, "error": str(exc)},
                )
            )
            continue

        if turn + 1 == policy.max_turns:
            session.append_protocol_error(
                "STEP_LIMIT_REACHED",
                "must answer on final step",
                tool_call_id=request.call_id,
                tool_name=request.name,
                max_invalid_retries=policy.max_invalid_retries,
            )
            events.append(
                TrajectoryEvent(
                    "protocol_error",
                    turn,
                    {
                        "code": "STEP_LIMIT_REACHED",
                        "error": "must answer on final step",
                    },
                )
            )
            return _result(
                status=GenerationStatus.FAILED,
                response=last.text,
                assistant_message=last.message,
                events=events,
                usage=usage,
                model_latency_ms=model_latency_ms,
                non_generation_ms=non_generation_ms,
                endpoint=last.endpoint,
                finish_reason="step_limit_reached",
                error="STEP_LIMIT_REACHED: must answer on final step",
            )

        if session.is_duplicate(request):
            session.append_no_progress(request)
            events.append(
                TrajectoryEvent(
                    "protocol_error",
                    turn,
                    {
                        "code": "NO_PROGRESS",
                        "error": "exact duplicate of a successful tool call",
                    },
                )
            )
            continue

        events.append(
            TrajectoryEvent(
                "tool_call",
                turn,
                {
                    "id": request.call_id,
                    "name": request.name,
                    "start": request.start_code,
                    "end": request.end_code,
                    "level": request.level,
                    "start_seconds": request.start,
                    "end_seconds": request.end,
                },
            )
        )
        tool_started = time.monotonic()
        try:
            bundle = await media.execute(request)
        except MediaExecutionError as exc:
            non_generation_ms += (time.monotonic() - tool_started) * 1000.0
            events.append(
                TrajectoryEvent(
                    "media_error",
                    turn,
                    {"tool": request.name, "error": str(exc)},
                )
            )
            return _result(
                status=GenerationStatus.FAILED,
                response=last.text,
                assistant_message=last.message,
                events=events,
                usage=usage,
                model_latency_ms=model_latency_ms,
                non_generation_ms=non_generation_ms,
                endpoint=last.endpoint,
                finish_reason="media_error",
                error=f"MediaExecutionError: {exc}",
            )
        non_generation_ms += (time.monotonic() - tool_started) * 1000.0
        session.append_tool_result(request, bundle)
        events.append(
            TrajectoryEvent(
                "tool_result",
                turn,
                {
                    **dict(bundle.manifest),
                    "header": bundle.header,
                    **({"parts": copy.deepcopy([dict(part) for part in bundle.parts])} if trace_complete else {}),
                },
            )
        )

    response = last.text if last is not None else ""
    assistant = last.message if last is not None else None
    endpoint = last.endpoint if last is not None else None
    events.append(
        TrajectoryEvent(
            "agent_summary",
            policy.max_turns,
            {
                "turns": policy.max_turns,
                "tool_calls": session.tool_calls_used,
                "invalid_actions": session.invalid_actions,
                "prompt_tokens_sum": usage.prompt_tokens,
                "prompt_tokens_max": max_prompt_tokens,
                "completion_tokens_sum": usage.completion_tokens,
            },
        )
    )
    return _result(
        status=GenerationStatus.TRUNCATED,
        response=response,
        assistant_message=assistant,
        events=events,
        usage=usage,
        model_latency_ms=model_latency_ms,
        non_generation_ms=non_generation_ms,
        endpoint=endpoint,
        finish_reason="max_turns",
        error="agent exhausted max_turns before a valid final answer",
    )


__all__ = ["generate"]

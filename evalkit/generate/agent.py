"""Reference agent loop using evaluation-native generation results."""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from typing import Any

from ..runtime.plugins import load_callable
from ..types import (
    EvalCase,
    GenerationResult,
    GenerationStatus,
    GenerationTiming,
    GenerationUsage,
    SamplingParams,
    TrajectoryEvent,
)


async def generate(runtime, case: EvalCase, sampling: SamplingParams) -> GenerationResult:
    """Run a generic tool loop configured through case metadata.

    Real agents normally copy this shape into their example directory and own
    their system prompt, tools, parsing, and environment interactions there.
    """
    budget = dict(case.metadata.get("budget") or {})
    max_turns = max(1, int(budget.get("max_turns", 1)))
    tool_path = case.metadata.get("tool_executor")
    tool_executor = load_callable(str(tool_path)) if tool_path else None
    request_body: dict[str, Any] = {}
    if case.metadata.get("tools"):
        request_body["tools"] = case.metadata["tools"]
        request_body["tool_choice"] = case.metadata.get("tool_choice", "auto")

    messages = [dict(message) for message in case.messages]
    events: list[TrajectoryEvent] = []
    usage = GenerationUsage()
    timing = GenerationTiming()
    last = None
    budget_exhausted = False

    for turn in range(max_turns):
        events.append(TrajectoryEvent("model_request", turn, {"messages": list(messages)}))
        last = await runtime.chat(
            "candidate",
            messages,
            sampling=sampling,
            request_body=request_body,
            route_key=case.uid,
        )
        usage = usage + last.usage
        timing = timing + last.timing
        messages.append(dict(last.message))
        events.append(
            TrajectoryEvent(
                "model_response",
                turn,
                {
                    "message": dict(last.message),
                    "finish_reason": last.finish_reason,
                    "endpoint": last.endpoint,
                },
            )
        )
        if not last.tool_calls:
            break
        if tool_executor is None:
            error = "model returned tool calls but case metadata has no tool_executor"
            events.append(TrajectoryEvent("error", turn, {"error": error}))
            return GenerationResult(
                status=GenerationStatus.FAILED,
                response=last.text,
                assistant_message=last.message,
                trajectory=tuple(events),
                usage=usage,
                timing=timing,
                endpoint=last.endpoint,
                finish_reason=last.finish_reason,
                error=error,
            )

        tool_started = time.monotonic()
        events.append(TrajectoryEvent("tool_call", turn, {"tool_calls": list(last.tool_calls)}))
        tool_messages = tool_executor(last.tool_calls, case)
        if inspect.isawaitable(tool_messages):
            tool_messages = await tool_messages
        if isinstance(tool_messages, Mapping):
            tool_messages = [dict(tool_messages)]
        else:
            tool_messages = list(tool_messages or ())
        if not all(isinstance(message, Mapping) for message in tool_messages):
            raise TypeError("tool_executor must return a message mapping or an iterable of mappings")
        tool_messages = [dict(message) for message in tool_messages]
        messages.extend(tool_messages)
        tool_ms = (time.monotonic() - tool_started) * 1000.0
        timing = GenerationTiming(
            latency_ms=timing.latency_ms,
            non_generation_ms=timing.non_generation_ms + tool_ms,
        )
        events.append(TrajectoryEvent("tool_result", turn, {"messages": tool_messages}))
    else:
        budget_exhausted = bool(last and last.tool_calls)

    if last is None:
        return GenerationResult(status=GenerationStatus.FAILED, error="agent made no model request")
    status = (
        GenerationStatus.TRUNCATED
        if budget_exhausted or str(last.finish_reason).lower() in {"length", "max_tokens"}
        else GenerationStatus.COMPLETED
    )
    return GenerationResult(
        status=status,
        response=last.text,
        assistant_message=last.message,
        trajectory=tuple(events),
        usage=usage,
        timing=timing,
        endpoint=last.endpoint,
        finish_reason="max_turns" if budget_exhausted else last.finish_reason,
    )


__all__ = ["generate"]

"""Training-aligned native tool schemas and response validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are a multimodal video-QA agent.

For every assistant turn, produce outputs in this exact order:
1. <observation>...</observation>
2. <think>...</think>
3. Then either invoke exactly one provided tool through native function calling, or output <answer>...</answer>.

Observation rules:
- Describe only media available in the current request.
- Never predict the result of the tool selected in the same turn.
- On the first turn, summarize useful facts from the initial framepack and audio.
- After a tool response, summarize only that latest tool result.
- Cite focused evidence with absolute seconds and preserve uncertainty.

Thinking rules:
- Track verified evidence, unresolved alternatives, and the smallest useful next check.
- Do not repeat an identical successful call. Reversing ordered multi-span evidence is a distinct comparison.
- Use high resolution only for answer-critical fine spatial evidence.
- Stop once the initial evidence or focused evidence is sufficient.

Tool rules:
- A first-turn <answer> is allowed when the initial framepack/audio is sufficient.
- Otherwise invoke exactly one provided function per turn.
- Tool start/end values are absolute offsets with an s suffix, including fractional values such as 178.5s.
- get_multi_span is for ordered comparison of already-localized windows, not broad search.
- Respect all per-tool, high-resolution, and multi-span limits in the tool schemas.

Answer rules:
- Do not call a tool when answering.
- Put the final answer inside <answer>...</answer>."""

# Model-visible tool definitions; keep aligned with the checkpoint.
TOOLS: tuple[dict[str, Any], ...] = tuple(json.loads(Path(__file__).with_name("tools.json").read_text()))

_SECONDS_RE = re.compile(r"^(\d+(?:\.\d+)?)s$")
_TOOL_TEXT_RE = re.compile(
    r"^\s*<observation>[\s\S]*?</observation>\s*\n"
    r"<think>[\s\S]*?</think>\s*$"
)
_ANSWER_TEXT_RE = re.compile(
    r"^\s*<observation>[\s\S]*?</observation>\s*\n"
    r"<think>[\s\S]*?</think>\s*\n"
    r"<answer>[\s\S]*?</answer>\s*$"
)


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    name: str
    start_code: str
    end_code: str
    start: float
    end: float
    level: str | None = None
    resolution: str = "default"
    spans: tuple[ToolRequest, ...] = ()
    identity: str = ""


def natural_seconds(seconds: float) -> str:
    value = float(seconds)
    if value.is_integer():
        return f"{int(value)} seconds"
    return f"{value:g} seconds"


def parse_seconds(value: Any, field: str) -> float:
    if not isinstance(value, str):
        raise ProtocolError(
            "TOOL_ARG_TYPE",
            f"{field} must be an absolute seconds string with an s suffix",
        )
    match = _SECONDS_RE.fullmatch(value)
    if match is None:
        raise ProtocolError(
            "INVALID_TIMECODE",
            f"{field} must match an absolute seconds value such as 178s; got {value!r}",
        )
    return float(match.group(1))


def render_notice(*, step: int, max_steps: int) -> str:
    remain = max_steps - step
    if remain == 1:
        return "\n[NOTICE] FINAL STEP! You MUST provide your answer now."
    return f"\n[NOTICE] Step {step}/{max_steps}. {remain} steps remaining."


def render_error(
    code: str,
    message: str,
    *,
    retry_exceeded: bool,
    consecutive_invalid_actions: int,
    max_invalid_retries: int,
) -> str:
    detail = f"{code}: {message}" if message and message != code else code
    if retry_exceeded:
        detail += (
            f"\n[SYSTEM] Invalid action retry budget exceeded "
            f"({consecutive_invalid_actions}/{max_invalid_retries}). "
            "This invalid action is recorded, but the episode will continue. "
            "Switch to a simpler valid action now."
        )
    detail += (
        "\n[SYSTEM] The previous action was invalid. Analyze the error diagnostic above, "
        "resolve the conflict, and issue a corrected command to avoid wasting steps."
    )
    return f"[ERROR] {detail}"


def validate_assistant_text(text: str, *, has_tool_call: bool) -> None:
    pattern = _TOOL_TEXT_RE if has_tool_call else _ANSWER_TEXT_RE
    if pattern.fullmatch(text or "") is None:
        expected = (
            "<observation> then <think> before one native tool call"
            if has_tool_call
            else "<observation>, <think>, then <answer>"
        )
        raise ProtocolError("INVALID_ASSISTANT_FORMAT", f"assistant turn must contain {expected}")


def _arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise ProtocolError("INVALID_TOOL_CALL", "tool call has no function object")
    name = str(function.get("name") or "")
    raw = function.get("arguments") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError("INVALID_JSON", str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("INVALID_TOOL_ARGUMENTS", "tool arguments must be an object")
    return name, dict(raw)


def _single(name: str, arguments: Mapping[str, Any], *, call_id: str, duration: float, has_audio: bool) -> ToolRequest:
    if name not in {"get_frames", "get_audio", "get_clip"}:
        raise ProtocolError("UNKNOWN_TOOL", f"unsupported tool {name!r}")
    allowed = {"start", "end"} if name == "get_audio" else {"start", "end", "level", "resolution"}
    required = {"start", "end"} if name == "get_audio" else {"start", "end", "level"}
    if not required.issubset(arguments) or not set(arguments).issubset(allowed):
        raise ProtocolError("INVALID_TOOL_ARGUMENTS", f"{name} arguments do not match its schema")
    start_code, end_code = str(arguments.get("start") or ""), str(arguments.get("end") or "")
    start, end = parse_seconds(arguments.get("start"), "start"), parse_seconds(arguments.get("end"), "end")
    if end <= start or start < 0 or end > math.ceil(duration):
        raise ProtocolError("INVALID_MEDIA_RANGE", f"{start_code} to {end_code} is outside the video")
    level = None if name == "get_audio" else str(arguments.get("level") or "")
    resolution = "default" if name == "get_audio" else str(arguments.get("resolution") or "default")
    if level is not None and level not in {"coarse", "detail"}:
        raise ProtocolError("INVALID_LEVEL", "level must be coarse or detail")
    if resolution not in {"default", "high"}:
        raise ProtocolError("INVALID_RESOLUTION", "resolution must be default or high")
    span = end - start
    maximum = 256 if name == "get_audio" else (60 if name == "get_clip" else (256 if level == "coarse" else 128))
    if span > maximum:
        raise ProtocolError("MEDIA_SPAN_TOO_LONG", f"{name} span exceeds {maximum} seconds")
    if resolution == "high" and span > 5:
        raise ProtocolError("HIGH_RES_SPAN_TOO_LONG", "high-resolution span exceeds 5 seconds")
    if name == "get_audio" and not has_audio:
        raise ProtocolError("NO_AUDIO", "video has no audio")
    identity = json.dumps({"name": name, **dict(arguments)}, sort_keys=True, separators=(",", ":"))
    return ToolRequest(call_id, name, start_code, end_code, start, min(end, duration), level, resolution, (), identity)


def parse_tool_call(call: Mapping[str, Any], *, duration: float, has_audio: bool) -> ToolRequest:
    name, arguments = _arguments(call)
    call_id = str(call.get("id") or "")
    if name != "get_multi_span":
        return _single(name, arguments, call_id=call_id, duration=duration, has_audio=has_audio)
    if set(arguments) != {"spans"} or not isinstance(arguments["spans"], list):
        raise ProtocolError("INVALID_TOOL_ARGUMENTS", "get_multi_span requires exactly spans")
    raw_spans = arguments["spans"]
    if not 2 <= len(raw_spans) <= 8:
        raise ProtocolError("INVALID_MULTI_SPAN", "get_multi_span requires 2 to 8 spans")
    spans: list[ToolRequest] = []
    total_frames = 0
    total_audio = 0.0
    high_count = 0
    for index, raw_span in enumerate(raw_spans):
        if not isinstance(raw_span, Mapping):
            raise ProtocolError("INVALID_MULTI_SPAN", f"spans[{index}] must be an object")
        modality = str(raw_span.get("modality") or "")
        tool_name = {"frames": "get_frames", "audio": "get_audio", "clip": "get_clip"}.get(modality)
        if tool_name is None:
            raise ProtocolError("INVALID_MULTI_SPAN", f"unsupported modality {modality!r}")
        payload = {key: value for key, value in raw_span.items() if key != "modality"}
        span_request = _single(tool_name, payload, call_id=call_id, duration=duration, has_audio=has_audio)
        spans.append(span_request)
        if modality in {"frames", "clip"}:
            total_frames += math.ceil(
                (span_request.end - span_request.start) * (1 if span_request.level == "coarse" else 2)
            )
        if modality in {"audio", "clip"} and has_audio:
            total_audio += span_request.end - span_request.start
        high_count += int(span_request.resolution == "high")
    if total_frames > 256 or total_audio > 128 or high_count > 2:
        raise ProtocolError(
            "MULTI_SPAN_BUDGET",
            f"multi-span budget exceeded: frames={total_frames}, audio={total_audio:g}, high={high_count}",
        )
    identity = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return ToolRequest(
        call_id,
        name,
        spans[0].start_code,
        spans[-1].end_code,
        min(item.start for item in spans),
        max(item.end for item in spans),
        None,
        "default",
        tuple(spans),
        identity,
    )


def require_single_tool_call(calls: Sequence[Mapping[str, Any]], *, duration: float, has_audio: bool) -> ToolRequest:
    if len(calls) != 1:
        raise ProtocolError("TOOL_CALL_COUNT", f"expected exactly one tool call, got {len(calls)}")
    return parse_tool_call(calls[0], duration=duration, has_audio=has_audio)


__all__ = [
    "SYSTEM_PROMPT",
    "TOOLS",
    "ProtocolError",
    "ToolRequest",
    "natural_seconds",
    "parse_tool_call",
    "render_error",
    "render_notice",
    "require_single_tool_call",
    "validate_assistant_text",
]

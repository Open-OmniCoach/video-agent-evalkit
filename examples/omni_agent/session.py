"""Training-aligned conversation state, initial-media prompt and keep_recent=1 rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evalkit.types import EvalCase, ProblemType

from .media import MediaBundle, VideoInfo
from .protocol import SYSTEM_PROMPT, ToolRequest, render_error, render_notice

MEDIA_OMITTED = "[MEDIA OMITTED - Refer to your Observation]"


def _fallback_question(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "user":
            continue
        content = message.get("content")
        chunks: list[str] = []
        if isinstance(content, str):
            chunks.append(content)
        elif content is not None:
            for part in content:
                if isinstance(part, Mapping) and str(part.get("type") or "").lower() == "text":
                    chunks.append(str(part.get("text") or ""))
        text = "\n".join(chunk for chunk in chunks if chunk).strip()
        suffixes = (
            "\nAnswer with only the option letter",
            "\nProvide the numerical value",
            "\nProvide the total count",
            '\nAnswer only "Yes" or "No"',
            "\nProvide the time span",
            "\nProvide your answer",
        )
        cut = min((text.find(value) for value in suffixes if value in text), default=-1)
        return text[:cut].strip() if cut >= 0 else text
    raise ValueError("video-QA case has no user question")


def _question(case: EvalCase) -> str:
    value = case.metadata.get("problem") or case.metadata.get("question")
    return str(value).strip() if value else _fallback_question(case.messages)


def _audio_guidance(info: VideoInfo, bundle: MediaBundle) -> str:
    attached = bool(bundle.manifest.get("audio"))
    available = bool(bundle.manifest.get("audio_available", info.has_audio))
    if attached:
        return "- The attached full audio may be used for broad speech/sound localization."
    if available:
        return (
            "- Full audio is not attached to this initial request; audio remains "
            "available through get_audio and get_clip."
        )
    return "- No audio is available; do not call get_audio. get_clip returns visual frames only."


def _answer_instruction(problem_type: ProblemType) -> str:
    if problem_type == ProblemType.MULTIPLE_CHOICE:
        return "Answer with exactly one uppercase option letter in action.content."
    if problem_type == ProblemType.TEMPORAL_GROUNDING:
        # Training-aligned temporal answer instruction.
        return (
            "Answer with one or more positive-width time intervals and no other text in action.content.\n"
            "- One interval: [83.4s,87.8s]\n"
            "- Multiple intervals: [83.4s,87.8s]; [182s,186.5s]\n"
            "- An isolated point such as 83.4s is invalid, including for a 'when' or 'moment' question.\n"
            "- Localize the event's supported onset and offset. Do not invent fixed-width padding around a point.\n"
            "- If either boundary is uncertain, use the media tools to resolve it before answering."
        )
    if problem_type in {ProblemType.NUMERICAL, ProblemType.REGRESSION, ProblemType.REC}:
        return "Answer with exactly one number inside <answer>."
    return "Put the concise final response inside <answer>."


def build_initial_user_text(case: EvalCase, info: VideoInfo, bundle: MediaBundle) -> str:
    question = _question(case)
    lines = [
        "Video:",
        f"- duration: {info.duration:.3f}s",
        f"- has_audio: {info.has_audio}",
        "- Tool start/end values use absolute seconds with an s suffix.",
        "",
        "Legal tool spans:",
        "- get_frames: coarse <= 256s, detail <= 128s.",
        "- get_audio: <= 256s.",
        "- get_clip: coarse <= 60s, detail <= 60s; same-span audio included.",
        "",
        f"Question: {question}",
    ]
    options = list(case.metadata.get("options") or ())
    if case.problem_type == ProblemType.MULTIPLE_CHOICE and options:
        lines.extend(["Options:", *(str(option) for option in options)])
    lines.extend(
        [
            _answer_instruction(case.problem_type),
            "",
            "Initial framepack media:",
            bundle.header,
            "- These timestamped frames are a global scan, not a continuous video.",
            _audio_guidance(info, bundle),
            "- A successful focused tool call replaces this initial media because media.keep_recent=1; "
            "preserve useful initial evidence in observation/think.",
        ]
    )
    return "\n".join(lines)


def _strip_notice(content: Any) -> Any:
    marker = "[NOTICE]"
    if isinstance(content, str):
        position = content.find(marker)
        return content[:position].rstrip() if position >= 0 else content
    if content is None:
        return content
    cleaned = []
    for part in content:
        if isinstance(part, Mapping) and str(part.get("type") or "").lower() == "text":
            text = str(part.get("text") or "")
            position = text.find(marker)
            if position >= 0:
                prefix = text[:position].rstrip()
                if prefix:
                    cleaned.append({**part, "text": prefix})
                continue
        cleaned.append(part)
    return cleaned


def _has_media(content: Any) -> tuple[bool, bool]:
    if isinstance(content, str) or content is None:
        return False, False
    kinds = {str(part.get("type") or "").lower() for part in content if isinstance(part, Mapping)}
    return "image" in kinds, bool(kinds.intersection({"audio", "video"}))


def _first_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is not None:
        for part in content:
            if isinstance(part, Mapping) and str(part.get("type") or "").lower() == "text":
                return str(part.get("text") or "")
    return "Media content"


def _append_control_state(message: dict[str, Any], content: str) -> None:
    """Fold control feedback into the preceding environment message.

    This mirrors the native-tool-v2 training exporter. In particular, it
    avoids an initial ``user -> user`` pair and a standalone user message after
    a tool response, both of which render differently under Swift/Hermes.
    """

    previous = message.get("content")
    if isinstance(previous, str):
        message["content"] = f"{previous.rstrip()}\n{content}" if previous.strip() else content
        return
    if previous is not None:
        message["content"] = [
            *previous,
            {"type": "text", "text": content},
        ]
        return
    message["content"] = content


@dataclass
class AgentSession:
    case: EvalCase
    info: VideoInfo
    messages: list[dict[str, Any]]
    focused_tool_succeeded: bool = False
    tool_calls_used: int = 0
    invalid_actions: int = 0
    consecutive_invalid_actions: int = 0
    total_no_progress: int = 0
    consecutive_no_progress: int = 0
    successful_requests: list[ToolRequest] = field(default_factory=list)

    @classmethod
    def create(cls, case: EvalCase, info: VideoInfo, initial: MediaBundle) -> AgentSession:
        return cls(
            case=case,
            info=info,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_initial_user_text(case, info, initial)},
                        *[dict(part) for part in initial.parts],
                    ],
                },
            ],
        )

    def append_assistant(self, message: Mapping[str, Any]) -> None:
        self.messages.append(dict(message))

    def append_tool_result(
        self,
        request: ToolRequest,
        bundle: MediaBundle,
    ) -> None:
        self._omit_visible_media()
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": request.call_id,
                "name": request.name,
                "content": [dict(part) for part in bundle.parts],
            }
        )
        self.focused_tool_succeeded = True
        self.tool_calls_used += 1
        self.successful_requests.append(request)
        self.consecutive_invalid_actions = 0
        self.consecutive_no_progress = 0

    def append_protocol_error(
        self,
        code: str,
        message: str,
        *,
        tool_call_id: str = "",
        tool_name: str = "",
        max_invalid_retries: int,
    ) -> None:
        self.invalid_actions += 1
        self.consecutive_invalid_actions += 1
        content = render_error(
            code,
            message,
            retry_exceeded=self.consecutive_invalid_actions > max_invalid_retries,
            consecutive_invalid_actions=self.consecutive_invalid_actions,
            max_invalid_retries=max_invalid_retries,
        )
        if tool_call_id:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content,
                }
            )
        else:
            self.messages.append({"role": "user", "content": content})

    def append_protocol_error_for_calls(
        self,
        calls: Sequence[Mapping[str, Any]],
        code: str,
        message: str,
        *,
        max_invalid_retries: int,
    ) -> None:
        """Close every native tool call so the next OpenAI request stays valid."""

        self.invalid_actions += 1
        self.consecutive_invalid_actions += 1
        content = render_error(
            code,
            message,
            retry_exceeded=self.consecutive_invalid_actions > max_invalid_retries,
            consecutive_invalid_actions=self.consecutive_invalid_actions,
            max_invalid_retries=max_invalid_retries,
        )
        for call in calls:
            function = call.get("function")
            name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "name": name,
                    "content": content,
                }
            )

    def append_control_state(self, *, step: int, max_steps: int) -> None:
        cleaned_messages: list[dict] = []
        for message in self.messages:
            cleaned = dict(message)
            cleaned["content"] = _strip_notice(message.get("content"))
            if cleaned.get("role") == "user" and (cleaned.get("content") is None or cleaned.get("content") == ""):
                continue
            cleaned_messages.append(cleaned)
        self.messages = cleaned_messages
        notice = render_notice(step=step, max_steps=max_steps)
        if self.messages and self.messages[-1].get("role") in {"user", "tool"}:
            _append_control_state(self.messages[-1], notice)
        else:
            self.messages.append({"role": "user", "content": notice})

    def is_duplicate(self, request: ToolRequest) -> bool:
        return any(prior.identity == request.identity for prior in self.successful_requests)

    def append_no_progress(self, request: ToolRequest) -> None:
        self.total_no_progress += 1
        self.consecutive_no_progress += 1
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": request.call_id,
                "name": request.name,
                "content": (
                    f"[NO_PROGRESS] {request.name} repeats an identical successful call. "
                    "Use different evidence or answer from what is already verified."
                ),
            }
        )

    def _omit_visible_media(self) -> None:
        for index, message in enumerate(self.messages):
            has_images, has_other = _has_media(message.get("content"))
            if not (has_images or has_other):
                continue
            header = _first_text(message.get("content")).strip() or "Media content"
            suffix = f" Timestamps: [] {MEDIA_OMITTED}" if has_images else f" {MEDIA_OMITTED}"
            replaced = dict(message)
            replaced["content"] = header + suffix
            self.messages[index] = replaced

    def request_summary(self) -> dict[str, Any]:
        media = {"image": 0, "audio": 0, "video": 0}
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, str) or content is None:
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                kind = str(part.get("type") or "").lower()
                if kind in media:
                    media[kind] += 1
        return {
            "message_count": len(self.messages),
            "visible_media": media,
            "focused_tool_succeeded": self.focused_tool_succeeded,
            "tool_calls_used": self.tool_calls_used,
            "invalid_actions": self.invalid_actions,
            "consecutive_invalid_actions": self.consecutive_invalid_actions,
            "total_no_progress": self.total_no_progress,
            "consecutive_no_progress": self.consecutive_no_progress,
        }


__all__ = ["MEDIA_OMITTED", "AgentSession", "build_initial_user_text"]

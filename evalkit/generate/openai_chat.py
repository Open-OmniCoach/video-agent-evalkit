"""Default single-turn generation over the normalized chat transport."""

from __future__ import annotations

from ..types import EvalCase, GenerationResult, GenerationStatus, SamplingParams


async def generate(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    chat = await runtime.chat("candidate", case.messages, sampling=sampling)
    status = (
        GenerationStatus.TRUNCATED
        if str(chat.finish_reason).lower() in {"length", "max_tokens"}
        else GenerationStatus.COMPLETED
    )
    return GenerationResult(
        status=status,
        response=chat.text,
        assistant_message=chat.message,
        usage=chat.usage,
        timing=chat.timing,
        endpoint=chat.endpoint,
        finish_reason=chat.finish_reason,
    )


__all__ = ["generate"]

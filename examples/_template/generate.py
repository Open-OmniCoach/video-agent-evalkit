"""Custom generation template.

Delete this file when the built-in ``evalkit.generate.openai_chat:generate`` is
sufficient. Scoring deliberately does not enter this function.
"""

from __future__ import annotations

from evalkit.types import EvalCase, GenerationResult, GenerationStatus, SamplingParams


async def generate(
    runtime,
    case: EvalCase,
    sampling: SamplingParams,
) -> GenerationResult:
    result = await runtime.chat(
        "candidate",
        case.messages,
        sampling=sampling,
        route_key=case.uid,
    )
    status = (
        GenerationStatus.TRUNCATED
        if str(result.finish_reason).lower() in {"length", "max_tokens"}
        else GenerationStatus.COMPLETED
    )
    return GenerationResult(
        status=status,
        response=result.text,
        assistant_message=result.message,
        usage=result.usage,
        timing=result.timing,
        endpoint=result.endpoint,
        finish_reason=result.finish_reason,
    )


__all__ = ["generate"]

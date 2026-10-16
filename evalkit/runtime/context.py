"""Explicit per-run resource context."""

from __future__ import annotations

import asyncio
import contextvars
import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import ModelCfg, RunConfig
from ..types import ChatResult, TrajectoryEvent
from .transport import ModelClient, resolve_endpoint

_MODEL_TRACE: contextvars.ContextVar[list[TrajectoryEvent] | None] = contextvars.ContextVar(
    "evalkit_model_trace", default=None
)


class EvalRuntime:
    def __init__(self, models: Mapping[str, ModelClient], *, capture_model_io: bool = False) -> None:
        self.models = dict(models)
        self.capture_model_io = capture_model_io

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "EvalRuntime":
        model_configs: dict[str, ModelCfg] = {"candidate": config.model, **config.aux_models}
        for model in model_configs.values():
            resolve_endpoint(model.endpoint)
        return cls(
            {
                name: ModelClient(name, model.endpoint, default_max_inflight=config.max_active_cases)
                for name, model in model_configs.items()
            },
            capture_model_io=config.trajectory == "trainable",
        )

    def begin_model_trace(self):
        return _MODEL_TRACE.set([]) if self.capture_model_io else None

    def end_model_trace(self, token) -> tuple[TrajectoryEvent, ...]:
        if token is None:
            return ()
        events = tuple(_MODEL_TRACE.get() or ())
        _MODEL_TRACE.reset(token)
        return events

    async def __aenter__(self) -> "EvalRuntime":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def close(self) -> None:
        await asyncio.gather(*(model.close() for model in self.models.values()))

    async def chat(
        self,
        model_name: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        sampling: Mapping[str, Any] | None = None,
        request_body: Mapping[str, Any] | None = None,
        route_key: str | None = None,
    ) -> ChatResult:
        try:
            model = self.models[model_name]
        except KeyError as exc:
            raise KeyError(f"unknown model {model_name!r}; configured: {sorted(self.models)}") from exc
        trace = _MODEL_TRACE.get()
        request = None
        if trace is not None:
            request = await asyncio.to_thread(
                copy.deepcopy,
                {"messages": list(messages), "request_body": dict(request_body or {})},
            )
        result = await model.chat(
            messages,
            sampling=sampling,
            request_body=request_body,
            route_key=route_key,
        )
        if trace is not None:
            trace.append(
                TrajectoryEvent(
                    "model_exchange",
                    len(trace),
                    {
                        "model": model_name,
                        "request": request,
                        "response": {"message": copy.deepcopy(dict(result.message))},
                    },
                )
            )
        return result

    def release_route_key(self, route_key: str) -> None:
        for model in self.models.values():
            model.release_route_key(route_key)

    def metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name, model in self.models.items():
            values = model.stats.to_dict()
            values.update(
                {
                    "model": model.resolved.model,
                    "urls": list(model.resolved.urls),
                    "adapter": model.resolved.adapter,
                    "transport": model.resolved.transport,
                }
            )
            metrics[name] = values
        return metrics


__all__ = ["EvalRuntime"]

"""Persistent OpenAI-compatible model clients and endpoint routing."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from ..config import EndpointCfg
from ..types import ChatResult, GenerationTiming, GenerationUsage

_RETRYABLE_STATUS = frozenset({408, 429})
_RESERVED_BODY_FIELDS = frozenset({"model", "messages"})
_AUDIO_MIME_TO_FORMAT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/aac": "aac",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
}


class EndpointPoolUnavailable(RuntimeError):
    """Raised after one request has exhausted every configured endpoint."""


@dataclass(frozen=True)
class ResolvedEndpoint:
    urls: tuple[str, ...]
    model: str
    adapter: str
    transport: str


@dataclass
class _EndpointState:
    url: str
    inflight: int = 0


@dataclass
class ModelStats:
    requests: int = 0
    failures: int = 0
    retries: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "failures": self.failures,
            "retries": self.retries,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
        }


def _load_service_model(path: str, name: str) -> Mapping[str, Any]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"service manifest not found: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("service manifest top-level must be a mapping")
    if manifest.get("schema_version", 1) != 1:
        raise ValueError(f"unsupported service manifest schema_version {manifest.get('schema_version')!r}")
    models = manifest.get("models") or {}
    if name not in models:
        raise KeyError(f"service manifest has no model {name!r}; available: {sorted(models)}")
    model = models[name]
    if not isinstance(model, Mapping):
        raise ValueError(f"service manifest model {name!r} must be a mapping")
    return model


def resolve_endpoint(config: EndpointCfg) -> ResolvedEndpoint:
    if config.service_manifest:
        service = _load_service_model(config.service_manifest, config.service_model)
        raw_urls = service.get("urls") or ()
        if not isinstance(raw_urls, (list, tuple)):
            raise ValueError("service manifest model urls must be a list")
        urls = tuple(str(url).strip() for url in raw_urls if str(url).strip())
        model = config.model or str(service.get("model") or "")
        adapter = str(service.get("adapter") or config.adapter)
        transport = str(service.get("transport") or config.transport)
    else:
        urls = config.base_urls or ((config.base_url,) if config.base_url else ())
        model = config.model
        adapter = config.adapter
        transport = config.transport
    if not urls:
        raise ValueError("resolved endpoint has no URLs")
    if len(set(urls)) != len(urls):
        raise ValueError("resolved endpoint contains duplicate URLs")
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"model endpoint URL must be HTTP(S), got {url!r}")
    if transport != "openai_chat":
        raise ValueError(f"service transport {transport!r} is not supported")
    if adapter not in {"openai", "vllm_multimodal"}:
        raise ValueError(f"service adapter {adapter!r} is not supported")
    if not model:
        raise ValueError("endpoint model is empty and service manifest did not provide one")
    return ResolvedEndpoint(urls=urls, model=model, adapter=adapter, transport=transport)


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _media_url(raw: str, media_transport: str) -> str:
    if raw.startswith(("http://", "https://", "data:", "file:")):
        return raw
    path = Path(raw).expanduser().resolve()
    return path.as_uri() if media_transport == "file_uri" else _data_uri(path)


def _audio_input(raw: str, *, adapter: str, media_transport: str) -> dict[str, Any]:
    if adapter == "vllm_multimodal":
        return {
            "type": "audio_url",
            "audio_url": {"url": _media_url(raw, media_transport)},
        }
    if raw.startswith(("http://", "https://", "file:", "data:")):
        return {"type": "audio_url", "audio_url": {"url": raw}}
    path = Path(raw).expanduser().resolve()
    mime, _ = mimetypes.guess_type(str(path))
    fmt = _AUDIO_MIME_TO_FORMAT.get(mime or "", path.suffix.lstrip(".").lower() or "wav")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}}


def adapt_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    adapter: str,
    media_transport: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        adapted = dict(message)
        adapted["role"] = str(message.get("role", "user"))
        content = message.get("content", "")
        if content is None:
            adapted["content"] = None
            out.append(adapted)
            continue
        if isinstance(content, str):
            adapted["content"] = content
            out.append(adapted)
            continue
        parts: list[dict[str, Any]] = []
        for raw_part in content or ():
            part = dict(raw_part)
            kind = str(part.get("type", "")).lower()
            if kind == "text":
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            elif kind == "image":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _media_url(str(part["image"]), media_transport)},
                    }
                )
            elif kind == "video":
                parts.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": _media_url(str(part["video"]), media_transport)},
                    }
                )
            elif kind == "audio":
                parts.append(
                    _audio_input(
                        str(part["audio"]),
                        adapter=adapter,
                        media_transport=media_transport,
                    )
                )
            else:
                parts.append(part)
        adapted["content"] = parts
        out.append(adapted)
    return out


def _payload(
    resolved: ResolvedEndpoint,
    config: EndpointCfg,
    messages: Sequence[Mapping[str, Any]],
    sampling: Mapping[str, Any],
    request_body: Mapping[str, Any],
) -> dict[str, Any]:
    for label, values in (
        ("sampling", sampling),
        ("endpoint.extra_body", config.extra_body),
        ("request_body", request_body),
    ):
        reserved = _RESERVED_BODY_FIELDS.intersection(values)
        if reserved:
            raise ValueError(f"{label} cannot override reserved fields: {sorted(reserved)}")
        if values.get("stream"):
            raise ValueError("streaming responses are not supported by the evaluation transport")
    params = {str(key): value for key, value in sampling.items() if value is not None}
    if "max_new_tokens" in params and "max_tokens" not in params:
        params["max_tokens"] = params.pop("max_new_tokens")
    params.setdefault("max_tokens", 1024)
    body: dict[str, Any] = {
        "model": resolved.model,
        "messages": adapt_messages(messages, adapter=resolved.adapter, media_transport=config.media_transport),
        **params,
    }
    body.update(config.extra_body)
    body.update(request_body)
    return body


def _response_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return str(content)


def _encode_json_body(body: Mapping[str, Any]) -> bytes:
    return json.dumps(
        body,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class ModelClient:
    """One logical model backed by one or more OpenAI-compatible endpoints."""

    def __init__(
        self,
        name: str,
        config: EndpointCfg,
        *,
        default_max_inflight: int,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.resolved = resolve_endpoint(config)
        self.endpoints = [_EndpointState(_chat_url(url)) for url in self.resolved.urls]
        max_inflight = config.max_inflight or default_max_inflight
        self._admission = asyncio.Semaphore(max(1, max_inflight))
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            limits=httpx.Limits(
                max_connections=max(1, max_inflight),
                max_keepalive_connections=max(1, max_inflight),
            ),
            transport=http_transport,
        )
        self._bindings: dict[str, int] = {}
        self._selection_lock = asyncio.Lock()
        self._cursor = 0
        self.stats = ModelStats()

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(endpoint.url for endpoint in self.endpoints)

    async def close(self) -> None:
        await self._client.aclose()

    def release_route_key(self, route_key: str) -> None:
        self._bindings.pop(route_key, None)

    async def _reserve(
        self,
        route_key: str | None,
        excluded: set[int],
    ) -> int:
        async with self._selection_lock:
            if route_key is not None:
                bound = self._bindings.get(route_key)
                if bound is not None and bound not in excluded:
                    self.endpoints[bound].inflight += 1
                    return bound
            available = [index for index in range(len(self.endpoints)) if index not in excluded]
            if not available:
                available = list(range(len(self.endpoints)))
            if route_key is None:
                loads = [endpoint.inflight for endpoint in self.endpoints]
            else:
                loads = [0] * len(self.endpoints)
                for bound in self._bindings.values():
                    loads[bound] += 1
            minimum = min(loads[index] for index in available)
            candidates = [index for index in available if loads[index] == minimum]
            index = candidates[self._cursor % len(candidates)]
            self._cursor += 1
            if route_key is not None:
                self._bindings[route_key] = index
            self.endpoints[index].inflight += 1
            return index

    async def _release(self, index: int) -> None:
        async with self._selection_lock:
            self.endpoints[index].inflight -= 1

    def _headers(self, route_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = self.config.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update({str(key): str(value) for key, value in self.config.headers.items()})
        if route_key and self.config.route_key_header:
            headers[self.config.route_key_header] = route_key
        return headers

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        sampling: Mapping[str, Any] | None = None,
        request_body: Mapping[str, Any] | None = None,
        route_key: str | None = None,
    ) -> ChatResult:
        body = await asyncio.to_thread(
            _payload,
            self.resolved,
            self.config,
            messages,
            sampling or {},
            request_body or {},
        )
        content = await asyncio.to_thread(_encode_json_body, body)
        excluded: set[int] = set()
        last_error: Exception | None = None
        attempts = self.config.request_retries + 1
        async with self._admission:
            for attempt in range(attempts):
                index = await self._reserve(route_key, excluded)
                endpoint = self.endpoints[index]
                started = time.monotonic()
                self.stats.requests += 1
                try:
                    response = await self._client.post(
                        endpoint.url,
                        content=content,
                        headers=self._headers(route_key),
                    )
                    if response.status_code in _RETRYABLE_STATUS or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"retryable model response {response.status_code}: {response.text[:400]}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices") or []
                    if not choices:
                        raise ValueError(f"model endpoint {endpoint.url} returned no choices")
                    choice = choices[0]
                    message = dict(choice.get("message") or {})
                    message.setdefault("role", "assistant")
                    text = _response_text(message.get("content"))
                    tool_calls = tuple(message.get("tool_calls") or ())
                    usage_data = data.get("usage") or {}
                    prompt_details = usage_data.get("prompt_tokens_details") or {}
                    usage = GenerationUsage(
                        prompt_tokens=int(usage_data.get("prompt_tokens") or 0),
                        completion_tokens=int(usage_data.get("completion_tokens") or 0),
                        cached_tokens=int(prompt_details.get("cached_tokens") or usage_data.get("cached_tokens") or 0),
                    )
                    timing = GenerationTiming(latency_ms=(time.monotonic() - started) * 1000.0)
                    self.stats.latency_ms += timing.latency_ms
                    self.stats.prompt_tokens += usage.prompt_tokens
                    self.stats.completion_tokens += usage.completion_tokens
                    self.stats.cached_tokens += usage.cached_tokens
                    if route_key is not None:
                        self._bindings[route_key] = index
                    return ChatResult(
                        message=message,
                        text=text,
                        tool_calls=tool_calls,
                        finish_reason=choice.get("finish_reason"),
                        usage=usage,
                        timing=timing,
                        endpoint=endpoint.url,
                        raw=data,
                    )
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError):
                        status_code = exc.response.status_code
                        if status_code not in _RETRYABLE_STATUS and status_code < 500:
                            self.stats.failures += 1
                            raise
                    last_error = exc
                    self.stats.failures += 1
                    excluded.add(index)
                    if route_key is not None:
                        self._bindings.pop(route_key, None)
                    if attempt + 1 >= attempts:
                        if len(excluded) >= len(self.endpoints):
                            raise EndpointPoolUnavailable(
                                f"all endpoints for model {self.name!r} are unavailable"
                            ) from exc
                        raise
                    self.stats.retries += 1
                    await asyncio.sleep(self.config.retry_backoff * (2**attempt))
                except Exception:
                    self.stats.failures += 1
                    raise
                finally:
                    await self._release(index)
        assert last_error is not None
        raise last_error


__all__ = [
    "EndpointPoolUnavailable",
    "ModelClient",
    "ModelStats",
    "ResolvedEndpoint",
    "adapt_messages",
    "resolve_endpoint",
]

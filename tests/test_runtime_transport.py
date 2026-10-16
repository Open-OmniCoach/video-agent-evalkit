"""Transport adaptation, routing, stickiness, and failover.

Run: python tests/test_runtime_transport.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.config import EndpointCfg
from evalkit.runtime import EvalRuntime
from evalkit.runtime.transport import (
    EndpointPoolUnavailable,
    ModelClient,
    adapt_messages,
    resolve_endpoint,
)


def _response(request: httpx.Request, text: str = "ok") -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        },
    )


def _test_message_adapters() -> None:
    with TemporaryDirectory() as tmp:
        image = Path(tmp) / "x.jpg"
        audio = Path(tmp) / "x.wav"
        image.write_bytes(b"jpeg")
        audio.write_bytes(b"wave")
        messages = (
            {
                "role": "user",
                "name": "caller",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "audio", "audio": str(audio)},
                    {"type": "text", "text": "question"},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        )

        inline = adapt_messages(messages, adapter="openai", media_transport="inline")
        assert inline[0]["name"] == "caller", inline
        assert inline[0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert inline[0]["content"][1]["type"] == "input_audio", inline
        assert inline[1]["tool_calls"][0]["id"] == "call-1", inline
        assert inline[1]["content"] is None, inline
        assert inline[2]["tool_call_id"] == "call-1", inline

        file_uri = adapt_messages(messages[:1], adapter="vllm_multimodal", media_transport="file_uri")
        assert file_uri[0]["content"][0]["image_url"]["url"] == image.resolve().as_uri()
        assert file_uri[0]["content"][1]["audio_url"]["url"] == audio.resolve().as_uri()


async def _test_sticky_and_balanced_routing() -> None:
    hosts: list[str] = []
    headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host or "")
        headers.append(request.headers.get("x-route-key"))
        body = json.loads(request.content)
        assert body["max_tokens"] == 7, body
        assert "max_new_tokens" not in body, body
        await asyncio.sleep(0.01)
        return _response(request, request.url.host or "")

    config = EndpointCfg(
        base_urls=("http://a.test/v1", "http://b.test/v1"),
        model="candidate",
        route_key_header="x-route-key",
        request_retries=0,
    )
    client = ModelClient(
        "candidate",
        config,
        default_max_inflight=4,
        http_transport=httpx.MockTransport(handler),
    )
    try:
        first = await client.chat(
            ({"role": "user", "content": "one"},),
            sampling={"max_new_tokens": 7},
            route_key="same",
        )
        second = await client.chat(
            ({"role": "user", "content": "two"},),
            sampling={"max_new_tokens": 7},
            route_key="same",
        )
        assert first.text == second.text == "a.test", (first, second)

        await client.chat(
            ({"role": "user", "content": "unbound"},),
            sampling={"max_new_tokens": 7},
        )
        other = await client.chat(
            ({"role": "user", "content": "other"},),
            sampling={"max_new_tokens": 7},
            route_key="other",
        )
        assert other.text == "b.test", other

        client.release_route_key("same")
        client.release_route_key("other")
        await asyncio.gather(
            client.chat(({"role": "user", "content": "three"},), sampling={"max_new_tokens": 7}),
            client.chat(({"role": "user", "content": "four"},), sampling={"max_new_tokens": 7}),
        )
        assert set(hosts[-2:]) == {"a.test", "b.test"}, hosts
        assert headers[:2] == ["same", "same"], headers
    finally:
        await client.close()


async def _test_failover_and_nonretryable_error() -> None:
    hosts: list[str] = []

    async def failover_handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host or "")
        if request.url.host == "a.test":
            return httpx.Response(503, request=request, text="busy")
        return _response(request, "recovered")

    config = EndpointCfg(
        base_urls=("http://a.test/v1", "http://b.test/v1"),
        model="candidate",
        request_retries=1,
        retry_backoff=0,
    )
    client = ModelClient(
        "candidate",
        config,
        default_max_inflight=2,
        http_transport=httpx.MockTransport(failover_handler),
    )
    try:
        result = await client.chat(({"role": "user", "content": "x"},))
        assert result.text == "recovered", result
        assert hosts == ["a.test", "b.test"], hosts
        assert client.stats.requests == 2 and client.stats.retries == 1, client.stats
    finally:
        await client.close()

    async def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, text="down")

    client = ModelClient(
        "candidate",
        config,
        default_max_inflight=2,
        http_transport=httpx.MockTransport(unavailable),
    )
    try:
        try:
            await client.chat(({"role": "user", "content": "x"},))
        except EndpointPoolUnavailable:
            pass
        else:
            raise AssertionError("exhausting every endpoint must stop the run")
    finally:
        await client.close()

    calls = 0

    async def bad_request(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, request=request, text="invalid")

    client = ModelClient(
        "candidate",
        config,
        default_max_inflight=2,
        http_transport=httpx.MockTransport(bad_request),
    )
    try:
        try:
            await client.chat(({"role": "user", "content": "x"},))
        except httpx.HTTPStatusError as exc:
            assert exc.response.status_code == 400, exc
        else:
            raise AssertionError("400 response must fail")
        assert calls == 1, calls
    finally:
        await client.close()


async def _test_request_guardrails() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(request)

    client = ModelClient(
        "candidate",
        EndpointCfg(base_url="http://guard.test/v1", model="candidate", request_retries=0),
        default_max_inflight=1,
        http_transport=httpx.MockTransport(handler),
    )
    try:
        for sampling, request_body in (
            ({"model": "other"}, None),
            ({"stream": True}, None),
            ({}, {"messages": []}),
        ):
            try:
                await client.chat(
                    ({"role": "user", "content": "x"},),
                    sampling=sampling,
                    request_body=request_body,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("reserved or streaming request fields must be rejected")
        assert calls == 0, calls
    finally:
        await client.close()


async def _test_trainable_trace() -> None:
    client = ModelClient(
        "candidate",
        EndpointCfg(base_url="http://trace.test/v1", model="candidate", request_retries=0),
        default_max_inflight=1,
        http_transport=httpx.MockTransport(_response),
    )
    runtime = EvalRuntime({"candidate": client}, capture_model_io=True)
    messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    token = runtime.begin_model_trace()
    try:
        await runtime.chat("candidate", messages, request_body={"tools": [{"type": "function"}]})
        events = runtime.end_model_trace(token)
        token = None
        messages[0]["content"][0]["text"] = "mutated"
        assert events[0].kind == "model_exchange"
        assert events[0].data["request"]["messages"][0]["content"][0]["text"] == "x"
        assert events[0].data["request"]["request_body"]["tools"]
        assert events[0].data["response"]["message"]["content"] == "ok"

        async def capture(text: str):
            local_token = runtime.begin_model_trace()
            await runtime.chat("candidate", [{"role": "user", "content": text}])
            return runtime.end_model_trace(local_token)

        left, right = await asyncio.gather(capture("left"), capture("right"))
        assert left[0].data["request"]["messages"][0]["content"] == "left"
        assert right[0].data["request"]["messages"][0]["content"] == "right"
    finally:
        if token is not None:
            runtime.end_model_trace(token)
        await runtime.close()


def _test_manifest_resolution() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "services.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": {
                        "candidate": {
                            "urls": ["http://one:8000/v1", "http://two:8000/v1"],
                            "model": "served",
                            "adapter": "vllm_multimodal",
                            "transport": "openai_chat",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        resolved = resolve_endpoint(EndpointCfg(service_manifest=str(path), service_model="candidate"))
        assert resolved.urls == ("http://one:8000/v1", "http://two:8000/v1"), resolved
        assert resolved.model == "served" and resolved.adapter == "vllm_multimodal", resolved


def main() -> int:
    _test_message_adapters()
    _test_manifest_resolution()
    asyncio.run(_test_sticky_and_balanced_routing())
    asyncio.run(_test_failover_and_nonretryable_error())
    asyncio.run(_test_request_guardrails())
    asyncio.run(_test_trainable_trace())
    print("runtime transport OK: media, tools, routing, failover, manifest")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())

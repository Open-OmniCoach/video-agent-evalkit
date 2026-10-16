"""Shared omni agent: a synthetic tool turn; no model or media IO.

Run: python tests/test_omni_agent.py
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from dataclasses import replace
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.types import ChatResult, EvalCase, GenerationStatus, ProblemType
from evalkit.config import load_run_config
from evalkit.runtime.datasource import load_specs
from evalkit.runtime.plugins import load_callable
from evalkit.serve import load_serve_config
from evalkit.serve.launcher import dry_run
from examples.omni_agent import generate, media, protocol, session
from examples.omni_agent.policy import AgentPolicy


def fixture():
    case = EvalCase(
        uid="synthetic",
        dataset="fixture",
        index=0,
        messages=(),
        target="A",
        problem_type=ProblemType.MULTIPLE_CHOICE,
        metadata={"question": "Which object moves?", "options": ["A. Ball", "B. Box"]},
    )
    info = media.VideoInfo(Path("video.mp4"), Path("audio.wav"), 4000, 224, 224, 2, 8000, True)
    initial = media.MediaBundle(
        "initial",
        "Initial frames",
        ({"type": "image", "image": "initial.jpg"},),
        {"audio": False, "frame_count": 1, "timestamps": [0.0]},
    )
    return case, info, initial


async def exercise_agent() -> None:
    case, info, initial = fixture()
    prompt = session.build_initial_user_text(case, info, initial)
    assert "Full audio is not attached" in prompt
    assert "get_audio and get_clip" in prompt
    call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "get_frames",
            "arguments": '{"start":"1s","end":"2s","level":"coarse"}',
        },
    }
    prefix = "<observation>A ball is visible.</observation>\n<think>Inspect its motion.</think>"
    answer = "<observation>The ball moved.</observation>\n<think>The evidence supports A.</think>\n<answer>A</answer>"
    responses = iter(
        [
            ChatResult(
                message={"role": "assistant", "content": prefix, "tool_calls": [call]},
                text=prefix,
                tool_calls=(call,),
                finish_reason="tool_calls",
            ),
            ChatResult(message={"role": "assistant", "content": answer}, text=answer, finish_reason="stop"),
        ]
    )
    requests = []

    async def chat(model, messages, **kwargs):
        assert model == "candidate" and kwargs["route_key"] == case.uid
        requests.append(copy.deepcopy(messages))
        return next(responses)

    tool = media.MediaBundle(
        "frames",
        "Focused evidence",
        ({"type": "image", "image": "focused.jpg"},),
        {"frame_count": 1, "timestamps": [1.0]},
    )
    fake_media = SimpleNamespace(info=info, execute=AsyncMock(return_value=tool))
    result = await generate._generate_with_media(
        SimpleNamespace(chat=chat),
        case,
        {"max_tokens": 4096},
        policy=AgentPolicy(media_cache_id="fixture", max_turns=2),
        media=fake_media,
        initial=initial,
        preparation_ms=0,
        trace_complete=True,
    )
    assert result.status == GenerationStatus.COMPLETED and result.response == answer, result
    assert fake_media.execute.await_count == 1 and len(requests) == 2
    images = [
        part["image"]
        for message in requests[1]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image"
    ]
    assert images == ["focused.jpg"], images
    assert {event.kind for event in result.trajectory} >= {"tool_call", "tool_result", "agent_summary"}
    parsed = protocol.require_single_tool_call([call], duration=4000, has_audio=True)
    assert (parsed.start, parsed.end) == (1, 2)


def test_agent() -> None:
    # Any accidental real decoder/process-control call fails before it can run.
    with ExitStack() as stack:
        for name in (
            "os.kill",
            "os.killpg",
            "subprocess.Popen",
            "subprocess.run",
            "asyncio.create_subprocess_exec",
            "asyncio.create_subprocess_shell",
        ):
            stack.enter_context(patch(name, side_effect=AssertionError("unexpected process IO")))
        asyncio.run(exercise_agent())


@pytest.mark.parametrize("initial_frames", [0, -1, 257, 1.5, True, "96", None])
def test_invalid_initial_frames(initial_frames):
    with pytest.raises(ValueError, match="initial_frames"):
        AgentPolicy.from_metadata({"budget": {"media_cache_id": "fixture", "initial_frames": initial_frames}})


@pytest.mark.parametrize(
    "name", ["max_turns", "max_invalid_retries", "max_context_tokens", "max_output_tokens", "initial_audio_max_seconds"]
)
@pytest.mark.parametrize("value", [True, 1.5, "96", None, 0, -1])
def test_integer_budgets_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        AgentPolicy.from_metadata({"budget": {"media_cache_id": "fixture", name: value}})


@pytest.mark.parametrize("initial_frames,cached_frames", [(96, 256), (256, 256), (96, 12), (1, 256)])
def test_initial_view_selects_cached_frames(tmp_path, initial_frames, cached_frames):
    policy = AgentPolicy.from_metadata({"budget": {"media_cache_id": "fixture", "initial_frames": initial_frames}})
    parent_timestamps = media.select_initial_timestamps(1000, fps=0.5, max_frames=256)[:cached_frames]
    payloads = [f"cached-frame-{index}".encode() for index in range(cached_frames)]
    entries = []
    offset = 0
    for timestamp, payload in zip(parent_timestamps, payloads, strict=True):
        entries.append([timestamp, offset, len(payload)])
        offset += len(payload)
    manifest = {"initial": {"index": entries, "width": 224, "height": 224}}
    original_manifest = copy.deepcopy(manifest)
    cache = tmp_path / "video"
    cache.mkdir()
    source = cache / "initial.mjpeg"
    source.write_bytes(b"".join(payloads))
    audio = cache / "audio.wav"
    audio.write_bytes(b"cached-audio")
    session_view = media.MediaSession(
        info=media.VideoInfo(Path("video.mp4"), audio, 1000, 224, 224, 2, 2000, True),
        policy=policy,
        namespace=tmp_path,
        media_id="video",
        manifest=manifest,
        cache_hit=True,
    )

    async def exercise():
        bundle = await session_view.initial_bundle()
        images = [Path(part["image"]) for part in bundle.parts if part["type"] == "image"]
        if cached_frames <= initial_frames:
            expected = list(range(cached_frames))
        elif initial_frames == 1:
            expected = [0]
        else:
            expected = [round(index * (cached_frames - 1) / (initial_frames - 1)) for index in range(initial_frames)]
        assert len(images) == min(initial_frames, cached_frames)
        assert [path.read_bytes() for path in images] == [payloads[index] for index in expected]
        assert len(list(images[0].parent.glob("*.jpg"))) == len(expected)
        assert bundle.manifest["timestamps"] == [parent_timestamps[index] for index in expected]
        assert bundle.manifest["frame_count"] == len(expected)
        assert f"selected_frames={len(expected)}" in bundle.header
        assert [part["audio"] for part in bundle.parts if part["type"] == "audio"] == [str(audio)]
        for part, following in zip(bundle.parts, bundle.parts[1:]):
            if part["type"] == "image":
                continue
            if following["type"] == "image":
                index = images.index(Path(following["image"]))
                assert str(int(parent_timestamps[expected[index]])) in part["text"]
        assert manifest == original_manifest
        await session_view.close()
        assert all(not path.exists() for path in images)
        assert source.read_bytes() == b"".join(payloads) and audio.exists()

    asyncio.run(exercise())


def test_initial_view_limit_does_not_change_cache_contract():
    policies = [
        AgentPolicy.from_metadata({"budget": {"media_cache_id": "shared", "initial_frames": count}})
        for count in (96, 256)
    ]
    assert media.cache_contract(policies[0]) == media.cache_contract(policies[1])
    assert media.cache_contract(policies[0])["preprocess"]["initial"]["max_frames"] == 256


@pytest.mark.parametrize(
    "model,frames,context,backend",
    [
        ("qwen2_5_omni", 96, 131072, "vllm"),
        ("qwen3_omni", 256, 65536, "sglang"),
    ],
)
def test_model_configs(tmp_path, monkeypatch, model, frames, context, backend):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OMNI_AGENT_MEDIA_CACHE_ROOT", str(tmp_path / "cache"))
    config_dir = REPO / "examples/omni_agent/configs" / model
    run = load_run_config(config_dir / "run.yaml")
    serve = load_serve_config(config_dir / "serve.yaml")
    assert load_callable(run.model.generate_fn) is generate.generate
    assert run.model.endpoint.service_manifest == serve.manifest_path
    assert Path(run.registry_path) == REPO / "configs/registry.yaml"
    assert Path(run.data_root) == REPO / "data"
    assert Path(run.out_dir) == REPO / "runs" / f"{model}_agent" / "predictions"
    assert serve.working_dir == str(REPO)
    policy = AgentPolicy.from_metadata(run.defaults)
    assert policy.initial_frames == frames and policy.max_context_tokens == context
    specs = load_specs(run)
    assert all(bench.name in specs for bench in run.eval_datasets)
    assert specs["sample_mcq"].annotation_path == REPO / "data/sample_mcq.jsonl"
    process = dry_run(serve)["processes"][0]
    assert serve.models["candidate"].backend == backend
    command = process["command"]
    template = REPO / command[command.index("--chat-template") + 1]
    assert template == REPO / "examples/omni_agent/templates" / f"{model}.jinja"
    assert template.is_file()
    if backend == "vllm":
        overrides = json.loads(command[command.index("--hf-overrides") + 1])
        text = overrides["thinker_config"]["text_config"]
        assert text["max_position_embeddings"] == context
        assert text["rope_parameters"] == {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
        }
        assert run.case_max_attempts == 1 and run.model.endpoint.request_retries == 0
    else:
        assert run.case_max_attempts == 2 and run.model.endpoint.request_retries == 3


def test_models_share_cache_namespace(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_AGENT_MEDIA_CACHE_ROOT", str(tmp_path))
    case, _, _ = fixture()
    configs = REPO / "examples/omni_agent/configs"
    policies = [
        AgentPolicy.from_metadata(load_run_config(configs / model / "run.yaml").defaults)
        for model in ("qwen2_5_omni", "qwen3_omni")
    ]
    assert media.cache_namespace(case, policies[0]) == media.cache_namespace(case, policies[1])
    assert media.cache_contract(policies[0]) == media.cache_contract(policies[1])
    assert media.cache_namespace(case, policies[0]).parent.parent == tmp_path
    monkeypatch.delenv("OMNI_AGENT_MEDIA_CACHE_ROOT")
    monkeypatch.setattr(media.tempfile, "gettempdir", lambda: str(tmp_path))
    assert media.media_cache_dir() == tmp_path / "evalkit-media/omni-agent"


@pytest.mark.parametrize("trace_complete,preserve", [("1", "0"), ("0", "1")])
def test_generate_environment_controls(tmp_path, monkeypatch, trace_complete, preserve):
    monkeypatch.setenv("OMNI_AGENT_TRACE_COMPLETE", trace_complete)
    monkeypatch.setenv("OMNI_AGENT_PRESERVE_TRACE_MEDIA", preserve)
    case, info, initial = fixture()
    case = replace(case, metadata={**case.metadata, "budget": {"media_cache_id": "fixture"}})
    answer = "<observation>A ball moved.</observation>\n<think>The ball is A.</think>\n<answer>A</answer>"
    runtime = SimpleNamespace(
        chat=AsyncMock(
            return_value=ChatResult(
                message={"role": "assistant", "content": answer},
                text=answer,
                finish_reason="stop",
            )
        )
    )
    case_view = tmp_path / "case-view"
    case_view.mkdir()
    view = media.MediaSession(
        info=info,
        policy=AgentPolicy(media_cache_id="fixture"),
        namespace=tmp_path,
        media_id="fixture",
        manifest={},
        cache_hit=True,
    )
    view._view_root = case_view
    with (
        patch.object(media.MediaSession, "open", new=AsyncMock(return_value=view)),
        patch.object(view, "initial_bundle", new=AsyncMock(return_value=initial)),
    ):
        result = asyncio.run(generate.generate(runtime, case, {"max_tokens": 4096}))
    assert result.status == GenerationStatus.COMPLETED and result.response == answer
    initial_event = next(event for event in result.trajectory if event.kind == "media_initial")
    assert ("parts" in initial_event.data) == (trace_complete == "1")
    assert case_view.exists() == (preserve == "1")


def main() -> int:
    test_agent()
    print("Omni agent: tools and conversation media replacement OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

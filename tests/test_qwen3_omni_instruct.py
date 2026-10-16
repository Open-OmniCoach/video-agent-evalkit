"""Instruct recipes and native/image/video-sequence media with real FFmpeg."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.config import load_run_config
from evalkit.environment import host_tool_command
from evalkit.runtime.datasource import load_specs
from evalkit.runtime.plugins import load_callable
from evalkit.serve import load_serve_config
from evalkit.serve.launcher import dry_run
from evalkit.types import ChatResult, EvalCase, ProblemType
from examples.qwen3_omni_instruct import generate, local_video, media


class FixedRuntime:
    async def chat(self, model, messages, **kwargs):
        assert model == "candidate"
        self.messages = messages
        self.kwargs = kwargs
        return ChatResult(message={"role": "assistant", "content": "A"}, text="A", finish_reason="stop")


def test_instruct_media(tmp_path, monkeypatch):
    try:
        ffmpeg = host_tool_command("ffmpeg")
        host_tool_command("ffprobe")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    video = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x96:rate=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "6",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        timeout=30,
    )
    cache = tmp_path / "cache"
    monkeypatch.setenv(media.CACHE_ENV, str(cache))
    case = EvalCase(
        uid="instruct-fixture",
        dataset="fixture",
        index=0,
        messages=(
            {
                "role": "user",
                "content": [{"type": "video", "video": str(video)}, {"type": "text", "text": "Choose A or B."}],
            },
        ),
        target="A",
        problem_type=ProblemType.MULTIPLE_CHOICE,
    )

    async def exercise():
        runtime = FixedRuntime()
        native = await generate.generate(runtime, case, {"temperature": 0})
        staged = Path(runtime.messages[0]["content"][0]["video"])
        assert staged != video and cache in staged.parents
        assert staged.read_bytes() == video.read_bytes()
        assert native.trajectory[0].kind == "media_staging"

        first, second = await asyncio.gather(
            local_video.prepare_video(str(video)), local_video.prepare_video(str(video))
        )
        assert first is second
        assert local_video.frame_count(10, 1) == 10
        assert local_video.frame_count(10, 2) == 20
        assert local_video.frame_count(200, 2) == local_video.MAX_FRAMES
        assert local_video.effective_fps(200, local_video.MAX_FRAMES, 2) == 1.28
        assert local_video.frame_timestamps(10, 20, 2)[-1] == 9.5
        assert local_video.frame_timestamps(200, local_video.MAX_FRAMES, 2)[-1] < 200

        images = await local_video.generate_timestamped_images(runtime, case, {})
        content = runtime.messages[0]["content"]
        assert [p["text"] for p in content if p["type"] == "text"][:6] == [
            f"<{index:.2f} seconds>" for index in range(6)
        ]
        frames = [p["image"] for p in content if p["type"] == "image"]
        assert len(frames) == 6 and all(frame.startswith("data:image/jpeg;base64,") for frame in frames)
        assert [p["audio"] for p in content if p["type"] == "audio"] == [str(staged)]
        assert runtime.kwargs["request_body"]["mm_processor_kwargs"] == {
            "min_pixels": local_video.MIN_PIXELS,
            "max_pixels": local_video.MAX_PIXELS,
            "use_audio_in_video": False,
        }
        assert images.trajectory[0].data["frame_count"] == 6
        assert images.trajectory[0].data["audio_derived_from_video"] is True
        assert images.trajectory[0].data["nominal_sample_fps"] == 1

        twice = await local_video.generate_timestamped_images_2fps(runtime, case, {})
        assert twice.trajectory[0].data["frame_count"] == 12
        assert twice.trajectory[0].data["nominal_sample_fps"] == 2

        sequence = await local_video.generate_video_sequence(runtime, case, {})
        videos = [p["video"] for p in runtime.messages[0]["content"] if p["type"] == "video"]
        assert len(videos) == 1 and videos[0].startswith("data:video/jpeg;base64,")
        assert len(videos[0].split(",", 1)[1].split(",")) == 6
        assert runtime.kwargs["request_body"]["media_io_kwargs"]["video"] == {
            "num_frames": -1,
            "fps": 1.0,
            "duration": 6.0,
            "do_sample_frames": False,
        }
        assert sequence.trajectory[0].data["frames_sha256"] == images.trajectory[0].data["frames_sha256"]

    asyncio.run(exercise())


@pytest.mark.parametrize("recipe", ["smoke", "dailyomni", "omnivideobench", "lvomnibench", "worldsense", "timelens"])
def test_instruct_configs(tmp_path, monkeypatch, recipe):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(media.CACHE_ENV, str(tmp_path / "cache"))
    configs = REPO / "examples/qwen3_omni_instruct/configs"
    run = load_run_config(configs / "run" / f"{recipe}.yaml")
    serve = load_serve_config(configs / "serve.yaml")
    assert run.model.endpoint.service_manifest == serve.manifest_path
    assert Path(run.registry_path) == REPO / "configs/registry.yaml"
    assert Path(run.data_root) == REPO / "data"
    assert Path(serve.working_dir) == REPO
    specs = load_specs(run)
    for bench in run.eval_datasets:
        assert bench.name in specs
        assert callable(load_callable(run.bench_generate_fn(bench)))
    command = dry_run(serve)["processes"][0]["command"]
    assert command[command.index("--max-model-len") + 1] == "65536"
    assert command[command.index("--allowed-local-media-path") + 1] == str(tmp_path / "cache")

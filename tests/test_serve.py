"""Serve topology validation without launching a backend.

Run: python tests/test_serve.py
"""

from __future__ import annotations

import sys
import asyncio
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evalkit.serve import load_serve_config
from evalkit.serve.launcher import dry_run
from evalkit.serve import launcher
from evalkit.serve.config import ReplicaConfig, ServeConfig, ServeModelConfig


def media_config(tmp_path, args):
    return ServeConfig(
        models={
            "candidate": ServeModelConfig(
                name="candidate",
                backend="vllm",
                model="fixture",
                served_model_name="fixture",
                adapter="vllm_multimodal",
                replicas=(ReplicaConfig(gpus=(0,), port=8000),),
                args=args,
            )
        },
        working_dir=str(tmp_path),
        manifest_path=str(tmp_path / "manifest.json"),
    )


def test_media_cache_path_from_environment_exists_before_launch(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    cache = real / "cache"
    monkeypatch.setenv("TEST_MEDIA_CACHE", str(alias / "cache"))
    monkeypatch.setattr(launcher, "backend_command", lambda backend: (sys.executable,))
    config = media_config(tmp_path, ("--allowed-local-media-path", "${TEST_MEDIA_CACHE}"))
    command = dry_run(config)["processes"][0]["command"]
    assert command[-1] == str(cache)
    assert not cache.exists(), "dry-run must not create directories"

    class LaunchObserved(Exception):
        pass

    async def launch(*command, **kwargs):
        assert cache.is_dir(), "vLLM checks the directory during startup"
        assert command[-1] == str(cache)
        raise LaunchObserved

    monkeypatch.setattr(launcher.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(LaunchObserved):
        asyncio.run(launcher.ServeSupervisor(config).start())


def test_unset_media_environment_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MEDIA_CACHE", raising=False)
    config = media_config(tmp_path, ("--allowed-local-media-path", "${TEST_MEDIA_CACHE}"))
    launch = AsyncMock()
    monkeypatch.setattr(launcher.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(ValueError, match="TEST_MEDIA_CACHE"):
        asyncio.run(launcher.ServeSupervisor(config).start())
    launch.assert_not_called()


@pytest.mark.parametrize("args", [("--allowed-local-media-path", "cache"), ("--allowed-local-media-path=cache",)])
def test_relative_media_path_uses_service_working_directory(tmp_path, monkeypatch, args):
    config = media_config(tmp_path, args)
    monkeypatch.chdir(REPO)
    command = dry_run(config)["processes"][0]["command"]
    assert str(tmp_path / "cache") in command[-1]


def test_arguments_use_replica_environment_without_expanding_other_text(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VALUE", "inherited")
    literal = '{"pattern":"${TEST_VALUE}"}'
    config = media_config(tmp_path, ("--example-value", "${TEST_VALUE}", "--json", literal))
    model = config.models["candidate"]
    model = replace(model, env={"TEST_VALUE": "model"})
    config = replace(config, models={"candidate": model})
    assert dry_run(config)["processes"][0]["command"][-4:] == ["--example-value", "model", "--json", literal]
    replica = replace(model.replicas[0], env={"TEST_VALUE": "replica"})
    config = replace(config, models={"candidate": replace(model, replicas=(replica,))})
    assert dry_run(config)["processes"][0]["command"][-4:] == ["--example-value", "replica", "--json", literal]


def _test_vllm_topology() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_path = root / "serve.yaml"
        config_path.write_text(
            "manifest_path: services.json\n"
            "working_dir: .\n"
            "advertise_host: 127.0.0.1\n"
            "models:\n"
            "  candidate:\n"
            "    backend: vllm\n"
            "    model: /path/to/model\n"
            "    replicas:\n"
            "      - {gpus: [0], port: 8000}\n"
        )
        rendered = dry_run(load_serve_config(config_path))
        assert Path(rendered["working_dir"]) == root.resolve(), rendered
        assert len(rendered["processes"]) == 1, rendered
        process = rendered["processes"][0]
        assert process["command"][1:3] == ["serve", "/path/to/model"], process
        assert process["cuda_visible_devices"] == "0", process
        assert process["base_url"] == "http://127.0.0.1:8000/v1", process


def _test_sglang_command_and_validation() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_path = root / "serve.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "manifest_path: services.json",
                    "working_dir: .",
                    "models:",
                    "  candidate:",
                    "    backend: sglang",
                    "    model: /models/test",
                    "    replicas:",
                    "      - {gpus: [0, 1], port: 30000}",
                    "    args: [--tp-size, '2', --enable-metrics]",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        rendered = dry_run(load_serve_config(config_path))
        command = rendered["processes"][0]["command"]
        assert command[1:3] == ["-m", "sglang.launch_server"], command
        assert command[-3:] == ["--tp-size", "2", "--enable-metrics"], command

        overlap = root / "overlap.yaml"
        overlap.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "      - {gpus: [0, 1], port: 30000}",
                "      - {gpus: [0, 1], port: 30000}\n      - {gpus: [1, 2], port: 30001}",
            ),
            encoding="utf-8",
        )
        try:
            load_serve_config(overlap)
        except ValueError as exc:
            assert "GPU overlap" in str(exc), exc
        else:
            raise AssertionError("overlapping replicas must be rejected")


def main() -> int:
    _test_vllm_topology()
    _test_sglang_command_and_validation()
    print("serve OK: commands, GPU validation, backend topology")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())

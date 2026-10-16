"""Dependency-light environment manager checks.

Run: python tests/test_environment.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import evalkit.environment as environment_module
from evalkit.environment import (
    backend_command,
    check_environment,
    environment_path,
    host_tool_command,
    sync_environment,
)


def _fake_tool(path: Path, name: str, version: str) -> None:
    path.write_text(f"#!/bin/sh\necho '{name} version {version}'\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    original = os.environ.get("EVALKIT_ENV_HOME")
    original_ffmpeg = os.environ.get("EVALKIT_FFMPEG")
    original_ffprobe = os.environ.get("EVALKIT_FFPROBE")
    try:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["EVALKIT_ENV_HOME"] = str(root / "envs")
            core_python = environment_path("core") / "bin" / "python"
            core_python.parent.mkdir(parents=True)
            core_python.symlink_to(Path(sys.executable).resolve())
            checked = check_environment("core")
            assert checked["ok"] is True, checked
            assert checked["returncode"] == 0, checked
            assert "evalkit" in checked["stdout"], checked

            assert backend_command("vllm")[0].endswith("/vllm/bin/vllm")
            assert backend_command("vllm_omni")[0].endswith("/vllm-omni/bin/vllm")
            assert backend_command("sglang")[1:] == ("-m", "sglang.launch_server")

            ffmpeg = root / "ffmpeg"
            ffprobe = root / "ffprobe"
            _fake_tool(ffmpeg, "ffmpeg", "6.1.2")
            _fake_tool(ffprobe, "ffprobe", "6.1.2")
            os.environ["EVALKIT_FFMPEG"] = str(ffmpeg)
            os.environ["EVALKIT_FFPROBE"] = str(ffprobe)
            tools = check_environment("tools")
            assert tools["ok"] is True, tools
            assert tools["tools"]["ffmpeg"]["version"] == "6.1.2", tools
            assert host_tool_command("ffprobe") == str(ffprobe.resolve())
            sync_environment("tools")

            os.environ.pop("EVALKIT_FFMPEG")
            os.environ.pop("EVALKIT_FFPROBE")
            bundled_ffmpeg = core_python.parent / "static_ffmpeg"
            bundled_ffprobe = core_python.parent / "static_ffprobe"
            _fake_tool(bundled_ffmpeg, "ffmpeg", "8.0")
            _fake_tool(bundled_ffprobe, "ffprobe", "8.0")
            original_executable = environment_module.sys.executable
            try:
                environment_module.sys.executable = str(core_python)
                bundled_tools = check_environment("tools")
                assert bundled_tools["ok"] is True, bundled_tools
                assert host_tool_command("ffmpeg") == str(bundled_ffmpeg.resolve())
            finally:
                environment_module.sys.executable = original_executable
    finally:
        if original is None:
            os.environ.pop("EVALKIT_ENV_HOME", None)
        else:
            os.environ["EVALKIT_ENV_HOME"] = original
        for name, value in (
            ("EVALKIT_FFMPEG", original_ffmpeg),
            ("EVALKIT_FFPROBE", original_ffprobe),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print("environment manager OK: isolated paths, host tools, core probe, backend commands")
    return 0


def test_script() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())

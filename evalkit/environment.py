"""Isolated backend environments and declared host tools."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = REPO_ROOT / "environments"
_BACKEND_PROJECTS = {
    "core": REPO_ROOT,
    "vllm": ENVIRONMENTS_DIR / "vllm",
    "vllm-omni": ENVIRONMENTS_DIR / "vllm-omni",
    "sglang": ENVIRONMENTS_DIR / "sglang",
}


def env_home() -> Path:
    return Path(os.environ.get("EVALKIT_ENV_HOME", "~/.venv/evalkit")).expanduser().resolve()


def environment_path(name: str) -> Path:
    if name not in _BACKEND_PROJECTS:
        raise KeyError(f"unknown environment {name!r}; available: {sorted(_BACKEND_PROJECTS)}")
    return env_home() / name


def backend_environment(backend: str) -> Path:
    name = {"vllm": "vllm", "vllm_omni": "vllm-omni", "sglang": "sglang"}.get(backend)
    if name is None:
        raise KeyError(f"unknown serving backend {backend!r}")
    return environment_path(name)


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _tools_config() -> dict[str, Any]:
    return _read_toml(ENVIRONMENTS_DIR / "tools.toml")


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _host_tool_config(name: str) -> dict[str, Any]:
    tools = _tools_config().get("tools") or {}
    try:
        config = tools[name]
    except KeyError as exc:
        raise KeyError(f"unknown host tool {name!r}; available: {sorted(tools)}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"host tool {name!r} configuration must be a table")
    return config


def host_tool_command(name: str) -> str:
    """Resolve one declared host executable or raise with its install guidance."""
    config = _host_tool_config(name)
    override = str(config.get("env_override") or "")
    requested = os.environ.get(override, "") if override else ""
    declared = config.get("command") or name
    candidates = [requested] if requested else ([declared] if isinstance(declared, str) else list(declared))
    environment_bin = str(Path(sys.executable).parent)
    for command in candidates:
        resolved = shutil.which(str(command))
        if resolved is None and not requested:
            resolved = shutil.which(str(command), path=environment_bin)
        if resolved is not None:
            return str(Path(resolved).resolve())
    hint = str(config.get("install_hint") or f"install {candidates[0]}")
    suffix = f" or set {override}" if override else ""
    raise RuntimeError(f"host tool {name!r} was not found; {hint}{suffix}")


def check_host_tools() -> dict[str, Any]:
    """Validate commands and minimum versions declared in tools.toml."""
    configured = _tools_config().get("tools") or {}
    results: dict[str, Any] = {}
    for name in configured:
        config = _host_tool_config(str(name))
        minimum = str(config.get("minimum_version") or "")
        item: dict[str, Any] = {"minimum_version": minimum}
        try:
            command = host_tool_command(str(name))
            probe = subprocess.run(
                [command, *(str(arg) for arg in config.get("version_args") or ("--version",))],
                capture_output=True,
                text=True,
                timeout=float(config.get("probe_timeout_seconds") or 10),
            )
            output = "\n".join(part for part in (probe.stdout, probe.stderr) if part)
            match = re.search(str(config.get("version_regex") or ""), output, re.MULTILINE)
            version = match.group("version") if match else ""
            version_ok = bool(version) and (not minimum or _version_tuple(version) >= _version_tuple(minimum))
            item.update(
                {
                    "ok": probe.returncode == 0 and version_ok,
                    "path": command,
                    "version": version,
                    "returncode": probe.returncode,
                }
            )
            if not version:
                item["error"] = "could not parse version output"
            elif not version_ok:
                item["error"] = f"version {version} is below required {minimum}"
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            item.update({"ok": False, "error": str(exc)})
        results[str(name)] = item
    return {"name": "tools", "ok": all(item["ok"] for item in results.values()), "tools": results}


def sync_host_tools() -> None:
    """Validate declared tools; probing a lazy provider such as static-ffmpeg downloads it once."""
    result = check_host_tools()
    if result["ok"]:
        return
    failures: list[str] = []
    for name, item in result["tools"].items():
        if item["ok"]:
            continue
        config = _host_tool_config(name)
        hint = str(config.get("install_hint") or "install the declared command")
        failures.append(f"{name}: {item.get('error', 'unavailable')}; {hint}")
    raise RuntimeError("host tool requirements are not satisfied:\n- " + "\n- ".join(failures))


def sync_python_environment(name: str) -> None:
    try:
        project = _BACKEND_PROJECTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown environment {name!r}; available: {sorted(_BACKEND_PROJECTS)}") from exc
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required; install uv 0.11.28 before syncing environments")
    target = environment_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(target)
    subprocess.run(
        [uv, "sync", "--project", str(project), "--locked", "--python", "3.12"],
        check=True,
        env=env,
    )


def sync_environment(name: str) -> None:
    if name == "tools":
        sync_host_tools()
        return
    if name == "all":
        sync_host_tools()
        for environment in _BACKEND_PROJECTS:
            sync_python_environment(environment)
        return
    sync_python_environment(name)


def check_environment(name: str) -> dict[str, Any]:
    if name == "tools":
        return check_host_tools()
    path = environment_path(name)
    python = path / "bin" / "python"
    result: dict[str, Any] = {"name": name, "ok": python.exists(), "path": str(path)}
    if not python.exists():
        return result
    probe_code = (
        "import json, evalkit; print(json.dumps({'evalkit': evalkit.__version__}))"
        if name == "core"
        else (
            "import json, torch; "
            "ok=torch.cuda.is_available(); "
            "x=(torch.ones(1, device='cuda')+1).item() if ok else None; "
            "print(json.dumps({'cuda':ok,'devices':torch.cuda.device_count(),'probe':x}))"
        )
    )
    try:
        probe = subprocess.run(
            [str(python), "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        result.update({"ok": False, "error": f"environment probe timed out: {exc}"})
        return result
    result.update(
        {
            "ok": result["ok"] and probe.returncode == 0,
            "returncode": probe.returncode,
            "stdout": probe.stdout.strip(),
            "stderr": probe.stderr.strip(),
        }
    )
    return result


def backend_command(backend: str) -> tuple[str, ...]:
    environment = backend_environment(backend)
    if backend in {"vllm", "vllm_omni"}:
        return (str(environment / "bin" / "vllm"),)
    return (str(environment / "bin" / "python"), "-m", "sglang.launch_server")


__all__ = [
    "backend_command",
    "backend_environment",
    "check_environment",
    "check_host_tools",
    "environment_path",
    "host_tool_command",
    "sync_environment",
]

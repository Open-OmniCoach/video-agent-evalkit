"""Foreground subprocess supervisor for local inference replicas."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ..environment import backend_command, backend_environment
from .config import ServeConfig


@dataclass(frozen=True)
class ProcessSpec:
    logical_model: str
    replica_index: int
    command: tuple[str, ...]
    env: dict[str, str]
    gpus: tuple[int, ...]
    health_urls: tuple[str, ...]
    base_url: str
    local_media_path: Path | None = None


def _expand_arg(arg: str, env: dict[str, str]) -> str:
    """Resolve a whole-argument ${NAME} reference; other arguments stay literal."""
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", arg)
    if match is None:
        return arg
    value = env.get(match[1], "").strip()
    if not value:
        raise ValueError(f"serve argument requires environment variable {match[1]}")
    return value


def _resolve_local_media_path(command: list[str], working_dir: str) -> Path | None:
    """Use the same canonical root as file:// requests, including through symlinks."""
    flag = "--allowed-local-media-path"
    selected = None
    for index, arg in enumerate(command):
        if arg == flag:
            index += 1
            if index == len(command) or command[index].startswith("--"):
                raise ValueError(f"{flag} requires a directory")
            raw = command[index]
            prefix = ""
        elif arg.startswith(flag + "="):
            raw = arg[len(flag) + 1 :]
            prefix = flag + "="
        else:
            continue
        if not raw.strip():
            raise ValueError(f"{flag} requires a directory")
        path = Path(raw).expanduser()
        selected = (Path(working_dir) / path).resolve()
        command[index] = prefix + str(selected)
    return selected


def build_process_specs(config: ServeConfig) -> tuple[ProcessSpec, ...]:
    specs: list[ProcessSpec] = []
    for logical_name, model in config.models.items():
        for index, replica in enumerate(model.replicas):
            command = list(backend_command(model.backend))
            if model.backend in {"vllm", "vllm_omni"}:
                command.extend(
                    [
                        "serve",
                        model.model,
                        "--host",
                        config.host,
                        "--port",
                        str(replica.port),
                        "--served-model-name",
                        model.served_model_name,
                    ]
                )
                if model.backend == "vllm_omni":
                    command.append("--omni")
            else:
                command.extend(
                    [
                        "--model-path",
                        model.model,
                        "--host",
                        config.host,
                        "--port",
                        str(replica.port),
                        "--served-model-name",
                        model.served_model_name,
                    ]
                )
            env = os.environ.copy()
            env.update(model.env)
            env.update(replica.env)
            command.extend(_expand_arg(arg, env) for arg in (*model.args, *replica.args))
            local_media_path = (
                _resolve_local_media_path(command, config.working_dir)
                if model.backend in {"vllm", "vllm_omni"}
                else None
            )
            backend_bin = str(backend_environment(model.backend) / "bin")
            inherited_path = env.get("PATH", "")
            env["PATH"] = backend_bin if not inherited_path else f"{backend_bin}{os.pathsep}{inherited_path}"
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in replica.gpus)
            root = f"http://{config.advertise_host}:{replica.port}"
            specs.append(
                ProcessSpec(
                    logical_model=logical_name,
                    replica_index=index,
                    command=tuple(command),
                    env=env,
                    gpus=replica.gpus,
                    health_urls=tuple(f"{root}{path}" for path in model.health_paths),
                    base_url=f"{root}/v1",
                    local_media_path=local_media_path,
                )
            )
    return tuple(specs)


def _manifest(
    config: ServeConfig, specs: tuple[ProcessSpec, ...], processes: list[asyncio.subprocess.Process]
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, model in config.models.items():
        selected = [(spec, proc) for spec, proc in zip(specs, processes, strict=True) if spec.logical_model == name]
        models[name] = {
            "transport": "openai_chat",
            "backend": model.backend,
            "adapter": model.adapter,
            "model": model.served_model_name,
            "urls": [spec.base_url for spec, _ in selected],
            "replicas": [
                {
                    "url": spec.base_url,
                    "pid": proc.pid,
                    "replica_index": spec.replica_index,
                    "gpus": list(spec.gpus),
                }
                for spec, proc in selected
            ],
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
    }


def _write_manifest(path: str, manifest: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)


class ServeSupervisor:
    def __init__(self, config: ServeConfig) -> None:
        self.config = config
        self.specs: tuple[ProcessSpec, ...] = ()
        self.processes: list[asyncio.subprocess.Process] = []
        self.manifest: dict[str, Any] | None = None

    async def start(self) -> dict[str, Any]:
        if self.processes:
            raise RuntimeError("serve supervisor has already been started")
        self.specs = build_process_specs(self.config)
        manifest_path = Path(self.config.manifest_path)
        manifest_path.unlink(missing_ok=True)
        try:
            for spec in self.specs:
                executable = Path(spec.command[0])
                if not executable.exists():
                    raise FileNotFoundError(
                        f"backend executable not found: {executable}; sync its evalkit environment first"
                    )
                if spec.local_media_path is not None:
                    spec.local_media_path.mkdir(parents=True, exist_ok=True)
                process = await asyncio.create_subprocess_exec(
                    *spec.command,
                    env=spec.env,
                    cwd=self.config.working_dir,
                    start_new_session=True,
                )
                self.processes.append(process)
            readiness = [
                asyncio.create_task(self._wait_ready(spec, process))
                for spec, process in zip(self.specs, self.processes, strict=True)
            ]
            try:
                await asyncio.gather(*readiness)
            except BaseException:
                for task in readiness:
                    task.cancel()
                await asyncio.gather(*readiness, return_exceptions=True)
                raise
        except BaseException:
            await self.stop()
            raise
        self.manifest = _manifest(self.config, self.specs, self.processes)
        _write_manifest(self.config.manifest_path, self.manifest)
        return self.manifest

    async def _wait_ready(self, spec: ProcessSpec, process: asyncio.subprocess.Process) -> None:
        deadline = asyncio.get_running_loop().time() + self.config.startup_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                if process.returncode is not None:
                    raise RuntimeError(
                        f"{spec.logical_model}[{spec.replica_index}] exited during startup with {process.returncode}"
                    )
                for url in spec.health_urls:
                    try:
                        response = await client.get(url)
                        if 200 <= response.status_code < 400:
                            return
                    except httpx.HTTPError:
                        pass
                await asyncio.sleep(1.0)
        raise TimeoutError(
            f"{spec.logical_model}[{spec.replica_index}] did not become ready within {self.config.startup_timeout}s"
        )

    async def wait(self) -> int:
        if not self.processes:
            return 0
        waiters = [asyncio.create_task(process.wait()) for process in self.processes]
        done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        exit_code = next(iter(done)).result()
        for waiter in pending:
            waiter.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self.stop()
        return exit_code if exit_code != 0 else 1

    async def stop(self) -> None:
        running = [process for process in self.processes if process.returncode is None]
        for process in running:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if running:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(process.wait() for process in running)),
                    timeout=self.config.terminate_timeout,
                )
            except asyncio.TimeoutError:
                for process in running:
                    if process.returncode is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                await asyncio.gather(*(process.wait() for process in running))
        Path(self.config.manifest_path).unlink(missing_ok=True)
        self.manifest = None


async def run_serve(config: ServeConfig) -> int:
    supervisor = ServeSupervisor(config)
    try:
        await supervisor.start()
        return await supervisor.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 130
    finally:
        await supervisor.stop()


def dry_run(config: ServeConfig) -> dict[str, Any]:
    specs = build_process_specs(config)
    return {
        "manifest_path": config.manifest_path,
        "working_dir": config.working_dir,
        "processes": [
            {
                "logical_model": spec.logical_model,
                "replica_index": spec.replica_index,
                "command": list(spec.command),
                "cuda_visible_devices": spec.env["CUDA_VISIBLE_DEVICES"],
                "base_url": spec.base_url,
            }
            for spec in specs
        ],
    }


def run_serve_sync(config: ServeConfig) -> int:
    return asyncio.run(run_serve(config))


__all__ = ["ProcessSpec", "ServeSupervisor", "build_process_specs", "dry_run", "run_serve_sync"]

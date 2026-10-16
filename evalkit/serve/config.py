"""Declarative local model-service topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import resolve_path


@dataclass(frozen=True)
class ReplicaConfig:
    gpus: tuple[int, ...]
    port: int
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServeModelConfig:
    name: str
    backend: str
    model: str
    served_model_name: str
    adapter: str
    replicas: tuple[ReplicaConfig, ...]
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    health_paths: tuple[str, ...] = ("/health", "/v1/models")


@dataclass(frozen=True)
class ServeConfig:
    models: dict[str, ServeModelConfig]
    manifest_path: str
    working_dir: str
    host: str = "0.0.0.0"
    advertise_host: str = "127.0.0.1"
    startup_timeout: float = 1800.0
    terminate_timeout: float = 15.0
    allow_gpu_overlap: bool = False


def load_serve_config(path: str | Path) -> ServeConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("serve config top-level must be a mapping")
    if not raw.get("manifest_path"):
        raise ValueError("serve config requires manifest_path")
    raw_models = raw.get("models") or {}
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("serve config requires a non-empty models mapping")

    models: dict[str, ServeModelConfig] = {}
    used_ports: set[int] = set()
    used_gpus: set[int] = set()
    allow_gpu_overlap = bool(raw.get("allow_gpu_overlap", False))
    for logical_name, value in raw_models.items():
        item = dict(value or {})
        backend = str(item.get("backend", ""))
        if backend not in {"vllm", "vllm_omni", "sglang"}:
            raise ValueError(f"model {logical_name!r} has unsupported backend {backend!r}")
        model_path = str(item.get("model", "")).strip()
        if not model_path:
            raise ValueError(f"model {logical_name!r} is missing model")
        adapter = str(item.get("adapter", "openai"))
        if adapter not in {"openai", "vllm_multimodal"}:
            raise ValueError(f"model {logical_name!r} has unsupported adapter {adapter!r}")
        replicas: list[ReplicaConfig] = []
        for replica_raw in item.get("replicas") or ():
            replica = dict(replica_raw or {})
            gpus = tuple(int(gpu) for gpu in replica.get("gpus") or ())
            port = int(replica.get("port", 0))
            if not gpus or any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
                raise ValueError(f"model {logical_name!r} replica requires unique non-negative GPUs")
            if not 1 <= port <= 65535 or port in used_ports:
                raise ValueError(f"model {logical_name!r} has invalid or duplicate port {port}")
            overlap = used_gpus.intersection(gpus)
            if overlap and not allow_gpu_overlap:
                raise ValueError(f"GPU overlap is not allowed: {sorted(overlap)}")
            used_ports.add(port)
            used_gpus.update(gpus)
            replicas.append(
                ReplicaConfig(
                    gpus=gpus,
                    port=port,
                    env={str(key): str(val) for key, val in dict(replica.get("env") or {}).items()},
                    args=tuple(str(arg) for arg in (replica.get("args") or ())),
                )
            )
        if not replicas:
            raise ValueError(f"model {logical_name!r} requires at least one replica")
        models[str(logical_name)] = ServeModelConfig(
            name=str(logical_name),
            backend=backend,
            model=model_path,
            served_model_name=str(item.get("served_model_name") or model_path),
            adapter=adapter,
            replicas=tuple(replicas),
            args=tuple(str(arg) for arg in (item.get("args") or ())),
            env={str(key): str(val) for key, val in dict(item.get("env") or {}).items()},
            health_paths=tuple(str(p) for p in (item.get("health_paths") or ("/health", "/v1/models"))),
        )

    startup_timeout = float(raw.get("startup_timeout", 1800.0))
    terminate_timeout = float(raw.get("terminate_timeout", 15.0))
    if startup_timeout <= 0 or terminate_timeout <= 0:
        raise ValueError("startup_timeout and terminate_timeout must be positive")
    for model in models.values():
        if not model.health_paths or any(not path.startswith("/") for path in model.health_paths):
            raise ValueError(f"model {model.name!r} health_paths must be absolute URL paths")

    working_dir = resolve_path(path.parent, raw.get("working_dir", "."))
    if not Path(working_dir).is_dir():
        raise ValueError(f"serve working_dir does not exist: {working_dir}")

    return ServeConfig(
        models=models,
        manifest_path=resolve_path(path.parent, raw["manifest_path"]),
        working_dir=working_dir,
        host=str(raw.get("host", "0.0.0.0")),
        advertise_host=str(raw.get("advertise_host", "127.0.0.1")),
        startup_timeout=startup_timeout,
        terminate_timeout=terminate_timeout,
        allow_gpu_overlap=allow_gpu_overlap,
    )


__all__ = ["ReplicaConfig", "ServeConfig", "ServeModelConfig", "load_serve_config"]

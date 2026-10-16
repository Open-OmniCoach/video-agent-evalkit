"""Typed YAML configuration for evaluation runs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_GENERATE_FN = "evalkit.generate.openai_chat:generate"
_DEFAULT_SCORE_FN = "evalkit.runtime.scoring:score"


def resolve_path(base_dir: Path, value: Any) -> str:
    path = Path(str(value)).expanduser()
    return str((path if path.is_absolute() else base_dir / path).resolve())


def _resolve_optional_path(base_dir: Path, value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return resolve_path(base_dir, value)


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _prompt_config(base_dir: Path, value: Any, *, field_name: str) -> dict[str, Any]:
    prompt = _mapping(value, field_name=field_name)
    if prompt.get("template"):
        prompt["template"] = resolve_path(base_dir, prompt["template"])
    return prompt


def _scoring_config(value: Any, *, field_name: str) -> dict[str, Any]:
    scoring = _mapping(value, field_name=field_name) or {"type": "rule"}
    kind = str(scoring.get("type", "rule")).lower()
    if kind not in {"rule", "judge"}:
        raise ValueError(f"{field_name}.type must be 'rule' or 'judge'")
    scoring["type"] = kind
    return scoring


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True)
class EndpointCfg:
    """One logical model's OpenAI-compatible transport configuration."""

    transport: str = "openai_chat"
    adapter: str = "openai"
    base_url: str = ""
    base_urls: tuple[str, ...] = ()
    service_manifest: str | None = None
    service_model: str = ""
    model: str = ""
    api_key: str = ""
    api_key_env: str = ""
    headers: dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    media_transport: str = "inline"
    route_key_header: str = ""
    max_inflight: int = 0
    request_retries: int = 2
    retry_backoff: float = 0.5

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    @staticmethod
    def from_dict(
        data: dict[str, Any] | None,
        *,
        config_dir: Path,
        logical_name: str,
    ) -> "EndpointCfg":
        raw = _mapping(data, field_name=f"model {logical_name!r}.endpoint")
        base_url = str(raw.get("base_url", "")).strip()
        raw_base_urls = raw.get("base_urls") or ()
        if isinstance(raw_base_urls, (str, bytes)):
            raise ValueError(f"model {logical_name!r}.endpoint.base_urls must be a list")
        base_urls = tuple(str(url).strip() for url in raw_base_urls if str(url).strip())
        if len(set(base_urls)) != len(base_urls):
            raise ValueError(f"model {logical_name!r}.endpoint.base_urls contains duplicates")
        service_manifest = _resolve_optional_path(config_dir, raw.get("service_manifest"))
        source_count = int(bool(base_url)) + int(bool(base_urls)) + int(bool(service_manifest))
        if source_count != 1:
            raise ValueError(
                f"model {logical_name!r} endpoint must set exactly one of base_url, base_urls, or service_manifest"
            )
        if not service_manifest and not str(raw.get("model", "")).strip():
            raise ValueError(f"model {logical_name!r}.endpoint.model is required for direct URLs")
        transport = str(raw.get("transport", "openai_chat"))
        if transport != "openai_chat":
            raise ValueError(f"unsupported transport {transport!r}; expected 'openai_chat'")
        adapter = str(raw.get("adapter", "openai"))
        if adapter not in {"openai", "vllm_multimodal"}:
            raise ValueError(f"unsupported endpoint adapter {adapter!r}")
        media_transport = str(raw.get("media_transport", "inline"))
        if media_transport not in {"inline", "file_uri"}:
            raise ValueError("media_transport must be 'inline' or 'file_uri'")
        max_inflight = int(raw.get("max_inflight", 0))
        if max_inflight < 0:
            raise ValueError("endpoint.max_inflight cannot be negative")
        request_retries = int(raw.get("request_retries", 2))
        if request_retries < 0:
            raise ValueError("endpoint.request_retries cannot be negative")
        timeout = float(raw.get("timeout", 300.0))
        if timeout <= 0:
            raise ValueError("endpoint.timeout must be positive")
        retry_backoff = float(raw.get("retry_backoff", 0.5))
        if retry_backoff < 0:
            raise ValueError("endpoint.retry_backoff cannot be negative")
        return EndpointCfg(
            transport=transport,
            adapter=adapter,
            base_url=base_url,
            base_urls=base_urls,
            service_manifest=service_manifest,
            service_model=str(raw.get("service_model") or logical_name),
            model=str(raw.get("model", "")),
            api_key=str(raw.get("api_key", "")),
            api_key_env=str(raw.get("api_key_env", "")),
            headers=_mapping(raw.get("headers"), field_name="endpoint.headers"),
            timeout=timeout,
            extra_body=_mapping(raw.get("extra_body"), field_name="endpoint.extra_body"),
            media_transport=media_transport,
            route_key_header=str(raw.get("route_key_header", "")),
            max_inflight=max_inflight,
            request_retries=request_retries,
            retry_backoff=retry_backoff,
        )


@dataclass(frozen=True)
class ModelCfg:
    name: str
    generate_fn: str
    endpoint: EndpointCfg

    @staticmethod
    def from_dict(
        name: str,
        data: dict[str, Any] | None,
        *,
        config_dir: Path,
        default_generate_fn: str,
    ) -> "ModelCfg":
        raw = _mapping(data, field_name=f"model {name!r}")
        return ModelCfg(
            name=name,
            generate_fn=str(raw.get("generate_fn") or default_generate_fn),
            endpoint=EndpointCfg.from_dict(raw.get("endpoint"), config_dir=config_dir, logical_name=name),
        )


@dataclass(frozen=True)
class BenchCfg:
    name: str
    generate_fn: str | None = None
    score_fn: str | None = None
    scoring: dict[str, Any] = field(default_factory=dict)
    aggregator: str | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    prompt: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    idempotent: bool | None = None


@dataclass(frozen=True)
class RunConfig:
    model: ModelCfg
    eval_datasets: tuple[BenchCfg, ...]
    aux_models: dict[str, ModelCfg] = field(default_factory=dict)
    max_active_cases: int = 64
    out_dir: str = "./runs/exp"
    registry_path: str | None = None
    data_root: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    case_max_attempts: int = 2
    flush_every: int = 20
    trajectory: str | None = None

    def bench_generate_fn(self, bench: BenchCfg) -> str:
        return str(bench.generate_fn or self.defaults.get("generate_fn") or self.model.generate_fn)

    def bench_score_fn(self, bench: BenchCfg) -> str:
        return str(bench.score_fn or self.defaults.get("score_fn") or _DEFAULT_SCORE_FN)

    def bench_sampling(self, bench: BenchCfg) -> dict[str, Any]:
        merged = dict(self.defaults.get("sampling") or {})
        merged.update(bench.sampling)
        return merged

    def bench_budget(self, bench: BenchCfg) -> dict[str, Any]:
        merged = dict(self.defaults.get("budget") or {})
        merged.update(bench.budget)
        return merged

    def bench_prompt(self, bench: BenchCfg) -> dict[str, Any]:
        merged = dict(self.defaults.get("prompt") or {})
        merged.update(bench.prompt)
        return merged

    def bench_scoring(self, bench: BenchCfg) -> dict[str, Any]:
        merged = dict(self.defaults.get("scoring") or {"type": "rule"})
        merged.update(bench.scoring)
        return merged

    def bench_idempotent(self, bench: BenchCfg) -> bool:
        if bench.idempotent is not None:
            return bench.idempotent
        return bool(self.defaults.get("idempotent", False))


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path).expanduser().resolve()
    config_dir = path.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"run config top-level must be a mapping, got {type(raw).__name__}")

    defaults = _mapping(raw.get("defaults"), field_name="defaults")
    defaults["prompt"] = _prompt_config(config_dir, defaults.get("prompt"), field_name="defaults.prompt")
    defaults["scoring"] = _scoring_config(defaults.get("scoring"), field_name="defaults.scoring")
    defaults["sampling"] = _mapping(defaults.get("sampling"), field_name="defaults.sampling")
    defaults["budget"] = _mapping(defaults.get("budget"), field_name="defaults.budget")
    if "idempotent" in defaults:
        defaults["idempotent"] = _optional_bool(defaults["idempotent"], field_name="defaults.idempotent")
    default_generate_fn = str(defaults.get("generate_fn") or _DEFAULT_GENERATE_FN)
    model = ModelCfg.from_dict(
        "candidate",
        raw.get("model"),
        config_dir=config_dir,
        default_generate_fn=default_generate_fn,
    )
    raw_aux_models = _mapping(raw.get("aux_models"), field_name="aux_models")
    if "candidate" in raw_aux_models:
        raise ValueError("aux_models cannot redefine the reserved model name 'candidate'")
    aux_models = {
        str(name): ModelCfg.from_dict(
            str(name),
            cfg,
            config_dir=config_dir,
            default_generate_fn=default_generate_fn,
        )
        for name, cfg in raw_aux_models.items()
    }

    benches: list[BenchCfg] = []
    raw_benches = raw.get("eval_datasets") or []
    if isinstance(raw_benches, (str, bytes, dict)):
        raise ValueError("eval_datasets must be a list of mappings")
    for entry in raw_benches:
        item = _mapping(entry, field_name="eval_datasets entry")
        if "name" not in item:
            raise ValueError("each eval_datasets entry must have a 'name'")
        limit = item.get("limit")
        if limit is not None and int(limit) < 0:
            raise ValueError("bench limit cannot be negative")
        benches.append(
            BenchCfg(
                name=str(item["name"]),
                generate_fn=item.get("generate_fn"),
                score_fn=item.get("score_fn"),
                scoring=(
                    _scoring_config(item["scoring"], field_name=f"benchmark {item['name']!r}.scoring")
                    if "scoring" in item
                    else {}
                ),
                aggregator=(str(item["aggregator"]) if item.get("aggregator") else None),
                budget=_mapping(item.get("budget"), field_name=f"benchmark {item['name']!r}.budget"),
                sampling=_mapping(item.get("sampling"), field_name=f"benchmark {item['name']!r}.sampling"),
                prompt=_prompt_config(
                    config_dir,
                    item.get("prompt"),
                    field_name=f"benchmark {item['name']!r}.prompt",
                ),
                report=_mapping(item.get("report"), field_name=f"benchmark {item['name']!r}.report"),
                limit=int(limit) if limit is not None else None,
                idempotent=(
                    _optional_bool(item["idempotent"], field_name=f"benchmark {item['name']!r}.idempotent")
                    if "idempotent" in item
                    else None
                ),
            )
        )
    if not benches:
        raise ValueError("run config must list at least one eval_datasets entry")

    scheduler = _mapping(raw.get("scheduler"), field_name="scheduler")
    max_active_cases = int(scheduler.get("max_active_cases", 64))
    if max_active_cases <= 0:
        raise ValueError("scheduler.max_active_cases must be positive")
    retry = _mapping(raw.get("retry"), field_name="retry")
    case_max_attempts = int(retry.get("case_max_attempts", 2))
    if case_max_attempts <= 0:
        raise ValueError("retry.case_max_attempts must be positive")
    flush_every = int(raw.get("flush_every", 20))
    if flush_every <= 0:
        raise ValueError("flush_every must be positive")
    trajectory = raw.get("trajectory")
    if trajectory not in (None, "trainable"):
        raise ValueError("trajectory must be 'trainable' when set")

    return RunConfig(
        model=model,
        eval_datasets=tuple(benches),
        aux_models=aux_models,
        max_active_cases=max_active_cases,
        out_dir=resolve_path(config_dir, raw.get("out_dir") or "./runs/exp"),
        registry_path=resolve_path(config_dir, raw.get("registry_path") or "registry.yaml"),
        data_root=_resolve_optional_path(config_dir, raw.get("data_root")),
        defaults=defaults,
        case_max_attempts=case_max_attempts,
        flush_every=flush_every,
        trajectory=trajectory,
    )


__all__ = ["BenchCfg", "EndpointCfg", "ModelCfg", "RunConfig", "load_run_config", "resolve_path"]

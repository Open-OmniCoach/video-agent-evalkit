"""Evalkit command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_run_config
from .data_tools.cli import add_data_subparser
from .environment import check_environment, sync_environment
from .runtime import driver
from .serve import load_serve_config, run_serve_sync
from .serve.launcher import dry_run

_ENV_TARGETS = ("core", "vllm", "vllm-omni", "sglang", "tools", "all")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="Run benchmark accuracy evaluation.")
    run_parser.add_argument("--config", required=True, help="Path to run YAML.")
    run_parser.add_argument("--log-level", default="INFO")

    serve_parser = sub.add_parser("serve", help="Launch local inference replicas.")
    serve_parser.add_argument("--config", required=True, help="Path to serve YAML.")
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.add_argument("--log-level", default="INFO")

    env_parser = sub.add_parser("env", help="Manage isolated inference environments.")
    env_sub = env_parser.add_subparsers(dest="env_cmd", required=True)
    for action in ("sync", "check"):
        action_parser = env_sub.add_parser(action)
        action_parser.add_argument("target", choices=_ENV_TARGETS)

    add_data_subparser(sub)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        _configure_logging(args.log_level)
        # Source-checkout generators are resolved from the caller's working directory.
        working_dir = str(Path.cwd())
        if working_dir not in sys.path:
            sys.path.insert(0, working_dir)
        result = driver.run_sync(load_run_config(args.config))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "serve":
        _configure_logging(args.log_level)
        config = load_serve_config(args.config)
        if args.dry_run:
            print(json.dumps(dry_run(config), indent=2, ensure_ascii=False))
            return 0
        return run_serve_sync(config)
    if args.cmd == "env":
        if args.env_cmd == "sync":
            sync_environment(args.target)
            return 0
        targets = _ENV_TARGETS[:-1] if args.target == "all" else (args.target,)
        results = [check_environment(target) for target in targets]
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0 if all(result["ok"] for result in results) else 1
    if args.cmd == "data":
        return args.func(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())

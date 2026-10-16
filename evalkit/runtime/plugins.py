"""Import-path based plugin loading without framework-specific dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_callable(path: str) -> Callable[..., Any]:
    if ":" in path:
        module_name, attribute = path.split(":", 1)
    else:
        module_name, attribute = path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"plugin {path!r} is not callable")
    return value


__all__ = ["load_callable"]

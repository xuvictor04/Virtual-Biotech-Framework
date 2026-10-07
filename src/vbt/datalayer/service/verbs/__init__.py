"""Hidden verbs of the data child (§11.8), discovered from the modules of this package.

Each module exports ``VERBS: dict[str, Callable[[ServiceContext, Mapping], dict]]``: a verb takes the
request payload (the JSON of its :mod:`vbt.datalayer.ipc` request model) and returns the JSON of
its response model. Later phases add verbs as new modules (``aggregate.py``, ``similar.py``, ...)
without editing ``server.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Mapping

__all__ = ["load_verbs", "Verb"]

Verb = Callable[[Any, Mapping[str, Any]], dict[str, Any]]


def load_verbs() -> dict[str, Verb]:
    """Every ``VERBS`` entry of every module of this package (a duplicate name is an error)."""
    out: dict[str, Verb] = {}
    origin: dict[str, str] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for name, fn in (getattr(module, "VERBS", None) or {}).items():
            if name in out:
                raise RuntimeError(f"verb {name!r} is defined by {origin[name]} and {module.__name__}")
            out[name] = fn
            origin[name] = module.__name__
    return out

"""The vbt data layer: descriptors, plugins and one gateway in front of every MCP server.

See ``docs/DATA_LAYER.md``. Harness-side modules never import pyarrow or pandas (I12); the
data child (``service/``) reads the data. Public names are resolved lazily so importing
``vbt.datalayer`` stays cheap and never pulls in the gateway before it is needed.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["DataResult", "GatewayError", "ErrorKind", "DataSettings", "load_catalog", "build_gateway"]

_LAZY = {
    "DataResult": ("vbt.datalayer.result", "DataResult"),
    "GatewayError": ("vbt.datalayer.errors", "GatewayError"),
    "ErrorKind": ("vbt.datalayer.errors", "ErrorKind"),
    "DataSettings": ("vbt.datalayer.settings", "DataSettings"),
    "load_catalog": ("vbt.datalayer.catalog", "load_catalog"),
}

_GATEWAY_MODULE = "vbt.datalayer.gateway.gateway"


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module, attr = _LAZY[name]
        value = getattr(importlib.import_module(module), attr)
        globals()[name] = value
        return value
    if name == "build_gateway":
        try:
            gateway = importlib.import_module(_GATEWAY_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name in (_GATEWAY_MODULE, "vbt.datalayer.gateway"):
                raise ImportError("vbt.datalayer.build_gateway is unavailable: the gateway package "
                                  f"({_GATEWAY_MODULE}) is not installed in this checkout") from exc
            raise
        value = getattr(gateway, "build_gateway")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

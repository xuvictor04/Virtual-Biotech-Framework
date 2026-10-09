"""Make a registration visible to the running session (docs/PROJECTS.md, "In-session").

After a data spec or plugin is registered, :func:`refresh_data_layer`:

1. updates the session's configuration (``data.project_dir``: discovery imports the project's approved plugins);
2. calls :meth:`vbt.runtime.Runtime.reload_data_layer`, which rebuilds the data catalog (shipped + project) and
   plugin registry, hands them to the running gateway (``DataGateway.reload_catalog``) and restarts the ``data``
   child (``MCPBridge.recycle``) with the updated settings, so it serves the new tables and its listing (the
   ``mcp__data__*`` table enums, derived by the gateway from the new catalog) is re-registered. A runtime without
   that method (an embedding of an older harness) gets the same steps from here.

Every step is best effort: what could not be refreshed is named in the returned note and becomes available in the
next session of the project, which builds everything from the project directory.
"""

from __future__ import annotations

import logging
from typing import Any

from .model import Project, activate

__all__ = ["refresh_data_layer"]

log = logging.getLogger(__name__)
DATA_SERVER = "data"


def _swap_catalog(gateway: Any, catalog: Any, registry: Any) -> None:
    reload = getattr(gateway, "reload_catalog", None)
    if callable(reload):
        reload(catalog, registry)
        return
    gateway.catalog = catalog
    gateway.registry = registry
    for part in ("readiness", "resolver"):
        obj = getattr(gateway, part, None)
        if obj is not None:
            if hasattr(obj, "catalog"):
                obj.catalog = catalog
            if hasattr(obj, "registry"):
                obj.registry = registry


async def refresh_data_layer(runtime: Any, project: Project) -> str:
    """Refresh the running session after a registration; returns a note for the agent ("" when all applied)."""
    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        return "the new item is available from the next session of the project"
    updated = activate(config, project, runs=False)
    data = config.setdefault("data", {})
    data["project_dir"] = updated["data"]["project_dir"]
    if "plugins" in updated["data"]:
        data["plugins"] = updated["data"]["plugins"]
    reload = getattr(runtime, "reload_data_layer", None)
    if callable(reload):
        problems = list(await reload())
    else:
        problems = await _refresh(runtime, project, config)
    if problems == ["no data gateway runs in this session"]:
        return ("no data gateway runs in this session: the new data is served from the next session with the data "
                "layer enabled")
    if problems:
        return "; ".join(problems) + ": the new data is served from the next session of the project"
    return ""


async def _refresh(runtime: Any, project: Project, config: dict[str, Any]) -> list[str]:
    """What :meth:`vbt.runtime.Runtime.reload_data_layer` does, for a runtime without it."""
    if hasattr(runtime, "_data_settings"):
        runtime._data_settings = None                  # noqa: SLF001 - parsed again from the updated config
    gateway = getattr(runtime, "gateway", None)
    bridge = getattr(runtime, "mcp", None)
    if gateway is None or bridge is None:
        return ["no data gateway runs in this session"]
    problems = []
    try:
        from ..datalayer.catalog import build_catalog
        from ..datalayer.descriptor.load import variables_from_config
        from ..datalayer.plugins.registry import discover
        from ..datalayer.settings import DataSettings

        settings = DataSettings.from_config(config)
        registry = discover(settings)
        run_vars = runtime.run_variables() if callable(getattr(runtime, "run_variables", None)) else None
        catalog = build_catalog(settings, registry, variables=variables_from_config(config), run=run_vars)
        _swap_catalog(gateway, catalog, registry)
        gateway.settings = settings
    except Exception as exc:  # noqa: BLE001 - the session keeps its previous catalog
        log.warning("refreshing the data catalog failed", exc_info=True)
        problems.append(f"the gateway's catalog could not be rebuilt ({type(exc).__name__}: {exc})")
    try:
        st = getattr(bridge, "_servers", {}).get(DATA_SERVER)
        if st is not None:
            cfg = st.cfg
            env = dict(getattr(cfg, "env", None) or {})
            env["VBT_DATA_SETTINGS"] = DataSettings.from_config(config).to_json()
            cfg.env = env
            ok = await bridge.recycle(DATA_SERVER, wait_s=60.0)
            if not ok:
                problems.append("the data child is busy and was not restarted")
        else:
            problems.append("this session runs no data child")
    except Exception as exc:  # noqa: BLE001
        log.warning("restarting the data child failed", exc_info=True)
        problems.append(f"the data child could not be restarted ({type(exc).__name__}: {exc})")
    clear = getattr(runtime, "_system_cache", None)
    if hasattr(clear, "clear"):
        clear.clear()
    return problems

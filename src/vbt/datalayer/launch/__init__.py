"""Launching stdio MCP servers under the reaper (§14.2). No pyarrow.

:func:`build_launch_spec` rewrites a server's command so it runs under
:mod:`vbt.datalayer.launch.reaper` (executed by path with the harness interpreter and ``-E``):
an ``RLIMIT_DATA`` memory limit, a status file next to the server log, a ``VBT_CHILD_EXIT``
marker in the server log when the child ends (the reaper tees the child's stderr, so a memory exit
carries its ``cause``: ``memory_error``, ``cgroup_oom_kill``, ``watchdog``, ``kernel_oom_kill`` or
``peak_rss``), and for Python children an explicit environment in which ``PYTHONHASHSEED=0`` actually
takes effect. The gateway's ``launch_spec(cfg)`` calls it
with the bridge's log directory (``MCPBridge.log_root()``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..api import LaunchSpec
from ..memory import sizing

__all__ = ["REAPER", "EXIT_MARKER", "CHILD_ENV", "DATA_SERVER", "LIMIT_KINDS", "build_launch_spec", "limit_kind",
           "server_limit_mb", "host_memory_mb", "status_path"]

#: The launcher script, executed by path (it never imports vbt).
REAPER = Path(__file__).resolve().with_name("reaper.py")
EXIT_MARKER = "VBT_CHILD_EXIT"
DATA_SERVER = "data"

#: Environment pins for every launched child (§14.2 step 3). With the system allocator, Arrow
#: allocation failures surface as MemoryError/ArrowMemoryError inside the tool, so FastMCP returns
#: isError and the server usually survives. PYTHONHASHSEED is set by the reaper itself.
CHILD_ENV: dict[str, str] = {
    "ARROW_DEFAULT_MEMORY_POOL": "system",
    "MALLOC_ARENA_MAX": "2",
    "PRELOAD_MCP_DATA": "0",
}

#: A server limit of ``auto`` on a host whose memory cannot be read (the shipped number before ``auto``).
UNKNOWN_HOST_SERVER_MB = 12000


def host_memory_mb() -> int | None:
    """The memory this process tree may use, in MB: the smaller of ``MemTotal`` and its memory cgroup limit
    (a container's ``--memory``), or ``$VBT_HOST_MEMORY_MB``; None when unknown (:mod:`..memory.sizing`)."""
    plan = sizing.plan_mb()
    return int(plan) if plan else None


def _as_mb(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    return None


def _memory_raw(settings: Any) -> dict[str, Any]:
    raw = getattr(settings, "raw", None) or {}
    mem = raw.get("memory") if isinstance(raw, dict) else None
    return dict(mem) if isinstance(mem, dict) else {}


LIMIT_KINDS = ("rlimit_data", "cgroup", "watchdog", "rss", "none")


def limit_kind(cfg: Any, settings: Any = None) -> str:
    """How a server's memory is contained: its own ``limit_kind`` (``configs/mcp_servers.yaml``), else
    ``data.memory.limit_kind``. ``rss`` contains resident memory only (a memory cgroup, else the RSS watchdog)
    and sets no ``RLIMIT_DATA``, which TileDB's Census reads and Arrow's thread stacks fail under."""
    own = None if isinstance(cfg, str) else getattr(cfg, "limit_kind", None)
    if own:
        if own not in LIMIT_KINDS:
            raise ValueError(f"server {getattr(cfg, 'name', cfg)!r}: limit_kind {own!r} is not one of {LIMIT_KINDS}")
        return str(own)
    return str(getattr(getattr(settings, "memory", None), "limit_kind", "rlimit_data") or "rlimit_data")


def server_limit_mb(cfg: Any, settings: Any = None) -> int:
    """The memory limit (MB) a server runs under; 0 means no limit.

    Per server ``mem_limit_mb`` wins; the data child defaults to ``data.service.mem_limit_mb``
    and every other server to ``data.memory.default_server_mb``. ``auto`` scales with the host
    (:mod:`..memory.sizing`): a server gets ``0.8 x`` the host budget (at least 2,048 MB), the data
    child 5% of the memory it may plan with (3,000-32,768 MB). A number stays as given.
    ``limit_kind: none`` disables the limit.
    """
    memory = getattr(settings, "memory", None)
    if limit_kind(cfg, settings) == "none":
        return 0
    name = cfg if isinstance(cfg, str) else getattr(cfg, "name", "")
    own = None if isinstance(cfg, str) else getattr(cfg, "mem_limit_mb", None)
    explicit = _as_mb(own)
    if explicit is not None:
        return max(0, explicit)
    raw = _memory_raw(settings)
    plan = sizing.plan_mb(raw)
    if name == DATA_SERVER and not sizing.is_auto(own):
        value = getattr(getattr(settings, "service", None), "mem_limit_mb", None) if settings is not None else None
        if _as_mb(value) is not None:
            return max(0, _as_mb(value) or 0)
        if settings is None or sizing.is_auto(value):
            return sizing.data_child_for(plan) if plan else sizing.CHILD_FLOOR_MB
    value = getattr(memory, "default_server_mb", None) if memory is not None else None
    if not sizing.is_auto(own) and _as_mb(value) is not None:
        return max(0, _as_mb(value) or 0)
    return sizing.server_limit_for(plan, raw) if plan else UNKNOWN_HOST_SERVER_MB


def status_path(log_dir: str | Path, server: str) -> Path:
    """``<log_dir>/<server>.status.json``, the reaper's status file."""
    return Path(log_dir) / f"{server}.status.json"


def build_launch_spec(cfg: Any, settings: Any = None, log_dir: str | Path | None = None) -> LaunchSpec | None:
    """The reaper command line for a stdio server, or None to launch it as configured.

    None for HTTP servers, servers with ``launcher: false``, an empty command, and hosts other
    than Linux (no ``RLIMIT_DATA`` semantics). The child command and its arguments follow
    ``--`` unchanged; the reaper strips ``-E``/``-I`` from Python children itself.
    """
    if getattr(cfg, "url", None) or not str(getattr(cfg, "command", "") or "").strip():
        return None
    if getattr(cfg, "launcher", None) is False:
        return None
    if not sys.platform.startswith("linux"):
        return None
    name = str(cfg.name)
    limit = server_limit_mb(cfg, settings)
    status = status_path(Path(log_dir) if log_dir is not None else Path.cwd(), name)
    args = ["-E", str(REAPER), "--limit-mb", str(limit), "--status", str(status), "--server", name]
    kind = limit_kind(cfg, settings)
    if kind in ("cgroup", "watchdog", "rss"):          # rlimit_data is the reaper's default; none sets limit 0
        args += ["--containment", str(kind)]
    relay_mb = float(getattr(getattr(settings, "memory", None), "relay_max_message_mb", 0) or 0)
    if relay_mb > 0:                                   # data.memory.relay_max_message_mb (off by default)
        args += ["--relay-max-mb", f"{relay_mb:g}"]
    args += ["--", str(cfg.command), *[str(a) for a in (cfg.args or [])]]
    return LaunchSpec(command=sys.executable, args=args, env=dict(CHILD_ENV), status_path=str(status))

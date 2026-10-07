"""Launching stdio MCP servers under the reaper (§14.2). No pyarrow.

:func:`build_launch_spec` rewrites a server's command so it runs under
:mod:`vbt.datalayer.launch.reaper` (executed by path with the harness interpreter and ``-E``):
an ``RLIMIT_DATA`` memory limit, a status file next to the server log, a ``VBT_CHILD_EXIT``
marker in the server log when the child ends, and for Python children an explicit environment
in which ``PYTHONHASHSEED=0`` actually takes effect. The gateway's ``launch_spec(cfg)`` calls it
with the bridge's log directory (``MCPBridge.log_root()``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..api import LaunchSpec

__all__ = ["REAPER", "EXIT_MARKER", "CHILD_ENV", "DATA_SERVER", "build_launch_spec", "server_limit_mb",
           "host_memory_mb", "status_path"]

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

_AUTO_HOST_SHARE = 0.8


def host_memory_mb() -> int | None:
    """``MemTotal`` of this host in MB (Linux), else None."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _as_mb(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    return None


def server_limit_mb(cfg: Any, settings: Any = None) -> int:
    """The memory limit (MB) a server runs under; 0 means no limit.

    Per server ``mem_limit_mb`` wins; the data child defaults to ``data.service.mem_limit_mb``
    and every other server to ``data.memory.default_server_mb``. ``auto`` (or a default that
    is not a number) is ``0.8 x`` host memory at launch time; the gateway may set a tighter
    per-server value from estimates (§14.2). ``limit_kind: none`` disables the limit.
    """
    memory = getattr(settings, "memory", None)
    if getattr(memory, "limit_kind", "rlimit_data") == "none":
        return 0
    name = cfg if isinstance(cfg, str) else getattr(cfg, "name", "")
    explicit = None if isinstance(cfg, str) else _as_mb(getattr(cfg, "mem_limit_mb", None))
    if explicit is not None:
        return max(0, explicit)
    if name == DATA_SERVER and settings is not None:
        value = _as_mb(getattr(getattr(settings, "service", None), "mem_limit_mb", None))
        if value is not None:
            return max(0, value)
    value = _as_mb(getattr(memory, "default_server_mb", None)) if memory is not None else 12000
    if value is not None:
        return max(0, value)
    host = host_memory_mb()
    return int(host * _AUTO_HOST_SHARE) if host else 0


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
    kind = getattr(getattr(settings, "memory", None), "limit_kind", "rlimit_data")
    if kind in ("cgroup", "watchdog"):                 # rlimit_data is the reaper's default; none sets limit 0
        args += ["--containment", str(kind)]
    relay_mb = float(getattr(getattr(settings, "memory", None), "relay_max_message_mb", 0) or 0)
    if relay_mb > 0:                                   # data.memory.relay_max_message_mb (off by default)
        args += ["--relay-max-mb", f"{relay_mb:g}"]
    args += ["--", str(cfg.command), *[str(a) for a in (cfg.args or [])]]
    return LaunchSpec(command=sys.executable, args=args, env=dict(CHILD_ENV), status_path=str(status))

"""Host-scaled memory defaults: what ``auto`` means for each memory setting (§14.2, §14.5, §14.6). No pyarrow.

The harness runs on hosts from a 16 GB workstation to a 1 TB server, so its memory budgets derive from
``plan_mb``, the memory this harness may plan with:

* ``data.memory.host_mb``: ``auto`` (the default) is the smaller of ``MemTotal`` and the memory cgroup limit of
  this process and its ancestors (a container's ``--memory``, a systemd slice: ``/proc/meminfo`` shows neither).
  A number declares the harness's share of a host it shares (a GPU host whose model server holds RAM too).
* ``data.memory.host_budget_mb: auto``, the sum of the upstream servers' resident memory (``host.py``):
  ``0.75 x plan - reserve``, ``reserve = max(data.memory.harness_reserve_mb (2,048), 5% of plan)``,
  at least 1,024.
* ``data.memory.default_server_mb: auto``, one server's limit: ``0.8 x host budget``, at least 2,048. On a 512 GB
  host that is 293,601 MB, so the unmodified servers load every Open Targets table they read whole as upstream
  does (genetics' whole-table loads of the full 25.09 release are about 80 GB, docs/DEPLOYMENT.md §2.2); on a
  16 GB host it is 8,192 MB, and admission refuses the tables that cannot fit with ``too_large`` naming the host
  and the setting.
* ``data.service.mem_limit_mb: auto``, the data child: 5% of plan, 3,000 to 32,768; ``max_resident_mb: auto`` is
  two thirds of it.
* ``data.memory.workspace_mb: auto``, one agent command (Bash, the notebooks it runs, a utility's tests):
  ``0.25 x plan / limits.max_parallel_agents`` within 8,000-65,536, never above half the plan (:func:`workspace_for`).
* The witness and readiness budgets (``data.witness.max_scan_bytes``, ``max_inflate_bytes``, ``max_key_set``,
  ``repair_max_bytes``; ``data.readiness.vocab_budget_bytes``) at ``auto``: the shipped value times
  ``data child limit / 3,000``, never less than the shipped value.

A number anywhere stays as given. :func:`resolve_auto` replaces every ``auto`` of a ``data`` mapping with its
number (the shipped values are the floors); :func:`describe` lists, per key, what was configured, what applies on
this host and the rule. :func:`effective_memory_mb` is the stdlib probe the harness, the launcher and ``vbt
validate`` share.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "AUTO", "HOST_SHARE", "DEFAULT_RESERVE_MB", "RESERVE_FRACTION", "BUDGET_FLOOR_MB", "SERVER_SHARE",
    "SERVER_FLOOR_MB", "CHILD_FRACTION", "CHILD_FLOOR_MB", "CHILD_CEILING_MB", "SCALED", "HOST_MB_ENV",
    "WORKSPACE_FLOOR_MB", "WORKSPACE_CEILING_MB", "WORKSPACE_PLAN_SHARE", "workspace_for",
    "meminfo_total_mb", "cgroup_limit_mb", "effective_memory_mb", "plan_mb", "plan_source", "is_auto",
    "host_budget_for",
    "server_limit_for", "plan_for_server", "data_child_for", "witness_scale", "resolve_auto", "describe",
]

AUTO = "auto"
HOST_SHARE = 0.75
DEFAULT_RESERVE_MB = 2048.0
RESERVE_FRACTION = 0.05
BUDGET_FLOOR_MB = 1024.0
SERVER_SHARE = 0.8
SERVER_FLOOR_MB = 2048
CHILD_FRACTION = 0.05
CHILD_FLOOR_MB = 3000
CHILD_CEILING_MB = 32768
#: ``data.memory.workspace_mb: auto`` (agent Bash, notebooks, utility tests): :func:`workspace_for`.
WORKSPACE_FLOOR_MB = 8000
WORKSPACE_CEILING_MB = 65536
WORKSPACE_PLAN_SHARE = 0.5
#: ``(section, key)``: the shipped value, which ``auto`` scales by ``data child / CHILD_FLOOR_MB`` and never goes under.
SCALED: dict[tuple[str, str], int] = {
    ("witness", "max_scan_bytes"): 2_000_000_000,
    ("witness", "max_inflate_bytes"): 20_000_000,
    ("witness", "max_key_set"): 20000,
    ("witness", "repair_max_bytes"): 500_000_000,
    ("readiness", "vocab_budget_bytes"): 500_000_000,
}
#: Overrides the probe (tests, or any ``vbt`` command simulating another host): the MB to plan with.
HOST_MB_ENV = "VBT_HOST_MEMORY_MB"
_UNLIMITED = 1 << 60


def is_auto(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- the host


def meminfo_total_mb(proc_root: str = "/proc") -> float | None:
    """``MemTotal`` in MB (Linux), else None."""
    try:
        with open(os.path.join(proc_root, "meminfo"), encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None


def _limit_value(text: str | None) -> float | None:
    if not text or text == "max":
        return None
    try:
        raw = int(text)
    except ValueError:
        return None
    return raw / (1024.0 * 1024.0) if 0 < raw < _UNLIMITED else None


def cgroup_limit_mb(proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> float | None:
    """The tightest memory cgroup limit over this process's cgroup and its ancestors (v2 ``memory.max``, v1
    ``memory.limit_in_bytes``), in MB; None when no limit applies or none can be read."""
    try:
        lines = Path(proc_root, "self", "cgroup").read_text(encoding="ascii", errors="replace").splitlines()
    except OSError:
        return None
    candidates: list[tuple[Path, str]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        rel = (parts[2] or "/").strip("/")
        if parts[0] == "0" and parts[1] == "":
            candidates.append((Path(cgroup_root, rel) if rel else Path(cgroup_root), "memory.max"))
        if "memory" in parts[1].split(","):
            base = Path(cgroup_root, "memory")
            candidates.append((base / rel if rel else base, "memory.limit_in_bytes"))
    root = Path(cgroup_root)
    limits: list[float] = []
    for start, name in candidates:
        top = root / "memory" if name == "memory.limit_in_bytes" else root
        here = start
        while True:
            value = _limit_value(_read(here / name))
            if value is not None:
                limits.append(value)
            if here == top or top not in here.parents:
                break
            here = here.parent
    return min(limits) if limits else None


def effective_memory_mb(proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> float | None:
    """What this process tree may use: the smaller of ``MemTotal`` and the cgroup limit (MB), else None."""
    values = [v for v in (meminfo_total_mb(proc_root), cgroup_limit_mb(proc_root, cgroup_root)) if v]
    return min(values) if values else None


def plan_mb(memory: Mapping[str, Any] | None = None, *, measured: float | None = None) -> float | None:
    """The MB the budgets are planned from: ``data.memory.host_mb`` when it is a number, else ``$VBT_HOST_MEMORY_MB``,
    else ``measured``, else :func:`effective_memory_mb`."""
    own = _number((memory or {}).get("host_mb")) if not is_auto((memory or {}).get("host_mb")) else None
    if own and own > 0:
        return own
    env = _number(os.environ.get(HOST_MB_ENV))
    if env and env > 0:
        return env
    if measured:
        return float(measured)
    return effective_memory_mb()


def plan_source(memory: Mapping[str, Any] | None = None, *, measured: float | None = None) -> str:
    """Where :func:`plan_mb` takes its number from, in the words ``vbt validate`` and ``vbt ds status`` print."""
    own = _number((memory or {}).get("host_mb")) if not is_auto((memory or {}).get("host_mb")) else None
    if own and own > 0:
        return "data.memory.host_mb"
    env = _number(os.environ.get(HOST_MB_ENV))
    if env and env > 0:
        return HOST_MB_ENV
    return "measured" if measured else "MemTotal and the cgroup limit"


# --------------------------------------------------------------------------- the rules


def _reserve(plan: float, memory: Mapping[str, Any] | None) -> float:
    reserve = _number((memory or {}).get("harness_reserve_mb"))
    return max(reserve if reserve is not None else DEFAULT_RESERVE_MB, RESERVE_FRACTION * plan)


def host_budget_for(plan: float, memory: Mapping[str, Any] | None = None) -> float:
    """``max(1,024, 0.75 x plan - max(harness_reserve_mb, 5% of plan))`` MB."""
    return max(BUDGET_FLOOR_MB, HOST_SHARE * float(plan) - _reserve(float(plan), memory))


def server_limit_for(plan: float, memory: Mapping[str, Any] | None = None) -> int:
    """One upstream server's limit for ``default_server_mb: auto``: ``0.8 x host budget``, at least 2,048 MB."""
    return int(max(SERVER_FLOOR_MB, SERVER_SHARE * host_budget_for(plan, memory)))


def plan_for_server(need_mb: float, memory: Mapping[str, Any] | None = None) -> float:
    """The smallest plan (MB) whose ``auto`` server limit is at least ``need_mb``: what a refusal names as the host
    that would admit the call. ``server_limit_for`` grows with the plan, so a bisection finds it."""
    need = float(need_mb)
    if server_limit_for(1.0, memory) >= need:
        return 1.0
    lo, hi = 1.0, 2.0
    while server_limit_for(hi, memory) < need:
        lo, hi = hi, hi * 2
    while hi - lo > 1.0:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if server_limit_for(mid, memory) >= need else (mid, hi)
    return hi


def data_child_for(plan: float) -> int:
    """The data child's limit for ``mem_limit_mb: auto``: 5% of plan, within 3,000-32,768 MB."""
    return int(max(CHILD_FLOOR_MB, min(CHILD_CEILING_MB, CHILD_FRACTION * float(plan))))


def workspace_for(plan: float, parallel: int = 8) -> int:
    """An agent command's limit for ``data.memory.workspace_mb: auto``: ``0.25 x plan / limits.max_parallel_agents``
    within 8,000-65,536 MB, and never above half the plan (at least 1,024): 8,000 MB on a 16 GB host, 32,768 on
    1 TB with 8 agents, 3,000 on a 6,000 MB share (where 8,000 was more than the harness had)."""
    rule = max(WORKSPACE_FLOOR_MB, min(WORKSPACE_CEILING_MB, 0.25 * float(plan) / max(1, int(parallel))))
    return int(min(rule, max(1024.0, WORKSPACE_PLAN_SHARE * float(plan))))


def witness_scale(child_mb: float) -> float:
    """``data child / 3,000``, never below 1: how far the witness and readiness budgets grow past the shipped ones."""
    return max(1.0, float(child_mb) / CHILD_FLOOR_MB)


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def resolve_auto(data: Mapping[str, Any], *, measured_mb: float | None = None) -> dict[str, Any]:
    """A copy of a ``data`` settings mapping with every ``auto`` budget replaced by its number (the module
    docstring's rules). ``memory.host_budget_mb`` and ``memory.default_server_mb`` stay ``auto``: the host budget
    and the launcher resolve them where they are used (``host.host_budget_mb``, ``launch.server_limit_mb``). A host
    whose memory is unknown gets the shipped values."""
    out = copy.deepcopy(dict(data or {}))
    memory = _section(out, "memory")
    plan = plan_mb(memory, measured=measured_mb)
    service = _section(out, "service")
    if is_auto(service.get("mem_limit_mb")):
        child = float(data_child_for(plan)) if plan else float(CHILD_FLOOR_MB)
        service["mem_limit_mb"] = int(child)
    else:
        child = _number(service.get("mem_limit_mb")) or float(CHILD_FLOOR_MB)
    if is_auto(service.get("max_resident_mb")):
        service["max_resident_mb"] = int(child * 2 / 3)
    if service:
        out["service"] = service
    scale = witness_scale(child)
    for (section, key), floor in SCALED.items():
        sec = _section(out, section)
        if is_auto(sec.get(key)):
            sec[key] = int(floor * scale)
            out[section] = sec
    return out


def describe(data: Mapping[str, Any] | None, *, measured_mb: float | None = None,
             servers: Mapping[str, Any] | None = None, parallel: int = 8) -> dict[str, Any]:
    """Per memory setting: ``{configured, effective, rule}`` on this host, plus the host facts (``vbt validate``,
    ``vbt ds status``). ``data`` is the merged ``data`` mapping before :func:`resolve_auto` (``DataSettings.raw``).
    ``servers``: ``{name: mem_limit_mb}`` of servers with their own limit; ``parallel``:
    ``limits.max_parallel_agents`` (the workspace rule)."""
    data = dict(data or {})
    memory = _section(data, "memory")
    total = meminfo_total_mb()
    cgroup = cgroup_limit_mb()
    plan = plan_mb(memory, measured=measured_mb)
    out: dict[str, Any] = {
        "host": {"memtotal_mb": round(total) if total else None, "cgroup_limit_mb": round(cgroup) if cgroup else None,
                 "plan_mb": round(plan) if plan else None,
                 "plan_from": plan_source(memory, measured=measured_mb)},
        "settings": {}}
    resolved = resolve_auto(data, measured_mb=measured_mb)

    def put(key: str, configured: Any, effective: Any, rule: str) -> None:
        out["settings"][key] = {"configured": configured, "effective": effective, "rule": rule}

    hb = memory.get("host_budget_mb", AUTO)
    if is_auto(hb):
        put("memory.host_budget_mb", hb, round(host_budget_for(plan, memory)) if plan else None,
            "0.75 x plan - max(harness_reserve_mb, 5% of plan)")
    else:
        put("memory.host_budget_mb", hb, hb, "as configured")
    ds = memory.get("default_server_mb", AUTO)
    put("memory.default_server_mb", ds, server_limit_for(plan, memory) if is_auto(ds) and plan else ds,
        "0.8 x host budget, at least 2,048" if is_auto(ds) else "as configured")
    put("memory.limit_kind", memory.get("limit_kind"), memory.get("limit_kind"),
        "rss: a memory cgroup when one can be created, else the RSS watchdog; no RLIMIT_DATA"
        if memory.get("limit_kind") == "rss" else "as configured")
    ws = memory.get("workspace_mb", AUTO)
    put("memory.workspace_mb", ws, workspace_for(plan, parallel) if is_auto(ws) and plan else
        (WORKSPACE_FLOOR_MB if is_auto(ws) else ws),
        "0.25 x plan / max_parallel_agents within 8,000-65,536, at most half the plan" if is_auto(ws)
        else "as configured")
    for key in ("mem_limit_mb", "max_resident_mb"):
        configured = _section(data, "service").get(key)
        put(f"service.{key}", configured, _section(resolved, "service").get(key),
            ("5% of plan within 3,000-32,768" if key == "mem_limit_mb" else "2/3 of the data child's limit")
            if is_auto(configured) else "as configured")
    for (section, key), floor in SCALED.items():
        configured = _section(data, section).get(key)
        put(f"{section}.{key}", configured, _section(resolved, section).get(key),
            f"{floor:,} x data child / 3,000 (at least {floor:,})" if is_auto(configured) else "as configured")
    for name, limit in sorted((servers or {}).items()):
        put(f"mcp_servers.{name}.mem_limit_mb", limit,
            server_limit_for(plan, memory) if is_auto(limit) and plan else limit,
            "0.8 x host budget" if is_auto(limit) else "as configured")
    return out

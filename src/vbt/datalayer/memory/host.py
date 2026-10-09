"""Host memory budget across upstream servers (§14.3, phase 4 F19). No pyarrow.

Per-server limits stop one server from exhausting the host, but the servers together can: the
target, drug and pathway servers each cache the ``target`` table (about 3.5 GB EST each). The host
budget caps the **sum** of resident memory over every server the bridge runs:

* ``data.memory.host_budget_mb``: a number of MB, ``auto`` (``0.75 x plan - reserve``, where plan is the memory
  this process tree may use, the smaller of ``MemTotal`` and its memory cgroup limit, or ``data.memory.host_mb``,
  and the reserve is the larger of ``data.memory.harness_reserve_mb`` (default 2,048 MB, for the orchestrator, the
  model client and the data child) and 5% of plan: :mod:`.sizing`), or ``off``/``0`` (no host budget);
* before an upstream call loads tables, :meth:`HostBudget.reserve` checks ``sum(resident) + need``;
  over budget, it recycles the **least recently used idle** server (no call in flight, not the one
  asking) and checks again, until the need fits or no idle server is left;
* still over budget: ``too_large`` with ``subkind: host_busy`` (retryable later: busy servers will
  finish), never a kill of a server that is answering a call.

Resident memory per server comes from the :class:`~.ledger.ResidencyLedger` (the reaper's RSS when
fresh, else the estimates). Calls mark servers busy with :meth:`HostBudget.begin` / :meth:`end`
(``AdmissionController`` does it for upstream calls), which also refreshes their LRU position.

The budget covers the **upstream** servers only. The harness's own servers (:data:`HARNESS_SERVERS`, the data
child) are inside the harness reserve and under their own limit (``data.service.mem_limit_mb``): the budget never
counts, admits or recycles them, whatever a caller's server listing or the ledger names (on a 6,000 MB share the
data child's 1.6 GB after a few finds filled the 2,452 MB budget, and every upstream call was refused host_busy).
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Awaitable, Callable, Iterable, Mapping

from ..errors import ErrorKind, GatewayError, too_large_payload
from . import sizing
from .ledger import ResidencyLedger

__all__ = ["HOST_SHARE", "DEFAULT_RESERVE_MB", "HARNESS_SERVERS", "host_total_mb", "host_budget_mb", "HostBudget"]

HOST_SHARE = sizing.HOST_SHARE
DEFAULT_RESERVE_MB = sizing.DEFAULT_RESERVE_MB
#: Servers the harness runs itself (``launch.DATA_SERVER``): never part of the upstream budget.
HARNESS_SERVERS = frozenset({"data"})


def host_total_mb() -> float | None:
    """``MemTotal`` of this host in MB (Linux), else None."""
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _raw_memory(settings: Any) -> Mapping[str, Any]:
    raw = getattr(settings, "raw", None) or {}
    mem = raw.get("memory") if isinstance(raw, Mapping) else None
    return mem if isinstance(mem, Mapping) else {}


def host_budget_mb(settings: Any = None, *, total_mb: float | None = None) -> float | None:
    """The host budget in MB, or None when it is off (``off``, ``0``, or ``auto`` on a host whose memory
    is unknown). ``auto`` plans from ``total_mb`` when given, else ``data.memory.host_mb``, else the memory this
    process tree may use (:func:`.sizing.plan_mb`)."""
    mem = getattr(settings, "memory", None)
    value = getattr(mem, "host_budget_mb", "auto") if mem is not None else "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("off", "none", "0", ""):
            return None
        if text != "auto":
            try:
                value = float(text)
            except ValueError:
                return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value > 0 else None
    raw = _raw_memory(settings)
    plan = float(total_mb) if total_mb else sizing.plan_mb(raw)
    if not plan:
        return None
    return sizing.host_budget_for(plan, raw)


class HostBudget:
    """Total resident memory across servers, with LRU idle recycling (§14.3 host budget)."""

    def __init__(self, budget_mb: float | None, ledger: ResidencyLedger, *,
                 recycle: Callable[..., Awaitable[bool] | bool] | None = None,
                 servers: Callable[[], Iterable[str]] | None = None, recycle_wait_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic, exclude: Iterable[str] = HARNESS_SERVERS) -> None:
        self.budget_mb = budget_mb
        self.ledger = ledger
        self.recycle = recycle
        self.servers = servers
        self.exclude = frozenset(exclude)       # not upstream: never counted, admitted or recycled
        self.recycle_wait_s = float(recycle_wait_s)
        self.clock = clock
        self.in_flight: dict[str, int] = {}
        self.last_used: dict[str, float] = {}
        self.evictions: list[tuple[str, float]] = []
        self.origin: str | None = None          # how the budget was set, for refusals

    @classmethod
    def from_settings(cls, settings: Any, ledger: ResidencyLedger, **kw: Any) -> "HostBudget":
        mem = getattr(settings, "memory", None)
        kw.setdefault("recycle_wait_s", float(getattr(mem, "recycle_wait_s", 30) or 30))
        host = cls(host_budget_mb(settings), ledger, **kw)
        configured = getattr(mem, "host_budget_mb", "auto") if mem is not None else "auto"
        plan = sizing.plan_mb(_raw_memory(settings))
        if sizing.is_auto(configured) and plan:
            host.origin = (f"data.memory.host_budget_mb is auto: 0.75 x this host's {plan:,.0f} MB less the harness "
                           "reserve")
        elif host.budget_mb is not None:
            host.origin = "data.memory.host_budget_mb is configured"
        return host

    @property
    def enabled(self) -> bool:
        return self.budget_mb is not None

    # ------------------------------------------------------------------ activity

    def begin(self, server: str) -> None:
        self.in_flight[server] = self.in_flight.get(server, 0) + 1
        self.last_used[server] = self.clock()

    def end(self, server: str) -> None:
        n = self.in_flight.get(server, 0) - 1
        if n > 0:
            self.in_flight[server] = n
        else:
            self.in_flight.pop(server, None)
        self.last_used[server] = self.clock()

    def touch(self, server: str) -> None:
        self.last_used[server] = self.clock()

    def busy(self, server: str) -> bool:
        return self.in_flight.get(server, 0) > 0

    # ------------------------------------------------------------------ accounting

    def known_servers(self) -> list[str]:
        """The upstream servers the budget covers (never :attr:`exclude`)."""
        names = set(self.last_used) | set(self.in_flight) | set(self.ledger.servers())
        if self.servers is not None:
            try:
                names |= set(self.servers())
            except Exception:  # noqa: BLE001 - a listing failure only narrows the view
                pass
        return sorted(names - self.exclude)

    def resident_mb(self, exclude: Iterable[str] = ()) -> dict[str, float]:
        skip = set(exclude)
        return {s: self.ledger.resident_mb(s) for s in self.known_servers() if s not in skip}

    def total_mb(self) -> float:
        return sum(self.resident_mb().values())

    def idle_lru(self, exclude: Iterable[str] = ()) -> list[str]:
        """Idle servers holding tables, least recently used first."""
        skip = set(exclude)
        cands = [s for s in self.known_servers() if s not in skip and not self.busy(s)
                 and (self.ledger.resident(s) or self.ledger.pending(s))]
        return sorted(cands, key=lambda s: (self.last_used.get(s, float("-inf")), s))

    # ------------------------------------------------------------------ admission

    async def reserve(self, server: str, need_mb: float, *, tool: str | None = None) -> list[str]:
        """Make room for ``need_mb`` more on ``server``; returns the servers recycled for it. Raises
        ``too_large`` (``host_busy``) when the budget cannot be met without touching a busy server."""
        if not self.enabled or need_mb <= 0 or server in self.exclude:
            return []
        evicted: list[str] = []
        while True:
            total = self.total_mb()
            if total + need_mb <= float(self.budget_mb or 0):
                return evicted
            victims = self.idle_lru(exclude=[server, *evicted])
            if not victims or self.recycle is None:
                busy = sorted(s for s in self.known_servers() if self.busy(s))
                payload = too_large_payload("host_busy", need_mb=round(need_mb, 1),
                                            limit_mb=round(float(self.budget_mb or 0), 1), alternative=None,
                                            hint="wait for the busy servers to finish, or narrow the call",
                                            learned=False)
                payload.update({"host_resident_mb": round(total, 1), "busy_servers": busy,
                                "recycled": list(evicted)})
                origin = f"; {self.origin}" if self.origin else ""
                raise GatewayError(ErrorKind.too_large,
                                   f"the host memory budget ({self.budget_mb:,.0f} MB{origin}) is in use: servers "
                                   f"hold about {total:,.0f} MB and {server} needs about {need_mb:,.0f} MB more; no "
                                   f"idle server is left to recycle", tool=tool, payload=payload, subkind="host_busy",
                                   retryable="later")
            victim = victims[0]
            ok = await self._recycle(victim)
            if not ok:
                raise GatewayError(ErrorKind.too_large, f"the host memory budget is exhausted and {victim} could "
                                   "not be recycled", tool=tool, subkind="host_busy", retryable="later",
                                   payload={"need_mb": round(need_mb, 1), "victim": victim})
            self.ledger.reset(victim)
            evicted.append(victim)
            self.evictions.append((victim, self.clock()))

    async def _recycle(self, server: str) -> bool:
        try:
            result = self.recycle(server, wait_s=self.recycle_wait_s)  # type: ignore[misc]
            if inspect.isawaitable(result):
                result = await result
        except Exception:  # noqa: BLE001 - a failed recycle refuses the call instead
            return False
        return result is not False

    def snapshot(self) -> dict[str, Any]:
        return {"budget_mb": self.budget_mb, "resident_mb": {k: round(v, 1) for k, v in self.resident_mb().items()},
                "busy": sorted(s for s in self.known_servers() if self.busy(s)),
                "evictions": [list(e) for e in self.evictions[-20:]]}

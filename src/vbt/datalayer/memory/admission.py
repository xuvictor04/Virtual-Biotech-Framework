"""Memory admission for upstream calls (§14.3). No pyarrow.

``prepare`` asks :meth:`AdmissionController.admit` before an upstream call whose tool loads
tables whole (``reads.<table>.access: full_table``) or scans them (``bounded_scan``):

1. Only the ``upstream`` route is admitted; derived and witness work is charged to the data
   child's own budget (rev 2), so other routes get an empty :class:`Admission`.
2. **Permanent** ``too_large``: the tool's full reads cannot fit even in a fresh server
   (``sum peak x safety > limit - baseline``).
3. **Learned** ``too_large``: the server was killed for memory loading these cold tables before.
4. **Resident pressure**: ``resident + need x safety > limit`` recycles the idle server (which
   empties the upstream forever-cache) unless the thrash guard (``max_recycles_per_10min``)
   says no, in which case the call is ``too_large`` and retryable later.
5. A call that will load a cold table gets the server's **cold-call lock** (``CallPlan.hold``):
   at most one loading call per server is in flight, so loads are never doubled and an OOM is
   attributable to the single cold call. Warm calls run concurrently.

``commit`` records the outcome (cold tables become resident; an OOM becomes a learned refusal).
:meth:`admit_remote` admits remote reads count-first (``total x est_row_bytes`` against a cap).

Phase 4 (F19): with a :class:`~.host.HostBudget` (``host=`` or :meth:`AdmissionController.enable_host_budget`)
the sum over servers is capped too, recycling the least recently used idle server before a cold load
and refusing with ``too_large``/``host_busy`` when only busy servers hold the memory. With
``feedback_dir`` (``data.cache_dir``), a committed cold load whose server has a fresh reaper status
records ``(peak RSS - resident before) / estimate`` as measured feedback (:mod:`.calibrate`).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping

from ..errors import ErrorKind, GatewayError, too_large_payload
from . import sizing
from .estimate import MB, MemoryEstimator
from .host import HostBudget
from .ledger import ResidencyLedger

__all__ = ["TableRead", "Admission", "AdmissionController", "RECYCLE_WINDOW_S"]

RECYCLE_WINDOW_S = 600.0
_HINT = "narrow the call (fewer rows or a more specific filter) or use the named alternative"


@dataclass(frozen=True)
class TableRead:
    """One table an upstream call reads, with the ``_stats`` payload used to estimate it."""

    table: str                              # "source.table"
    access: str = "full_table"              # overlay ReadSpec.access
    stats: Any = None                       # TableStatsModel | its JSON form | None (not estimable)
    selectivity: float | None = None        # bounded_scan: matching share under the predicate


@dataclass
class Admission:
    """The memory decision for one call (empty for routes other than ``upstream``)."""

    server: str | None = None
    cold_tables: tuple[str, ...] = ()
    lock: asyncio.Lock | None = None
    need_mb: float = 0.0
    resident_mb: float | None = None
    limit_mb: float | None = None
    generation: int | None = None
    recycled: bool = False
    unestimated: tuple[str, ...] = ()
    peaks_mb: dict[str, float] = field(default_factory=dict)
    reservation: int | None = None
    committed: bool = False
    fingerprints: dict[str, str] = field(default_factory=dict)
    tiers: dict[str, str] = field(default_factory=dict)        # estimate tier per cold table (measured|sample|seed)
    host_evicted: tuple[str, ...] = ()
    host_held: bool = False

    @property
    def admitted(self) -> bool:
        return self.server is not None

    def to_record(self) -> dict[str, Any]:
        """The ``memory`` block of the provenance record."""
        if not self.admitted:
            return {"admission": "not_applicable"}
        out: dict[str, Any] = {"admission": "admitted", "cold_tables": list(self.cold_tables),
                               "need_mb": round(self.need_mb, 1), "generation": self.generation}
        if self.resident_mb is not None:
            out["resident_mb"] = round(self.resident_mb, 1)
        if self.limit_mb is not None:
            out["limit_mb"] = round(self.limit_mb, 1)
        if self.recycled:
            out["recycled"] = True
        if self.unestimated:
            out["unestimated"] = list(self.unestimated)
        if self.tiers:
            out["estimate_tiers"] = dict(sorted(self.tiers.items()))
        if self.host_evicted:
            out["host_recycled"] = list(self.host_evicted)
        return out


def _reads(reads: Iterable[Any] | Mapping[str, Any]) -> list[TableRead]:
    """``TableRead`` objects from TableReads, ``(table, access, stats[, selectivity])`` tuples,
    or a ``{table: stats}`` mapping of full-table reads."""
    if isinstance(reads, Mapping):
        return [TableRead(str(t), "full_table", s) for t, s in reads.items()]
    out = []
    for r in reads:
        if isinstance(r, TableRead):
            out.append(r)
        elif isinstance(r, (tuple, list)):
            out.append(TableRead(*r))
        else:
            raise TypeError(f"not a table read: {r!r}")
    return out


class AdmissionController:
    """Admits upstream calls against per-server memory limits (§14.3)."""

    def __init__(self, settings: Any, estimator: MemoryEstimator | None = None,
                 ledger: ResidencyLedger | None = None,
                 recycle: Callable[..., Awaitable[bool] | bool] | None = None,
                 limits: Mapping[str, float] | Callable[[str], float | None] | None = None, *,
                 generation: Callable[[str], int | None] | None = None,
                 clock: Callable[[], float] = time.monotonic, host: HostBudget | None = None,
                 feedback_dir: Any = None) -> None:
        self.settings = settings
        self.est = estimator or MemoryEstimator.from_settings(settings)
        self.ledger = ledger or ResidencyLedger()
        self.recycle = recycle
        self.limits = limits
        self.generation = generation
        self.clock = clock
        mem = getattr(settings, "memory", None)
        self.recycle_idle = bool(getattr(mem, "recycle_idle_servers", True))
        self.recycle_wait_s = float(getattr(mem, "recycle_wait_s", 30))
        self.max_recycles = int(getattr(mem, "max_recycles_per_10min", 4))
        self.learned_refusals: set[tuple[str, frozenset[str]]] = set()
        self._cold_locks: dict[str, asyncio.Lock] = {}
        self._recycles: dict[str, deque[float]] = {}
        self.host = host
        self.feedback_dir = feedback_dir
        self._servers: Callable[[], Any] | None = None

    def enable_host_budget(self, budget_mb: float | None = None) -> HostBudget:
        """Cap the sum of resident memory over servers (``data.memory.host_budget_mb`` unless given)."""
        host = HostBudget.from_settings(self.settings, self.ledger, recycle=self.recycle, servers=self._servers)
        if budget_mb is not None:
            host.budget_mb = float(budget_mb) if budget_mb > 0 else None
        self.host = host
        return host

    def bind_bridge(self, bridge: Any) -> None:
        """Take recycle, generations and status files from an ``MCPBridge``."""
        self.recycle = bridge.recycle

        def _generation(server: str) -> int | None:
            return (bridge.status().get(server) or {}).get("generation")

        def _servers() -> list[str]:
            return sorted(bridge.status())

        self.generation = _generation
        self._servers = _servers
        if self.host is not None:
            self.host.recycle = bridge.recycle
            self.host.servers = _servers

    # ------------------------------------------------------------------ limits and state

    def limit_mb(self, server: str) -> float:
        """The server's memory limit in MB (explicit, else the reaper's, else the configured default);
        0 means no limit."""
        value: Any = None
        if callable(self.limits):
            value = self.limits(server)
        elif isinstance(self.limits, Mapping):
            value = self.limits.get(server)
        if value is None:
            value = self.ledger.limit_mb(server)
        if value is None:
            from ..launch import server_limit_mb     # launch imports memory.sizing: imported on use

            value = server_limit_mb(server, self.settings)
        return max(0.0, float(value))

    def limit_origin(self, server: str, limit_mb: float, need_mb: float) -> tuple[str, dict[str, Any]]:
        """Where ``limit_mb`` comes from on this host, in words, and the payload keys that say it: ``host_mb`` (the
        memory the limits are planned from), ``limit_source`` (``auto`` when the limit is the host-scaled one, else
        ``configured``) and, for ``auto``, ``host_mb_needed`` (the host whose auto limit admits ``need_mb``)."""
        raw = getattr(self.settings, "raw", None) or {}
        memory = raw.get("memory") if isinstance(raw, Mapping) else None
        memory = memory if isinstance(memory, Mapping) else {}
        plan = sizing.plan_mb(memory)
        auto = sizing.server_limit_for(plan, memory) if plan else None
        extra: dict[str, Any] = {"host_mb": round(plan) if plan else None}
        if auto is not None and abs(float(limit_mb) - auto) < 1.0:
            needed = sizing.plan_for_server(need_mb, memory)
            extra.update({"limit_source": "auto", "host_mb_needed": round(needed)})
            return (f"data.memory.default_server_mb is auto: 0.8 x the host budget of this {plan:,.0f} MB host; a "
                    f"host with about {needed / 1024:,.0f} GB admits it, or set the server's mem_limit_mb in "
                    "configs/mcp_servers.yaml"), extra
        extra["limit_source"] = "configured"
        host = f" on this {plan:,.0f} MB host" if plan else ""
        return (f"the limit is configured{host}: data.memory.default_server_mb or the server's mem_limit_mb in "
                "configs/mcp_servers.yaml"), extra

    def cold_lock(self, server: str) -> asyncio.Lock:
        lock = self._cold_locks.get(server)
        if lock is None:
            lock = self._cold_locks[server] = asyncio.Lock()
        return lock

    def _sync(self, server: str) -> int | None:
        gen = None
        if self.generation is not None:
            try:
                gen = self.generation(server)
            except Exception:  # noqa: BLE001 - a missing server has no generation yet
                gen = None
        self.ledger.sync(server, gen)
        return self.ledger.generation(server)

    def can_recycle(self, server: str) -> bool:
        """Recycling is enabled, possible, and under ``max_recycles_per_10min`` for this server."""
        if not self.recycle_idle or self.recycle is None:
            return False
        times = self._recycles.setdefault(server, deque())
        now = self.clock()
        while times and now - times[0] > RECYCLE_WINDOW_S:
            times.popleft()
        return len(times) < self.max_recycles

    def is_learned(self, server: str, cold: Iterable[str]) -> bool:
        cold = frozenset(cold)
        return bool(cold) and any(s == server and tables <= cold for s, tables in self.learned_refusals)

    def learn_refusal(self, server: str, tables: Iterable[str]) -> None:
        """Remember that ``server`` was killed for memory loading ``tables`` (non-empty)."""
        tables = frozenset(tables)
        if tables:
            self.learned_refusals.add((server, tables))

    # ------------------------------------------------------------------ admission

    def _too_large(self, reason: str, message: str, *, tool: str | None, need_mb: float | None,
                   limit_mb: float | None, alternative: str | None, retryable: str | None = None,
                   learned: bool = False, extra: Mapping[str, Any] | None = None) -> GatewayError:
        payload = too_large_payload(reason, need_mb=None if need_mb is None else round(need_mb, 1),
                                    limit_mb=None if limit_mb is None else round(limit_mb, 1),
                                    alternative=alternative, hint=_HINT, learned=learned)
        payload.update(extra or {})
        return GatewayError(ErrorKind.too_large, message, tool=tool, payload=payload, subkind=reason,
                            retryable=retryable)

    async def admit(self, server: str, reads: Iterable[Any] | Mapping[str, Any], route: str, *,
                    tool: str | None = None, alternative: str | None = None) -> Admission:
        """Admit one call or raise ``too_large`` before anything is allocated (§14.3)."""
        if route != "upstream":
            return Admission()
        reads = _reads(reads)
        full = [r for r in reads if r.access == "full_table"]
        scans = [r for r in reads if r.access == "bounded_scan"]
        generation = self._sync(server)
        limit = self.limit_mb(server)
        safety = self.est.safety
        peaks = {r.table: self.est.peak_upstream(r.stats, r.table) / MB for r in full if r.stats is not None}
        fingerprints = {r.table: str(fp) for r in full if r.stats is not None
                        and (fp := (r.stats.get("fingerprint") if isinstance(r.stats, Mapping)
                                    else getattr(r.stats, "fingerprint", None)))}
        tiers = {r.table: self.est.tier(r.stats, r.table) for r in full if r.stats is not None}
        unestimated = tuple(sorted({r.table for r in full + scans if r.stats is None}))
        transient = sum(self.est.transient(r.stats, r.selectivity) / MB for r in scans if r.stats is not None)

        baseline = self.ledger.baseline_mb(server)
        everything = sum(peaks.values()) + transient
        if limit > 0 and everything * safety > limit - baseline:
            where, sizing_extra = self.limit_origin(server, limit, everything * safety + baseline)
            raise self._too_large(
                "over_limit", f"{server}: this call needs about {everything * safety:,.0f} MB "
                f"(x{safety:g} safety) but the server's limit is {limit:,.0f} MB with a {baseline:,.0f} MB baseline; "
                f"it cannot run on this host ({where})", tool=tool, need_mb=everything * safety, limit_mb=limit,
                alternative=alternative, extra={"permanent": True, **sizing_extra})

        recycled = False
        while True:
            resident = self.ledger.resident(server)
            cold = tuple(sorted({r.table for r in full} - resident))
            if self.is_learned(server, cold):
                raise self._too_large(
                    "learned_refusal", f"{server} was killed for memory loading {', '.join(cold)} earlier in "
                    "this session; the same load is refused", tool=tool,
                    need_mb=sum(peaks.get(t, 0.0) for t in cold) * safety, limit_mb=limit or None,
                    alternative=alternative, learned=True)
            pending = self.ledger.pending(server)
            need = sum(peaks.get(t, 0.0) for t in cold if t not in pending) + transient
            resident_mb = self.ledger.resident_mb(server)
            if limit <= 0 or resident_mb + need * safety <= limit:
                break
            if recycled or not self.can_recycle(server) or not await self._recycle(server):
                raise self._too_large(
                    "resident_memory", f"{server} holds about {resident_mb:,.0f} MB; loading "
                    f"{', '.join(cold) or 'this scan'} needs about {need * safety:,.0f} MB more than its "
                    f"{limit:,.0f} MB limit allows and the server could not be recycled now", tool=tool,
                    need_mb=need * safety, limit_mb=limit, alternative=alternative, retryable="later",
                    extra={"resident_mb": round(resident_mb, 1)})
            recycled = True
            generation = self._sync(server)
            self.ledger.reset(server, generation)

        evicted: list[str] = []
        if self.host is not None and self.host.enabled and need > 0:
            evicted = await self.host.reserve(server, need * safety, tool=tool)
        reservation = None
        if cold:
            reservation = self.ledger.reserve(server, {t: peaks.get(t, 0.0) for t in cold})
        held = False
        if self.host is not None:
            self.host.begin(server)
            held = True
        return Admission(server=server, cold_tables=cold, lock=self.cold_lock(server) if cold else None,
                         need_mb=need * safety, resident_mb=resident_mb, limit_mb=limit or None,
                         generation=generation, recycled=recycled, unestimated=unestimated,
                         peaks_mb={t: round(peaks.get(t, 0.0), 1) for t in cold}, reservation=reservation,
                         fingerprints={t: fingerprints[t] for t in cold if t in fingerprints},
                         tiers={t: tiers[t] for t in cold if t in tiers}, host_evicted=tuple(evicted),
                         host_held=held)

    async def recycle_after_oom(self, server: str) -> bool:
        """§14.4: restart ``server`` after an in-tool memory error (it may hold a partial cache near its
        limit), within the thrash guard. False when recycling is off, guarded or failed."""
        if not self.can_recycle(server):
            return False
        return await self._recycle(server)

    async def _recycle(self, server: str) -> bool:
        """Recycle ``server`` (counted by the thrash guard). False when it could not be done."""
        self._recycles.setdefault(server, deque()).append(self.clock())
        try:
            result = self.recycle(server, wait_s=self.recycle_wait_s)  # type: ignore[misc]
            if inspect.isawaitable(result):
                result = await result
        except Exception:  # noqa: BLE001 - a failed recycle refuses the call instead
            return False
        return result is not False

    def commit(self, admission: Admission, ok: bool = True, *, oom: bool = False) -> None:
        """Record the outcome of an admitted call: on success its cold tables are resident; an
        OOM makes ``(server, cold tables)`` a learned refusal. Idempotent."""
        if not admission.admitted or admission.committed:
            return
        admission.committed = True
        server = admission.server or ""
        if admission.host_held and self.host is not None:
            self.host.end(server)
        self.ledger.release(server, admission.reservation)
        if oom:
            self.learn_refusal(server, admission.cold_tables)
            return
        if ok and admission.cold_tables:
            gen = self._sync(server)
            if admission.generation is not None and gen != admission.generation:
                return       # the server restarted during the call: what it loaded is gone
            self.ledger.add(server, {t: admission.peaks_mb.get(t, 0.0) for t in admission.cold_tables},
                            generation=gen)
            self._record_feedback(admission)

    def _record_feedback(self, admission: Admission) -> None:
        """Measured feedback (§10.4 tier ``measured``): the server's peak RSS after a cold load over the
        estimate of the tables it loaded. Needs a fresh reaper status and ``feedback_dir``."""
        if not self.feedback_dir or admission.resident_mb is None:
            return
        from .calibrate import feedback_from_status, record_feedback

        server = admission.server or ""
        before = admission.resident_mb - self.ledger.reserved_mb(server)
        measured = feedback_from_status(self.ledger.status(server), before)
        tables = {t: {"estimated_mb": admission.peaks_mb.get(t, 0.0), "fingerprint": admission.fingerprints.get(t)}
                  for t in admission.cold_tables if admission.peaks_mb.get(t)}
        if measured is None or not tables:
            return
        try:
            data = record_feedback(self.feedback_dir, tables, measured, server=server)
        except OSError:
            return
        for t, info in tables.items():
            key = str(info.get("fingerprint") or t)
            entry = (data.get("tables") or {}).get(key)
            if entry:
                self.est.add_feedback(key, entry["factor"])

    def admit_remote(self, total: int | None, est_row_bytes: int | None, cap: int, *, tool: str | None = None,
                     table: str | None = None, alternative: str | None = None) -> int | None:
        """Count-first admission of a remote read (``size_from``/``count_via``): ``total x
        est_row_bytes`` over ``cap`` is ``too_large`` before the call. Returns the estimated bytes,
        or None when the count or the row size is unknown."""
        if total is None or not est_row_bytes:
            return None
        need = int(total) * int(est_row_bytes)
        if need > int(cap):
            raise self._too_large(
                "remote_size", f"{table or 'the remote source'}: {int(total):,} rows x about {int(est_row_bytes):,} "
                f"bytes is about {need / MB:,.1f} MB, over the {int(cap) / MB:,.1f} MB cap; narrow the request",
                tool=tool, need_mb=need / MB, limit_mb=int(cap) / MB, alternative=alternative,
                extra={"total": int(total), "est_row_bytes": int(est_row_bytes), "need_bytes": need,
                       "cap_bytes": int(cap)})
        return need

    def snapshot(self) -> dict[str, Any]:
        out = {"ledger": self.ledger.snapshot(),
               "learned_refusals": sorted([s, sorted(t)] for s, t in self.learned_refusals)}
        if self.host is not None:
            out["host"] = self.host.snapshot()
        return out

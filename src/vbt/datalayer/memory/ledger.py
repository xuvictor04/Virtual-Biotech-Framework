"""Residency ledger: which tables each upstream server holds in memory (§14.3). No pyarrow.

Upstream servers cache every table they load for the life of the process. The ledger records,
per server **generation** (bumped by the bridge on every start, restart or recycle), the tables
loaded in that generation with their estimated size, plus reservations for cold loads admitted
but not yet finished. A generation change empties it: a restarted server holds nothing.

Resident memory prefers the reaper's status file (``<log_dir>/<server>.status.json``, rewritten
every second with the child's RSS) and falls back to ``baseline + sum(resident estimates)``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["DEFAULT_BASELINE_MB", "STATUS_MAX_AGE_S", "RESERVATION_TTL_S", "read_status", "Residency",
           "ResidencyLedger"]

#: RSS of an idle upstream server (interpreter, FastMCP, pandas imported) before any table loads.
DEFAULT_BASELINE_MB = 300.0
#: A status file older than this is stale (the reaper rewrites it every second).
STATUS_MAX_AGE_S = 5.0
#: A cold-load reservation that was never committed (the call timed out) lapses after this.
RESERVATION_TTL_S = 1800.0


def read_status(path: str | Path | None) -> dict[str, Any] | None:
    """The reaper's status JSON at ``path``, or None when missing or unreadable."""
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@dataclass
class Residency:
    """One server's memory state in its current generation."""

    generation: int | None = None
    tables: dict[str, float] = field(default_factory=dict)            # table -> estimated MB
    reserved: dict[int, tuple[dict[str, float], float]] = field(default_factory=dict)  # id -> (tables, at)
    reset_at: float = field(default_factory=time.time)
    baseline_mb: float | None = None
    status_path: str | None = None
    loaded_at: dict[str, float] = field(default_factory=dict)


class ResidencyLedger:
    """Resident tables and memory per server and generation."""

    def __init__(self, *, baseline_mb: float = DEFAULT_BASELINE_MB, status_max_age_s: float = STATUS_MAX_AGE_S,
                 reservation_ttl_s: float = RESERVATION_TTL_S, clock: Callable[[], float] = time.time) -> None:
        self.default_baseline_mb = float(baseline_mb)
        self.status_max_age_s = float(status_max_age_s)
        self.reservation_ttl_s = float(reservation_ttl_s)
        self.clock = clock
        self._servers: dict[str, Residency] = {}
        self._next_id = 0

    def _get(self, server: str) -> Residency:
        r = self._servers.get(server)
        if r is None:
            r = self._servers[server] = Residency(reset_at=self.clock())
        return r

    # ------------------------------------------------------------------ generations

    def generation(self, server: str) -> int | None:
        return self._get(server).generation

    def sync(self, server: str, generation: int | None) -> bool:
        """Follow the bridge's generation of ``server``; a change empties the ledger. Returns
        True when it was reset."""
        r = self._get(server)
        if generation is None or r.generation == generation:
            return False
        if r.generation is None and not r.tables and not r.reserved:
            r.generation = generation
            return False
        self.reset(server, generation)
        return True

    def reset(self, server: str, generation: int | None = None) -> None:
        """Forget everything ``server`` held (restart, recycle). The status file written before
        now is ignored from here on."""
        r = self._get(server)
        r.tables.clear()
        r.reserved.clear()
        r.loaded_at.clear()
        r.baseline_mb = None
        r.reset_at = self.clock()
        if generation is not None:
            r.generation = generation

    # ------------------------------------------------------------------ status file

    def set_status_path(self, server: str, path: str | Path | None) -> None:
        self._get(server).status_path = str(path) if path else None

    def status_path(self, server: str) -> str | None:
        return self._get(server).status_path

    def status(self, server: str) -> dict[str, Any] | None:
        """The reaper status of ``server`` when it is fresh and written in this generation."""
        r = self._get(server)
        data = read_status(r.status_path)
        if not data or data.get("exit"):
            return None
        ts = data.get("ts")
        if not isinstance(ts, (int, float)):
            return None
        now = self.clock()
        if ts < r.reset_at or now - ts > self.status_max_age_s:
            return None
        return data

    # ------------------------------------------------------------------ residency

    def resident(self, server: str) -> frozenset[str]:
        """Tables loaded by ``server`` in its current generation."""
        return frozenset(self._get(server).tables)

    def pending(self, server: str) -> frozenset[str]:
        """Tables of admitted cold loads not yet committed."""
        self._expire(server)
        out: set[str] = set()
        for tables, _ in self._get(server).reserved.values():
            out.update(tables)
        return frozenset(out)

    def baseline_mb(self, server: str) -> float:
        r = self._get(server)
        return r.baseline_mb if r.baseline_mb is not None else self.default_baseline_mb

    def estimated_mb(self, server: str) -> float:
        """``baseline + sum(resident estimates)`` (no status file, no reservations)."""
        return self.baseline_mb(server) + sum(self._get(server).tables.values())

    def reserved_mb(self, server: str) -> float:
        self._expire(server)
        seen: dict[str, float] = {}
        for tables, _ in self._get(server).reserved.values():
            for t, mb in tables.items():
                seen[t] = max(seen.get(t, 0.0), mb)
        return sum(mb for t, mb in seen.items() if t not in self._get(server).tables)

    def resident_mb(self, server: str) -> float:
        """Memory ``server`` holds now: the reaper's RSS when fresh, else the estimate, plus
        reservations of admitted cold loads."""
        r = self._get(server)
        status = self.status(server)
        rss = status.get("rss_mb") if status else None
        if isinstance(rss, (int, float)):
            if not r.tables and not r.reserved and r.baseline_mb is None:
                r.baseline_mb = float(rss)       # first reading of a fresh generation: its baseline
            base = float(rss)
        else:
            base = self.estimated_mb(server)
        return base + self.reserved_mb(server)

    def limit_mb(self, server: str) -> float | None:
        """The limit the reaper enforces for ``server`` (from its status file), if known."""
        data = read_status(self._get(server).status_path)
        value = data.get("limit_mb") if data else None
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    def reserve(self, server: str, tables: Mapping[str, float]) -> int:
        """Reserve memory for an admitted cold load; returns the reservation id."""
        self._next_id += 1
        self._get(server).reserved[self._next_id] = (dict(tables), self.clock())
        return self._next_id

    def release(self, server: str, reservation: int | None) -> None:
        if reservation is not None:
            self._get(server).reserved.pop(reservation, None)

    def add(self, server: str, tables: Mapping[str, float], generation: int | None = None) -> None:
        """Record ``tables`` as loaded by ``server`` (ignored for a generation that has passed)."""
        r = self._get(server)
        if generation is not None and r.generation is not None and generation != r.generation:
            return
        now = self.clock()
        for t, mb in tables.items():
            r.tables[t] = float(mb)
            r.loaded_at[t] = now

    def _expire(self, server: str) -> None:
        r = self._get(server)
        cutoff = self.clock() - self.reservation_ttl_s
        for rid in [rid for rid, (_, at) in r.reserved.items() if at < cutoff]:
            r.reserved.pop(rid, None)

    def snapshot(self, server: str | None = None) -> dict[str, Any]:
        """Per server: generation, resident tables (MB), reservations, resident and baseline MB."""
        names = [server] if server else sorted(self._servers)
        out: dict[str, Any] = {}
        for name in names:
            r = self._get(name)
            out[name] = {"generation": r.generation, "tables": dict(sorted(r.tables.items())),
                         "pending": sorted(self.pending(name)), "resident_mb": round(self.resident_mb(name), 1),
                         "baseline_mb": round(self.baseline_mb(name), 1)}
        return out

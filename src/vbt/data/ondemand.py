"""On-demand acquisition: what a ``not_ready`` refusal leads to under ``data.acquisition.auto``. No pyarrow.

A call refused ``not_ready`` because a table's files are absent (or partial, or of another release) carries how to
acquire them (:func:`vbt.datalayer.gateway.readiness.acquisition_hint`: the command, size, files, licence and
login notes). What happens next is the operator's policy, ``data.acquisition``::

    auto: off            # default: the refusal says how; a person runs `vbt data acquire`
    auto: ask            # the tables are queued in <cache_dir>/acquisition/pending.json; `vbt data acquire
                         # --pending` (a person's approval) acquires them
    auto: under_budget   # between turns the system acquires them itself when the whole acquisition (declared
    budget_bytes: 5 GB   # sizes, prepare inputs included) fits the budget; larger ones are queued as with `ask`

:func:`between_turns` is that step: the harness calls it after a turn with the turn's refusals (or ``None``, to take
every table the readiness cache reports missing). Every acquisition it makes is recorded in
``<data.provenance.dir>/acquisitions.jsonl`` and, with ``run_dir``, in the run's ``data_acquisitions.jsonl``
(``by: auto``, the policy and the budget).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .acquire import (
    AcquisitionSettings,
    execute,
    fmt_bytes,
    plan_acquisition,
    record_provenance,
)
from .manifest import atomic_write, load_manifest

__all__ = ["ACQUIRE_STATUSES", "refused_tables", "missing_tables", "between_turns", "pending", "queue",
           "clear_pending"]

#: Table statuses that acquiring the table's files can fix.
ACQUIRE_STATUSES = frozenset({"missing", "partial", "stale"})


def _pending_path(settings: AcquisitionSettings) -> Path | None:
    return Path(settings.cache_dir) / "acquisition" / "pending.json" if settings.cache_dir else None


def pending(settings: AcquisitionSettings) -> dict[str, list[str]]:
    """``{source: [tables]}`` queued for an operator's approval."""
    path = _pending_path(settings)
    data = load_manifest(path) if path is not None else {}
    want = data.get("wanted") if isinstance(data.get("wanted"), dict) else {}
    return {str(k): [str(t) for t in v] for k, v in want.items() if isinstance(v, list)}


def queue(settings: AcquisitionSettings, wanted: Mapping[str, Sequence[str]], *, reason: str = "") -> None:
    path = _pending_path(settings)
    if path is None:
        return
    cur = pending(settings)
    for s, ts in wanted.items():
        names = cur.setdefault(s, [])
        names.extend(t for t in ts if t not in names)
    atomic_write(path, json.dumps({"wanted": cur, "reason": reason,
                                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1).encode())


def clear_pending(settings: AcquisitionSettings, wanted: Mapping[str, Sequence[str]]) -> None:
    path = _pending_path(settings)
    if path is None or not path.is_file():
        return
    cur = pending(settings)
    for s, ts in wanted.items():
        cur[s] = [t for t in cur.get(s, []) if t not in ts]
    cur = {s: ts for s, ts in cur.items() if ts}
    atomic_write(path, json.dumps({"wanted": cur, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                  indent=1).encode())


def refused_tables(catalog: Any, refusals: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """``{source: [tables]}`` named by ``not_ready`` payload entries (``{"tables": [{name, check, ...}]}`` or the
    entries themselves) whose table has an acquisition entry."""
    out: dict[str, list[str]] = {}
    for r in refusals:
        entries = r.get("tables") if isinstance(r.get("tables"), list) else [r]
        for e in entries:
            if not isinstance(e, Mapping):
                continue
            acq = e.get("acquire") if isinstance(e.get("acquire"), Mapping) else None
            ref = str((acq or {}).get("table") or e.get("name") or "")
            if "." not in ref:
                continue
            _add(catalog, out, ref)
    return out


def _add(catalog: Any, out: dict[str, list[str]], ref: str) -> None:
    try:
        t = catalog.table(ref)
    except Exception:  # noqa: BLE001
        return
    phys = str(t.physical)
    source, _, name = phys.partition(".")
    acq = t.descriptor.acquisition
    if acq is None or acq.mode != "download" or name not in acq.tables:
        return
    names = out.setdefault(source, [])
    if name not in names:
        names.append(name)


def missing_tables(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """``{source: [tables]}`` the readiness cache reports missing, partial or stale and that can be acquired."""
    from ..datalayer.gateway.readiness import ReadinessCache, table_status
    from ..preflight import data_catalog

    settings, catalog, registry = data_catalog(dict(config))
    cache = ReadinessCache(settings.cache_dir, catalog, registry)
    cache.load()
    out: dict[str, list[str]] = {}
    for ref, m in sorted(cache.tables.items()):
        if table_status(m) in ACQUIRE_STATUSES:
            _add(catalog, out, ref)
    return out


def between_turns(config: Mapping[str, Any], *, refusals: Sequence[Mapping[str, Any]] | None = None,
                  run_dir: Path | None = None, catalog: Any = None, session: Any = None,
                  settings: AcquisitionSettings | None = None, root: Path | None = None) -> dict[str, Any]:
    """Apply ``data.acquisition.auto`` to the tables ``refusals`` name (``None``: every table the readiness cache
    reports missing). Returns ``{policy, wanted, bytes, decision, report?}``; never raises for a failed transfer
    (the report says it)."""
    settings = settings or AcquisitionSettings.from_config(config)
    if catalog is None:
        from ..preflight import data_catalog

        catalog = data_catalog(dict(config))[1]
    wanted = refused_tables(catalog, refusals) if refusals is not None else missing_tables(config)
    out: dict[str, Any] = {"policy": settings.auto, "budget_bytes": settings.budget_bytes, "wanted": wanted}
    if not wanted:
        out["decision"] = "nothing to acquire"
        return out
    if settings.auto == "off":
        out["decision"] = "off: the refusal says how to acquire it"
        return out
    offline = plan_acquisition(catalog, wanted, settings, root=root, offline=True)
    need = offline.bytes_remaining
    out["bytes"] = need
    if settings.auto == "ask" or need is None or need > settings.budget_bytes:
        queue(settings, wanted, reason=f"policy {settings.auto}, {fmt_bytes(need)}")
        out["decision"] = ("queued for approval (vbt data acquire --pending)" if settings.auto == "ask" else
                           f"queued: {fmt_bytes(need)} is over the budget of {fmt_bytes(settings.budget_bytes)}")
        return out
    plan = plan_acquisition(catalog, wanted, settings, root=root, session=session)
    rem = plan.bytes_remaining
    if rem is not None and rem > settings.budget_bytes:
        queue(settings, wanted, reason=f"listed size {fmt_bytes(rem)} over the budget")
        out["decision"] = f"queued: the listing says {fmt_bytes(rem)}, over the budget"
        out["bytes"] = rem
        return out
    try:
        report = execute(plan, settings, session=session, max_bytes=settings.budget_bytes, config=config)
    except ValueError as exc:
        queue(settings, wanted, reason=str(exc))
        out["decision"] = f"queued: {exc}"
        return out
    out["decision"] = "acquired" if report.ok else "acquisition failed"
    out["report"] = report
    out["provenance"] = record_provenance(settings, report, by="auto", run_dir=run_dir,
                                          extra={"wanted": wanted, "trigger": "refusals" if refusals is not None
                                                 else "readiness cache"})
    if report.ok:
        clear_pending(settings, wanted)
    return out

"""Readiness scoped to each call (§10.3, §13). No pyarrow.

The data child's ``_check`` computes per-table results with per-column, per-container,
per-partition and per-item-table statuses (R1-R10). :class:`ReadinessCache` keeps them in the
harness, persisted in ``${data.cache_dir}/readiness/<source>.json`` keyed by ``(descriptor
sha256, table signature)``: a cached result is valid while the descriptor is unchanged and the
table's stat-only layout signature (computed here with the layout plugin, stdlib only) still
matches. :meth:`ReadinessCache.shallow_refresh` is the per-turn check: it recomputes the
signatures of the tables one call reads and drops results whose signature moved.

:func:`call_readiness` decides one call from exactly the parts it reads: its ``reads.columns``,
the columns its arguments and result fields bind, its order columns, the bound table's key, the
containers it returns, the partitions its partition predicate can include, the tables of coverage
universes, and the item tables it serves. One drifted field no longer takes down every tool on
the table, and a partial partition blocks only calls that can include it. Resolver indexes are
decided per accepted kind by the gateway (a kind without its index is dropped with a note).

A reason whose status acquiring the files fixes (``missing``, ``partial``, ``stale``) on a table whose descriptor
declares an ``acquisition`` entry also says how (:func:`acquisition_hint`): ``acquire`` holds the command
(``vbt data acquire <source>.<table>``), the declared bytes and files, the prepare steps, the licence and login
notes and, when the cache knows the ``data.acquisition`` policy (``ReadinessCache.acquisition``), what the policy
decides (:func:`auto_decision`); the reason's ``hint`` says the same in words.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..ipc import CheckResponse, TableCheckModel
from ..plugins.base import LayoutSpec

__all__ = ["READY_STATUSES", "TABLE_LEVEL_STATUSES", "CallReadiness", "ReadinessCache", "call_readiness",
           "section_tables", "layout_spec", "table_signature", "tables_read", "columns_read", "degraded_tools", "norm_path",
           "parse_partition_label", "partitions_selected", "table_status", "ACQUIRE_STATUSES", "acquisition_hint",
           "auto_decision"]

READY_STATUSES = frozenset({"ready", "awaiting_producer", "unbound"})
#: Statuses that make the whole table unservable whatever part a call reads.
TABLE_LEVEL_STATUSES = frozenset({"missing", "stale", "unreachable", "plugin_unavailable"})
_HINTS = {
    "missing": "the table's files are absent; download or prepare them, then rerun `vbt ds check`",
    "partial": "a partial download or unreadable fragment; complete the download, then rerun `vbt ds check`",
    "schema_drift": "the columns differ from the descriptor; update the descriptor or the data",
    "encoding_drift": "stored codes differ from the descriptor's encoding",
    "key_violation": "the declared key is not unique or has nulls",
    "stale": "the release differs from the one the descriptor expects",
    "unreachable": "the remote source did not answer",
    "plugin_unavailable": "a plugin the table needs is not installed",
}


#: Statuses that acquiring a table's files fixes (absent, incomplete, or of another release).
ACQUIRE_STATUSES = frozenset({"missing", "partial", "stale"})


def _fmt_bytes(n: float | None) -> str:
    if n is None:
        return "size unknown"
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{int(n)} B"


def _budget(value: Any) -> int:
    """``budget_bytes`` as bytes (an int, or ``"5 GB"``, ``"500MiB"``)."""
    if value is None or value == "" or isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().upper().replace(" ", "")
    for unit, mult in (("TIB", 1 << 40), ("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10), ("TB", 10**12),
                       ("GB", 10**9), ("MB", 10**6), ("KB", 10**3), ("B", 1)):
        if text.endswith(unit):
            try:
                return int(float(text[: -len(unit)]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def auto_decision(policy: Mapping[str, Any] | None, nbytes: int | None) -> tuple[str, str]:
    """``(decision, sentence)`` of ``data.acquisition`` (``auto: off | ask | under_budget``, ``budget_bytes``) for an
    acquisition of ``nbytes``: ``off``, ``ask`` (queued for an operator's approval), ``auto`` (the system acquires
    it between turns) or ``over_budget``."""
    auto = (policy or {}).get("auto", "off")
    auto = {False: "off", True: "under_budget"}.get(auto, auto) if isinstance(auto, bool) else str(auto)
    if auto == "ask":
        return "ask", "data.acquisition.auto is ask: ask the operator to approve it (`vbt data acquire --pending`)"
    if auto == "under_budget":
        budget = _budget((policy or {}).get("budget_bytes", (policy or {}).get("budget")))
        if nbytes is not None and nbytes <= budget:
            return "auto", (f"data.acquisition.auto is under_budget ({_fmt_bytes(budget)}): the system acquires it "
                            "between turns; retry the call in the next turn")
        return "over_budget", (f"over the data.acquisition budget ({_fmt_bytes(budget)}): an operator must run it")
    return "off", "an operator runs it (data.acquisition.auto is off)"


def acquisition_hint(catalog: Any, ref: str, policy: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """How to acquire the files behind ``ref`` (an item table: its parent's), from its descriptor's
    ``acquisition`` section; None when the source is read live or declares no acquisition for the table."""
    try:
        t = catalog.table(ref)
    except Exception:  # noqa: BLE001
        return None
    desc = t.descriptor
    acq = getattr(desc, "acquisition", None)
    source, _, name = str(t.physical).partition(".")
    if acq is None or acq.mode == "remote" or name not in acq.tables:
        return None
    groups: list[str] = []
    steps: list[str] = []
    entry = acq.tables[name]
    if entry.files:
        groups.append(name)
    if entry.prepared_by:
        steps.append(entry.prepared_by)
        groups.extend(g for g in acq.prepare[entry.prepared_by].needs if g not in groups)
    entries = [acq.tables.get(g) or acq.extra.get(g) for g in groups]
    nbytes = sum(int(e.bytes) for e in entries if e is not None and e.bytes is not None) \
        if all(e is not None and e.bytes is not None for e in entries) else None
    files = sum(int(e.count) for e in entries if e is not None and e.count is not None) \
        if all(e is not None and e.count is not None for e in entries) else None
    release = str(acq.release or desc.release.expect or "current")
    out: dict[str, Any] = {"command": f"vbt data acquire {source}.{name}", "source": source, "table": f"{source}.{name}",
                           "release": release, "bytes": nbytes, "files": files, "prepare": steps,
                           "mode": acq.mode}
    if acq.licence:
        out["licence"] = acq.licence
    if acq.login:
        out["login"] = acq.login
    if acq.mode == "manual":
        out["command"] = f"see the descriptor's acquisition.login ({source})"
    if policy is not None:
        out["policy"], out["decision"] = auto_decision(policy, nbytes)
    return out


def _acquire_text(h: Mapping[str, Any]) -> str:
    size = _fmt_bytes(h.get("bytes")) + (f" in {h['files']} file(s)" if h.get("files") is not None else "")
    text = f"acquire them with `{h['command']}` ({size}, release {h['release']}"
    if h.get("prepare"):
        text += f", then the prepare step {', '.join(h['prepare'])}"
    text += ")"
    if h.get("licence"):
        text += f"; licence: {h['licence']}"
    if h.get("login"):
        text += f"; login: {h['login']}"
    if h.get("decision"):
        text += f"; {h['decision']}"
    return text + "; then rerun `vbt ds check`"


def _with_acquisition(reason: dict[str, Any], cache: Any, ref: str, status: str | None) -> dict[str, Any]:
    if status not in ACQUIRE_STATUSES:
        return reason
    h = acquisition_hint(getattr(cache, "catalog", None), ref, getattr(cache, "acquisition", None))
    if h is None:
        return reason
    reason["acquire"] = h
    base = {"missing": "the table's files are absent", "partial": "the table's files are incomplete",
            "stale": "the files are of another release"}[status]
    reason["hint"] = f"{base}; {_acquire_text(h)}"
    return reason


def norm_path(path: str) -> str:
    """A column path without list brackets and leading ``/`` (``go[].id`` -> ``go.id``)."""
    text = str(path).lstrip("/")
    while "[" in text:
        start = text.index("[")
        end = text.find("]", start)
        if end < 0:
            break
        text = text[:start] + text[end + 1:]
    return text.strip(".")


def _overlaps(a: str, b: str) -> bool:
    a, b = norm_path(a), norm_path(b)
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def _within(path: str, column: str) -> bool:
    """``path`` is ``column`` or a field under it."""
    p, c = norm_path(path), norm_path(column)
    return p == c or p.startswith(c + ".")


def parse_partition_label(label: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(label).split("/"):
        k, sep, v = part.partition("=")
        if sep:
            out[k] = v
    return out


def partitions_selected(scope: Mapping[str, Any] | None, labels: Iterable[str]) -> list[str] | None:
    """The partitions (``col=value[/col=value]``) a call reads, from the scope it fixed: the labels whose
    values agree with every fixed scope dimension that is a partition column. None when the scope fixes
    no partition column (the call reads every partition)."""
    names = list(labels)
    cols = {k for n in names for k in parse_partition_label(n)}
    fixed = {str(k).split(".")[-1]: v for k, v in (scope or {}).items() if str(k).split(".")[-1] in cols}
    if not fixed:
        return None

    def agrees(name: str) -> bool:
        parts = parse_partition_label(name)
        for col, value in fixed.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            if col in parts and parts[col] not in {str(v) for v in values}:
                return False
        return True

    return sorted(n for n in names if agrees(n))


def layout_spec(table: Any) -> LayoutSpec:
    """The layout plugin's view of a catalog table (its physical table for item tables)."""
    desc = table.descriptor
    spec = table.physical_spec
    ref = spec.layout if spec.layout is not None else desc.defaults.layout
    options = dict(getattr(ref, "options", {}) or {})
    partitions = {name: p.type for name, p in spec.partitions.items()}
    expect: dict[str, Any] = {}
    for name, p in spec.partitions.items():
        vocab = getattr(p.column, "vocab", None)
        expect[name] = list(vocab) if p.expect == "declared" and isinstance(vocab, list) else None
    fk = spec.fragment_key.model_dump(by_alias=True) if spec.fragment_key is not None else None
    return LayoutSpec(table=str(table.physical), path=spec.path, options=options, partitions=partitions,
                      partition_expect=expect, fragment_key=fk, format=table.format)


def table_signature(table: Any, registry: Any) -> str | None:
    """The stat-only layout signature of ``table`` (None when it cannot be computed here)."""
    name = table.layout
    if not name or registry is None or table.descriptor.kind == "remote":
        return None
    plugin = registry.find("layout", name)
    if plugin is None or not hasattr(plugin, "signature"):
        return None
    root = table.descriptor.root
    if not root:
        return None
    try:
        return str(plugin.signature(os.path.expanduser(str(root)), layout_spec(table)))
    except Exception:  # noqa: BLE001 - an unreadable location is decided by _check
        return None


@dataclass
class CallReadiness:
    ready: bool = True
    reasons: list[dict[str, Any]] = field(default_factory=list)   # not_ready payload entries
    unchecked: list[str] = field(default_factory=list)            # tables without a cached result
    notes: list[str] = field(default_factory=list)
    unavailable_partitions: dict[str, list[str]] = field(default_factory=dict)
    soft: list[dict[str, Any]] = field(default_factory=list)      # the reasons on section-only tables

    @property
    def hard(self) -> list[dict[str, Any]]:
        """The reasons that fail the call (an unready section table only marks its section unavailable)."""
        return [x for x in self.reasons if x not in self.soft]


def section_tables(contract: Any) -> set[str]:
    """Tables read only for result or derived sections: an unready one marks its section
    unavailable (``partial``) instead of failing the call."""
    b = getattr(contract, "binding", None)
    out: set[str] = set()
    if b is None:
        return out
    main = {b.result.rows_of, contract.bound_table, b.derived.table if b.derived is not None else None}
    for s in b.result.sections.values():
        out.add(s.table)
    if b.derived is not None:
        for s in b.derived.sections.values():
            out.add(s.table)
    return {t for t in out if t and t not in main}


def _strip_table(column: str, table: str) -> str:
    return column[len(table) + 1:] if column.startswith(table + ".") else column


def tables_read(contract: Any, bound_table: str | None, args: Mapping[str, Any] | None = None) -> list[str]:
    """The tables (and item tables) one call reads: ``reads`` (``when`` honoured; an ``upstream`` read of
    a derived tool left out), the bound table after selector resolution, the derived table and its
    sections, ``rows_of``, result sections and coverage-universe tables. Tables only reachable through
    an unselected selector value are left out."""
    b = getattr(contract, "binding", None)
    if b is None:
        return []
    args = dict(args or {})
    alternatives: set[str] = set()
    for name in getattr(contract, "selector_args", []):
        a = contract.args[name]
        targets = [t for t in a.values.values() if isinstance(t, str)]
        if isinstance(a.binds, dict):
            targets.extend(a.binds.values())
        alternatives.update(".".join(str(t).split(".")[:2]) for t in targets)
    out: list[str] = []

    def add(ref: str | None) -> None:
        if ref and ref not in out and (ref not in alternatives or ref == bound_table):
            out.append(ref)

    for ref, rs in b.reads.items():
        if rs.when and any(args.get(k) != v for k, v in rs.when.items()):
            continue
        if rs.access == "upstream" and b.serve == "derived":
            continue                                   # only the upstream tool reads it; a derived call never does
        add(ref)
    add(bound_table)
    if b.derived is not None:
        add(b.derived.table)
        for s in b.derived.sections.values():
            add(s.table)
    add(b.result.rows_of)
    for s in b.result.sections.values():
        add(s.table)
    for ref in list(out):
        t = contract.tables.get(ref)
        cov = getattr(getattr(t, "spec", None), "coverage", None) if t is not None else None
        if cov is not None and cov.universe is not None:
            u = cov.universe.table
            add(u if "." in u else f"{t.ref.source}.{u}")
    return out


def columns_read(contract: Any, table: str) -> list[str]:
    """Column paths (relative to ``table``) the call reads from it."""
    b = getattr(contract, "binding", None)
    if b is None:
        return []
    cols: list[str] = []
    rs = b.reads.get(table)
    if rs is not None:
        cols.extend(_strip_table(c, table) for c in rs.columns)
    for name in contract.args:
        for t, c in contract.arg_columns(name):
            if t == table:
                cols.append(c)
    t = contract.tables.get(table)
    rows_table = b.result.rows_of or (b.derived.table if b.derived is not None else None) or contract.bound_table
    if rows_table == table:
        for fm in b.result.fields.values():
            if fm.column:
                cols.append(fm.column.lstrip("/"))
            elif fm.computed and fm.computed.get("from"):
                cols.append(str(fm.computed["from"]))
        cols.extend(o.column for o in b.result.order)
        if b.derived is not None:
            cols.extend(b.derived.columns)
    if t is not None:
        cols.extend(c for c in t.key if not c.endswith("#"))
    seen: dict[str, None] = {}
    for c in cols:
        seen.setdefault(c, None)
    return list(seen)


class ReadinessCache:
    """Per-table ``_check`` results with their signatures (see the module docstring)."""

    def __init__(self, cache_dir: str | Path | None, catalog: Any, registry: Any = None, *,
                 refresh_interval_s: float = 1.0, clock: Any = time.monotonic,
                 acquisition: Mapping[str, Any] | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.catalog = catalog
        self.registry = registry
        #: the ``data.acquisition`` policy (``auto``, ``budget_bytes``); None when the caller does not know it, and
        #: then a not_ready reason says how to acquire a table without saying what the policy decides
        self.acquisition: Mapping[str, Any] | None = acquisition
        self.refresh_interval_s = float(refresh_interval_s)
        self.clock = clock
        self.tables: dict[str, TableCheckModel] = {}
        self.signatures: dict[str, str | None] = {}
        self.indexes: dict[str, dict[str, Any]] = {}   # "source:id_type" -> {status, fingerprint, detail}
        self.hash_randomization: int | None = None
        self.checked_at: dict[str, float] = {}
        self._refreshed: dict[str, float] = {}

    # -- persistence -------------------------------------------------------

    def _digest(self, source: str) -> str | None:
        try:
            return self.catalog.descriptor_digest(source)
        except Exception:  # noqa: BLE001
            return None

    def _file(self, source: str) -> Path | None:
        return self.cache_dir / "readiness" / f"{source}.json" if self.cache_dir else None

    def load(self) -> int:
        """Load persisted results whose descriptor digest and signature still match; returns how many."""
        n = 0
        for source in getattr(self.catalog, "sources", {}):
            path = self._file(source)
            if path is None or not path.is_file():
                continue
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if data.get("descriptor_sha256") != self._digest(source):
                continue
            for ref, entry in (data.get("tables") or {}).items():
                try:
                    model = TableCheckModel.model_validate(entry.get("check") or {})
                except Exception:  # noqa: BLE001 - a stale format is re-checked
                    continue
                sig = entry.get("signature")
                current = self._current_signature(ref)
                if sig is not None and current is not None and sig != current:
                    continue
                self.tables[ref] = model
                self.signatures[ref] = sig
                n += 1
        return n

    def save(self, source: str) -> None:
        path = self._file(source)
        if path is None:
            return
        body = {"descriptor_sha256": self._digest(source), "tables": {
            ref: {"signature": self.signatures.get(ref), "check": m.model_dump(mode="json")}
            for ref, m in sorted(self.tables.items()) if ref.split(".")[0] == source}}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
            os.replace(tmp, path)
        except OSError:
            pass

    # -- results -------------------------------------------------------------

    def _catalog_table(self, ref: str) -> Any:
        try:
            return self.catalog.table(ref)
        except Exception:  # noqa: BLE001
            return None

    def _current_signature(self, ref: str) -> str | None:
        t = self._catalog_table(ref)
        return table_signature(t, self.registry) if t is not None else None

    def load_check_results(self, response: CheckResponse | Mapping[str, Any], *, save: bool = True) -> list[str]:
        """Store a ``_check`` response; returns the table names stored."""
        if not isinstance(response, CheckResponse):
            response = CheckResponse.model_validate(response)
        if response.hash_randomization is not None:
            self.hash_randomization = response.hash_randomization
        stored = []
        sources = set()
        for ref, model in response.tables.items():
            self.tables[ref] = model
            self.signatures[ref] = model.signature or self._current_signature(ref)
            self.checked_at[ref] = self.clock()
            self._refreshed[ref] = self.clock()
            stored.append(ref)
            sources.add(ref.split(".")[0])
        if save:
            for s in sorted(sources):
                self.save(s)
        return stored

    def get(self, ref: str) -> TableCheckModel | None:
        return self.tables.get(ref)

    def physical(self, ref: str) -> tuple[str, str | None]:
        """``(physical table, item table name or None)`` under which ``ref``'s result is stored."""
        t = self._catalog_table(ref)
        if t is None or not t.is_item_table:
            return ref, None
        return str(t.physical), ref.split(".", 1)[1]

    def shallow_refresh(self, tables: Iterable[str]) -> list[str]:
        """Recompute the signatures of ``tables`` (at most once per ``refresh_interval_s``); results
        whose signature changed are dropped. Returns the dropped table names."""
        stale = []
        now = self.clock()
        for ref in tables:
            phys, _ = self.physical(ref)
            if phys not in self.tables:
                continue
            if now - self._refreshed.get(phys, -1e18) < self.refresh_interval_s:
                continue
            self._refreshed[phys] = now
            current = self._current_signature(phys)
            old = self.signatures.get(phys)
            if current is not None and old is not None and current != old:
                self.tables.pop(phys, None)
                stale.append(phys)
        return stale

    def set_index(self, qualified: str, status: str, *, fingerprint: str | None = None, detail: str = "") -> None:
        self.indexes[qualified] = {"status": status, "fingerprint": fingerprint, "detail": detail}

    def index_ready(self, qualified: str) -> bool | None:
        entry = self.indexes.get(qualified)
        return None if entry is None else entry.get("status") == "ready"

    def snapshot(self) -> dict[str, Any]:
        """A JSON-able view: ``{tables: {ref: {table, status (:func:`table_status`), worst_status, columns,
        partitions, containers, item_tables, failed_checks}}, indexes: {qualified id_type: {name (bare), id_type, status, ...}}}``."""
        tables = {}
        for ref, m in sorted(self.tables.items()):
            failed = [c.model_dump(mode="json") for c in m.checks if not c.ok]
            tables[ref] = {"table": ref, "status": table_status(m), "worst_status": m.status,
                           "fingerprint": m.fingerprint,
                           "columns": {k: v for k, v in m.columns.items() if v not in READY_STATUSES},
                           "containers": {k: v for k, v in m.containers.items() if v not in READY_STATUSES},
                           "partitions": {k: v for k, v in m.partitions.items() if v not in READY_STATUSES},
                           "item_tables": dict(m.item_tables), "failed_checks": failed}
        indexes = {k: {"name": k.partition(":")[2] or k, "id_type": k, **v} for k, v in sorted(self.indexes.items())}
        return {"tables": tables, "indexes": indexes, "hash_randomization": self.hash_randomization}


def table_status(m: TableCheckModel) -> str:
    """The table's own status: ``m.status`` when a table-level check failed (or the status is one
    only a whole table has), else ready. Findings scoped to a column, container or partition block
    only the calls that read that part (see :func:`call_readiness`), so they do not make the table
    unready; ``m.status`` stays the worst status over every part."""
    if m.status in READY_STATUSES or m.status in TABLE_LEVEL_STATUSES:
        return m.status
    if any(c.column is None and c.partition is None for c in _failed_checks(m)):
        return m.status
    return "ready"


def _failed_checks(m: TableCheckModel) -> list[Any]:
    return [c for c in m.checks if not c.ok and c.level == "error"]


def _reason(name: str, check: str, detail: str, *, column: str | None = None, partition: str | None = None,
            hint: str | None = None, status: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "check": check, "detail": detail,
                           "hint": hint or _HINTS.get(status or "", "run `vbt ds check` for details")}
    if column is not None:
        out["column"] = column
    if partition is not None:
        out["partition"] = partition
    return out


def _partition_selected(label: str, partition_values: Mapping[str, Sequence[Any]] | None) -> bool:
    """Can a call with these fixed partition values read the partition ``label``?"""
    if not partition_values:
        return True
    parts = parse_partition_label(label)
    for col, values in partition_values.items():
        if col in parts and str(parts[col]) not in {str(v) for v in values}:
            return False
    return True


def call_readiness(contract: Any, cache: ReadinessCache, *, bound_table: str | None,
                   args: Mapping[str, Any] | None = None,
                   partition_values: Mapping[str, Mapping[str, Sequence[Any]]] | None = None) -> CallReadiness:
    """Readiness of one call over exactly the parts it reads (see the module docstring).
    ``partition_values`` is ``{table: {partition column: values fixed by the call}}``."""
    out = CallReadiness()
    for ref in tables_read(contract, bound_table, args):
        phys, item = cache.physical(ref)
        m = cache.get(phys)
        if m is None:
            out.unchecked.append(phys)
            continue
        status = m.status
        if item is not None and m.item_tables.get(item) not in (None, *READY_STATUSES):
            out.reasons.append(_with_acquisition(
                _reason(ref, "item_table", f"item table {item} is {m.item_tables[item]}", status=m.item_tables[item]),
                cache, ref, m.item_tables[item]))
            continue
        failed = _failed_checks(m)
        table_level = [c for c in failed if c.column is None and c.partition is None]
        if status in TABLE_LEVEL_STATUSES or table_level:
            c = table_level[0] if table_level else None
            out.reasons.append(_with_acquisition(
                _reason(phys, c.name if c else status, (c.detail if c else "") or f"table is {status}",
                        hint=(c.hint or None) if c else None, status=status), cache, phys, status))
            continue
        wanted = columns_read(contract, ref)
        if item is not None:
            t = cache._catalog_table(ref)
            prefix = norm_path(t.items_path or "") if t is not None else ""
            wanted = [f"{prefix}.{c}" if prefix else c for c in wanted]
        # a refuted or unconfirmed descriptor fact (R6) disables that fact, not the column: reading stays
        # possible and filters that need the fact are refused by the contracts (unsupported_filter, I9)
        facts_only = {c.column for c in failed if c.column} - {c.column for c in failed if c.column and c.name != "R6"}
        bad_cols = {k: v for k, v in {**m.columns, **m.containers}.items()
                    if v not in READY_STATUSES and k not in facts_only}
        for c in failed:
            if c.column is not None and c.partition is None and c.column not in facts_only:
                bad_cols.setdefault(c.column, status if status not in READY_STATUSES else "schema_drift")
        # an optional column the data lacks (R4 warning, status missing) blocks the calls that read it or a field
        # under it, never the container that would hold it (25.09 tissues.protein.cell_type, absent elsewhere)
        absent = {k for k, v in m.columns.items() if v == "missing"} - {c.column for c in failed if c.column}
        hit = False
        for col, st in sorted(bad_cols.items()):
            if any(_within(w, col) if col in absent else _overlaps(col, w) for w in wanted):
                detail = next((c.detail for c in failed if c.column == col), "") or f"column {col} is {st}"
                check = next((c.name for c in failed if c.column == col), st)
                out.reasons.append(_reason(phys, check, detail, column=col, status=st))
                hit = True
                break
        if hit:
            continue
        fixed = (partition_values or {}).get(ref) or (partition_values or {}).get(phys)
        bad_parts = {k: v for k, v in m.partitions.items() if v not in READY_STATUSES}
        for c in failed:
            if c.partition is not None:
                bad_parts.setdefault(c.partition, "partial")
        blocking = sorted(lbl for lbl in bad_parts if _partition_selected(lbl, fixed))
        if blocking:
            lbl = blocking[0]
            detail = next((c.detail for c in failed if c.partition == lbl), "") or f"partition {lbl} is {bad_parts[lbl]}"
            out.reasons.append(_with_acquisition(
                _reason(phys, next((c.name for c in failed if c.partition == lbl), bad_parts[lbl]), detail,
                        partition=lbl, status=bad_parts[lbl]), cache, phys, bad_parts[lbl]))
            out.unavailable_partitions[phys] = blocking
            continue
        own = table_status(m)                          # findings scoped to parts this call skips do not count
        if own not in READY_STATUSES and not bad_cols and not bad_parts and own != "partial":
            out.reasons.append(_with_acquisition(_reason(phys, own, f"table is {own}", status=own), cache, phys, own))
    sections = section_tables(contract)
    if sections:
        tails = {t.split(".")[-1] for t in sections}
        out.soft = [x for x in out.reasons if x["name"] in sections or x["name"].split(".")[-1] in tails]
    out.ready = not out.reasons
    return out


def degraded_tools(catalog: Any, cache: ReadinessCache) -> dict[str, str]:
    """``{mcp__server__tool: reason}`` for reviewed tools that every call would find unready (a
    table or column they always read failed; a failed partition may be excluded by arguments, and an
    unready section table only marks that section unavailable)."""
    out: dict[str, str] = {}
    for server in catalog.servers():
        for tool in catalog.tools(server):
            try:
                contract = catalog.contract(server, tool)
            except Exception:  # noqa: BLE001
                continue
            if contract.binding is None or contract.binding.serve == "block":
                continue
            r = call_readiness(contract, cache, bound_table=contract.bound_table)
            always = [x for x in r.hard if "partition" not in x]      # a partition may be excluded by arguments
            if always:
                x = always[0]
                out[f"mcp__{server}__{tool}"] = f"{x['name']} not ready: {x['check']} ({x['detail']})"[:300]
    return out

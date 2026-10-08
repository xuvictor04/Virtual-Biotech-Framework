"""The bounded reader (§11.6, §14.5): two-pass scans with exact totals and global top-k. pyarrow inside methods.

One :class:`TableReader` serves a table or an item table (§6.2). It never names a plugin or a
dataset: fragments come from the table's layout plugin (listed by name, hence an unreadable listed
fragment is an error naming it and its partition, never a smaller table), values from its format
plugin, ordering from statistic plugins.

**Pruning.** Before reading, fragments are pruned on partition-only conjuncts (three-valued: a null
partition never satisfies a filter); conjuncts on a column that mirrors a partition
(``mirrored_by``) are rewritten to the partition for this. Row groups are pruned with footer
min/max statistics and, for columns with an ``access_paths`` entry ``via: sidecar_index``, with
the row-group value index of ``service/sidecar.py`` (built on first use when missing).
:meth:`TableReader.estimate_scan_bytes` sums the decoded bytes of the leaves that remain; over
the budget the reader raises :class:`BudgetExceeded` before reading anything.

**Two passes.** Pass 1 projects only the key and filter leaves of each remaining row group
(``read_leaves``, which reaches leaves inside lists), evaluates the predicate with the
three-valued oracle (:func:`~vbt.datalayer.predicate.evaluate`: ``Any``/``All``/``CmpAbs``/
``CensoredCmp``/``NonEmpty``/``KindMatch``; NaN, ``missing_values``, placeholders and
``unknown_when`` read as null) and keeps the matching row indices. Pass 2 reads the order and
output leaves of the row groups that matched and takes the matching rows. Partition columns are
constant per fragment and injected from the fragment, never read per row.

Results: exact totals with unknown attribution (``excluded_unknown`` per column with ``_rows``,
``excluded_not_applicable``, ``unknown_total``), top-k under statistic sort keys (nulls last, ties
by the canonical row key, per group), output rows, distinct values and grain counts in the
column's storage type, and key sets.
"""

from __future__ import annotations

import heapq
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..catalog import POSITION_MARK, TableRef
from ..descriptor.columns import is_container
from ..plugins.base import Fragment, Normalized, ValueSnapshot
from ..plugins.formats import conjuncts, round_to_storage, storage_typed
from ..plugins.layouts import prune_fragments
from ..predicate import (
    map_columns,
    All,
    And,
    Any as AnyItem,
    CensoredCmp,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    IsNull,
    KindMatch,
    NonEmpty,
    Not,
    Or,
    Param,
    Predicate,
    PredicateError,
    Range,
    RankKey,
    TextMatch,
    evaluate,
    facet_predicate,
    from_json,
    is_null,
    kleene_and,
)
from ..roles import format_name, parse_path
from ..rowkey import canonical, element_type, render_value
from . import ServiceContext, ServiceError, layout_spec
from . import items as _items
from .sidecar import FooterInfo, build_access_index, footer_reader

__all__ = [
    "TableReader", "BudgetExceeded", "TableUnavailable", "UnboundParameter", "ScanStats", "Match", "Aggregate",
    "predicate_paths", "bind_params", "logical_leaves", "rendered",
]

ALL_COLUMNS = "*"
_ROW_INDEX = "__vbt_row_index__"


class BudgetExceeded(ServiceError):
    """The scan would decode more than the budget allows (witness: ``total_method: unknown``)."""

    def __init__(self, reason: str, *, need_bytes: int | None = None, limit_bytes: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.need_bytes = need_bytes
        self.limit_bytes = limit_bytes


class TableUnavailable(ServiceError):
    """The table's location is missing or holds no data file (never an empty table)."""


class UnboundParameter(ServiceError):
    """A ``{"param": name}`` in the predicate has no value: the witness cannot express the filter."""


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

def _join(prefix: str, inner: str) -> str:
    if not prefix:
        return inner
    if inner.startswith("[]"):
        return prefix + inner[2:] if prefix.endswith("[]") else prefix + inner
    return f"{prefix}.{inner}" if inner else prefix


def _resolve(stack: list[str], column: str) -> str:
    if column.startswith("/"):
        return column[1:]
    level = len(stack) - 1
    while column.startswith("^."):
        column = column[2:]
        level -= 1
    return _join(stack[max(level, 0)], column)


def predicate_paths(p: Predicate | None, stack: list[str] | None = None) -> list[str]:
    """Physical paths a predicate reads (quantifier bodies resolved against their containers; ``^.``
    and ``/`` honoured), in first-seen order."""
    out: dict[str, None] = {}

    def rec(q: Predicate, st: list[str]) -> None:
        if isinstance(q, (And, Or)):
            for x in q.preds:
                rec(x, st)
        elif isinstance(q, Not):
            rec(q.pred, st)
        elif isinstance(q, (AnyItem, All)):
            container = _resolve(st, q.path)
            before = len(out)
            rec(q.pred, st + [container if container.endswith("]") else container + "[]"])
            if len(out) == before:
                out.setdefault(container, None)        # a body without columns reads the container
        elif isinstance(q, (Contains, NonEmpty, KindMatch)):
            out.setdefault(_resolve(st, q.path), None)
        elif isinstance(q, CensoredCmp):
            out.setdefault(_resolve(st, q.time), None)
            out.setdefault(_resolve(st, q.event), None)
        else:
            out.setdefault(_resolve(st, q.column), None)  # type: ignore[union-attr]

    if p is not None:
        rec(p, list(stack or [""]))
    return list(out)


_map_pred = map_columns


def bind_params(p: Predicate | None, params: Mapping[str, Any] | None) -> Predicate | None:
    """``p`` with every :class:`Param` replaced by its value; a missing value raises :class:`UnboundParameter`."""
    if p is None:
        return None
    params = dict(params or {})

    def val(v: Any) -> Any:
        if isinstance(v, Param):
            if v.name not in params:
                raise UnboundParameter(f"the predicate needs parameter {v.name!r}, which the call does not give")
            return params[v.name]
        if isinstance(v, tuple):
            return tuple(val(x) for x in v)
        return v

    def rec(q: Predicate) -> Predicate:
        if isinstance(q, And):
            return And(tuple(rec(x) for x in q.preds))
        if isinstance(q, Or):
            return Or(tuple(rec(x) for x in q.preds))
        if isinstance(q, Not):
            return Not(rec(q.pred))
        if isinstance(q, (AnyItem, All)):
            return type(q)(q.path, rec(q.pred), q.skip_null_items)
        if isinstance(q, Eq):
            return Eq(q.column, val(q.value))
        if isinstance(q, In):
            vals: list[Any] = []
            for v in q.values:
                b = val(v)
                vals.extend(b if isinstance(b, (list, tuple, set, frozenset)) else [b])
            return In(q.column, tuple(vals))
        if isinstance(q, Cmp):
            return Cmp(q.column, q.op, val(q.value))
        if isinstance(q, CmpAbs):
            return CmpAbs(q.column, q.op, val(q.value))
        if isinstance(q, Range):
            return Range(q.column, val(q.lo), val(q.hi), q.lo_inclusive, q.hi_inclusive)
        if isinstance(q, Contains):
            return Contains(q.path, val(q.value))
        if isinstance(q, CensoredCmp):
            return CensoredCmp(q.time, q.event, q.op, val(q.value))
        return q

    return rec(p)


def _as_predicate(p: Any) -> Predicate | None:
    if p is None or (isinstance(p, Mapping) and not p):
        return None
    if isinstance(p, Mapping):
        return from_json(dict(p))
    return p


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------

def _is_list(pa: Any, t: Any) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t)


def logical_leaves(schema: Any) -> list[tuple[str, Any]]:
    """``(§6.4 path, leaf Arrow type)`` of every leaf of an Arrow schema (``items[].id``, ``tags[]``, ``m[].key``)."""
    import pyarrow as pa

    out: list[tuple[str, Any]] = []

    def rec(t: Any, prefix: str) -> None:
        if pa.types.is_dictionary(t):
            t = t.value_type
        if _is_list(pa, t):
            rec(t.value_type, prefix + "[]")
        elif pa.types.is_map(t):
            rec(t.key_type, prefix + "[].key")
            rec(t.item_type, prefix + "[].value")
        elif pa.types.is_struct(t):
            for f in t:
                rec(f.type, f"{prefix}.{format_name(f.name)}")
        else:
            out.append((prefix, t))

    for f in schema:
        rec(f.type, format_name(f.name))
    return out


def _type_at(schema: Any, path: str) -> Any:
    import pyarrow as pa

    names = [s.name for s in parse_path(path).segments if s.name]
    if not names or names[0] not in schema.names:
        return None
    t = schema.field(names[0]).type
    for name in names[1:]:
        while _is_list(pa, t) or pa.types.is_dictionary(t) or pa.types.is_map(t):
            if pa.types.is_map(t):
                t = pa.struct([pa.field("key", t.key_type), pa.field("value", t.item_type)])
            else:
                t = t.value_type
        if not pa.types.is_struct(t) or t.get_field_index(name) < 0:
            return None
        t = t.field(t.get_field_index(name)).type
    while _is_list(pa, t) or pa.types.is_dictionary(t):
        t = t.value_type
    return t


def _leaf_array(arr: Any, tokens: Sequence[str]) -> Any:
    """Flatten an Arrow array along ``tokens`` (field names and ``[]``) to the leaf values."""
    import pyarrow as pa
    import pyarrow.compute as pc

    for tok in tokens:
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if pa.types.is_dictionary(arr.type):
            arr = arr.dictionary_decode()
        if tok == "[]":
            arr = pc.list_flatten(arr) if not pa.types.is_map(arr.type) else arr.flatten()
        else:
            if pa.types.is_struct(arr.type):
                arr = pc.struct_field(arr, [arr.type.get_field_index(tok)])
            else:
                return pa.nulls(len(arr))
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_decode()
    return arr


def rendered(value: Any, storage_type: str | None) -> Any:
    """A value as the canonical JSON of its storage type parses back (float32 0.05 -> 0.05)."""
    if value is None:
        return None
    try:
        return json.loads(render_value(value, storage_type))
    except (TypeError, ValueError):
        return str(value)


class _Rev:
    """Reverses the order of a string (descending text keys)."""

    __slots__ = ("v",)

    def __init__(self, v: str) -> None:
        self.v = v

    def __lt__(self, other: "_Rev") -> bool:
        return self.v > other.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Rev) and self.v == other.v

    def __hash__(self) -> int:
        return hash(self.v)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ScanStats:
    total: int = 0
    unknown_total: int = 0
    excluded_unknown: dict[str, int] = field(default_factory=dict)
    excluded_not_applicable: dict[str, int] = field(default_factory=dict)
    scanned_bytes: int = 0
    estimated_bytes: int = 0
    row_groups: int = 0
    pruned_row_groups: int = 0
    fragments: int = 0
    pruned_fragments: int = 0
    counts: _items.ContainerCounts = field(default_factory=_items.ContainerCounts)
    used_sidecar: bool = False
    footer: bool = False                               # the total came from footers, nothing was scanned


@dataclass
class Match:
    row: dict[str, Any]                               # the native row, or the singleton item view
    key: tuple[Any, ...]
    positions: tuple[int, ...] = ()
    fragment: Fragment | None = None
    row_group: int = 0


@dataclass
class Aggregate:
    """What one pass computed (:meth:`TableReader.aggregate`)."""

    stats: ScanStats
    topk: list[list[Any]] | dict[str, list[list[Any]]] = field(default_factory=list)
    key_set: list[list[Any]] | None = None
    distinct: dict[str, list[Any]] = field(default_factory=dict)
    distinct_complete: dict[str, bool] = field(default_factory=dict)
    distinct_counts: dict[str, int] = field(default_factory=dict)
    group_totals: dict[str, int] = field(default_factory=dict)
    one_to_many: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

class TableReader:
    """Bounded reads of one table or item table of the catalog."""

    def __init__(self, ctx: ServiceContext, ref: str | TableRef) -> None:
        self.ctx = ctx
        self.ref = str(TableRef.parse(ref))
        self.table = ctx.table(self.ref)
        self.desc = self.table.descriptor
        self.spec = self.table.physical_spec
        layout_name = self.table.layout
        format_name = self.table.format
        if not layout_name or not format_name or format_name == "none":
            raise ServiceError(f"{self.ref} is served upstream only (layout {layout_name!r}, format {format_name!r})")
        self.layout = ctx.plugin("layout", layout_name)
        if "scan" not in (getattr(self.layout, "capabilities", ()) or ()):
            raise ServiceError(f"{self.ref} is served upstream only (layout {layout_name!r} has nothing to scan)")
        self.fmt = ctx.format_plugin(self.table)     # FormatRef options and the matrix spec applied
        self.lspec = layout_spec(self.table)
        self.root = self.desc.root or ""
        self.levels = _items.levels(self.table.items_path) if self.table.is_item_table else []
        self.key = tuple(self.table.key)
        self.partitions = dict(self.lspec.partitions)
        self.mirrors: dict[str, str] = {m: name for name, p in self.spec.partitions.items() for m in p.mirrored_by}
        self._footer = footer_reader(self.fmt)
        self._sig: str | None = None
        self._frags: list[Fragment] = []
        self._schema: Any = None
        self._cleaner: list[Any] | None = None
        self._kinds: dict[str, Any] = {}

    # -- identity -------------------------------------------------------------------

    def signature(self) -> str:
        return self.layout.signature(self.root, self.lspec)

    def refresh(self) -> None:
        sig = self.signature()
        if sig != self._sig:
            self._sig = sig
            self._frags = list(self.layout.fragments(self.root, self.lspec))
            self._schema = None

    def fragments(self, *, required: bool = True) -> list[Fragment]:
        self.refresh()
        if required and not self._frags:
            raise TableUnavailable(f"{self.table.physical}: no data files under {self.lspec.path or self.root!r} "
                                   f"(missing; an empty location is not an empty table)")
        return list(self._frags)

    def fingerprint(self) -> str:
        self.refresh()
        return self._fingerprint()

    def _fingerprint(self) -> str:
        """The fingerprint of the listing ``refresh()`` last made (no new stat walk)."""
        if self._sig is None:
            self.refresh()
        key = (str(self.table.physical), self._sig or "")
        fp = self.ctx.fingerprints.get(key)
        if fp is None:
            fp = self.layout.fingerprint(self._frags, self.ctx.manifest(self.desc.source))
            self.ctx.fingerprints[key] = fp
        return fp

    def partition_fingerprints(self) -> dict[str, str]:
        self.refresh()
        return dict(self.layout.partition_fingerprints(self._frags, self.ctx.manifest(self.desc.source)))

    def fragment_name(self, frag: Fragment) -> str:
        base = self.lspec.path or ""
        location = os.path.join(self.root, base) if self.root and not os.path.isabs(base) else (base or self.root)
        path = frag.uri[len("file://"):] if frag.uri.startswith("file://") else frag.uri
        if os.path.isdir(location):
            return os.path.relpath(path, location).replace(os.sep, "/")
        return os.path.basename(path)

    def schema(self) -> Any:
        """The logical schema of the current listing (re-read when ``refresh()`` sees a new signature)."""
        if self._schema is None:
            frags = self.fragments()
            self._schema = self.fmt.logical_schema(frags[0])
        return self._schema

    def footer(self, frag: Fragment) -> FooterInfo | None:
        if self._footer is None:
            return None
        key = (self._fingerprint(), frag.uri)
        info = self.ctx.footers.get(key)
        if info is None:
            info = self._footer(frag)
            with self.ctx.lock:
                self.ctx.footers[key] = info
        return info

    def fragment_stats(self, frag: Fragment) -> Any:
        key = (self._fingerprint(), "stats:" + frag.uri)
        st = self.ctx.footers.get(key)
        if st is None:
            st = self.fmt.stats(frag)
            with self.ctx.lock:
                self.ctx.footers[key] = st
        return st

    # -- columns ----------------------------------------------------------------------

    def physical_path(self, name: str) -> str:
        """A request path (bare item field, ``^.x``, ``/x`` or a full path) as a path of the physical table."""
        if not self.levels:
            name = name[1:] if name.startswith("/") else name
            # a descriptor column name need not be a §6.4 path ('moa-broad'): quote it as one segment
            if name in self.spec.columns or name in self.spec.partitions:
                return format_name(name)
            return name
        tops = list(self.spec.columns) + list(self.spec.partitions)
        return _items.qualify_item_path(name, self.levels, self._level_fields(), tops)

    def _level_fields(self) -> list[Mapping[str, Any]]:
        out: list[Mapping[str, Any]] = []
        cols: Mapping[str, Any] = self.spec.columns
        for lvl in self.levels:
            col: Any = None
            for n in lvl.names:
                col = cols.get(n) if cols is not None else None
                cols = dict(getattr(col, "fields", {}) or {}) if col is not None else {}
            out.append(dict(getattr(col, "fields", {}) or {}) if col is not None else {})
            cols = out[-1]
        return out

    def column_spec(self, path: str) -> Any:
        """The ColumnSpec at a physical path (struct and container fields descended), or None. A positional key part
        (``hallmarks.cancerHallmarks[]#``, an item's position) has none: counting such an item table raised a
        PathError, so ``_stats`` failed on the 25.09 target_cancer_hallmarks table."""
        if path.endswith(POSITION_MARK):
            return None
        cols: Mapping[str, Any] = dict(self.spec.columns)
        for name, p in self.spec.partitions.items():
            cols = {**cols, name: p.column} if name not in cols else cols
        col: Any = None
        for seg in parse_path(path.lstrip("/")).segments:
            if not seg.name:
                continue
            col = cols.get(seg.name)
            if col is None:
                return None
            cols = dict(getattr(col, "fields", {}) or {})
        return col

    def top_columns(self) -> list[str]:
        return [n for n in self.schema().names if n not in self.partitions]

    def storage_type(self, path: str) -> str | None:
        try:
            t = _type_at(self.schema(), path)
        except (ValueError, TypeError):
            return None
        return None if t is None else str(t)

    def storage_types(self, paths: Sequence[str]) -> list[str | None]:
        out: list[str | None] = []
        for p in paths:
            if p.endswith(POSITION_MARK):
                out.append("int64")
            elif p in self.partitions:
                out.append(self.partitions[p])
            else:
                out.append(self.storage_type(p))
        return out

    def leaf(self, path: str) -> str | None:
        """``path_in_schema`` (or group prefix) of a physical path; None for partition columns and absent paths."""
        p = path.lstrip("/")
        if p.endswith(POSITION_MARK):
            p = p[: -len(POSITION_MARK)]
        if parse_path(p).head in self.partitions:
            return None                                # constant per fragment: injected, never read
        try:
            return self.fmt.leaf_path(p, self.schema())
        except (ValueError, KeyError):
            return None

    def leaves(self, paths: Iterable[str]) -> list[str]:
        out: dict[str, None] = {}
        for p in paths:
            if p == ALL_COLUMNS:
                for c in self.top_columns():
                    out.setdefault(c, None)
                continue
            leaf = self.leaf(p)
            if leaf is not None:
                out.setdefault(leaf, None)
        return list(out)

    # -- cleaning (I6: in-band unknowns read as null) --------------------------------------

    def companions(self, path: str) -> list[str]:
        """Columns read with ``path`` because its facets refer to them: ``applies_when`` and ``unknown_when``
        (siblings in the same struct or item), ``placeholder_when: equals_key`` (its ``of`` column)."""
        spec = self.column_spec(path)
        if spec is None:
            return []
        parsed = parse_path(path)
        prefix = ".".join(s.name + ("[]" * len(s.brackets)) for s in parsed.segments[:-1])
        names = list((getattr(spec, "applies_when", None) or {}).keys())
        for cond in getattr(spec, "unknown_when", None) or []:
            if isinstance(cond, Mapping) and cond.get("column"):
                names.append(str(cond["column"]))
        if getattr(spec, "placeholder_when", None) == "equals_key" and getattr(spec, "of", None):
            names.append(str(spec.of))
        return [f"{prefix}.{n}" if prefix else n for n in names]

    def _clean_tree(self, columns: Mapping[str, Any]) -> list[Any]:
        tree: list[Any] = []
        for name, col in columns.items():
            if is_container(col):
                kids = self._clean_tree(col.fields)
                if kids:
                    tree.append((name, col, kids, (), (), (), None))
                continue
            role = getattr(col, "role", None)
            codes: list[Any] = list(getattr(col, "missing_values", []) or [])
            plugin = self.ctx.statistic(getattr(col, "statistic", None)) if role == "measure" else None
            if plugin is not None and hasattr(plugin, "missing_codes"):
                for c in plugin.missing_codes(col):
                    if c not in codes:
                        codes.append(c)
            holders = list(getattr(col, "placeholders", []) or [])
            conds: list[Predicate] = []
            for cond in list(getattr(col, "unknown_when", []) or []):
                cond = dict(cond)
                target = str(cond.pop("column", name))
                try:
                    conds.append(facet_predicate(cond, target))
                except (PredicateError, ValueError, TypeError):
                    continue
            of = getattr(col, "of", None) if getattr(col, "placeholder_when", None) == "equals_key" else None
            if codes or holders or conds or of:
                # the codes rendered once here: rendering them again for every value was half the time of the 25.09
                # expression check (34.6 M render_value calls for its nested rna/protein codes)
                rendered = frozenset(render_value(c) for c in (*codes, *holders))
                tree.append((name, col, None, rendered, (), tuple(conds), of))
        return tree

    def clean(self, obj: dict[str, Any]) -> dict[str, Any]:
        if self._cleaner is None:
            self._cleaner = self._clean_tree(self.spec.columns)
        if self._cleaner:
            _clean_dict(obj, self._cleaner)
        return obj

    # -- kinds -----------------------------------------------------------------------

    def kind_of(self, value: Any, id_type: str) -> bool | None:
        if is_null(value):
            return None
        plugin = self._kinds.get(id_type)
        if plugin is None:
            try:
                plugin = self.ctx.identifier(id_type if ":" in id_type else f"{self.desc.source}:{id_type}")
            except Exception:  # noqa: BLE001 - an id_type the catalog does not know is decided by the plugin name
                plugin = self.ctx.registry.find("identifier", id_type)
            self._kinds[id_type] = plugin
        if plugin is None:
            return None
        return isinstance(plugin.normalize_stored(str(value)), Normalized)

    # -- planning ---------------------------------------------------------------------------

    def _prune_fragments(self, pred: Predicate | None) -> list[Fragment]:
        frags = self.fragments()
        if pred is None:
            return frags
        parts = conjuncts(pred)
        if self.mirrors:
            parts = parts + [_map_pred(c, lambda c: self.mirrors.get(c, c)) for c in parts
                             if set(predicate_paths(c)) & set(self.mirrors)]
        return prune_fragments(frags, And(tuple(parts)) if len(parts) > 1 else parts[0])

    def _sidecar_columns(self) -> dict[str, Any]:
        out = {}
        for ap in self.spec.access_paths:
            if ap.via == "sidecar_index" and ap.columns:
                out[ap.columns[0]] = ap
        return out

    def _sidecar_allowed(self, pred: Predicate | None) -> set[tuple[str, int]] | None:
        """Row groups the sidecar indexes allow (None: no sidecar applies)."""
        indexed = self._sidecar_columns()
        if pred is None or not indexed:
            return None
        owner = self.ctx.reader(str(self.table.physical)) if self.levels else self
        allowed: set[tuple[str, int]] | None = None
        for c in conjuncts(pred):
            values: list[Any] | None = None
            col: str | None = None
            if isinstance(c, Eq):
                col, values = c.column, [c.value]
            elif isinstance(c, In):
                col, values = c.column, list(c.values)
            elif isinstance(c, Contains):
                col, values = c.path, [c.value]
            if col is None or values is None:
                continue
            physical = col.lstrip("/")
            name = next((k for k in indexed if k == physical), None)
            if name is None:
                continue
            path, _rows = build_access_index(owner, name)
            idx = self.ctx.sidecars[str(path)]
            hit = idx.lookup(values)
            allowed = hit if allowed is None else (allowed & hit)
        return allowed

    def _rg_may_match(self, info: FooterInfo, rg: int, parts: Sequence[Predicate]) -> bool:
        group = info.row_groups[rg]
        for c in parts:
            col = getattr(c, "column", None)
            if col is None or isinstance(c, (TextMatch, IsNull, CmpAbs)):
                continue
            physical = col.lstrip("/")
            if physical in self.partitions or parse_path(physical).crosses_list:
                continue
            leaf = self.leaf(physical)
            chunk = group.chunks.get(leaf) if leaf else None
            # rows where this comparison is unknown (nulls, in-band codes) are attributed, never pruned
            if chunk is None or chunk.null_count is None or chunk.null_count > 0 or self._in_band_codes(physical):
                continue
            if not chunk.has_minmax:
                continue
            st = self.storage_type(physical)
            try:
                if not _interval_may_match(c, chunk.min, chunk.max, st):
                    return False
            except TypeError:
                continue
        return True

    def plan(self, pred: Predicate | None, leaves: Sequence[str], *, use_sidecars: bool = True
             ) -> tuple[list[tuple[Fragment, list[int] | None]], ScanStats]:
        """``[(fragment, row groups or None)]`` that may hold matches, with the decoded-bytes estimate."""
        stats = ScanStats()
        all_frags = self.fragments()
        frags = self._prune_fragments(pred)
        stats.fragments = len(all_frags)
        stats.pruned_fragments = len(all_frags) - len(frags)
        allowed = self._sidecar_allowed(pred) if use_sidecars else None
        stats.used_sidecar = allowed is not None
        parts = conjuncts(pred) if pred is not None else []
        out: list[tuple[Fragment, list[int] | None]] = []
        for frag in frags:
            info = self.footer(frag)
            if info is None:
                out.append((frag, None))
                stats.row_groups += 1
                stats.estimated_bytes += int((frag.size or 0) * 3)
                continue
            name = self.fragment_name(frag) if allowed is not None else ""
            keep = []
            for rg in range(len(info.row_groups)):
                stats.row_groups += 1
                if allowed is not None and (name, rg) not in allowed:
                    stats.pruned_row_groups += 1
                    continue
                if parts and not self._rg_may_match(info, rg, parts):
                    stats.pruned_row_groups += 1
                    continue
                keep.append(rg)
            if keep:
                out.append((frag, keep))
                stats.estimated_bytes += info.bytes(leaves, keep)
        return out, stats

    def estimate_scan_bytes(self, leaves: Sequence[str], predicate: Any = None, *, params: Mapping[str, Any] | None
                            = None, use_sidecars: bool = True) -> int:
        """Decoded bytes of ``leaves`` (``path_in_schema``) over the row groups left after partition,
        statistics and sidecar pruning."""
        _, stats = self.plan(self.prepare(predicate, params), list(leaves), use_sidecars=use_sidecars)
        return stats.estimated_bytes

    def prepare(self, predicate: Any, params: Mapping[str, Any] | None = None) -> Predicate | None:
        """The request predicate with parameters bound and (item tables) paths made physical."""
        return self._prepare(predicate, params)[0]

    def _prepare(self, predicate: Any, params: Mapping[str, Any] | None = None
                 ) -> tuple[Predicate | None, dict[str, str]]:
        names: dict[str, str] = {}

        def physical(name: str) -> str:
            path = self.physical_path(name)
            names.setdefault(path, name)
            return path

        pred = bind_params(_as_predicate(predicate), params)
        if pred is not None and self.levels:
            pred = _map_pred(pred, physical)
        if pred is not None:
            # literals as stored (a float32 0.05 is 0.05000000074505806): equality is exact (F-10)
            pred = storage_typed(pred, lambda p: self.partitions.get(p) or self.storage_type(p))
        return pred, names

    # -- reading ------------------------------------------------------------------------------

    def _read(self, frag: Fragment, rg: int | None, leaves: Sequence[str], info: FooterInfo | None) -> list[dict]:
        return self._read_rows(frag, rg, leaves, info)[0]

    def _read_rows(self, frag: Fragment, rg: int | None, leaves: Sequence[str], info: FooterInfo | None,
                   push: Sequence[tuple[Predicate, list[str]]] = (),
                   arrow_filter: Callable[[Any], Any] | None = None) -> tuple[list[dict], list[int]]:
        """``(native rows, their row indices in the row group)``. ``push`` conjuncts are applied in Arrow
        first, relaxed to keep rows where they are unknown (null or NaN): only rows on which a pushed
        conjunct is false are dropped, and those can never match or count as unknown. ``arrow_filter``
        (a boolean mask of the Arrow table read) drops rows before they are converted."""
        rows: list[dict] = []
        index: list[int] = []
        for r, i in self._row_chunks(frag, rg, leaves, info, push, arrow_filter, None):
            rows.extend(r)
            index.extend(i)
        return rows, index

    def _row_chunks(self, frag: Fragment, rg: int | None, leaves: Sequence[str], info: FooterInfo | None,
                    push: Sequence[tuple[Predicate, list[str]]] = (),
                    arrow_filter: Callable[[Any], Any] | None = None,
                    chunk: int | None = None) -> Iterator[tuple[list[dict], list[int]]]:
        """:meth:`_read_rows` converted ``chunk`` rows at a time (None: all at once). The row group is read into
        Arrow whole; only its conversion to Python is sliced."""
        import pyarrow as pa

        if info is None:
            tops = sorted({leaf.split(".")[0] for leaf in leaves}) or None
            batches = list(self.fmt.scan([frag], columns=tops, predicate=None, partitions=self.partitions))
            if not batches:
                return
            tbl = pa.Table.from_batches(batches)
            index = list(range(tbl.num_rows))
        elif not leaves:
            tbl = None
            index = list(range(info.row_groups[rg or 0].rows))
        else:
            tbl = self.fmt.read_leaves(frag, list(leaves), [rg])
            expr = self._pushdown(tbl.schema, push) if push else None
            if expr is not None:
                tbl = tbl.append_column(_ROW_INDEX, pa.array(range(tbl.num_rows), pa.int64())).filter(expr)
                index = tbl.column(_ROW_INDEX).to_pylist()
                tbl = tbl.drop_columns([_ROW_INDEX])
            else:
                index = list(range(tbl.num_rows))
        if tbl is not None and arrow_filter is not None:
            tbl, index = _masked(tbl, index, arrow_filter)
        part = dict(frag.partition or {})
        step = chunk or max(len(index), 1)
        for start in range(0, len(index), step):
            if tbl is None:
                rows: list[dict] = [{} for _ in range(min(step, len(index) - start))]
            else:
                rows = self.fmt.to_native(tbl.slice(start, step) if step < len(index) else tbl)
            for row in rows:
                for k, v in part.items():
                    row[k] = v
                self.clean(row)
            yield rows, index[start:start + step]

    def _pushable(self, conj: Sequence[Predicate], conj_paths: Sequence[list[str]]
                  ) -> list[tuple[Predicate, list[str]]]:
        """Conjuncts on flat file columns without in-band unknowns (cleaning must see those rows)."""
        if "pushdown" not in (getattr(self.fmt, "capabilities", ()) or ()):
            return []
        out = []
        for c, paths in zip(conj, conj_paths):
            if not paths or any(p in self.partitions or parse_path(p).crosses_list or self._unclean(p)
                                for p in paths):
                continue
            out.append((c, paths))
        return out

    def _in_band_codes(self, path: str) -> bool:
        """Does the column store unknowns as values (codes, placeholders), which min/max cannot tell apart?"""
        spec = self.column_spec(path)
        if spec is None:
            return False
        if any(getattr(spec, f, None) for f in ("missing_values", "unknown_when", "placeholders", "placeholder_when")):
            return True
        plugin = self.ctx.statistic(getattr(spec, "statistic", None)) if getattr(spec, "role", None) == "measure" \
            else None
        return bool(plugin is not None and hasattr(plugin, "missing_codes") and plugin.missing_codes(spec))

    def _unclean(self, path: str) -> bool:
        spec = self.column_spec(path)
        if spec is None:
            return False
        if any(getattr(spec, f, None) for f in ("missing_values", "unknown_when", "placeholders",
                                                  "placeholder_when", "missing")):
            return True
        plugin = self.ctx.statistic(getattr(spec, "statistic", None)) if getattr(spec, "role", None) == "measure" \
            else None
        return bool(plugin is not None and hasattr(plugin, "missing_codes") and plugin.missing_codes(spec))

    def _pushdown(self, schema: Any, push: Sequence[tuple[Predicate, list[str]]]) -> Any:
        import pyarrow as pa
        import pyarrow.compute as pc

        exprs = []
        for c, paths in push:
            try:
                expr, _residual = self.fmt.compile(c, schema)
            except (ValueError, TypeError, KeyError, pa.ArrowException):
                continue
            if expr is None:
                continue
            for p in paths:
                names = [s.name for s in parse_path(p).segments]
                if names[0] not in schema.names:
                    expr = None
                    break
                f = pc.field(*names)
                expr = expr | f.is_null()
                t = _type_at(schema, p)
                if t is not None and pa.types.is_floating(t):
                    expr = expr | pc.is_nan(f)
            if expr is not None:
                exprs.append(expr)
        if not exprs:
            return None
        out = exprs[0]
        for e in exprs[1:]:
            out = out & e
        return out

    def scan(self, predicate: Any = None, *, columns: Sequence[str] | None = None,
             params: Mapping[str, Any] | None = None, unknown_columns: Sequence[str] | None = None,
             budget_bytes: int | None = None, stats: ScanStats | None = None,
             attribute_unknown: bool = True,
             row_filter: Callable[[Mapping[str, Any]], bool] | None = None,
             arrow_filter: Callable[[Any], Any] | None = None) -> Iterator[Match]:
        """Matches of ``predicate`` (rows, or items of an item table) with the key and ``columns``
        (physical paths; ``"*"`` for every column) materialised. Fills ``stats`` with the totals.
        ``row_filter`` skips a stored row (with all its items) before anything else looks at it;
        ``arrow_filter`` (an Arrow table of the key and predicate leaves -> a boolean mask) skips it before it
        is converted to Python at all."""
        pred, names = self._prepare(predicate, params)
        conj = conjuncts(pred) if pred is not None else []
        conj_paths = [predicate_paths(c) for c in conj]
        key_paths = list(self.key)
        out_paths = [ALL_COLUMNS] if columns is None else list(columns)
        filters = [p for ps in conj_paths for p in ps]
        filters += [c for p in filters for c in self.companions(p)]
        pass1 = self.leaves([*key_paths, *filters])
        if self.levels:
            inner = self.leaf(self.levels[-1].text)
            if inner is not None and not any(x == inner or x.startswith(inner + ".") for x in pass1):
                pass1.append(inner)                    # the container must be read to find its items
        outputs = [*out_paths, *(c for p in out_paths if p != ALL_COLUMNS for c in self.companions(p))]
        pass2 = self.leaves([*key_paths, *filters, *outputs])
        plan, st = self.plan(pred, sorted(set(pass1) | set(pass2)))
        limit = self.ctx.scan_budget(self.table, budget_bytes)
        if st.estimated_bytes > limit:
            raise BudgetExceeded(f"reading {self.ref} needs ~{st.estimated_bytes} decoded bytes after pruning, over "
                                 f"the {limit}-byte scan budget", need_bytes=st.estimated_bytes, limit_bytes=limit)
        if stats is None:
            stats = ScanStats()
        for name in ("estimated_bytes", "row_groups", "pruned_row_groups", "fragments", "pruned_fragments",
                     "used_sidecar"):
            setattr(stats, name, getattr(st, name))
        tracked = _tracked(unknown_columns, conj_paths, self.physical_path) if attribute_unknown else None
        same = set(pass2) <= set(pass1)
        kind_of = self.kind_of
        push = self._pushable(conj, conj_paths) if not self.levels else []
        for frag, rgs in plan:
            info = self.footer(frag)
            for rg in (rgs if rgs is not None else [None]):
                if info is not None:
                    stats.scanned_bytes += info.bytes(pass1, [rg or 0])
                # one row group's second pass, read once however many of its slices hold matches
                second: dict[str, Any] = {}
                # converted SCAN_CHUNK_ROWS rows at a time: a 25.09 interaction shard is one row group of 1.3 M
                # rows, and its key and filter columns converted whole ran the data child out of memory
                for rows, index in self._row_chunks(frag, rg, pass1, info, push, arrow_filter, SCAN_CHUNK_ROWS):
                    hits: list[tuple[int, tuple[int, ...]]] = []
                    for i, row in zip(index, rows):
                        if row_filter is not None and not row_filter(row):
                            continue
                        views = _items.explode(row, self.levels, stats.counts) if self.levels else [(row, ())]
                        for view, pos in views:
                            truths = [evaluate(c, view, None, kind_of=kind_of) for c in conj]
                            overall = kleene_and(truths)
                            if overall is True:
                                hits.append((i, pos))
                            elif overall is None:
                                stats.unknown_total += 1
                                if tracked is not None:
                                    self._attribute(view, conj, conj_paths, truths, tracked, stats, names)
                    if not hits:
                        continue
                    if same:
                        full = dict(zip(index, rows))
                    else:
                        wanted = sorted({i for i, _ in hits})
                        if info is None:
                            if "rows" not in second:
                                second["rows"] = dict(enumerate(self._read(frag, rg, pass2, info)))
                            full = second["rows"]
                        else:
                            if "table" not in second:
                                extra = [x for x in pass2 if x not in pass1]
                                second["table"] = self.fmt.read_leaves(frag, list(pass2), [rg or 0]) if pass2 \
                                    else None
                                stats.scanned_bytes += info.bytes(extra, [rg or 0])
                            full_rows = self._read_take(frag, rg or 0, pass2, wanted, table=second["table"])
                            full = dict(zip(wanted, full_rows))
                    for i, pos in hits:
                        row = full[i]
                        view = _items.view_at(row, self.levels, pos) if self.levels else row
                        if view is None:
                            continue
                        stats.total += 1
                        yield Match(view, _items.key_values(view, self.key, pos, self.levels), pos, frag, rg or 0)

    def _read_take(self, frag: Fragment, rg: int, leaves: Sequence[str], indices: Sequence[int],
                   table: Any = None) -> list[dict]:
        """Rows ``indices`` of a row group (``table``: its ``leaves`` already read) converted to Python."""
        import pyarrow as pa

        tbl = table if table is not None else self.fmt.read_leaves(frag, list(leaves), [rg]) if leaves else None
        if tbl is None:
            return [dict(frag.partition or {}) for _ in indices]
        rows = self.fmt.to_native(tbl.take(pa.array(list(indices), pa.int64())))
        for row in rows:
            for k, v in (frag.partition or {}).items():
                row[k] = v
            self.clean(row)
        return rows

    def _attribute(self, view: Mapping[str, Any], conj: Sequence[Predicate], conj_paths: Sequence[list[str]],
                   truths: Sequence[bool | None], tracked: set[str] | None, stats: ScanStats,
                   names: Mapping[str, str] | None = None) -> None:
        """Count a row whose predicate is unknown against every column whose comparison is unknown while every
        other conjunct is true (reported under the name the request used)."""
        names = names or {}
        if any(t is False for t in truths):
            return
        attributed = False
        seen: set[str] = set()                         # one count per column, however many conjuncts read it
        for i, t in enumerate(truths):
            if t is not None:
                continue
            cols = conj_paths[i]
            null_cols = [c for c in cols if all(is_null(v) for v in _items.path_values(view, c) or [None])]
            for col in (null_cols or cols):
                if (tracked is not None and col not in tracked) or col in seen:
                    continue
                seen.add(col)
                name = names.get(col, col)
                if self._not_applicable(view, col):
                    stats.excluded_not_applicable[name] = stats.excluded_not_applicable.get(name, 0) + 1
                else:
                    stats.excluded_unknown[name] = stats.excluded_unknown.get(name, 0) + 1
                    attributed = True
        if attributed:
            stats.excluded_unknown["_rows"] = stats.excluded_unknown.get("_rows", 0) + 1

    def _not_applicable(self, view: Mapping[str, Any], col: str) -> bool:
        spec = self.column_spec(col)
        when = getattr(spec, "applies_when", None) if spec is not None else None
        if not when:
            return False
        parsed = parse_path(col)
        prefix = ".".join(s.name + ("[]" if s.is_list else "") for s in parsed.segments[:-1])
        for other, allowed in when.items():
            path = f"{prefix}.{other}" if prefix else other
            vals = _items.path_values(view, path)
            if vals and all(v not in allowed for v in vals):
                return True
        return False

    # -- one-pass aggregation ------------------------------------------------------------------

    def order_keys(self, order: Sequence[Any]) -> list[tuple[RankKey, Any, Any, str]]:
        out = []
        for o in order or ():
            rk = o if isinstance(o, RankKey) else RankKey.from_json(o if isinstance(o, Mapping) else
                                                                    o.model_dump())
            path = self.physical_path(rk.column)
            spec = self.column_spec(path)
            name = rk.statistic or getattr(spec, "statistic", None)
            out.append((rk, self.ctx.statistic(name), spec, path))
        return out

    def sort_key(self, view: Mapping[str, Any], keys: Sequence[tuple[RankKey, Any, Any, str]],
                 tie: str) -> tuple[Any, ...]:
        parts: list[Any] = []
        for rk, plugin, spec, path in keys:
            value = _items.path_value(view, path)
            if isinstance(value, list):
                value = None
            v: Any = None
            if not is_null(value):
                if plugin is not None and hasattr(plugin, "rank_value"):
                    try:
                        v = plugin.rank_value(value, spec)
                    except (TypeError, ValueError):
                        v = None
                    if v is None and isinstance(value, str):
                        v = value
                elif isinstance(value, bool):
                    v = float(value)
                elif isinstance(value, (int, float)):
                    v = float(value)
                else:
                    v = str(value)
            if v is None:
                parts.append((1 if rk.nulls == "last" else -1,))
                continue
            if isinstance(v, str):
                parts.append((0, 1, _Rev(v) if rk.direction.startswith("desc") else v))
                continue
            if rk.direction.endswith("_abs"):
                v = abs(v)
            if rk.direction.startswith("desc"):
                v = -v
            parts.append((0, 0, v + 0.0))
        parts.append(tie)
        return tuple(parts)

    def canonical_key(self, values: Sequence[Any], paths: Sequence[str] | None = None) -> str:
        return canonical(list(values), self.storage_types(list(paths or self.key)))

    def key_json(self, values: Sequence[Any], paths: Sequence[str] | None = None) -> list[Any]:
        """A key as JSON values rendered in their storage types."""
        return json.loads(self.canonical_key(values, paths))

    def aggregate(self, predicate: Any = None, *, order: Sequence[Any] = (), k: int | None = None,
                  group_by: Sequence[str] = (), key: Sequence[str] = (), distinct: Sequence[str] = (),
                  max_values: int | None = None, grains: Mapping[str, Any] | None = None,
                  key_set_max: int | None = None, one_to_many: bool = False,
                  params: Mapping[str, Any] | None = None, unknown_columns: Sequence[str] = (),
                  budget_bytes: int | None = None) -> Aggregate:
        """One scan computing totals, top-k (per group), key set, distinct values, grain counts, group
        totals and keys shared by several rows."""
        if not (order and k) and not group_by and not key and not distinct and not grains and key_set_max is None \
                and not one_to_many and not self.levels and self.prepare(predicate, params) is None:
            fast = self._footer_count()
            if fast is not None:
                return Aggregate(fast)
        key_paths = [self.physical_path(c) for c in key] if key else list(self.key)
        okeys = self.order_keys(order)
        groups = [self.physical_path(g) for g in group_by]
        dpaths = [self.physical_path(d) for d in distinct]
        grain_specs = {name: self._grain(g) for name, g in (grains or {}).items()}
        need = [*key_paths, *(p for *_, p in okeys), *groups, *dpaths,
                *(c for g in grain_specs.values() for c in g["columns"])]
        stats = ScanStats()
        heaps: dict[str, list[tuple[Any, ...]]] = {}
        key_set: set[str] | None = set() if key_set_max is not None else None
        dvals: dict[str, dict[str, Any]] = {d: {} for d in distinct}
        dcomplete = {d: True for d in distinct}
        dsets: dict[str, set[str]] = {name: set() for name in grain_specs}
        gtotals: dict[str, int] = {}
        counts: dict[str, int] = {}
        counter_open = True
        cap = max_values if max_values is not None else int(self.ctx.settings.witness.max_key_set)
        key_types = self.storage_types(key_paths)
        seq = 0
        for m in self.scan(predicate, columns=need, params=params, unknown_columns=unknown_columns,
                           budget_bytes=budget_bytes, stats=stats):
            kv = tuple(_items.key_values(m.row, key_paths, m.positions, self.levels)) if key else m.key
            ckey = canonical(list(kv), key_types)
            gkey = canonical([_items.path_value(m.row, g) for g in groups]) if groups else ""
            if groups:
                gtotals[gkey] = gtotals.get(gkey, 0) + 1
            if k is not None and k > 0:
                sk = self.sort_key(m.row, okeys, ckey)
                heap = heaps.setdefault(gkey, [])
                seq += 1
                item = (_Neg(sk), seq, ckey)
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if key_set is not None and len(key_set) <= key_set_max:  # type: ignore[operator]
                key_set.add(ckey)
            for name, path in zip(distinct, dpaths):
                if not dcomplete[name]:
                    continue
                st = self.storage_type(path) if path not in self.partitions else self.partitions[path]
                for v in _items.path_values(m.row, path) if parse_path(path).crosses_list else \
                        [_items.path_value(m.row, path)]:
                    if is_null(v):
                        continue
                    r = render_value(v, st)
                    if r not in dvals[name]:
                        if len(dvals[name]) >= cap:
                            dcomplete[name] = False
                            break
                        dvals[name][r] = rendered(v, st)
            for name, g in grain_specs.items():
                dsets[name].add(self._grain_value(m.row, m.positions, g))
            if one_to_many and counter_open:
                counts[ckey] = counts.get(ckey, 0) + 1
                if len(counts) > cap:
                    counter_open = False
        agg = Aggregate(stats)
        if k is not None and k > 0:
            per = {g: [json.loads(ck) for _neg, _s, ck in sorted(h, reverse=True)] for g, h in heaps.items()}
            agg.topk = per if groups else per.get("", [])
        if key_set is not None:
            agg.key_set = None if len(key_set) > key_set_max else [json.loads(c) for c in sorted(key_set)]  # type: ignore[operator]
        agg.distinct = {d: [dvals[d][r] for r in sorted(dvals[d])] for d in distinct}
        agg.distinct_complete = dcomplete
        agg.distinct_counts = {name: len(s) for name, s in dsets.items()}
        agg.group_totals = dict(sorted(gtotals.items()))
        agg.one_to_many = {c: n for c, n in sorted(counts.items()) if n > 1}
        return agg

    def _footer_count(self) -> ScanStats | None:
        """Rows of an unfiltered table from its footers (``total_method: footer``), None without footers."""
        stats = ScanStats(footer=True)
        for frag in self.fragments():
            info = self.footer(frag)
            if info is None:
                return None
            stats.total += info.rows
            stats.fragments += 1
            stats.row_groups += len(info.row_groups)
        return stats

    # -- grains ----------------------------------------------------------------------------------

    def _grain(self, g: Any) -> dict[str, Any]:
        if isinstance(g, str):
            named = self.table.spec.grains.get(g)
            if named is None:
                raise ServiceError(f"{self.ref} declares no grain {g!r}")
            g = named
        if isinstance(g, (list, tuple)):
            return {"columns": [self.physical_path(c) for c in g], "unordered": [], "canonicalize": None, "by": []}
        data = g if isinstance(g, Mapping) else g.model_dump()
        cols = [self.physical_path(c) for c in data.get("columns") or []]
        unordered = [self.physical_path(c) for c in data.get("unordered") or []]
        by = [self.physical_path(c) for c in data.get("by") or []]
        canon = data.get("canonicalize")
        parents = self._parents(canon) if canon else None
        return {"columns": [*cols, *unordered, *by], "plain": cols, "unordered": unordered, "by": by,
                "canonicalize": parents}

    def _grain_value(self, view: Mapping[str, Any], pos: Sequence[int], g: Mapping[str, Any]) -> str:
        plain = g.get("plain", g["columns"])
        vals = [_items.key_values(view, [c], pos, self.levels)[0] for c in plain]
        parents = g.get("canonicalize")
        if parents is not None and vals:
            vals[0] = parents.get(str(vals[0]), vals[0]) if vals[0] is not None else None
        if g.get("unordered"):
            pair = sorted((render_value(_items.path_value(view, c)) for c in g["unordered"]))
            vals.append(pair)
        for c in g.get("by", []):
            vals.append(_items.path_value(view, c))
        return canonical(vals)

    def _parents(self, ref: str) -> dict[str, str]:
        """``id_type.parent`` -> {key: parent key} from the id_type's ``canonicalize.parent`` column."""
        id_type = ref.rsplit(".", 1)[0]
        cached = self.ctx.parents.get(id_type)
        if cached is not None:
            return cached
        src, spec = self.ctx.catalog.id_type(id_type, self.desc.source)
        if spec.canonicalize is None:
            raise ServiceError(f"id_type {id_type} declares no canonicalize.parent")
        table, _, column = spec.canonicalize.parent.partition(".")
        reader = self.ctx.reader(f"{src}.{table}")
        key = reader.key[0] if reader.key else "id"
        out: dict[str, str] = {}
        for m in reader.scan(None, columns=[key, column], attribute_unknown=False):
            k = _items.path_value(m.row, key)
            p = _items.path_value(m.row, column)
            if k is not None:
                out[str(k)] = str(p) if p not in (None, "") else str(k)
        self.ctx.parents[id_type] = out
        return out

    # -- convenience API (one scan each) -------------------------------------------------------------

    def count(self, predicate: Any = None, grain: Any = None, *, params: Mapping[str, Any] | None = None,
              unknown_columns: Sequence[str] = (), budget_bytes: int | None = None
              ) -> tuple[int, dict[str, int], dict[str, int], int]:
        """``(total, excluded_unknown, excluded_not_applicable, unknown_total)``; with ``grain`` the total
        counts distinct grain values."""
        agg = self.aggregate(predicate, grains={"_": grain} if grain else None, params=params,
                             unknown_columns=unknown_columns, budget_bytes=budget_bytes)
        total = agg.distinct_counts["_"] if grain else agg.stats.total
        return total, dict(agg.stats.excluded_unknown), dict(agg.stats.excluded_not_applicable), agg.stats.unknown_total

    def top_k(self, predicate: Any, order: Sequence[Any], k: int, key: Sequence[str] = (),
              group_by: Sequence[str] = (), *, params: Mapping[str, Any] | None = None,
              budget_bytes: int | None = None) -> list[list[Any]] | dict[str, list[list[Any]]]:
        return self.aggregate(predicate, order=order, k=k, key=key, group_by=group_by, params=params,
                              budget_bytes=budget_bytes).topk

    def distinct(self, predicate: Any, columns: Sequence[str], max_values: int | None = None, *,
                 params: Mapping[str, Any] | None = None) -> dict[str, list[Any]]:
        return self.aggregate(predicate, distinct=columns, max_values=max_values, params=params).distinct

    def distinct_counts(self, predicate: Any, grains: Mapping[str, Any], *,
                        params: Mapping[str, Any] | None = None) -> dict[str, int]:
        return self.aggregate(predicate, grains=grains, params=params).distinct_counts

    def key_set(self, predicate: Any, key: Sequence[str] = (), max_n: int | None = None, *,
                params: Mapping[str, Any] | None = None) -> list[list[Any]] | None:
        cap = max_n if max_n is not None else int(self.ctx.settings.witness.max_key_set)
        return self.aggregate(predicate, key=key, key_set_max=cap, params=params).key_set

    def rows(self, predicate: Any = None, columns: Sequence[str] = (), order: Sequence[Any] = (),
             limit: int | None = None, limit_grain: Any = None, *, explode: Sequence[str] = (),
             carry: Sequence[str] = (), rename: Mapping[str, str] | None = None,
             params: Mapping[str, Any] | None = None, budget_bytes: int | None = None,
             stats: ScanStats | None = None, group_by: Sequence[str] = (),
             distinct: Sequence[str] = ()) -> tuple[list[dict[str, Any]], list[list[Any]], ScanStats]:
        """``(rows, canonical row keys, stats)``: matching rows as native dicts, ordered (``order``, else
        the table's ``rank``, then the canonical key), cut to ``limit`` (per ``limit_grain`` value: each
        grain's best row), with ``explode``/``carry``/``rename`` applied to the output.

        With ``group_by`` (default: the ``within`` columns of the order) rows are ranked within each group,
        groups follow each other in key order and ``limit`` applies per group. With ``distinct`` only the
        first row of each combination of those columns is kept."""
        okeys = self.order_keys(order or [r.model_dump() for r in self.table.spec.rank])
        groups = [self.physical_path(g) for g in group_by] or \
            list(dict.fromkeys(self.physical_path(w) for rk, *_ in okeys for w in rk.within))
        dpaths = [self.physical_path(d) for d in distinct]
        stats = stats or ScanStats()
        need: list[str] = [ALL_COLUMNS] if not columns else [self.physical_path(c) for c in (*columns, *carry)]
        need += [p for *_, p in okeys] + groups + dpaths
        grain = self._grain(limit_grain) if limit_grain else None
        if grain:
            need += grain["columns"]
        bounded = limit is not None and grain is None and not groups and not dpaths
        key_types = self.storage_types(list(self.key))
        heap: list[tuple[Any, ...]] = []
        collected: list[tuple[Any, ...]] = []
        seq = 0
        for m in self.scan(predicate, columns=need, params=params, budget_bytes=budget_bytes, stats=stats):
            ckey = canonical(list(m.key), key_types)
            sk = self.sort_key(m.row, okeys, ckey)
            if groups:
                sk = (canonical([_items.path_value(m.row, g) for g in groups]),) + sk
            seq += 1
            item = (sk, seq, m, ckey)
            if bounded:
                neg = (_Neg(sk), seq, m, ckey)
                if len(heap) < int(limit):  # type: ignore[arg-type]
                    heapq.heappush(heap, neg)
                elif neg > heap[0]:
                    heapq.heapreplace(heap, neg)
            else:
                collected.append(item)
        if bounded:
            ordered = [(x[0].v, x[1], x[2], x[3]) for x in sorted(heap, reverse=True)]
        else:
            ordered = sorted(collected, key=lambda x: (x[0], x[1]))
        if grain is not None:
            seen: set[str] = set()
            best = []
            for item in ordered:
                g = self._grain_value(item[2].row, item[2].positions, grain)
                if g in seen:
                    continue
                seen.add(g)
                best.append(item)
            ordered = best
        if dpaths:
            combos: set[str] = set()
            unique = []
            for item in ordered:
                c = canonical([_items.path_value(item[2].row, d) for d in dpaths])
                if c not in combos:
                    combos.add(c)
                    unique.append(item)
            ordered = unique
        if limit is not None and groups:
            per: dict[Any, int] = {}
            cut = []
            for item in ordered:
                n = per.get(item[0][0], 0)
                if n < int(limit):
                    per[item[0][0]] = n + 1
                    cut.append(item)
            ordered = cut
        elif limit is not None:
            ordered = ordered[: int(limit)]
        out_rows: list[dict[str, Any]] = []
        keys: list[list[Any]] = []
        if self.levels:
            explode = ()                               # rows of an item table already are the items
        for _sk, _s, m, ckey in ordered:
            row = self.output_row(m, columns)
            for r in _explode_rows(row, m.row, explode, carry, self):
                out_rows.append(_renamed(r, rename))
                keys.append(json.loads(ckey))
        return out_rows, keys, stats

    def output_row(self, m: Match, columns: Sequence[str] = ()) -> dict[str, Any]:
        if not columns:
            if self.levels:
                return _items.item_row(m.row, self.levels, self.key, m.positions)
            return dict(m.row)
        out: dict[str, Any] = {}
        for c in columns:
            path = self.physical_path(c)
            if path.endswith(POSITION_MARK):
                out[c] = _items.key_values(m.row, [path], m.positions, self.levels)[0]
            else:
                out[c] = _items.path_value(m.row, path)
        return out

    # -- leaf values, snapshots, samples ----------------------------------------------------------------

    def leaf_values(self, path: str, predicate: Any = None) -> Iterator[tuple[Fragment, int, list[Any]]]:
        """``(fragment, row group, values)`` of one leaf (list elements flattened), one row group at a time."""
        for frag, rg, arr in self.leaf_arrays(path, predicate):
            yield frag, rg, arr.to_pylist()

    def leaf_arrays(self, path: str, predicate: Any = None) -> Iterator[tuple[Fragment, int, Any]]:
        import pyarrow as pa

        physical = self.physical_path(path)
        if physical in self.partitions:
            for frag in self._prune_fragments(_as_predicate(predicate)):
                info = self.footer(frag)
                rows = info.rows if info is not None else 1
                yield frag, 0, pa.array([frag.partition.get(physical)] * rows)
            return
        leaf = self.leaf(physical)
        if leaf is None:
            raise ServiceError(f"{self.ref}: column {path!r} is not in the data")
        tokens = _tokens(physical)
        plan, _ = self.plan(_as_predicate(predicate), [leaf], use_sidecars=False)
        for frag, rgs in plan:
            info = self.footer(frag)
            for rg in (rgs if rgs is not None else [None]):
                if info is None:
                    batches = list(self.fmt.scan([frag], columns=[tokens[0]], predicate=None,
                                                 partitions=self.partitions))
                    if not batches:
                        continue
                    tbl = pa.Table.from_batches(batches)
                else:
                    tbl = self.fmt.read_leaves(frag, [leaf], [rg])
                yield frag, rg or 0, _leaf_array(tbl.column(tokens[0]), tokens[1:])

    def snapshot(self, path: str, *, budget_bytes: int | None = None, max_values: int | None = None
                 ) -> ValueSnapshot:
        """Distinct raw values of a column with counts (null and NaN counted apart), complete unless the
        budget or ``max_values`` cut it short. Uses row-group ``min == max`` statistics when every row group
        of a flat column is single-valued."""
        import pyarrow as pa
        import pyarrow.compute as pc

        physical = self.physical_path(path)
        storage = self.partitions.get(physical) or self.storage_type(physical)
        while element_type(storage):
            storage = element_type(storage)     # a list column's values are its elements (see below)
        fast = self._single_valued(physical)
        if fast is not None:
            values = tuple(sorted(fast, key=lambda v: render_value(v, storage)))
            return ValueSnapshot(values, None, 0, 0, True, _snapshot_id(values, storage), storage)
        leaf = self.leaf(physical)
        limit = budget_bytes if budget_bytes is not None else int(self.ctx.settings.readiness.vocab_budget_bytes)
        cap = max_values if max_values is not None else 100_000
        counts: dict[str, int] = {}
        raw: dict[str, Any] = {}
        nulls = nans = 0
        complete = True
        spent = 0
        for frag, rg, arr in self.leaf_arrays(physical):
            info = self.footer(frag)
            if info is not None and leaf is not None:
                spent += info.bytes([leaf], [rg])
                if spent > limit:
                    complete = False
                    break
            while pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) \
                    or pa.types.is_fixed_size_list(arr.type):
                arr = pc.list_flatten(arr)          # a list column's values are its elements
            nulls += arr.null_count
            if pa.types.is_floating(arr.type):
                nan_mask = pc.is_nan(arr)
                nans += int(pc.sum(nan_mask).as_py() or 0)
                arr = arr.filter(pc.invert(pc.fill_null(nan_mask, True)))
            for v, n in _value_counts(arr.drop_null()):
                r = render_value(v, storage)
                if r not in raw:
                    if len(raw) >= cap:
                        complete = False
                        continue
                    raw[r] = v
                counts[r] = counts.get(r, 0) + n
        values = tuple(raw[r] for r in sorted(raw))
        return ValueSnapshot(values, {raw[r]: counts[r] for r in sorted(raw)} if _hashable(values) else None,
                             nulls, nans, complete, _snapshot_id(values, storage), storage)

    def _single_valued(self, physical: str) -> list[Any] | None:
        """Distinct values from row-group statistics when every row group has ``min == max`` and no nulls."""
        if parse_path(physical).crosses_list or physical in self.partitions:
            return None
        leaf = self.leaf(physical)
        seen: dict[str, Any] = {}
        for frag in self.fragments():
            info = self.footer(frag)
            if info is None:
                return None
            for group in info.row_groups:
                chunk = group.chunks.get(leaf or "")
                if chunk is None or not chunk.has_minmax or chunk.min != chunk.max or chunk.null_count != 0:
                    return None
                seen.setdefault(render_value(chunk.min), chunk.min)
        return list(seen.values())

    def sample_rows(self, n: int, *, seed: int = 0, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Up to ``n`` rows (items of an item table) from seeded random row groups, for size estimates and
        relation checks."""
        rng = random.Random(seed)
        plan, _ = self.plan(None, self.leaves([ALL_COLUMNS] if columns is None else columns), use_sidecars=False)
        pairs = [(frag, rg) for frag, rgs in plan for rg in (rgs if rgs is not None else [None])]
        rng.shuffle(pairs)
        out: list[dict[str, Any]] = []
        leaves = self.leaves([ALL_COLUMNS] if columns is None else [*columns, *self.key])
        for frag, rg in pairs:
            info = self.footer(frag)
            want = n - len(out)
            rows = self._sample_group(frag, rg, leaves, info, want, rng)
            if self.levels:
                rows = [_items.item_row(v, self.levels, self.key, p)
                        for r in rows for v, p in _items.explode(r, self.levels)]
            if len(rows) > n - len(out):
                rows = rng.sample(rows, n - len(out))
            out.extend(rows)
            if len(out) >= n:
                break
        return out

    def _sample_group(self, frag: Fragment, rg: int | None, leaves: Sequence[str], info: FooterInfo | None,
                      want: int, rng: random.Random) -> list[dict[str, Any]]:
        """The rows of one row group a sample of ``want`` needs, as native rows: never the whole group when it is
        larger. Every Open Targets 25.09 shard is one row group (study: 1,964,234 rows), and converting it whole
        to Python rows took 4.6 GB for a 500-row ``_stats`` sample.

        A table's rows are drawn exactly as before (``rng.sample`` over the group's row positions, then only those
        rows are converted). An item table needs whole rows to find its items: a group over
        :data:`SAMPLE_WHOLE_GROUP_ROWS` rows or :data:`SAMPLE_WHOLE_GROUP_BYTES` is converted in chunks of rows in a
        seeded order until ``want`` items are found; a smaller one is converted whole, as before."""
        rows_in = info.row_groups[rg or 0].rows if info is not None and rg is not None else None
        if rows_in is None or not leaves or rows_in <= want:
            return self._read(frag, rg, leaves, info)
        if not self.levels:
            return self._read_take(frag, rg or 0, leaves, rng.sample(range(rows_in), want))
        if rows_in <= SAMPLE_WHOLE_GROUP_ROWS and info is not None and \
                info.bytes(leaves, [rg or 0]) <= SAMPLE_WHOLE_GROUP_BYTES:
            return self._read(frag, rg, leaves, info)
        import pyarrow as pa

        tbl = self.fmt.read_leaves(frag, list(leaves), [rg])
        order = list(range(rows_in))
        rng.shuffle(order)
        out: list[dict[str, Any]] = []
        items = 0
        for i in range(0, rows_in, SAMPLE_CHUNK_ROWS):
            chunk = sorted(order[i:i + SAMPLE_CHUNK_ROWS])
            rows = self.fmt.to_native(tbl.take(pa.array(chunk, pa.int64())))
            for row in rows:
                for k, v in (frag.partition or {}).items():
                    row[k] = v
                self.clean(row)
                items += sum(1 for _ in _items.explode(row, self.levels))
            out.extend(rows)
            if items >= want:
                break
        return out


#: An item table's row group with more rows than this, or more uncompressed bytes in the leaves read, is sampled in
#: chunks of :data:`SAMPLE_CHUNK_ROWS` rows instead of converted whole. Footer bytes understate nested rows as
#: Python objects: one 25.09 expression group (11,082 genes, 11.4 MB) took 2.4 GB converted whole.
SAMPLE_WHOLE_GROUP_ROWS = 2048
SAMPLE_WHOLE_GROUP_BYTES = 16 * 1024 * 1024
SAMPLE_CHUNK_ROWS = 256
#: Rows of a row group a scan converts to Python at a time.
SCAN_CHUNK_ROWS = 65_536


def _masked(tbl: Any, index: list[int], arrow_filter: Callable[[Any], Any]) -> tuple[Any, list[int]]:
    """``tbl`` and its row ``index`` without the rows ``arrow_filter``'s mask does not keep (null: not kept)."""
    mask = arrow_filter(tbl)
    keep = mask.to_pylist()
    return tbl.filter(mask), [i for i, k in zip(index, keep) if k]


def _value_counts(arr: Any) -> list[tuple[Any, int]]:
    """``(value, count)`` pairs of an Arrow array; types without a ``value_counts`` kernel (structs of
    lists, maps) are counted on their rendered Python values."""
    import pyarrow as pa
    import pyarrow.compute as pc

    try:
        return [(item["values"], int(item["counts"])) for item in pc.value_counts(arr).to_pylist()]
    except (pa.ArrowNotImplementedError, pa.ArrowTypeError):
        first: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for v in arr.to_pylist():
            r = render_value(v)
            first.setdefault(r, v)
            counts[r] = counts.get(r, 0) + 1
        return [(first[r], counts[r]) for r in first]


class _Neg:
    """Inverts tuple order: a min-heap of these keeps the k smallest sort keys."""

    __slots__ = ("v",)

    def __init__(self, v: tuple[Any, ...]) -> None:
        self.v = v

    def __lt__(self, other: "_Neg") -> bool:
        return other.v < self.v

    def __gt__(self, other: "_Neg") -> bool:
        return self.v < other.v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Neg) and self.v == other.v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(path: str) -> list[str]:
    out: list[str] = []
    for seg in parse_path(path).segments:
        if seg.name:
            out.append(seg.name)
        out.extend("[]" for _ in seg.brackets)
    return out


def _snapshot_id(values: Sequence[Any], storage: str | None) -> str:
    import hashlib

    h = hashlib.sha256()
    for v in values:
        h.update(render_value(v, storage).encode("utf-8"))
        h.update(b"\n")
    return "vs1:" + h.hexdigest()[:16]


def _hashable(values: Sequence[Any]) -> bool:
    try:
        hash(tuple(values))
    except TypeError:
        return False
    return True


def _tracked(unknown_columns: Sequence[str] | None, conj_paths: Sequence[list[str]],
             physical: Callable[[str], str]) -> set[str] | None:
    """Columns whose unknown comparisons are attributed (all predicate columns when none are named)."""
    if unknown_columns:
        named = {physical(c) for c in unknown_columns} | set(unknown_columns)
        return {p for ps in conj_paths for p in ps if p in named}
    return {p for ps in conj_paths for p in ps}


def _interval_may_match(c: Predicate, lo: Any, hi: Any, storage: str | None) -> bool:
    """Can a row group with values in ``[lo, hi]`` hold a row for which ``c`` is true? (TypeError: undecided)."""
    def v(x: Any) -> Any:
        if isinstance(x, bool) or x is None:
            return x
        return round_to_storage(x, storage)

    if isinstance(c, Eq):
        x = v(c.value)
        if x is None:
            return False
        return bool(lo <= x <= hi)
    if isinstance(c, In):
        vals = [v(x) for x in c.values if x is not None]
        return any(lo <= x <= hi for x in vals)
    if isinstance(c, Cmp):
        x = v(c.value)
        if x is None:
            return False
        if c.op == "<":
            return bool(lo < x)
        if c.op == "<=":
            return bool(lo <= x)
        if c.op == ">":
            return bool(hi > x)
        if c.op == ">=":
            return bool(hi >= x)
        if c.op == "!=":
            return not (lo == hi == x)
        return True
    if isinstance(c, Range):
        ok = True
        if c.lo is not None:
            x = v(c.lo)
            ok = ok and bool(hi > x or (c.lo_inclusive and hi >= x))
        if c.hi is not None:
            x = v(c.hi)
            ok = ok and bool(lo < x or (c.hi_inclusive and lo <= x))
        return ok
    return True


def _clean_dict(d: dict[str, Any], tree: Sequence[Any]) -> None:
    for name, col, kids, codes, holders, conds, of in tree:
        if name not in d:
            continue
        value = d[name]
        if kids is not None:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _clean_dict(item, kids)
            elif isinstance(value, dict):
                _clean_dict(value, kids)
            continue
        if value is None:
            continue
        bad = codes                                    # rendered by _clean_tree
        if holders:
            bad = bad | {render_value(h) for h in holders}
        if of is not None and d.get(of) is not None:
            bad = bad | {render_value(d.get(of))}
        if bad:
            if isinstance(value, list):
                d[name] = [None if x is not None and render_value(x) in bad else x for x in value]
            elif render_value(value) in bad:
                d[name] = None
        if conds and d.get(name) is not None:
            # a condition whose columns were not read is not evaluated (the reader reads them with the column)
            if any(evaluate(cond, d) is not False for cond in conds
                   if all(parse_path(c).head in d for c in predicate_paths(cond))):
                d[name] = None


def _renamed(row: dict[str, Any], rename: Mapping[str, str] | None) -> dict[str, Any]:
    if not rename:
        return row
    # an item row (exploded, or a row of an item table) holds the item's own field names: a container
    # path ("screens[].geneEffect") renames that field too
    names = {**{k.rpartition("[].")[2]: v for k, v in rename.items() if "[]." in k}, **rename}
    return {names.get(k, k): v for k, v in row.items()}


def _explode_rows(row: dict[str, Any], view: Mapping[str, Any], explode: Sequence[str], carry: Sequence[str],
                  reader: TableReader) -> Iterator[dict[str, Any]]:
    if not explode:
        yield row
        return
    current = [row]
    for path in explode:
        nxt = []
        for r in current:
            items = _items.path_values(r, path if path.endswith("]") else path + "[]")
            for item in items:
                out = dict(item) if isinstance(item, Mapping) else {"value": item}
                for c in carry:
                    out.setdefault(c, _items.path_value(view, reader.physical_path(c)))
                nxt.append(out)
        current = nxt
    yield from current

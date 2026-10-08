"""Role-driven result transforms T1-T14 (§11.7). No pyarrow.

Pure functions over **logical rows** (the field-mapped views of :mod:`.fields`), each returning
the rows and its counters; the gateway applies them in this order and the witness checks
(W1-W6) run after T6:

* T1 leakage (:mod:`.leakage`); T2 in-band unknowns, NaN and placeholders become null; T3
  existence per row (``exists_when``, null non-nullable references: phantom rows, W6);
* T4 honour bound arguments (re-applied, ``binds_any`` as ``Or``, ``item_filter`` per nested item
  with ``drop_empty_parents``) and T5 unknown never passes (null beats T4: counted in
  ``excluded_unknown``, or ``excluded_not_applicable`` outside ``applies_when``);
* T6 negation (rows and nested items; ``include_negated`` keeps them);
* T7 duplicates; T8 key-completeness disclosure (``pooled_over``); T9 levels split into
  ``<level>s`` sections with one row per level key; T10 order and cut (statistic plugin keys,
  nulls last, ties by canonical key, per group, ``limit_grain``, never splitting a grain
  boundary); T11 counts, summaries, dropped and vetoed fields, ``arg_echo``; T12 ordered trims of
  nested arrays with ``_vbt.trimmed``; T13 flag partition of nested items; T14 measure validity
  (``undefined_when``, non-finite values).
"""

from __future__ import annotations

import math
import statistics as _stats
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..descriptor.columns import DEFAULT_STATISTIC
from ..errors import json_value
from ..predicate import (
    Any as AnyItem,
)
from ..predicate import (
    And,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    Not,
    Or,
    Predicate,
    Range,
    RankKey,
    TextMatch,
    columns,
    evaluate,
    facet_predicate,
    is_null,
    map_columns,
)
from ..rowkey import canonical
from .fields import concrete_paths, get_path, jp_first, jp_get, jp_set, parse_jsonpath, recount, set_path
from .leakage import withhold_rows

__all__ = [
    "Counters", "t1_leakage", "t2_unknowns", "t3_existence", "t4_honour", "t5_unknown", "honour_arguments",
    "t6_negation", "t7_duplicates", "t8_pooled_over", "t9_levels", "t10_order_cut", "t11_counts", "t12_trim",
    "t13_flag_partition", "t14_validity", "row_key", "split_container", "rank_keys", "on_item_rows", "trim_column",
]


@dataclass
class Counters:
    """Everything the transforms count, merged into the ``_vbt`` header."""

    excluded: dict[str, int] = field(default_factory=dict)
    excluded_unknown: dict[str, int] = field(default_factory=dict)
    excluded_not_applicable: dict[str, int] = field(default_factory=dict)
    excluded_negated: int = 0
    withheld: dict[str, int] = field(default_factory=dict)
    nulled: dict[str, int] = field(default_factory=dict)
    dropped_parents: int = 0
    items_removed: dict[str, int] = field(default_factory=dict)
    trimmed: dict[str, dict[str, int]] = field(default_factory=dict)
    removed_fields: dict[str, str] = field(default_factory=dict)
    leakage_unchecked: int = 0                         # rows whose dates could not be compared with the ceiling
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def bump(d: dict[str, int], key: str, n: int = 1) -> None:
        d[key] = d.get(key, 0) + n


def _row(r: Any) -> Mapping[str, Any]:
    return r if isinstance(r, Mapping) else {}


# ---------------------------------------------------------------------------
# T1, T2, T3
# ---------------------------------------------------------------------------

def t1_leakage(rows: Iterable[Any], spec: Any, ceiling: Any, counters: Counters) -> list[Any]:
    kept, counts = withhold_rows(rows, spec, ceiling)
    if counts["withheld"]:
        Counters.bump(counters.withheld, "leakage", counts["withheld"])
    counters.leakage_unchecked += counts["unchecked"]
    if counts["redacted"]:
        counters.notes.append(f"{counts['redacted']} row(s) had fields changed after the evidence ceiling redacted")
    return kept


def _null_codes(spec: Any) -> list[Any]:
    return list(getattr(spec, "missing_values", []) or [])


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _unknown_value(row: Mapping[str, Any], name: str, value: Any, spec: Any, key_value: Any) -> bool:
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is None:
        return False
    if any(_same(value, c) for c in _null_codes(spec)):
        return True
    if any(_same(value, p) for p in getattr(spec, "placeholders", []) or []):
        return True
    if getattr(spec, "placeholder_when", None) == "equals_key" and key_value is not None and value == key_value:
        return True
    for cond in getattr(spec, "unknown_when", []) or []:
        try:
            if evaluate(facet_predicate(cond, name), row) is True:
                return True
        except Exception:  # noqa: BLE001 - a malformed facet never nulls a value
            continue
    return False


def t2_unknowns(rows: Iterable[Any], columns_spec: Mapping[str, Any], counters: Counters, *,
                key_column: str | None = None, prefix: str = "") -> list[Any]:
    """In-band unknown codes, ``unknown_when`` rows, NaN and placeholders become null (in place),
    recursively into nested containers; counts in ``counters.nulled``."""
    from ..descriptor.columns import is_container
    rows = list(rows)
    for row in rows:
        if not isinstance(row, dict):
            continue
        key_value = row.get(key_column) if key_column else None
        for name, spec in columns_spec.items():
            if name not in row:
                continue
            value = row[name]
            if is_container(spec):
                items = value if isinstance(value, list) else ([value] if isinstance(value, dict) else [])
                t2_unknowns([i for i in items if isinstance(i, dict)], dict(spec.fields), counters,
                            prefix=f"{prefix}{name}[].")
                continue
            if isinstance(value, list) and getattr(spec, "role", None) not in ("vector",):
                cleaned = [None if (v is not None and _unknown_value(row, name, v, spec, key_value)) else v
                           for v in value]
                n = sum(1 for a, b in zip(value, cleaned) if a is not None and b is None)
                if n:
                    row[name] = cleaned
                    Counters.bump(counters.nulled, prefix + name, n)
                continue
            if _unknown_value(row, name, value, spec, key_value):
                row[name] = None
                Counters.bump(counters.nulled, prefix + name)
    return rows


def t3_existence(rows: Iterable[Any], exists_when: str | None, nonnull_refs: Sequence[str],
                 counters: Counters) -> tuple[list[Any], list[Any]]:
    """W6 per row: ``(kept, phantom rows)``."""
    from .fields import jp_test
    kept: list[Any] = []
    phantom: list[Any] = []
    for row in rows:
        ok = True
        if exists_when and isinstance(row, Mapping):
            try:
                ok = jp_test(row, exists_when)
            except Exception:  # noqa: BLE001
                ok = True
        if ok and isinstance(row, Mapping):
            for ref in nonnull_refs:
                if ref in row and row[ref] is None:
                    ok = False
                    break
        (kept if ok else phantom).append(row)
    if phantom:
        Counters.bump(counters.withheld, "phantom", len(phantom))
    return kept, phantom


# ---------------------------------------------------------------------------
# T4, T5
# ---------------------------------------------------------------------------

def split_container(pred: Predicate) -> tuple[str, Predicate] | None:
    """``(container column, item-relative predicate)`` of a predicate over one nested container."""
    if isinstance(pred, AnyItem):
        return pred.path.replace("[]", ""), pred.pred
    col = None
    if isinstance(pred, (Eq, In, Cmp, CmpAbs, Range, TextMatch)):
        col = pred.column
    elif isinstance(pred, Contains):
        col = pred.path
    if not col or "[]" not in col:
        return None
    head, _, rest = col.rpartition("[]")
    rest = rest.lstrip(".")
    container = head.replace("[]", "")
    if isinstance(pred, Contains):
        return container, Eq(rest or "[]", pred.value)
    inner = rest or "[]"
    if isinstance(pred, Eq):
        return container, Eq(inner, pred.value)
    if isinstance(pred, In):
        return container, In(inner, pred.values)
    if isinstance(pred, Cmp):
        return container, Cmp(inner, pred.op, pred.value)
    if isinstance(pred, CmpAbs):
        return container, CmpAbs(inner, pred.op, pred.value)
    if isinstance(pred, Range):
        return container, Range(inner, pred.lo, pred.hi, pred.lo_inclusive, pred.hi_inclusive)
    return container, TextMatch(inner, pred.text, pred.mode)  # type: ignore[union-attr]


def _applies(row: Mapping[str, Any], spec: Any) -> bool:
    aw = getattr(spec, "applies_when", None)
    if not aw:
        return True
    for col, allowed in aw.items():
        v = get_path(row, col)
        if v not in allowed:
            return False
    return True


def _first_column(pred: Predicate) -> str:
    cols = sorted(columns(pred))
    return cols[0].replace("[]", "").split(".")[-1] if cols else "?"


def on_item_rows(pred: Predicate, items_path: str, rename: Mapping[str, str] | None = None) -> Predicate:
    """``pred`` (over the physical table's column paths) for **flat item rows** of the item table at
    ``items_path`` (``tissues[]``): item columns become item-relative (``tissues[].label`` ->
    ``label``, or their ``rename``), an ``Any`` over the container becomes its body, and parent
    columns are left as they are (the caller reads them from the row's parent key)."""
    prefix = items_path.rstrip(".") + "."
    rename = dict(rename or {})

    def col(c: str) -> str:
        if c in rename:
            return rename[c]
        return c[len(prefix):] if c.startswith(prefix) else c

    if isinstance(pred, AnyItem):
        # Any(a[], Any(b[], p)) down to the items' list: the body is a predicate on the item itself
        path, body = pred.path, pred.pred
        while isinstance(body, AnyItem) and path != items_path:
            path, body = f"{path}.{body.path}", body.pred
        if path == items_path:
            full = {c: f"{items_path}.{c}" for c in columns(body)}
            return map_columns(body, lambda c: rename.get(full.get(c, c), c))
    if isinstance(pred, (And, Or)):
        return type(pred)(tuple(on_item_rows(q, items_path, rename) for q in pred.preds))
    if isinstance(pred, Not):
        return Not(on_item_rows(pred.pred, items_path, rename))
    return map_columns(pred, col)


def honour_arguments(rows: Iterable[Any], preds: Mapping[str, tuple[Predicate, Any]],
                     column_specs: Callable[[str], Any], counters: Counters, *,
                     params: Mapping[str, Any] | None = None, parents: Mapping[str, str | None] | None = None
                     ) -> list[Any]:
    """T4 + T5: re-apply each bound argument's predicate; ``preds`` is ``{arg: (predicate, binding)}``.
    Unknown (null) beats false: such rows count in ``excluded_unknown``/``excluded_not_applicable``.

    ``parents`` (rows that are items of a parent record) maps each parent column to the row field that
    holds it (the ``parent_key`` part, e.g. ``id`` -> ``/id``, since an item can have its own ``id``):
    a predicate over parent columns is evaluated on those values, never on the item's own fields; a
    parent column with no such field is not re-checked (upstream scoped the items to their parent)."""
    kept = []
    parents = dict(parents or {})
    for row in rows:
        if not isinstance(row, Mapping):
            kept.append(row)
            continue
        unknown_col: str | None = None
        false_arg: str | None = None
        drop_parent = False
        for arg, (pred, binding) in preds.items():
            if getattr(binding, "item_filter", False):
                split = split_container(pred)
                if split is not None:
                    container, inner = split
                    items = get_path(row, container)
                    if isinstance(items, list):
                        good = [i for i in items if evaluate(inner, i, params) is True]
                        removed = len(items) - len(good)
                        if removed:
                            Counters.bump(counters.items_removed, arg, removed)
                            set_path(row, container, good)  # type: ignore[arg-type]
                        if not good and getattr(binding, "drop_empty_parents", False):
                            drop_parent = True
                        continue
            target: Mapping[str, Any] = row
            if parents:
                cols = {c for c in columns(pred) if "[" not in c}
                hit = cols & parents.keys()
                if hit:
                    if any(parents[c] is None for c in hit):
                        continue                       # the item does not carry its parent's key
                    target = {**row, **{c: row.get(parents[c]) for c in hit}}  # type: ignore[arg-type]
            v = evaluate(pred, target, params)
            if v is True:
                continue
            if v is None and unknown_col is None:
                unknown_col = _first_column(pred)
            elif v is False and false_arg is None:
                false_arg = arg
        if unknown_col is not None:
            spec = column_specs(unknown_col)
            if spec is not None and not _applies(row, spec):
                Counters.bump(counters.excluded_not_applicable, unknown_col)
            else:
                Counters.bump(counters.excluded_unknown, unknown_col)
            continue
        if false_arg is not None:
            Counters.bump(counters.excluded, false_arg)
            continue
        if drop_parent:
            counters.dropped_parents += 1
            continue
        kept.append(row)
    return kept


t4_honour = honour_arguments
t5_unknown = honour_arguments


# ---------------------------------------------------------------------------
# T6, T7, T8
# ---------------------------------------------------------------------------

def _truthy_flag(v: Any) -> bool:
    return v is True or (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 1) or \
        (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes", "y", "not"))


def t6_negation(rows: Iterable[Any], negate_paths: Sequence[str], include_negated: bool,
                counters: Counters, *, drop_empty_parents: bool = False) -> list[Any]:
    """Rows (top-level qualifier) or nested items (``container[].qualifier``) whose negating
    qualifier is true are removed unless ``include_negated``. ``excluded_negated`` counts the records
    withheld: negated rows, and parents that negation alone left without items (§0 item 4, §19 CT-6:
    a pair whose every evidence record is negated has no support, whatever the tool; the argument
    flag ``drop_empty_parents`` governs only T4 item filters and is accepted here for compatibility);
    negated items of kept parents are counted in ``items_removed["negated"]``."""
    rows = list(rows)
    if include_negated or not negate_paths:
        return rows
    kept = []
    for row in rows:
        if not isinstance(row, Mapping):
            kept.append(row)
            continue
        negated = emptied = False
        for path in negate_paths:
            if "[]" in path:
                head, _, leaf = path.rpartition("[]")
                container = head.replace("[]", "")
                items = get_path(row, container)
                if isinstance(items, list):
                    good = [i for i in items
                            if not (isinstance(i, Mapping) and _truthy_flag(get_path(i, leaf.lstrip("."))))]
                    if len(good) != len(items):
                        Counters.bump(counters.items_removed, "negated", len(items) - len(good))
                        set_path(row, container, good)  # type: ignore[arg-type]
                        emptied = emptied or not good
            elif _truthy_flag(get_path(row, path)):
                negated = True
        if negated or emptied:
            counters.excluded_negated += 1
        else:
            kept.append(row)
    return kept


def t7_duplicates(rows: Iterable[Any], duplicate_cols: Sequence[str], include_duplicates: bool,
                  counters: Counters) -> list[Any]:
    rows = list(rows)
    if include_duplicates or not duplicate_cols:
        return rows
    kept = []
    for row in rows:
        if isinstance(row, Mapping) and any(_truthy_flag(get_path(row, c)) for c in duplicate_cols):
            Counters.bump(counters.withheld, "duplicates")
            continue
        kept.append(row)
    return kept


def t8_pooled_over(rows: Sequence[Any], key_columns: Sequence[str]) -> list[str]:
    """Key columns no row carries (disclosed in ``_vbt.pooled_over``)."""
    rows = [r for r in rows if isinstance(r, Mapping)]
    if not rows:
        return []
    out = []
    for k in key_columns:
        if "[]" in k or k.endswith("#"):
            continue
        top = k.split(".")[0]
        if not any(top in r for r in rows):
            out.append(k)
    return out


# ---------------------------------------------------------------------------
# T9 levels
# ---------------------------------------------------------------------------

def t9_levels(rows: Sequence[Any], levels: Mapping[str, Sequence[str]],
              level_keys: Mapping[str, Sequence[str]], counters: Counters) -> tuple[list[Any], dict[str, list[Any]],
                                                                                    dict[str, int]]:
    """Move level columns into ``<level>s`` sections with one row per level key (I15).
    Returns ``(rows, sections, {level: distinct keys})``."""
    sections: dict[str, list[Any]] = {}
    grains: dict[str, int] = {}
    rows = list(rows)
    for level, cols in levels.items():
        keys = list(level_keys.get(level) or [f"{level}Id"])
        by_key: dict[str, dict[str, Any]] = {}
        conflicts = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            kv = [row.get(k) for k in keys]
            if all(v is None for v in kv):
                continue
            label = canonical(kv)
            entry = by_key.setdefault(label, {k: row.get(k) for k in keys})
            for c in cols:
                if c not in row:
                    continue
                v = row.pop(c)
                if c in entry and entry[c] is not None and v is not None and entry[c] != v:
                    conflicts += 1
                if entry.get(c) is None:
                    entry[c] = v
        sections[f"{level}s"] = [by_key[k] for k in sorted(by_key)]
        grains[level] = len(by_key)
        if conflicts:
            counters.notes.append(f"{conflicts} {level}-level value(s) differ between rows of one {level}")
    return rows, sections, grains


# ---------------------------------------------------------------------------
# T10 order and cut
# ---------------------------------------------------------------------------

def row_key(row: Any, key_columns: Sequence[str], storage_types: Sequence[str | None] | None = None) -> str:
    return canonical([get_path(row, k) for k in key_columns], storage_types)


def rank_keys(order: Sequence[Any], table: Any, registry: Any) -> list[tuple[RankKey, Any, Any]]:
    """``(RankKey, statistic plugin, column spec)`` for each declared order entry."""
    out = []
    for o in order:
        col = getattr(o, "column", None) or (o.get("column") if isinstance(o, Mapping) else None)
        if not col:
            continue
        get = (lambda k, d=None: getattr(o, k, d)) if not isinstance(o, Mapping) else o.get
        spec = None
        if table is not None:
            spec = table.columns.get(str(col).split(".")[0])
        stat = get("statistic") or getattr(spec, "statistic", None)
        measure = getattr(spec, "role", None) in ("measure", "count") or spec is None
        if stat is None and measure:
            stat = DEFAULT_STATISTIC
        plugin = registry.find("statistic", stat) if registry is not None and stat else None
        if plugin is None and registry is not None and measure:
            plugin = registry.find("statistic", getattr(spec, "fallback", None) or DEFAULT_STATISTIC)
        rk = RankKey(column=str(col), direction=get("direction") or "desc", nulls=get("nulls") or "last",
                     statistic=stat, within=tuple(get("within") or ()))
        out.append((rk, plugin, spec))
    return out


def _sort(rows: list[Any], keys: Sequence[tuple[RankKey, Any, Any]], tie: Callable[[Any], str]) -> list[Any]:
    from ..plugins.statistics import order_rows
    if not keys:
        return sorted(rows, key=tie)
    return list(order_rows(rows, keys, tie_key=tie))


def t10_order_cut(rows: Sequence[Any], keys: Sequence[tuple[RankKey, Any, Any]], key_columns: Sequence[str],
                  limit: int | None, *, within: Sequence[str] = (), limit_grain: Sequence[str] | None = None,
                  boundary: Sequence[str] | None = None,
                  storage_types: Sequence[str | None] | None = None) -> tuple[list[Any], dict[str, Any]]:
    """Sort (declared order, nulls last, ties by canonical key) and cut to ``limit`` (per group
    when ``within``; counting ``limit_grain`` keys, each grain's best row; never splitting a
    ``boundary`` grain). Returns ``(rows, {total, returned, truncated, groups})``."""
    from .scope import per_group_cut
    rows = [r for r in rows]
    ordered = _sort(rows, keys, lambda r: row_key(r, key_columns, storage_types))
    info: dict[str, Any] = {"total": len(ordered)}
    if limit_grain:
        seen: dict[str, Any] = {}
        best = []
        for r in ordered:
            g = canonical([get_path(r, c) for c in list(within) + list(limit_grain)])
            if g not in seen:
                seen[g] = r
                best.append(r)
        ordered = best
        info["grain_total"] = len(best)
    if limit is None:
        info.update(returned=len(ordered), truncated=False)
        return ordered, info
    kept, groups = per_group_cut(ordered, list(within), limit)
    if boundary and not within and len(kept) < len(ordered) and kept:
        last = canonical([get_path(kept[-1], c) for c in boundary])
        for r in ordered[len(kept):]:
            if canonical([get_path(r, c) for c in boundary]) != last:
                break
            kept.append(r)
    info.update(returned=len(kept), truncated=len(kept) < len(ordered), groups=groups)
    return kept, info


# ---------------------------------------------------------------------------
# T11 counts and summaries
# ---------------------------------------------------------------------------

def _aggregate(values: list[Any], agg: str) -> Any:
    vals = [v for v in values if not is_null(v)]
    if agg == "count_true":
        return sum(1 for v in vals if v is True)
    if agg == "count":
        return len(vals)
    if agg == "count_distinct":
        return len({json_value(v) if not isinstance(v, (list, dict)) else repr(v) for v in vals})
    nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    if agg == "sum":
        return sum(nums)
    if agg == "mean":
        return sum(nums) / len(nums)
    if agg == "median":
        return _stats.median(nums)
    if agg == "min":
        return min(nums)
    if agg == "max":
        return max(nums)
    return None


def _value(row: Any, column: str) -> Any:
    """``column`` on a row; a container path (``screens[].geneEffect``) also as the item's own field, since
    the rows of an item table are the items."""
    v = get_path(row, column)
    if v is None and "[]." in column:
        v = get_path(row, column.rpartition("[].")[2])
    return v


def _recompute(rows: Sequence[Any], spec: Any) -> Any:
    of = getattr(spec, "of", None)
    group_by = list(getattr(spec, "group_by", []) or [])
    if not group_by:
        values = [_value(r, of) if of else 1 for r in rows]
        return _aggregate(values, spec.agg)
    groups: dict[str, list[Any]] = {}
    for r in rows:
        k = "/".join(str(_value(r, g)) for g in group_by)
        groups.setdefault(k, []).append(_value(r, of) if of else 1)
    return {k: _aggregate(v, spec.agg) for k, v in sorted(groups.items())}


def _remove(obj: Any, path: str) -> bool:
    parts = parse_jsonpath(path)
    if not parts:
        return False
    parent_path = "$" + "".join(f"[{p}]" if isinstance(p, int) else (".*" if p == "*" else f".{p}")
                                for p in parts[:-1])
    parents = jp_get(obj, parent_path) if len(parts) > 1 else [obj]
    removed = False
    for p in parents:
        if isinstance(p, dict) and parts[-1] in p:
            del p[parts[-1]]
            removed = True
    return removed


def t11_counts(obj: Any, rows: Sequence[Any], *, count_fields: Sequence[str] | Mapping[str, str] = (),
               summary_fields: Mapping[str, Any] | None = None, full_rows: Sequence[Any] | None = None,
               drop_fields: Mapping[str, str] | None = None, vetoed: Mapping[str, str] | None = None,
               arg_echo: Mapping[str, str] | None = None, args_raw: Mapping[str, Any] | None = None,
               counters: Counters, upstream_returned: int | None = None) -> Any:
    """Set count fields to returned counts, recompute or drop summaries (``full_rows`` is the
    complete matching set; None means it is not known, so recomputes are dropped), remove
    known-wrong and vetoed fields, and restore echoed arguments. Returns ``obj``.

    A list-form count field counts the rows the reply lists: with ``upstream_returned`` (the rows upstream
    listed, before the transforms) a field whose upstream value differs counts something else (the rows written
    to a file, the high-quality probes) and is left as upstream returned it, with a note."""
    if isinstance(count_fields, (list, tuple)) and upstream_returned is not None:
        kept = []
        for path in count_fields:
            v = jp_first(obj, path)
            if isinstance(v, int) and not isinstance(v, bool) and v != upstream_returned:
                counters.notes.append(f"{path} ({v}) is not the number of rows upstream listed ({upstream_returned}); "
                                      "left as returned")
                continue
            kept.append(path)
        count_fields = kept
    if isinstance(obj, dict) or isinstance(obj, list):
        recount(obj, count_fields, len(rows))
    for path, spec in (summary_fields or {}).items():
        if spec == "drop" or full_rows is None:
            if _remove(obj, path) or jp_get(obj, path):
                counters.removed_fields[path] = ("dropped: computed by upstream over a truncated or unverified "
                                                 "row set" if spec != "drop" else "dropped by the overlay")
            continue
        targets = concrete_paths(obj, path)
        wildcard = targets != [path]
        if len(targets) > 1:
            # one summary per group ([*]): the rows are not attributed to their groups here, so none is kept
            if _remove(obj, path):
                counters.removed_fields[path] = "dropped: a per-group summary the gateway cannot recompute per group"
            continue
        value = _recompute(full_rows, spec)
        if wildcard and isinstance(value, dict):
            # the one group's value (the rows need not carry the group column: they sit under it)
            value = next(iter(value.values())) if len(value) == 1 else None
        for target in targets:
            if jp_get(obj, target):
                jp_set(obj, target, value)
    for path, reason in (drop_fields or {}).items():
        if _remove(obj, path):
            counters.removed_fields[path] = reason
    for name, reason in (vetoed or {}).items():
        hit = False
        for r in rows:
            if isinstance(r, dict) and name in r:
                del r[name]
                hit = True
        if hit:
            counters.removed_fields[name] = reason
    for arg, path in (arg_echo or {}).items():
        if args_raw is not None and arg in args_raw and jp_get(obj, path):
            jp_set(obj, path, args_raw[arg])
    return obj


# ---------------------------------------------------------------------------
# T12, T13, T14
# ---------------------------------------------------------------------------

def trim_column(path: str) -> str:
    """The column a ``result.trim`` path names: ``$.descendants`` and ``descendants[]`` are ``descendants``."""
    col = path[2:] if path.startswith("$.") else path
    return col.replace("[]", "")


def t12_trim(rows: Iterable[Any], trims: Mapping[str, Any], counters: Counters, *,
             orders: Mapping[str, Sequence[Any]] | None = None,
             item_keys: Mapping[str, Sequence[str]] | None = None) -> None:
    """Trim nested arrays to ``trims[path]`` (an int or ``{max, order}``) in declared order (the
    container's ``rank``, else its item key), keeping ``{returned, total}`` per path: ``total`` is the
    list's length as stored. Paths naming the same column (``$.descendants``, ``descendants``) are trimmed
    once, by the first entry. ``order: key`` sorts items by their item key, a list of scalars by value;
    ``order: depth`` sorts items by their ``depth`` field (a list without depths keeps its key order)."""
    done: set[str] = set()
    for path, spec in trims.items():
        cap = spec if isinstance(spec, int) else getattr(spec, "max", None)
        how = "declared" if isinstance(spec, int) else getattr(spec, "order", "declared")
        col = trim_column(path)
        if cap is None or col in done:
            continue
        done.add(col)
        returned = total = 0
        for row in rows:
            items = get_path(row, col) if isinstance(row, Mapping) else None
            if not isinstance(items, list):
                continue
            total += len(items)
            if len(items) > cap:
                order = (orders or {}).get(col) if how == "declared" else None
                keys = list((item_keys or {}).get(col) or [])
                scalars = all(not isinstance(i, (Mapping, list)) for i in items)
                if order:
                    items = _sort(list(items), rank_keys(order, None, None), lambda i: row_key(i, keys))
                elif how == "depth" and not scalars:
                    items = sorted(items, key=lambda i: (_depth(i), row_key(i, keys) if keys else canonical(i)))
                elif how in ("key", "depth") and scalars:
                    items = sorted(items, key=lambda i: canonical([i]))
                elif how == "key" and keys:
                    items = sorted(items, key=lambda i: row_key(i, keys))
                set_path(row, col, items[:cap])  # type: ignore[arg-type]
                returned += cap
            else:
                returned += len(items)
        if returned < total:
            counters.trimmed[col] = {"returned": returned, "total": total}


def _depth(item: Any) -> float:
    d = item.get("depth") if isinstance(item, Mapping) else None
    return float(d) if isinstance(d, (int, float)) and not isinstance(d, bool) else math.inf


def t13_flag_partition(rows: Iterable[Any], containers: Mapping[str, str]) -> int:
    """Split nested items by a ``partition_items`` flag: ``<path>`` (true), ``<path>_not_met``
    (false), ``<path>_unknown`` (null). ``containers`` maps container paths to the flag field.
    Returns the number of rows changed."""
    n = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        for container, flag in containers.items():
            items = row.get(container)
            if not isinstance(items, list):
                continue
            met: list[Any] = []
            not_met: list[Any] = []
            unknown: list[Any] = []
            for i in items:
                v = i.get(flag) if isinstance(i, Mapping) else None
                (met if v is True else not_met if v is False else unknown).append(i)
            row[container] = met
            row[f"{container}_not_met"] = not_met
            row[f"{container}_unknown"] = unknown
            n += 1
    return n


def t14_validity(rows: Iterable[Any], measures: Mapping[str, Any], counters: Counters,
                 computed: Sequence[str] = ()) -> None:
    """Null ``undefined_when`` values and non-finite measures and computed values (excluded from ranking)."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        for name, spec in measures.items():
            if name not in row:
                continue
            v = row[name]
            undefined = False
            for cond in getattr(spec, "undefined_when", []) or []:
                try:
                    if evaluate(facet_predicate(cond, name), row) is True:
                        undefined = True
                except Exception:  # noqa: BLE001
                    continue
            if undefined or (isinstance(v, float) and not math.isfinite(v)):
                if v is not None:
                    row[name] = None
                    Counters.bump(counters.nulled, name)
        for name in computed:
            v = row.get(name)
            if isinstance(v, float) and not math.isfinite(v):
                row[name] = None
                Counters.bump(counters.nulled, name)

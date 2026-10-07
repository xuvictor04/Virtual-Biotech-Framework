"""``_serve``: rows served from the data (derived tools, repairs, native verbs; §8.2, §11.8).

Verbs (phase 1):

* ``lookup`` / ``find`` / ``members``: rows matching ``predicate``, ordered by ``order`` (else the table's
  ``rank``; nulls last, ties by the canonical key; within the ``group_by`` groups, else within the
  order's ``within`` columns, with ``limit`` per group), cut to ``limit`` (per ``limit_grain``: each
  grain's best row), one row per ``distinct`` combination, with ``explode``/``carry``/``rename``. ``nest: {group_by, items, count_as, having,
  item_filter, include_negated}`` groups rows **after** dropping negated rows and items (a ``qualifier``
  with ``effect: negate`` that is true) and items failing ``item_filter``, keeps groups whose count
  satisfies ``having`` (``min_arg`` reads the bound from an argument) and cuts the groups to ``limit``; ``split: {by, limit, values}`` returns ``{value: rows}`` with a limit per list.
  ``sections: {name: {table, verb, key, columns, order, limit, single, value}}`` are served
  per section; a section over an ``entity_detail`` table is always a list.
* ``search``: ``search_text`` against the key, label and synonym leaves, ranked by match class
  (exact > casefold > previous symbol > alias > other synonym > prefix > word > substring), then
  the table's rank, then the key; each row carries ``match`` (the class) and ``_match: {class, column,
  value, rule}`` (``synonym:<kind>`` for a whole-synonym match); broad and narrow synonyms are labelled
  ``broad_synonym``/``narrow_synonym``.
* ``count``: the total (and every declared grain's distinct count) with the unknown attribution.
* ``aggregate`` (phase 1 subset): one row per ``group_by`` value with ``count_distinct``, ``count``,
  ``first`` and ``distinct`` outputs (``gene_count`` of ``search_go_terms``), sorted by ``order``.

Verbs (phase 2, answered by the phase-2 modules):

* ``similar`` (``similar.serve_similar``): the cosine top-k around ``anchor`` (``find_similar_entities``),
  or with two ``anchors`` the cosine of the pair (``compute_entity_similarity``);
* ``compare``, or any request with ``split: {compare_with}`` (``setcompare.serve_compare``): the set
  comparison of two tables on complete keys (``compare_direct_indirect``);
* ``view``: the main record looked up like ``lookup``, with its sections.

``sections`` of every request are read one by one (``views.serve_sections``): a section whose table
cannot be read becomes ``{_vbt_unavailable, status: not_ready}`` instead of failing the call.

``served_by`` is ``derived``. Over the scan budget the response has no rows, ``total: null`` and a
``reason`` starting with ``too_large:`` (never a partial answer presented as complete). Counters that
have no field of their own (``excluded_negated``, groups ``filtered`` empty) are reported in
``sections["_excluded"]``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Mapping, Sequence

from ...descriptor.columns import is_container
from ...ipc import VERB_SERVE, ServeRequest, ServeResponse
from ...predicate import evaluate, from_json
from ...roles import parse_path
from .. import ServiceContext, ServiceError
from .. import items as _items
from ..reader import BudgetExceeded, ScanStats, TableReader, UnboundParameter
from .index_build import text_leaf

__all__ = ["serve", "search", "aggregate_rows", "VERBS", "MATCH_CLASSES"]

MATCH_CLASSES = ("exact", "casefold", "previous_synonym", "alias", "related_synonym", "broad_synonym",
                 "narrow_synonym", "prefix", "word", "substring")
_SYNONYM_CLASS = {"previous": "previous_synonym", "obsolete": "previous_synonym", "alias": "alias",
                  "exact": "alias", "related": "related_synonym", "broad": "broad_synonym",
                  "narrow": "narrow_synonym"}


def _fold(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip().casefold()


def _negate_columns(reader: TableReader) -> list[str]:
    out = []

    def walk(cols: Mapping[str, Any], prefix: str) -> None:
        for name, col in cols.items():
            path = f"{prefix}{name}"
            if getattr(col, "role", None) == "qualifier" and getattr(col, "effect", None) == "negate":
                out.append(path)
            fields = getattr(col, "fields", None)
            if fields and not prefix:
                walk(fields, "")                       # item tables see container fields at the top
    walk(reader.spec.columns, "")
    return out


def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes", "not"))


def _nest(rows: list[dict[str, Any]], spec: Mapping[str, Any], negate: Sequence[str],
          params: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], int, int]:
    """Group rows; returns (groups, n negated, n filtered). Negated rows, and negated items of a list
    field, are dropped (unless ``include_negated``), as are items failing ``item_filter`` (the argument
    predicates on the items). A group left without items is dropped: counted as negated when items
    matched the filters but every one was negated, else as filtered. ``having`` is checked on what
    remains, so a limit applied to the groups never selects a group the filters empty."""
    negated = 0
    include_negated = bool(spec.get("include_negated"))
    if not include_negated:
        kept = []
        for r in rows:
            if any(_truthy(r.get(c)) for c in negate if c in r):
                negated += 1
                continue
            kept.append(r)
        rows = kept
    item_preds = [from_json(p) for p in spec.get("item_filter") or []]
    group_by = list(spec.get("group_by") or [])
    items = spec.get("items") or "items"
    name = items if isinstance(items, str) else str(items.get("name", "items"))
    cols = None if isinstance(items, str) else list(items.get("columns") or []) or None
    count_as = spec.get("count_as")
    groups: dict[str, dict[str, Any]] = {}
    matched: dict[str, bool] = {}                      # a group had an item passing the filters
    dropped: dict[str, bool] = {}                      # a group lost items to the filters or negation
    for r in rows:
        gkey = json.dumps([r.get(g) for g in group_by], default=str)
        g = groups.get(gkey)
        if g is None:
            g = groups[gkey] = {c: r.get(c) for c in group_by}
            g[name] = []
        if cols is None and isinstance(r.get(name), list):
            for it in r[name]:                         # items is a list field of the rows: merge the lists
                if item_preds and not all(evaluate(p, it, params) is True for p in item_preds):
                    dropped[gkey] = True
                    continue
                matched[gkey] = True
                if not include_negated and isinstance(it, Mapping) and \
                        any(_truthy(it.get(c)) for c in negate if c in it):
                    dropped[gkey] = True
                    continue
                g[name].append(it)
            continue
        if item_preds and not all(evaluate(p, r, params) is True for p in item_preds):
            dropped[gkey] = True                       # the rows are the items (an item table)
            continue
        matched[gkey] = True
        item = {k: v for k, v in r.items() if k not in group_by} if cols is None else {c: r.get(c) for c in cols}
        g[name].append(item)
    out = []
    filtered = 0
    having = _resolve_having(spec.get("having") or {}, params or {})
    for gkey, g in groups.items():
        n = len(g[name])
        if count_as:
            g[str(count_as)] = n
        if n == 0 and dropped.get(gkey):
            if matched.get(gkey):
                negated += 1                           # every matching item was negated
            else:
                filtered += 1                          # no item passed the filters
            continue
        if not _having_ok(n, having):
            continue
        out.append(g)
    return out, negated, filtered


def _resolve_having(having: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """``{min_arg: <argument>}`` (and ``max_arg``) take their bound from the call's argument; unset, no bound."""
    out = {k: v for k, v in having.items() if k not in ("min_arg", "max_arg")}
    for k, bound in (("min_arg", "min"), ("max_arg", "max")):
        arg = having.get(k)
        v = params.get(str(arg)) if arg else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[bound] = v
    return out


def _having_ok(n: int, having: Mapping[str, Any]) -> bool:
    for k, v in having.items():
        if isinstance(v, Mapping):
            for op, x in v.items():
                if not {"ge": n >= x, "gt": n > x, "le": n <= x, "lt": n < x, "eq": n == x}.get(op, True):
                    return False
        elif k in ("min", "min_items") and n < int(v):
            return False
        elif k in ("max", "max_items") and n > int(v):
            return False
    return True


def _split(rows: list[dict[str, Any]], spec: Mapping[str, Any], per_list: int | None = None
           ) -> dict[str, list[dict[str, Any]]]:
    """Rows into named lists: ``by_sign`` (``"+"``/``"-"`` by the sign of a column; zero and unknown values
    go to neither list) or ``by`` (the column's values name the lists). ``per_list`` (the request limit)
    or ``spec.limit`` caps each list."""
    sign = spec.get("by_sign")
    if sign:
        out_sign: dict[str, list[dict[str, Any]]] = {"+": [], "-": []}
        for r in rows:
            v = r.get(sign)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v == 0:
                continue
            bucket = out_sign["+" if v > 0 else "-"]
            if per_list is None or len(bucket) < int(per_list):
                bucket.append(r)
        return out_sign
    by = spec.get("by") or spec.get("column")
    if not by:
        raise ServiceError("split needs `by_sign` or `by` (the column whose values name the lists)")
    limit = spec.get("limit", per_list)
    values = spec.get("values")
    out: dict[str, list[dict[str, Any]]] = {str(v): [] for v in values} if values else {}
    for r in rows:
        key = str(r.get(by))
        if values and key not in out:
            continue
        bucket = out.setdefault(key, [])
        cap = limit.get(key) if isinstance(limit, Mapping) else limit
        if cap is None or len(bucket) < int(cap):
            bucket.append(r)
    return out


def search(reader: TableReader, text: str, predicate: Any, limit: int | None, params: Mapping[str, Any],
           budget: int | None) -> tuple[list[dict[str, Any]], list[list[Any]], int]:
    """Rows ranked by match class, then the table's rank, then the key; ``(rows, keys, total matches)``."""
    phys: list[tuple[str, str]] = [(k, "key") for k in reader.key if not k.endswith("#")]

    def walk(cols: Mapping[str, Any], prefix: str) -> None:
        for name, col in cols.items():
            dotted = f"{prefix}{name}"
            if getattr(col, "role", None) in ("label", "synonym"):
                phys.extend(text_leaf(reader, dotted, col))
            elif is_container(col):
                walk(col.fields, dotted + ".")

    if reader.levels:
        walk(reader._level_fields()[-1], ".".join(n for lvl in reader.levels for n in lvl.names) + ".")
    else:
        walk(reader.spec.columns, "")
    needle = _fold(text)
    word = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
    okeys = reader.order_keys([r.model_dump() for r in reader.table.spec.rank])
    scored = []
    stats = ScanStats()
    for m in reader.scan(predicate, columns=[p for p, _ in phys] + [p for *_, p in okeys], params=params,
                         budget_bytes=budget, stats=stats, attribute_unknown=False):
        best: tuple[int, str, str, Any] | None = None
        for path, kind in phys:
            for v in _items.path_values(m.row, path):
                if not isinstance(v, (str, int)) or isinstance(v, bool):
                    continue
                s = str(v)
                f = _fold(s)
                if kind in ("key", "label"):
                    cls = "exact" if s == text else ("casefold" if f == needle else None)
                else:
                    cls = _SYNONYM_CLASS.get(kind, "alias") if f == needle else None
                if cls is None:
                    if f.startswith(needle):
                        cls = "prefix"
                    elif word.search(f):
                        cls = "word"
                    elif needle in f:
                        cls = "substring"
                if cls is None:
                    continue
                rank = MATCH_CLASSES.index(cls)
                if best is None or rank < best[0]:
                    best = (rank, cls, path, s, f"synonym:{kind}" if kind not in ("key", "label") and
                            f == needle else cls)
        if best is None:
            continue
        ckey = reader.canonical_key(m.key)
        scored.append((best[0], reader.sort_key(m.row, okeys, ckey), ckey, m, best))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    total = len(scored)
    if limit is not None:
        scored = scored[: int(limit)]
    rows, keys = [], []
    for _r, _sk, ckey, m, best in scored:
        row = reader.output_row(m)
        row["match"] = best[1]                  # §10.6: each row carries its match class
        row["_match"] = {"class": best[1], "column": best[2], "value": best[3], "rule": best[4]}
        rows.append(row)
        keys.append(json.loads(ckey))
    return rows, keys, total


def _grain_counts(reader: TableReader, pred: Any, params: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
                  ) -> dict[str, dict[str, int | None]]:
    grains = dict(reader.table.spec.grains)
    if not grains:
        return {}
    try:
        totals = reader.distinct_counts(pred, grains, params=params)
    except (BudgetExceeded, ServiceError):
        totals = {}
    out: dict[str, dict[str, int | None]] = {}
    for name, g in grains.items():
        cols = g if isinstance(g, list) else list(g.columns) + list(g.unordered) + list(g.by)
        names = [parse_path(c).segments[-1].name if not reader.levels else c for c in cols]
        returned = {json.dumps([r.get(n, r.get(c)) for n, c in zip(names, cols)], default=str, sort_keys=True)
                    for r in rows}
        out[name] = {"returned": len(returned) if rows else 0, "total": totals.get(name)}
    return out


_AGGREGATES = ("count_distinct", "count", "first", "distinct")


def aggregate_rows(reader: TableReader, req: ServeRequest, params: Mapping[str, Any],
                   order: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """One row per ``group_by`` value with the ``aggregate`` outputs (phase 1: ``count_distinct``, ``count``,
    ``first``, ``distinct``), sorted by ``order`` (nulls last) and renamed; ``(rows, groups)``."""
    specs: dict[str, tuple[str, str]] = {}
    for name, spec in req.aggregate.items():
        if len(spec) != 1 or next(iter(spec)) not in _AGGREGATES:
            raise ServiceError(f"aggregate {name}: one of {', '.join(_AGGREGATES)} per output (got {dict(spec)})")
        fn, col = next(iter(spec.items()))
        specs[name] = (fn, col)
    cols = list(dict.fromkeys([*req.group_by, *(c for _, c in specs.values())]))
    rows, _keys, _stats = reader.rows(req.predicate, cols, [], None, None, params=params,
                                      budget_bytes=req.budget_bytes)

    def canon(v: Any) -> str:
        return json.dumps(v, default=str, sort_keys=True)

    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        k = tuple(canon(r.get(c)) for c in req.group_by)
        g = groups.setdefault(k, {"row": {c: r.get(c) for c in req.group_by}, "vals": {n: [] for n in specs}})
        for n, (_fn, c) in specs.items():
            g["vals"][n].append(r.get(c))
    out: list[dict[str, Any]] = []
    for g in groups.values():
        row = dict(g["row"])
        for n, (fn, _c) in specs.items():
            vals = [v for v in g["vals"][n] if v is not None]
            if fn == "count_distinct":
                row[n] = len({canon(v) for v in vals})
            elif fn == "count":
                row[n] = len(vals)
            elif fn == "first":
                row[n] = vals[0] if vals else None
            else:
                row[n] = [json.loads(v) for v in sorted({canon(v) for v in vals})]
        out.append(row)
    for o in reversed(list(order)):                   # stable sorts, last key first
        col, desc = o.get("column"), str(o.get("direction", "asc")).startswith("desc")
        present = [r for r in out if r.get(col) is not None]
        missing = [r for r in out if r.get(col) is None]
        present.sort(key=lambda r: (canon(r.get(col)) if not isinstance(r.get(col), (int, float)) else r.get(col)),
                     reverse=desc)
        out = present + missing                       # nulls last
    if req.rename:
        out = [{req.rename.get(k, k): v for k, v in r.items()} for r in out]
    return out, len(groups)


def serve(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = ServeRequest.model_validate(dict(payload))
    if req.verb == "similar":
        from .similar import serve_similar

        return serve_similar(ctx, req.model_dump(mode="json"))
    if req.verb == "compare" or (req.split or {}).get("compare_with"):
        from .setcompare import serve_compare

        return serve_compare(ctx, req.model_dump(mode="json"))
    reader = ctx.reader(req.table)
    params = dict(req.params)
    order = [o.model_dump() for o in req.order]
    try:
        if req.verb == "count":
            total, eu, ena, _ut = reader.count(req.predicate, params=params, budget_bytes=req.budget_bytes)
            grains = {name: {"returned": None, "total": n}
                      for name, n in reader.distinct_counts(req.predicate, dict(reader.table.spec.grains),
                                                            params=params).items()}
            resp = ServeResponse(rows=[], total=total, key_columns=list(reader.key), grains=grains,
                                 excluded_unknown=eu, excluded_not_applicable=ena)
            return resp.model_dump(mode="json")
        if req.verb == "search":
            if not req.search_text:
                raise ServiceError("search needs search_text")
            rows, keys, total = search(reader, req.search_text, req.predicate, req.limit, params, req.budget_bytes)
            resp = ServeResponse(rows=rows, total=total, truncated=total > len(rows), key_columns=list(reader.key),
                                 row_keys=keys)
            return resp.model_dump(mode="json")
        if req.verb == "aggregate":
            rows, total = aggregate_rows(reader, req, params, order)
            out = rows[: int(req.limit)] if req.limit is not None else rows
            resp = ServeResponse(rows=out, total=total, truncated=total > len(out),
                                 key_columns=[req.rename.get(c, c) for c in req.group_by])
            return resp.model_dump(mode="json")
        if req.verb not in ("lookup", "find", "members", "view"):
            raise ServiceError(f"_serve verb {req.verb!r} arrives in a later phase (lookup, find, search, members, "
                               "count, aggregate, similar, compare, view)")
        stats = ScanStats()
        limit = None if (req.nest or req.split) else req.limit   # nest and split limit per group / list
        rows, keys, stats = reader.rows(req.predicate, req.columns, order, limit, req.limit_grain,
                                        explode=req.explode, carry=req.carry, rename=req.rename, params=params,
                                        budget_bytes=req.budget_bytes, stats=stats, group_by=req.group_by,
                                        distinct=req.distinct)
        sections: dict[str, Any] = {}
        total = stats.total
        truncated = total > len(rows)
        out_rows: Any = rows
        nest_grains: dict[str, dict[str, int | None]] = {}
        if req.nest:
            nest = dict(req.nest)
            group_by = list(nest.get("group_by") or [])
            nest["group_by"] = [req.rename.get(c, c) for c in group_by]   # rows are renamed
            nested, negated, filtered = _nest(rows, nest, _negate_columns(reader), params)
            total = len(nested)
            truncated = req.limit is not None and total > int(req.limit)
            out_rows = nested[: int(req.limit)] if req.limit is not None else nested
            keys = []
            sections["_excluded"] = {"negated": negated, "filtered": filtered}
            # the grain the nest groups by is counted exactly: one row per qualifying group
            for name, g in reader.table.spec.grains.items():
                cols = g if isinstance(g, list) else list(g.columns) + list(g.unordered) + list(g.by)
                if cols and sorted(cols) == sorted(group_by):
                    nest_grains[name] = {"returned": len(out_rows), "total": total}
        if req.split:
            out_rows = _split(out_rows, req.split, req.limit)
            truncated = sum(len(v) for v in out_rows.values()) < len(rows)
        if req.sections:
            from .views import serve_sections

            sections.update(serve_sections(ctx, req.sections, params))
        grains = _grain_counts(reader, req.predicate, params, rows) if not req.nest else nest_grains
        resp = ServeResponse(rows=out_rows, total=total, truncated=truncated, key_columns=list(reader.key),
                             grains=grains, excluded_unknown=dict(stats.excluded_unknown),
                             excluded_not_applicable=dict(stats.excluded_not_applicable), served_by="derived",
                             sections=sections, row_keys=keys)
        return resp.model_dump(mode="json")
    except BudgetExceeded as exc:
        return ServeResponse(rows=[], total=None, truncated=True, key_columns=list(reader.key),
                             reason=f"too_large: {exc.reason}").model_dump(mode="json")
    except UnboundParameter as exc:
        raise ServiceError(str(exc)) from None


VERBS = {VERB_SERVE: serve}

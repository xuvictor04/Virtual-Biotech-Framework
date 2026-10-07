"""``_compare``: full-key set comparison of two tables under one filter (§10.6; replaces
``compare_direct_indirect``, defect OT-ASSOC-004).

Upstream compared the top-``limit`` prefixes of the direct and indirect association tables, so
"direct only" depended on the limit, not on the data. Here both key sets are read **untruncated**
(every row matching the filter, keyed by ``keys``; YAML reads a key named ``on`` as true), the counts are computed on the complete sets, and
``limit`` bounds only the rows returned per part (each part ranked by the tables' rank, ties by key).

Request (hidden verb)::

    {left, right, where?, keys?, limit?, names?: {left, right, left_only, right_only, both}}

Through the gateway's ``_serve`` (derived bindings) the request is a ``find`` on the left table with
``split: {compare_with: <right table>, keys?, counts?: <section name>, count_names?: {...}}``: rows come
back as ``{left_only, right_only, both}`` lists (the binding's ``split.into`` places them) and the
section named by ``counts`` receives the counts (``count_names`` renames them, e.g. to upstream's
``unique_to_direct_count``). A predicate column the right table lacks is an error, never dropped.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...ipc import ServeResponse
from ...predicate import Predicate, columns as predicate_columns, from_json
from ...result import inject_header
from ...rowkey import canonical
from .. import ServiceContext, ServiceError
from .public import LongView, _invalid, _limit, compile_where, guarded, header, long_view, order_rows, table_access

__all__ = ["compare_sets", "serve_compare", "compare", "PARTS", "VERBS"]

PARTS = ("left_only", "right_only", "both")
COUNTS = {"left": "left_count", "right": "right_count", "left_only": "left_only_count",
          "right_only": "right_only_count", "both": "both_count"}


def compare_sets(left: LongView, right: LongView, pred: Predicate | None, on: Sequence[str], limit: int | None, *,
                 budget: int | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """``({left_only, right_only, both}: rows cut to limit, {left, right, left_only, right_only, both}: counts)``
    over the complete key sets."""
    missing = [c for c in (predicate_columns(pred) if pred is not None else set()) | set(on)
               if c.split(".")[0].lstrip("/") not in right.columns]
    if missing:
        raise ServiceError(f"{right.ref} has no column {', '.join(sorted(missing))}: the filter cannot apply to both")
    lrows, _lt, _le = left.rows(pred, limit=None, order=[], budget=budget)
    rrows, _rt, _re = right.rows(pred, limit=None, order=[], budget=budget)
    lkeys = {canonical([r.get(c) for c in on]): r for r in lrows}
    rkeys = {canonical([r.get(c) for c in on]): r for r in rrows}
    parts = {"left_only": [lkeys[k] for k in lkeys.keys() - rkeys.keys()],
             "right_only": [rkeys[k] for k in rkeys.keys() - lkeys.keys()],
             "both": [lkeys[k] for k in lkeys.keys() & rkeys.keys()]}
    counts = {"left": len(lkeys), "right": len(rkeys), **{p: len(v) for p, v in parts.items()}}
    out = {}
    for p, rows in parts.items():
        view = right if p == "right_only" else left
        ordered = order_rows(rows, view.default_order(), on)
        out[p] = ordered[: int(limit)] if limit is not None else ordered
    return out, counts


def compare(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The hidden verb: compare two tables' key sets under one ``where``."""
    lt = table_access(ctx, payload.get("left"), agent=payload.get("agent"), argument="left")
    rt = table_access(ctx, payload.get("right"), agent=payload.get("agent"), argument="right")
    left, right = long_view(ctx, str(lt.ref)), long_view(ctx, str(rt.ref))
    on = list(payload.get("keys") or left.key)
    bad = [c for c in on if c not in left.columns or c not in right.columns]
    if bad:
        raise _invalid("keys", bad[0], "keys names key columns both tables have", sorted(set(left.columns) &
                                                                                     set(right.columns)))
    notes: list[str] = []
    resolved: dict[str, str] = {}
    pred, _keys = compile_where(left, payload.get("where"), notes=notes, resolved=resolved)
    limit = _limit(payload, 50, 1000)
    parts, counts = compare_sets(left, right, pred, on, limit, budget=payload.get("budget_bytes"))
    names = {**COUNTS, **dict(payload.get("names") or {})}
    rows = [dict(r, comparison=p) for p in PARTS for r in parts[p]]
    notes.append("counts are computed on the complete key sets; limit bounds only the rows returned per part")
    hdr = header(left, rows=rows, total=counts["left"] + counts["right_only"], truncated=any(
        len(parts[p]) < counts[p] for p in PARTS), key=on, resolved=resolved, notes=notes,
        extra={"tables": [left.ref, right.ref]})
    return inject_header({"counts": {names[k]: v for k, v in counts.items()}, "rows": rows}, hdr)


def serve_compare(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """The ``_serve`` form (``split: {compare_with}``) used by derived bindings."""
    split = dict(req.get("split") or {})
    left = long_view(ctx, str(req.get("table")))
    right = long_view(ctx, str(split["compare_with"]))
    on = list(split.get("keys") or left.key)
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    parts, counts = compare_sets(left, right, pred, on, req.get("limit"), budget=req.get("budget_bytes"))
    names = {**COUNTS, **dict(split.get("count_names") or {})}
    sections: dict[str, Any] = {}
    target = split.get("counts")
    for name in (req.get("sections") or {}):
        if name == target:
            sections[name] = {names[k]: v for k, v in counts.items()}
    if target and target not in sections:
        sections[str(target)] = {names[k]: v for k, v in counts.items()}
    total = counts["left"] + counts["right_only"]
    returned = sum(len(v) for v in parts.values())
    resp = ServeResponse(rows={p: parts[p] for p in PARTS}, total=total, truncated=returned < total,
                         key_columns=on, sections=sections, served_by="derived",
                         row_keys=[[r.get(c) for c in on] for p in PARTS for r in parts[p]])
    return resp.model_dump(mode="json")


VERBS = {"_compare": guarded("compare", compare)}

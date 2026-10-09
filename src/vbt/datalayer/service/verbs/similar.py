"""``similar`` / ``_similar``: cosine top-k over a vectors table (§10.6, rev 2; S12 ``literature_vector``).

Request: ``{table, anchor, where?, top_k?, agent?}``. The anchor resolves like any key argument (the
key column's id_type and every kind with an edge to it), and:

* the anchor is excluded from the candidates **and** from ``_vbt.total``;
* an anchor that is not a row of the table is ``not_found``; an anchor whose vector has norm 0 (or a
  non-finite component) has no defined similarity: ``invalid_argument`` with ``subkind: undefined``;
* a candidate whose score is not finite (a zero-norm or non-finite vector) is excluded and counted in
  ``_vbt.excluded_unknown.similarity``, never sorted somewhere arbitrary;
* rows are ordered by similarity descending, ties by the canonical key; ``top_k`` cuts them and
  ``_vbt.total`` counts every candidate with a finite score.

``serve_similar`` answers the gateway's ``_serve`` requests with ``verb: similar`` (derived serving and
repairs of ``find_similar_entities``): ``anchor: {column, value}`` already resolved, the predicate in
its JSON form, ``limit`` as ``top_k``. With two ``anchors`` (``[{name, column, value}]``, every anchor
argument of the call) it is the cosine of the pair (repairs of ``compute_entity_similarity``): one row
with each anchor under its argument name, the request's ``columns`` of each as ``<name>_<column>`` and
``similarity``; a missing anchor gives no row, a zero-norm one ``invalid_argument`` (``undefined``).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ...errors import ErrorKind, GatewayError, json_value
from ...ipc import VERB_SIMILAR, ServeResponse
from ...predicate import Eq, Predicate, from_json
from ...result import inject_header
from ...rowkey import canonical
from .. import ServiceContext, ServiceError
from .public import LongView, _invalid, _limit, compile_where, guarded, header, key_label, long_view, table_access

__all__ = ["similar", "serve_similar", "cosine_ranking", "pair_similarity", "vector_column", "VERBS"]


def vector_column(view: LongView) -> str:
    cols = [n for n, c in view.columns.items() if getattr(c, "role", None) == "vector"]
    if not cols:
        raise ServiceError(f"{view.ref} has no vector column")
    return cols[0]


def _vec(v: Any) -> list[float] | None:
    if v is None:
        return None
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    return out


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_ranking(view: LongView, anchor: Any, pred: Predicate | None, k: int | None, *,
                   columns: Sequence[str] = (), budget: int | None = None
                   ) -> tuple[list[dict[str, Any]], int, int, bool]:
    """``(top rows with 'similarity', total finite candidates, non-finite candidates, anchor found)``."""
    vcol = vector_column(view)
    key = view.key[0]
    anchor_rows, _t, _e = view.rows(Eq(key, anchor), columns=[key, vcol], limit=1, budget=budget)
    if not anchor_rows:
        return [], 0, 0, False
    a = _anchor_vector(anchor, anchor_rows[0], vcol)
    na = _norm(a)
    out_cols = list(columns) or [c for c in view.columns if c != vcol]
    rows, _total, _eu = view.rows(pred, columns=list(dict.fromkeys([*out_cols, key, vcol])), order=[], limit=None,
                                  budget=budget)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    bad = 0
    for r in rows:
        if r.get(key) == anchor:
            continue                                   # the anchor is never its own neighbour, nor counted
        v = _vec(r.get(vcol))
        score = float("nan")
        if v is not None and len(v) == len(a):
            nv = _norm(v)
            if nv > 0 and all(math.isfinite(x) for x in v):
                score = sum(x * y for x, y in zip(a, v)) / (na * nv)
        if not math.isfinite(score):
            bad += 1
            continue
        out = {c: r.get(c) for c in out_cols if c != vcol}
        out["similarity"] = round(score, 12)
        scored.append((score, canonical([r.get(c) for c in view.key]), out))
    scored.sort(key=lambda x: (-x[0], x[1]))
    total = len(scored)
    if k is not None:
        scored = scored[: int(k)]
    return [s[2] for s in scored], total, bad, True


def _anchor_vector(anchor: Any, row: Mapping[str, Any], vcol: str) -> list[float]:
    a = _vec(row.get(vcol))
    if a is None or not all(math.isfinite(x) for x in a) or _norm(a) == 0:
        raise GatewayError(ErrorKind.invalid_argument, f"the anchor {anchor!r} has no defined similarity (its vector "
                           "has norm 0 or non-finite components)", argument="anchor", value=anchor,
                           subkind="undefined")
    return a


def pair_similarity(view: LongView, anchors: Sequence[Mapping[str, Any]], *, columns: Sequence[str] = (),
                    budget: int | None = None) -> dict[str, Any] | None:
    """The cosine of the first two ``anchors`` (``{name, value}``) as one row, None when one has no row."""
    vcol = vector_column(view)
    key = view.key[0]
    extra = [c for c in columns if c not in (key, vcol)]
    out: dict[str, Any] = {}
    vectors = []
    for a in anchors[:2]:
        rows, _t, _e = view.rows(Eq(key, a["value"]), columns=[key, vcol, *extra], limit=1, budget=budget)
        if not rows:
            return None
        vectors.append(_anchor_vector(a["value"], rows[0], vcol))
        name = str(a.get("name") or f"entity_{len(vectors)}")
        out[name] = rows[0].get(key)
        out.update({f"{name}_{c}": rows[0].get(c) for c in extra})
    x, y = vectors
    if len(x) != len(y):
        raise ServiceError(f"{view.ref}: the anchors' vectors differ in length ({len(x)} and {len(y)})")
    out["similarity"] = round(sum(p * q for p, q in zip(x, y)) / (_norm(x) * _norm(y)), 12)
    return out


def similar(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Cosine top-k (see the module docstring)."""
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    try:
        vector_column(view)
    except ServiceError:
        raise _invalid("table", view.ref, f"{view.ref} has no vector column") from None
    anchor = payload.get("anchor")
    if anchor in (None, ""):
        raise _invalid("anchor", anchor, "anchor names the reference entity")
    notes: list[str] = []
    resolved: dict[str, str] = {}
    _p, keys = compile_where(view, {view.key[0]: anchor}, argument="anchor", notes=notes, resolved=resolved)
    canon = (keys.get(view.key[0]) or [anchor])[0]
    pred, _k = compile_where(view, payload.get("where"), notes=notes, resolved=resolved)
    k = _limit(payload, 10, 1000, "top_k")
    rows, total, bad, found = cosine_ranking(view, canon, pred, k, budget=payload.get("budget_bytes"))
    if not found:
        raise GatewayError(ErrorKind.not_found, f"anchor={anchor!r} has no vector in {view.ref}", argument="anchor",
                           value=anchor, payload={"table": view.ref})
    notes.append("the anchor is excluded from the results and from the total")
    hdr = header(view, rows=rows, total=total, truncated=total > len(rows), order="similarity desc, ties by key",
                 resolved=resolved, excluded_unknown={"similarity": bad}, notes=notes,
                 extra={"anchor": json_value(canon)})
    return inject_header({"rows": json_value(rows)}, hdr)


def serve_similar(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """The ``_serve`` form (``verb: similar``) used by derived bindings."""
    view = long_view(ctx, str(req.get("table")))
    anchors = [a for a in req.get("anchors") or [] if isinstance(a, Mapping) and a.get("value") is not None]
    if len(anchors) >= 2:
        row = pair_similarity(view, anchors, columns=list(req.get("columns") or []), budget=req.get("budget_bytes"))
        if row is None:
            return ServeResponse(rows=[], total=0, key_columns=[key_label(k) for k in view.key],
                                 reason="an anchor has no vector").model_dump(mode="json")
        return ServeResponse(rows=[json_value(row)], total=1, key_columns=[str(a.get("name")) for a in anchors[:2]],
                             served_by="derived").model_dump(mode="json")
    anchor = req.get("anchor") or {}
    if not isinstance(anchor, Mapping) or anchor.get("value") is None:
        raise ServiceError("similar needs anchor {column, value}")
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    rows, total, bad, found = cosine_ranking(view, anchor["value"], pred, req.get("limit"),
                                             columns=list(req.get("columns") or []), budget=req.get("budget_bytes"))
    if not found:
        return ServeResponse(rows=[], total=0, key_columns=[key_label(k) for k in view.key],
                             reason=f"anchor {anchor['value']!r} has no vector").model_dump(mode="json")
    keys = [[r.get(c) for c in view.key] for r in rows]
    resp = ServeResponse(rows=json_value(rows), total=total, truncated=total > len(rows), key_columns=[key_label(k) for k in view.key],
                         excluded_unknown={"similarity": bad} if bad else {}, served_by="derived",
                         row_keys=json_value(keys))
    return resp.model_dump(mode="json")


VERBS = {VERB_SIMILAR: guarded("similar", similar)}

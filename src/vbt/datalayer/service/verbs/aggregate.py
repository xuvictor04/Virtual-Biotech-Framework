"""``aggregate`` / ``_aggregate``: grouped aggregation over a table's long view (§10.6, rev 2).

Works on facts, item tables and matrix long views (whose row attributes are joined through
``attributes_from``, so DepMap gene effects aggregate by ``OncotreeLineage``). Request::

    {table, where?, group_by: [column, ...], measure?, how: count | count_distinct | <aggregation>,
     min_n?, limit?, agent?}

* ``count`` counts rows per group; ``count_distinct`` counts distinct ``measure`` values;
* any other ``how`` (``mean``, ``median``, ``min``, ``max``, ``sum``, ...) is the measure column's
  statistic plugin's aggregation (``fallback: numeric`` when the declared statistic is not registered),
  so an aggregation the statistic forbids is ``invalid_argument`` and values comparable only within a
  scope are never pooled;
* unknown measure values (null, NaN, in-band codes the plugin knows) are excluded and counted, per
  group (``n_unknown``) and in ``_vbt.excluded_unknown``; every group reports its ``n`` (the known
  values aggregated);
* groups with ``n < min_n`` are dropped and counted (``_vbt.extra.groups_below_min_n``), never shown
  with a value computed from too few observations;
* groups are ordered by the aggregate (descending; ``count`` and ``count_distinct`` too), then by the
  group key; ``limit`` cuts the groups and ``_vbt.total`` counts them all. A group whose key is
  unknown is listed with a null key (rows are not silently pooled into another group).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ...errors import ErrorKind, GatewayError, invalid_argument_payload, json_value
from ...ipc import VERB_AGGREGATE
from ...result import inject_header
from .. import ServiceContext
from .public import _invalid, _limit, compile_where, guarded, header, long_view, table_access

__all__ = ["aggregate", "group_rows", "VERBS"]

_COUNTING = ("count", "count_distinct")


def _plugin(ctx: ServiceContext, spec: Any) -> Any:
    name = getattr(spec, "statistic", None)
    plugin = ctx.statistic(name) if name else None
    if plugin is None and (getattr(spec, "fallback", None) == "numeric" or name is None):
        plugin = ctx.statistic("numeric")
    return plugin


def group_rows(rows: list[Mapping[str, Any]], group_by: list[str]) -> dict[str, dict[str, Any]]:
    """``{canonical group key: {"key": {...}, "rows": [...]}}`` in first-seen order."""
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = {c: r.get(c) for c in group_by}
        k = json.dumps([json_value(key[c]) for c in group_by], sort_keys=True, default=str)
        groups.setdefault(k, {"key": key, "rows": []})["rows"].append(r)
    return groups


def aggregate(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Grouped aggregation (see the module docstring)."""
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    group_by = payload.get("group_by") or []
    if isinstance(group_by, str):
        group_by = [group_by]
    if not isinstance(group_by, list) or not group_by:
        raise _invalid("group_by", group_by, "group_by names at least one long-view column", sorted(view.columns))
    for c in group_by:
        if c not in view.columns:
            raise _invalid("group_by", c, f"{view.ref} has no column {c!r}", sorted(view.columns))
    how = str(payload.get("how") or "count")
    measure = payload.get("measure")
    spec = view.column(measure) if measure else None
    if how != "count":
        if measure is None or spec is None:
            raise _invalid("measure", measure, f"how={how} needs a measure column of {view.ref}",
                           [n for n, c in view.columns.items() if getattr(c, "role", None) in ("measure", "count")])
    if how not in _COUNTING and getattr(spec, "role", None) not in ("measure", "count"):
        raise _invalid("measure", measure, f"{measure} is not a measure or count column",
                       [n for n, c in view.columns.items() if getattr(c, "role", None) in ("measure", "count")])
    min_n = payload.get("min_n", 1)
    if isinstance(min_n, bool) or not isinstance(min_n, int) or min_n < 1:
        raise GatewayError(ErrorKind.invalid_argument, "min_n is a positive integer",
                           payload=invalid_argument_payload("min_n", min_n, None))
    limit = _limit(payload, 100, 10_000)
    notes: list[str] = []
    resolved: dict[str, str] = {}
    pred, keys = compile_where(view, payload.get("where"), notes=notes, resolved=resolved)
    col_keys = row_keys = None
    if view.matrix is not None:
        col_keys = [str(k) for k in keys.get(view.col_key[0], [])] or None
        row_keys = [str(k) for k in keys.get(view.row_key[0], [])] or None
    columns = list(dict.fromkeys([*group_by, *([measure] if measure else [])]))
    rows, total_rows, eu = view.rows(pred, columns=columns, order=[], limit=None, budget=payload.get("budget_bytes"),
                                     col_keys=col_keys, row_keys=row_keys)
    plugin = _plugin(ctx, spec) if how not in _COUNTING else None
    if how not in _COUNTING and plugin is None:
        raise _invalid("how", how, f"no statistic plugin aggregates {measure}")
    out: list[dict[str, Any]] = []
    unknown_total = 0
    below = 0
    for g in group_rows(rows, group_by).values():
        vals = [r.get(measure) for r in g["rows"]] if measure else [1] * len(g["rows"])
        if how == "count":
            n, unknown, value = len(g["rows"]), 0, len(g["rows"])
        elif how == "count_distinct":
            known = [v for v in vals if v is not None and v == v]
            n, unknown = len(known), len(vals) - len(known)
            value = len({json.dumps(json_value(v), sort_keys=True) for v in known})
        else:
            try:
                res = plugin.aggregate(vals, how, spec)
            except Exception as exc:  # noqa: BLE001 - UnsupportedFilter: the statistic forbids this aggregation
                raise GatewayError(ErrorKind.invalid_argument, str(exc),
                                   payload=invalid_argument_payload("how", how,
                                                                    sorted(getattr(plugin, "aggregations", ()))))
            n, unknown, value = int(res.n), int(res.n_excluded), res.value
        unknown_total += unknown
        if n < min_n:
            below += 1
            continue
        out.append({**g["key"], how if how in _COUNTING else f"{how}_{measure}": json_value(value), "n": n,
                    "n_unknown": unknown})
    vname = how if how in _COUNTING else f"{how}_{measure}"
    present = [r for r in out if r[vname] is not None]
    absent = [r for r in out if r[vname] is None]
    present.sort(key=lambda r: (-float(r[vname]) if isinstance(r[vname], (int, float)) else 0,
                                json.dumps([json_value(r.get(c)) for c in group_by], default=str)))
    out = present + absent
    total = len(out)
    shown = out[:limit]
    if below:
        notes.append(f"{below} group(s) with fewer than {min_n} known values were dropped")
    excluded = dict(eu)
    if unknown_total and measure:
        excluded[measure] = excluded.get(measure, 0) + unknown_total
    hdr = header(view, rows=shown, total=total, truncated=total > len(shown), order=f"{vname} desc, then group key",
                 resolved=resolved, excluded_unknown=excluded, notes=notes, key=list(group_by),
                 extra={"groups_below_min_n": below, "rows_aggregated": total_rows, "how": how})
    return inject_header({"rows": shown}, hdr)


VERBS = {VERB_AGGREGATE: guarded("aggregate", aggregate)}

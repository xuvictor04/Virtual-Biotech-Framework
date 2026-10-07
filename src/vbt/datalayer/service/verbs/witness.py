"""``_witness``: an independent bounded count, top-k and key set of a bound table (§11.6).

One scan of the table or item table computes everything the request asks for: ``total`` (rows,
items, or distinct values of ``grain``), the top-k canonical keys under ``order`` (per comparable
group: ``group_by``, else the ``within`` groups of the order), the key set when the total is at most
``key_set_max`` (``data.witness.max_key_set``), distinct values, grain counts, group totals, keys
shared by several rows (``one_to_many``) and the unknown attribution (``excluded_unknown`` per
column with ``_rows``, ``excluded_not_applicable``, ``unknown_total``).

When the witness cannot express the request (a parameter without a value, a table served upstream
only) or the scan would exceed the budget, the response is ``total_method: unknown`` with a reason,
and the gateway makes no count or ranking claim from it.

**Remote witness** (phase 4, F20): a table whose layout declares the ``count`` capability (``live_api``,
``soma``) is not scanned. The bound predicate is compiled into one independent count request (CT.gov
``countTotal=true&pageSize=0``, E-utilities ``esearch`` ``rettype=count``, a SOMA ``value_filter`` read of
``soma_joinid``) within the source's request budget (``budget`` of the descriptor). Only the total is
answered: no top-k, no key set. A predicate that does not compile completely, a request that fails or
a source without a total is ``unknown`` (never a guess). The method is reported as ``scan`` with the
reason ``remote count request``.

``_live_find`` (phase 4) is the derived ``lookup``/``find`` of live tables: the pages of one request
within the source's budget (``max_pages`` and ``max_requests_per_call``; more pages are ``truncated``,
never presented as complete), ``pivot`` tables reshaped to one row per index (cBioPortal patient-level
clinical data: one row per ``(studyId, patientId)`` with one column per attribute), and record versions
(``key.version``) observed so that a record the source changed since an earlier call is listed under
``source_updated`` for provenance.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...ipc import VERB_WITNESS, WitnessRequest, WitnessResponse
from ...plugins.base import FormatError
from ...predicate import from_json
from .. import ServiceContext, ServiceError, layout_spec
from ..reader import BudgetExceeded, TableUnavailable, UnboundParameter

__all__ = ["witness", "remote_witness", "remote_names", "is_remote", "live_find", "pivot_rows", "REMOTE_REASON",
           "LIVE_FIND", "VERBS"]

REMOTE_REASON = "remote count request"
LIVE_FIND = "_live_find"


def remote_names(columns: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """``{column path: remote_name}`` of a table's columns (nested fields included)."""
    out: dict[str, str] = {}
    for name, col in (columns or {}).items():
        path = f"{prefix}{name}"
        rn = getattr(col, "remote_name", None)
        if rn:
            out[path] = str(rn)
        fields = getattr(col, "fields", None)
        if fields:
            out.update(remote_names(fields, path + "."))
    return out


def is_remote(ctx: ServiceContext, ref: str) -> bool:
    """True when ``ref``'s layout answers counts remotely (capability ``count`` without ``scan``)."""
    try:
        t = ctx.table(ref)
        layout = ctx.plugin("layout", t.layout)
    except Exception:  # noqa: BLE001 - unknown tables are the reader's to report
        return False
    caps = set(getattr(layout, "capabilities", ()) or ())
    return "count" in caps and "scan" not in caps


def remote_witness(ctx: ServiceContext, req: WitnessRequest) -> WitnessResponse:
    """The remote witness: one independent count request (see the module docstring)."""
    t = ctx.table(req.table)
    layout = ctx.plugin("layout", t.layout)
    if req.grain not in (None, "row", "rows") or req.group_by:
        return WitnessResponse(total_method="unknown",
                               reason=f"{REMOTE_REASON}s count rows only (grain {req.grain!r}, groups {req.group_by})")
    try:
        predicate = from_json(req.predicate) if req.predicate else None
    except Exception as exc:  # noqa: BLE001
        return WitnessResponse(total_method="unknown", reason=f"predicate: {exc}")
    kwargs: dict[str, Any] = {"predicate": predicate, "budget": t.descriptor.budget}
    names = remote_names(getattr(t.spec, "columns", {}) or {})
    if names and "remote_names" in inspect.signature(layout.count).parameters:
        kwargs["remote_names"] = names
    try:
        total = layout.count(layout_spec(t), **kwargs)
    except Exception as exc:  # noqa: BLE001 - an outage leaves the count unknown, never zero
        return WitnessResponse(total_method="unknown", reason=f"{REMOTE_REASON} failed: {exc}"[:500])
    if total is None:
        return WitnessResponse(total_method="unknown",
                               reason="the bound predicate cannot be expressed as a count request to the source")
    return WitnessResponse(total=int(total), total_method="scan", reason=REMOTE_REASON, scanned_bytes=0)


def witness(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = WitnessRequest.model_validate(dict(payload))
    if is_remote(ctx, req.table):
        return remote_witness(ctx, req).model_dump(mode="json")
    try:
        reader = ctx.reader(req.table)
    except TableUnavailable:
        raise
    except ServiceError as exc:
        return WitnessResponse(total_method="unknown", reason=str(exc)).model_dump(mode="json")
    settings = ctx.settings.witness
    group_by = list(req.group_by)
    if not group_by:
        for o in req.order:
            for w in o.within:
                if w not in group_by:
                    group_by.append(w)
    grains: dict[str, Any] = dict(req.grains)
    grain = None if req.grain in ("row", "rows") else req.grain   # the reserved row grain counts rows
    if grain:
        grains.setdefault("__grain__", grain)
    k = req.k if settings.topk and req.order else None
    try:
        agg = reader.aggregate(req.predicate, order=[o.model_dump() for o in req.order], k=k, group_by=group_by,
                               key=req.key, distinct=req.distinct, grains=grains,
                               key_set_max=req.key_set_max if req.key_set_max is not None else settings.max_key_set,
                               one_to_many=bool(req.key), params=req.params, unknown_columns=req.unknown_columns,
                               budget_bytes=req.budget_bytes)
    except BudgetExceeded as exc:
        return WitnessResponse(total_method="unknown", reason=f"over budget: {exc.reason}").model_dump(mode="json")
    except UnboundParameter as exc:
        return WitnessResponse(total_method="unknown", reason=str(exc)).model_dump(mode="json")
    except FormatError:
        raise
    st = agg.stats
    total = agg.distinct_counts.pop("__grain__") if grain else st.total
    resp = WitnessResponse(
        total=total, total_method="footer" if st.footer else ("index" if st.used_sidecar else "scan"),
        topk=agg.topk if k else [],
        key_set=agg.key_set, distinct=agg.distinct, excluded_unknown=dict(st.excluded_unknown),
        excluded_not_applicable=dict(st.excluded_not_applicable), unknown_total=st.unknown_total,
        distinct_counts=agg.distinct_counts, group_totals=agg.group_totals, one_to_many=agg.one_to_many,
        scanned_bytes=st.scanned_bytes,
        reason=None if all(agg.distinct_complete.values()) else "distinct values truncated at the value cap")
    return resp.model_dump(mode="json")


def pivot_rows(rows: Sequence[Mapping[str, Any]], index: Sequence[str], name_column: str,
               value_column: str) -> list[dict[str, Any]]:
    """Long ``(index..., name, value)`` rows -> one row per index with one column per name (first value
    kept; a second, different value for one name is listed under ``_conflicts``)."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        key = tuple(r.get(c) for c in index)
        row = out.setdefault(key, {c: r.get(c) for c in index})
        name = r.get(name_column)
        if name is None:
            continue
        value = r.get(value_column)
        if name in row and row[name] != value:
            row.setdefault("_conflicts", {}).setdefault(str(name), [row[name]]).append(value)
            continue
        row[str(name)] = value
    return [out[k] for k in sorted(out, key=lambda k: tuple(str(x) for x in k))]


def live_find(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``{table, predicate?, columns?, limit?}`` -> ``{rows, total, as_of, truncated, pages, source_updated}``."""
    from ...plugins.layouts.live_api import RecordVersions, fetch_all

    ref = str(payload["table"])
    t = ctx.table(ref)
    layout = ctx.plugin("layout", t.layout)
    if "live" not in (getattr(layout, "capabilities", ()) or ()):
        raise ServiceError(f"{ref} is not a live table (layout {t.layout!r})")
    predicate = from_json(payload["predicate"]) if payload.get("predicate") else None
    columns = list(payload.get("columns") or [])
    lspec = layout_spec(t)
    kwargs: dict[str, Any] = {}
    names = remote_names(getattr(t.spec, "columns", {}) or {})
    if names and "remote_names" in inspect.signature(layout.request).parameters:
        kwargs["remote_names"] = names
    pivot = t.spec.pivot
    sent = predicate
    if pivot is not None:
        # the source holds long rows: only conjuncts on the index columns can be sent; the rest is applied
        # to the pivoted rows
        from ...predicate import And, columns as predicate_columns

        parts = list(predicate.preds) if isinstance(predicate, And) else ([predicate] if predicate else [])
        keep = [p for p in parts if predicate_columns(p) <= set(pivot.index)]
        sent = None if not keep else (keep[0] if len(keep) == 1 else And(tuple(keep)))
    got = fetch_all(layout, lspec, sent, t.descriptor.budget, projection=[] if pivot else columns, **kwargs)
    rows = list(got["rows"])
    if pivot is not None:
        from ...predicate import evaluate

        rows = pivot_rows(rows, pivot.index, pivot.name_column, pivot.value_column)
        if predicate is not None:
            rows = [r for r in rows if evaluate(predicate, r) is True]
        if columns:
            rows = [{k: v for k, v in r.items() if k in columns or k in pivot.index} for r in rows]
    flags: dict[str, Any] = {"source_updated": [], "versions": {}}
    version = t.spec.key.version
    if version and len(t.spec.key.columns) == 1:
        store = RecordVersions(Path(ctx.settings.cache_dir) / t.physical.source / "record_versions.json")
        flags = store.observe_rows(str(t.physical), rows, t.spec.key.columns[0], version)
        store.save()
    limit = payload.get("limit")
    total = got["total"] if got["total"] is not None and pivot is None else (None if got["truncated"] else len(rows))
    shown = rows[: int(limit)] if limit is not None else rows
    return {"table": ref, "rows": shown, "total": total, "total_method": "unknown" if total is None else "remote",
            "as_of": got["as_of"], "truncated": bool(got["truncated"]) or len(shown) < len(rows),
            "pages": got["pages"], "source_updated": flags["source_updated"],
            "record_versions": {str(t.physical): flags["versions"]} if flags["versions"] else {}}


VERBS = {VERB_WITNESS: witness, LIVE_FIND: live_find}

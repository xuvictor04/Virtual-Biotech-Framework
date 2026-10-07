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
"""

from __future__ import annotations

from typing import Any, Mapping

from ...ipc import VERB_WITNESS, WitnessRequest, WitnessResponse
from ...plugins.base import FormatError
from .. import ServiceContext, ServiceError
from ..reader import BudgetExceeded, TableUnavailable, UnboundParameter

__all__ = ["witness", "VERBS"]


def witness(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = WitnessRequest.model_validate(dict(payload))
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
    if req.grain:
        grains.setdefault("__grain__", req.grain)
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
    total = agg.distinct_counts.pop("__grain__") if req.grain else st.total
    resp = WitnessResponse(
        total=total, total_method="footer" if st.footer else ("index" if st.used_sidecar else "scan"),
        topk=agg.topk if k else [],
        key_set=agg.key_set, distinct=agg.distinct, excluded_unknown=dict(st.excluded_unknown),
        excluded_not_applicable=dict(st.excluded_not_applicable), unknown_total=st.unknown_total,
        distinct_counts=agg.distinct_counts, group_totals=agg.group_totals, one_to_many=agg.one_to_many,
        scanned_bytes=st.scanned_bytes,
        reason=None if all(agg.distinct_complete.values()) else "distinct values truncated at the value cap")
    return resp.model_dump(mode="json")


VERBS = {VERB_WITNESS: witness}

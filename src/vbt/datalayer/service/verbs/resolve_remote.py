"""``_resolve_remote``: existence of values in a huge identity universe (``index: remote``; §11.5).

``rsid``, ``ot_variant`` and ``study_locus_id`` have universes too large for a sidecar, so the data
child answers membership by a bounded scan of the universe column, pruned by row-group statistics
and sidecar indexes where declared. Each value is normalised with the id_type's plugin (a rejected
value gets no resolution); found values resolve to themselves (``exact``, or ``normalized:<steps>``).

``existence`` is ``exists`` when every value was found, ``absent`` when none was, ``unknown`` when
some were (each resolution then carries its own ``existence``), and ``unknown`` whenever the scan
is over budget or the table cannot be read: the gateway never turns an undecidable lookup into
``not_found``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...descriptor.columns import UniverseSpec
from ...ipc import VERB_RESOLVE_REMOTE, ResolveRemoteRequest, ResolveRemoteResponse
from ...plugins.base import FormatError, Normalized
from ...predicate import In
from .. import ServiceContext, ServiceError
from .. import items as _items
from ..reader import BudgetExceeded

__all__ = ["resolve_remote", "VERBS"]


def _universe(source: str, spec: Any) -> tuple[str, str]:
    u = spec.universe
    if isinstance(u, list):
        u = u[0] if u else None
    if u is None:
        raise ServiceError("the id_type declares no universe")
    if isinstance(u, UniverseSpec):
        table, column = u.table, u.keys[-1]
    else:
        table, _, column = str(u).partition(".")
    return (table if "." in table else f"{source}.{table}"), column


def map_remote(ctx: ServiceContext, req: ResolveRemoteRequest, bare: str) -> dict[str, Any]:
    """``id_type: "a>b"``: map each ``a`` value to the ``b`` keys of the rows that hold it, through the table
    ``a``'s ``maps_to`` names (``rsid>ot_variant`` reads ``variant.rsIds[]`` and ``variant.variantId``)."""
    a, _, b = bare.partition(">")
    try:
        src, spec = ctx.catalog.id_type(f"{req.source}:{a}")
        _bsrc, bspec = ctx.catalog.id_type(f"{req.source}:{b}")
        mt = next(m for m in spec.maps_to if (m.id_type.partition(":")[2] or m.id_type) == b)
        table, column = _universe(src, spec)
        btable, bcolumn = _universe(src, bspec)
        if btable != table or not mt.via or not table.endswith("." + mt.via.split(".")[-1]):
            raise ServiceError(f"{a} maps to {b} through {mt.via}, whose rows do not hold both keys")
        plugin = ctx.identifier(f"{src}:{a}")
        reader = ctx.reader(table)
    except (ServiceError, LookupError, StopIteration) as exc:
        return ResolveRemoteResponse(existence="unknown",
                                     resolutions=[{"value": v, "note": str(exc)} for v in req.values]
                                     ).model_dump(mode="json")
    normalized = {v: n for v in req.values if isinstance((n := plugin.normalize(v)), Normalized)}
    wanted = sorted({n.value for n in normalized.values()})
    targets: dict[str, list[str]] = {w: [] for w in wanted}
    if wanted:
        try:
            for m in reader.scan(In("/" + column, tuple(wanted)), columns=[column, bcolumn], attribute_unknown=False):
                keys = [str(x) for x in _items.path_values(m.row, bcolumn) if x is not None]
                for x in _items.path_values(m.row, column):
                    if x is not None and str(x) in targets:
                        targets[str(x)].extend(k for k in keys if k not in targets[str(x)])
        except (BudgetExceeded, ServiceError, FormatError) as exc:
            return ResolveRemoteResponse(existence="unknown",
                                         resolutions=[{"value": v, "existence": "unknown", "note": str(exc)}
                                                      for v in req.values]).model_dump(mode="json")
    resolutions: list[dict[str, Any]] = []
    for v in req.values:
        n = normalized.get(v)
        hits = targets.get(n.value, []) if n is not None else []
        if not hits:
            resolutions.append({"value": v, "existence": "absent" if n is not None else "unknown"})
        resolutions.extend({"value": v, "canonical": k, "rule": f"maps_to:{b}", "label": None, "existence": "exists"}
                           for k in sorted(hits))
    states = {r["existence"] for r in resolutions}
    existence = "exists" if states == {"exists"} else ("absent" if states == {"absent"} else "unknown")
    return ResolveRemoteResponse(resolutions=resolutions, existence=existence).model_dump(mode="json")


def resolve_remote(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = ResolveRemoteRequest.model_validate(dict(payload))
    bare = req.id_type.partition(":")[2] or req.id_type
    if ">" in bare:
        return map_remote(ctx, req, bare)
    try:
        src, spec = ctx.catalog.id_type(f"{req.source}:{bare}")
        table, column = _universe(src, spec)
        plugin = ctx.identifier(f"{src}:{bare}")
        reader = ctx.reader(table)
    except (ServiceError, LookupError) as exc:
        return ResolveRemoteResponse(existence="unknown",
                                     resolutions=[{"value": v, "note": str(exc)} for v in req.values]
                                     ).model_dump(mode="json")
    normalized: dict[str, Normalized] = {}
    for v in req.values:
        n = plugin.normalize(v)
        if isinstance(n, Normalized):
            normalized[v] = n
    wanted = sorted({n.value for n in normalized.values()})
    found: set[str] = set()
    if wanted:
        try:
            for m in reader.scan(In("/" + column, tuple(wanted)), columns=[column], attribute_unknown=False):
                for x in _items.path_values(m.row, column):
                    if x is not None and str(x) in wanted:
                        found.add(str(x))
        except (BudgetExceeded, ServiceError, FormatError) as exc:
            return ResolveRemoteResponse(existence="unknown",
                                         resolutions=[{"value": v, "existence": "unknown", "note": str(exc)}
                                                      for v in req.values]).model_dump(mode="json")
    resolutions = []
    for v in req.values:
        n = normalized.get(v)
        if n is None:
            resolutions.append({"value": v, "existence": "unknown", "note": "rejected by the identifier plugin"})
            continue
        if n.value in found:
            rule = "exact" if not n.steps else "normalized:" + "+".join(n.steps)
            resolutions.append({"value": v, "canonical": n.value, "rule": rule, "label": None, "existence": "exists"})
        else:
            resolutions.append({"value": v, "existence": "absent"})
    states = {r["existence"] for r in resolutions}
    existence = "exists" if states == {"exists"} else ("absent" if states == {"absent"} else "unknown")
    return ResolveRemoteResponse(resolutions=resolutions, existence=existence).model_dump(mode="json")


VERBS = {VERB_RESOLVE_REMOTE: resolve_remote}

"""``_resolve_remote``: existence of values in a huge identity universe (``index: remote``; §11.5).

``rsid``, ``ot_variant`` and ``study_locus_id`` have universes too large for a sidecar, so the data
child answers membership by a bounded scan of the universe column, pruned by row-group statistics
and sidecar indexes where declared. Each value is normalised with the id_type's plugin (a rejected
value gets no resolution); found values resolve to themselves (``exact``, or ``normalized:<steps>``).

``existence`` is ``exists`` when every value was found, ``absent`` when none was, ``unknown`` when
some were (each resolution then carries its own ``existence``), and ``unknown`` whenever the scan
is over budget or the table cannot be read: the gateway never turns an undecidable lookup into
``not_found``.

An id_type with ``universe_via`` (cBioPortal ``cbio_cancer_type``, ``cbio_study``: a listing, not a table
column) is decided against the listing of the live table whose key it is (``self: true``): the source's own
cancer types or studies, read whole within the source's budget and kept for ``universe_via.ttl_s``. A listing
cut by the budget decides nothing (``unknown``).
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from ...descriptor.columns import UniverseSpec
from ...ipc import VERB_RESOLVE_REMOTE, ResolveRemoteRequest, ResolveRemoteResponse
from ...plugins.base import FormatError, Normalized
from ...predicate import In
from .. import ServiceContext, ServiceError
from .. import items as _items
from ..reader import BudgetExceeded

__all__ = ["resolve_remote", "VERBS"]


_LISTED: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}


def _listed_universe(ctx: ServiceContext, source: str, bare: str, spec: Any) -> frozenset[str]:
    """The keys of the live table whose ``self`` key is this id_type (``universe_via``), read whole and cached for
    ``ttl_s``; ServiceError when there is no such table or the listing was cut."""
    from .public import is_live
    from .witness import live_find

    via = spec.universe_via
    ttl = float(via.ttl_s if via is not None and via.ttl_s is not None else 3600)
    hit = _LISTED.get((source, bare))
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return hit[1]
    desc = ctx.catalog.sources[source]
    for name, t in desc.tables.items():
        for col, c in (t.columns or {}).items():
            if getattr(c, "self_", False) and str(getattr(c, "id_type", "")).rpartition(":")[2] == bare:
                table = ctx.table(f"{source}.{name}")
                if not is_live(ctx, table):
                    continue
                got = live_find(ctx, {"table": f"{source}.{name}", "columns": [col]})
                if got.get("truncated"):
                    raise ServiceError(f"the {source}.{name} listing was cut by the source's budget")
                keys = frozenset(str(r.get(col)) for r in got.get("rows") or [] if isinstance(r, Mapping)
                                 and r.get(col) is not None)
                _LISTED[(source, bare)] = (time.monotonic(), keys)
                return keys
    raise ServiceError(f"no live table of {source} lists the {bare} universe")


def _listed(ctx: ServiceContext, req: ResolveRemoteRequest, src: str, bare: str, spec: Any) -> dict[str, Any]:
    """Membership in a ``universe_via`` listing (see the module docstring)."""
    try:
        keys = _listed_universe(ctx, src, bare, spec)
        plugin = ctx.identifier(f"{src}:{bare}")
    except Exception as exc:  # noqa: BLE001 - an unreadable listing decides nothing
        return ResolveRemoteResponse(existence="unknown",
                                     resolutions=[{"value": v, "existence": "unknown", "note": str(exc)}
                                                  for v in req.values]).model_dump(mode="json")
    resolutions = []
    for v in req.values:
        n = plugin.normalize(v)
        if not isinstance(n, Normalized):
            resolutions.append({"value": v, "existence": "unknown", "note": "rejected by the identifier plugin"})
        elif n.value in keys or v in keys:
            canon = n.value if n.value in keys else v
            rule = "exact" if not n.steps else "normalized:" + "+".join(n.steps)
            resolutions.append({"value": v, "canonical": canon, "rule": rule, "label": None, "existence": "exists"})
        else:
            resolutions.append({"value": v, "existence": "absent"})
    states = {r["existence"] for r in resolutions}
    existence = "exists" if states == {"exists"} else ("absent" if states == {"absent"} else "unknown")
    return ResolveRemoteResponse(resolutions=resolutions, existence=existence).model_dump(mode="json")


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
        if spec.universe is None and getattr(spec, "universe_via", None) is not None:
            return _listed(ctx, req, src, bare, spec)
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

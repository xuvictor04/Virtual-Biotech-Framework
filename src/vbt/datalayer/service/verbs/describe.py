"""``describe(source[, table])``: what a source and its tables hold, from the descriptors (§10.6).

Without ``table``: the source's title, release, coverage notes and its tables (kind, grain, key and
whether the native tools serve them). With ``table``: the table's long view, with each column's role
and the facets an agent needs to query it (id_type, vocabulary, scale, unit, direction, cutoff,
level, family, censoring, missing-value codes, and ``ops``: the operators a ``where`` entry on it takes),
its rank, coverage statement, evidence nature and which verbs apply to it. This is where an agent reads a
table's column map: the listed ``find``/``aggregate`` schemas name ``describe`` instead of carrying every
table's (``derive.tools.WHERE_POINTER``). Tables withheld from the calling ``agent`` (``expose.withhold_from``) and
tables with ``expose.native: false`` are not listed and cannot be described.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...errors import json_value
from ...result import Header, inject_header
from .. import ServiceContext
from .public import _invalid, exposed_tables, key_label, long_view, table_access

__all__ = ["describe", "verbs_for", "column_facets"]

_FACETS = ("id_type", "vocab", "scale", "unit", "direction", "encoding", "level", "family", "missing_values",
           "placeholders", "of", "synonym_kind", "side", "comparable_within")


def verbs_for(view: Any) -> list[str]:
    """The public verbs that apply to a table, from its kind and roles."""
    roles = {getattr(c, "role", None) for c in view.columns.values()}
    out = ["find", "lookup", "aggregate"]
    if view.matrix is None and roles & {"label", "synonym"}:
        out.append("search")
    if roles & {"category", "scope"}:
        out.append("vocab")
    if "member" in roles:
        out.append("members")
    if "vector" in roles:
        out.append("similar")
    if view.table.spec.edge is not None:
        out.append("neighbors")
    return out


def column_facets(spec: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": getattr(spec, "role", None)}
    for f in _FACETS:
        v = getattr(spec, f, None)
        if v not in (None, [], {}, ""):
            out[f] = json_value(v if not hasattr(v, "model_dump") else v.model_dump(exclude_none=True))
    cutoff = getattr(spec, "cutoff", None)
    if cutoff is not None:
        out["cutoff"] = json_value(cutoff.model_dump(exclude_none=True))
    if getattr(spec, "event_of", None):
        out["censoring"] = f"{spec.event_of} is right-censored when this flag is false"
    membership = getattr(spec, "membership", None)
    if membership is not None:
        out["propagation"] = membership.propagation
    return out


def _where_ops(ctx: ServiceContext, ref: str) -> dict[str, list[str]]:
    """``{column: operators}`` a ``where`` entry takes on each filterable long-view column (the listing's per-column
    map, ``derive.tools.where_schema``); empty when it cannot be derived."""
    from ...derive.tools import where_schema

    try:
        schema = where_schema(ctx.catalog, ref)
    except Exception:  # noqa: BLE001 - describe never fails over the operators
        return {}
    return {name: list(col.get("x-vbt-ops") or []) for name, col in schema.items() if col.get("x-vbt-ops")}


def describe(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The source, or one of its tables (see the module docstring)."""
    agent = payload.get("agent")
    source = payload.get("source")
    table = payload.get("table")
    if table and "." in str(table) and not source:
        source = str(table).split(".", 1)[0]
    if source not in ctx.catalog.sources:
        raise _invalid("source", source, f"unknown source {source!r}", sorted(ctx.catalog.sources))
    desc = ctx.catalog.source(str(source))
    rel = desc.release.expect
    if not table:
        refs = [r for r in exposed_tables(ctx, agent) if r.startswith(f"{source}.")]
        rows = []
        for ref in refs:
            t = ctx.catalog.table(ref)
            rows.append({"table": ref, "kind": t.kind, "grain": t.grain, "key": [key_label(k) for k in t.key],
                         "item_table_of": str(t.physical) if t.is_item_table else None})
        hdr = Header(status="ok" if rows else "empty", source=f"{source}@{rel}" if rel else str(source),
                     returned=len(rows), total=len(rows), served_by="derived", key=["table"])
        return inject_header({"title": desc.title, "release": rel, "tables": rows}, hdr)
    ref = str(table) if "." in str(table) else f"{source}.{table}"
    t = table_access(ctx, ref, agent=agent)
    view = long_view(ctx, ref)
    cols = {name: column_facets(spec) for name, spec in view.columns.items()}
    for name, where in _where_ops(ctx, ref).items():
        if name in cols:
            cols[name]["ops"] = where
    cov = t.spec.coverage
    body = {
        "table": ref, "kind": t.kind, "grain": t.grain, "key": [key_label(k) for k in view.key], "columns": cols,
        "rank": [r.model_dump(exclude_none=True) for r in t.spec.rank],
        "coverage": cov.model_dump(exclude_none=True) if cov is not None else None,
        "evidence": t.spec.evidence_nature.caveat if t.spec.evidence_nature is not None else None,
        "verbs": verbs_for(view),
    }
    if view.matrix is not None:
        body["matrix"] = {"row": view.row_key, "col": view.col_key, "values": view.values,
                          "attributes": list(getattr(view, "attr_columns", []))}
    hdr = Header(status="ok", source=f"{source}@{rel}" if rel else str(source), tables=[ref], served_by="derived")
    return inject_header(json_value(body), hdr)

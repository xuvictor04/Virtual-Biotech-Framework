"""``_view``: multi-table views with a status per section (§10.6; replaces the comprehensive target profile).

Upstream's ``get_comprehensive_target_profile`` turned a failed section into ``[]`` and a count of 0, so
an outage read as "no drugs" (OT-TARGET-011). A view section is read on its own and reports its own
status:

* ``ok``: rows (or the record of a ``single`` section) with ``total``;
* ``empty``: the entity has no rows in the section's table, with that table's coverage and statement
  (``covered`` means an absence; ``unknown`` means not evidence of absence);
* ``not_ready``: the section's table could not be read (missing, unreadable, over budget), with the
  reason and no rows and **no count**: a count of 0 is never fabricated for data that was not read.

Sections come from (in this order) ``sections`` in the request, a descriptor view
(``view: "source.name"``, ``SourceDescriptor.views``) or a derived binding (``server`` + ``tool``: the
binding's ``derived.sections``, e.g. the target profile's ``target``, ``known_drug``, ``pathways``,
``openfda`` reactions and ``mouse_phenotype``). ``args`` fill the sections' ``key_from_args``; each
argument is resolved like the binding's argument (its ``accepts``), so an unknown entity is
``not_found`` before any section is read.

:func:`serve_sections` is the same per-section read for the sections of a gateway ``_serve`` request
(derived bindings): a section that cannot be read becomes ``{_vbt_unavailable, status: not_ready}``
(the gateway's own marker for unready sections) instead of failing the whole call.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...errors import ErrorKind, GatewayError, json_value
from ...predicate import And, Eq
from ...result import Header, inject_header
from .. import ServiceContext, ServiceError
from ..reader import BudgetExceeded
from .public import _invalid, _resolve_value, coverage_of, guarded

__all__ = ["view", "read_section", "serve_sections", "view_sections", "VERBS"]

_UNREADABLE = (ServiceError, BudgetExceeded, OSError, ValueError)


def read_section(ctx: ServiceContext, name: str, sec: Mapping[str, Any], params: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """One section with its status (see the module docstring)."""
    table = str(sec.get("table") or "")
    try:
        t = ctx.table(table)
        reader = ctx.reader(table)
        key = sec.get("key") or {}
        parts = [Eq("/" + str(c), v) for c, v in key.items()]
        pred = parts[0] if len(parts) == 1 else (And(tuple(parts)) if parts else None)
        rows, _keys, stats = reader.rows(pred, sec.get("columns") or (), sec.get("order") or (), sec.get("limit"),
                                         params=dict(params or {}), budget_bytes=ctx.scan_budget(t, None))
    except _UNREADABLE as exc:
        return {"status": "not_ready", "table": table, "reason": f"{table} not ready: {exc}"}
    out: dict[str, Any] = {"table": table}
    if not rows:
        coverage, statement = coverage_of(t)
        out.update(status="empty", total=0, coverage=coverage, coverage_statement=statement)
        if sec.get("single") and t.kind != "entity_detail":
            out["record"] = None
        else:
            out["rows"] = []
        return out
    out["status"] = "ok"
    if sec.get("single") and t.kind != "entity_detail":
        row = rows[0]
        field = sec.get("value")
        out["record"] = json_value(row.get(str(field)) if field else row)
    else:
        out["rows"] = json_value(rows)
        out["total"] = int(stats.total)
        out["truncated"] = int(stats.total) > len(rows)
    return out


def serve_sections(ctx: ServiceContext, sections: Mapping[str, Mapping[str, Any]], params: Mapping[str, Any]
                   ) -> dict[str, Any]:
    """The sections of a ``_serve`` request in the phase-1 value form (rows, or the single record), a section
    that cannot be read as ``{_vbt_unavailable, status: not_ready}``."""
    out: dict[str, Any] = {}
    for name, sec in sections.items():
        got = read_section(ctx, name, sec, params)
        if got["status"] == "not_ready":
            out[name] = {"_vbt_unavailable": got["reason"], "status": "not_ready"}
        elif "record" in got:
            out[name] = got["record"]
        else:
            out[name] = got["rows"]
    return out


def view_sections(ctx: ServiceContext, payload: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], Any]:
    """``({name: section spec dict}, binding or None)`` named by the request."""
    if payload.get("sections"):
        return {k: dict(v) for k, v in dict(payload["sections"]).items()}, None
    if payload.get("view"):
        ref = str(payload["view"])
        src, _, name = ref.partition(".")
        try:
            v = ctx.catalog.source(src).views[name]
        except (LookupError, KeyError):
            raise _invalid("view", ref, f"unknown view {ref!r}") from None
        return {k: s.model_dump(exclude_none=True) for k, s in v.sections.items()}, None
    server, tool = payload.get("server"), payload.get("tool")
    try:
        contract = ctx.catalog.contract(str(server), str(tool))
    except Exception:  # noqa: BLE001
        contract = None
    b = getattr(contract, "binding", None)
    if b is None or b.derived is None or not b.derived.sections:
        raise _invalid("tool", f"{server}.{tool}", "name a view, sections, or a derived binding with sections")
    secs = {k: s.model_dump(exclude_none=True) for k, s in b.derived.sections.items()}
    main = {"table": b.derived.table, "verb": "lookup", "single": True, "key_from_args": {}}
    for name, a in b.args.items():
        if isinstance(a.binds, str) and a.binds.startswith(b.derived.table + "."):
            main["key_from_args"][a.binds[len(b.derived.table) + 1:]] = name
    return {"_main": main, **secs}, contract


def view(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Every section of a view for one entity, each with its own status."""
    secs, contract = view_sections(ctx, payload)
    args = dict(payload.get("args") or {})
    notes: list[str] = []
    resolved: dict[str, str] = {}
    values: dict[str, Any] = {}
    for name, value in args.items():
        a = contract.args.get(name) if contract is not None else None
        bound = None
        if a is not None and a.accepts and isinstance(a.binds, str):
            table, _, col = a.binds.rpartition(".")
            t = ctx.table(table)
            idt = getattr(t.columns.get(col), "id_type", None)
            bound = ctx.catalog.qualify_id_type(idt, t.descriptor.source) if idt else None
        values[name] = _resolve_value(ctx, name, value, bound, notes, resolved) if bound else value
    sections: dict[str, Any] = {}
    for name, sec in secs.items():
        key = {}
        for col, arg in (sec.get("key_from_args") or {}).items():
            if arg not in values:
                raise GatewayError(ErrorKind.invalid_argument, f"section {name} needs argument {arg!r}",
                                   argument=arg)
            key[col] = values[arg]
        sections[name] = read_section(ctx, name, {**sec, "key": key}, args)
    main = sections.pop("_main", None)
    if main is not None and main["status"] == "empty":
        raise GatewayError(ErrorKind.not_found, f"no {secs['_main']['table']} record for {values}",
                           payload={"table": secs["_main"]["table"]})
    statuses = {n: s["status"] for n, s in sections.items()}
    status = "partial" if "not_ready" in statuses.values() else "ok"
    hdr = Header(status=status, served_by="derived", resolved=resolved or None, tables=sorted(
        {str(s.get("table")) for s in secs.values()}), notes=notes, extra={"sections": statuses})
    body: dict[str, Any] = {}
    if main is not None:
        body["record"] = main.get("record")
    body["sections"] = sections
    return inject_header(body, hdr)


VERBS = {"_view": guarded("view", view)}

"""Public verbs of the data child (§10.6, phase 2): role-derived tools listed as ``mcp__data__*``.

``resolve(id_type, values)``, ``describe(source[, table])``, ``lookup(table, key)``, ``find(table, where,
rank_by, limit, distinct, group_by)``, ``search(table, text)``, ``vocab(table, column)``,
``members(table, set_id, propagate)``, ``aggregate(table, group_by, measure, how, min_n)``,
``similar(table, anchor, where, top_k)``, ``neighbors(table, node)`` (phase 3: ``network.py`` wraps it for
``nodes``, ``hops`` 1-4 and ``max_nodes``; ``hierarchy.py`` wraps ``members`` for ``propagate: true``), and
the phase-3 ``expand(id_type, values, direction)`` and ``enrich(table, genes, ...)`` (the hidden ``_expand``
and ``_enrich`` handlers, listed publicly).

Every verb works on the table's **long view**: the columns of a table or item table, or for a matrix
the row-axis key and fields, the column-axis key and parsed header fields, the values and the row
attributes joined through ``attributes_from`` (:class:`LongView`). Every ``where`` and ``key`` entry is
bound to one long-view column and resolved like an overlay argument: an identifier column's values go
through the same :class:`~vbt.datalayer.resolve.resolver.Resolver` the gateway uses (labels,
synonyms, retired IDs, xrefs and crosswalks of the column's id_type and of every id_type with a
declared edge to it), so a miss is ``not_found``, several entities are ``ambiguous`` and a wrong kind
is ``invalid_argument``; category values outside a complete vocabulary are ``invalid_argument`` with
the nearest valid values. An identifier that resolves to an entity the table does not cover (a gene
that is not a column of the DepMap matrix) gives ``status: empty`` with ``coverage: not_covered``.

Results carry the ``_vbt`` header first (status, returned, total, truncated, order, coverage and its
statement, resolutions, excluded unknowns); errors are the typed envelopes of §12.1
(``status: tool_error``). Tables with ``expose.native: false`` are refused, and so are tables whose
``expose.withhold_from`` names the calling ``agent`` (the payload's ``agent``, set by the gateway).

The gateway's ``_serve`` requests for the phase-2 derived bindings (``verb: similar``, set
comparisons, sections with a per-section status) are routed by ``serve.py`` to ``similar.py``,
``setcompare.py`` and ``views.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ...errors import ErrorKind, GatewayError, error_envelope, invalid_argument_payload, json_value, nearest
from ...predicate import And, Cmp, Contains, Eq, In, Not, Or, Predicate, evaluate
from ...result import Header, inject_header
from ...rowkey import canonical
from .. import ServiceContext, ServiceError

__all__ = [
    "PUBLIC_VERBS", "LongView", "long_view", "table_access", "compile_where", "resolver", "header", "guarded",
    "find", "lookup", "search", "vocab", "members", "neighbors", "resolve", "coverage_of", "order_rows",
    "VERBS",
]

#: The public verbs, in listing order (each is ``mcp__data__<name>``).
PUBLIC_VERBS = ("resolve", "describe", "lookup", "find", "search", "vocab", "members", "aggregate", "similar",
                "neighbors", "expand", "enrich")
#: ``where`` operators: ``{column: value}`` is ``eq`` (``in`` for a list).
WHERE_OPS = ("eq", "in", "ne", "ge", "gt", "le", "lt", "contains")
_CMP = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "ne": "!="}
_RANKABLE = ("measure", "count", "time")
#: The hard cap on a native call's ``limit`` (the memory budget may lower it).
MAX_ROWS = 1000


# ---------------------------------------------------------------------------- errors and headers

def guarded(name: str, fn: Callable[[ServiceContext, Mapping[str, Any]], dict[str, Any]]
            ) -> Callable[[ServiceContext, Mapping[str, Any]], dict[str, Any]]:
    """A verb whose typed failures come back as §12.1 envelopes (``status: tool_error``)."""
    from ..reader import BudgetExceeded, TableUnavailable

    def verb(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
        tool = f"data.{name}"
        try:
            return fn(ctx, payload)
        except GatewayError as exc:
            return exc.with_tool(tool).envelope()
        except BudgetExceeded as exc:
            return error_envelope(ErrorKind.too_large, str(exc), tool=tool)
        except TableUnavailable as exc:
            return error_envelope(ErrorKind.not_ready, str(exc), tool=tool)
        except ServiceError as exc:
            return error_envelope(ErrorKind.invalid_argument, str(exc), tool=tool)

    verb.__name__ = name
    verb.__doc__ = fn.__doc__
    return verb


def _invalid(argument: str, value: Any, message: str, valid: Sequence[Any] | None = None) -> GatewayError:
    near = nearest(value, valid) if valid else []
    return GatewayError(ErrorKind.invalid_argument, message,
                        payload=invalid_argument_payload(argument, value, list(valid) if valid is not None else None,
                                                         near=near[:5]))


def coverage_of(table: Any, *, covered: bool | None = None, excluded_unknown: int = 0) -> tuple[str, str | None]:
    """``(coverage, statement)`` of an empty or filtered result over ``table`` (as the gateway states it)."""
    cov = getattr(getattr(table, "spec", None), "coverage", None)
    statement = cov.statement if cov is not None else None
    if covered is False:
        return "not_covered", statement
    if cov is None or cov.absence_means == "unknown":
        return "unknown", statement
    if cov.absence_means == "censored":
        return "censored", statement
    return ("partial_unknown" if excluded_unknown else "covered"), statement


def header(view: "LongView", *, rows: Sequence[Any], total: int | None, truncated: bool = False,
           order: str | None = None, resolved: Mapping[str, str] | None = None,
           excluded_unknown: Mapping[str, int] | None = None, covered: bool | None = None,
           notes: Sequence[str] = (), extra: Mapping[str, Any] | None = None, key: Sequence[str] | None = None
           ) -> Header:
    """The ``_vbt`` header of a native result."""
    eu = {k: int(v) for k, v in (excluded_unknown or {}).items() if v}
    status = "ok" if rows else "empty"
    if rows and truncated:
        status = "partial" if total is None else "ok"
    coverage, statement = (None, None)
    if not rows:
        coverage, statement = coverage_of(view.table, covered=covered, excluded_unknown=sum(eu.values()))
    desc = view.table.descriptor
    rel = desc.release.expect
    return Header(status=status, source=f"{desc.source}@{rel}" if rel else desc.source, tables=[view.ref],
                  key=list(key if key is not None else view.key), returned=len(rows), total=total,
                  total_method="data_child" if total is not None else "unknown", truncated=truncated, order=order,
                  resolved=dict(resolved) or None if resolved else None, excluded_unknown=eu or None,
                  coverage=coverage, coverage_statement=statement,
                  evidence=(view.table.spec.evidence_nature.caveat if view.table.spec.evidence_nature else None),
                  served_by="derived", extra=dict(extra or {}), notes=list(notes))


def _result(view: "LongView", rows: list[Any], hdr: Header, **body: Any) -> dict[str, Any]:
    return inject_header({"rows": json_value(rows), **body}, hdr)


# ---------------------------------------------------------------------------- tables and exposure

def table_access(ctx: ServiceContext, ref: Any, *, agent: str | None = None, argument: str = "table",
                 native: bool = True) -> Any:
    """The catalog table of a public call: known, exposed to native tools (``native``; harness readers of the
    in-process client may read ``expose.native: false`` tables) and not withheld from ``agent``."""
    if not isinstance(ref, str) or ref.count(".") != 1:
        raise _invalid(argument, ref, f"{argument} names a table as 'source.table'", ctx.table_refs())
    try:
        table = ctx.catalog.table(ref)
    except LookupError:
        raise _invalid(argument, ref, f"unknown table {ref!r}", ctx.table_refs()) from None
    expose = table.spec.expose
    if native and not expose.native:
        raise GatewayError(ErrorKind.invalid_argument, f"{ref} is not served by the native tools"
                           + (f" ({expose.reason})" if expose.reason else ""), argument=argument, value=ref,
                           subkind="not_exposed")
    if agent and agent in expose.withhold_from:
        raise GatewayError(ErrorKind.invalid_argument, f"{ref} is withheld from {agent}"
                           + (f" ({expose.reason})" if expose.reason else ""), argument=argument, value=ref,
                           subkind="withheld")
    return table


def exposed_tables(ctx: ServiceContext, agent: str | None = None) -> list[str]:
    out = []
    for ref in ctx.table_refs():
        try:
            table_access(ctx, ref, agent=agent)
        except GatewayError:
            continue
        out.append(ref)
    return out


def long_view(ctx: ServiceContext, ref: str) -> "LongView":
    cache = ctx.__dict__.setdefault("_long_views", {})
    view = cache.get(ref)
    if view is None:
        view = cache[ref] = LongView(ctx, ref)
    return view


# ---------------------------------------------------------------------------- the long view

class LongView:
    """The long view of one table: its columns with their specs, its key, and bounded row reads.

    Non-matrix tables and item tables read through :class:`~vbt.datalayer.service.reader.TableReader`;
    matrices read through the format's ``axis_values`` and ``slice`` (one long row per cell, empty
    cells null), with ``attributes_from`` columns joined onto the row axis."""

    def __init__(self, ctx: ServiceContext, ref: str) -> None:
        self.ctx = ctx
        self.ref = ref
        self.table = ctx.table(ref)
        self.source = self.table.descriptor.source
        self.matrix = self.table.physical_spec.matrix if self.table.kind == "matrix" else None
        self.columns: dict[str, Any] = {}
        self._attr: dict[str, Any] | None = None
        if self.matrix is None:
            self.reader = ctx.reader(ref)
            self.columns = dict(self.table.columns)
            self.key = [k for k in self.table.key]
            return
        from ...plugins.formats.csv import matrix_layout

        self.reader = ctx.reader(ref)
        self.layout = matrix_layout(self.matrix)
        row, col = self.matrix.axes["row"], self.matrix.axes["col"]
        for ax in (row, col):
            for name, spec in ax.columns.items():
                self.columns[name] = spec
            if ax.parse is not None:
                for name, spec in ax.parse.fields.items():
                    self.columns[name] = spec
        self.columns.update(self.matrix.values)
        self.attr_columns: list[str] = []
        af = row.attributes_from or None
        if af:
            other = self._qualify_table(str(af.get("table")))
            spec = ctx.table(other)
            for c in af.get("columns") or []:
                if c in spec.columns:
                    self.columns[c] = spec.columns[c]
                    self.attr_columns.append(c)
            self._attr = {"table": other, "key": af.get("key"), "columns": list(self.attr_columns)}
        self.row_key = list(self.layout.row.key)
        self.col_key = list(self.layout.col.key)
        self.col_fields = [f for f in self.layout.col.fields if f not in self.col_key]
        self.values = list(self.layout.values)
        self.key = [*self.row_key, *self.col_key]

    def _qualify_table(self, name: str) -> str:
        return name if "." in name else f"{self.source}.{name}"

    # -- columns ---------------------------------------------------------------------------------

    def column(self, name: str) -> Any:
        return self.columns.get(name)

    def id_type(self, name: str) -> str | None:
        spec = self.column(name)
        idt = getattr(spec, "id_type", None)
        if not idt or not isinstance(idt, str):
            return None
        try:
            return self.ctx.catalog.qualify_id_type(idt, self.source)
        except LookupError:
            return None

    def rankable(self) -> list[str]:
        return sorted(n for n, c in self.columns.items() if getattr(c, "role", None) in _RANKABLE)

    def default_order(self) -> list[dict[str, Any]]:
        return [r.model_dump() for r in self.table.spec.rank]

    # -- reads -------------------------------------------------------------------------------------

    def rows(self, pred: Predicate | None, *, columns: Sequence[str] = (), order: Sequence[Mapping[str, Any]] = (),
             limit: int | None = None, group_by: Sequence[str] = (), distinct: Sequence[str] = (),
             budget: int | None = None, col_keys: Sequence[str] | None = None,
             row_keys: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        """``(rows, total matches, excluded unknown per column)`` under ``order`` (else the table's rank),
        nulls last, ties by the canonical key, cut to ``limit`` (per ``group_by`` group)."""
        if self.matrix is None:
            from ..reader import ScanStats

            stats = ScanStats()
            out, _keys, stats = self.reader.rows(pred, list(columns), list(order) or self.default_order(), limit,
                                                 group_by=list(group_by), distinct=list(distinct),
                                                 budget_bytes=self.ctx.scan_budget(self.table, budget), stats=stats)
            return out, int(stats.total), dict(stats.excluded_unknown)
        cells = list(self.cells(pred, budget=budget, col_keys=col_keys, row_keys=row_keys))
        total = len(cells)
        ordered = order_rows(cells, list(order) or self.default_order(), self.key, group_by=group_by)
        if distinct:
            seen: set[str] = set()
            kept = []
            for r in ordered:
                c = canonical([r.get(d) for d in distinct])
                if c not in seen:
                    seen.add(c)
                    kept.append(r)
            ordered = kept
        ordered = _cut(ordered, limit, group_by)
        if columns:
            ordered = [{c: r.get(c) for c in columns} for r in ordered]
        return ordered, total, {}

    def axis(self, axis: str) -> list[dict[str, Any]]:
        """The members of a matrix axis over every fragment (``row`` with its joined attributes)."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for frag in self.reader.fragments():
            for r in self.reader.fmt.to_native(self.reader.fmt.axis_values(frag, axis)):
                r = {k: v for k, v in r.items() if k != "position"}
                k = canonical([r.get(x) for x in (self.row_key if axis == "row" else self.col_key)])
                if k not in seen:
                    seen.add(k)
                    out.append(r)
        if axis == "row" and self._attr:
            attrs = self._attributes([r.get(self.row_key[0]) for r in out])
            for r in out:
                r.update(attrs.get(str(r.get(self.row_key[0])), {}))
        return out

    def _attributes(self, keys: Iterable[Any]) -> dict[str, dict[str, Any]]:
        if not self._attr:
            return {}
        want = sorted({str(k) for k in keys if k is not None})
        if not want:
            return {}
        reader = self.ctx.reader(self._attr["table"])
        kcol = str(self._attr["key"])
        rows, _k, _s = reader.rows(In(kcol, tuple(want)), [kcol, *self._attr["columns"]], [], None)
        return {str(r.get(kcol)): {c: r.get(c) for c in self._attr["columns"]} for r in rows}

    def cells(self, pred: Predicate | None, *, budget: int | None = None, col_keys: Sequence[str] | None = None,
              row_keys: Sequence[str] | None = None) -> Iterable[dict[str, Any]]:
        """Long-view rows of a matrix (one per cell, every value column) that satisfy ``pred``."""
        fmt = self.reader.fmt
        rpred = In(self.row_key[0], tuple(row_keys)) if row_keys is not None else None
        budget = self.ctx.scan_budget(self.table, budget)
        attrs_cache: dict[str, dict[str, Any]] = {}
        for frag in self.reader.fragments():
            merged: dict[str, dict[str, Any]] = {}
            for value in self.values:
                for batch in fmt.slice(frag, value, row_predicate=rpred, col_keys=col_keys, budget_bytes=budget,
                                       attributes=self.col_fields):
                    for r in fmt.to_native(_table_of(batch)):
                        k = canonical([r.get(x) for x in self.key])
                        cell = merged.setdefault(k, {x: r.get(x) for x in (*self.key, *self.col_fields)})
                        v = r.get(value)
                        cell[value] = None if isinstance(v, float) and math.isnan(v) else v
            if self._attr:
                missing = {str(c[self.row_key[0]]) for c in merged.values()} - set(attrs_cache)
                attrs_cache.update(self._attributes(missing))
            for cell in merged.values():
                if self._attr:
                    cell.update(attrs_cache.get(str(cell[self.row_key[0]]), {c: None for c in self.attr_columns}))
                if pred is None or evaluate(pred, cell) is True:
                    yield cell


def _table_of(batch: Any) -> Any:
    import pyarrow as pa

    return pa.Table.from_batches([batch])


def _sortable(v: Any) -> Any:
    if isinstance(v, bool):
        return (0, float(v))
    if isinstance(v, (int, float)):
        return (0, float(v))
    return (1, str(v))


def order_rows(rows: Sequence[Mapping[str, Any]], order: Sequence[Mapping[str, Any]], key: Sequence[str], *,
               group_by: Sequence[str] = ()) -> list[Any]:
    """Rows sorted by ``order`` (``asc``/``desc``/``*_abs``; nulls last), then the canonical key; within
    ``group_by`` groups when given (groups in key order)."""
    def sk(r: Mapping[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = [canonical([r.get(g) for g in group_by])] if group_by else []
        for o in order:
            v = r.get(str(o.get("column")))
            if v is None or (isinstance(v, float) and math.isnan(v)):
                parts.append((1, 0))
                continue
            d = str(o.get("direction", "desc"))
            if d.endswith("_abs") and isinstance(v, (int, float)):
                v = abs(v)
            s = _sortable(v)
            if d.startswith("desc"):
                s = (s[0], -s[1]) if s[0] == 0 else (s[0], _Rev(s[1]))
            parts.append((0, s))
        parts.append(canonical([r.get(k) for k in key]))
        return tuple(parts)

    return sorted(rows, key=sk)


class _Rev(str):
    def __lt__(self, other: str) -> bool:  # type: ignore[override]
        return str.__gt__(self, other)

    def __gt__(self, other: str) -> bool:  # type: ignore[override]
        return str.__lt__(self, other)


def _cut(rows: list[Any], limit: int | None, group_by: Sequence[str]) -> list[Any]:
    if limit is None:
        return rows
    if not group_by:
        return rows[: int(limit)]
    per: dict[str, int] = {}
    out = []
    for r in rows:
        g = canonical([r.get(c) for c in group_by])
        if per.get(g, 0) < int(limit):
            per[g] = per.get(g, 0) + 1
            out.append(r)
    return out


# ---------------------------------------------------------------------------- resolution

def resolver(ctx: ServiceContext) -> Any:
    """The data child's :class:`Resolver`: sidecar indexes built on first use (matrix-axis universes from
    the axis members), cached per universe fingerprint."""
    r = ctx.__dict__.get("_resolver")
    if r is None:
        from ...resolve.resolver import Resolver

        r = ctx.__dict__["_resolver"] = Resolver(ctx.registry, ctx.catalog, lambda s, t: _index(ctx, s, t),
                                                 settings=ctx.settings)
    return r


def _index(ctx: ServiceContext, source: str, id_type: str) -> Any:
    from ...resolve.index import IndexMissing
    from .index_build import _universes, build_resolver_index

    cache = ctx.__dict__.setdefault("_indexes", {})
    try:
        src, spec = ctx.catalog.id_type(f"{source}:{id_type}")
        unis = _universes(src, spec)
        if not unis or spec.index == "remote":
            return None
        fp = ctx.reader(unis[0]["table"]).fingerprint()
    except Exception:  # noqa: BLE001 - an id_type whose universe cannot be read resolves as unknown
        return None
    hit = cache.get((src, id_type))
    if hit is not None and hit[0] == fp:
        return hit[1]
    try:
        if not ctx.index_store.exists(src, fp, id_type):
            if any(str(k).startswith("@") for u in unis for k in u["keys"]):
                ctx.index_store.write_sidecar(src, fp, id_type, _axis_rows(ctx, src, id_type, spec, unis))
            else:
                build_resolver_index(ctx, src, id_type)
        index = ctx.index_store.load(src, fp, id_type)
    except (IndexMissing, ServiceError, OSError, ValueError):
        return None
    cache[(src, id_type)] = (fp, index)
    return index


def _axis_rows(ctx: ServiceContext, src: str, id_type: str, spec: Any, unis: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Resolver rows of an id_type whose universe is a matrix axis (``@col.entrez_id``): ``exact`` per axis
    key, ``label_exact`` per label field of the axis, and the crosswalks the id_type owns."""
    from ...resolve.index import Entry
    from .index_build import _table_ref

    members: dict[str, str] = {}
    labels: list[tuple[str, str, str]] = []
    for u in unis:
        view = long_view(ctx, u["table"])
        if view.matrix is None:
            continue
        for k in u["keys"]:
            axis, _, field = str(k).lstrip("@").partition(".")
            ax = view.matrix.axes[axis]
            label_fields = [n for n, c in {**ax.columns, **(ax.parse.fields if ax.parse else {})}.items()
                            if getattr(c, "role", None) == "label" and getattr(c, "of", None) in (field, None)]
            for r in view.axis(axis):
                key = r.get(field)
                if key in (None, ""):
                    continue
                members.setdefault(str(key), str(next((r.get(f) for f in label_fields if r.get(f)), "") or ""))
                for f in label_fields:
                    if r.get(f) not in (None, ""):
                        labels.append((str(r[f]), str(key), f))
    plugin = ctx.identifier(f"{src}:{id_type}", list(members)[:5000])
    lk = plugin.label_key
    rows = [Entry(lk(k), k, "exact", label) for k, label in members.items()]
    rows += [Entry(lk(text), key, f"label_exact:{field}", text) for text, key, field in labels]
    for cw in spec.crosswalks:
        reader = ctx.reader(_table_ref(src, cw.table))
        for m in reader.scan(None, columns=[cw.from_, cw.to], attribute_unknown=False):
            a, b = m.row.get(cw.from_), m.row.get(cw.to)
            if a is not None and b is not None:
                rows.append(Entry(lk(str(a)), str(b), f"crosswalk:{cw.name}", str(a)))
    return sorted(set(rows), key=Entry.as_row)


def accepted_kinds(ctx: ServiceContext, bound: str) -> list[str]:
    """``bound`` and every id_type with a declared edge path to it (labels, unions, crosswalks, maps_to) whose
    universe can be read here (cached per data child)."""
    cache = ctx.__dict__.setdefault("_accepts", {})
    if bound in cache:
        return list(cache[bound])
    r = resolver(ctx)
    out = [bound]
    for src in sorted(ctx.catalog.sources):
        for name in sorted(ctx.catalog.sources[src].id_types):
            q = f"{src}:{name}"
            if q == bound:
                continue
            try:
                path = r.path(q, bound)
            except Exception:  # noqa: BLE001
                path = None
            if path is None or len(path) > int(ctx.settings.resolution.max_hops):
                continue
            try:
                state = r._universe_state(q)[0]
            except Exception:  # noqa: BLE001
                continue
            if state in ("local", "none"):              # a kind whose universe cannot be read here is not offered
                out.append(q)
    cache[bound] = tuple(out)
    return out


def _resolve_value(ctx: ServiceContext, argument: str, value: Any, bound: str, notes: list[str],
                   resolved: dict[str, str]) -> Any:
    from ...resolve.resolver import error_for

    res = resolver(ctx).resolve(value, accepted_kinds(ctx, bound), bound_id_type=bound)
    err = error_for(res, argument)
    if err is not None:
        raise err
    if res.status == "unknown" or res.canonical is None:
        notes.append(f"{argument}={value!r}: existence could not be decided (index unavailable); used as given")
        return value
    resolved[argument if argument not in resolved else f"{argument}[{len(resolved)}]"] = res.summary()
    return res.canonical


def _snapshot(view: LongView, column: str) -> list[Any] | None:
    spec = view.column(column)
    declared = getattr(spec, "vocab", None)
    if isinstance(declared, list):
        return list(declared)
    if view.matrix is not None:
        return None
    try:
        snap = view.reader.snapshot(view.reader.physical_path(column), max_values=1000)
    except Exception:  # noqa: BLE001 - an unreadable vocabulary leaves the value unchecked
        return None
    return list(snap.values) if snap.complete else None


def compile_where(view: LongView, where: Mapping[str, Any] | None, *, argument: str = "where",
                  notes: list[str] | None = None, resolved: dict[str, str] | None = None
                  ) -> tuple[Predicate | None, dict[str, list[Any]]]:
    """``where`` as a predicate on long-view columns, identifiers resolved; also ``{column: canonical values}``
    for the identifier columns (the matrix reads narrow the axes with them)."""
    notes = notes if notes is not None else []
    resolved = resolved if resolved is not None else {}
    if where is None:
        return None, {}
    if not isinstance(where, Mapping):
        raise _invalid(argument, where, f"{argument} maps long-view columns to values")
    parts: list[Predicate] = []
    keys: dict[str, list[Any]] = {}
    for col, raw in where.items():
        spec = view.column(col)
        if spec is None:
            raise _invalid(f"{argument}.{col}", col, f"{view.ref} has no column {col!r}", sorted(view.columns))
        op, value = _op_value(f"{argument}.{col}", raw)
        bound = view.id_type(col)
        role = getattr(spec, "role", None)
        if bound and op in ("eq", "in", "ne") and getattr(spec, "resolvable", True) is not False:
            values = value if op == "in" else [value]
            canon = [_resolve_value(view.ctx, f"{argument}.{col}", v, bound, notes, resolved) for v in values]
            keys[col] = list(canon)
            value = canon if op == "in" else canon[0]
        elif role in ("category", "scope") and op in ("eq", "in", "ne"):
            vocab = _snapshot(view, col)
            if vocab:                                   # an empty table has no vocabulary to check against
                rendered = {str(json_value(v)): v for v in vocab}
                for v in (value if op == "in" else [value]):
                    if str(json_value(v)) not in rendered:
                        raise _invalid(f"{argument}.{col}", v, f"{v!r} is not a value of {view.ref}.{col}",
                                       [json_value(x) for x in vocab])
        elif op in _CMP and op != "ne" and not isinstance(value, (int, float)):
            raise _invalid(f"{argument}.{col}", value, f"{op} compares numbers")
        parts.append(_leaf(col, op, value))
    if not parts:
        return None, keys
    return (parts[0] if len(parts) == 1 else And(tuple(parts))), keys


def _op_value(argument: str, raw: Any) -> tuple[str, Any]:
    if isinstance(raw, Mapping):
        if len(raw) != 1 or next(iter(raw)) not in WHERE_OPS:
            raise _invalid(argument, raw, f"one operator of {', '.join(WHERE_OPS)} per column", list(WHERE_OPS))
        op, value = next(iter(raw.items()))
        if op == "in" and not isinstance(value, list):
            raise _invalid(argument, value, "in takes a list")
        return str(op), value
    if isinstance(raw, list):
        if not raw:
            raise _invalid(argument, raw, "an empty list matches nothing; pass at least one value")
        return "in", raw
    return "eq", raw


def _leaf(col: str, op: str, value: Any) -> Predicate:
    if op == "eq":
        return Eq(col, value)
    if op == "in":
        return In(col, tuple(value))
    if op == "contains":
        return Contains(col, value)
    if op == "ne":
        return Not(Eq(col, value))
    return Cmp(col, _CMP[op], value)


def parse_order(view: LongView, rank_by: Any, argument: str = "rank_by") -> list[dict[str, Any]]:
    """``rank_by`` (``"col"``, ``"col desc"``, ``{column, direction}`` or a list of them) on measure, count
    or time columns of the long view."""
    if rank_by in (None, "", []):
        return []
    items = rank_by if isinstance(rank_by, list) else [rank_by]
    out = []
    allowed = view.rankable()
    for it in items:
        if isinstance(it, str):
            col, _, direction = it.strip().partition(" ")
            it = {"column": col, "direction": direction.strip() or "desc"}
        if not isinstance(it, Mapping) or it.get("column") not in allowed:
            raise _invalid(argument, it, f"{argument} names a measure, count or time column of {view.ref}", allowed)
        d = str(it.get("direction", "desc"))
        if d not in ("asc", "desc", "asc_abs", "desc_abs"):
            raise _invalid(argument, d, "direction is asc, desc, asc_abs or desc_abs",
                           ["asc", "desc", "asc_abs", "desc_abs"])
        out.append({"column": it["column"], "direction": d, "nulls": "last"})
    return out


def _limit(payload: Mapping[str, Any], default: int, maximum: int, name: str = "limit") -> int:
    v = payload.get(name, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > maximum:
        raise GatewayError(ErrorKind.invalid_argument, f"{name} is an integer from 1 to {maximum}",
                           payload=invalid_argument_payload(name, v, None))
    return v


def _columns_arg(view: LongView, payload: Mapping[str, Any]) -> list[str]:
    cols = payload.get("columns") or []
    if not isinstance(cols, list):
        raise _invalid("columns", cols, "columns is a list of long-view columns", sorted(view.columns))
    bad = [c for c in cols if c not in view.columns]
    if bad:
        raise _invalid("columns", bad[0], f"{view.ref} has no column {bad[0]!r}", sorted(view.columns))
    return list(cols)


def _order_text(order: Sequence[Mapping[str, Any]]) -> str | None:
    if not order:
        return None
    return ", ".join(f"{o['column']} {o['direction']}" for o in order) + " (data child; nulls last, ties by key)"


def _not_covered(view: LongView, keys: Mapping[str, list[Any]]) -> list[str]:
    """Resolved axis keys that are not members of the matrix (the entity exists but is not measured)."""
    if view.matrix is None:
        return []
    out = []
    for axis, cols in (("row", view.row_key), ("col", view.col_key)):
        wanted = [v for c in cols for v in keys.get(c, [])]
        if not wanted:
            continue
        members = {str(r.get(cols[0])) for r in view.axis(axis)}
        out.extend(str(v) for v in wanted if str(v) not in members)
    return out


# ---------------------------------------------------------------------------- verbs

def _max_rows(ctx: ServiceContext) -> int:
    """The largest ``limit`` of a native call: rows of about 2 KB within ``data.memory.max_result_bytes``."""
    return max(1, min(MAX_ROWS, int(ctx.settings.memory.max_result_bytes) // 2000))


# ---------------------------------------------------------------------------- live tables (F20)

_LIVE_OPS = {"eq", "in", "ne", "ge", "gt", "le", "lt", "contains", "search"}
#: Notes of a live answer whose filter included a text matched by the source's engine (``search``).
ENGINE_MATCH_NOTE = ("{cols} matched by the source's search engine (words and synonyms, not the literal text): "
                     "the rows and the total are the engine's matches")


def is_live(ctx: ServiceContext, table: Any) -> bool:
    """The table's layout reads a live source (capability ``live`` without ``scan``): ``find`` and ``lookup``
    are answered by ``_live_find`` (one request of a few pages within the source's budget), never a scan."""
    try:
        layout = ctx.plugin("layout", table.layout)
    except Exception:  # noqa: BLE001
        return False
    caps = set(getattr(layout, "capabilities", ()) or ())
    return "live" in caps and "scan" not in caps


def live_column(table: Any, path: str) -> Any:
    """The declared column a dotted path names on a live table (``protocolSection.identificationModule.nctId``),
    or True for a path inside a column declared without fields (a payload: its JSON is the source's), else None."""
    parts = [p for p in str(path).replace("[]", "").split(".") if p]
    cols: Any = table.columns
    for i, part in enumerate(parts):
        col = (cols or {}).get(part) if isinstance(cols, Mapping) else None
        if col is None:
            return None
        if i == len(parts) - 1:
            return col
        fields = getattr(col, "fields", None)
        if not fields:
            return True if getattr(col, "role", None) == "payload" else None
        cols = fields
    return None


def _engine_text(table: Any, path: str) -> bool:
    """A text column the source's search engine matches (``remote_name`` on a ``text`` role): words and synonyms,
    never the literal value (``AREA[BriefTitle]Cancer`` matched titles without the word)."""
    col = live_column(table, path)
    return getattr(col, "role", None) == "text" and bool(getattr(col, "remote_name", None))


def _live_predicate(table: Any, where: Any, notes: list[str] | None = None) -> Predicate | None:
    """``where`` of a live find: ``{column: value | [values] | {op: value}}`` on the table's columns (dotted
    paths into a nested column allowed; a pivoted table's attribute columns are named by the source);
    identifiers are sent as given (the source resolves them). A text column the source's engine matches takes
    ``{search: text}`` only (an equality would be answered with the engine's word and synonym matches)."""
    from ...predicate import TextMatch

    if where is None:
        return None
    if not isinstance(where, Mapping):
        raise _invalid("where", where, "where maps columns to values")
    parts: list[Predicate] = []
    engine: list[str] = []
    for col, v in where.items():
        if table.spec.pivot is None and str(col).split(".")[0].split("[")[0] not in table.columns:
            raise _invalid("where", col, f"{table.ref} has no column {col!r}", sorted(table.columns))
        text = table.spec.pivot is None and _engine_text(table, str(col))
        ops = dict(v) if isinstance(v, Mapping) else {"in" if isinstance(v, list) else "eq": v}
        if text and set(ops) != {"search"}:
            raise GatewayError(ErrorKind.invalid_argument,
                               f"{col} is matched by the source's search engine (words and synonyms), never compared "
                               f"as a value: pass {{\"{col}\": {{\"search\": \"<text>\"}}}}",
                               payload=invalid_argument_payload("where", {col: v}, ["search"]))
        for op, x in ops.items():
            if op not in _LIVE_OPS:
                raise _invalid("where", op, f"unknown operator {op!r}", sorted(_LIVE_OPS))
            if op == "search":
                if not text or not isinstance(x, str) or not x.strip():
                    raise _invalid("where", {col: v}, f"search takes a text and applies to text columns the source's "
                                   f"engine matches; {col} is not one")
                parts.append(TextMatch(str(col), x.strip(), "word"))
                engine.append(str(col))
                continue
            parts.append(Eq(col, x) if op == "eq" else In(col, tuple(x if isinstance(x, list) else [x]))
                         if op == "in" else Not(Eq(col, x)) if op == "ne" else Contains(col, x)
                         if op == "contains" else Cmp(col, op, x))
    if engine and notes is not None:
        notes.append(ENGINE_MATCH_NOTE.format(cols=", ".join(engine)))
    return None if not parts else (parts[0] if len(parts) == 1 else And(tuple(parts)))


def _live_columns(table: Any, payload: Mapping[str, Any]) -> list[str]:
    """``columns`` of a live find: declared column paths (a pivoted table's attributes are the source's)."""
    cols = payload.get("columns") or []
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise _invalid("columns", cols, "columns is a list of column paths", sorted(table.columns))
    if table.spec.pivot is None:
        bad = [c for c in cols if live_column(table, c) is None]
        if bad:
            raise _invalid("columns", bad[0], f"{table.ref} has no column {bad[0]!r} (name nested columns by their "
                           "dotted path)", sorted(table.columns))
    return list(cols)


def _default_filters(table: Any, pred: Predicate | None, where: Any, notes: list[str]) -> Predicate | None:
    """Qualifier columns with ``default_filter`` (Census ``is_primary_data``: a cell's other copies are duplicates)
    are fixed to true unless ``where`` names them; the filter is disclosed. A native find counted 158 duplicate
    cells where single_cell.count_cells, which adds the same filter, answered 0."""
    named = {str(c).split(".")[0] for c in (where or {})} if isinstance(where, Mapping) else set()
    added: list[str] = []
    parts = [] if pred is None else (list(pred.preds) if isinstance(pred, And) else [pred])
    for name, col in (table.columns or {}).items():
        if getattr(col, "role", None) == "qualifier" and getattr(col, "default_filter", False) and name not in named:
            parts.append(Eq(name, True))
            added.append(name)
    if added:
        effect = {getattr(table.columns[n], "effect", None) for n in added}
        why = "duplicate rows excluded" if effect == {"duplicate"} else "the declared default"
        notes.append(f"{', '.join(f'{n} == true' for n in added)} added ({why}; name "
                     f"{added[0]} in where, e.g. [true, false], to include them)")
    return None if not parts else (parts[0] if len(parts) == 1 else And(tuple(parts)))


def _last_field(path: str) -> str:
    """``studyFirstPostDateStruct`` of ``protocolSection.statusModule.studyFirstPostDateStruct.date``: the last named
    field of a date path (a trailing ``date`` names the struct that holds it)."""
    parts = [p for p in str(path).replace("[]", "").split(".") if p]
    if len(parts) > 1 and parts[-1] == "date":
        return parts[-2]
    return parts[-1] if parts else str(path)


def _live_find(ctx: ServiceContext, table: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    from ...predicate import to_json
    from .. import layout_spec
    from .witness import live_find, requested_keys

    ref = str(table.ref)
    notes: list[str] = []
    pred = _live_predicate(table, payload.get("where"), notes)
    if requested_keys(pred, [c for c in table.key if not str(c).endswith("#")]) is None:
        pred = _default_filters(table, pred, payload.get("where"), notes)   # a named record is read as named
    columns = _live_columns(table, payload)
    limit = _limit(payload, 50, _max_rows(ctx))
    budget = getattr(table.descriptor, "budget", None)
    max_pages = getattr(budget, "max_pages", None) if budget is not None else None
    if not max_pages:
        # a single unpaged read (a SOMA frame): admitted count-first, never an unbounded read
        layout = ctx.plugin("layout", table.layout)
        try:
            n = layout.count(layout_spec(table), predicate=pred, budget=budget)
        except Exception as exc:  # noqa: BLE001 - an unanswerable count admits nothing
            n, why = None, str(exc)
        else:
            why = "the filter does not compile into the source's own filter"
        if n is None or n > _max_rows(ctx):
            raise GatewayError(ErrorKind.too_large, f"{ref}: a find must select at most {_max_rows(ctx)} rows "
                               f"({'counted ' + str(n) if n is not None else why}); narrow where",
                               payload={"table": ref, "count": n})
    got = live_find(ctx, {"table": ref, "predicate": to_json(pred) if pred is not None else None,
                          "columns": columns, "limit": limit})
    rows = list(got.get("rows") or [])
    if got.get("cut_after_read"):
        # the whole match was read (one unpaged request: the Census) and cut to limit here
        notes.append(f"every matching row was read ({got.get('total')}); the first {len(rows)} in the source's order "
                     "are returned (limit), not a page prefix")
    elif got.get("truncated"):
        notes.append(f"{got.get('pages')} page(s) read within the source's budget: the rows are a prefix, not all")
    if got.get("source_updated"):
        notes.append("records changed at the source since an earlier call: " + ", ".join(
            map(str, got["source_updated"][:10])))
    leakage = got.get("leakage") or None
    withheld = None
    if leakage:
        n_out = int(leakage.get("withheld") or 0)
        withheld = {"leakage": n_out} if n_out else None
        how = ("sent with the request and the count (records available and last changed by then)"
               if leakage.get("sent") else "checked on the records named")
        notes.append(f"evidence ceiling {leakage['ceiling']} applied: {how}; {n_out} row(s) withheld as dated "
                     "after it")
        if len(leakage.get("bounded_on") or []) > 1:
            # two totals for one filter under one ceiling: this find counts what it may return, an upstream count
            # tool counts every record available by the ceiling (the registry's own first-posted bound)
            avail, changed = leakage["bounded_on"][0], leakage["bounded_on"][1]
            notes.append(f"the total counts records available ({_last_field(avail)}) and last changed "
                         f"({_last_field(changed)}) by {leakage['ceiling']}, the rows this find can return; an "
                         f"upstream count under the same ceiling bounds only {_last_field(avail)} and also counts the "
                         "records changed after it (which are withheld from rows), so its total is larger")
        if leakage.get("redacted"):
            notes.append(f"{leakage['redacted']} row(s) redacted (fields changed after the ceiling nulled)")
    # keys the filter named that the source did not return (and that no ceiling withheld): unknown to the source
    asked = got.get("requested_keys")
    missing: list[Any] = []
    if asked is not None and not got.get("truncated"):
        key = [c for c in table.key if not str(c).endswith("#")]
        from ...gateway.fields import get_path

        seen = {json.dumps([get_path(r, c) for c in key], default=str) for r in rows if isinstance(r, Mapping)}
        held = {json.dumps([w.get(c) for c in key] if isinstance(w, Mapping) else [w], default=str)
                for w in got.get("withheld_keys") or []}
        for k in asked:
            parts = [k.get(c) for c in key] if isinstance(k, Mapping) else [k]
            text = json.dumps(parts, default=str)
            if text not in seen and text not in held:
                missing.append(k)
    if missing and payload.get("_lookup") and len(asked or []) == 1:
        k = missing[0]
        raise GatewayError(ErrorKind.not_found, f"{ref}: no record with key {json.dumps(k, default=str)} at the source",
                           payload={"table": ref, "key": k})
    if missing:
        notes.append(f"{len(missing)} requested key(s) not at the source: "
                     + ", ".join(json.dumps(k, default=str) for k in missing[:10]))
    status = "ok" if rows else "empty"
    if rows and (got.get("truncated") or missing):
        status = "partial"
    coverage, statement = coverage_of(table) if not rows else (None, None)
    as_of = got.get("as_of")
    src = table.descriptor.source
    hdr = Header(status=status, source=f"{src}@{as_of}" if as_of else src, tables=[ref],  # type: ignore[arg-type]
                 key=list(table.key), returned=len(rows), total=got.get("total"),
                 total_method="remote" if got.get("total") is not None else "unknown",
                 truncated=bool(got.get("truncated")), withheld=withheld, not_found_items=missing or None,
                 coverage=coverage, coverage_statement=statement,  # type: ignore[arg-type]
                 served_by="derived", notes=notes,
                 extra={"source_updated": got.get("source_updated") or None,
                        "leakage": {k: leakage[k] for k in ("ceiling", "withheld") if k in leakage} | {
                            "risk": bool(leakage.get("unchecked"))} if leakage else None})
    return inject_header({"rows": json_value(rows), "record_versions": got.get("record_versions") or None}, hdr)


def find(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rows of a table's long view matching ``where`` (identifiers resolved), ranked by ``rank_by`` (else
    the table's rank), cut to ``limit`` (per ``group_by`` group), one per ``distinct`` combination. A live
    table (``live`` layout) is read by ``_live_find`` within the source's request budget."""
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    if is_live(ctx, table):
        for arg in ("rank_by", "group_by", "distinct"):
            if payload.get(arg):
                raise GatewayError(ErrorKind.unsupported_combination, f"{arg} is not available on the live table "
                                   f"{table.ref}: the source orders and pages it", argument=arg)
        return _live_find(ctx, table, payload)
    view = long_view(ctx, str(table.ref))
    notes: list[str] = []
    resolved: dict[str, str] = {}
    pred, keys = compile_where(view, payload.get("where"), notes=notes, resolved=resolved)
    order = parse_order(view, payload.get("rank_by"))
    limit = _limit(payload, 50, _max_rows(ctx))
    group_by = list(payload.get("group_by") or [])
    distinct = list(payload.get("distinct") or [])
    for arg, cols in (("group_by", group_by), ("distinct", distinct)):
        bad = [c for c in cols if c not in view.columns]
        if bad:
            raise _invalid(arg, bad[0], f"{view.ref} has no column {bad[0]!r}", sorted(view.columns))
    columns = _columns_arg(view, payload)
    missing = _not_covered(view, keys)
    if missing:
        notes.append(f"not measured in {view.ref}: {', '.join(missing[:10])}")
        hdr = header(view, rows=[], total=0, covered=False, resolved=resolved, notes=notes)
        return _result(view, [], hdr)
    col_keys = row_keys = None
    if view.matrix is not None:
        col_keys = keys.get(view.col_key[0]) if view.col_key else None
        row_keys = keys.get(view.row_key[0]) if view.row_key else None
        col_keys = [str(k) for k in col_keys] if col_keys else None
        row_keys = [str(k) for k in row_keys] if row_keys else None
    rows, total, eu = view.rows(pred, columns=columns, order=order, limit=limit, group_by=group_by,
                                distinct=distinct, budget=payload.get("budget_bytes"), col_keys=col_keys,
                                row_keys=row_keys)
    returned_total = total if not distinct and not group_by else None
    hdr = header(view, rows=rows, total=returned_total, truncated=len(rows) < total,
                 order=_order_text(order or view.default_order()), resolved=resolved, excluded_unknown=eu,
                 notes=notes)
    return _result(view, rows, hdr)


def lookup(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The record of a complete key (every key column of the long view; identifiers resolved)."""
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    key = payload.get("key")
    live = is_live(ctx, table)
    view = None if live else long_view(ctx, str(table.ref))
    names = [k for k in (table.key if view is None else view.key) if not k.endswith("#")]
    if not isinstance(key, Mapping):
        raise _invalid("key", key, "key maps every key column to one value", names)
    # a key column is named by its full path or its last field (nctId for protocolSection...nctId), live or not
    short = {k.split(".")[-1].replace("[]", ""): k for k in names}
    missing = [s for s, k in short.items() if s not in key and k not in key]
    if missing:
        raise GatewayError(ErrorKind.incomplete_key, f"key misses {', '.join(missing)} of {table.ref}",
                           payload={"argument": "key", "missing": missing, "key": names})
    if view is None:
        where = {k: key.get(k, key.get(s)) for s, k in short.items()}
        # a live record: the source is the authority on its keys (a key it does not hold is not_found)
        return find(ctx, {"table": str(table.ref), "where": where, "limit": 10, "agent": payload.get("agent"),
                          "columns": payload.get("columns"), "_lookup": True})
    where = {(k if k in view.columns else s): key.get(s, key.get(k)) for s, k in short.items()}
    return find(ctx, {"table": str(table.ref), "where": where, "limit": 10, "agent": payload.get("agent")})


def search(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rows whose key, label or synonym matches ``text``, ordered by match class (exact, casefold,
    previous, alias, ..., prefix, word, substring), then the table's rank, then the key."""
    from .serve import search as _search

    table = table_access(ctx, payload.get("table") or payload.get("entity"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _invalid("text", text, "text is a non-empty string")
    if view.matrix is not None:
        raise GatewayError(ErrorKind.unsupported_combination, f"{view.ref} is a matrix: search its axes with find",
                           payload={"alternatives": ["data.find"]})
    limit = _limit(payload, 20, _max_rows(ctx))
    rows, _keys, total = _search(view.reader, text, None, limit, {}, ctx.scan_budget(view.table, None))
    hdr = header(view, rows=rows, total=total, truncated=total > len(rows), order="match class, rank, key")
    return _result(view, rows, hdr)


def vocab(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The distinct values of a category or scope column (storage-typed, with counts when scanned)."""
    from .vocab import vocab as _vocab

    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    col = payload.get("column")
    if col not in view.columns:
        raise _invalid("column", col, f"{view.ref} has no column {col!r}", sorted(view.columns))
    if view.matrix is not None:
        values = sorted({str(r.get(col)) for ax in ("row", "col") for r in view.axis(ax) if r.get(col) is not None})
        hdr = header(view, rows=values, total=len(values), key=[col])
        return _result(view, values, hdr, complete=True)
    out = _vocab(ctx, {"table": str(table.ref), "column": col, "max_values": payload.get("max_values")})
    hdr = header(view, rows=out["values"], total=len(out["values"]), truncated=not out["complete"], key=[col])
    return _result(view, out["values"], hdr, complete=out["complete"], counts=out.get("counts"))


def members(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The members of one set of a ``sets`` table (direct membership; ``hierarchy.py`` wraps this verb for
    ``propagate: true``)."""
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    if payload.get("propagate"):
        raise GatewayError(ErrorKind.unsupported_combination, "propagate: membership propagated over a hierarchy "
                           "arrives in phase 3; this returns direct members", payload={"argument": "propagate"})
    member_cols = [n for n, c in view.columns.items() if getattr(c, "role", None) == "member"]
    if not member_cols:
        raise _invalid("table", view.ref, f"{view.ref} has no member column")
    mcol = view.columns[member_cols[0]]
    parent = getattr(getattr(mcol, "membership", None), "set", None)
    set_col = getattr(parent, "parent", None) or view.key[0]
    where = {set_col: payload.get("set_id")}
    notes: list[str] = []
    resolved: dict[str, str] = {}
    pred, _keys = compile_where(view, where, argument="set_id", notes=notes, resolved=resolved)
    rows, total, _eu = view.rows(pred, columns=[set_col, member_cols[0]], limit=None)
    out = []
    for r in rows:
        for m in r.get(member_cols[0]) or []:
            out.append({set_col: r.get(set_col), "member": m})
    prop = getattr(getattr(mcol, "membership", None), "propagation", None)
    if prop:
        notes.append(f"membership is {prop}")
    hdr = header(view, rows=out, total=len(out), resolved=resolved, notes=notes, key=[set_col, "member"])
    return _result(view, out, hdr)


def neighbors(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Edges of an ``edges`` table touching ``node`` on either side, one hop (``network.py`` wraps this verb
    for several nodes and hops)."""
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    hops = payload.get("hops", 1)
    if hops != 1:
        raise GatewayError(ErrorKind.unsupported_combination, "neighbors is limited to one hop until phase 3",
                           payload={"argument": "hops", "value": hops})
    edge = view.table.spec.edge
    if edge is None:
        raise _invalid("table", view.ref, f"{view.ref} is not an edges table")
    notes: list[str] = []
    resolved: dict[str, str] = {}
    node = payload.get("node")
    pa_, _k = compile_where(view, {edge.a: node}, argument="node", notes=notes, resolved=resolved)
    canon = next(iter(_k.get(edge.a) or [node]))
    pred: Predicate = Or((Eq(edge.a, canon), Eq(edge.b, canon)))
    extra, _ = compile_where(view, payload.get("where"), notes=notes, resolved=resolved)
    if extra is not None:
        pred = And((pred, extra))
    order = parse_order(view, payload.get("rank_by"))
    limit = _limit(payload, 50, _max_rows(ctx))
    within = [w for o in view.table.spec.rank for w in o.within]
    rows, total, eu = view.rows(pred, order=order, limit=limit, group_by=within)
    for r in rows:
        r["partner"] = r.get(edge.b) if r.get(edge.a) == canon else r.get(edge.a)
    hdr = header(view, rows=rows, total=total, truncated=len(rows) < total,
                 order=(_order_text(order or view.default_order()) or "key") + (f" within {', '.join(within)}" if within else ""),
                 resolved=resolved, excluded_unknown=eu, notes=notes)
    return _result(view, rows, hdr)


def resolve(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Each value resolved to the id_type's canonical key by a recorded rule (or its typed outcome)."""
    id_type = payload.get("id_type")
    try:
        bound = ctx.catalog.qualify_id_type(str(id_type))
    except LookupError:
        names = sorted(f"{s}:{n}" for s, d in ctx.catalog.sources.items() for n in d.id_types)
        raise _invalid("id_type", id_type, f"unknown id_type {id_type!r}", names) from None
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise _invalid("values", values, "values is a non-empty list")
    r = resolver(ctx)
    accepts = accepted_kinds(ctx, bound)
    out = []
    for v in values:
        res = r.resolve(v, accepts, bound_id_type=bound)
        out.append({"value": v, "status": res.status, "canonical": res.canonical, "rule": res.rule,
                    "matched_id_type": res.matched_id_type, "label": res.label,
                    "candidates": [{"id": c.id, "label": c.label, "via": c.via} for c in res.candidates][:10],
                    "suggestions": [c.id for c in res.suggestions][:5], "tried": list(res.tried)[:10]})
    ok = sum(1 for o in out if o["status"] in ("resolved", "resolved_unverified"))
    hdr = Header(status="ok" if ok else "empty", returned=len(out), total=len(out), served_by="derived",
                 key=["value"], extra={"id_type": bound, "resolved_count": ok})
    return inject_header({"rows": json_value(out)}, hdr)


def _describe(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    from .describe import describe

    return describe(ctx, payload)


def _aggregate(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    from .aggregate import aggregate

    return aggregate(ctx, payload)


def _similar(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    from .similar import similar

    return similar(ctx, payload)


# ---------------------------------------------------------------------------- the in-process client's reads


def rooted(ctx: ServiceContext, source: str, root: str | None) -> ServiceContext:
    """``ctx`` with ``source`` read from ``root`` instead of its configured root (the client's readers name
    the directory they were given); cached per (source, root)."""
    if not root:
        return ctx
    from ...catalog import Catalog

    key = (source, str(Path(root).resolve()))
    cache = ctx.__dict__.setdefault("_rooted", {})
    hit = cache.get(key)
    if hit is None:
        sources = dict(ctx.catalog.sources)
        if source not in sources:
            raise _invalid("table", source, f"unknown source {source!r}", sorted(sources))
        sources[source] = sources[source].model_copy(update={"root": key[1]})
        catalog = Catalog(sources, ctx.catalog.overlays, ctx.catalog.generic, aliases=ctx.catalog.aliases,
                          registry=ctx.registry)
        hit = cache[key] = ServiceContext(ctx.settings, catalog=catalog, registry=ctx.registry)
    return hit


def ready_or_raise(ctx: ServiceContext, ref: str) -> str:
    """Shallow readiness of one table (layout listing, readable fragments, manifest), else ``not_ready``.
    Returns the table's fingerprint."""
    from ...gateway.readiness import READY_STATUSES, table_status
    from ..checks import check_table

    model = check_table(ctx, ref, "shallow")
    status = table_status(model)
    if status not in READY_STATUSES:
        failed = [f"{c.name}: {c.detail}" for c in model.checks if not c.ok and c.level == "error"]
        raise GatewayError(ErrorKind.not_ready, f"{ref} is not ready ({status}): {'; '.join(failed[:3])}",
                           payload={"tables": [ref], "status": status})
    return str(getattr(model, "fingerprint", None) or ctx.reader(ref).fingerprint())


def _client_table(ctx: ServiceContext, payload: Mapping[str, Any]) -> tuple[ServiceContext, Any]:
    ref = payload.get("table")
    if not isinstance(ref, str) or ref.count(".") != 1:
        raise _invalid("table", ref, "table names a table as 'source.table'", ctx.table_refs())
    sub = rooted(ctx, ref.split(".")[0], payload.get("root"))
    table = table_access(sub, ref, agent=payload.get("agent"), native=bool(payload.get("native", True)))
    return sub, table


def _expand(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``expand`` (§10.6): the hidden ``_expand`` verb (hierarchy.py) as a public tool."""
    from .hierarchy import expand_verb

    return expand_verb(ctx, payload)


def _enrich(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``enrich`` (§10.6): the hidden ``_enrich`` verb (enrich.py) on a table exposed to the native tools."""
    from .enrich import enrich

    table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    return enrich(ctx, payload)


def materialize(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``_materialize``: the rows of ``table`` matching ``where`` (identifiers resolved), projected on
    ``columns``, written as one Parquet file under ``out_dir`` for the in-process client (a backed read:
    the rows never travel through JSON). Parquet tables without partitions are copied in file order with
    their storage types; other tables are written from the long view."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sub, table = _client_table(ctx, payload)
    ref = str(table.ref)
    fingerprint = ready_or_raise(sub, ref)
    view = long_view(sub, ref)
    columns = _columns_arg(view, payload)
    notes: list[str] = []
    resolved: dict[str, str] = {}
    pred, _keys = compile_where(view, payload.get("where"), notes=notes, resolved=resolved)
    out_dir = Path(str(payload.get("out_dir") or sub.settings.cache_dir)) / "client"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(json.dumps([ref, fingerprint, columns, payload.get("where"), payload.get("root")],
                                    sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = out_dir / f"{ref}.{tag}.parquet"
    reader = view.reader
    files: list[str | None] = []
    plain = view.matrix is None and not table.is_item_table and table.format == "parquet" and not reader.partitions
    if plain:
        from ...plugins.layouts.zip_member import local_path

        files = [local_path(f) for f in reader.fragments()]
        plain = all(f is not None for f in files)
    if plain:
        import pyarrow.dataset as pads

        data = pads.dataset(sorted(str(f) for f in files), format="parquet")
        expr = _arrow_filter(pred, set(data.schema.names))
        if pred is None or expr is not None:
            tbl = data.to_table(columns=columns or None, filter=expr)
        else:                                          # predicates arrow cannot express: the three-valued oracle
            full = data.to_table()
            keep = [i for i, r in enumerate(full.to_pylist()) if evaluate(pred, r) is True]
            tbl = full.take(pa.array(keep, pa.int64()))
            tbl = tbl.select(columns) if columns else tbl
    else:
        rows, _total, _eu = view.rows(pred, columns=columns, order=[], limit=None, budget=payload.get("budget_bytes"))
        tbl = pa.Table.from_pylist(rows)
    pq.write_table(tbl, path)
    hdr = header(view, rows=[None] * tbl.num_rows, total=tbl.num_rows, resolved=resolved, notes=notes,
                 extra={"fingerprint": fingerprint})
    return inject_header({"path": str(path), "n_rows": tbl.num_rows, "columns": tbl.column_names,
                          "fingerprint": fingerprint, "table": ref}, hdr)


def _arrow_filter(pred: Predicate | None, names: set[str]) -> Any:
    """An Arrow expression for ``Eq``/``In`` conjunctions on top-level columns (null never matches, as in the
    three-valued oracle); None for anything else."""
    import pyarrow.dataset as pads

    if pred is None:
        return None
    if isinstance(pred, Eq) and pred.column in names and pred.value is not None:
        return pads.field(pred.column) == pred.value
    if isinstance(pred, In) and pred.column in names and None not in pred.values:
        return pads.field(pred.column).isin(list(pred.values))
    if isinstance(pred, And):
        parts = [_arrow_filter(p, names) for p in pred.preds]
        if all(p is not None for p in parts):
            out = parts[0]
            for p in parts[1:]:
                out = out & p
            return out
    return None


def matrix(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``_matrix``: the fragments of a matrix table (``fragment``: a file name, stem or fragment key; all when
    omitted), checked ready, with local path, fingerprint and shape, for the client's ``open_matrix``."""
    from ...plugins.layouts.zip_member import local_path

    sub, table = _client_table(ctx, payload)
    ref = str(table.ref)
    if table.kind != "matrix":
        raise _invalid("table", ref, f"{ref} is not a matrix table")
    fingerprint = ready_or_raise(sub, ref)
    reader = sub.reader(ref)
    want = payload.get("fragment")
    frags = reader.fragments()

    def names(f: Any) -> set[Any]:
        p = Path(local_path(f) or f.uri)
        return {f.fragment_key, p.name, p.stem}

    chosen = [f for f in frags if want is None or want in names(f)]
    if not chosen:
        raise GatewayError(ErrorKind.not_found, f"{ref} has no fragment {want!r}", argument="fragment", value=want,
                           payload={"valid_values": sorted(str(f.fragment_key or Path(f.uri).name) for f in frags)})
    out = []
    for f in chosen:
        try:
            shape = reader.fmt.stats(f).shape
        except Exception:  # noqa: BLE001 - the shape is informative only
            shape = None
        out.append({"path": local_path(f) or f.uri, "fragment_key": f.fragment_key, "size": f.size,
                    "shape": list(shape) if shape else None})
    hdr = header(long_view(sub, ref), rows=out, total=len(out), extra={"fingerprint": fingerprint})
    return inject_header({"table": ref, "fingerprint": fingerprint, "fragments": out}, hdr)


VERBS = {
    "resolve": guarded("resolve", resolve),
    "describe": guarded("describe", _describe),
    "lookup": guarded("lookup", lookup),
    "find": guarded("find", find),
    "search": guarded("search", search),
    "vocab": guarded("vocab", vocab),
    "members": guarded("members", members),
    "aggregate": guarded("aggregate", _aggregate),
    "similar": guarded("similar", _similar),
    "neighbors": guarded("neighbors", neighbors),
    "expand": guarded("expand", lambda ctx, payload: _expand(ctx, payload)),
    "enrich": guarded("enrich", lambda ctx, payload: _enrich(ctx, payload)),
}
assert tuple(VERBS) == PUBLIC_VERBS
VERBS.update({"_materialize": guarded("materialize", materialize), "_matrix": guarded("matrix", matrix)})


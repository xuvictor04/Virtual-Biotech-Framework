"""``_expand`` and hierarchy semantics (§11.5 hierarchy expansion, phase 3, F18).

An id_type's hierarchy is declared once (``IdTypeSpec.hierarchy: {table, columns, predicates,
reflexive}``), possibly by another source that ``extends`` the type (``gene_ontology:go_term`` adds the
go-basic.obo DAG to ``open_targets:go_term``). An id_type without one falls back to the hierarchy
columns of its universe table (``disease.children``, ``reactome.parents``). From it this module builds a
**closure**: the direct parent edges whose predicate the hierarchy declares (GO ``is_a`` and
``part_of`` propagate, ``regulates`` does not: an edge from a nested ``relationship[].target`` counts
only when its sibling ``type`` is a declared predicate), checked for cycles, and kept in memory and
in a sidecar ``<cache>/<source>/<fingerprint>/closure/<id_type>.json.gz`` keyed by the hierarchy
table's fingerprint and the predicates (a changed OBO file is a new fingerprint, never a stale
closure).

* :func:`expand` returns ``{term: depth}`` (descendants or ancestors; the term itself at depth 0 only
  when asked) and an **expansion record** (id_type, term, direction, predicates, hierarchy table,
  fingerprint, size, maximum depth) that goes into provenance. More than ``max_expand`` terms
  (``data.resolution.max_expand``) is ``too_large`` with ``subkind: expansion``; a term inside a
  cycle is ``not_ready`` (``subkind: hierarchy_cycle``), never a silently truncated walk.
* :func:`descendant_predicate` is ``include_descendants(X)``: ``Eq(col, X) OR Contains(<ancestor
  column>, X)`` when the bound table carries an ancestor list of that column (OT ``known_drug.ancestors``
  excludes the term itself), else ``In(col, expand(X) ∪ {X})``. A column declared ``propagated_over``
  (indirect associations) refuses it (``unsupported_combination``): its rows already include descendants.
* The ``_expand`` hidden verb: ``{id_type, values | value, direction?, include_self?, max_expand?}``
  (values resolved like any identifier argument) -> rows ``{of, term, depth}``.
* ``members`` with ``propagate: true`` (and derived ``members`` requests whose set id_type has a
  hierarchy and whose membership is ``mixed`` or ``propagated``): the members of the set **and of
  every descendant set**, each member once, with ``via`` naming the set it was annotated to (direct
  annotations first) and a note disclosing the propagation. ``find_genes_in_pathway`` on a top-level
  Reactome pathway therefore lists the genes of its sub-pathways instead of an empty answer.

Routing. :func:`serve_extension` registers phase-3 handlers for the gateway's ``_serve`` requests and
wraps the ``_serve`` verb once (``functools.wraps``; requests no handler claims go to ``serve.py``
unchanged). Handlers here claim ``verb: expand`` and requests whose ``params`` set
``include_descendants`` (the predicate is rewritten before the ordinary read), and ``verb: members``
over a hierarchy. ``enrich.py``, ``network.py`` and ``pairs.py`` register theirs the same way.
"""

from __future__ import annotations

import functools
import gzip
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ...errors import ErrorKind, GatewayError, json_value, not_ready_payload, too_large_payload
from ...ipc import VERB_EXPAND, VERB_SERVE, ServeResponse
from ...predicate import And, Contains, Eq, In, Or, Predicate, from_json, to_json
from ...result import inject_header
from ...rowkey import canonical
from .. import ServiceContext, ServiceError
from . import public as _public
from . import serve as _serve_module
from .public import _invalid, compile_where, guarded, header, long_view, table_access

__all__ = [
    "Closure", "closure_for", "hierarchy_of", "expand", "descendant_predicate", "rewrite_include_descendants",
    "propagated_members", "members", "expand_verb", "serve_extension", "base_serve", "member_column",
    "HierarchySource", "VERBS",
]

#: ``_serve`` handlers of phase 3: ``(name, claims(request) -> bool, handle(ctx, request) -> response)``.
_EXTENSIONS: list[tuple[str, Callable[[Mapping[str, Any]], bool], Callable[[ServiceContext, Mapping[str, Any]], Any]]] = []


def serve_extension(name: str, claims: Callable[[Mapping[str, Any]], bool]
                    ) -> Callable[[Callable[[ServiceContext, Mapping[str, Any]], Any]], Any]:
    """Register a ``_serve`` handler claiming the requests ``claims`` accepts (first registered wins)."""

    def deco(fn: Callable[[ServiceContext, Mapping[str, Any]], Any]) -> Any:
        if not any(n == name for n, _c, _f in _EXTENSIONS):
            _EXTENSIONS.append((name, claims, fn))
        _install()
        return fn

    return deco


def _install() -> None:
    current = _serve_module.VERBS[VERB_SERVE]
    if getattr(current, "_phase3", False):
        return
    base = current

    @functools.wraps(base)
    def serve(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
        for _name, claims, fn in _EXTENSIONS:
            try:
                claimed = claims(payload)
            except Exception:  # noqa: BLE001 - a malformed request is left to serve.py's validation
                claimed = False
            if claimed:
                return fn(ctx, payload)
        return base(ctx, payload)

    serve._phase3 = True  # type: ignore[attr-defined]
    serve._base = base  # type: ignore[attr-defined]
    _serve_module.VERBS[VERB_SERVE] = serve


def base_serve(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``serve.py``'s own ``_serve`` (no phase-3 routing)."""
    fn = _serve_module.VERBS[VERB_SERVE]
    return getattr(fn, "_base", fn)(ctx, payload)


# ---------------------------------------------------------------------------- the hierarchy of an id_type


@dataclass(frozen=True)
class HierarchySource:
    """Where an id_type's hierarchy lives: ``table`` (qualified), its edge ``columns`` and ``predicates``."""

    id_type: str
    table: str
    columns: tuple[str, ...]
    predicates: tuple[str, ...]
    reflexive: bool = False
    declared_by: str = ""


def hierarchy_of(ctx: ServiceContext, id_type: str) -> HierarchySource | None:
    """The hierarchy of ``id_type`` (``source:name``): its own ``hierarchy``, one declared by an id_type that
    ``extends`` it, else the parent/child hierarchy columns of its universe table; None without any."""
    try:
        src, spec = ctx.catalog.id_type(id_type)
    except LookupError:
        return None
    qualified = f"{src}:{id_type.partition(':')[2] or id_type}" if ":" in id_type else f"{src}:{id_type}"
    candidates: list[tuple[str, str, Any]] = []
    if spec.hierarchy is not None:
        candidates.append((src, qualified, spec.hierarchy))
    for other_src in sorted(ctx.catalog.sources):
        for name, other in sorted(ctx.catalog.sources[other_src].id_types.items()):
            if other.extends == qualified and other.hierarchy is not None:
                candidates.append((other_src, f"{other_src}:{name}", other.hierarchy))
    for owner, declared_by, h in candidates:
        table = h.table if "." in h.table else f"{owner}.{h.table}"
        try:
            ctx.table(table)
        except ServiceError:
            continue
        return HierarchySource(qualified, table, tuple(h.columns), tuple(h.predicates), bool(h.reflexive), declared_by)
    universe = spec.universe
    if isinstance(universe, str) and "." in universe:
        table = f"{src}.{universe.split('.')[0]}"
        try:
            t = ctx.table(table)
        except ServiceError:
            return None
        cols = [n for n, c in t.columns.items() if getattr(c, "role", None) == "hierarchy" and
                getattr(c, "relation", None) in ("parent", "child") and getattr(c, "closure", None) != "transitive"]
        if cols:
            return HierarchySource(qualified, table, tuple(cols), (), False, f"{table} hierarchy columns")
    return None


# ---------------------------------------------------------------------------- the closure


@dataclass
class Closure:
    """Direct parent edges of one hierarchy (the declared predicates only) with cycle information."""

    source: HierarchySource
    fingerprint: str
    parents: dict[str, list[str]] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    cycles: list[list[str]] = field(default_factory=list)
    edges_skipped: dict[str, int] = field(default_factory=dict)       # predicate -> edges not followed
    sidecar: str | None = None

    @property
    def in_cycle(self) -> set[str]:
        return {t for c in self.cycles for t in c}

    def walk(self, term: str, direction: str) -> dict[str, int]:
        """``{term: shortest depth}`` reachable from ``term`` (the term itself excluded)."""
        step = self.children if direction == "descendants" else self.parents
        depth = {term: 0}
        frontier = [term]
        while frontier:
            nxt: list[str] = []
            for t in frontier:
                for u in step.get(t, ()):
                    if u not in depth:
                        depth[u] = depth[t] + 1
                        nxt.append(u)
            frontier = sorted(nxt)
        depth.pop(term, None)
        return depth

    def to_json(self) -> dict[str, Any]:
        return {"id_type": self.source.id_type, "table": self.source.table, "predicates": list(self.source.predicates),
                "fingerprint": self.fingerprint, "parents": self.parents, "cycles": self.cycles,
                "edges_skipped": self.edges_skipped}


def _edge_columns(view: Any, source: HierarchySource) -> list[tuple[str, str | None, str | None, Any, str]]:
    """``(container or column, item field or None, type field or None, column spec, relation)`` per declared
    hierarchy column."""
    out = []
    for path in source.columns:
        container, _, fieldname = path.partition("[].")
        if fieldname:
            spec = view.column(container)
            fields = dict(getattr(spec, "fields", {}) or {})
            fspec = fields.get(fieldname)
            type_field = next((n for n, c in fields.items() if n != fieldname and getattr(c, "role", None) == "category"),
                              None)
            out.append((container, fieldname, type_field, fspec, str(getattr(fspec, "relation", "parent"))))
        else:
            spec = view.column(path)
            out.append((path, None, None, spec, str(getattr(spec, "relation", "parent"))))
    return out


def _predicate_ok(spec: Any, edge_type: str | None, predicates: Sequence[str]) -> tuple[bool, str]:
    if not predicates:
        return True, edge_type or ""
    if edge_type is not None:
        return edge_type in predicates, edge_type
    declared = getattr(spec, "predicate", None)
    names = [declared] if isinstance(declared, str) else list(declared or [])
    if not names:
        return True, ""
    hit = [n for n in names if n in predicates]
    return bool(hit), (hit[0] if hit else names[0])


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _find_cycles(parents: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Every strongly connected component with more than one term, or a self loop (Tarjan, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    nodes = sorted(set(parents) | {p for ps in parents.values() for p in ps})
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(sorted(parents.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on.add(root)
        while work:
            node, it = work[-1]
            nxt = next(it, None)
            if nxt is not None:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on.add(nxt)
                    work.append((nxt, iter(sorted(parents.get(nxt, ())))))
                elif nxt in on:
                    low[node] = min(low[node], index[nxt])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    t = stack.pop()
                    on.discard(t)
                    comp.append(t)
                    if t == node:
                        break
                if len(comp) > 1 or node in parents.get(node, ()):
                    out.append(sorted(comp))
    return sorted(out)


def closure_for(ctx: ServiceContext, id_type: str) -> Closure:
    """The closure of ``id_type``'s hierarchy (cached per hierarchy fingerprint; sidecar on disk)."""
    source = hierarchy_of(ctx, id_type)
    if source is None:
        raise GatewayError(ErrorKind.unsupported_combination, f"{id_type} has no declared hierarchy to expand",
                           payload={"id_type": id_type, "reason": "no_hierarchy"})
    reader = ctx.reader(source.table)
    try:
        fp = reader.fingerprint()
    except Exception as exc:  # noqa: BLE001 - an unreadable hierarchy is not ready, never an empty expansion
        raise GatewayError(ErrorKind.not_ready, f"the hierarchy of {id_type} ({source.table}) cannot be read: {exc}",
                           payload=not_ready_payload([{"name": source.table, "check": "hierarchy",
                                                       "detail": str(exc)[:200]}])) from None
    tag = canonical([fp, sorted(source.predicates), list(source.columns)])
    cache = ctx.__dict__.setdefault("_closures", {})
    hit = cache.get(source.id_type)
    if hit is not None and hit[0] == tag:
        return hit[1]
    name = source.id_type.replace(":", "__")
    src = source.table.split(".")[0]
    path = ctx.cache_path(src, fp, "closure", f"{name}.json.gz")
    closure: Closure | None = None
    if path.exists():
        try:
            data = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
            if data.get("tag") == tag:
                closure = Closure(source, fp, {k: list(v) for k, v in data["parents"].items()},
                                  cycles=[list(c) for c in data.get("cycles", [])],
                                  edges_skipped=dict(data.get("edges_skipped") or {}), sidecar=str(path))
        except (OSError, ValueError, KeyError):
            closure = None
    if closure is None:
        closure = _build(ctx, source, fp)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(gzip.compress(json.dumps({**closure.to_json(), "tag": tag}, sort_keys=True).encode()))
        tmp.replace(path)
        closure.sidecar = str(path)
    children: dict[str, list[str]] = {}
    for child, ps in closure.parents.items():
        for p in ps:
            children.setdefault(p, []).append(child)
    closure.children = {k: sorted(v) for k, v in children.items()}
    cache[source.id_type] = (tag, closure)
    return closure


def _build(ctx: ServiceContext, source: HierarchySource, fp: str) -> Closure:
    view = long_view(ctx, source.table)
    key = view.key[0]
    cols = _edge_columns(view, source)
    need = list(dict.fromkeys([key, *(c for c, *_ in cols)]))
    rows, _total, _eu = view.rows(None, columns=need, order=[], limit=None)
    parents: dict[str, set[str]] = {}
    skipped: dict[str, int] = {}
    for r in rows:
        term = r.get(key)
        if term in (None, ""):
            continue
        term = str(term)
        parents.setdefault(term, set())
        for container, fieldname, type_field, spec, relation in cols:
            if fieldname is None:
                targets = [(t, None) for t in _as_list(r.get(container))]
            else:
                targets = [(it.get(fieldname), it.get(type_field) if type_field else None)
                           for it in _as_list(r.get(container)) if isinstance(it, Mapping)]
            for target, etype in targets:
                if target in (None, ""):
                    continue
                ok, pname = _predicate_ok(spec, None if etype is None else str(etype), source.predicates)
                if not ok:
                    skipped[pname] = skipped.get(pname, 0) + 1
                    continue
                if relation == "child":
                    parents.setdefault(str(target), set()).add(term)
                else:
                    parents[term].add(str(target))
    plain = {k: sorted(v) for k, v in sorted(parents.items())}
    return Closure(source, fp, plain, cycles=_find_cycles(plain), edges_skipped=dict(sorted(skipped.items())))


# ---------------------------------------------------------------------------- expansion


def expand(ctx: ServiceContext, id_type: str, term: Any, *, direction: str = "descendants",
           max_expand: int | None = None, include_self: bool = False) -> tuple[dict[str, int], dict[str, Any]]:
    """``({term: depth}, expansion record)`` of ``term`` under ``id_type``'s hierarchy (see the module docstring)."""
    if direction not in ("descendants", "ancestors"):
        raise _invalid("direction", direction, "direction is descendants or ancestors", ["descendants", "ancestors"])
    closure = closure_for(ctx, id_type)
    t = str(term)
    limit = int(max_expand if max_expand is not None else ctx.settings.resolution.max_expand)
    reached = closure.walk(t, direction)
    cyclic = closure.in_cycle & ({t} | set(reached))
    if cyclic:
        raise GatewayError(ErrorKind.not_ready, f"the hierarchy of {id_type} has a cycle through "
                           f"{', '.join(sorted(cyclic)[:5])}; it cannot be expanded faithfully",
                           subkind="hierarchy_cycle",
                           payload=not_ready_payload([{"name": closure.source.table, "check": "acyclic",
                                                       "detail": f"cycle: {' -> '.join(sorted(cyclic)[:10])}"}]))
    if len(reached) > limit:
        raise GatewayError(ErrorKind.too_large, f"{t} has {len(reached)} {direction} under {id_type}, more than "
                           f"max_expand={limit}; name a narrower term", subkind="expansion",
                           payload=too_large_payload("expansion", hint="choose a more specific term"))
    out = dict(sorted(reached.items(), key=lambda kv: (kv[1], kv[0])))
    if include_self or closure.source.reflexive:
        out = {t: 0, **out}
    record = {"id_type": id_type, "term": t, "direction": direction, "hierarchy": closure.source.table,
              "declared_by": closure.source.declared_by, "predicates": list(closure.source.predicates),
              "fingerprint": closure.fingerprint, "n": len(reached), "max_depth": max(reached.values(), default=0),
              "known": t in closure.parents or t in closure.children,
              "edges_not_followed": dict(closure.edges_skipped), "closure": closure.sidecar}
    return out, record


def _column_id_type(ctx: ServiceContext, table: Any, column: str) -> str | None:
    spec = table.columns.get(column)
    idt = getattr(spec, "id_type", None)
    if not idt:
        return None
    try:
        return ctx.catalog.qualify_id_type(str(idt), table.descriptor.source)
    except LookupError:
        return None


def descendant_predicate(ctx: ServiceContext, table_ref: str, column: str, value: Any, *,
                         max_expand: int | None = None) -> tuple[Predicate, dict[str, Any]]:
    """``include_descendants`` on ``table_ref.column = value`` (see the module docstring)."""
    table = ctx.table(table_ref)
    spec = table.columns.get(column)
    if spec is None:
        raise ServiceError(f"{table_ref} has no column {column!r}")
    over = getattr(spec, "propagated_over", None)
    if over:
        raise GatewayError(ErrorKind.unsupported_combination,
                           f"{table_ref}.{column} is already propagated over {over}: its rows include every "
                           "descendant, so include_descendants would count them twice; query it as is",
                           payload={"arguments": ["include_descendants"], "reason": "propagated_over",
                                    "propagated_over": over})
    for name, c in table.columns.items():
        if getattr(c, "role", None) == "hierarchy" and getattr(c, "relation", None) == "ancestor" and \
                getattr(c, "of", None) == column:
            record = {"column": column, "term": json_value(value), "via": f"{table_ref}.{name}",
                      "form": "eq_or_contains_ancestor", "reflexive": bool(getattr(c, "reflexive", False))}
            return Or((Eq(column, value), Contains(name, value))), record
    id_type = _column_id_type(ctx, table, column)
    if id_type is None:
        raise GatewayError(ErrorKind.unsupported_combination, f"{table_ref}.{column} has no hierarchy to expand",
                           payload={"arguments": ["include_descendants"], "reason": "no_hierarchy"})
    terms, record = expand(ctx, id_type, value, max_expand=max_expand)
    values = tuple(dict.fromkeys([value, *terms]))
    record = {**record, "column": column, "form": "in_expansion"}
    return (Eq(column, value) if len(values) == 1 else In(column, values)), record


def rewrite_include_descendants(ctx: ServiceContext, table_ref: str, pred: Predicate | None, *,
                                max_expand: int | None = None) -> tuple[Predicate | None, list[dict[str, Any]]]:
    """Every ``Eq(col, X)`` on a hierarchical identifier column of ``pred`` replaced by its descendant
    predicate (outside negations); ``(predicate, expansion records)``."""
    table = ctx.table(table_ref)
    records: list[dict[str, Any]] = []

    def hierarchical(col: str) -> bool:
        spec = table.columns.get(col)
        if getattr(spec, "propagated_over", None):
            return True                                # refused by descendant_predicate
        if any(getattr(c, "role", None) == "hierarchy" and getattr(c, "relation", None) == "ancestor" and
               getattr(c, "of", None) == col for c in table.columns.values()):
            return True
        idt = _column_id_type(ctx, table, col)
        return idt is not None and hierarchy_of(ctx, idt) is not None

    def rec(p: Predicate) -> Predicate:
        if isinstance(p, Eq) and isinstance(p.column, str) and hierarchical(p.column):
            out, record = descendant_predicate(ctx, table_ref, p.column, p.value, max_expand=max_expand)
            records.append(record)
            return out
        if isinstance(p, And):
            return And(tuple(rec(q) for q in p.preds))
        if isinstance(p, Or):
            return Or(tuple(rec(q) for q in p.preds))
        return p                                       # Not(...) and leaves stay as they are

    if pred is None:
        return None, records
    out = rec(pred)
    if not records:
        raise GatewayError(ErrorKind.unsupported_combination, f"include_descendants: no argument of this call binds a "
                           f"hierarchical identifier of {table_ref}",
                           payload={"arguments": ["include_descendants"], "reason": "no_hierarchy"})
    return out, records


def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"))


# ---------------------------------------------------------------------------- propagated membership


def member_column(view: Any) -> tuple[str, Any]:
    cols = [(n, c) for n, c in view.columns.items() if getattr(c, "role", None) == "member"]
    if cols:
        return cols[0]
    container = getattr(view.table, "container", None)
    if container is not None and getattr(container, "role", None) == "member":
        return (view.table.items_path or "").rstrip("[]").split(".")[-1], container
    raise _invalid("table", view.ref, f"{view.ref} has no member column")


def propagated_members(ctx: ServiceContext, table_ref: str, set_column: str, member_key: str, set_id: Any, *,
                       set_id_type: str | None, max_expand: int | None = None, columns: Sequence[str] = (),
                       ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int]:
    """Members of ``set_id`` and of every descendant set of it (each member once, first by its direct
    annotation, else by the shallowest descendant): ``(rows with 'via', expansion record, direct count)``."""
    view = long_view(ctx, table_ref)
    sets: dict[str, int] = {str(set_id): 0}
    record = None
    if set_id_type and hierarchy_of(ctx, set_id_type) is not None:
        desc, record = expand(ctx, set_id_type, set_id, max_expand=max_expand)
        sets.update(desc)
    rows, _total, _eu = view.rows(In(set_column, tuple(sets)), columns=list(dict.fromkeys([set_column, member_key,
                                                                                            *columns])),
                                  order=[], limit=None)
    best: dict[str, tuple[int, str, dict[str, Any]]] = {}
    for r in rows:
        m = r.get(member_key)
        s = r.get(set_column)
        if m in (None, "") or s is None:
            continue
        rank = (sets.get(str(s), 1 << 30), str(s))
        k = canonical([m])
        cur = best.get(k)
        if cur is None or rank < (cur[0], cur[1]):
            best[k] = (rank[0], rank[1], r)
    out = []
    for k in sorted(best):
        depth, via, r = best[k]
        out.append({**{c: r.get(c) for c in columns if c not in (set_column,)}, member_key: r.get(member_key),
                    "via": via, "propagated": depth > 0})
    direct = sum(1 for o in out if not o["propagated"])
    return out, record, direct


def mixed_fraction(ctx: ServiceContext, table_ref: str, set_column: str, member_key: str, set_id_type: str
                   ) -> dict[str, Any]:
    """``propagation: mixed`` measured: the share of stored (member, set) annotations whose member is also
    annotated to a descendant of that set (an ancestor listed next to the lowest-level annotation)."""
    closure = closure_for(ctx, set_id_type)
    view = long_view(ctx, table_ref)
    rows, _t, _e = view.rows(None, columns=[set_column, member_key], order=[], limit=None)
    by_member: dict[str, set[str]] = {}
    for r in rows:
        if r.get(set_column) is not None and r.get(member_key) is not None:
            by_member.setdefault(str(r[member_key]), set()).add(str(r[set_column]))
    total = sum(len(s) for s in by_member.values())
    ancestors_listed = 0
    for sets in by_member.values():
        for s in sets:
            if sets & set(closure.walk(s, "descendants")):
                ancestors_listed += 1
    return {"annotations": total, "ancestor_annotations": ancestors_listed,
            "fraction": (ancestors_listed / total) if total else None, "fingerprint": closure.fingerprint}


def members(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``members(table, set_id, propagate?)``: direct members, or with ``propagate: true`` the members of the
    set and its descendant sets (each once, ``via`` the set it is annotated to)."""
    if not payload.get("propagate"):
        return _DIRECT_MEMBERS(ctx, payload)
    table = table_access(ctx, payload.get("table"), agent=payload.get("agent"))
    view = long_view(ctx, str(table.ref))
    name, mcol = member_column(view)
    membership = getattr(mcol, "membership", None)
    set_end = getattr(membership, "set", None)
    set_col = getattr(set_end, "parent", None) or getattr(set_end, "path", None) or view.key[0]
    if set_col not in view.columns:
        set_col = next((c for c in view.columns if c.endswith(str(set_col))), set_col)
    notes: list[str] = []
    resolved: dict[str, str] = {}
    _pred, keys = compile_where(view, {set_col: payload.get("set_id")}, argument="set_id", notes=notes,
                                resolved=resolved)
    canon = next(iter(keys.get(set_col) or [payload.get("set_id")]))
    set_idt = view.id_type(set_col)
    if set_idt is None or hierarchy_of(ctx, set_idt) is None:
        raise GatewayError(ErrorKind.unsupported_combination, f"{view.ref}: the sets have no hierarchy to propagate "
                           "over; call without propagate", payload={"arguments": ["propagate"], "reason": "no_hierarchy"})
    member_key = _member_key(view, membership, name)
    rows, record, direct = propagated_members(ctx, view.ref, set_col, member_key, canon, set_id_type=set_idt,
                                              max_expand=payload.get("max_expand"))
    out = [{set_col: canon, "member": r[member_key], "via": r["via"], "propagated": r["propagated"]} for r in rows]
    notes.append(f"propagated over {set_idt}: {direct} direct, {len(out) - direct} via descendant sets")
    extra: dict[str, Any] = {"expansion": [summary(record)] if record else []}
    body: dict[str, Any] = {"rows": json_value(out), "expansion": [record] if record else []}
    if getattr(membership, "propagation", None) == "mixed":
        mixed = mixed_fraction(ctx, view.ref, set_col, member_key, set_idt)
        extra["mixed"] = body["mixed"] = mixed
        if mixed["fraction"] is not None:
            notes.append(f"membership is mixed: {mixed['fraction']:.1%} of annotations also list an ancestor")
    hdr = header(view, rows=out, total=len(out), resolved=resolved, notes=notes, key=[set_col, "member"],
                 extra=extra)
    return inject_header(body, hdr)


def summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """The provenance summary of an expansion record (the full record travels in the body)."""
    return {k: record.get(k) for k in ("id_type", "term", "direction", "n", "max_depth", "fingerprint")}


def _member_key(view: Any, membership: Any, name: str) -> str:
    """The row field holding one member: the parent key of an item table (``/id`` when an item field is also
    named ``id``), else the member column."""
    parent = getattr(getattr(membership, "member", None), "parent", None)
    if parent:
        return f"/{parent}" if view.table.is_item_table and parent in view.columns else str(parent)
    return name


def expand_verb(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``_expand``: descendants (or ancestors) of each value under an id_type's hierarchy."""
    id_type = payload.get("id_type")
    try:
        bound = ctx.catalog.qualify_id_type(str(id_type))
    except LookupError:
        names = sorted(f"{s}:{n}" for s, d in ctx.catalog.sources.items() for n in d.id_types)
        raise _invalid("id_type", id_type, f"unknown id_type {id_type!r}", names) from None
    values = payload.get("values")
    if values is None and payload.get("value") is not None:
        values = [payload["value"]]
    if not isinstance(values, list) or not values:
        raise _invalid("values", values, "values is a non-empty list of terms")
    direction = str(payload.get("direction") or "descendants")
    from .public import _resolve_value

    notes: list[str] = []
    resolved: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    records = []
    for i, v in enumerate(values):
        canon = _resolve_value(ctx, f"values[{i}]", v, bound, notes, resolved)
        terms, record = expand(ctx, bound, canon, direction=direction, max_expand=payload.get("max_expand"),
                               include_self=bool(payload.get("include_self")))
        records.append(record)
        rows.extend({"of": canon, "term": t, "depth": d} for t, d in terms.items())
    closure = closure_for(ctx, bound)
    from ...result import Header

    hdr = Header(status="ok" if rows else "empty", returned=len(rows), total=len(rows), served_by="derived",
                 tables=[closure.source.table], key=["of", "term"], resolved=resolved or None,
                 order="of, depth, term", notes=notes, extra={"expansion": [summary(r) for r in records]})
    return inject_header({"rows": json_value(rows), "expansion": json_value(records)}, hdr)


# ---------------------------------------------------------------------------- _serve routing


def _params(req: Mapping[str, Any]) -> Mapping[str, Any]:
    return req.get("params") or {}


@serve_extension("include_descendants", lambda req: _truthy(_params(req).get("include_descendants")))
def serve_include_descendants(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """A ``_serve`` request whose call set ``include_descendants``: the predicate is rewritten, the read is the
    ordinary one, and the expansion records travel in ``sections._expansion``."""
    table = str(req.get("table"))
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    new, records = rewrite_include_descendants(ctx, table, pred)
    params = {k: v for k, v in _params(req).items() if k != "include_descendants"}
    out = dict(_serve_module.VERBS[VERB_SERVE](ctx, {**req, "predicate": to_json(new) if new is not None else None,
                                                        "params": params}))
    sections = dict(out.get("sections") or {})
    sections["_expansion"] = records
    out["sections"] = sections
    return out


def _claims_members(req: Mapping[str, Any]) -> bool:
    if req.get("verb") == "expand":
        return True
    split = req.get("split") or {}
    return req.get("verb") == "members" and bool(split.get("propagate"))


@serve_extension("expand", _claims_members)
def serve_expand(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """``verb: expand`` (rows ``{of, term, depth}`` for the predicate's ``Eq``/``In`` values on the table's key)
    and ``verb: members`` with ``split: {propagate: true}`` (members of the bound set and its descendants)."""
    table = str(req.get("table"))
    view = long_view(ctx, table)
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    if req.get("verb") == "expand":
        col = view.key[0]
        terms = _values_of(pred, col)
        idt = view.id_type(col)
        if not terms or idt is None:
            raise ServiceError(f"expand needs {table}.{col} fixed to one or more terms")
        rows: list[dict[str, Any]] = []
        records = []
        for t in terms:
            got, record = expand(ctx, idt, t, direction=str((req.get("split") or {}).get("direction") or
                                                            "descendants"))
            records.append(record)
            rows.extend({"of": t, "term": k, "depth": d} for k, d in got.items())
        limit = req.get("limit")
        shown = rows[: int(limit)] if limit is not None else rows
        return ServeResponse(rows=shown, total=len(rows), truncated=len(shown) < len(rows), key_columns=["of", "term"],
                             sections={"_expansion": records}, served_by="derived").model_dump(mode="json")
    name, mcol = member_column(view)
    set_col = _set_column(view, mcol)
    values = _values_of(pred, set_col)
    if len(values) != 1:
        return base_serve(ctx, {**req, "split": None})
    member_key = _member_key(view, getattr(mcol, "membership", None), name)
    set_idt = _set_id_type(ctx, view, mcol, set_col)
    columns = [c for c in (req.get("columns") or []) if c != set_col]
    rows, record, direct = propagated_members(ctx, table, set_col, member_key, values[0], set_id_type=set_idt,
                                              columns=columns)
    out_rows = [{**{c: r.get(c) for c in columns}, member_key: r[member_key]} for r in rows]
    rename = dict(req.get("rename") or {})
    if rename:
        out_rows = [{rename.get(k, k): v for k, v in r.items()} for r in out_rows]
    note = {"set": json_value(values[0]), "direct": direct, "propagated": len(rows) - direct,
            "expansion": record}
    return ServeResponse(rows=out_rows, total=len(out_rows), truncated=False, key_columns=[member_key],
                         sections={"_expansion": [record] if record else [], "_propagation": note},
                         served_by="derived", row_keys=[[r.get(member_key)] for r in out_rows]).model_dump(mode="json")


def _set_column(view: Any, mcol: Any) -> str:
    set_end = getattr(getattr(mcol, "membership", None), "set", None)
    name = getattr(set_end, "parent", None) or getattr(set_end, "path", None) or view.key[0]
    if name in view.columns:
        return str(name)
    return next((k for k in view.key if str(k).endswith(f".{name}") or str(k) == name), str(name))


def _set_id_type(ctx: ServiceContext, view: Any, mcol: Any, set_col: str) -> str | None:
    idt = view.id_type(set_col) or view.id_type(set_col.split(".")[-1])
    if idt:
        return idt
    raw = getattr(getattr(getattr(mcol, "membership", None), "set", None), "id_type", None)
    if not raw:
        return None
    try:
        return ctx.catalog.qualify_id_type(str(raw), view.source)
    except LookupError:
        return None


def _values_of(pred: Predicate | None, column: str) -> list[Any]:
    """The values ``pred`` fixes ``column`` to (``Eq``/``In``, possibly inside an ``And``)."""
    short = column.split(".")[-1].replace("[]", "")

    def same(c: str) -> bool:
        return c == column or c.split(".")[-1].replace("[]", "") == short

    if isinstance(pred, Eq) and same(pred.column):
        return [pred.value]
    if isinstance(pred, In) and same(pred.column):
        return list(pred.values)
    if isinstance(pred, And):
        for p in pred.preds:
            got = _values_of(p, column)
            if got:
                return got
    return []


_current = _public.VERBS["members"]
_DIRECT_MEMBERS = getattr(_current, "_phase2", _current)            # direct membership (public.py)
if not getattr(_current, "_phase3", False):
    _wrapped = guarded("members", members)
    _wrapped._phase3 = True  # type: ignore[attr-defined]
    _wrapped._phase2 = _current  # type: ignore[attr-defined]
    _public.VERBS["members"] = _wrapped

VERBS = {VERB_EXPAND: guarded("expand", expand_verb)}

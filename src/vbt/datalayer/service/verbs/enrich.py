"""``_enrich``: set enrichment from a member column's ``enrichment`` contract (§6.8 S9, §9.4; phase 3, F16).

Upstream tested only the sets that overlap the query (BH's ``m`` counted pathways with a hit), used
"every annotated gene" as the background whatever the GO aspect, and dropped unresolvable genes
silently. Here everything comes from the descriptor (``membership.enrichment: {universe, family,
test, correction}``):

* **query**: the gene list resolved like any identifier argument (symbols, versioned or lower-case
  Ensembl IDs); a list resolving below ``min_resolved_fraction`` (the call's, else the membership's)
  is ``insufficient_resolution``; ``resolution_summary`` lists what did not resolve;
* **universe** per scope: the declared universe table and ``where`` (``{param: aspect}`` bound from the
  call), one universe per ``per_scope`` value when the call does not fix it (GO: each aspect has its
  own background of genes with at least one annotation in that aspect); a caller ``universe`` list
  overrides it (resolved, disclosed);
* **propagation**: annotations are propagated to every ancestor set through the set id_type's
  hierarchy (Reactome ``parents``; GO via the ``gene_ontology`` source: ``is_a`` and ``part_of``
  propagate, ``regulates`` does not) when the membership declares ``propagate_via`` and the hierarchy
  can be read; otherwise direct annotations only, disclosed;
* **family**: every set whose size, counted **within the universe**, lies in the size bounds, the sets
  with zero overlap included (``include_zero_overlap``); BH (``correction``) runs over that whole family;
* **test**: the ``test`` statistic plugin (``hypergeom_enrichment`` or ``fisher``) on
  ``(overlap, set size, query size, universe size)``, all counted within the universe.

Each tested set is a row ``{scope?, set_id, set_label, set_size, overlap, overlap_members, expected,
fold_enrichment, pvalue, fdr, top_level?}`` (``fold_enrichment`` = observed / expected overlap;
``top_level`` lists the hierarchy roots above the set, the exact alternative to the lossy
``topLevelTerm`` category), ordered by p-value, then set ID. The ``statistics`` block of the result
records the test, the correction, the family size, the size bounds, the universe (table, where, size,
override) and the propagation (hierarchy, predicates, closure fingerprint), so the numbers can be
recomputed.

Request (hidden verb): ``{table, column, genes, universe?, scope?: {param: value}, min_size?, max_size?,
alpha?, limit?, propagate?: true | false | "auto", min_resolved_fraction?}``. Derived bindings reach it
through ``_serve`` with ``split: {enrich: {column, args: {min_size: <arg>, max_size: <arg>, alpha: <arg>,
aspect: <arg>}, defaults: {...}}}``; the gene list is the predicate's ``In`` on the bound column
(already resolved by the gateway) and the rows are the significant sets (``fdr <= alpha``, every set
when no alpha is given), cut to ``limit``, with ``total`` counting them; the statistics block travels
in ``sections._statistics``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...errors import ErrorKind, GatewayError, insufficient_resolution_payload, json_value
from ...ipc import VERB_ENRICH, ServeResponse
from ...predicate import In, Param, Predicate, from_json
from ...result import inject_header
from .. import ServiceContext, ServiceError
from .hierarchy import _values_of, closure_for, hierarchy_of, serve_extension
from .public import _invalid, guarded, header, long_view, resolver, table_access

__all__ = ["enrich", "enrich_sets", "EnrichmentResult", "serve_enrich", "VERBS"]

ROW_FIELDS = ("scope", "set_id", "set_label", "set_size", "overlap", "overlap_members", "expected",
              "fold_enrichment", "pvalue", "fdr", "top_level")


class EnrichmentResult:
    """The rows of every tested set and the statistics block."""

    def __init__(self, rows: list[dict[str, Any]], statistics: dict[str, Any], notes: list[str]) -> None:
        self.rows = rows
        self.statistics = statistics
        self.notes = notes


# ---------------------------------------------------------------------------- the contract


def _contract(ctx: ServiceContext, table_ref: str, column: str | None) -> tuple[Any, str, Any, Any]:
    """``(table, member column name, member spec, enrichment spec)``."""
    table = ctx.table(table_ref)
    if table.is_item_table and table.container is not None:
        parent = ctx.table(str(table.physical))
        name = (table.items_path or "").split("[]")[0]
        table, column = parent, column or name
    cols = {n: c for n, c in table.columns.items() if getattr(c, "role", None) == "member"}
    if column is None:
        column = next((n for n, c in cols.items() if getattr(getattr(c, "membership", None), "enrichment", None)),
                      None)
    spec = cols.get(str(column)) if column else None
    enrichment = getattr(getattr(spec, "membership", None), "enrichment", None)
    if spec is None or enrichment is None:
        valid = [n for n, c in cols.items() if getattr(getattr(c, "membership", None), "enrichment", None)]
        raise _invalid("column", column, f"{table.ref} has no member column with an enrichment contract", valid)
    return table, str(column), spec, enrichment


def _item_table(ctx: ServiceContext, table: Any, column: str) -> str | None:
    for ref in ctx.item_tables_of(str(table.physical)):
        t = ctx.table(ref)
        if (t.items_path or "").rstrip("[]") == column:
            return ref
    return None


def _table_name(ctx: ServiceContext, source: str, name: str) -> str:
    first = name.split(".")[0]
    if "." in name and first in ctx.catalog.sources:
        return name
    return f"{source}.{name}"


def _field(view: Any, path: str) -> str:
    """The long-view field of a descriptor path: item fields by their name, parent columns as ``/name``."""
    t = view.table
    if not t.is_item_table:
        return path.replace("[]", "").split(".")[-1] if "[]." in path else path
    prefix = (t.items_path or "").rstrip("[]") + "[]."
    if path.startswith(prefix):
        return path[len(prefix):]
    return f"/{path.lstrip('/')}"


def _key_id_type(ctx: ServiceContext, view: Any, path: str) -> str | None:
    """The id_type of a universe key (a parent column of an item table is read from the parent table)."""
    f = _field(view, path)
    if f.startswith("/"):
        parent = ctx.table(str(view.table.physical))
        idt = getattr(parent.columns.get(f[1:]), "id_type", None)
        try:
            return ctx.catalog.qualify_id_type(str(idt), view.source) if idt else None
        except LookupError:
            return None
    return view.id_type(f)


def _bind(pred: Predicate, params: Mapping[str, Any]) -> Predicate | None:
    """``pred`` with every ``Param`` bound from ``params``; None when a parameter is missing."""
    missing: list[str] = []

    def walk(p: Any) -> Any:
        if isinstance(p, Param):
            if p.name not in params or params[p.name] is None:
                missing.append(p.name)
                return p
            return params[p.name]
        if isinstance(p, tuple):
            return tuple(walk(x) for x in p)
        if hasattr(p, "__dataclass_fields__"):
            kw = {f: walk(getattr(p, f)) for f in p.__dataclass_fields__}
            return type(p)(**kw)
        return p

    out = walk(pred)
    return None if missing else out


# ---------------------------------------------------------------------------- reading sets and universes


def _universe(ctx: ServiceContext, enrichment: Any, source: str, scope: Mapping[str, Any]
              ) -> tuple[dict[Any, set[str]], dict[str, Any]]:
    """``({scope value or None: universe keys}, record)``."""
    uni = enrichment.universe
    if uni is None:
        raise ServiceError("the enrichment contract declares no universe")
    ref = _table_name(ctx, source, uni.table)
    view = long_view(ctx, ref)
    key = _field(view, uni.keys[0])
    per_scope = [_field(view, s) for s in uni.per_scope]
    where = from_json(uni.where) if uni.where else None
    bound = _bind(where, scope) if where is not None else None
    pred = bound
    by_scope = where is not None and bound is None and per_scope
    if where is not None and bound is None and not per_scope:
        raise GatewayError(ErrorKind.invalid_argument, "the enrichment universe needs a scope value "
                           f"({', '.join(sorted(_params_of(where)))})", payload={"argument": "scope"})
    rows, _t, _e = view.rows(None if by_scope else pred, columns=[key, *per_scope], order=[], limit=None)
    out: dict[Any, set[str]] = {}
    for r in rows:
        k = r.get(key)
        if k in (None, ""):
            continue
        sv = tuple(r.get(s) for s in per_scope) if by_scope else None
        if by_scope and any(x is None for x in sv or ()):
            continue
        out.setdefault(sv[0] if sv and len(sv) == 1 else sv, set()).add(str(k))
    record = {"table": ref, "keys": list(uni.keys), "where": uni.where, "per_scope": list(uni.per_scope),
              "scope": dict(scope) if scope else None, "sizes": {str(k): len(v) for k, v in out.items()}}
    return out, record


def _params_of(p: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(p, Param):
        out.add(p.name)
    elif isinstance(p, tuple):
        for x in p:
            out |= _params_of(x)
    elif hasattr(p, "__dataclass_fields__"):
        for f in p.__dataclass_fields__:
            out |= _params_of(getattr(p, f))
    return out


def _sets(ctx: ServiceContext, table: Any, column: str, spec: Any, scope_fields: Sequence[str]
          ) -> tuple[list[tuple[str, str, Any]], dict[str, str], str | None]:
    """``([(set_id, member, scope value)], {set_id: label}, set id_type)`` of the membership."""
    membership = spec.membership
    set_end, member_end = membership.set, membership.member
    labels: dict[str, str] = {}
    out: list[tuple[str, str, Any]] = []
    source = table.descriptor.source
    item_ref = _item_table(ctx, table, column) if set_end.path else None
    if item_ref is not None:
        view = long_view(ctx, item_ref)
        set_field = _field(view, f"{column}[].{set_end.path}")
        member_field = _field(view, str(member_end.parent)) if member_end.parent else _field(
            view, f"{column}[].{member_end.path}")
        label_field = next((n for n, c in view.columns.items() if getattr(c, "role", None) == "label" and
                            getattr(c, "of", None) == set_end.path), None)
        scope_cols = [s for s in scope_fields if s]
        need = list(dict.fromkeys([set_field, member_field, *([label_field] if label_field else []), *scope_cols]))
        rows, _t, _e = view.rows(None, columns=need, order=[], limit=None)
        for r in rows:
            s, m = r.get(set_field), r.get(member_field)
            if s in (None, "") or m in (None, ""):
                continue
            sv = tuple(r.get(c) for c in scope_cols)
            out.append((str(s), str(m), sv[0] if len(sv) == 1 else (sv or None)))
            if label_field and r.get(label_field) and str(s) not in labels:
                labels[str(s)] = str(r[label_field])
        set_idt = set_end.id_type
    else:
        view = long_view(ctx, str(table.ref))
        set_col = str(set_end.parent or view.key[0])
        rows, _t, _e = view.rows(None, columns=[set_col, column], order=[], limit=None)
        for r in rows:
            s = r.get(set_col)
            for m in r.get(column) or []:
                if s not in (None, "") and m not in (None, ""):
                    out.append((str(s), str(m), None))
        set_idt = set_end.id_type or getattr(view.column(set_col), "id_type", None)
    try:
        qualified = ctx.catalog.qualify_id_type(str(set_idt), source) if set_idt else None
    except LookupError:
        qualified = None
    return out, labels, qualified


def _labels(ctx: ServiceContext, id_type: str, ids: Sequence[str]) -> dict[str, str]:
    """Labels of set IDs from the id_type's universe table (its first ``resolve_via`` column there)."""
    src, spec = ctx.catalog.id_type(id_type)
    if not isinstance(spec.universe, str) or "." not in spec.universe:
        return {}
    table, key = spec.universe.split(".", 1)
    label = next((c.split(".", 1)[1] for c in spec.resolve_via if c.split(".", 1)[0] == table), None)
    if label is None:
        return {}
    try:
        view = long_view(ctx, f"{src}.{table}")
        rows, _t, _e = view.rows(In(key, tuple(ids)), columns=[key, label], order=[], limit=None)
    except Exception:  # noqa: BLE001 - labels are informative; an unreadable label table leaves them null
        return {}
    return {str(r[key]): str(r[label]) for r in rows if r.get(key) is not None and r.get(label) is not None}


def _map_members(ctx: ServiceContext, spec: Any, source: str, members: set[str]) -> tuple[dict[str, str], list[str]]:
    """Members in the universe's identity space: ``({member: canonical}, unmapped)`` through the member end's
    ``maps_to`` (MSigDB symbols -> Ensembl genes); identity when nothing is declared."""
    end = spec.membership.member
    if not end.maps_to or not end.id_type:
        return {m: m for m in members}, []
    r = resolver(ctx)
    try:
        bound = ctx.catalog.qualify_id_type(str(end.maps_to), source)
        kind = ctx.catalog.qualify_id_type(str(end.id_type), source)
    except LookupError:
        return {m: m for m in members}, []
    out: dict[str, str] = {}
    unmapped = []
    for m in sorted(members):
        res = r.resolve(m, [kind, bound], bound_id_type=bound)
        if res.status in ("resolved", "resolved_unverified") and res.canonical is not None:
            out[m] = str(res.canonical)
        else:
            unmapped.append(m)
    return out, unmapped


# ---------------------------------------------------------------------------- the analysis


def resolve_query(ctx: ServiceContext, values: Sequence[Any], bound: str, *, argument: str = "genes",
                  min_fraction: float | None = None) -> tuple[list[str], dict[str, Any]]:
    """``(canonical keys, resolution summary)``; ``insufficient_resolution`` below ``min_fraction``."""
    from .public import accepted_kinds

    r = resolver(ctx)
    kinds = accepted_kinds(ctx, bound)
    canon: list[str] = []
    unresolved: list[Any] = []
    ambiguous: list[Any] = []
    duplicates: list[Any] = []
    distinct = list(dict.fromkeys(str(v) for v in values))
    for v in distinct:
        res = r.resolve(v, kinds, bound_id_type=bound)
        if res.status in ("resolved", "resolved_unverified") and res.canonical is not None:
            if str(res.canonical) in canon:
                duplicates.append(v)
            else:
                canon.append(str(res.canonical))
        elif res.status == "ambiguous":
            ambiguous.append(v)
        else:
            unresolved.append(v)
    summary = {"requested": len(distinct), "resolved": len(canon) + len(duplicates), "unresolved": unresolved,
               "ambiguous": ambiguous, "duplicates": duplicates}
    if min_fraction is not None and distinct and (len(canon) + len(duplicates)) / len(distinct) < float(min_fraction):
        raise GatewayError(ErrorKind.insufficient_resolution,
                           f"{len(canon) + len(duplicates)} of {len(distinct)} {argument} resolved "
                           f"(minimum {float(min_fraction):.0%})",
                           payload=insufficient_resolution_payload(argument, len(distinct), len(canon) + len(duplicates),
                                                                   unresolved, ambiguous, [], float(min_fraction)))
    return canon, summary


def enrich_sets(ctx: ServiceContext, table_ref: str, column: str | None, query: Sequence[str], *,
                scope: Mapping[str, Any] | None = None, universe_override: Sequence[str] | None = None,
                min_size: int | None = None, max_size: int | None = None, propagate: Any = "auto"
                ) -> EnrichmentResult:
    """Every set of the family tested against ``query`` (canonical universe keys); see the module docstring."""
    table, column, spec, enrichment = _contract(ctx, table_ref, column)
    source = table.descriptor.source
    notes: list[str] = []
    scope = {k: v for k, v in (scope or {}).items() if v is not None}
    plugin = ctx.statistic(enrichment.test)
    if plugin is None or not callable(getattr(plugin, "test", None)):
        raise GatewayError(ErrorKind.not_ready, f"the set test {enrichment.test!r} is not registered",
                           payload={"tables": [str(table.ref)], "status": "plugin_unavailable"})
    correction = ctx.statistic(enrichment.correction) if enrichment.correction else None
    uni_view = long_view(ctx, _table_name(ctx, source, enrichment.universe.table)) if enrichment.universe else None
    scope_fields = []
    if uni_view is not None and enrichment.universe.per_scope:
        item_ref = _item_table(ctx, table, column)
        if item_ref is not None:
            scope_fields = [_field(long_view(ctx, item_ref), s) for s in enrichment.universe.per_scope]
    universes, uni_record = _universe(ctx, enrichment, source, scope)
    if universe_override is not None:
        universes = {k: set(map(str, universe_override)) for k in (universes or {None: set()})}
        uni_record = {**uni_record, "override": True, "sizes": {str(k): len(v) for k, v in universes.items()}}
        notes.append(f"universe overridden by the caller ({len(set(universe_override))} keys)")
    pairs, labels, set_idt = _sets(ctx, table, column, spec, scope_fields)
    mapping, unmapped = _map_members(ctx, spec, source, {m for _s, m, _v in pairs})
    if unmapped:
        notes.append(f"{len(unmapped)} set member(s) do not map to the universe's identifiers and are not counted")
    fixed_scope = None
    if scope_fields and scope:
        fixed_scope = next(iter(scope.values()))
    # propagation
    closure = None
    prop_record: dict[str, Any] = {"propagated": False}
    via = getattr(spec.membership, "propagate_via", None)
    if propagate and via and set_idt:
        try:
            closure = closure_for(ctx, set_idt) if hierarchy_of(ctx, set_idt) is not None else None
        except GatewayError as exc:
            if propagate is True:
                raise
            notes.append(f"annotations not propagated: the hierarchy of {set_idt} is unavailable ({exc.message})")
        if closure is not None:
            prop_record = {"propagated": True, "id_type": set_idt, "hierarchy": closure.source.table,
                           "declared_by": closure.source.declared_by, "predicates": list(closure.source.predicates),
                           "fingerprint": closure.fingerprint, "edges_not_followed": dict(closure.edges_skipped),
                           "closure": closure.sidecar}
            if closure.cycles:
                raise GatewayError(ErrorKind.not_ready, f"the hierarchy of {set_idt} has cycles; annotations cannot "
                                   "be propagated faithfully", subkind="hierarchy_cycle",
                                   payload={"tables": [closure.source.table]})
    elif propagate is True and not (via and set_idt):
        raise GatewayError(ErrorKind.unsupported_combination, f"{table.ref}.{column} declares no hierarchy to "
                           "propagate over", payload={"arguments": ["propagate"], "reason": "no_hierarchy"})
    members_of: dict[Any, dict[str, set[str]]] = {}
    anc_cache: dict[str, list[str]] = {}
    for s, m, sv in pairs:
        if fixed_scope is not None and sv != fixed_scope:
            continue
        key = sv if scope_fields else None
        canon = mapping.get(m)
        if canon is None:
            continue
        targets = [s]
        if closure is not None:
            if s not in anc_cache:
                anc_cache[s] = sorted(closure.walk(s, "ancestors"))
            targets += anc_cache[s]
        for t in targets:
            members_of.setdefault(key, {}).setdefault(t, set()).add(canon)
    unlabelled = sorted({s for sets_ in members_of.values() for s in sets_} - set(labels))
    if unlabelled and set_idt:
        labels.update(_labels(ctx, set_idt, unlabelled))
    roots: dict[str, list[str]] = {}
    if closure is not None:
        for sets_ in members_of.values():
            for s in sets_:
                anc = anc_cache.get(s) or sorted(closure.walk(s, "ancestors"))
                roots[s] = sorted(a for a in anc if not closure.parents.get(a)) or ([] if closure.parents.get(s) else [s])
    fam = enrichment.family
    lo = int(min_size) if min_size is not None else 1
    hi = int(max_size) if max_size is not None else None
    query_set = {str(q) for q in query}
    rows: list[dict[str, Any]] = []
    family_sizes: dict[str, int] = {}
    stats_scopes: dict[str, Any] = {}
    scopes = sorted(members_of, key=lambda k: str(k)) if members_of else []
    for key in scopes:
        uni = universes.get(key) if key in universes else universes.get(None, set()) if None in universes else set()
        if fixed_scope is not None:
            uni = universes.get(fixed_scope, universes.get(None, set()))
        N = len(uni)
        q = query_set & uni
        n = len(q)
        family: list[tuple[str, set[str]]] = []
        for s, mem in members_of[key].items():
            within = mem & uni
            K = len(within)
            if K < lo or (hi is not None and K > hi):
                continue
            if not fam.include_zero_overlap and not (within & q):
                continue
            family.append((s, within))
        family.sort(key=lambda x: x[0])
        pvals: list[float] = []
        part: list[dict[str, Any]] = []
        for s, within in family:
            K = len(within)
            hits = within & q
            k = len(hits)
            p = float(plugin.test(k, K, n, N, None)) if N else 1.0
            pvals.append(p)
            part.append({"scope": key, "set_id": s, "set_label": labels.get(s), "set_size": K, "overlap": k,
                         "overlap_members": sorted(hits), "expected": (K * n / N) if N else None,
                         "fold_enrichment": ((k / n) / (K / N)) if n and K and N else None, "pvalue": p,
                         "top_level": roots.get(s) if closure is not None else None})
        qs = correction.adjust(pvals, len(pvals)) if correction is not None and pvals else [None] * len(pvals)
        for r, qv in zip(part, qs):
            r["fdr"] = qv
        rows.extend(part)
        family_sizes[str(key)] = len(family)
        stats_scopes[str(key)] = {"universe_n": N, "query_in_universe": n, "family_m": len(family),
                                  "zero_overlap_sets": sum(1 for r in part if r["overlap"] == 0)}
    rows.sort(key=lambda r: (r["pvalue"], str(r["scope"]), r["set_id"]))
    if not scope_fields:
        for r in rows:
            r.pop("scope", None)
    if closure is None:
        for r in rows:
            r.pop("top_level", None)
    statistics = {
        "test": enrichment.test, "correction": enrichment.correction,
        "family": {"m": family_sizes, "include_zero_overlap": fam.include_zero_overlap,
                   "size_bounds": {"min": lo, "max": hi, "counted_within": fam.size_bounds.get("counted_within",
                                                                                              "universe")},
                   "per_scope": bool(scope_fields)},
        "universe": uni_record, "scopes": stats_scopes, "propagation": prop_record,
        "members_unmapped": len(unmapped), "query": {"given": len(query_set)},
    }
    return EnrichmentResult(rows, statistics, notes)


def _scope_values(spec: Any, scope: Mapping[str, Any], item_view: Any | None) -> dict[str, Any]:
    """Scope values with the scope column's aliases applied (``biological_process`` -> ``P``)."""
    out = {}
    for k, v in scope.items():
        if v is None:
            continue
        col = None
        if item_view is not None:
            col = item_view.column(k)
        aliases = dict(getattr(col, "aliases", {}) or {})
        out[k] = aliases.get(v, aliases.get(str(v), v))
    return out


def _scope_spec(ctx: ServiceContext, table: Any, column: str, enrichment: Any) -> Any:
    item_ref = _item_table(ctx, table, column)
    return long_view(ctx, item_ref) if item_ref else None


def enrich(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The hidden verb (see the module docstring)."""
    t = table_access(ctx, payload.get("table"), agent=payload.get("agent"), native=False)
    table, column, spec, enrichment = _contract(ctx, str(t.ref), payload.get("column"))
    genes = payload.get("genes")
    if not isinstance(genes, list) or not genes:
        raise _invalid("genes", genes, "genes is a non-empty list of identifiers")
    uni = enrichment.universe
    uview = long_view(ctx, _table_name(ctx, table.descriptor.source, uni.table))
    bound = _key_id_type(ctx, uview, uni.keys[0])
    if bound is None:
        raise ServiceError(f"the universe key {uni.keys[0]} of {uview.ref} has no id_type")
    mrf = payload.get("min_resolved_fraction", spec.membership.min_resolved_fraction)
    canon, summary = resolve_query(ctx, genes, bound, min_fraction=mrf)
    override = None
    if payload.get("universe") is not None:
        if not isinstance(payload["universe"], list) or not payload["universe"]:
            raise _invalid("universe", payload["universe"], "universe is a non-empty list of identifiers")
        override, _usum = resolve_query(ctx, payload["universe"], bound, argument="universe")
        missing = sorted(set(canon) - set(override))
        if missing:
            summary["outside_universe"] = missing
    scope = _scope_values(spec, dict(payload.get("scope") or {}), _scope_spec(ctx, table, column, enrichment))
    res = enrich_sets(ctx, str(table.ref), column, canon, scope=scope, universe_override=override,
                      min_size=payload.get("min_size"), max_size=payload.get("max_size"),
                      propagate=payload.get("propagate", "auto"))
    rows = res.rows
    alpha = payload.get("alpha")
    if alpha is not None:
        rows = [r for r in rows if r.get("fdr") is not None and r["fdr"] <= float(alpha)]
    limit = payload.get("limit")
    shown = rows[: int(limit)] if limit is not None else rows
    notes = list(res.notes)
    notes.append("every set of the family is tested, zero-overlap sets included; BH runs over the whole family")
    hdr = header(long_view(ctx, str(table.ref)), rows=shown, total=len(rows), truncated=len(shown) < len(rows),
                 order="pvalue asc, then set_id", key=["scope", "set_id"] if any("scope" in r for r in rows) else
                 ["set_id"], notes=notes, covered=True,
                 extra={"statistics": res.statistics, "resolution_summary": summary})
    out = inject_header({"rows": json_value(shown), "statistics": json_value(res.statistics)}, hdr)
    out["_vbt"]["coverage"] = "covered"
    return out


# ---------------------------------------------------------------------------- _serve (derived bindings)


@serve_extension("enrich", lambda req: bool((req.get("split") or {}).get("enrich")))
def serve_enrich(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """The ``_serve`` form (``split: {enrich: {...}}``) of ``get_pathway_enrichment`` and ``get_go_enrichment``."""
    opts = dict((req.get("split") or {}).get("enrich") or {})
    params = dict(req.get("params") or {})
    args = dict(opts.get("args") or {})
    defaults = dict(opts.get("defaults") or {})

    def arg(name: str) -> Any:
        a = args.get(name)
        v = params.get(a) if a else None
        return v if v is not None else defaults.get(name)

    table, column, spec, enrichment = _contract(ctx, str(req.get("table")), opts.get("column"))
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    view = long_view(ctx, str(req.get("table")))
    genes: list[Any] = []
    for c in (view.key[0], *(k for k in view.key)):
        genes = _values_of(pred, c)
        if genes:
            break
    scope = {}
    if args.get("aspect"):
        scope_name = next(iter(_params_of(from_json(enrichment.universe.where)))) if enrichment.universe and \
            enrichment.universe.where else "aspect"
        scope[scope_name] = arg("aspect")
    scope = _scope_values(spec, scope, _scope_spec(ctx, table, column, enrichment))
    res = enrich_sets(ctx, str(table.ref), column, [str(g) for g in genes], scope=scope,
                      min_size=arg("min_size"), max_size=arg("max_size"), propagate=defaults.get("propagate", "auto"))
    rows = res.rows
    alpha = arg("alpha")
    if alpha is not None:
        rows = [r for r in rows if r.get("fdr") is not None and r["fdr"] <= float(alpha)]
    total = len(rows)
    limit = req.get("limit")
    shown = rows[: int(limit)] if limit is not None else rows
    drop = set(opts.get("drop") or [])
    rename = dict(req.get("rename") or {})
    out_rows = [{rename.get(k, k): v for k, v in r.items() if k not in drop} for r in shown]
    stats = {**res.statistics, "returned": len(out_rows), "significant": total, "alpha": alpha,
             "notes": res.notes}
    sections = {"_statistics": json_value(stats)}
    for name in (req.get("sections") or {}):
        sections.setdefault(name, json_value(stats))
    return ServeResponse(rows=json_value(out_rows), total=total, truncated=len(shown) < total,
                         key_columns=[rename.get("set_id", "set_id")], sections=sections,
                         served_by="derived").model_dump(mode="json")


# ---------------------------------------------------------------------------- grouped essentiality (DepMap)


def _effect_stats(effects: Sequence[Any], spec: Any, threshold: Any = None) -> dict[str, Any]:
    """``n``, ``n_unknown``, ``mean_gene_effect`` and ``essential_fraction`` of screen effects: unknown effects
    are excluded and counted; the fraction uses the measure's declared cutoff (its inclusivity), or
    ``<= threshold`` when the call gives one; ``None`` (never 0) without a known effect."""
    from ...plugins.statistics.gene_effect import meets_cutoff

    known = [float(e) for e in effects if isinstance(e, (int, float)) and not isinstance(e, bool) and e == e]
    out: dict[str, Any] = {"n": len(known), "n_unknown": len(effects) - len(known),
                           "mean_gene_effect": (sum(known) / len(known)) if known else None}
    if threshold is not None:
        flags = [e <= float(threshold) for e in known]
    elif getattr(spec, "cutoff", None) is not None:
        flags = [bool(meets_cutoff(e, spec)) for e in known]
    else:
        flags = []
    out["essential_count"] = sum(flags) if flags or known else None
    out["essential_fraction"] = (sum(flags) / len(flags)) if flags else None
    return out


def _screen_paths(opts: Mapping[str, Any]) -> dict[str, str]:
    cols = dict(opts.get("columns") or {})
    missing = [k for k in ("gene", "effect", "disease") if k not in cols]
    if missing:
        raise ServiceError(f"split.essentiality.columns needs {', '.join(missing)}")
    return cols


@serve_extension("essentiality", lambda req: bool((req.get("split") or {}).get("essentiality")))
def serve_essentiality(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """Grouped screen statistics for the DepMap essentiality tools (``split: {essentiality: {mode, columns,
    args}}``) over the screens item table: ``by_tissue`` (one gene's tissues, with its screens),
    ``by_disease`` (one gene per disease), ``by_gene`` (genes essential in one disease) and ``selective``
    (genes whose mean effect in a target disease is lower than in a comparison disease). Diseases are exact
    vocabulary values (the gateway checked them); groups below ``min_cell_lines`` known effects are dropped and
    counted; unknown effects never count as non-essential."""
    opts = dict((req.get("split") or {}).get("essentiality") or {})
    params = dict(req.get("params") or {})
    args = dict(opts.get("args") or {})
    cols = _screen_paths(opts)

    def p(name: str, default: Any) -> Any:
        v = params.get(args.get(name, name))
        return default if v is None else v

    view = long_view(ctx, str(req.get("table")))
    spec = view.column(cols["effect"].split(".")[-1]) or view.column(cols["effect"])
    pred = from_json(req["predicate"]) if req.get("predicate") else None
    mode = opts.get("mode")
    want = list(dict.fromkeys(v for v in cols.values()))
    if mode == "selective":
        target, comparison = params.get(args.get("target", "target_disease")), params.get(
            args.get("comparison", "comparison_disease"))
        if target == comparison:
            raise GatewayError(ErrorKind.invalid_argument, "the target and comparison diseases are the same group")
        pred = In(cols["disease"], (target, comparison))
    rows, _total, _eu = view.rows(pred, columns=want, order=[], limit=None, budget=req.get("budget_bytes"))
    min_n = int(p("min_cell_lines", 1))
    below = 0
    out: list[dict[str, Any]] = []
    if mode == "by_tissue":
        genes = {r.get(cols["gene"]) for r in rows}
        groups: dict[Any, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r.get(cols["tissue_id"]), []).append(r)
        tissues = []
        for tid in sorted(groups, key=lambda t: str(t)):
            g = groups[tid]
            st = _effect_stats([r.get(cols["effect"]) for r in g], spec)
            screens = sorted(({k: r.get(v) for k, v in dict(opts.get("screen_fields") or {}).items()} for r in g),
                             key=lambda s: str(s.get("depmapId")))
            tissues.append({"tissue": {"id": tid, "name": g[0].get(cols.get("tissue_name", ""))},
                            "num_cell_lines": len(g), "num_with_effect": st["n"],
                            "mean_gene_effect": st["mean_gene_effect"], "essential_fraction": st["essential_fraction"],
                            "cell_lines": screens})
        flags = [r.get(cols["essential_flag"]) for r in rows if cols.get("essential_flag")]
        known_flags = [f for f in flags if f is not None]
        record = {"gene_id": next(iter(genes)) if len(genes) == 1 else None, "found": True,
                  "is_essential": known_flags[0] if known_flags else None, "num_tissues": len(tissues),
                  "num_cell_lines": sum(t["num_cell_lines"] for t in tissues), "essentiality_by_tissue": tissues}
        return ServeResponse(rows=[json_value(record)], total=1, key_columns=[], served_by="derived",
                             sections={"_essentiality": {"cutoff": _cutoff_text(spec)}}).model_dump(mode="json")
    if mode == "by_disease":
        groups = {}
        for r in rows:
            groups.setdefault(r.get(cols["disease"]), []).append(r.get(cols["effect"]))
        for d, effects in groups.items():
            st = _effect_stats(effects, spec)
            if st["n"] < min_n:
                below += 1
                continue
            out.append({"disease": d, "num_cell_lines": st["n"], "num_unknown": st["n_unknown"],
                        "mean_gene_effect": st["mean_gene_effect"], "essential_fraction": st["essential_fraction"]})
        out.sort(key=lambda r: (r["mean_gene_effect"] is None, r["mean_gene_effect"] or 0.0, str(r["disease"])))
        for i, r in enumerate(out, start=1):
            r["rank"] = i
    elif mode == "by_gene":
        threshold = p("min_effect_threshold", None)
        groups = {}
        for r in rows:
            groups.setdefault(r.get(cols["gene"]), []).append(r)
        for gene, g in groups.items():
            st = _effect_stats([r.get(cols["effect"]) for r in g], spec, threshold)
            if st["n"] < min_n:
                below += 1
                continue
            if not st["essential_count"]:
                continue
            top = sorted((r for r in g if isinstance(r.get(cols["effect"]), (int, float))),
                         key=lambda r: (r[cols["effect"]], str(r.get(cols.get("cell_line", "")))))[:5]
            out.append({"gene_id": gene, "mean_gene_effect": st["mean_gene_effect"], "num_cell_lines": st["n"],
                        "num_unknown": st["n_unknown"], "essential_fraction": st["essential_fraction"],
                        "top_cell_lines": [{"cellLineName": r.get(cols.get("cell_line", "")),
                                            "disease": r.get(cols["disease"]), "geneEffect": r.get(cols["effect"])}
                                           for r in top]})
        out.sort(key=lambda r: (r["mean_gene_effect"], str(r["gene_id"])))
    elif mode == "selective":
        min_diff = float(p("min_effect_difference", 0.3))
        groups = {}
        for r in rows:
            side = "target" if r.get(cols["disease"]) == target else "comparison"
            groups.setdefault(r.get(cols["gene"]), {"target": [], "comparison": []})[side].append(r.get(cols["effect"]))
        for gene, sides in groups.items():
            t, c = _effect_stats(sides["target"], spec), _effect_stats(sides["comparison"], spec)
            if t["n"] < min_n or c["n"] < min_n:
                below += 1
                continue
            diff = t["mean_gene_effect"] - c["mean_gene_effect"]
            if diff >= -min_diff:
                continue
            out.append({"gene_id": gene, "target_effect": t["mean_gene_effect"], "comparison_effect": c["mean_gene_effect"],
                        "effect_difference": diff, "target_cell_lines": t["n"], "comparison_cell_lines": c["n"]})
        out.sort(key=lambda r: (r["effect_difference"], str(r["gene_id"])))
    else:
        raise ServiceError(f"unknown essentiality mode {mode!r}")
    total = len(out)
    limit = req.get("limit")
    shown = out[: int(limit)] if limit is not None else out
    meta = {"groups_below_min_cell_lines": below, "min_cell_lines": min_n, "cutoff": _cutoff_text(spec)}
    return ServeResponse(rows=json_value(shown), total=total, truncated=len(shown) < total, served_by="derived",
                         key_columns=["disease"] if mode == "by_disease" else ["gene_id"],
                         sections={"_essentiality": meta}).model_dump(mode="json")


@serve_extension("tissue_specificity", lambda req: bool((req.get("split") or {}).get("tissue_specificity")))
def serve_tissue_specificity(ctx: ServiceContext, req: Mapping[str, Any]) -> dict[str, Any]:
    """Genes expressed more in one tissue than in the median of the others (``find_tissue_specific_genes``).

    Unit guard: each gene is compared within the unit of its target-tissue value (``rna.unit``); a blank unit
    is not a unit (the gene is excluded and counted), values in other units are never pooled. When every
    other tissue is 0 the fold change is undefined: the gene is reported as ``expressed_only_in_target`` with
    ``fold_change: null`` (upstream wrote a 999.9 sentinel), ranked first."""
    from statistics import median

    opts = dict((req.get("split") or {}).get("tissue_specificity") or {})
    params = dict(req.get("params") or {})
    args = dict(opts.get("args") or {})
    cols = dict(opts.get("columns") or {})
    tissue = params.get(args.get("tissue", "tissue"))
    if tissue in (None, ""):
        raise GatewayError(ErrorKind.invalid_argument, "tissue is required")
    threshold = float(params.get(args.get("threshold", "fold_change_threshold")) or 2.0)
    view = long_view(ctx, str(req.get("table")))
    want = [cols["gene"], cols["tissue_id"], cols["tissue_label"], cols["value"], cols["unit"]]
    if cols.get("zscore"):
        want.append(cols["zscore"])
    rows, _t, _e = view.rows(None, columns=want, order=[], limit=None, budget=req.get("budget_bytes"))
    key = str(tissue).strip().casefold()

    def is_target(r: Mapping[str, Any]) -> bool:
        return str(r.get(cols["tissue_id"]) or "").casefold() == key or \
            str(r.get(cols["tissue_label"]) or "").strip().casefold() == key

    by_gene: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_gene.setdefault(str(r.get(cols["gene"])), []).append(r)
    if not any(is_target(r) for r in rows):
        labels = sorted({str(r.get(cols["tissue_label"])) for r in rows if r.get(cols["tissue_label"])})
        raise GatewayError(ErrorKind.invalid_argument, f"tissue {tissue!r} is not a tissue of this table (exact label "
                           "or EFO code)", payload={"argument": "tissue", "valid_values": labels[:50]})
    out: list[dict[str, Any]] = []
    blank_unit = undefined = 0
    for gene, entries in by_gene.items():
        targets = [e for e in entries if is_target(e) and isinstance(e.get(cols["value"]), (int, float))]
        if not targets:
            continue
        t = targets[0]
        unit = t.get(cols["unit"])
        if unit is None or not str(unit).strip():
            blank_unit += 1
            continue
        others = [float(e[cols["value"]]) for e in entries if not is_target(e) and e.get(cols["unit"]) == unit and
                  isinstance(e.get(cols["value"]), (int, float)) and e[cols["value"]] == e[cols["value"]]]
        if not others:
            continue
        med = float(median(others))
        value = float(t[cols["value"]])
        if med == 0:
            if value <= 0:
                continue
            undefined += 1
            fold, only = None, True
        else:
            fold, only = value / med, False
            if fold < threshold:
                continue
        out.append({"gene_id": gene, "tissue_expression": value, "median_other_tissues": med,
                    "fold_change": fold, "expressed_only_in_target": only, "n_other_tissues": len(others),
                    "zscore_in_tissue": t.get(cols["zscore"]) if cols.get("zscore") else None, "unit": unit})
    out.sort(key=lambda r: (not r["expressed_only_in_target"], -(r["fold_change"] or 0.0),
                            -r["tissue_expression"], r["gene_id"]))
    total = len(out)
    limit = req.get("limit")
    shown = out[: int(limit)] if limit is not None else out
    meta = {"blank_unit_excluded": blank_unit, "expressed_only_in_target": undefined, "threshold": threshold}
    return ServeResponse(rows=json_value(shown), total=total, truncated=len(shown) < total, served_by="derived",
                         key_columns=["gene_id"], sections={"_specificity": meta}).model_dump(mode="json")


def _cutoff_text(spec: Any) -> str | None:
    cutoff = getattr(spec, "cutoff", None)
    if cutoff is None:
        return None
    return f"{cutoff.op} {cutoff.value} ({'inclusive' if cutoff.op in ('le', 'ge') else 'exclusive'})"


VERBS = {VERB_ENRICH: guarded("enrich", enrich)}

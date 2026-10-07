"""``_build_index``: resolver sidecars per id_type and row-group value indexes per access path (§11.5, §11.6).

**Resolver sidecar** (``{source, id_type}``). The id_type's identity universe is projected (a column,
a ``table.column[]`` list, several tables, or a ``UniverseSpec`` with ``keys``/``mode``/``where``)
and the index rows of ``resolve/index.py`` are written:

* ``exact``: one per universe key, ``label`` its primary label, ``family`` its parent (``canonicalize``);
* ``label_exact:<col>`` / ``synonym:<kind>`` for every ``resolve_via`` column (containers expanded
  to their label and synonym leaves); labels stored in another table carry ``stored_table``;
* ``retired:<col>`` from ``retired.listed_in`` (the row's key is the replacement), ``flag`` and
  ``label_prefix`` (the term is retired, ``replaced_by`` names the replacement, ``consider`` the
  candidates; retired terms are not universe members);
* ``xref:<NS>`` for ``xref_via`` CURIEs whose namespace is mapped (or named by ``id_type_from``);
* ``crosswalk:<name>`` for crosswalks owned by the id_type (canonical = owner key, label = the other key);
* ``stored_form`` for every table spelling of a key that differs from its canonical form
  (``stored_forms`` entries, and identifier columns referencing the universe), via ``normalize_stored``;
* ``attr:<col>`` with the ``disambiguate_with`` values (JSON scalars).

``label_key`` is the configured plugin's ``label_key`` (the plugin is configured with the universe
sample, so ``prefixes: from_universe`` works). The sidecar is written under the **universe table's
fingerprint** (the first table of a list universe): ``<cache>/<source>/<fp>/index/<id_type>.tsv.gz``.

**Access path** (``{table, access_path}``): the row-group value index of ``service/sidecar.py``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from ...descriptor.columns import UniverseSpec, is_container
from ...ipc import VERB_BUILD_INDEX, BuildIndexRequest, BuildIndexResponse
from ...plugins.base import Normalized
from ...predicate import from_json
from ...resolve.index import Entry
from .. import ServiceContext, ServiceError
from .. import items as _items
from ..checks import _schema_path
from ..reader import BudgetExceeded
from ..sidecar import build_access_index

__all__ = ["build_index", "build_resolver_index", "build_access_path", "resolver_rows", "text_leaf", "VERBS"]

UNIVERSE_SAMPLE = 5000


def _table_ref(source: str, table: str) -> str:
    return table if table.count(".") == 1 else f"{source}.{table}"


def _split_ref(source: str, ref: str) -> tuple[str, str]:
    """``table.column`` / ``source.table.column`` -> (``source.table``, column path)."""
    parts = ref.split(".")
    if len(parts) >= 3 and "[" not in parts[1]:
        return f"{parts[0]}.{parts[1]}", ".".join(parts[2:])
    return f"{source}.{parts[0]}", ".".join(parts[1:])


def _universes(source: str, spec: Any) -> list[dict[str, Any]]:
    u = spec.universe
    refs = u if isinstance(u, list) else [u]
    out = []
    for r in refs:
        if r is None:
            continue
        if isinstance(r, UniverseSpec):
            out.append({"table": _table_ref(source, r.table), "keys": list(r.keys), "mode": r.mode, "where": r.where})
        else:
            table, column = _split_ref(source, str(r))
            out.append({"table": table, "keys": [column], "mode": "tuple", "where": None})
    return out


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v
    return json.dumps(v, default=str, ensure_ascii=False, sort_keys=True)


def _values(row: Mapping[str, Any], path: str) -> list[Any]:
    """Non-empty values at ``path`` (a list value is its elements)."""
    out: list[Any] = []
    for v in _items.path_values(row, path):
        for x in (v if isinstance(v, list) else [v]):
            if x is not None and x != "":
                out.append(x)
    return out


def _label_columns(reader: Any, key_column: str) -> list[str]:
    """Label columns of a key column of a table (role label, ``of`` the key or unspecified)."""
    out = []
    for name, col in reader.spec.columns.items():
        if getattr(col, "role", None) == "label" and getattr(col, "of", None) in (None, key_column):
            out.append(name + (getattr(col, "path", None) or ""))
    return out


def _owner_column(reader: Any, label_col: str, id_type: str) -> str:
    """The key column a label column of ``reader``'s table labels."""
    col = reader.column_spec(label_col.split("[")[0])
    of = getattr(col, "of", None)
    if of:
        return str(of)
    for name, c in reader.spec.columns.items():
        if getattr(c, "role", None) == "identifier" and getattr(c, "id_type", None) == id_type:
            return name
    return reader.key[0]


def _sibling_key(unis: Sequence[Mapping[str, Any]], table: str, column: str) -> tuple[str, str, str] | None:
    """``(container, label field, key field)`` when ``column`` is a field of the items of a list whose items
    also hold the id_type's universe key (both under the same ``a[].b[]`` container), else None."""
    if "[]" not in column:
        return None
    container, _, label_field = column.rpartition("[].")
    if not container or "[" in label_field:
        return None
    for u in unis:
        if u["table"] != table:
            continue
        for k in u["keys"]:
            kc, _, key_field = str(k).rpartition("[].")
            if kc == container and key_field and "[" not in key_field:
                return container + "[]", label_field, key_field
    return None


def _leaves_of(reader: Any, column: str) -> list[tuple[str, str]]:
    """``(path, kind)`` label/synonym leaves of a column (a container expands to its label/synonym fields)."""
    col = reader.column_spec(column)
    if col is None:
        return [(column, "label")]
    if is_container(col):
        out = []
        for name, f in col.fields.items():
            if getattr(f, "role", None) in ("label", "synonym"):
                out.extend(text_leaf(reader, f"{column}.{name}", f))
        return out
    return text_leaf(reader, column, col)


def text_leaf(reader: Any, dotted: str, col: Any) -> list[tuple[str, str]]:
    """``[(path, kind)]`` of one label or synonym column at a dotted declared path (lists inserted from the
    schema, the column's ``path`` facet applied); kind is ``label`` or the synonym kind."""
    base = _schema_path(reader, dotted)
    sub = getattr(col, "path", None) or ""
    path = base + sub if sub.startswith("[") else (f"{base}.{sub}" if sub else base)
    if path.endswith("[][]"):
        path = path[:-2]
    if getattr(col, "role", None) == "synonym":
        return [(path, str(getattr(col, "synonym_kind", None) or "alias"))]
    return [(path, "label")]


def resolver_rows(ctx: ServiceContext, source: str, id_type: str) -> tuple[list[Entry], str]:
    """``(index rows, fingerprint)`` of one id_type's resolver sidecar."""
    src, spec = ctx.catalog.id_type(f"{source}:{id_type}")
    qid = f"{src}:{id_type}"
    if spec.index == "remote":
        raise ServiceError(f"{qid} is resolved remotely (index: remote): use _resolve_remote")
    unis = _universes(src, spec)
    if not unis:
        raise ServiceError(f"{qid} declares no universe (label kinds live in their key type's index)")
    first = ctx.reader(unis[0]["table"])
    fingerprint = first.fingerprint()
    # -- the universe: keys, primary labels, families, attributes, retired terms ----------------
    retired = spec.retired
    keys: dict[str, dict[str, Any]] = {}
    rows: list[Entry] = []
    for u in unis:
        reader = ctx.reader(u["table"])
        key_cols = list(u["keys"])
        own = key_cols[-1]
        labels = _label_columns(reader, own.split("[")[0])
        extra: list[str] = [*labels, *spec.disambiguate_with]
        parent_col = None
        if spec.canonicalize is not None:
            ptable, pcol = _split_ref(src, spec.canonicalize.parent)
            if ptable == u["table"]:
                parent_col = pcol
                extra.append(pcol)
        if retired is not None:
            extra += [c for c in (retired.flag, retired.replaced_by, retired.consider) if c]
        pred = from_json(u["where"]) if u["where"] else None
        for m in reader.scan(pred, columns=[*key_cols, *extra], attribute_unknown=False):
            parts = [_values(m.row, k) for k in key_cols]
            if u["mode"] == "union":
                candidates = [v for vs in parts for v in vs]
            else:
                if not all(parts):
                    continue
                candidates = list(parts[-1])
            label = next((v for c in labels for v in _values(m.row, c)), None)
            for k in candidates:
                info = keys.setdefault(str(k), {"label": label, "family": "", "attrs": [], "retired": None})
                if parent_col:
                    p = _items.path_value(m.row, parent_col)
                    info["family"] = str(p) if p not in (None, "") and str(p) != str(k) else ""
                if u["mode"] == "tuple" and len(key_cols) > 1:
                    for kc, vs in zip(key_cols[:-1], parts[:-1]):
                        info["attrs"].append((kc, vs[0] if vs else None))
                for col in spec.disambiguate_with:
                    v = _items.path_value(m.row, col)
                    if v is not None:
                        info["attrs"].append((col, v))
                if retired is not None and _is_retired(m.row, label, retired):
                    rep = _items.path_value(m.row, retired.replaced_by) if retired.replaced_by else None
                    cons = _values(m.row, retired.consider) if retired.consider else []
                    info["retired"] = (rep, cons, retired.flag or (labels[0] if labels else own))
    sample = list(keys)[:UNIVERSE_SAMPLE]
    plugin = ctx.identifier(qid, sample)
    lk = plugin.label_key
    for k, info in keys.items():
        if info["retired"] is not None:
            rep, cons, col = info["retired"]
            rows.append(Entry(lk(k), str(rep) if rep else "", f"retired:{_arg(col)}", k))
            for c in cons:
                rows.append(Entry(lk(k), str(c), f"retired:{_arg(retired.consider)}", k))  # type: ignore[union-attr]
            continue
        rows.append(Entry(lk(k), k, "exact", _text(info["label"]) if info["label"] is not None else "",
                          family=info["family"]))
        for col, v in info["attrs"]:
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if x is not None:
                    rows.append(Entry("", k, f"attr:{_arg(col)}", _text(x)))
    # -- labels and synonyms (resolve_via) ----------------------------------------------------------
    universe_tables = {u["table"] for u in unis}
    for ref in spec.resolve_via:
        table, column = _split_ref(src, ref)
        reader = ctx.reader(table)
        leaves = _leaves_of(reader, column)
        sibling = _sibling_key(unis, table, column)
        if sibling is not None:
            # a label on a nested item (screens[].cellLineName) names the identifier on the same item
            # (screens[].depmapId), not the row's key
            container, label_field, key_field = sibling
            seen: set[tuple[str, str]] = set()
            for m in reader.scan(None, columns=[f"{container}.{label_field}", f"{container}.{key_field}"],
                                 attribute_unknown=False):
                for item in _items.path_values(m.row, container):
                    if not isinstance(item, Mapping) or item.get(key_field) in (None, ""):
                        continue
                    for v in _values(item, label_field):
                        s = _text(v)
                        if (s, str(item[key_field])) in seen:
                            continue                   # the same line screened for many genes
                        seen.add((s, str(item[key_field])))
                        rows.append(Entry(lk(s), str(item[key_field]), f"label_exact:{_arg(column)}", s,
                                          stored_table=table if table not in universe_tables else ""))
            continue
        owner = _owner_column(reader, column, id_type)
        for m in reader.scan(None, columns=[owner, *(p for p, _ in leaves)], attribute_unknown=False):
            k = _items.path_value(m.row, owner)
            if k is None:
                continue
            for path, kind in leaves:
                for v in _values(m.row, path):
                    s = _text(v)
                    if kind == "label":
                        rows.append(Entry(lk(s), str(k), f"label_exact:{_arg(column)}", s,
                                          stored_table=table if table not in universe_tables else ""))
                    else:
                        rows.append(Entry(lk(s), str(k), f"synonym:{kind}", s))
    # -- retired IDs listed on current terms ------------------------------------------------------------
    for ref in (retired.listed_in if retired is not None else []):
        table, column = _split_ref(src, ref)
        reader = ctx.reader(table)
        owner = reader.key[0]
        for m in reader.scan(None, columns=[owner, column], attribute_unknown=False):
            k = _items.path_value(m.row, owner)
            for v in _values(m.row, column if column.endswith("]") else column):
                for x in (v if isinstance(v, list) else [v]):
                    rows.append(Entry(lk(str(x)), str(k), f"retired:{_arg(column)}", str(x)))
    # -- cross-references -----------------------------------------------------------------------------
    for x in spec.xref_via:
        table, column = _split_ref(src, x.column)
        reader = ctx.reader(table)
        owner = reader.key[0]
        for m in reader.scan(None, columns=[owner, column], attribute_unknown=False):
            k = _items.path_value(m.row, owner)
            if k is None:
                continue
            for curie, ns in _xrefs(m.row, column, x):
                rows.append(Entry(lk(curie), str(k), f"xref:{ns}", curie))
    # -- crosswalks owned by this id_type ---------------------------------------------------------------
    for cw in spec.crosswalks:
        reader = ctx.reader(_table_ref(src, cw.table))
        for m in reader.scan(None, columns=[cw.from_, cw.to], attribute_unknown=False):
            a, b = _items.path_value(m.row, cw.from_), _items.path_value(m.row, cw.to)
            if a is None or b is None:
                continue
            rows.append(Entry(lk(str(a)), str(b), f"crosswalk:{cw.name}", str(a)))
    # -- stored forms ---------------------------------------------------------------------------------------
    rows.extend(_stored_form_rows(ctx, src, id_type, spec, plugin, keys))
    return _dedupe(rows), fingerprint


def _arg(column: Any) -> str:
    """A rule argument from a table-relative column path (list brackets dropped: ``obsoleteTerms``,
    ``genomicLocation.chromosome``)."""
    return str(column).replace("[]", "")


def _is_retired(row: Mapping[str, Any], label: Any, retired: Any) -> bool:
    if retired.flag and _items.path_value(row, retired.flag) is True:
        return True
    if retired.label_prefix and isinstance(label, str) and label.startswith(retired.label_prefix):
        return True
    return False


def _xrefs(row: Mapping[str, Any], column: str, spec: Any) -> Iterable[tuple[str, str]]:
    values = _items.path_values(row, column if column.endswith("]") else column + "[]")
    if not values:
        values = _values(row, column)
    for v in values:
        if isinstance(v, Mapping) and spec.id_type_from is not None:
            field = spec.id_type_from.field.lstrip("^.")
            ns = v.get(field)
            if ns is None or (spec.id_type_from.map and str(ns) not in spec.id_type_from.map):
                continue
            for k, x in v.items():
                if k == field:
                    continue
                for curie in (x if isinstance(x, list) else [x]):
                    if curie is not None:
                        yield str(curie), str(ns)
        elif isinstance(v, str) and ":" in v:
            ns = v.split(":", 1)[0]
            if spec.namespaces and ns not in spec.namespaces:
                continue
            yield v, ns


def _stored_form_rows(ctx: ServiceContext, src: str, id_type: str, spec: Any, plugin: Any,
                      keys: Mapping[str, Any]) -> list[Entry]:
    lk = plugin.label_key
    targets: dict[str, str] = {}
    for ref in spec.stored_forms:
        table, column = _split_ref(src, ref)
        targets[f"{table}\t{column}"] = "declared"
    desc = ctx.catalog.source(src)
    uni = {f"{u['table']}.{u['keys'][-1]}" for u in _universes(src, spec)}
    for tname, t in desc.tables.items():
        if t.items_of is not None:
            continue
        for cname, col in t.columns.items():
            ref = getattr(col, "ref", None)
            if getattr(col, "role", None) == "identifier" and getattr(col, "id_type", None) == id_type and \
                    isinstance(ref, str) and _table_ref(src, ref.rsplit(".", 1)[0]) + "." + ref.rsplit(".", 1)[1] \
                    in uni:
                targets.setdefault(f"{src}.{tname}\t{cname}", "ref")
    out: list[Entry] = []
    for target, why in targets.items():
        table, column = target.split("\t")
        try:
            snap = ctx.reader(table).snapshot(column)
        except (ServiceError, BudgetExceeded):
            continue
        for v in snap.values:
            if v is None:
                continue
            s = str(v)
            n = plugin.normalize_stored(s)
            if not isinstance(n, Normalized):
                continue
            canonical = n.value
            if canonical not in keys:
                continue
            if why == "ref" and s == canonical:
                continue
            out.append(Entry(lk(s), canonical, "stored_form", stored_table=table, stored_value=s))
    return out


def _dedupe(rows: Iterable[Entry]) -> list[Entry]:
    seen: dict[tuple[str, ...], Entry] = {}
    for e in rows:
        seen.setdefault(e.as_row(), e)
    return list(seen.values())


def build_resolver_index(ctx: ServiceContext, source: str, id_type: str, *, force: bool = False
                         ) -> BuildIndexResponse:
    bare = id_type.partition(":")[2] or id_type
    src, _spec = ctx.catalog.id_type(f"{source}:{bare}")
    rows, fp = resolver_rows(ctx, src, bare)
    path = ctx.index_store.write_sidecar(src, fp, bare, rows)
    return BuildIndexResponse(path=str(path), rows=len(rows), fingerprint=fp)


def build_access_path(ctx: ServiceContext, table: str, columns: list[str], *, force: bool = False
                      ) -> BuildIndexResponse:
    reader = ctx.reader(table)
    declared = [ap for ap in reader.spec.access_paths if ap.via == "sidecar_index"]
    match = next((ap for ap in declared if list(ap.columns) == list(columns) or ap.columns[:1] == columns[:1]), None)
    if match is None:
        raise ServiceError(f"{table} declares no sidecar_index access path on {columns} "
                           f"(declared: {[ap.columns for ap in declared]})")
    phys = ctx.reader(str(reader.table.physical))
    path, rows = build_access_index(phys, match.columns[0], force=force)
    return BuildIndexResponse(path=str(path), rows=rows, fingerprint=phys.fingerprint())


def build_index(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = BuildIndexRequest.model_validate(dict(payload))
    if req.source is not None and req.id_type is not None:
        resp = build_resolver_index(ctx, req.source, req.id_type, force=req.force)
    else:
        resp = build_access_path(ctx, str(req.table), list(req.access_path or []), force=req.force)
    return resp.model_dump(mode="json")


VERBS = {VERB_BUILD_INDEX: build_index}

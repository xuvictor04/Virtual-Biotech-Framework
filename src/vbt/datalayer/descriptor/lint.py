"""Descriptor and overlay lint (§6.9). No pyarrow.

:func:`lint_descriptor` checks one source descriptor (other loaded sources resolve
``source.table.col`` references and qualified id_types); :func:`lint_overlay` checks one
overlay against a catalog (or a ``{source: descriptor}`` mapping). Both return
:class:`Finding` lists; ``level == "error"`` findings fail ``vbt ds lint`` and CI.

Rules (``Finding.rule``):

``strict_roles``      every column and nested field roled (strict); error with ``strict=True``, else a warning
``key``               key columns exist, keyable, not all nullable, no nullable ``self``; item-table keys composable
``scope_key``         every scope column is in the key / an enclosing item key, or declares ``determined_by``,
                      or the table declares ``aggregated_over`` for it
``scope_binding``     a forbid-pooled scope key column of a bound table is fixed by an equality binding or left
                      to the auto-derived gateway-only argument
``reference``         ``ref``, ``of``, ``unit_from``, ``comparable_within``, ``event_of``, ``length_of``,
                      ``norm_column``, ``item_key``, ``family``, ``determined_by`` ... resolve (§6.4); refs to
                      sources that are not loaded are warnings
``reference_soft``    other column mentions (constraints, sentinels, access paths, edge sides ...): errors under
                      strict, warnings otherwise (non-strict descriptors need not role every physical column)
``id_type``           id_type names resolve to exactly one qualified id_type
``plugin``            plugin names registered (only when a registry is given); statistics may carry ``fallback``
``verified``          ``verified: false`` facts have a confirmation path or ``verified_by: manifest|upstream_doc``
``coverage``          ``censored`` coverage needs ``censor``
``partition``         ``expect: declared`` partitions have a vocabulary
``digits_kind``       a digits-only kind is not accepted next to other kinds without ``input_requires_prefix``
``resolver_columns``  ``canonicalize``, ``retired``, ``xref_via``, ``crosswalks``, ``stored_forms``,
                      ``resolve_via`` and universes name existing tables and columns
``binding``           overlay binds/fields/reads/args resolve; serve modes are complete
``accepts``           accepted kinds resolve and reach the bound column's id_type (label_of/maps_to/crosswalk/union)
``flag_codes``        ``when_true``/``when_false`` name codes positively (no ``ne``)
``echo``              echo paths read the record
``defect``            defect ``where`` is ``file:line``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from ..roles import KEYABLE_ROLES, LIST, Role, parse_path, PathError
from .columns import (
    CompositeRef,
    IdTypeFrom,
    MemberCol,
    NestedCol,
    ParseSpec,
    is_container,
)
from .models import GrainSpec, SourceDescriptor, TableSpec, UniverseSpec, id_type_identity, plugin_name
from .overlay import ArgBinding, DerivedSpec, EchoSpec, Overlay, ToolBinding
from .scoping import AmbiguousReference, Resolved, ScopeError, TableScope, UnknownReference

__all__ = [
    "Finding", "lint_descriptor", "lint_overlay", "errors", "warnings", "DEFECT_WHERE", "RESERVED_DIMENSIONS",
    "qualify", "split_qualified",
]

Level = Literal["error", "warning"]

DEFECT_WHERE = re.compile(r"^[\w/.]+:\d+(?:[-,]\d+)*$")
#: Pseudo-dimensions a facet may name besides columns (``comparable_within: [release]``).
RESERVED_DIMENSIONS = frozenset({"release"})
#: Rank columns that are not table columns (§6.1 RankSpec).
SPECIAL_RANK_COLUMNS = frozenset({"match_class", "similarity"})
_EQUALITY_OPS = ("eq", "in")


@dataclass(frozen=True)
class Finding:
    level: Level
    where: str
    message: str
    rule: str = ""
    target: str | None = None                          # the unresolved name, when there is one
    table: str | None = None                           # "source.table" the finding is about
    scope: str | None = None                           # container path a scoped name was looked up from

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "error"]


def warnings(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "warning"]


def split_qualified(name: str) -> tuple[str | None, str]:
    """``"src:kind"`` -> ``("src", "kind")``; bare -> ``(None, name)``."""
    if ":" in name:
        src, _, bare = name.partition(":")
        return src, bare
    return None, name


def qualify(name: str, source: str) -> str:
    return name if ":" in name else f"{source}:{name}"


def _sources_of(catalog: Any) -> dict[str, SourceDescriptor]:
    if isinstance(catalog, Mapping):
        return dict(catalog)
    srcs = getattr(catalog, "sources", None)
    if srcs is None:
        srcs = getattr(catalog, "descriptors", {})
    return dict(srcs() if callable(srcs) else srcs)


def _overlays_of(catalog: Any) -> dict[str, Overlay]:
    ovs = getattr(catalog, "overlays", None)
    if ovs is None:
        return {}
    return dict(ovs() if callable(ovs) else ovs)


def _role(col: Any) -> Role | None:
    r = getattr(col, "role", None)
    return Role(r) if r else None


@dataclass(frozen=True)
class _Site:
    table: str
    container: tuple[str, ...]                         # container names from the table root
    name: str
    col: Any
    axis: str | None = None                            # matrix axis ("row"/"col"), "value", "section"

    @property
    def path(self) -> str:
        return ".".join(self.container + (self.name,))

    @property
    def per_element(self) -> bool:
        p = getattr(self.col, "path", None)
        return bool((isinstance(p, str) and "[]" in p) or getattr(self.col, "list_delimiter", None))


def _walk_fields(table: str, fields: Mapping[str, Any], container: tuple[str, ...],
                 axis: str | None = None) -> Iterator[_Site]:
    for name, col in fields.items():
        yield _Site(table, container, name, col, axis)
        if is_container(col):
            yield from _walk_fields(table, col.fields, container + (name,), axis)
        parse = getattr(col, "parse", None)
        if isinstance(parse, ParseSpec):
            yield from _walk_fields(table, parse.fields, container + (name,), axis)


# ---------------------------------------------------------------------------
# Descriptor lint
# ---------------------------------------------------------------------------

class _DescriptorLinter:
    def __init__(self, desc: SourceDescriptor, registry: Any, strict: bool | None,
                 loaded: Mapping[str, SourceDescriptor]) -> None:
        self.desc = desc
        self.registry = registry
        self.strict = strict
        self.loaded = {k: v for k, v in loaded.items() if k != desc.source}
        self.out: list[Finding] = []
        self._scopes: dict[str, TableScope | None] = {}

    # -- helpers -------------------------------------------------------------

    def add(self, level: Level, where: str, message: str, rule: str, target: str | None = None,
            table: str | None = None, scope: Sequence[str] = ()) -> None:
        container = ".".join(n for n in scope if n != LIST) or None
        self.out.append(Finding(level, f"{self.desc.source}.{where}" if where else self.desc.source, message, rule,
                                target, f"{self.desc.source}.{table}" if table else None, container))

    def strict_level(self, table: str) -> Level | None:
        if self.strict is True:
            return "error"
        if self.strict is None and self.desc.table_strict(table):
            return "warning"
        return None

    def soft_level(self, table: str | None) -> Level:
        if self.strict is True:
            return "error"
        return "warning"

    def scope(self, table: str) -> TableScope | None:
        if table not in self._scopes:
            try:
                self._scopes[table] = TableScope(self.desc, table, self.loaded)
            except ScopeError as exc:
                items = self.desc.tables[table].items_of if table in self.desc.tables else None
                missing = items.table if items is not None and items.table not in self.desc.tables else None
                self.add("error", f"tables.{table}.items_of" if missing else f"tables.{table}", str(exc),
                         "key" if missing else "reference", missing, table)
                self._scopes[table] = None
        return self._scopes[table]

    def resolve(self, table: str, name: str, container: Sequence[str] = (), *, mode: str = "column",
                where: str, rule: str = "reference", level: Level = "error",
                innermost: bool = False) -> Resolved | None:
        """Resolve a reference and record a finding when it does not resolve."""
        if not isinstance(name, str) or "{" in name:
            return None                                # templated (column_patterns) or not a name
        if mode == "column" and name in RESERVED_DIMENSIONS:
            return None
        sc = self.scope(table)
        if sc is None:
            return None
        # A table.col reference is not looked up in the table's scope: the finding names no table.
        in_scope = (table, tuple(container)) if mode != "table_column" else (None, ())
        try:
            r = sc.resolve(name, tuple(container), mode=mode)  # type: ignore[arg-type]
        except AmbiguousReference as exc:
            self.add("error", where, str(exc), rule, name, *in_scope)
            return None
        except UnknownReference as exc:
            self.add(level, where, str(exc), rule, name, *in_scope)
            return None
        if not r.loaded:
            self.add("warning", where, f"{name!r} names source {r.source!r}, which is not loaded (not checked)",
                     rule, name, table)
            return r
        if innermost and r.kind not in ("element",) and r.level != len(sc.levels(container)) - 1:
            self.add(level, where, f"{name!r} must name a field of the container's items", rule, name, table,
                     container)
            return None
        return r

    def table_ref(self, ref: str, where: str, rule: str = "resolver_columns",
                  level: Level = "error") -> tuple[SourceDescriptor, str] | None:
        """``table`` or ``source.table`` -> (descriptor, table)."""
        if "." in ref:
            src, _, tab = ref.partition(".")
            d = self.desc if src == self.desc.source else self.loaded.get(src)
            if d is None:
                self.add("warning", where, f"source {src!r} of {ref!r} is not loaded (not checked)", rule, ref)
                return None
        else:
            d, tab = self.desc, ref
        if tab not in d.tables:
            self.add(level, where, f"unknown table {ref!r}", rule, ref)
            return None
        return d, tab

    def column_in(self, d: SourceDescriptor, table: str, column: str, where: str, rule: str,
                  level: Level = "error") -> Resolved | None:
        try:
            sc = TableScope(d, table, {**self.loaded, self.desc.source: self.desc})
            r = sc.resolve(column, (), mode="column")
        except ScopeError as exc:
            self.add(level, where, f"{table}.{column}: {exc}", rule, f"{table}.{column}" if d is self.desc
                     else f"{d.source}.{table}.{column}")
            return None
        return r

    def id_type(self, name: str | None, where: str, *, table: str | None = None) -> None:
        """Every id_type mention resolves to exactly one qualified id_type."""
        if not name:
            return
        src, bare = split_qualified(name)
        if src is None or src == self.desc.source:
            if bare not in self.desc.id_types:
                self.add("error", where, f"unknown id_type {name!r} (not declared in {self.desc.source})",
                         "id_type", name, table)
            return
        other = self.loaded.get(src)
        if other is None:
            self.add("warning", where, f"id_type {name!r}: source {src!r} is not loaded (not checked)", "id_type",
                     name, table)
        elif bare not in other.id_types:
            self.add("error", where, f"unknown id_type {name!r}", "id_type", name, table)

    # -- run ---------------------------------------------------------------

    def run(self) -> list[Finding]:
        for name, spec in self.desc.tables.items():
            if self.scope(name) is None:
                continue
            self.lint_table(name, spec)
        for name, it in self.desc.id_types.items():
            self.lint_id_type(name, it)
        self.lint_source()
        return self.out

    def sites(self, name: str, spec: TableSpec) -> list[_Site]:
        sites = list(_walk_fields(name, spec.columns, ()))
        for pname, part in spec.partitions.items():
            sites.append(_Site(name, (), pname, part.column))
        for pattern, col in spec.column_patterns.items():
            sites.append(_Site(name, (), pattern, col))
        if spec.matrix is not None:
            for axis, ax in spec.matrix.axes.items():
                sites.extend(_walk_fields(name, ax.columns, (), axis))
                if ax.parse is not None:
                    sites.extend(_walk_fields(name, ax.parse.fields, (), axis))
            sites.extend(_walk_fields(name, spec.matrix.values, (), "value"))
            sites.extend(_walk_fields(name, spec.matrix.sections, (), "section"))
        return sites

    # -- tables ------------------------------------------------------------

    def lint_table(self, name: str, spec: TableSpec) -> None:
        where = f"tables.{name}"
        sites = self.sites(name, spec)
        self.lint_strict_roles(name, spec, sites)
        self.lint_key(name, spec)
        self.lint_items_of(name, spec)
        for site in sites:
            self.lint_column(site)
        self.lint_scope_columns(name, spec, sites)
        lvl = self.soft_level(name)
        for i, r in enumerate(spec.rank):
            if r.column not in SPECIAL_RANK_COLUMNS:
                self.resolve(name, r.column, where=f"{where}.rank[{i}]", rule="reference")
            for w in r.within:
                self.resolve(name, w, where=f"{where}.rank[{i}].within", rule="reference")
        for gname, grain in spec.grains.items():
            gcols = grain if isinstance(grain, list) else list(grain.columns) + list(grain.unordered) + list(grain.by)
            for c in gcols:
                self.resolve(name, c, where=f"{where}.grains.{gname}", rule="reference")
            if isinstance(grain, GrainSpec) and grain.canonicalize:
                self.lint_canonical_grain(grain.canonicalize, f"{where}.grains.{gname}", name)
        for i, c in enumerate(spec.constraints):
            self.resolve(name, c.column, where=f"{where}.constraints[{i}]", rule="reference_soft", level=lvl)
        if spec.coverage is not None:
            self.lint_coverage(name, spec.coverage, f"{where}.coverage")
        if spec.sentinels is not None:
            for kind in ("present", "absent"):
                for i, s in enumerate(getattr(spec.sentinels, kind)):
                    sw = f"{where}.sentinels.{kind}[{i}]"
                    for k in s.key:
                        self.resolve(name, k, where=sw, rule="reference_soft", level=lvl)
                    for k, v in s.expect.items():
                        cols: list[str] = []
                        if k in ("nonempty", "is_null", "is_empty"):
                            cols = list(v)
                        elif k in ("contains", "items"):
                            cols = list(v)
                        elif k != "min_rows":
                            cols = [k]
                        for c in cols:
                            self.resolve(name, c, where=sw, rule="reference_soft", level=lvl)
        for pname, part in spec.partitions.items():
            pw = f"{where}.partitions.{pname}"
            vocab = getattr(part.column, "vocab", None)
            if part.expect == "declared" and not isinstance(vocab, list):
                self.add("error", pw, "expect: declared needs the partition column's vocabulary as a list",
                         "partition", table=name)
        for i, ap in enumerate(spec.access_paths):
            for c in ap.columns:
                self.resolve(name, c, where=f"{where}.access_paths[{i}]", rule="reference_soft", level=lvl)
            if ap.via == "partition":
                for c in ap.columns:
                    if c not in spec.partitions:
                        self.add("error", f"{where}.access_paths[{i}]", f"{c!r} is not a partition column",
                                 "partition", c, name)
        if spec.conditions is not None:
            for c in spec.conditions.columns:
                self.resolve(name, c, where=f"{where}.conditions", rule="reference")
            if spec.conditions.from_ != "self":
                self.table_ref(spec.conditions.from_, f"{where}.conditions.from", rule="reference")
        if spec.edge is not None:
            self.resolve(name, spec.edge.a, where=f"{where}.edge.a", rule="reference")
            self.resolve(name, spec.edge.b, where=f"{where}.edge.b", rule="reference")
            for side, cols in spec.edge.sides.items():
                for c in cols:
                    self.resolve(name, c, where=f"{where}.edge.sides.{side}", rule="reference_soft", level=lvl)
            for c in spec.edge.directed_when:
                self.resolve(name, c, where=f"{where}.edge.directed_when", rule="reference")
        elif spec.kind == "edges":
            self.add("error", where, "kind: edges needs an `edge` block (endpoints, orientation)", "reference",
                     table=name)
        if spec.roles_from is not None:
            self.table_ref(spec.roles_from.table, f"{where}.roles_from", rule="reference_soft", level=lvl)
        if spec.size_from is not None:
            self.table_ref(spec.size_from.table, f"{where}.size_from", rule="reference_soft", level=lvl)
        if spec.implements is not None and not re.match(r"^\w+:\w+(@[\w.\-]+)?$", spec.implements):
            self.add("error", f"{where}.implements", "implements is 'source:table@release'", "reference",
                     spec.implements, name)
        if spec.matrix is not None:
            for axis, ax in spec.matrix.axes.items():
                # an axis key is a declared axis field: a column, a parsed header field or the exposed index
                fields = set(ax.columns) | set(ax.parse.fields if ax.parse is not None else ()) | (
                    {ax.index_name} if ax.index_name else set())
                for c in ax.key.columns:
                    if c not in fields:
                        self.add("error", f"{where}.matrix.axes.{axis}.key", f"axis key {c!r} is not an axis field",
                                 "key", c, name)
        self.lint_plugins_table(name, spec)

    def lint_strict_roles(self, name: str, spec: TableSpec, sites: list[_Site]) -> None:
        level = self.strict_level(name)
        if level is None:
            return
        for s in sites:
            if isinstance(s.col, NestedCol) and not s.col.fields:
                self.add(level, f"tables.{name}.columns.{s.path}", "nested container has no roled fields",
                         "strict_roles", table=name)
        if spec.items_of is None and spec.matrix is None and not spec.columns and spec.roles_from is None:
            self.add(level, f"tables.{name}", "no roled columns", "strict_roles", table=name)

    def _key_column(self, table: str, column: str, where: str) -> Resolved | None:
        return self.resolve(table, column, where=where, rule="key")

    def lint_key(self, name: str, spec: TableSpec) -> None:
        where = f"tables.{name}.key"
        if spec.items_of is not None:
            if spec.key.columns:
                self.add("error", where, "an item table's key is composed from its parent: key.columns must be []",
                         "key", table=name)
            return
        keys = [("key", spec.key)] + [(f"alternate_keys[{i}]", k) for i, k in enumerate(spec.alternate_keys)]
        for label, key in keys:
            kw = f"tables.{name}.{label}"
            if not key.columns:
                self.add("error", kw, "key.columns is empty (a table needs its complete key)", "key", table=name)
                continue
            for c in key.columns:
                r = self._key_column(name, c, kw)
                if r is None or r.column is None:
                    continue
                role = _role(r.column)
                if role is not None and role not in KEYABLE_ROLES:
                    self.add("error", kw, f"key part {c!r} has role {role.value}, which cannot be a key part", "key",
                             c, name)
                if c in key.nullable and getattr(r.column, "self_", False):
                    self.add("error", kw, f"key part {c!r} is the table's self identifier and cannot be nullable",
                             "key", c, name)
            # An alternate key may be one nullable column (unique where present).
            if label == "key" and key.nullable and set(key.nullable) >= set(key.columns):
                self.add("error", kw, "every key part is nullable", "key", table=name)

    def lint_items_of(self, name: str, spec: TableSpec) -> None:
        if spec.items_of is None:
            return
        where = f"tables.{name}.items_of"
        parent = spec.items_of.table
        if parent not in self.desc.tables:
            self.add("error", where, f"items_of names unknown table {parent!r}", "key", parent, name)
            return
        try:
            segs = [s for s in parse_path(spec.items_of.path).segments if s.name]
        except PathError as exc:
            self.add("error", where, f"bad path: {exc}", "key", spec.items_of.path, name)
            return
        psc = self.scope(parent)
        if psc is None:
            return
        try:
            fields = psc.levels()[-1]
        except ScopeError:
            return                                     # the parent's own items_of is reported on the parent
        for i, seg in enumerate(segs):
            so_far = ".".join(s.name for s in segs[: i + 1])
            col = fields.get(seg.name)
            if col is None:
                self.add("error", where, f"container {so_far!r} is not declared on {parent!r}", "key",
                         f"{parent}.{so_far}", name)
                return
            if not is_container(col):
                self.add("error", where, f"{so_far!r} is not a nested or member container", "key", table=name)
                return
            # Every list container on the path contributes its item key to the composed key.
            if seg.is_list and col.item_key is None:
                self.add("error", where, f"container {so_far!r} has no item_key, so the item table's key cannot be "
                         "composed", "key", table=name)
            fields = dict(col.fields)

    def lint_column(self, s: _Site) -> None:
        col, table = s.col, s.table
        where = f"tables.{table}.columns.{s.path}" if s.axis is None else f"tables.{table}.matrix.{s.axis}.{s.path}"
        ctx = s.container + ((LIST,) if s.per_element else ())
        inner = s.container + (s.name,)
        role = _role(col)
        if role is None:
            return

        def ref_column(value: str | None, facet: str, *, mode: str = "column", rule: str = "reference",
                       level: Level = "error", container: Sequence[str] = ctx, innermost: bool = False) -> None:
            if value:
                self.resolve(table, value, container, mode=mode, where=f"{where}.{facet}", rule=rule, level=level,
                             innermost=innermost)

        soft = self.soft_level(table)
        # common facets
        for c in getattr(col, "unique_within", []) or []:
            ref_column(c, "unique_within")
        scope = getattr(col, "scope", None)
        if scope is not None:
            for c in scope.determined_by:
                ref_column(c, "scope.determined_by")
        for c in (getattr(col, "applies_when", None) or {}):
            ref_column(c, "applies_when")
        ref_column(getattr(col, "equals", None), "equals", rule="reference_soft", level=soft)
        for pv, override in (getattr(col, "by_partition", {}) or {}).items():
            if isinstance(override, dict) and override.get("id_type"):
                self.id_type(override["id_type"], f"{where}.by_partition.{pv}", table=table)
        # role facets
        if role in (Role.identifier, Role.endpoint, Role.label):
            self.id_type(getattr(col, "id_type", None), f"{where}.id_type", table=table)
        if role in (Role.identifier, Role.endpoint, Role.member):
            ref = getattr(col, "ref", None)
            if isinstance(ref, str):
                ref_column(ref, "ref", mode="table_column")
            elif isinstance(ref, CompositeRef):
                hit = self.table_ref(ref.table, f"{where}.ref", rule="reference")
                if hit is not None:
                    d, t = hit
                    for target, local in ref.on.items():
                        self.column_in(d, t, target, f"{where}.ref.on", "reference")
                        ref_column(local, "ref.on")
        if role == Role.identifier:
            self.id_type(col.maps_to, f"{where}.maps_to", table=table)
            ref_column(col.retired_into, "retired_into")
            if isinstance(col.id_type_from, IdTypeFrom):
                ref_column(col.id_type_from.field, "id_type_from.field")
                for v in col.id_type_from.map.values():
                    self.id_type(v, f"{where}.id_type_from.map", table=table)
            if col.kind_from is not None:
                ref_column(col.kind_from.column, "kind_from.column")
                for v in list(col.kind_from.map.values()) + ([col.kind_from.otherwise] if col.kind_from.otherwise
                                                             else []):
                    self.id_type(v, f"{where}.kind_from", table=table)
        if role in (Role.label, Role.synonym, Role.hierarchy):
            ref_column(getattr(col, "of", None), "of")
        if role == Role.synonym:
            ref_column(col.synonym_kind_from, "synonym_kind_from")
        if role in (Role.category, Role.scope):
            ref_column(col.hierarchy_via, "hierarchy_via")
            if col.aliases_from is not None:
                hit = self.table_ref(col.aliases_from.table, f"{where}.aliases_from", rule="reference")
                if hit is not None:
                    self.column_in(hit[0], hit[1], col.aliases_from.column, f"{where}.aliases_from", "reference")
        if role == Role.scope:
            ref_column(col.unit_from, "unit_from")
        if role == Role.measure:
            ref_column(col.unit_from, "unit_from")
            ref_column(col.of, "of")
            ref_column(col.part_of, "part_of")
            for c in col.comparable_within:
                ref_column(c, "comparable_within")
            for c in col.family or []:
                ref_column(c, "family")
            for k, c in (col.columns or {}).items():
                ref_column(c, f"columns.{k}", rule="reference_soft", level=soft)
        if role == Role.count:
            ref_column(col.length_of, "length_of")
            for c in col.comparable_within:
                ref_column(c, "comparable_within")
            dist = col.distinct_by if isinstance(col.distinct_by, list) else ([col.distinct_by] if col.distinct_by
                                                                              else [])
            for c in dist:
                ref_column(c, "distinct_by", rule="reference_soft", level=soft)
        if role == Role.flag:
            ref_column(col.event_of, "event_of")
        if role == Role.time:
            for fb in col.fallback:
                ref_column(fb.column, "fallback")
        if role == Role.hierarchy:
            ref_column(col.closure_of, "closure_of")
            self.id_type(col.target_id_type, f"{where}.target_id_type", table=table)
        if role == Role.vector:
            ref_column(col.norm_column, "norm_column")
        if role == Role.member:
            m = col.membership
            for end_name in ("set", "member"):
                end = getattr(m, end_name)
                if end.parent:
                    ref_column(end.parent, f"membership.{end_name}.parent")
                if end.path:
                    ref_column(end.path, f"membership.{end_name}.path", container=inner)
                self.id_type(end.id_type, f"{where}.membership.{end_name}.id_type", table=table)
                self.id_type(end.maps_to, f"{where}.membership.{end_name}.maps_to", table=table)
            if m.propagate_via and m.propagate_via.get("id_type"):
                self.id_type(m.propagate_via["id_type"], f"{where}.membership.propagate_via", table=table)
            if m.enrichment is not None and m.enrichment.universe is not None:
                self.lint_universe(m.enrichment.universe, f"{where}.membership.enrichment.universe", table)
        if isinstance(col, (NestedCol, MemberCol)):
            ik = col.item_key
            if ik is not None:
                for c in ik.columns:
                    if c == LIST:
                        continue
                    ref_column(c, "item_key", container=inner, rule="reference", innermost=True)
            for i, r in enumerate(col.rank):
                if r.column not in SPECIAL_RANK_COLUMNS:
                    ref_column(r.column, f"rank[{i}]", container=inner)
            if col.coverage is not None:
                self.lint_coverage(table, col.coverage, f"{where}.coverage", container=inner)
        if role == Role.reference and self.registry is not None:
            for k in col.ref_kinds:
                if k != "url" and not self.registry.has("identifier", k):
                    self.add("error", f"{where}.ref_kinds", f"unknown identifier plugin {k!r}", "plugin", k, table)
        self.lint_verified(s, where)
        self.lint_statistic(col, where, table)

    def lint_verified(self, s: _Site, where: str) -> None:
        col = s.col
        if getattr(col, "verified", True) is not False:
            return
        if getattr(col, "verified_by", "data") in ("manifest", "upstream_doc"):
            return
        role = _role(col)
        if role == Role.measure and col.family:
            self.add("warning", where, "a multiple-testing family cannot be confirmed from data; declare "
                     "verified_by: upstream_doc (or manifest) with the citation", "verified", table=s.table)
        elif role in (Role.text, Role.reference, Role.payload, Role.ignore, Role.synonym):
            self.add("warning", where, f"verified: false on a {role.value} column has no confirmation path",
                     "verified", table=s.table)

    def lint_statistic(self, col: Any, where: str, table: str) -> None:
        if self.registry is None:
            return
        role = _role(col)
        if role not in (Role.measure, Role.scope):
            return
        stat = getattr(col, "statistic", None)
        if not stat or self.registry.has("statistic", stat):
            return
        fallback = getattr(col, "fallback", None)
        if fallback and self.registry.has("statistic", fallback):
            return
        msg = f"statistic {stat!r} is not registered" + (f" and fallback {fallback!r} is not either" if fallback
                                                          else "; add `fallback:` with a registered statistic")
        self.add("error", f"{where}.statistic", msg, "plugin", stat, table)

    def lint_plugins_table(self, name: str, spec: TableSpec) -> None:
        if self.registry is None:
            return
        where = f"tables.{name}"
        if spec.items_of is not None:
            return
        layout = self.desc.table_layout(name)
        fmt = self.desc.table_format(name)
        if layout and not self.registry.has("layout", layout):
            self.add("error", f"{where}.layout", f"unknown layout plugin {layout!r}", "plugin", layout, name)
        if fmt and fmt != "none" and not self.registry.has("format", fmt):
            self.add("error", f"{where}.format", f"unknown format plugin {fmt!r}", "plugin", fmt, name)

    def lint_scope_columns(self, name: str, spec: TableSpec, sites: list[_Site]) -> None:
        if spec.items_of is not None:
            return                                     # checked on the parent table, where the columns live
        key = set(spec.key.columns)
        for k in spec.alternate_keys:
            key |= set(k.columns)
        key_paths = {_strip(c) for c in key}
        aggregated = {a.dimension for a in spec.aggregated_over}
        sc = self.scope(name)
        root_self = self._self_key(spec)
        for s in sites:
            sf = getattr(s.col, "scope", None)
            if sf is None or s.axis in ("value", "section"):
                continue
            if sf.determined_by or s.name in aggregated or s.path in aggregated:
                continue
            if s.axis in ("row", "col"):
                continue                               # axis attributes are determined by the axis key
            if not s.container:
                if s.name in key or s.name in key_paths or root_self:
                    continue
            else:
                container_col = None
                if sc is not None:
                    try:
                        lv = sc.levels(s.container[:-1])[-1]
                        container_col = lv.get(s.container[-1])
                    except ScopeError:
                        container_col = None
                ik = getattr(container_col, "item_key", None)
                if ik is not None and (s.name in ik.columns):
                    continue
                if _strip(s.path) in key_paths:
                    continue
            self.add("error", f"tables.{name}.columns.{s.path}",
                     f"scope column {s.path!r} is not in the key or an enclosing item key; declare "
                     "scope.determined_by or the table's aggregated_over", "scope_key", s.path, name)

    @staticmethod
    def _self_key(spec: TableSpec) -> bool:
        """An entity keyed by its own identifier: every column is an attribute of that entity."""
        if len(spec.key.columns) != 1:
            return False
        col = spec.columns.get(spec.key.columns[0])
        return bool(getattr(col, "self_", False))

    def lint_coverage(self, table: str, cov: Any, where: str, container: Sequence[str] = ()) -> None:
        if cov.absence_means == "censored" and cov.censor is None:
            self.add("error", where, "absence_means: censored needs `censor` (the stored filter)", "coverage",
                     table=table)
        if cov.absence_means in ("absent", "censored") and not (cov.statement or "").strip():
            self.add("error", where, f"absence_means: {cov.absence_means} needs a statement", "coverage",
                     table=table)
        if cov.censor is not None:
            self.resolve(table, cov.censor.column, container, where=f"{where}.censor", rule="reference_soft",
                         level=self.soft_level(table))
        if cov.universe is not None:
            self.lint_universe(cov.universe, f"{where}.universe", table)

    def lint_universe(self, u: UniverseSpec, where: str, table: str | None) -> None:
        hit = self.table_ref(u.table, where, rule="resolver_columns")
        if hit is None:
            return
        d, t = hit
        for k in u.keys:
            self.column_in(d, t, k, where, "resolver_columns")
        for c in u.per_scope:
            self.column_in(d, t, c, f"{where}.per_scope", "resolver_columns")

    def lint_canonical_grain(self, text: str, where: str, table: str) -> None:
        idt, _, _attr = text.rpartition(".")
        if not idt:
            self.add("error", where, f"canonicalize is '<id_type>.parent', got {text!r}", "reference", text, table)
            return
        self.id_type(idt, where, table=table)
        src, bare = split_qualified(idt)
        d = self.desc if src is None or src == self.desc.source else self.loaded.get(src)
        if d is not None and bare in d.id_types and d.id_types[bare].canonicalize is None:
            self.add("error", where, f"id_type {idt!r} declares no canonicalize (needed by {text!r})", "reference",
                     idt, table)

    # -- id_types ------------------------------------------------------------

    def lint_id_type(self, name: str, it: Any) -> None:
        where = f"id_types.{name}"
        if self.registry is not None and not self.registry.has("identifier", it.plugin):
            self.add("error", f"{where}.plugin", f"unknown identifier plugin {it.plugin!r}", "plugin", it.plugin)
        universes = it.universe if isinstance(it.universe, list) else ([it.universe] if it.universe else [])
        for u in universes:
            if isinstance(u, UniverseSpec):
                self.lint_universe(u, f"{where}.universe", None)
            else:
                self.lint_table_column(u, f"{where}.universe")
        self.id_type(it.label_of, f"{where}.label_of")
        for m in it.union:
            self.id_type(m, f"{where}.union")
        if it.extends:
            self.id_type(it.extends, f"{where}.extends")
        crosswalk_names = {c.name for t in self.desc.id_types.values() for c in t.crosswalks}
        for i, m in enumerate(it.maps_to):
            self.id_type(m.id_type, f"{where}.maps_to[{i}]")
            if m.via and m.via not in crosswalk_names:
                self.table_ref(m.via, f"{where}.maps_to[{i}].via", rule="resolver_columns")
        for c in it.resolve_via:
            self.lint_table_column(c, f"{where}.resolve_via")
        if it.retired is not None:
            for c in it.retired.listed_in:
                self.lint_table_column(c, f"{where}.retired.listed_in")
            for facet in ("flag", "replaced_by", "consider"):
                v = getattr(it.retired, facet)
                if v:
                    self.lint_table_column(v, f"{where}.retired.{facet}")
        for i, x in enumerate(it.xref_via):
            self.lint_table_column(x.column, f"{where}.xref_via[{i}]")
            for v in x.namespaces.values():
                self.id_type(v, f"{where}.xref_via[{i}].namespaces")
            if x.id_type_from is not None:
                for v in x.id_type_from.map.values():
                    self.id_type(v, f"{where}.xref_via[{i}].id_type_from")
        for i, cw in enumerate(it.crosswalks):
            hit = self.table_ref(cw.table, f"{where}.crosswalks[{i}]", rule="resolver_columns")
            if hit is not None:
                for c in (cw.from_, cw.to):
                    self.column_in(hit[0], hit[1], c, f"{where}.crosswalks[{i}]", "resolver_columns")
        for key in it.stored_forms:
            self.lint_table_column(key, f"{where}.stored_forms")
        if it.canonicalize is not None:
            self.lint_table_column(it.canonicalize.parent, f"{where}.canonicalize")
        if it.hierarchy is not None:
            hit = self.table_ref(it.hierarchy.table, f"{where}.hierarchy", rule="resolver_columns")
            if hit is not None:
                for c in it.hierarchy.columns:
                    self.column_in(hit[0], hit[1], c, f"{where}.hierarchy", "resolver_columns")
        if it.disambiguate_with:
            home = self._universe_table(it)
            if home is not None:
                for c in it.disambiguate_with:
                    self.column_in(home[0], home[1], c, f"{where}.disambiguate_with", "reference_soft",
                                   level=self.soft_level(None))
        if it.label_of and it.universe:
            self.add("warning", where, "a label kind takes existence from its label_of type; universe ignored",
                     "id_type")

    def _universe_table(self, it: Any) -> tuple[SourceDescriptor, str] | None:
        u = it.universe[0] if isinstance(it.universe, list) and it.universe else it.universe
        if isinstance(u, UniverseSpec):
            t = u.table
        elif isinstance(u, str):
            parts = u.split(".")
            if len(parts) < 2:
                return None
            t = parts[0]
        else:
            return None
        if "." in t:
            src, _, tab = t.partition(".")
            d = self.desc if src == self.desc.source else self.loaded.get(src)
            return (d, tab) if d is not None and tab in d.tables else None
        return (self.desc, t) if t in self.desc.tables else None

    def lint_table_column(self, ref: str, where: str, rule: str = "resolver_columns") -> None:
        """``table.col`` / ``source.table.col`` must exist."""
        try:
            parts = [s.name for s in parse_path(ref).segments]
        except PathError as exc:
            self.add("error", where, f"bad path {ref!r}: {exc}", rule, ref)
            return
        if len(parts) < 2:
            self.add("error", where, f"{ref!r} must be table.column", rule, ref)
            return
        first = parts[0]
        if first in self.desc.tables:
            d, t, col = self.desc, first, ref[len(first) + 1:]
        elif len(parts) >= 3 and (first == self.desc.source or first in self.loaded):
            d = self.desc if first == self.desc.source else self.loaded[first]
            t = parts[1]
            col = ref[len(first) + len(t) + 2:]
            if t not in d.tables:
                self.add("error", where, f"unknown table {first}.{t}", rule, ref)
                return
        elif len(parts) >= 3:
            # source.table.col of a source that is not loaded (or a table.nested.col of an unknown table)
            self.add("warning", where, f"{ref!r}: neither a table of {self.desc.source} nor a loaded source "
                     f"(not checked)", rule, ref)
            return
        else:
            self.add("error", where, f"unknown table {first!r} in {ref!r}", rule, ref)
            return
        self.column_in(d, t, col, where, rule)

    # -- source --------------------------------------------------------------

    def lint_source(self) -> None:
        d = self.desc
        if d.leakage is not None:
            for facet in ("available_at", "changed_at"):
                v = getattr(d.leakage, facet)
                if v and not self._in_any_table(v):
                    self.add(self.soft_level(None), f"leakage.{facet}", f"{v!r} is not a column of any table",
                             "reference_soft", v)
            for v in d.leakage.redact:
                if not self._in_any_table(v):
                    self.add(self.soft_level(None), "leakage.redact", f"{v!r} is not a column of any table",
                             "reference_soft", v)
        for concept, refs in d.concepts.items():
            for r in refs:
                self.lint_table_column(r, f"concepts.{concept}", rule="reference_soft")
        for vname, view in d.views.items():
            for sname, sec in view.sections.items():
                self.table_ref(sec.table, f"views.{vname}.sections.{sname}", rule="reference")
        if self.registry is not None:
            for kind in ("layout", "format"):
                p = plugin_name(getattr(d.defaults, kind))
                if p and p != "none" and not self.registry.has(kind, p):
                    self.add("error", f"defaults.{kind}", f"unknown {kind} plugin {p!r}", "plugin", p)

    def _in_any_table(self, column: str) -> bool:
        for t in self.desc.tables:
            sc = self.scope(t)
            if sc is None:
                continue
            try:
                sc.resolve(column, ())
                return True
            except ScopeError:
                continue
        return False


def _gateway_languages() -> frozenset[str]:
    """``escape`` values the gateway parses itself instead of quoting through a format plugin
    (SOMA value filters, §11.4)."""
    from ..gateway.soma_filter import LANGUAGE
    return frozenset({LANGUAGE})


def _strip(path: str) -> str:
    try:
        return parse_path(path).strip_brackets().text
    except PathError:
        return path


def lint_descriptor(desc: SourceDescriptor, registry: Any = None, strict: bool | None = None,
                    loaded_sources: Mapping[str, SourceDescriptor] | None = None) -> list[Finding]:
    """Lint one descriptor. ``strict=True`` makes role completeness and soft references errors (``vbt ds lint
    --strict``); ``None`` follows the descriptor's ``strict`` with warnings (phase 1). Plugin names are
    checked only when ``registry`` is given."""
    return _DescriptorLinter(desc, registry, strict, loaded_sources or {}).run()


# ---------------------------------------------------------------------------
# Overlay lint
# ---------------------------------------------------------------------------

class _OverlayLinter:
    def __init__(self, overlay: Overlay, catalog: Any, registry: Any) -> None:
        self.ov = overlay
        self.sources = _sources_of(catalog)
        self.overlays = _overlays_of(catalog)
        self.registry = registry
        self.out: list[Finding] = []

    def add(self, level: Level, where: str, message: str, rule: str, target: str | None = None,
            table: str | None = None) -> None:
        self.out.append(Finding(level, f"{self.ov.server}.{where}", message, rule, target, table))

    # -- resolution helpers ------------------------------------------------

    def split_binding(self, text: str) -> tuple[str, str, str] | None:
        """``source.table.col...`` -> (source, table, column path)."""
        try:
            names = [s.name for s in parse_path(text).segments]
        except PathError:
            return None
        if len(names) < 3:
            return None
        src, tab = names[0], names[1]
        rest = text[len(src) + len(tab) + 2:]
        return src, tab, rest

    def table_of(self, ref: str, where: str, rule: str = "binding") -> tuple[SourceDescriptor, str] | None:
        src, _, tab = ref.partition(".")
        if not tab:
            self.add("error", where, f"{ref!r} must be source.table", rule, ref)
            return None
        d = self.sources.get(src)
        if d is None:
            self.add("error", where, f"unknown source {src!r} in {ref!r}", rule, ref, ref)
            return None
        if tab not in d.tables:
            self.add("error", where, f"unknown table {ref!r}", rule, ref, ref)
            return None
        return d, tab

    def column(self, text: str, where: str, rule: str = "binding") -> tuple[str, Resolved] | None:
        """Resolve a binding ``source.table.col`` -> ("source.table", Resolved)."""
        parts = self.split_binding(text)
        if parts is None:
            self.add("error", where, f"{text!r} must be source.table.column", rule, text)
            return None
        src, tab, rest = parts
        hit = self.table_of(f"{src}.{tab}", where, rule)
        if hit is None:
            return None
        d, t = hit
        return self.column_in(d, t, rest, where, rule)

    def column_in(self, d: SourceDescriptor, table: str, column: str, where: str,
                  rule: str = "binding") -> tuple[str, Resolved] | None:
        try:
            r = TableScope(d, table, self.sources).resolve(column, (), mode="column")
        except ScopeError as exc:
            self.add("error", where, f"{d.source}.{table}: {exc}", rule, f"{d.source}.{table}.{column}",
                     f"{d.source}.{table}")
            return None
        return f"{d.source}.{table}", r

    def id_type(self, name: str, bound_source: str | None, where: str) -> str | None:
        """Qualified id_type for an ``accepts`` entry, or None (finding recorded)."""
        src, bare = split_qualified(name)
        if src is not None:
            d = self.sources.get(src)
            if d is None:
                self.add("warning", where, f"accepted kind {name!r}: source {src!r} is not loaded (not checked)",
                         "accepts", name)
                return None
            if bare not in d.id_types:
                self.add("error", where, f"unknown accepted kind {name!r}", "accepts", name)
                return None
            return name
        if bound_source and bound_source in self.sources and bare in self.sources[bound_source].id_types:
            return f"{bound_source}:{bare}"
        hits = sorted(s for s, d in self.sources.items() if bare in d.id_types)
        if len({id_type_identity(self.sources[s], self.sources[s].id_types[bare]) for s in hits}) == 1:
            return f"{hits[0]}:{bare}"                 # one source, or several with the same identity
        if not hits:
            self.add("error", where, f"unknown accepted kind {name!r} (no loaded source declares it)", "accepts",
                     name)
        else:
            self.add("error", where, f"accepted kind {name!r} is declared by {hits}; qualify it (source:{bare})",
                     "accepts", name)
        return None

    def id_spec(self, qualified: str) -> Any:
        src, bare = split_qualified(qualified)
        d = self.sources.get(src or "")
        return d.id_types.get(bare) if d is not None else None

    def neighbours(self, qualified: str) -> set[str]:
        """Declared id_type edges (label_of, maps_to, crosswalk via maps_to, union, extends) in both directions."""
        out: set[str] = set()
        src, bare = split_qualified(qualified)
        spec = self.id_spec(qualified)
        if spec is not None and src:
            for ref in ([spec.label_of] if spec.label_of else []) + list(spec.union) + \
                       ([spec.extends] if spec.extends else []) + [m.id_type for m in spec.maps_to]:
                out.add(qualify(ref, src))
        for s, d in self.sources.items():
            for n, other in d.id_types.items():
                q = f"{s}:{n}"
                refs = ([other.label_of] if other.label_of else []) + list(other.union) + \
                    ([other.extends] if other.extends else []) + [m.id_type for m in other.maps_to]
                if qualified in {qualify(r, s) for r in refs}:
                    out.add(q)
        return out

    def reachable(self, start: str, goal: str) -> bool:
        seen = {start}
        frontier = [start]
        while frontier:
            nxt = []
            for node in frontier:
                if node == goal:
                    return True
                for n in self.neighbours(node):
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return goal in seen

    @staticmethod
    def bound_id_type(col: Any) -> str | None:
        role = _role(col)
        if role in (Role.identifier, Role.endpoint, Role.label):
            return getattr(col, "id_type", None)
        if role == Role.member:
            return col.membership.member.id_type
        return None

    # -- run ---------------------------------------------------------------

    def run(self) -> list[Finding]:
        for src in self.ov.sources:
            if src not in self.sources:
                self.add("warning", "sources", f"source {src!r} is not loaded", "binding", src)
        defect_ids: dict[str, str] = {}
        for tool, b in self.ov.tools.items():
            self.lint_tool(tool, b)
            for d in b.defects:
                if d.id in defect_ids:
                    self.add("warning", f"tools.{tool}.defects", f"defect id {d.id!r} also used by "
                             f"{defect_ids[d.id]}", "defect", d.id)
                defect_ids[d.id] = tool
        return self.out

    def result_table(self, b: ToolBinding) -> tuple[SourceDescriptor, str] | None:
        """The table the result rows belong to (rows_of, derived.table, first bound table, single read)."""
        cands: list[str] = []
        if b.result.rows_of:
            cands.append(b.result.rows_of)
        if b.derived is not None:
            cands.append(b.derived.table)
        for a in b.args.values():
            for c in a.bound_columns:
                parts = self.split_binding(c)
                if parts:
                    cands.append(f"{parts[0]}.{parts[1]}")
        if not cands and len(b.reads) == 1:
            cands.append(next(iter(b.reads)))
        for ref in cands:
            src, _, tab = ref.partition(".")
            d = self.sources.get(src)
            if d is not None and tab in d.tables:
                return d, tab
        return None

    def lint_tool(self, tool: str, b: ToolBinding) -> None:
        w = f"tools.{tool}"
        for ref in b.reads:
            self.table_of(ref, f"{w}.reads")
        args = b.args
        selectors = [n for n, a in args.items() if a.role == "selector"]
        for name, a in args.items():
            self.lint_arg(tool, name, a, b, selectors)
        for group_kind in ("require_any", "exclusive"):
            for group in getattr(b, group_kind):
                for n in group:
                    if n not in args:
                        self.add("error", f"{w}.{group_kind}", f"{n!r} is not an argument of the binding", "binding",
                                 n)
        # serve modes
        if b.serve == "derived" and b.derived is None:
            self.add("error", f"{w}.serve", "serve: derived needs a `derived` spec", "binding")
        if b.serve == "block":
            hidden = b.hidden or (b.block is not None and b.block.hidden)
            if b.block is None or (not b.block.alternatives and not hidden):
                self.add("error", f"{w}.serve", "serve: block needs block.alternatives (or hidden: true)", "binding")
        if b.on_contradiction == "derived" and b.derived is None and b.serve == "pass" and b.witness:
            pass  # without `derived` a contradiction becomes tool_defect (§8.2); not an error
        if b.derived is not None:
            self.lint_derived(b.derived, f"{w}.derived")
        if b.leakage_filter is not None and b.leakage_filter.arg not in args:
            self.add("error", f"{w}.leakage_filter", f"{b.leakage_filter.arg!r} is not an argument", "binding",
                     b.leakage_filter.arg)
        self.lint_result(tool, b)
        for i, d in enumerate(b.defects):
            if not d.where or not DEFECT_WHERE.match(d.where):
                self.add("error", f"{w}.defects[{i}]", f"defect {d.id}: where must be file:line (got {d.where!r})",
                         "defect", d.where)
        for alias in b.same_as:
            server, _, other = alias.partition(".")
            if not server or not other or "." in other:
                self.add("error", f"{w}.same_as", f"{alias!r} must be server.tool", "binding", alias)
                continue
            ov = self.overlays.get(server)
            if ov is None:
                if server != self.ov.server:
                    self.add("warning", f"{w}.same_as", f"no overlay loaded for server {server!r} (not checked)",
                             "binding", alias)
            elif other in ov.tools and ov is not self.ov:
                self.add("error", f"{w}.same_as", f"{alias} has its own binding; same_as would bind it twice",
                         "binding", alias)
            elif ov is self.ov and other not in self.ov.tools and other == tool:
                self.add("error", f"{w}.same_as", f"{alias} names this tool itself", "binding", alias)
        self.lint_scope_bindings(tool, b)

    def lint_arg(self, tool: str, name: str, a: ArgBinding, b: ToolBinding, selectors: list[str]) -> None:
        w = f"tools.{tool}.args.{name}"
        if isinstance(a.binds, dict) and not selectors:
            self.add("error", f"{w}.binds", "binds as {selector value: column} needs a role: selector argument",
                     "binding")
        if a.role in ("filter", "anchor") and not a.bound_columns:
            self.add("error", w, f"a {a.role} argument must bind a column (use role: unbound for an unbindable "
                     "filter)", "binding")
        resolved: list[tuple[str, Resolved]] = []
        for c in a.bound_columns:
            hit = self.column(c, f"{w}.binds")
            if hit is not None:
                resolved.append(hit)
        if a.role == "selector":
            for value, target in a.values.items():
                if isinstance(target, str) and target.count(".") == 1:
                    self.table_of(target, f"{w}.values.{value}")
        if a.role == "order_by":
            for value, target in a.values.items():
                if isinstance(target, dict) and target.get("column"):
                    rt = self.result_table(b)
                    if rt is not None:
                        self.column_in(rt[0], rt[1], str(target["column"]), f"{w}.values.{value}")
        if a.pattern is not None:
            try:
                re.compile(a.pattern)
            except re.error as exc:
                self.add("error", f"{w}.pattern", f"pattern does not compile: {exc}", "binding")
        if a.escape and a.escape not in _gateway_languages() and self.registry is not None \
                and not self.registry.has("format", a.escape):
            self.add("error", f"{w}.escape", f"unknown format plugin {a.escape!r}", "plugin", a.escape)
        for flag_key in ("when_true", "when_false"):
            pred = getattr(a, flag_key)
            if pred and "ne" in pred:
                coded = any(getattr(r.column, "encoding", None) or isinstance(getattr(r.column, "vocab", None), list)
                            for _, r in resolved)
                self.add("error" if coded else "warning", f"{w}.{flag_key}",
                         "name the favourable codes positively ({in: [...]}); `ne` lets NaN, values between codes "
                         "and new codes count as the favourable state", "flag_codes")
        if a.accepts:
            self.lint_accepts(tool, name, a, resolved)

    def lint_accepts(self, tool: str, name: str, a: ArgBinding, resolved: list[tuple[str, Resolved]]) -> None:
        w = f"tools.{tool}.args.{name}.accepts"
        bound_source = resolved[0][0].split(".")[0] if resolved else None
        quals = [(raw, self.id_type(raw, bound_source, w)) for raw in a.accepts]
        if self.registry is not None:
            self.lint_digits(w, [q for _, q in quals if q])
        if not resolved:
            return
        table, r = resolved[0]
        bound = self.bound_id_type(r.column)
        if bound is None:
            if r.column is not None and _role(r.column) not in (Role.reference, Role.category, Role.scope):
                self.add("warning", w, f"bound column {table}.{r.path} has no id_type; accepted kinds cannot be "
                         "checked", "accepts")
            return
        src = r.source or table.split(".")[0]
        bound_q = qualify(bound, src)
        if self.id_spec(bound_q) is None:
            return                                     # reported by the descriptor lint
        for raw, q in quals:
            if q is None or q == bound_q:
                continue
            spec = self.id_spec(q)
            if spec is not None and not spec.resolvable:
                continue                               # syntax-only kinds are not resolved into the column
            if not self.reachable(q, bound_q):
                self.add("error", w, f"accepted kind {raw!r} does not reach the bound column's id_type {bound_q} "
                         "through label_of, maps_to, a crosswalk or a union", "accepts", raw)

    def lint_digits(self, where: str, kinds: list[str]) -> None:
        """A digits-only kind is ambiguous next to other kinds unless it requires a prefix for agent input."""
        if len(kinds) < 2:
            return
        for q in kinds:
            spec = self.id_spec(q)
            if spec is None:
                continue
            plugin = self.registry.find("identifier", spec.plugin)
            examples = tuple(getattr(plugin, "examples", ()) or ())
            if examples and all(str(e).isdigit() for e in examples) and \
                    not spec.options.get("input_requires_prefix"):
                self.add("error", where, f"digits-only kind {q!r} accepted next to other kinds; set options."
                         "input_requires_prefix or accept it alone", "digits_kind", q)

    def lint_derived(self, d: DerivedSpec, where: str) -> None:
        self.table_of(d.table, f"{where}.table")
        for sname, sec in d.sections.items():
            self.table_of(sec.table, f"{where}.sections.{sname}")
        for i, sub in enumerate(d.compose):
            self.lint_derived(sub, f"{where}.compose[{i}]")

    def lint_result(self, tool: str, b: ToolBinding) -> None:
        w = f"tools.{tool}.result"
        res = b.result
        if res.codec and self.registry is not None and not self.registry.has("envelope", res.codec):
            self.add("error", f"{w}.codec", f"unknown envelope plugin {res.codec!r}", "plugin", res.codec)
        if res.rows_of:
            hit = self.table_of(res.rows_of, f"{w}.rows_of")
            if hit is not None and hit[0].tables[hit[1]].items_of is None:
                self.add("error", f"{w}.rows_of", f"{res.rows_of!r} is not an item table", "binding", res.rows_of)
        if res.grain not in (None, "row", "rows") and b.witness:
            # the witness counts this grain: the result table (or the table it is an item table of) declares it
            hit = self.result_table(b)
            if hit is not None:
                spec_t = hit[0].tables[hit[1]]
                parent = hit[0].tables.get(spec_t.items_of.table) if spec_t.items_of is not None else None
                if res.grain not in spec_t.grains and (parent is None or res.grain not in parent.grains):
                    self.add("error", f"{w}.grain", f"{hit[0].source}.{hit[1]} declares no grain {res.grain!r} "
                             "(use row, a declared grain, or witness: false)", "binding", res.grain)
        for arg, spec in res.echo_specs().items():
            if isinstance(spec, EchoSpec) and spec.source == "request":
                self.add("error", f"{w}.echo.{arg}", "an echo must read the record (source: record), not copy the "
                         "request", "echo")
            if arg not in b.args:
                self.add("error", f"{w}.echo.{arg}", f"{arg!r} is not an argument", "echo", arg)
        if res.echo_set is not None and res.echo_set.arg not in b.args:
            self.add("error", f"{w}.echo_set", f"{res.echo_set.arg!r} is not an argument", "echo", res.echo_set.arg)
        for col, arg in res.key_from_args.items():
            if arg not in b.args:
                self.add("error", f"{w}.key_from_args", f"{arg!r} is not an argument", "binding", arg)
        if res.order_from_arg is not None:
            a = b.args.get(res.order_from_arg)
            if a is None or a.role != "order_by":
                self.add("error", f"{w}.order_from_arg", f"{res.order_from_arg!r} is not a role: order_by argument",
                         "binding", res.order_from_arg)
        rt = self.result_table(b)
        computed = {k for k, f in res.fields.items() if f.computed is not None}
        if rt is None:
            if res.fields or res.order:
                self.add("warning", w, "cannot tell which table the rows belong to; fields and order not checked",
                         "binding")
            return
        d, t = rt
        for field_name, fm in res.fields.items():
            if fm.column is not None:
                self.column_in(d, t, fm.column, f"{w}.fields.{field_name}")
        for i, r in enumerate(res.order):
            if r.column in SPECIAL_RANK_COLUMNS or r.column in computed:
                continue
            self.column_in(d, t, r.column, f"{w}.order[{i}]")
            if r.statistic and self.registry is not None and not self.registry.has("statistic", r.statistic):
                self.add("error", f"{w}.order[{i}]", f"unknown statistic plugin {r.statistic!r}", "plugin",
                         r.statistic)
        for path, summary in res.summary_fields.items():
            if not isinstance(summary, str) and summary.of:
                try:
                    TableScope(d, t, self.sources).resolve(summary.of, ())
                except ScopeError as exc:
                    self.add("warning", f"{w}.summary_fields", str(exc), "binding", summary.of)
        for sname, sec in res.sections.items():
            self.table_of(sec.table, f"{w}.sections.{sname}")

    def lint_scope_bindings(self, tool: str, b: ToolBinding) -> None:
        """A forbid-pooled scope key column is fixed by an equality binding; an unbound one gets the
        auto-derived gateway-only argument (§10.1), a non-equality binding cannot fix it."""
        by_table: dict[str, dict[str, ArgBinding]] = {}
        for name, a in b.args.items():
            for c in a.bound_columns:
                parts = self.split_binding(c)
                if parts:
                    by_table.setdefault(f"{parts[0]}.{parts[1]}", {})[_strip(parts[2])] = a
        for ref, bound in by_table.items():
            src, _, tab = ref.partition(".")
            d = self.sources.get(src)
            if d is None or tab not in d.tables:
                continue
            spec = d.tables[tab]
            for k in spec.key.columns:
                col = spec.columns.get(k) or (spec.partitions[k].column if k in spec.partitions else None)
                sf = getattr(col, "scope", None)
                if sf is None or sf.pooling != "forbid":
                    continue
                arg = bound.get(_strip(k))
                if arg is not None and arg.op not in _EQUALITY_OPS:
                    self.add("error", f"tools.{tool}.args", f"scope key column {ref}.{k} (pooling: forbid) is bound "
                             f"with op {arg.op!r}, which cannot fix one value", "scope_binding", k, ref)


def lint_overlay(overlay: Overlay, catalog: Any, registry: Any = None) -> list[Finding]:
    """Lint one overlay against a catalog (or a ``{source: SourceDescriptor}`` mapping)."""
    return _OverlayLinter(overlay, catalog, registry).run()

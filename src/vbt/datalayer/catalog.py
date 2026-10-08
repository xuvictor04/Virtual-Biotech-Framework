"""The catalog: descriptors + overlays + id_type qualification, and per-tool contracts. No pyarrow.

* :meth:`Catalog.contract` finds the binding of ``server.tool``: a reviewed binding, a
  binding that lists the tool in ``same_as``, or the generic guard (``generic=True``; also for
  ``status: unreviewed`` bindings).
* :meth:`Catalog.table` resolves ``source.table``; an item table is resolved to its parent's
  fragments (``physical``) with its complete key composed from the parent key and the item
  keys along the container path (§6.2). ``data.sources.alias`` serves a source's tables from
  the tables of another source that declare ``implements`` (rev 2).
* :meth:`Catalog.id_type` honours qualification: ``source:name`` is exact; a bare name means
  the hint source's own type, else a unique match across loaded sources.
* R8: :func:`build_catalog` quarantines a descriptor or overlay file that does not load on its own
  (:class:`~.descriptor.load.Quarantined`, ``Catalog.quarantined``). :meth:`Catalog.contract` marks
  only the tools that depend on it (``ToolContract.quarantined``): every tool of a server whose overlay
  is quarantined, a tool whose binding references a table or id_type a quarantined descriptor declares
  (or may declare, when its YAML does not parse), and a tool on the generic guard while a generic
  overlay of its server is quarantined. :meth:`Catalog.lint` reports each file as an error.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .descriptor.columns import is_container
from .descriptor.load import digest as _digest
from .descriptor.load import Quarantined, load_descriptors, load_overlays, variables_from_config
from .descriptor.models import IdTypeSpec, SourceDescriptor, TableSpec, id_type_identity, plugin_name
from .descriptor.overlay import ArgBinding, GenericSpec, Overlay, ToolBinding
from .errors import ErrorKind, GatewayError, invalid_argument_payload
from .roles import LIST, Role, parse_path

__all__ = [
    "TableRef", "CatalogTable", "ToolContract", "Catalog", "CatalogError", "UnknownTable", "UnknownIdType",
    "AmbiguousIdType", "build_catalog", "load_catalog", "compose_key", "compose_nullable", "POSITION_MARK",
]

#: Marks a positional item-key part (``identity: position`` without ``max_items: 1``): ``"<container>[]#"``.
POSITION_MARK = "#"

log = logging.getLogger(__name__)
_WARNED: set[tuple[str, str]] = set()


class CatalogError(LookupError):
    pass


class UnknownTable(CatalogError):
    pass


class UnknownIdType(CatalogError):
    pass


class AmbiguousIdType(CatalogError):
    pass


@dataclass(frozen=True, order=True)
class TableRef:
    source: str
    table: str

    @classmethod
    def parse(cls, ref: "str | TableRef") -> "TableRef":
        if isinstance(ref, TableRef):
            return ref
        src, sep, tab = str(ref).partition(".")
        if not sep or not src or not tab or "." in tab:
            raise UnknownTable(f"table references are 'source.table', got {ref!r}")
        return cls(src, tab)

    def __str__(self) -> str:
        return f"{self.source}.{self.table}"


@dataclass
class CatalogTable:
    """One table as the gateway and the data child see it."""

    ref: TableRef                                      # the name bindings use
    spec: TableSpec
    descriptor: SourceDescriptor                       # the source whose spec this is
    physical: TableRef                                 # the table whose fragments hold the rows
    items_path: str | None = None                      # container path from the physical table ("items[]")
    columns: dict[str, Any] = field(default_factory=dict)   # the table's columns (container fields for items)
    key: tuple[str, ...] = ()                          # complete key (composed for item tables)
    container: Any = None                              # the container column of an item table
    served_from: TableRef | None = None                # data.sources.alias: the implementing table

    @property
    def is_item_table(self) -> bool:
        return self.items_path is not None

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def grain(self) -> str:
        return self.spec.grain

    @property
    def physical_spec(self) -> TableSpec:
        return self.descriptor.tables[self.physical.table]

    @property
    def layout(self) -> str | None:
        return self.descriptor.table_layout(self.physical.table)

    @property
    def format(self) -> str | None:
        return self.descriptor.table_format(self.physical.table)

    @property
    def nullable_key(self) -> tuple[str, ...]:
        """Key parts that may be null: the table's ``key.nullable``; for an item table, the parent's and the
        ``item_key.nullable`` parts of every container on its path (as composed key paths)."""
        if not self.is_item_table:
            return tuple(self.spec.key.nullable)
        return compose_nullable(self.descriptor, self.ref.table if self.served_from is None else
                                self.served_from.table)

    def scope_columns(self) -> dict[str, Any]:
        """Columns of this table that are scope dimensions (role scope or a scope facet), by path."""
        out: dict[str, Any] = {}
        for name, col in self.columns.items():
            if getattr(col, "scope", None) is not None:
                out[name] = col
        for name, part in self.physical_spec.partitions.items():
            if getattr(part.column, "scope", None) is not None:
                out.setdefault(name, part.column)
        return out


def _segments(path: str) -> list[Any]:
    return [s for s in parse_path(path).segments if s.name]


def compose_key(desc: SourceDescriptor, table: str) -> tuple[tuple[str, ...], str | None, Any, dict[str, Any], str]:
    """``(key, items_path, container, columns, physical table)`` of ``table``.

    An item table's key is the parent key + the item keys of every list container on the
    path (prefixed by the container path), so an item table over ``items[]`` (item key ``[k1, k2]``)
    is keyed ``(id, items[].k1, items[].k2)``. ``identity: value`` contributes the container itself
    (``members[]``); ``identity: position`` contributes ``<container>[]#`` unless
    ``max_items: 1`` (a singleton contributes nothing)."""
    spec = desc.tables[table]
    if spec.items_of is None:
        return tuple(spec.key.columns), None, None, dict(spec.columns), table
    parent = spec.items_of.table
    if parent not in desc.tables:
        raise UnknownTable(f"{desc.source}.{table}: items_of names unknown table {parent!r}")
    pkey, ppath, pcont, pfields, physical = compose_key(desc, parent)
    key = list(pkey)
    fields = pfields
    prefix = (ppath + ".") if ppath else ""
    container = pcont
    text = ""
    for seg in _segments(spec.items_of.path):
        col = fields.get(seg.name)
        if col is None or not is_container(col):
            raise UnknownTable(f"{desc.source}.{table}: container {seg.name!r} not found on {parent!r}")
        text = prefix + seg.name + (LIST * len(seg.brackets) if seg.is_list else "")
        if seg.is_list:
            ik = col.item_key
            if ik is not None:
                if ik.identity == "value":
                    key.append(text)
                elif ik.identity == "position":
                    if ik.max_items != 1:
                        key.append(text + POSITION_MARK)
                else:
                    key.extend(f"{text}.{c}" if c != LIST else text for c in ik.columns)
        prefix = text + "."
        container = col
        fields = dict(col.fields)
    return tuple(key), text, container, fields, physical


def compose_nullable(desc: SourceDescriptor, table: str) -> tuple[str, ...]:
    """The nullable parts of :func:`compose_key`'s key of ``table``: the physical table's ``key.nullable`` and
    each list container's ``item_key.nullable`` (prefixed by the container path)."""
    spec = desc.tables[table]
    if spec.items_of is None:
        return tuple(spec.key.nullable)
    parent = spec.items_of.table
    if parent not in desc.tables:
        raise UnknownTable(f"{desc.source}.{table}: items_of names unknown table {parent!r}")
    out = list(compose_nullable(desc, parent))
    _key, ppath, _cont, fields, _physical = compose_key(desc, parent)
    prefix = (ppath + ".") if ppath else ""
    for seg in _segments(spec.items_of.path):
        col = fields.get(seg.name)
        if col is None or not is_container(col):
            raise UnknownTable(f"{desc.source}.{table}: container {seg.name!r} not found on {parent!r}")
        text = prefix + seg.name + (LIST * len(seg.brackets) if seg.is_list else "")
        ik = col.item_key if seg.is_list else None
        if ik is not None and ik.identity == "key":
            out.extend(f"{text}.{c}" if c != LIST else text for c in ik.nullable)
        prefix = text + "."
        fields = dict(col.fields)
    return tuple(out)


@dataclass
class ToolContract:
    """The binding of one tool plus the tables and descriptors it involves."""

    server: str
    tool: str
    binding: ToolBinding | None
    tables: dict[str, CatalogTable] = field(default_factory=dict)
    descriptors: dict[str, SourceDescriptor] = field(default_factory=dict)
    generic: bool = False
    overlay: Overlay | None = None
    alias_of: str | None = None                        # "server.tool" whose binding lists this tool in same_as
    generic_spec: GenericSpec | None = None
    quarantined: tuple[Quarantined, ...] = ()          # R8: files this tool depends on that do not load
    missing: tuple[str, ...] = ()                      # table references the loaded catalog cannot resolve

    @property
    def quarantine_reason(self) -> str:
        """``<file>: <error>`` of each quarantined file the tool depends on ('' when none)."""
        return "; ".join(f"{q.file}: {q.summary}" for q in self.quarantined)

    @property
    def name(self) -> str:
        return f"{self.server}.{self.tool}"

    @property
    def args(self) -> dict[str, ArgBinding]:
        return dict(self.binding.args) if self.binding is not None else {}

    @property
    def full_table_reads(self) -> tuple[str, ...]:
        """Tables the upstream tool loads whole (admission estimates them)."""
        if self.binding is None:
            return ()
        return tuple(ref for ref, rs in self.binding.reads.items() if rs.access == "full_table")

    @property
    def identifier_args(self) -> dict[str, ArgBinding]:
        """Arguments resolved by the resolver: ``accepts`` set, ``role: anchor``, or bound to an
        identifier/endpoint column."""
        out: dict[str, ArgBinding] = {}
        for name, a in self.args.items():
            if a.accepts or a.role == "anchor":
                out[name] = a
                continue
            for table, col in self.arg_columns(name):
                spec = self._column(table, col)
                if getattr(spec, "role", None) in (Role.identifier.value, Role.endpoint.value):
                    out[name] = a
                    break
        return out

    @property
    def limit_arg(self) -> str | None:
        for name, a in self.args.items():
            if a.role == "limit":
                return name
        return None

    @property
    def selector_args(self) -> list[str]:
        return [n for n, a in self.args.items() if a.role == "selector"]

    def arg_columns(self, name: str) -> list[tuple[str, str]]:
        """``[(source.table, column path)]`` an argument binds."""
        a = self.args.get(name)
        if a is None:
            return []
        out = []
        for c in a.bound_columns:
            segs = parse_path(c).segments
            if len(segs) >= 3:
                src, tab = segs[0].name, segs[1].name
                out.append((f"{src}.{tab}", c[len(src) + len(tab) + 2:]))
        return out

    def _column(self, table: str, column: str) -> Any:
        t = self.tables.get(table)
        if t is None:
            return None
        current: Mapping[str, Any] = t.columns
        col = None
        for seg in parse_path(column).segments:
            if not seg.name:
                continue
            col = current.get(seg.name)
            if col is None:
                cand = t.physical_spec.columns.get(seg.name) or (
                    t.physical_spec.partitions[seg.name].column if seg.name in t.physical_spec.partitions else None)
                col = cand
            if col is None:
                return None
            current = dict(getattr(col, "fields", {}) or {})
        return col

    @property
    def bound_table(self) -> str | None:
        """The table the tool's arguments filter (before selector resolution): the first table an
        argument binds, else ``derived.table``, else ``result.rows_of``'s, else the only read."""
        if self.binding is None:
            return None
        for name, a in self.args.items():
            if a.role in ("filter", "anchor", "flag") and isinstance(a.binds, str):
                cols = self.arg_columns(name)
                if cols:
                    return cols[0][0]
        if self.binding.derived is not None:
            return self.binding.derived.table
        if self.binding.result.rows_of:
            return self.binding.result.rows_of
        if len(self.binding.reads) == 1:
            return next(iter(self.binding.reads))
        return None

    def selected_table(self, args: Mapping[str, Any]) -> str | None:
        """The bound table after a ``role: selector`` argument chose one (``method: <value>`` ->
        ``source.table``). An unknown selector value raises ``invalid_argument``."""
        for name in self.selector_args:
            a = self.args[name]
            choices: dict[Any, str] = {}
            for value, target in a.values.items():
                if isinstance(target, str):
                    choices[value] = target if target.count(".") == 1 else ".".join(target.split(".")[:2])
            for value, target in (a.binds.items() if isinstance(a.binds, dict) else []):
                choices.setdefault(value, ".".join(str(target).split(".")[:2]))
            if not choices:
                continue
            raw = args.get(name)
            if raw is None:
                continue
            for value, table in choices.items():
                if value == raw or (a.match == "casefold" and str(value).casefold() == str(raw).casefold()):
                    return table
            raise GatewayError(ErrorKind.invalid_argument, f"unknown value for {name}",
                               tool=f"mcp__{self.server}__{self.tool}", argument=name, value=raw,
                               payload=invalid_argument_payload(name, raw, [str(v) for v in choices]))
        return self.bound_table

    def table(self, ref: str) -> CatalogTable:
        try:
            return self.tables[ref]
        except KeyError:
            raise UnknownTable(f"{self.name} does not involve table {ref!r}") from None

    def item_table_of(self, rows_of: str | None = None) -> CatalogTable | None:
        """The item table the result rows are items of (``result.rows_of``)."""
        ref = rows_of or (self.binding.result.rows_of if self.binding is not None else None)
        if not ref:
            return None
        t = self.tables.get(ref)
        return t if t is not None and t.is_item_table else None


def _table_refs(binding: ToolBinding) -> list[str]:
    refs: list[str] = list(binding.reads)
    for a in binding.args.values():
        for c in a.bound_columns:
            segs = parse_path(c).segments
            if len(segs) >= 3:
                refs.append(f"{segs[0].name}.{segs[1].name}")
        if a.role == "selector":
            for target in a.values.values():
                if isinstance(target, str) and target.count(".") >= 1:
                    refs.append(".".join(target.split(".")[:2]))
    if binding.result.rows_of:
        refs.append(binding.result.rows_of)
    for sec in binding.result.sections.values():
        refs.append(sec.table)
    stack = [binding.derived] if binding.derived is not None else []
    while stack:
        d = stack.pop()
        refs.append(d.table)
        refs.extend(s.table for s in d.sections.values())
        stack.extend(d.compose)
    seen: dict[str, None] = {}
    for r in refs:
        seen.setdefault(r, None)
    return list(seen)


class Catalog:
    """Descriptors, overlays and generic overlays, with lookups for the gateway and the data child."""

    def __init__(self, descriptors: Mapping[str, SourceDescriptor], overlays: Mapping[str, Overlay] | None = None,
                 generic: Sequence[Overlay] = (), *, aliases: Mapping[str, str] | None = None,
                 registry: Any = None, quarantined: Sequence[Quarantined] = ()) -> None:
        self._sources = dict(descriptors)
        self._overlays = dict(overlays or {})
        self._generic = list(generic)
        self.aliases = dict(aliases or {})
        self.registry = registry
        self.quarantined: list[Quarantined] = sorted(quarantined, key=lambda q: (q.path, q.error))
        self._tables: dict[TableRef, CatalogTable] = {}

    # -- collections -------------------------------------------------------

    @property
    def sources(self) -> dict[str, SourceDescriptor]:
        return self._sources

    @property
    def overlays(self) -> dict[str, Overlay]:
        return self._overlays

    @property
    def generic(self) -> list[Overlay]:
        return self._generic

    def source(self, name: str) -> SourceDescriptor:
        try:
            return self._sources[name]
        except KeyError:
            raise CatalogError(f"unknown source {name!r} (loaded: {sorted(self._sources)})") from None

    def servers(self) -> list[str]:
        return sorted(self._overlays)

    def tools(self, server: str) -> list[str]:
        ov = self._overlays.get(server)
        return sorted(ov.tools) if ov is not None else []

    def table_refs(self) -> list[TableRef]:
        return sorted(TableRef(s, t) for s, d in self._sources.items() for t in d.tables)

    # -- tables ------------------------------------------------------------

    def table(self, ref: str | TableRef) -> CatalogTable:
        r = TableRef.parse(ref)
        if r in self._tables:
            return self._tables[r]
        served = self._alias_table(r)
        if served is not None:
            desc, table = served
            key, items, cont, cols, physical = compose_key(desc, table)
            out = CatalogTable(r, desc.tables[table], desc, TableRef(desc.source, physical), items, cols, key, cont,
                               served_from=TableRef(desc.source, table))
        else:
            own = self._sources.get(r.source)
            if own is None:
                raise UnknownTable(f"unknown source {r.source!r} in {r}")
            if r.table not in own.tables:
                raise UnknownTable(f"unknown table {r}")
            key, items, cont, cols, physical = compose_key(own, r.table)
            out = CatalogTable(r, own.tables[r.table], own, TableRef(r.source, physical), items, cols, key, cont)
        self._tables[r] = out
        return out

    def _alias_table(self, r: TableRef) -> tuple[SourceDescriptor, str] | None:
        target = self.aliases.get(r.source)
        if not target:
            return None
        desc = self._sources.get(target)
        if desc is None:
            raise UnknownTable(f"data.sources.alias maps {r.source!r} to {target!r}, which is not loaded")
        for name, spec in desc.tables.items():
            impl = spec.implements
            if impl and impl.split("@", 1)[0] == f"{r.source}:{r.table}":
                return desc, name
        return None

    # -- id_types ------------------------------------------------------------

    def id_type(self, name: str, source_hint: str | None = None) -> tuple[str, IdTypeSpec]:
        """``(source, spec)``: ``source:name`` exactly; a bare name in ``source_hint`` first, then the
        unique source declaring it."""
        if ":" in name:
            src, _, bare = name.partition(":")
            d = self._sources.get(src)
            if d is None or bare not in d.id_types:
                raise UnknownIdType(f"unknown id_type {name!r}")
            return src, d.id_types[bare]
        if source_hint and source_hint in self._sources and name in self._sources[source_hint].id_types:
            return source_hint, self._sources[source_hint].id_types[name]
        hits = sorted(s for s, d in self._sources.items() if name in d.id_types)
        if not hits:
            raise UnknownIdType(f"unknown id_type {name!r}")
        identities = {id_type_identity(self._sources[s], self._sources[s].id_types[name]) for s in hits}
        if len(identities) > 1:
            raise AmbiguousIdType(f"id_type {name!r} is declared by {hits} with different identity universes; "
                                  f"qualify it as source:{name}")
        return hits[0], self._sources[hits[0]].id_types[name]

    def qualify_id_type(self, name: str, source_hint: str | None = None) -> str:
        src, _ = self.id_type(name, source_hint)
        return f"{src}:{name.partition(':')[2] or name}" if ":" in name else f"{src}:{name}"

    # -- contracts -----------------------------------------------------------

    def contract(self, server: str, tool: str) -> ToolContract:
        own = tuple(q for q in self.quarantined if q.kind == "overlay" and q.name == server)
        if own:
            # the server's own overlay does not load: no binding of it can be trusted, nor the generic guard
            return ToolContract(server, tool, None, quarantined=own)
        ov = self._overlays.get(server)
        binding = ov.tools.get(tool) if ov is not None else None
        alias_of = None
        owner = ov
        if binding is None:
            for other in self._overlays.values():
                for name, b in other.tools.items():
                    if f"{server}.{tool}" in b.same_as:
                        binding, alias_of, owner = b, f"{other.server}.{name}", other
                        break
                if binding is not None:
                    break
        generic_spec = self._generic_spec(server)
        if binding is None:
            # a tool another server's overlay binds with same_as depends on that file: when it does not load, the
            # tool is quarantined, never served by the generic guard (target.get_pharmacogenomics in drug.yaml)
            elsewhere = self._same_as_quarantine(server, tool)
            if elsewhere:
                return ToolContract(server, tool, None, quarantined=elsewhere)
            for g in self._generic:
                if g.server in (server, "*") and tool in g.tools:
                    binding, owner = g.tools[tool], g
                    break
            if binding is None:
                return ToolContract(server, tool, None, generic=True, generic_spec=generic_spec,
                                    quarantined=self._generic_quarantine(server))
            generic = True
        else:
            generic = binding.status == "unreviewed"
        tables: dict[str, CatalogTable] = {}
        descriptors: dict[str, SourceDescriptor] = {}
        missing: list[str] = []
        for ref in _table_refs(binding):
            try:
                t = self.table(ref)
            except CatalogError:
                missing.append(ref)                    # lint reports unresolvable references
                continue
            tables[str(t.ref)] = t
            descriptors[t.descriptor.source] = t.descriptor
            descriptors.setdefault(t.ref.source, self._sources.get(t.ref.source, t.descriptor))
        quarantined = self._descriptor_quarantine(binding, missing, tables)
        if generic:
            quarantined += tuple(q for q in self._generic_quarantine(server) if q not in quarantined)
        return ToolContract(server, tool, binding, tables, descriptors, generic=generic, overlay=owner,
                            alias_of=alias_of, generic_spec=generic_spec, quarantined=quarantined,
                            missing=tuple(missing))

    # -- quarantine (R8) -----------------------------------------------------

    def _same_as_quarantine(self, server: str, tool: str) -> tuple[Quarantined, ...]:
        """Quarantined overlays of other servers that bind ``server.tool`` with ``same_as``; one whose ``same_as``
        cannot be read may bind any tool of a reviewed server that has no binding of its own."""
        name = f"{server}.{tool}"
        return tuple(q for q in self.quarantined if q.kind == "overlay" and q.name != server
                     and (name in (q.same_as or ()) or (q.same_as is None and server in self._overlays)))

    def _generic_quarantine(self, server: str) -> tuple[Quarantined, ...]:
        """Quarantined generic overlays that would apply to ``server`` (``*``, its name, or unknown)."""
        return tuple(q for q in self.quarantined if q.kind == "generic" and q.name in (None, "*", server))

    def _unloaded(self, source: str) -> set[str]:
        """The sources a reference to ``source`` needs that are not loaded (itself, or its alias target)."""
        target = self.aliases.get(source)
        if target:
            return set() if target in self._sources else {target}
        return set() if source in self._sources else {source}

    def _descriptor_quarantine(self, binding: ToolBinding, missing: Sequence[str],
                               tables: Mapping[str, CatalogTable]) -> tuple[Quarantined, ...]:
        """Quarantined descriptors a binding depends on: one that declares (or, unreadable, may declare) a
        source its unresolved table references or qualified id_types need, or a bare id_type no loaded
        source declares."""
        quarantined = [q for q in self.quarantined if q.kind == "descriptor"]
        if not quarantined:
            return ()
        sources: set[str] = set()
        bare: set[str] = set()
        for ref in missing:
            sources |= self._unloaded(ref.split(".", 1)[0])
        kinds: list[tuple[str, str | None]] = []
        for a in binding.args.values():
            kinds.extend((k, None) for k in a.accepts)
            for c in a.bound_columns:
                segs = parse_path(c).segments
                if len(segs) < 3:
                    continue
                t = tables.get(f"{segs[0].name}.{segs[1].name}")
                spec = t.columns.get(segs[2].name) if t is not None else None
                if t is not None and getattr(spec, "id_type", None):
                    kinds.append((str(spec.id_type), t.descriptor.source))
        for kind, hint in kinds:
            if ":" in kind:
                sources |= self._unloaded(kind.partition(":")[0])
                continue
            try:
                self.id_type(kind, hint)
            except UnknownIdType:
                bare.add(kind)
            except CatalogError:
                continue
        out = []
        for q in quarantined:
            if q.name is None:
                hit = bool(sources or bare)
            else:
                hit = q.name in sources or bool(bare and (q.id_types is None or bare & set(q.id_types)))
            if hit:
                out.append(q)
        return tuple(out)

    def quarantined_servers(self) -> dict[str, list[Quarantined]]:
        """``{server: [its quarantined overlay files]}``: servers no binding can be read for."""
        out: dict[str, list[Quarantined]] = {}
        for q in self.quarantined:
            if q.kind == "overlay" and q.name:
                out.setdefault(q.name, []).append(q)
        return out

    def quarantined_tools(self, servers: Sequence[str] | None = None) -> dict[str, tuple[Quarantined, ...]]:
        """``{"server.tool": files}`` of the loaded overlays' tools that depend on a quarantined file."""
        if not self.quarantined:
            return {}
        wanted = None if servers is None else set(servers)
        out: dict[str, tuple[Quarantined, ...]] = {}
        named = [(server, tool) for server in self.servers() for tool in self.tools(server)]
        # tools whose binding sits in a quarantined overlay of another server (same_as)
        named += [tuple(n.split(".", 1)) for q in self.quarantined if q.kind == "overlay" for n in q.same_as or ()
                  if "." in n]
        for server, tool in dict.fromkeys(named):
            if wanted is not None and server not in wanted:
                continue
            try:
                c = self.contract(server, tool)
            except Exception:  # noqa: BLE001 - lint reports a broken binding
                continue
            if c.quarantined:
                out[f"{server}.{tool}"] = c.quarantined
        return out

    def _generic_spec(self, server: str) -> GenericSpec | None:
        specs = [g.generic for g in self._generic if g.generic is not None and g.server in (server, "*")]
        if not specs:
            return None
        merged = GenericSpec()
        for s in specs:
            merged = GenericSpec(not_found_when=merged.not_found_when + s.not_found_when,
                                 empty_when=merged.empty_when + s.empty_when,
                                 param_kinds={**s.param_kinds, **merged.param_kinds},
                                 notes=merged.notes + s.notes)
        return merged

    # -- digests and lint --------------------------------------------------

    def descriptor_digest(self, source: str) -> str:
        return _digest(self.source(source))

    def overlay_digest(self, server: str) -> str | None:
        ov = self._overlays.get(server)
        return _digest(ov) if ov is not None else None

    def digest(self) -> str:
        """``sha256:`` over every descriptor, overlay and generic overlay digest and the aliases (and the
        quarantined files, when there are any)."""
        body: dict[str, Any] = {
            "descriptors": {s: _digest(d) for s, d in sorted(self._sources.items())},
            "overlays": {s: _digest(o) for s, o in sorted(self._overlays.items())},
            "generic": sorted(_digest(g) for g in self._generic),
            "aliases": dict(sorted(self.aliases.items())),
        }
        if self.quarantined:
            body["quarantined"] = [[q.file, q.kind, q.name, q.error] for q in self.quarantined]
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def plugin_names(self) -> dict[str, set[str]]:
        """Plugin names the descriptors use, by kind (preflight probes their ``requires``)."""
        out: dict[str, set[str]] = {"format": set(), "layout": set(), "statistic": set(), "identifier": set()}
        for d in self._sources.values():
            for it in d.id_types.values():
                out["identifier"].add(it.plugin)
            for name, t in d.tables.items():
                if t.items_of is not None:
                    continue
                for kind in ("format", "layout"):
                    p = getattr(d, f"table_{kind}")(name)
                    if p and p != "none":
                        out[kind].add(p)
        for d in self._sources.values():
            for kind in ("format", "layout"):
                p = plugin_name(getattr(d.defaults, kind))
                if p and p != "none":
                    out[kind].add(p)
        return out

    def lint(self, registry: Any = None, strict: bool | None = None) -> list[Any]:
        """Lint every descriptor and overlay (``registry`` defaults to the catalog's). A quarantined file is
        an error naming the file and why it does not load; an overlay's unresolved reference to what that
        file declares is a warning pointing at it (the overlay is not what is broken)."""
        from .descriptor.lint import Finding, lint_descriptor, lint_overlay
        reg = registry if registry is not None else self.registry
        findings: list[Any] = [Finding("error", q.path, f"does not load, quarantined: {q.summary}", rule="quarantined")
                               for q in self.quarantined]
        for d in self._sources.values():
            findings.extend(lint_descriptor(d, reg, strict, self._sources))
        for ov in list(self._overlays.values()) + self._generic:
            for f in lint_overlay(ov, self, reg):
                q = self._consequence_of(f)
                findings.append(f if q is None else dataclasses.replace(
                    f, level="warning", message=f"{f.message} (it depends on the quarantined {q.file})"))
        return findings

    def _consequence_of(self, finding: Any) -> Quarantined | None:
        """The quarantined descriptor an overlay error follows from: an unknown source it declares (or an
        unreadable one may declare), or an unknown accepted kind it declares."""
        target = finding.target                        # a lint Finding: the unresolved name, if any
        if finding.level != "error" or not target:
            return None
        source = str(target).split(".", 1)[0].split(":", 1)[0]
        for q in self.quarantined:
            if q.kind != "descriptor":
                continue
            if finding.rule == "accepts":
                if q.id_types is None or str(target).rsplit(":", 1)[-1] in q.id_types:
                    return q
            elif source not in self._sources and q.name in (None, source):
                return q
        return None


def build_catalog(settings: Any, registry: Any = None, *, variables: Mapping[str, str] | None = None,
                  run: Mapping[str, Any] | None = None, quarantine: bool = True) -> Catalog:
    """Load descriptors and overlays from ``settings.descriptors_dir`` / ``overlays_dir``, with
    ``data.sources.alias`` for ``implements`` tables. A file that does not load is quarantined on its
    own (``Catalog.quarantined``, logged once per process); ``quarantine=False`` raises on it instead."""
    variables = dict(variables or {})
    variables.setdefault("project_root", str(getattr(settings, "project_root", "")))
    quarantined: list[Quarantined] | None = [] if quarantine else None
    descriptors = load_descriptors(Path(settings.descriptors_dir), variables, run, quarantine=quarantined)
    overlays, generic = load_overlays(Path(settings.overlays_dir), variables, quarantine=quarantined)
    aliases = dict(getattr(getattr(settings, "sources", None), "alias", {}) or {})
    catalog = Catalog(descriptors, overlays, generic, aliases=aliases, registry=registry,
                      quarantined=quarantined or ())
    for q in catalog.quarantined:
        if (q.path, q.error) not in _WARNED:
            _WARNED.add((q.path, q.error))
            log.warning("data catalog: %s %s quarantined (only the tools that depend on it are refused): %s",
                        q.kind, q.path, q.summary)
    return catalog


def load_catalog(config: Mapping[str, Any] | None = None, registry: Any = None, *,
                 run: Mapping[str, Any] | None = None, quarantine: bool = True) -> Catalog:
    """:func:`build_catalog` from a loaded config (``vbt.config.load_config``)."""
    from .settings import DataSettings
    settings = DataSettings.from_config(config or {})
    return build_catalog(settings, registry, variables=variables_from_config(config), run=run,
                         quarantine=quarantine)

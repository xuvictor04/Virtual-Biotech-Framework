"""Reference scoping for facets that name another column (§6.4). No pyarrow.

A bare name resolves to a sibling field in the same struct or item first, then to the
fields of each enclosing item outward, then to a table-level column. ``^.name`` skips the
innermost level (repeatable), ``/name`` forces table level. Dotted names that are not found
in scope are read as ``table.col`` (same source) and ``source.table.col`` (another loaded
source, or this one qualified). A per-element column (``path: "[]"``, ``list_delimiter``)
adds an element level with no fields, so ``^.source`` on ``crossReferences.ids`` reaches the
``crossReferences`` item that holds ``ids``.

Levels are addressed by a *container path*: the container names from the table root to the
current level (``("crossReferences",)``), optionally ending in ``"[]"`` for an element level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..roles import LIST, Path, PathError, parse_path
from .columns import MemberCol, NestedCol, is_container

__all__ = [
    "ScopeError", "UnknownReference", "AmbiguousReference", "Resolved", "TableScope", "resolve_reference",
    "container_names",
]

Mode = Literal["column", "table_column", "any"]


class ScopeError(ValueError):
    """A reference that cannot be resolved under the §6.4 scoping rules."""


class UnknownReference(ScopeError):
    pass


class AmbiguousReference(ScopeError):
    pass


@dataclass(frozen=True)
class Resolved:
    """Where a reference points."""

    kind: Literal["column", "element", "partition", "virtual", "axis", "table_column", "foreign"]
    path: str                                          # column path within the target table ("" for an element)
    table: str | None = None                           # target table (None: the scope's own table)
    source: str | None = None                          # target source (None: the scope's own source)
    level: int = 0                                     # scope level where it was found (0 = table level)
    column: Any = None                                 # the ColumnSpec when one is declared
    loaded: bool = True                                # False: a source.table.col ref to a source not loaded


def container_names(path: str | Sequence[str] | None) -> tuple[str, ...]:
    """``"a[].b[]"`` or ``["a", "b"]`` -> ``("a", "b")``; a trailing element marker is kept as ``"[]"``."""
    if path is None or path == "" or path == ():
        return ()
    if isinstance(path, str):
        p = parse_path(path)
        return tuple(s.name for s in p.segments if s.name)
    return tuple(path)


def _fields(col: Any) -> dict[str, Any]:
    if isinstance(col, (NestedCol, MemberCol)):
        return dict(col.fields)
    return {}


def _walk(fields: Mapping[str, Any], path: Path) -> Any:
    """Resolve ``path`` (no prefixes) inside ``fields``; returns the column or raises KeyError."""
    current: Mapping[str, Any] = fields
    col: Any = None
    segs = [s for s in path.segments if s.name]
    if not segs:
        raise KeyError("")
    for i, seg in enumerate(segs):
        if seg.name not in current:
            raise KeyError(seg.name)
        col = current[seg.name]
        if i < len(segs) - 1:
            if not is_container(col):
                parse = getattr(col, "parse", None)
                sub = getattr(parse, "fields", None)
                if sub is None:
                    raise KeyError(segs[i + 1].name)
                current = dict(sub)
            else:
                current = _fields(col)
    return col


class TableScope:
    """Scope of one table of one source (an item table scopes inside its parent's container)."""

    def __init__(self, descriptor: Any, table: str, loaded_sources: Mapping[str, Any] | None = None) -> None:
        self.descriptor = descriptor
        self.source: str = getattr(descriptor, "source", "")
        self.table = table
        self.loaded = dict(loaded_sources or {})
        tables = getattr(descriptor, "tables", {}) or {}
        if table not in tables:
            raise UnknownReference(f"unknown table {table!r} in source {self.source!r}")
        self.spec = tables[table]
        self.root_table, self.base_path = self._root(table, set())
        self.root_spec = tables[self.root_table]

    # -- structure ---------------------------------------------------------

    def _root(self, table: str, seen: set[str]) -> tuple[str, tuple[str, ...]]:
        if table in seen:
            raise ScopeError(f"items_of cycle through {table!r}")
        seen.add(table)
        spec = self.descriptor.tables[table]
        if spec.items_of is None:
            return table, ()
        parent = spec.items_of.table
        if parent not in self.descriptor.tables:
            raise UnknownReference(f"items_of names unknown table {parent!r}")
        root, base = self._root(parent, seen)
        return root, base + container_names(spec.items_of.path)

    def top_fields(self) -> dict[str, Any]:
        """Table-level names: columns, partition columns, the fragment-key column and matrix axis fields."""
        spec = self.root_spec
        out: dict[str, Any] = dict(spec.columns)
        for name, part in spec.partitions.items():
            out.setdefault(name, part.column)
        if spec.fragment_key is not None:
            out.setdefault(spec.fragment_key.name, None)
        return out

    def axis_fields(self, axis: str) -> dict[str, Any]:
        m = getattr(self.root_spec, "matrix", None)
        if m is None or axis not in m.axes:
            return {}
        ax = m.axes[axis]
        out: dict[str, Any] = dict(ax.columns)
        if ax.parse is not None:
            for k, v in ax.parse.fields.items():
                out.setdefault(k, v)
        for k in ax.key.columns:
            out.setdefault(k, None)
        if ax.index_name:
            out.setdefault(ax.index_name, None)
        return out

    def levels(self, container_path: str | Sequence[str] | None = ()) -> list[dict[str, Any]]:
        """Field dicts from table level (0) to the innermost level of ``container_path`` (relative to
        this table: an item table's own container levels come first)."""
        names = self.base_path + container_names(container_path)
        levels: list[dict[str, Any]] = [self.top_fields()]
        current = levels[0]
        for name in names:
            if name == LIST:
                levels.append({})                      # element level: no fields
                continue
            if name not in current:
                raise UnknownReference(f"container {name!r} not found in table {self.root_table!r}")
            col = current[name]
            current = _fields(col)
            if not current and getattr(col, "parse", None) is not None:
                current = dict(col.parse.fields)
            levels.append(current)
        return levels

    def container(self) -> Any:
        """The container column of an item table (None for a plain table)."""
        if not self.base_path:
            return None
        current = self.top_fields()
        col = None
        for name in self.base_path:
            col = current.get(name)
            current = _fields(col)
        return col

    # -- resolution --------------------------------------------------------

    def resolve(self, name: str, container_path: str | Sequence[str] | None = (), *,
                mode: Mode = "column") -> Resolved:
        """Resolve ``name`` from the level given by ``container_path``.

        ``mode="column"``: scoped columns, then ``table.col``/``source.table.col``;
        ``"table_column"``: only ``table.col``/``source.table.col`` (``ref``, ``resolve_via`` ...);
        ``"any"``: both, and a name that resolves both ways is ambiguous."""
        try:
            path = parse_path(name)
        except PathError as exc:
            raise UnknownReference(f"{name!r} is not a valid path: {exc}") from None
        found: Resolved | None = None
        if mode in ("column", "any"):
            found = self._scoped(path, container_path)
        if found is not None and mode == "column":
            return found
        foreign: Resolved | None = None
        if not (path.absolute or path.up or path.axis) and len(path.segments) >= 2:
            foreign = self._qualified(path)
        if found is not None and foreign is not None:
            raise AmbiguousReference(f"{name!r} is both a column path of {self.table!r} and a table reference")
        if found is not None:
            return found
        if foreign is not None:
            return foreign
        where = f"{self.table}" + (f" ({'.'.join(container_names(container_path))})" if container_path else "")
        raise UnknownReference(f"{name!r} does not resolve from {where}")

    def _scoped(self, path: Path, container_path: str | Sequence[str] | None) -> Resolved | None:
        if path.axis is not None:
            fields = self.axis_fields(path.axis)
            try:
                col = _walk(fields, path)
            except KeyError:
                return None
            return Resolved("axis", path.text, level=0, column=col)
        levels = self.levels(container_path)
        if path.absolute:
            candidates = [0]
        else:
            start = len(levels) - 1 - path.up
            if start < 0:
                raise UnknownReference(f"{path.text!r} goes above the table level")
            candidates = list(range(start, -1, -1))
        if path.is_item_relative and not any(s.name for s in path.segments):
            return Resolved("element", "", level=len(levels) - 1)   # "[]": the item/element itself
        names = self.base_path + container_names(container_path)
        for lvl in candidates:
            try:
                col = _walk(levels[lvl], path)
            except KeyError:
                continue
            kind: Any = "column"
            if lvl == 0 and path.head not in self.root_spec.columns:
                kind = "partition" if path.head in self.root_spec.partitions else "virtual"
            # The structural path from the root table (no brackets, no scope prefixes).
            prefix = [n for n in names[:lvl] if n != LIST]
            full = ".".join(prefix + [s.name for s in path.segments if s.name])
            return Resolved(kind, full, level=lvl, column=col)
        if self.root_spec.matrix is not None and not path.absolute and not path.up:
            hits = []
            for axis in ("row", "col"):
                try:
                    hits.append((axis, _walk(self.axis_fields(axis), path)))
                except KeyError:
                    pass
            for vname, vcol in self.root_spec.matrix.values.items():
                if vname == path.head and len(path.segments) == 1:
                    hits.append(("value", vcol))
            if len(hits) > 1:
                raise AmbiguousReference(f"{path.text!r} names fields of several matrix axes/values; "
                                         f"prefix it with @row./@col.")
            if hits:
                axis, col = hits[0]
                return Resolved("axis", (f"@{axis}." if axis != "value" else "") + path.text, column=col)
        return None

    def _qualified(self, path: Path) -> Resolved | None:
        names = [s.name for s in path.segments]
        tables = self.descriptor.tables
        # table.col... within this source
        hit: Resolved | None = None
        if names[0] in tables:
            col = self._column_of(self.descriptor, names[0], Path(path.segments[1:]))
            if col is not _MISSING:
                hit = Resolved("table_column", Path(path.segments[1:]).strip_brackets().text, table=names[0],
                               source=self.source, column=col)
        # source.table.col...
        if len(names) >= 3:
            src = self.descriptor if names[0] == self.source else self.loaded.get(names[0])
            if src is not None:
                if names[1] in src.tables:
                    col = self._column_of(src, names[1], Path(path.segments[2:]))
                    if col is not _MISSING:
                        other = Resolved("foreign" if src is not self.descriptor else "table_column",
                                         Path(path.segments[2:]).strip_brackets().text, table=names[1],
                                         source=names[0], column=col)
                        if hit is not None:
                            raise AmbiguousReference(f"{path.text!r} is both table.col and source.table.col")
                        return other
            elif hit is None and names[0] != self.source:
                # A source that is not loaded: cannot be checked (lint reports a warning).
                return Resolved("foreign", Path(path.segments[2:]).text, table=names[1], source=names[0],
                                loaded=False)
        return hit

    @staticmethod
    def _column_of(descriptor: Any, table: str, path: Path) -> Any:
        try:
            scope = TableScope(descriptor, table)
        except ScopeError:
            return _MISSING
        if not path.segments:
            return _MISSING
        fields = scope.levels()[-1]
        try:
            return _walk(fields, path)
        except KeyError:
            if scope.root_spec.matrix is not None:
                for axis in ("row", "col"):
                    try:
                        return _walk(scope.axis_fields(axis), path)
                    except KeyError:
                        pass
            return _MISSING


_MISSING = object()


def resolve_reference(table: TableScope | Any, container_path: str | Sequence[str] | None, name: str, *,
                      descriptor: Any = None, table_name: str | None = None,
                      loaded_sources: Mapping[str, Any] | None = None, mode: Mode = "column") -> Resolved:
    """Resolve ``name`` from ``container_path`` of ``table`` under the §6.4 rules.

    ``table`` is a :class:`TableScope`, or a ``TableSpec`` together with ``descriptor`` and
    ``table_name`` (a lone ``TableSpec`` is wrapped in a one-table descriptor, so only scoped
    names resolve). Raises :class:`UnknownReference` or :class:`AmbiguousReference`."""
    if isinstance(table, TableScope):
        scope = table
    else:
        if descriptor is None:
            descriptor = _OneTable(table_name or "_table", table)
            table_name = table_name or "_table"
        if table_name is None:
            matches = [k for k, v in descriptor.tables.items() if v is table]
            if len(matches) != 1:
                raise ScopeError("pass table_name with a descriptor")
            table_name = matches[0]
        scope = TableScope(descriptor, table_name, loaded_sources)
    return scope.resolve(name, container_path, mode=mode)


class _OneTable:
    def __init__(self, name: str, spec: Any) -> None:
        self.source = ""
        self.tables = {name: spec}

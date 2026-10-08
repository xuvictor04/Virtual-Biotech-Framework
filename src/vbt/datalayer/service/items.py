"""Item tables (§6.2, rev 2): rows that are the items of a nested container, at any depth.

An item table over ``a[].b[].c[]`` has one row per item of the innermost list. The reader keeps
each item in its parent row as a **singleton view**: a copy of the row in which every list on the
container path holds only the item being considered (and its ancestors). Predicates with paths
relative to the physical table (``go[].id``, ``tissues[].rna.value``, ``/id``) then evaluate with
the ordinary three-valued semantics, and every condition applies to the same item, which is
what makes ``Any(liver AND value >= 10)`` return only an item that satisfies both (C21).

* :func:`levels` splits a container path into list levels (struct steps kept inside a level);
* :func:`explode` yields ``(view, positions)`` per item, skipping null containers, empty lists and
  null items while counting them (:class:`ContainerCounts`: null vs empty vs null items are three
  different facts and readiness reports them separately);
* :func:`key_values` reads the composed key (parent key + intermediate item keys + item key;
  ``identity: value`` keys the item itself and ``identity: position`` its index, ``<path>[]#``);
* :func:`item_row` is the item as an output row with its ancestors' key parts injected;
* :func:`item_key_duplicates` checks that item keys are unique within each parent row;
* :func:`arrow_items` reads the items of an Arrow table (one row group) without building a Python row: each
  list level is flattened with the index of its parent, so a key check counts tens of millions of items
  (25.09 l2g_prediction features, target_essentiality screens) in Arrow and numpy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from ..catalog import POSITION_MARK
from ..predicate import is_null
from ..roles import ItemCond, parse_path
from ..rowkey import canonical

__all__ = [
    "Level", "ContainerCounts", "levels", "explode", "view_at", "innermost", "key_values", "item_row", "path_values",
    "path_value", "item_key_duplicates", "container_counts", "qualify_item_path", "ArrowItems", "ArrowUnsupported",
    "arrow_items", "part_level",
]


@dataclass(frozen=True)
class Level:
    """One list on a container path: the struct steps from the enclosing item (or row) to the list."""

    names: tuple[str, ...]
    text: str                                          # the container path up to and including this list ("a[].b[]")


@dataclass
class ContainerCounts:
    rows: int = 0                                      # parent rows (or enclosing items) seen
    null: int = 0                                      # null containers (not assessed)
    empty: int = 0                                     # empty lists (assessed, none)
    nonempty: int = 0
    items: int = 0
    null_items: int = 0                                # list items that are themselves null
    by_level: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, other: "ContainerCounts") -> None:
        for name in ("rows", "null", "empty", "nonempty", "items", "null_items"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for lvl, counts in other.by_level.items():
            mine = self.by_level.setdefault(lvl, {})
            for k, v in counts.items():
                mine[k] = mine.get(k, 0) + v

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "null": self.null, "empty": self.empty, "nonempty": self.nonempty,
                "items": self.items, "null_items": self.null_items, "by_level": self.by_level}


def levels(items_path: str) -> list[Level]:
    """``hallmarks.attributes[]`` -> one level (``hallmarks``, ``attributes``);
    ``a[].b[].c[]`` -> three levels. The path must end at a list."""
    out: list[Level] = []
    names: list[str] = []
    text = ""
    for seg in parse_path(items_path).segments:
        if not seg.name:
            continue
        names.append(seg.name)
        text = f"{text}.{seg.name}" if text else seg.name
        if seg.is_list:
            text += "[]" * len(seg.brackets)
            out.append(Level(tuple(names), text))
            names = []
    if names:
        raise ValueError(f"item path {items_path!r} must end at a list container")
    return out


def _get(obj: Any, names: Sequence[str]) -> Any:
    cur = obj
    for n in names:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(n)
    return cur


def _with(obj: Mapping[str, Any], names: Sequence[str], value: Any) -> dict[str, Any]:
    """A shallow copy of ``obj`` with ``names`` replaced by ``value`` (struct steps copied on the way)."""
    out = dict(obj)
    if len(names) == 1:
        out[names[0]] = value
        return out
    inner = out.get(names[0])
    out[names[0]] = _with(inner if isinstance(inner, Mapping) else {}, names[1:], value)
    return out


def explode(row: Mapping[str, Any], lvls: Sequence[Level], counts: ContainerCounts | None = None
            ) -> Iterator[tuple[dict[str, Any], tuple[int, ...]]]:
    """Singleton views of every item under ``lvls`` with their positions (one index per level)."""
    yield from _explode(row, row, list(lvls), 0, (), counts)


def _explode(root: Mapping[str, Any], node: Mapping[str, Any], lvls: list[Level], depth: int,
             positions: tuple[int, ...], counts: ContainerCounts | None
             ) -> Iterator[tuple[dict[str, Any], tuple[int, ...]]]:
    lvl = lvls[depth]
    value = _get(node, lvl.names)
    tally = counts.by_level.setdefault(lvl.text, {}) if counts is not None else None
    if counts is not None and depth == 0:
        counts.rows += 1
    if tally is not None:
        tally["rows"] = tally.get("rows", 0) + 1
    if value is None:
        if counts is not None and depth == len(lvls) - 1:
            counts.null += 1
        if tally is not None:
            tally["null"] = tally.get("null", 0) + 1
        return
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    if not items:
        if counts is not None and depth == len(lvls) - 1:
            counts.empty += 1
        if tally is not None:
            tally["empty"] = tally.get("empty", 0) + 1
        return
    if counts is not None and depth == len(lvls) - 1:
        counts.nonempty += 1
    for i, item in enumerate(items):
        if item is None:
            if counts is not None and depth == len(lvls) - 1:
                counts.null_items += 1
            if tally is not None:
                tally["null_items"] = tally.get("null_items", 0) + 1
            continue
        pos = positions + (i,)
        if depth == len(lvls) - 1:
            if counts is not None:
                counts.items += 1
            yield _rebuild(root, lvls, pos, item), pos
        elif isinstance(item, Mapping):
            yield from _explode(root, item, lvls, depth + 1, pos, counts)


def _rebuild(root: Mapping[str, Any], lvls: Sequence[Level], pos: tuple[int, ...], leaf_item: Any) -> dict[str, Any]:
    """The singleton view of the item at ``pos``."""
    chain = [root]
    node: Any = root
    for d, lvl in enumerate(lvls[:-1]):
        node = (_get(node, lvl.names) or [])[pos[d]]
        chain.append(node)
    view: Any = leaf_item
    for d in range(len(lvls) - 1, -1, -1):
        view = _with(chain[d], lvls[d].names, [view])
    return view


def view_at(row: Mapping[str, Any], lvls: Sequence[Level], positions: Sequence[int]) -> dict[str, Any] | None:
    """The singleton view of the item at ``positions`` (None when the row no longer has it)."""
    node: Any = row
    for d, lvl in enumerate(lvls):
        lst = _get(node, lvl.names)
        if not isinstance(lst, (list, tuple)) or positions[d] >= len(lst):
            return None
        node = lst[positions[d]]
    if node is None:
        return None
    return _rebuild(row, lvls, tuple(positions), node)


def innermost(view: Mapping[str, Any], lvls: Sequence[Level]) -> Any:
    node: Any = view
    for lvl in lvls:
        lst = _get(node, lvl.names)
        if not lst:
            return None
        node = lst[0]
    return node


def path_values(obj: Any, path: str) -> list[Any]:
    """Every leaf value at ``path`` (lists crossed existentially; item conditions applied)."""
    parsed = parse_path(path.lstrip("/"))
    cur: list[Any] = [obj]
    for seg in parsed.segments:
        if not seg.name:
            nxt = cur
        else:
            nxt = [c.get(seg.name) if isinstance(c, Mapping) else None for c in cur]
        for b in seg.brackets:
            flat: list[Any] = []
            for v in nxt:
                if isinstance(v, (list, tuple)):
                    items = list(v)
                    if isinstance(b, ItemCond):
                        items = [x for x in items if b.matches(x)]
                    flat.extend(items)
            nxt = flat
        cur = nxt
    return cur


def path_value(obj: Any, path: str) -> Any:
    """The single value at ``path`` (a singleton list collapses; several values are returned as a list)."""
    if path.endswith(POSITION_MARK):
        raise ValueError("positional key parts are read with key_values()")
    parsed = parse_path(path.lstrip("/"))
    if not parsed.crosses_list:
        cur: Any = obj
        for seg in parsed.segments:
            cur = cur.get(seg.name) if isinstance(cur, Mapping) else None
        return cur
    vals = path_values(obj, path)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def key_values(view: Mapping[str, Any], key: Sequence[str], positions: Sequence[int] = (),
               lvls: Sequence[Level] = ()) -> tuple[Any, ...]:
    """The composed key of a row or singleton item view."""
    out: list[Any] = []
    for part in key:
        if part.endswith(POSITION_MARK):
            container = part[: -len(POSITION_MARK)]
            idx = next((i for i, lvl in enumerate(lvls) if lvl.text == container), None)
            out.append(positions[idx] if idx is not None and idx < len(positions) else None)
            continue
        v = path_value(view, part)
        out.append(None if is_null(v) else v)
    return tuple(out)


def item_row(view: Mapping[str, Any], lvls: Sequence[Level], key: Sequence[str],
             positions: Sequence[int] = ()) -> dict[str, Any]:
    """The item as an output row: its fields, plus the key parts of its ancestors (named by their last
    field, or by their full path when an item field has that name; positional parts as ``<container>#``)."""
    item = innermost(view, lvls)
    row: dict[str, Any] = dict(item) if isinstance(item, Mapping) else {"value": item}
    inner = lvls[-1].text if lvls else ""
    for part, value in zip(key, key_values(view, key, positions, lvls)):
        if part.endswith(POSITION_MARK):
            name = part
        elif inner and part.startswith(inner + ".") and "[]" not in part[len(inner) + 1:]:
            continue                                   # a field of the item itself
        elif part == inner:
            continue                                   # identity: value (the item is the row's value)
        else:
            name = parse_path(part).segments[-1].name
            if name in row:
                name = "/" + part if "[]" not in part else part
        row[name] = value
    return row


def qualify_item_path(name: str, lvls: Sequence[Level], fields_by_level: Sequence[Mapping[str, Any]],
                      top_columns: Sequence[str]) -> str:
    """A request path on an item table as a path from the physical table (§6.4 scoping).

    ``/x`` is a table-level column, ``^.x`` a field one level up, a path that already starts with
    the container path is kept, and a bare name resolves to a field of the item first, then of each
    enclosing item outward, then to a table-level column."""
    if name.startswith("/"):
        return name[1:]
    if lvls and (name == lvls[-1].text or name.startswith(lvls[-1].text + ".") or
                 any(name.startswith(lv.text + ".") for lv in lvls)):
        return name
    up = 0
    rest = name
    while rest.startswith("^."):
        up += 1
        rest = rest[2:]
    head = parse_path(rest).head if rest else ""
    depth = len(lvls) - 1 - up
    if up:
        if depth < 0:
            return rest
        return f"{lvls[depth].text}.{rest}"
    for d in range(len(lvls) - 1, -1, -1):
        if head in fields_by_level[d]:
            return f"{lvls[d].text}.{rest}"
    if head in top_columns:
        return rest
    return f"{lvls[-1].text}.{rest}" if lvls else rest


def container_counts(rows: Sequence[Mapping[str, Any]], items_path: str) -> ContainerCounts:
    counts = ContainerCounts()
    lvls = levels(items_path)
    for row in rows:
        for _ in explode(row, lvls, counts):
            pass
    return counts


def item_key_duplicates(row: Mapping[str, Any], container: str, item_key: Sequence[str],
                        identity: str = "key") -> list[str]:
    """Canonical item keys that occur more than once under one parent (``container`` is the list path
    relative to the row, item key columns relative to the item). The parent of a list nested in a list
    (``indications[].references[]``) is its enclosing item: one source cited under two indications of a
    drug is not a repeat."""
    path = container if container.endswith("]") else container + "[]"
    lvls = levels(path)
    if len(lvls) > 1:
        inner = path[len(lvls[-2].text) + 1:]
        out: set[str] = set()
        for view, _ in explode(row, lvls[:-1]):
            parent = innermost(view, lvls[:-1])
            if isinstance(parent, Mapping):
                out.update(_repeated(path_values(parent, inner), item_key, identity))
        return sorted(out)
    return _repeated(path_values(row, path), item_key, identity)


def _repeated(items: Sequence[Any], item_key: Sequence[str], identity: str) -> list[str]:
    seen: dict[str, int] = {}
    for item in items:
        if item is None:
            continue
        if identity == "value":
            k = canonical([item])
        else:
            k = canonical([path_value(item, c) if isinstance(item, Mapping) else None for c in item_key])
        seen[k] = seen.get(k, 0) + 1
    return sorted(k for k, n in seen.items() if n > 1)


def deep_copy(row: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(row))


# ---------------------------------------------------------------------------- items as Arrow arrays


class ArrowUnsupported(ValueError):
    """The Arrow item path cannot read this container or key part; the caller reads rows instead."""


@dataclass
class ArrowItems:
    """The non-null innermost items of an Arrow table under a container path (:func:`arrow_items`).

    ``rows`` is each item's row in the table and ``parents`` its parent: the enclosing item (its index among the
    non-null items of the level above) for a list in a list, the row for a list in the row. ``parts`` holds one
    Arrow array per requested key part, one value per item (positions as int64)."""

    rows: Any
    parents: Any
    parts: dict[str, Any]

    def __len__(self) -> int:
        return len(self.rows)


def part_level(part: str, lvls: Sequence[Level]) -> tuple[int, str]:
    """``(level, rest)`` of a key part on the container path: the innermost level whose container is a prefix of
    ``part``, and the struct path inside that level's item (``""``: the item itself, ``#``: its position)."""
    for d in range(len(lvls) - 1, -1, -1):
        text = lvls[d].text
        if part == text:
            return d, ""
        if part == text + POSITION_MARK:
            return d, POSITION_MARK
        if part.startswith(text + "."):
            rest = part[len(text) + 1:]
            if "[" in rest:
                raise ArrowUnsupported(f"key part {part!r} crosses a list inside its item")
            return d, rest
    raise ArrowUnsupported(f"key part {part!r} is not a field of an item on {lvls[-1].text if lvls else '?'}")


def _child(arr: Any, name: str) -> Any:
    """A struct field with the struct's own nulls applied (``StructArray.field`` would keep the child's values)."""
    import pyarrow as pa

    if not pa.types.is_struct(arr.type):
        raise ArrowUnsupported(f"{name!r} is not a field of a struct ({arr.type})")
    index = arr.type.get_field_index(name)
    if index < 0:
        raise ArrowUnsupported(f"field {name!r} is not in the data")
    return arr.flatten()[index]


def _list_values(arr: Any) -> tuple[Any, Any]:
    """``(values, lengths)`` of a list array: ``flatten()`` skips null lists (length 0 here). ``list_parent_indices``
    is not used: it also counts the values a null list slot still spans, so it can disagree with ``flatten()``."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type)):
        raise ArrowUnsupported(f"not a list ({arr.type})")
    lengths = np.asarray(pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
    values = arr.flatten()
    if len(values) != int(lengths.sum()):
        raise ArrowUnsupported("list values do not match the list lengths")
    return values, lengths


def arrow_items(table: Any, lvls: Sequence[Level], parts: Sequence[str], rows: Any = None) -> ArrowItems:
    """The items under ``lvls`` of an Arrow table (one row group as read), as :func:`explode` yields them (null
    containers, empty lists and null items skipped), with the key ``parts`` that lie on the container path
    (``a[].b``, ``a[].b[].c``, ``a[]#``, ``a[]``). ``rows`` (row indices) restricts the items to those rows.
    Nothing is converted to Python. Raises :class:`ArrowUnsupported` when a step is not a struct or list."""
    import numpy as np
    import pyarrow as pa

    if not lvls:
        raise ArrowUnsupported("no container path")
    placed = {p: part_level(p, lvls) for p in parts}
    head = lvls[0].names[0]
    if head not in table.column_names:
        raise ArrowUnsupported(f"column {head!r} was not read")
    arr = table.column(head)
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    row_of = np.arange(len(arr), dtype=np.int64)
    if rows is not None:
        row_of = np.asarray(rows, dtype=np.int64)
        arr = arr.take(pa.array(row_of))
    parents = row_of
    carried: dict[str, Any] = {}
    for depth, lvl in enumerate(lvls):
        for name in (lvl.names[1:] if depth == 0 else lvl.names):
            arr = _child(arr, name)
        values, lengths = _list_values(arr)
        node = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)      # the parent of each list value
        positions = np.arange(len(node), dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        if values.null_count:                                                   # null items are skipped, as rows
            keep = np.flatnonzero(values.is_valid().to_numpy(zero_copy_only=False))
            values = values.take(pa.array(keep))
            node, positions = node[keep], positions[keep]
        if carried:
            index = pa.array(node)
            carried = {p: a.take(index) for p, a in carried.items()}
        row_of = row_of[node]
        parents = node if depth else row_of
        for p, (d, rest) in placed.items():
            if d != depth:
                continue
            if rest == POSITION_MARK:
                carried[p] = pa.array(positions)
            elif not rest:
                carried[p] = values
            else:
                a = values
                for name in rest.split("."):
                    a = _child(a, name)
                carried[p] = a
        arr = values
    return ArrowItems(rows=row_of, parents=parents, parts=carried)

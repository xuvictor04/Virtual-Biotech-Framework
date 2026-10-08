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
* :func:`item_key_duplicates` checks that item keys are unique within each parent row.
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
    "path_value", "item_key_duplicates", "container_counts", "qualify_item_path",
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

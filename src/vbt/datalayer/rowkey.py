"""Canonical row keys (§6.3, rev 2). Stdlib only (no pyarrow).

One function renders a key as JSON on **both** sides of every comparison (witness key
sets, ``row_keys``, ``row_keys_sha256``, ``content_hash``, replay):

* floats as the shortest repr that round-trips in the column's **storage** type, so a
  float32 ``0.05`` renders ``0.05`` and never ``0.05000000074505806``;
* explicit ``null``; NaN is ``null`` (unknown, I6);
* list and set parts sorted by their canonical JSON (order-insensitive);
* dicts with sorted keys.

Storage types are Arrow type strings as the data child reports them (``float``,
``double``, ``halffloat``, ``int32``, ``large_string``, ``list<element: float>`` ...);
``None`` renders by the Python type of the value.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FLOAT32_TYPES", "FLOAT64_TYPES", "FLOAT16_TYPES", "render_float", "render_value", "canonical",
    "canonical_sha256", "content_hash", "element_type",
]

FLOAT32_TYPES = frozenset({"float", "float32"})
FLOAT64_TYPES = frozenset({"double", "float64"})
FLOAT16_TYPES = frozenset({"halffloat", "float16"})
_FLOAT_TYPES = FLOAT32_TYPES | FLOAT64_TYPES | FLOAT16_TYPES
_LIST_PREFIXES = ("list<", "large_list<", "fixed_size_list<", "list_view<", "large_list_view<")


def _normalise_type(storage_type: str | None) -> str | None:
    if storage_type is None:
        return None
    t = str(storage_type).strip().lower()
    if t.startswith("dictionary<"):  # dictionary<values=string, indices=int32>: the value type counts
        inner = t[len("dictionary<"):-1]
        for part in inner.split(","):
            k, _, v = part.partition("=")
            if k.strip() == "values":
                return _normalise_type(v.strip())
    return t


def element_type(storage_type: str | None) -> str | None:
    """Element type of a list storage type (``list<element: float>`` -> ``float``), else None."""
    t = _normalise_type(storage_type)
    if not t or not t.startswith(_LIST_PREFIXES):
        return None
    inner = t[t.index("<") + 1: t.rindex(">")] if ">" in t else t[t.index("<") + 1:]
    # fixed_size_list<element: float>[3] keeps the size after '>'; "item: float" / "element: float"
    name, sep, rest = inner.partition(":")
    if sep and name.strip() in {"item", "element", "elem", "value"}:
        inner = rest
    return inner.strip() or None


def _pack_roundtrip(fmt: str, value: float) -> float:
    return struct.unpack(fmt, struct.pack(fmt, value))[0]


def render_float(value: float, storage_type: str | None = "double") -> str:
    """Shortest decimal that round-trips to the same value in ``storage_type``.

    float64 (and unknown types) use ``repr``; float32/float16 search increasing
    precision for the shortest string whose parse rounds to the same stored value.
    NaN renders ``null``; infinities render as ``json.dumps`` does.
    """
    v = float(value)
    if math.isnan(v):
        return "null"
    if math.isinf(v):
        return json.dumps(v)
    t = _normalise_type(storage_type)
    fmt = "f" if t in FLOAT32_TYPES else "e" if t in FLOAT16_TYPES else None
    if fmt is None:
        return repr(v)
    try:
        stored = _pack_roundtrip(fmt, v)
    except (OverflowError, struct.error):
        return repr(v)
    for precision in range(1, 18):
        candidate = float(f"{stored:.{precision}g}")
        if _pack_roundtrip(fmt, candidate) == stored:
            # repr of the parsed candidate is its shortest float64 spelling, which still
            # parses to the same float64 and therefore to the same stored value.
            return repr(candidate)
    return repr(stored)


def _is_float_type(t: str | None) -> bool:
    return t in _FLOAT_TYPES


def _scalar(value: Any, storage_type: str | None) -> Any:
    """A JSON-ready scalar, or a pre-rendered float marked by ``_Raw``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float) or (_is_float_type(storage_type) and isinstance(value, int)):
        rendered = render_float(float(value), storage_type)
        return None if rendered == "null" else _Raw(rendered)
    if isinstance(value, int):
        return value
    item = getattr(value, "item", None)  # numpy scalars, should one slip through
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return _scalar(item(), storage_type)
        except (TypeError, ValueError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


class _Raw(str):
    """A float already rendered in its storage type (inserted verbatim)."""


def render_value(value: Any, storage_type: str | None = None) -> str:
    """Canonical JSON of one key part."""
    t = _normalise_type(storage_type)
    if isinstance(value, Mapping):
        items = sorted(((str(k), render_value(v, None)) for k, v in value.items()), key=lambda kv: kv[0])
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{v}" for k, v in items) + "}"
    if isinstance(value, (list, tuple, set, frozenset)):
        elem = element_type(t)
        parts = sorted(render_value(v, elem) for v in value)
        return "[" + ",".join(parts) + "]"
    s = _scalar(value, t)
    if isinstance(s, _Raw):
        return str(s)
    return json.dumps(s, ensure_ascii=False)


def canonical(values: Sequence[Any], storage_types: Sequence[str | None] | None = None) -> str:
    """Canonical JSON array of one row key (one entry per key part, in key order)."""
    types: Sequence[str | None] = storage_types if storage_types is not None else [None] * len(values)
    if len(types) != len(values):
        raise ValueError(f"{len(values)} key parts but {len(types)} storage types")
    return "[" + ",".join(render_value(v, t) for v, t in zip(values, types)) + "]"


def canonical_sha256(keys: Iterable[str | Sequence[Any]], storage_types: Sequence[str | None] | None = None) -> str:
    """sha256 hex of a set of row keys (order-insensitive; the order is recorded separately).

    ``keys`` may be canonical strings already or raw key tuples (rendered with ``storage_types``).
    """
    rendered = sorted(k if isinstance(k, str) else canonical(k, storage_types) for k in keys)
    h = hashlib.sha256()
    for line in rendered:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def content_hash(row: Mapping[str, Any], columns: Sequence[str] | None = None,
                 storage_types: Mapping[str, str | None] | None = None) -> str:
    """sha256 hex identifying a row by its content (``KeySpec.row_identity: content_hash``)."""
    cols = list(columns) if columns is not None else sorted(str(k) for k in row)
    types = [(storage_types or {}).get(c) for c in cols]
    return hashlib.sha256(canonical([row.get(c) for c in cols], types).encode("utf-8")).hexdigest()

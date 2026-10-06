"""Builtin format plugins (phase 1: ``parquet``) and format-agnostic helpers. No pyarrow at import.

:func:`storage_typed` rewrites a predicate's literals to the value they have in the column's
**storage** type (a float32 ``0.05`` becomes ``0.05000000074505806``), so the reader's residual
evaluation and the conformance oracle compare exactly as a storage-typed pushdown does (F-10).
Format plugins return residuals already rewritten; the reader evaluates them with
:func:`~vbt.datalayer.predicate.evaluate` on :meth:`to_native` rows.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Callable

from ...predicate import (
    All,
    And,
    Any as AnyItem,
    Cmp,
    CmpAbs,
    Contains,
    Eq,
    In,
    Not,
    Or,
    Param,
    Predicate,
    Range,
)
from ...roles import parse_path

__all__ = ["round_to_storage", "storage_typed", "conjuncts", "join_path", "top_column", "nan_to_none"]

_PACK = {"float": "f", "float32": "f", "halffloat": "e", "float16": "e"}


def round_to_storage(value: Any, storage_type: str | None) -> Any:
    """``value`` as the column stores it: floats rounded to float32/float16 for those storage types
    (a value outside the type's range is left as is); everything else unchanged."""
    fmt = _PACK.get(str(storage_type or "").strip().lower())
    if fmt is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return value
    try:
        stored = struct.unpack(fmt, struct.pack(fmt, v))[0]
    except (OverflowError, struct.error):
        return value
    return value if math.isinf(stored) else stored     # beyond the type's range: compared as double


def join_path(prefix: str, inner: str) -> str:
    """An item-relative path joined to its container path (``drugs[]`` + ``x`` -> ``drugs[].x``)."""
    if not prefix or inner.startswith("/"):
        return inner.lstrip("/")
    if inner.startswith("[]"):
        return prefix + inner[2:] if prefix.endswith("[]") else prefix + inner
    return f"{prefix}.{inner}" if inner else prefix


def top_column(path: str) -> str:
    """The table-level column a path reads (``drugs[].drugId`` -> ``drugs``; ``s.a`` -> ``s``)."""
    parsed = parse_path(path)
    return parsed.segments[0].name if parsed.segments else path


def conjuncts(p: Predicate | None) -> list[Predicate]:
    """Top-level conjuncts (nested ``And`` flattened)."""
    if p is None:
        return []
    if isinstance(p, And):
        out: list[Predicate] = []
        for q in p.preds:
            out.extend(conjuncts(q))
        return out
    return [p]


def _resolve(stack: list[str], column: str) -> str:
    if column.startswith("/"):
        return column[1:]
    level = len(stack) - 1
    while column.startswith("^."):
        column = column[2:]
        level -= 1
    return join_path(stack[max(level, 0)], column)


def storage_typed(p: Predicate | None, type_of: Callable[[str], str | None]) -> Predicate | None:
    """``p`` with every literal rounded to the storage type of the column it is compared with.
    ``type_of(path)`` returns the storage type of a full path (descending through lists)."""
    if p is None:
        return None
    return _typed(p, type_of, [""])


def _typed(p: Predicate, type_of: Callable[[str], str | None], stack: list[str]) -> Predicate:
    def cast(column: str, value: Any) -> Any:
        if isinstance(value, Param):
            return value
        return round_to_storage(value, type_of(_resolve(stack, column)))

    if isinstance(p, And):
        return And(tuple(_typed(q, type_of, stack) for q in p.preds))
    if isinstance(p, Or):
        return Or(tuple(_typed(q, type_of, stack) for q in p.preds))
    if isinstance(p, Not):
        return Not(_typed(p.pred, type_of, stack))
    if isinstance(p, Eq):
        return Eq(p.column, cast(p.column, p.value))
    if isinstance(p, Cmp):
        return Cmp(p.column, p.op, cast(p.column, p.value))
    if isinstance(p, CmpAbs):
        return CmpAbs(p.column, p.op, cast(p.column, p.value))
    if isinstance(p, In):
        return In(p.column, tuple(cast(p.column, v) for v in p.values))
    if isinstance(p, Range):
        lo = None if p.lo is None else cast(p.column, p.lo)
        hi = None if p.hi is None else cast(p.column, p.hi)
        return Range(p.column, lo, hi, p.lo_inclusive, p.hi_inclusive)
    if isinstance(p, Contains):
        path = p.path if p.path.endswith("[]") else p.path + "[]"
        return Contains(p.path, cast(path, p.value))
    if isinstance(p, (AnyItem, All)):
        inner = _typed(p.pred, type_of, stack + [_resolve(stack, p.path)])
        return type(p)(p.path, inner, p.skip_null_items)
    return p


def nan_to_none(value: Any) -> Any:
    """NaN -> None recursively through lists and dicts (I6: NaN is null)."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, list):
        return [nan_to_none(v) for v in value]
    if isinstance(value, dict):
        return {k: nan_to_none(v) for k, v in value.items()}
    return value

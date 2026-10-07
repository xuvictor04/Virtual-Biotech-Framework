"""``soma``: TileDB-SOMA dataframes (CELLxGENE Census obs/var; phase 4, F20). No pyarrow at import.

SOMA reads are filtered by a ``value_filter`` string. :meth:`SomaFormat.compile` builds that string
from the predicate IR with correct quoting (strings single-quoted with ``\\`` and ``'`` escaped, the
same literals ``gateway/soma_filter.py`` parses), which fixes the upstream quote-joining bug class
(``single_cell_mcp/tools.py:458-462``): ``Eq``, ``In``, ``Cmp``, ``Range``, ``Not(Eq)``/``Not(In)``,
``And`` and ``Or``. Anything SOMA cannot express (``IsNull``, text matching, list membership) or a
column name that is not a plain identifier is returned as the residual, which the caller applies to
the rows read; in a disjunction, one inexpressible branch makes the whole ``Or`` residual.

``scan`` reads a fragment whose ``uri`` names a SOMA dataframe (``soma://<census_version>/<path>``) through
the :mod:`~vbt.datalayer.plugins.layouts.soma` layout's opener; recorded reads in tests use the
``census`` stub instead of the network.
"""

from __future__ import annotations

import math
import re
from typing import Any, ClassVar, Iterator, Mapping, Sequence

from ...predicate import And, Cmp, Eq, In, Not, Or, Predicate, Range
from ..base import Fragment, FormatError, FragmentStats, PluginBase
from ..registry import register

__all__ = ["SomaFormat", "soma_quote", "SomaCompileError"]

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SomaCompileError(ValueError):
    """A predicate node SOMA cannot express."""


def soma_quote(value: Any) -> str:
    """A SOMA literal: strings single-quoted with ``\\`` and ``'`` escaped; ``True``/``False``; numbers."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SomaCompileError(f"non-finite number {value!r} in a filter")
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _col(name: str) -> str:
    if not _IDENT.fullmatch(str(name)):
        raise SomaCompileError(f"{name!r} is not a SOMA column name")
    return str(name)


def _expr(p: Predicate) -> str:
    if isinstance(p, Eq):
        return f"{_col(p.column)} == {soma_quote(p.value)}"
    if isinstance(p, In):
        if not p.values:
            raise SomaCompileError("an empty 'in' list")
        return f"{_col(p.column)} in [{', '.join(soma_quote(v) for v in p.values)}]"
    if isinstance(p, Cmp):
        return f"{_col(p.column)} {p.op} {soma_quote(p.value)}"
    if isinstance(p, Range):
        parts = []
        if p.lo is not None:
            parts.append(f"{_col(p.column)} {'>=' if p.lo_inclusive else '>'} {soma_quote(p.lo)}")
        if p.hi is not None:
            parts.append(f"{_col(p.column)} {'<=' if p.hi_inclusive else '<'} {soma_quote(p.hi)}")
        if not parts:
            raise SomaCompileError("an unbounded range")
        return " and ".join(parts) if len(parts) == 1 else "(" + " and ".join(parts) + ")"
    if isinstance(p, Not):
        inner = p.pred
        if isinstance(inner, Eq):
            return f"{_col(inner.column)} != {soma_quote(inner.value)}"
        if isinstance(inner, In) and inner.values:
            return f"{_col(inner.column)} not in [{', '.join(soma_quote(v) for v in inner.values)}]"
        raise SomaCompileError("only 'not' of == and 'in' is expressible")
    if isinstance(p, And):
        return " and ".join(_group(q) for q in p.preds)
    if isinstance(p, Or):
        return " or ".join(_group(q) for q in p.preds)
    raise SomaCompileError(f"cannot express {type(p).__name__} in a SOMA filter")


def _group(p: Predicate) -> str:
    text = _expr(p)
    return f"({text})" if isinstance(p, (And, Or)) else text


@register
class SomaFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "soma"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"string_compile"})

    def quote(self, value: Any) -> str:
        return soma_quote(value)

    def compile(self, predicate: Predicate | None, schema: Any = None) -> tuple[str | None, Predicate | None]:
        """``(value_filter, residual)``: the conjuncts SOMA can express as one filter string, the rest as
        the residual (None when everything compiled)."""
        if predicate is None:
            return None, None
        parts = list(predicate.preds) if isinstance(predicate, And) else [predicate]
        pushed: list[str] = []
        residual: list[Predicate] = []
        for p in parts:
            try:
                pushed.append(_group(p))
            except SomaCompileError:
                residual.append(p)
        text = " and ".join(pushed) if pushed else None
        rest = None if not residual else (residual[0] if len(residual) == 1 else And(tuple(residual)))
        return text, rest

    # ------------------------------------------------------------------ protocol

    def _frame(self, frag: Fragment) -> Any:
        from ..layouts.soma import open_dataframe

        try:
            return open_dataframe(frag.uri)
        except Exception as exc:  # noqa: BLE001 - an unreachable store is a FormatError, never empty
            raise FormatError(f"cannot open {frag.uri}: {exc}", fragment=frag.uri) from exc

    def logical_schema(self, frag: Fragment) -> Any:
        return getattr(self._frame(frag), "schema", None)

    def leaf_path(self, path: str, schema: Any) -> str:
        return str(path)

    def stats(self, frag: Fragment) -> FragmentStats:
        frame = self._frame(frag)
        rows = getattr(frame, "count", None)
        return FragmentStats(rows=int(rows) if isinstance(rows, int) else None, row_groups=1, columns={},
                             method="size_only")

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        return {"uri": frag.uri}

    def scan(self, frags: Sequence[Fragment], *, columns: list[str], predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None) -> Iterator[Any]:
        import pyarrow as pa

        from ...predicate import evaluate

        value_filter, residual = self.compile(predicate)
        for frag in frags:
            frame = self._frame(frag)
            kwargs: dict[str, Any] = {}
            if value_filter:
                kwargs["value_filter"] = value_filter
            if columns:
                kwargs["column_names"] = list(columns)
            try:
                table = frame.read(**kwargs).concat()
            except Exception as exc:  # noqa: BLE001
                raise FormatError(f"reading {frag.uri} failed: {exc}", fragment=frag.uri) from exc
            if not isinstance(table, pa.Table):
                table = pa.Table.from_pandas(table.to_pandas(), preserve_index=False)
            rows = table.to_pylist()
            if residual is not None:
                rows = [r for r in rows if evaluate(residual, r) is True]
            for i in range(0, len(rows), max(1, batch_rows)):
                yield pa.RecordBatch.from_pylist(rows[i:i + batch_rows])

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in table.to_pylist()]

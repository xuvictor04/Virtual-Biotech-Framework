"""The ``parquet`` format plugin (§9.2, §9.4, §6.4). pyarrow is imported inside methods only.

Capabilities: ``tabular``, ``pushdown``, ``stats``, ``nested``, ``leaf_projection``.

* :meth:`ParquetFormat.logical_schema` is the file's Arrow schema (``large_*`` and int32 types
  kept) with the footer's leaf paths recorded under :data:`PATHS_KEY`, plus the fragment's
  partition columns. :meth:`ParquetFormat.leaf_path` maps a §6.4 path to ``path_in_schema``
  from those recorded paths, so the legacy list encodings (``a.array``,
  ``a.bag.array_element``, ``a.element``, ``a.list.item``) map as well as ``a.list.element``.
* :meth:`ParquetFormat.compile` returns ``(pushdown, residual)``. Each top-level conjunct is
  compiled in negation normal form to an expression that is true on exactly the rows the
  :func:`~vbt.datalayer.predicate.evaluate` oracle makes true: literals are cast to the
  column's storage type (``pa.scalar(v, field.type)``, so float32 equality works and an
  inexact literal such as 2.5 on an int column never truncates); NaN is null for every float
  column (``!=``, ``NOT IN`` and ``NOT IS NULL`` carry ``is_nan`` guards); ``IsNull`` on a float
  column stays residual because Parquet row-group statistics ignore NaN and would prune the NaN
  rows. Casefold text matches push down a superset (``match_substring(ignore_case=True)`` OR the
  value holds a character with a multi-character case fold) and stay in the residual too.
  ``Contains``/``Any``/``All``/``NonEmpty``/``KindMatch``/``CensoredCmp``, paths through lists and
  ``CmpAbs`` on non-numeric columns are residual. The residual is :func:`storage_typed`.
* :meth:`ParquetFormat.scan` reads explicit fragment paths through ``pyarrow.dataset`` (one
  fragment at a time, never ``exclude_invalid_files``), with the fragment's partition values as
  its partition expression (hive columns restored with their declared types and pruned by the
  pushdown), ``batch_readahead=0``/``fragment_readahead=0``, and the residual's columns added to
  the projection. It applies the pushdown only; the caller applies the residual. Partition-only
  conjuncts are evaluated three-valued first (:func:`~vbt.datalayer.plugins.layouts.prune_fragments`).
  Any unreadable fragment raises :class:`FormatError` naming it and its partition (I14).
* :meth:`ParquetFormat.read_leaves` reads leaves inside lists by ``path_in_schema``
  (``drugs.list.element.drugId``) for whole files or chosen row groups.
* :meth:`ParquetFormat.to_native` returns plain Python: lists, dicts (maps as ``[{key, value}]``),
  NaN as ``None`` at every level, never numpy.
* Footers (``logical_schema``, ``stats``, ``metadata``) and ``read_leaves`` also open ``http(s)://``
  (``http_range``) and ``zip://`` fragments through
  :func:`~vbt.datalayer.plugins.layouts.zip_member.open_fragment`; :func:`footer_stats` turns one
  footer into per-leaf statistics for local and remote files alike.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
from datetime import date, datetime
from typing import Any, Callable, ClassVar, Iterator, Mapping, Sequence

from ...predicate import (
    And,
    Cmp,
    CmpAbs,
    Eq,
    In,
    IsNull,
    Not,
    Or,
    Param,
    Predicate,
    Range,
    TextMatch,
    columns as predicate_columns,
)
from ...roles import parse_path
from ..base import ColumnStats, Fragment, FragmentStats, FormatError, PluginBase
from ..layouts import prune_fragments
from ..registry import register
from . import conjuncts, storage_typed, top_column

__all__ = ["ParquetFormat", "PATHS_KEY", "PARTITION_TYPES", "leaf_index", "modern_paths", "arrow_type", "type_at",
           "footer_stats"]

#: Schema metadata key holding the footer's leaf paths (JSON list) in :meth:`ParquetFormat.logical_schema`.
PATHS_KEY = b"vbt.path_in_schema"
#: Declared partition types (``PartitionSpec.type``) -> Arrow type aliases.
PARTITION_TYPES = {"string": "string", "int64": "int64", "date": "date32"}
_LEGACY_WRAPPERS = 2                                   # a list adds one or two wrapper levels to a leaf path


def _arrow() -> Any:
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - the plugin declares requires=("pyarrow",)
        raise FormatError("the parquet format needs pyarrow in the data child's interpreter") from exc
    return pyarrow


def _path(frag: Fragment | str) -> str:
    uri = frag.uri if isinstance(frag, Fragment) else str(frag)
    return uri[len("file://"):] if uri.startswith("file://") else uri


def _unreadable(frag: Fragment, exc: BaseException, what: str = "not a readable Parquet file") -> FormatError:
    return FormatError(f"{_path(frag)}: {what} ({type(exc).__name__}: {exc})", fragment=frag.uri,
                       partition=frag.partition)


def arrow_type(declared: str) -> Any:
    """The Arrow type of a declared partition type (``string``, ``int64``, ``date``) or Arrow alias."""
    pa = _arrow()
    return pa.type_for_alias(PARTITION_TYPES.get(str(declared), str(declared)))


def _infer_type(value: Any) -> Any:
    pa = _arrow()
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, datetime):
        return pa.timestamp("us")
    if isinstance(value, date):
        return pa.date32()
    return pa.string()


# ---------------------------------------------------------------------------
# Leaf paths (§6.4 -> path_in_schema)
# ---------------------------------------------------------------------------

def _is_list(t: Any) -> bool:
    pa = _arrow()
    return pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t) or \
        getattr(pa.types, "is_list_view", lambda _t: False)(t) or \
        getattr(pa.types, "is_large_list_view", lambda _t: False)(t)


def _walk(t: Any, comps: Sequence[str], i: int, tokens: tuple[str, ...], ends: tuple[int, ...]
          ) -> Iterator[tuple[tuple[str, ...], tuple[int, ...], Any]]:
    """Logical tokens (names and ``[]``), the physical index after each token, and the leaf type,
    for the physical leaf path ``comps[i:]`` under Arrow type ``t``."""
    pa = _arrow()
    if pa.types.is_dictionary(t):
        t = t.value_type
    if _is_list(t):
        for skip in range(_LEGACY_WRAPPERS, 0, -1):        # 3-level (list.element) first, then 2-level
            if i + skip <= len(comps):
                yield from _walk(t.value_type, comps, i + skip, tokens + ("[]",), ends + (i + skip,))
        return
    if pa.types.is_map(t):
        if i + 1 < len(comps) and comps[i + 1] in ("key", "value"):
            sub = t.key_type if comps[i + 1] == "key" else t.item_type
            yield from _walk(sub, comps, i + 2, tokens + ("[]", comps[i + 1]), ends + (i + 1, i + 2))
        return
    if pa.types.is_struct(t):
        if i < len(comps):
            idx = t.get_field_index(comps[i])
            if idx >= 0:
                yield from _walk(t.field(idx).type, comps, i + 1, tokens + (comps[i],), ends + (i + 1,))
        return
    if i == len(comps):
        yield tokens, ends, t


def modern_paths(schema: Any) -> list[str]:
    """``path_in_schema`` of every leaf as pyarrow writes it (``list.element``, ``key_value``)."""
    pa = _arrow()
    out: list[str] = []

    def rec(t: Any, prefix: str) -> None:
        if pa.types.is_dictionary(t):
            t = t.value_type
        if _is_list(t):
            rec(t.value_type, f"{prefix}.list.element")
        elif pa.types.is_map(t):
            rec(t.key_type, f"{prefix}.key_value.key")
            rec(t.item_type, f"{prefix}.key_value.value")
        elif pa.types.is_struct(t):
            for f in t:
                rec(f.type, f"{prefix}.{f.name}")
        else:
            out.append(prefix)

    for f in schema:
        rec(f.type, f.name)
    return out


def leaf_index(schema: Any, physical: Sequence[str] | None = None
               ) -> dict[tuple[str, ...], tuple[str, tuple[int, ...], Any]]:
    """``{logical tokens: (path_in_schema, ends, leaf type)}`` for every leaf of ``schema``."""
    if physical is None:
        meta = schema.metadata or {}
        physical = json.loads(meta[PATHS_KEY]) if PATHS_KEY in meta else modern_paths(schema)
    out: dict[tuple[str, ...], tuple[str, tuple[int, ...], Any]] = {}
    for leaf in physical:
        comps = leaf.split(".")
        if comps[0] not in schema.names:
            continue
        top = schema.field(comps[0]).type
        for tokens, ends, t in _walk(top, comps, 1, (comps[0],), (1,)):
            out.setdefault(tokens, (leaf, ends, t))
            break
    return out


def _tokens(path: str) -> tuple[str, ...]:
    parsed = parse_path(path)
    if parsed.axis or parsed.up:
        raise ValueError(f"{path!r} is not a table column path")
    out: list[str] = []
    for seg in parsed.segments:
        if seg.name:
            out.append(seg.name)
        out.extend("[]" for _ in seg.brackets)             # item conditions select items: same leaf
    return tuple(out)


def type_at(schema: Any, path: str) -> Any:
    """The Arrow type at ``path`` (struct fields and lists descended; None when absent)."""
    pa = _arrow()
    names = [t for t in _tokens(path) if t != "[]"]
    if not names or names[0] not in schema.names:
        return None
    t = schema.field(names[0]).type
    for name in names[1:]:
        while _is_list(t) or pa.types.is_map(t) or pa.types.is_dictionary(t):
            if pa.types.is_dictionary(t):
                t = t.value_type
            elif pa.types.is_map(t):
                t = pa.struct([pa.field("key", t.key_type), pa.field("value", t.item_type)])
            else:
                t = t.value_type
        if not pa.types.is_struct(t) or t.get_field_index(name) < 0:
            return None
        t = t.field(t.get_field_index(name)).type
    while _is_list(t) or pa.types.is_dictionary(t):
        t = t.value_type
    return t


# ---------------------------------------------------------------------------
# Predicate compilation
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _multifold_class() -> str:
    """A regex class of the characters whose case fold is several characters (``ß`` -> ``ss``),
    which RE2's simple case folding does not match the way ``str.casefold`` does."""
    chars = [chr(c) for c in range(0x80, 0x10000) if not 0xD800 <= c <= 0xDFFF and len(chr(c).casefold()) > 1]
    return "[" + "".join(chars) + "]"


class _Compiler:
    """Compiles one conjunct to ``(expression, exact)`` or ``None`` (residual)."""

    def __init__(self, schema: Any) -> None:
        import pyarrow.compute as pc

        self.pa = _arrow()
        self.pc = pc
        self.schema = schema

    # -- fields and literals ----------------------------------------------

    def field(self, column: str) -> tuple[Any, Any] | None:
        try:
            parsed = parse_path(column)
        except ValueError:
            return None
        if parsed.axis or parsed.up or parsed.crosses_list:
            return None
        names = [s.name for s in parsed.segments]
        if not names or names[0] not in self.schema.names:
            return None
        t = self.schema.field(names[0]).type
        for name in names[1:]:
            if not self.pa.types.is_struct(t) or t.get_field_index(name) < 0:
                return None
            t = t.field(t.get_field_index(name)).type
        return self.pc.field(*names), t

    def kind(self, t: Any) -> str | None:
        types = self.pa.types
        if types.is_floating(t):
            return "float"
        if types.is_integer(t):
            return "int"
        if types.is_boolean(t):
            return "bool"
        if types.is_string(t) or types.is_large_string(t) or getattr(types, "is_string_view", lambda _t: False)(t):
            return "string"
        if types.is_timestamp(t) or types.is_date(t):
            return "temporal"
        return None

    def literal(self, value: Any, t: Any, *, equality: bool = False) -> Any:
        """A scalar the oracle would compare with ``value``: ``None`` when no faithful cast exists
        (the conjunct stays residual), ``_NEVER`` when the oracle never finds the two equal. With
        ``equality`` the scalar has exactly the column type (``is_in`` value sets)."""
        pa = self.pa
        k = self.kind(t)
        if isinstance(value, Param) or k is None:
            return None
        try:
            if k == "bool":
                return pa.scalar(value, t) if isinstance(value, bool) else _NEVER
            if k == "string":
                return pa.scalar(value, t) if isinstance(value, str) else _NEVER
            if k == "temporal":
                return pa.scalar(value, t) if isinstance(value, (date, datetime)) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return _NEVER
            if k == "int":
                if isinstance(value, float) and not value.is_integer():
                    # compared as double, never truncated; never equal to an integer
                    return _NEVER if equality else pa.scalar(value, pa.float64())
                try:
                    return pa.scalar(int(value), t)
                except (pa.ArrowInvalid, OverflowError):
                    return _NEVER if equality else pa.scalar(float(value), pa.float64())
            scalar = pa.scalar(float(value), t)
            if scalar.as_py() in (float("inf"), float("-inf")) and abs(float(value)) != float("inf"):
                # outside the float32 range: compare as double (no stored value equals it)
                return _NEVER if equality else pa.scalar(float(value), pa.float64())
            return scalar
        except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError, ValueError, TypeError):
            return None

    # -- leaves -------------------------------------------------------------

    def valid(self, f: Any, t: Any) -> Any:
        """True exactly for non-null, non-NaN values."""
        expr = f.is_valid()
        if self.kind(t) == "float":
            expr = expr & ~self.pc.is_nan(f)
        return expr

    def cmp(self, f: Any, t: Any, op: str, lit: Any) -> Any:
        if op == "==":
            return f == lit
        if op == "!=":
            expr = f != lit
            return expr & ~self.pc.is_nan(f) if self.kind(t) == "float" else expr
        return {"<": f < lit, "<=": f <= lit, ">": f > lit, ">=": f >= lit}[op]

    def leaf_cmp(self, column: str, op: str, value: Any, neg: bool, absolute: bool = False) -> Any:
        hit = self.field(column)
        if hit is None:
            return None
        f, t = hit
        if absolute:
            if self.kind(t) not in ("int", "float"):
                return None
            if self.kind(t) == "int":
                f, t = f.cast(self.pa.float64(), safe=False), self.pa.float64()
            f = self.pc.abs(f)
        if value is None or (isinstance(value, float) and value != value):
            return self.pc.scalar(False)                   # comparing with null is unknown: never true
        lit = self.literal(value, t)
        if lit is None:
            return None
        if neg:
            op = _COMPLEMENT[op]
        if op not in ("==", "!=") and self.kind(t) in ("bool", "temporal"):
            return None                                    # the oracle does not order these: residual
        if lit is _NEVER:
            # the oracle never finds the value equal (and cannot order it): only != can be true
            return self.valid(f, t) if op == "!=" else self.pc.scalar(False)
        if self.kind(t) in ("int", "float") and lit.type != t:
            # a double literal the column type cannot hold: compare as double (Arrow would otherwise
            # pick float32 for the pair); an unsafe int cast rounds as the oracle's float() does
            f = f.cast(self.pa.float64(), safe=False)
        return self.cmp(f, t, op, lit)

    def leaf_in(self, p: In, neg: bool) -> Any:
        hit = self.field(p.column)
        if hit is None:
            return None
        f, t = hit
        wanted: list[Any] = []
        for v in p.values:
            if isinstance(v, Param):
                return None
            wanted.extend(v if isinstance(v, (list, tuple, set, frozenset)) else [v])
        has_null = any(v is None or (isinstance(v, float) and v != v) for v in wanted)
        lits = []
        for v in wanted:
            if v is None or (isinstance(v, float) and v != v):
                continue
            lit = self.literal(v, t, equality=True)
            if lit is None:
                return None
            if lit is not _NEVER:
                lits.append(lit.as_py())
        if self.kind(t) == "float" and any(v == 0 for v in lits):
            lits += [0.0, -0.0]                            # is_in hashes bits: -0.0 must equal 0.0 as == does
        if not neg:
            if not lits:
                return self.pc.scalar(False)
            return self.pc.is_in(f, value_set=self.pa.array(lits, t))
        if has_null:
            return self.pc.scalar(False)                   # NOT IN (.., null) is never true
        if not lits:
            return self.valid(f, t)
        return ~self.pc.is_in(f, value_set=self.pa.array(lits, t)) & self.valid(f, t)

    def leaf_range(self, p: Range, neg: bool) -> Any:
        hit = self.field(p.column)
        if hit is None:
            return None
        f, t = hit
        if isinstance(p.lo, Param) or isinstance(p.hi, Param):
            return None
        parts = []
        if p.lo is not None:
            parts.append(Cmp(p.column, ">=" if p.lo_inclusive else ">", p.lo))
        if p.hi is not None:
            parts.append(Cmp(p.column, "<=" if p.hi_inclusive else "<", p.hi))
        if not parts:
            return self.pc.scalar(False) if neg else self.valid(f, t)
        exprs = [self.leaf_cmp(c.column, c.op, c.value, neg) for c in parts]
        if any(e is None for e in exprs):
            return None
        return functools.reduce((lambda a, b: a | b) if neg else (lambda a, b: a & b), exprs)

    def leaf_null(self, p: IsNull, neg: bool) -> Any:
        hit = self.field(p.column)
        if hit is None:
            return None
        f, t = hit
        if neg:
            return self.valid(f, t)
        if self.kind(t) == "float":
            return None                                    # statistics ignore NaN: would prune NaN rows
        return f.is_null()

    def leaf_text(self, p: TextMatch, neg: bool) -> tuple[Any, bool] | None:
        hit = self.field(p.column)
        if hit is None:
            return None
        f, t = hit
        if self.kind(t) != "string":
            return None
        if p.mode == "exact":
            return (f != p.text if neg else f == p.text), True
        if p.mode == "substring":
            m = self.pc.match_substring(f, p.text)
            return (~m if neg else m), True
        if p.mode in ("casefold", "casefold_substring") and not neg:
            if any(len(ch.casefold()) > 1 for ch in p.text):
                return None
            superset = self.pc.match_substring(f, p.text, ignore_case=True) | \
                self.pc.match_substring_regex(f, _multifold_class())
            return superset, False                         # superset: the conjunct stays residual too
        return None

    # -- structure ------------------------------------------------------------

    def compile(self, p: Predicate, neg: bool = False) -> tuple[Any, bool] | None:
        if isinstance(p, Not):
            return self.compile(p.pred, not neg)
        if isinstance(p, (And, Or)):
            parts = [self.compile(q, neg) for q in p.preds]
            if any(r is None for r in parts) or not parts:
                return None
            disjunctive = isinstance(p, Or) != neg
            exprs = [r[0] for r in parts]                   # type: ignore[index]
            exact = all(r[1] for r in parts)                # type: ignore[index]
            return functools.reduce((lambda a, b: a | b) if disjunctive else (lambda a, b: a & b), exprs), exact
        if isinstance(p, TextMatch):
            return self.leaf_text(p, neg)
        expr: Any = None
        if isinstance(p, Eq):
            expr = self.leaf_cmp(p.column, "==", p.value, neg)
        elif isinstance(p, Cmp):
            expr = self.leaf_cmp(p.column, p.op, p.value, neg)
        elif isinstance(p, CmpAbs):
            expr = self.leaf_cmp(p.column, p.op, p.value, neg, absolute=True)
        elif isinstance(p, In):
            expr = self.leaf_in(p, neg)
        elif isinstance(p, Range):
            expr = self.leaf_range(p, neg)
        elif isinstance(p, IsNull):
            expr = self.leaf_null(p, neg)
        return None if expr is None else (expr, True)


_COMPLEMENT = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "!=": "==", "==": "!="}
_NEVER = object()


# ---------------------------------------------------------------------------
# Native conversion
# ---------------------------------------------------------------------------

def _converter(t: Any) -> Callable[[Any], Any] | None:
    """A per-value converter for Arrow type ``t`` (None when ``to_pylist`` output is already native)."""
    pa = _arrow()
    if pa.types.is_dictionary(t):
        return _converter(t.value_type)
    if pa.types.is_floating(t):
        return lambda v: None if v is not None and v != v else v
    if pa.types.is_map(t):
        kc, vc = _converter(t.key_type), _converter(t.item_type)
        return lambda v: None if v is None else [{"key": kc(k) if kc else k, "value": vc(x) if vc else x} for k, x in v]
    if _is_list(t):
        inner = _converter(t.value_type)
        if inner is None:
            return None
        return lambda v: None if v is None else [inner(x) for x in v]
    if pa.types.is_struct(t):
        convs = {f.name: _converter(f.type) for f in t}
        if not any(convs.values()):
            return None
        return lambda v: None if v is None else {k: (convs[k](x) if convs.get(k) else x) for k, x in v.items()}
    return None


def _row_group_chunks(piece: Any, wanted: Sequence[int] | None, pushdown: Any, schema: Any, batch_rows: int
                      ) -> Iterator[list[int]]:
    """Row-group ids to read, in file order and in chunks of at most ``batch_rows`` rows (one row group
    when it is larger), after row-group statistics pruning: the decode never holds more than one chunk."""
    infos = {rg.id: rg.num_rows for rg in piece.row_groups}
    ids = [i for i in (wanted if wanted is not None else sorted(infos)) if i in infos]
    if pushdown is not None and ids:
        kept = {rg.id for rg in piece.subset(filter=pushdown, schema=schema).row_groups}
        ids = [i for i in ids if i in kept]
    chunk: list[int] = []
    rows = 0
    for i in ids:
        if chunk and rows + infos[i] > batch_rows:
            yield chunk
            chunk, rows = [], 0
        chunk.append(i)
        rows += infos[i]
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Footer statistics
# ---------------------------------------------------------------------------

def _is_text(t: Any) -> bool:
    pa = _arrow()
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) or \
        pa.types.is_large_binary(t)


def footer_stats(schema: Any, md: Any) -> FragmentStats:
    """Per-leaf statistics of one file from its footer alone: ``schema`` is the file's Arrow schema and
    ``md`` its ``FileMetaData`` (a local file or a footer fetched with range requests, so local and
    remote fragments report the same storage types, null counts and min/max)."""
    physical = [md.schema.column(i).path for i in range(md.num_columns)]
    types = {leaf: t for leaf, _ends, t in leaf_index(schema, physical).values()}
    cols: dict[str, ColumnStats] = {}
    for i, leaf in enumerate(physical):
        col = md.schema.column(i)
        unc = nv = 0
        nulls: int | None = 0
        mn = mx = None
        minmax = md.num_row_groups > 0
        for rg in range(md.num_row_groups):
            cc = md.row_group(rg).column(i)
            unc += int(cc.total_uncompressed_size or 0)
            nv += int(cc.num_values or 0)
            st = cc.statistics
            if st is None or not st.has_null_count:
                nulls = None
            elif nulls is not None:
                nulls += int(st.null_count)
            if st is None or not st.has_min_max:
                minmax = False
            elif minmax:
                try:
                    mn = st.min if mn is None or st.min < mn else mn
                    mx = st.max if mx is None or st.max > mx else mx
                except TypeError:
                    minmax = False
        t = types.get(leaf)
        storage = str(t) if t is not None else str(col.physical_type).lower()
        if col.max_repetition_level > 0:
            kind = "nested"
        elif t is not None and _is_text(t):
            kind = "string"
        else:
            kind = "flat"
        if isinstance(mn, bytes) or isinstance(mx, bytes):
            mn = mn.decode("utf-8", "replace") if isinstance(mn, bytes) else mn
            mx = mx.decode("utf-8", "replace") if isinstance(mx, bytes) else mx
        cols[leaf] = ColumnStats(uncompressed_bytes=unc, null_count=nulls, num_values=nv,
                                 max_rep_level=int(col.max_repetition_level),
                                 max_def_level=int(col.max_definition_level),
                                 min=mn if minmax else None, max=mx if minmax else None,
                                 storage_type=storage, kind=kind)  # type: ignore[arg-type]
    return FragmentStats(rows=int(md.num_rows), row_groups=int(md.num_row_groups), columns=cols, method="footer")


# ---------------------------------------------------------------------------
# The plugin
# ---------------------------------------------------------------------------

@register
class ParquetFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "parquet"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"tabular", "pushdown", "stats", "nested", "leaf_projection"})
    requires: ClassVar[tuple[str, ...]] = ("pyarrow",)

    # -- footers ----------------------------------------------------------------

    @contextlib.contextmanager
    def _open(self, frag: Fragment) -> Iterator[Any]:
        """The fragment as a ``ParquetFile``, closed on exit. A local path is opened by name; an
        ``http(s)://`` URL (``http_range``) or a ``zip://`` member is read through
        :func:`~vbt.datalayer.plugins.layouts.zip_member.open_fragment`, so a footer read transfers
        only the footer."""
        pa = _arrow()
        import pyarrow.parquet as pq

        from ..layouts.zip_member import local_path, open_fragment

        path = local_path(frag)
        raw = None
        try:
            if path is None:
                raw = open_fragment(frag)
            pf = pq.ParquetFile(raw if raw is not None else path)
        except (pa.ArrowException, OSError, ValueError) as exc:
            if raw is not None:
                raw.close()
            raise _unreadable(frag, exc) from exc
        try:
            yield pf
        finally:
            close = getattr(pf, "close", None)
            if callable(close):
                close()
            if raw is not None:
                raw.close()

    def _footer(self, frag: Fragment) -> tuple[Any, Any]:
        """``(arrow schema, FileMetaData)`` of one fragment (FormatError when unreadable)."""
        pa = _arrow()
        with self._open(frag) as pf:
            try:
                return pf.schema_arrow, pf.metadata
            except (pa.ArrowException, OSError, ValueError) as exc:
                raise _unreadable(frag, exc) from exc

    def logical_schema(self, frag: Fragment) -> Any:
        pa = _arrow()
        schema, md = self._footer(frag)
        physical = [md.schema.column(i).path for i in range(md.num_columns)]
        meta = dict(schema.metadata or {})
        meta[PATHS_KEY] = json.dumps(physical).encode()
        schema = schema.with_metadata(meta)
        for key, value in (frag.partition or {}).items():
            if key not in schema.names:
                schema = schema.append(pa.field(key, _infer_type(value)))
        return schema

    def leaf_path(self, path: str, schema: Any) -> str:
        want = _tokens(path)
        leaves = leaf_index(schema)
        if want in leaves:
            return leaves[want][0]
        for k in range(1, 4):                               # list<T> named without brackets: its elements
            if want + ("[]",) * k in leaves:
                return leaves[want + ("[]",) * k][0]
        n = len(want)
        for tokens, (leaf, ends, _t) in leaves.items():     # a group (struct, list of structs): its prefix
            if tokens[:n] == want and len(tokens) > n:
                return ".".join(leaf.split(".")[:ends[n - 1]])
        if len(want) == 1 and want[0] in schema.names:
            return want[0]                                  # a partition column (not in the file)
        raise ValueError(f"{path!r} is not a column path of this schema")

    def stats(self, frag: Fragment) -> FragmentStats:
        schema, md = self._footer(frag)
        return footer_stats(schema, md)

    @staticmethod
    def _is_text(t: Any) -> bool:
        return _is_text(t)

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        schema, md = self._footer(frag)
        out = {"created_by": str(md.created_by or ""), "format_version": str(md.format_version),
               "num_rows": str(md.num_rows), "num_row_groups": str(md.num_row_groups)}
        for k, v in (md.metadata or {}).items():
            key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
            if key == "ARROW:schema":
                continue                                    # base64 IPC schema: not a header field
            out[key] = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
        return out

    # -- predicates ---------------------------------------------------------------

    def storage_type_of(self, schema: Any) -> Callable[[str], str | None]:
        def type_of(path: str) -> str | None:
            try:
                t = type_at(schema, path)
            except ValueError:
                return None
            return None if t is None else str(t)

        return type_of

    def compile(self, predicate: Predicate, schema: Any) -> tuple[Any, Predicate | None]:
        if predicate is None:
            return None, None
        comp = _Compiler(schema)
        pushed: list[Any] = []
        residual: list[Predicate] = []
        for c in conjuncts(predicate):
            try:
                hit = comp.compile(c)
            except (ValueError, TypeError):                 # malformed paths stay with the oracle
                hit = None
            if hit is None:
                residual.append(c)
                continue
            expr, exact = hit
            pushed.append(expr)
            if not exact:
                residual.append(c)
        pushdown = functools.reduce(lambda a, b: a & b, pushed) if pushed else None
        rest: Predicate | None = None
        if residual:
            rest = residual[0] if len(residual) == 1 else And(tuple(residual))
        return pushdown, storage_typed(rest, self.storage_type_of(schema))

    # -- scans --------------------------------------------------------------------

    def scan_schema(self, frags: Sequence[Fragment], partitions: Mapping[str, str]) -> Any:
        """The dataset schema of a scan: the first fragment's logical schema with the declared
        partition types (partition columns that are also stored in the file are dropped from it)."""
        pa = _arrow()
        schema = self.logical_schema(frags[0])
        declared = {k: arrow_type(v) for k, v in (partitions or {}).items()}
        found = sorted({k for f in frags for k in (f.partition or {})} - set(declared))
        keys = [*declared, *found]
        fields = [f for f in schema if f.name not in keys]
        for key in keys:
            if key in declared:
                fields.append(pa.field(key, declared[key]))
            else:
                sample = next((f.partition[key] for f in frags if f.partition.get(key) is not None), None)
                fields.append(pa.field(key, _infer_type(sample)))
        return pa.schema(fields, metadata=schema.metadata)

    def _partition_expr(self, frag: Fragment, schema: Any) -> Any:
        pa = _arrow()
        import pyarrow.compute as pc

        expr = pc.scalar(True)
        for key, value in (frag.partition or {}).items():
            if key not in schema.names:
                continue
            if value is None:
                expr = expr & pc.field(key).is_null()
                continue
            try:
                expr = expr & (pc.field(key) == pa.scalar(value, schema.field(key).type))
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
                raise _unreadable(frag, exc, f"partition {key}={value!r} is not a {schema.field(key).type}") from exc
        return expr

    def _projection(self, columns: Sequence[str] | None, residual: Predicate | None, schema: Any) -> Any:
        import pyarrow.compute as pc

        wanted = list(columns) if columns else list(schema.names)
        extra = sorted({top_column(c) for c in predicate_columns(residual)}) if residual is not None else []
        out: dict[str, Any] = {}
        plain = True
        for c in [*wanted, *extra]:
            parsed = parse_path(c)
            head = parsed.segments[0].name if parsed.segments else c
            if head not in schema.names:
                raise ValueError(f"unknown column {c!r} (columns: {', '.join(schema.names)})")
            if parsed.crosses_list or len(parsed.segments) == 1:
                out.setdefault(head, pc.field(head))
            else:
                plain = False
                out.setdefault(c, pc.field(*[s.name for s in parsed.segments]))
        return list(out) if plain else out

    def scan(self, frags: Sequence[Fragment], *, columns: list[str] | None, predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None, memory_pool: Any = None) -> Iterator[Any]:
        """Batches of the pushdown's rows (the caller applies the residual). ``memory_pool`` (an Arrow
        pool) receives the scan's allocations, so callers can bound or measure them."""
        frags = list(frags)
        if not frags:
            return iter(())
        return self._scan(frags, columns, predicate, partitions or {}, batch_rows, row_groups, memory_pool)

    def _scan(self, frags: list[Fragment], columns: list[str] | None, predicate: Predicate | None,
              partitions: Mapping[str, str], batch_rows: int, row_groups: Mapping[str, Sequence[int]] | None,
              memory_pool: Any) -> Iterator[Any]:
        pa = _arrow()
        import pyarrow.dataset as ds
        import pyarrow.fs as pafs

        keep = prune_fragments(frags, predicate)
        if not keep:
            return
        schema = self.scan_schema(keep, partitions)       # a pruned fragment is never opened
        pushdown, residual = self.compile(predicate, schema) if predicate is not None else (None, None)
        projection = self._projection(columns, residual, schema)
        fmt = ds.ParquetFileFormat()
        fs = pafs.LocalFileSystem()
        for frag in keep:
            expr = self._partition_expr(frag, schema)
            try:
                dataset = ds.FileSystemDataset.from_paths([os.path.abspath(_path(frag))], schema=schema, format=fmt,
                                                          filesystem=fs, partitions=[expr])
                fragments = list(dataset.get_fragments(filter=pushdown) if pushdown is not None
                                 else dataset.get_fragments())
                for piece in fragments:
                    for ids in _row_group_chunks(piece, (row_groups or {}).get(frag.uri), pushdown, schema,
                                                 batch_rows):
                        for batch in piece.subset(row_group_ids=ids).to_batches(
                                schema=schema, columns=projection, filter=pushdown,
                                batch_size=max(1, int(batch_rows)), batch_readahead=0, fragment_readahead=0,
                                memory_pool=memory_pool):
                            yield batch
                            del batch                       # never hold two batches while the next decodes
            except (pa.ArrowException, OSError) as exc:
                raise _unreadable(frag, exc) from exc

    def read_leaves(self, frag: Fragment, leaves: list[str], row_groups: Sequence[int] | None) -> Any:
        pa = _arrow()
        schema = None
        physical: list[str] = []
        for leaf in leaves:
            if "[" in leaf or "@" in leaf or "^" in leaf:
                schema = schema if schema is not None else self.logical_schema(frag)
                physical.append(self.leaf_path(leaf, schema))
            else:
                physical.append(leaf)
        with self._open(frag) as pf:
            try:
                if row_groups is None:
                    return pf.read(columns=physical)
                return pf.read_row_groups(list(row_groups), columns=physical)
            except (pa.ArrowException, OSError) as exc:
                raise _unreadable(frag, exc) from exc

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        rows = table.to_pylist()
        convs = {f.name: c for f in table.schema if (c := _converter(f.type)) is not None}
        if convs:
            for row in rows:
                for name, conv in convs.items():
                    row[name] = conv(row.get(name))
        return rows

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases, parquet_with_paths, write_parquet

        return FormatCases(extension=".parquet", write=write_parquet, with_physical_paths=parquet_with_paths)

"""The ``csv`` and ``tsv`` format plugins (§9.2, §9.4, §6.7; phase 2, F12). pyarrow is imported inside
methods only.

Capabilities: ``tabular``, ``stats_scan``, ``matrix``.

* **Reading.** Files are read with ``pyarrow.csv`` streaming readers under **positional** column
  names (``c0``, ``c1`` ...), then renamed to their header names, so a duplicated header never
  selects the wrong column (pyarrow keeps duplicate CSV headers and ``include_columns`` silently
  returns the first, VERIFIED); a duplicated name is exposed as ``name#<position>``. Column types are
  inferred once per fragment from the first block (``options.column_types`` overrides) and applied
  to every block, so a later block that does not convert is a :class:`FormatError`, never a silent
  change of type. Unquoted empty cells are null; a quoted ``""`` stays the empty string. Local
  paths, ``.gz`` files, archive members (``zip://``) and ``http(s)://`` URLs are read alike
  (:func:`~vbt.datalayer.plugins.layouts.zip_member.open_fragment`).
* **Unreadable files** raise :class:`FormatError` naming the fragment (I14, F-8): a zero-length file,
  binary content (NUL bytes), a ragged row, a value that does not convert, or a file that does not
  end with a newline (a download cut mid-row; ``options.allow_unterminated: true`` accepts it).
  A file cut exactly at a row boundary parses: its shape is checked against the GCT dimension line
  or the manifest's byte count (R2).
* **stats_scan.** There are no footers: :meth:`CsvFormat.stats` makes one bounded streaming pass
  (one block at a time) and caches the result per fragment identity (uri, size, mtime, sha256) and
  configuration in :data:`STATS_CACHE`, so readiness and admission never rescan an unchanged file.
* **Predicates** are never pushed down (no ``pushdown`` capability): :meth:`CsvFormat.compile`
  returns the whole predicate as the storage-typed residual and the reader evaluates it.
* **matrix** (header-axis matrices: DepMap ``CRISPRGeneEffect.csv``, GCT). ``configure(options,
  matrix)`` binds a :class:`~vbt.datalayer.descriptor.models.MatrixSpec` (or its dict): the row
  axis is a column (``{index: 0}`` or a name with ``aliases``), the column axis is the header after
  ``exclude``, parsed by ``ParseSpec`` (``"A1BG (1)"`` -> ``symbol``, ``entrez_id``). Values are read
  **by position after the header checks**. :meth:`CsvFormat.matrix_checks` reports unparseable and
  duplicate headers, duplicate axis keys, all-empty columns (inferred as ``null``) and a GCT
  dimension line that disagrees with the data. :meth:`CsvFormat.slice` returns long-view batches
  ``(<row key>, <col key>, <value>)`` in row-major order; empty and NaN cells are null (the value
  role's ``missing: unknown``).

The helpers below the plugin (:func:`native_rows`, :data:`STATS_CACHE`, :func:`stats_from_batches`,
:class:`MatrixLayout`, :func:`long_batch`, :class:`SliceBudgetExceeded`) are shared by the other
phase-2 formats (``jsonl``, ``h5ad``, ``zarr``, ``obo``, ``gmt``, ``npy``).
"""

from __future__ import annotations

import copy
import csv as _csv
import gzip
import hashlib
import io
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Iterable, Iterator, Mapping, Self, Sequence

from ...predicate import Predicate, columns as predicate_columns, evaluate
from ...roles import parse_path
from ..base import CheckItem, ColumnStats, FormatError, Fragment, FragmentStats, PluginBase
from ..registry import register
from . import storage_typed, top_column

__all__ = [
    "CsvFormat", "TsvFormat", "STATS_CACHE", "ScanStatsCache", "SliceBudgetExceeded", "MatrixAxis", "MatrixLayout",
    "matrix_layout", "long_batch", "long_schema", "native_rows", "stats_from_batches", "fragment_identity",
    "unreadable", "residual_compile", "generic_leaf_path", "storage_type_of", "open_bytes", "fragment_name",
    "write_csv", "write_gct",
]

#: Bytes of the first block used for type inference and header checks.
INFER_BYTES = 1 << 20
#: Header lines of a GCT 1.2 file before the column header (``#1.2`` and the dimension line).
GCT_PREAMBLE = 2
#: The statistics scan (``stats``) decodes blocks of about this many rows, at most :data:`STATS_BLOCK_BYTES` each.
STATS_BATCH_ROWS = 65536
STATS_BLOCK_BYTES = 32 << 20


def _arrow() -> Any:
    try:
        import pyarrow
    except ImportError as exc:  # pragma: no cover - the plugin declares requires=("pyarrow",)
        raise FormatError("this format needs pyarrow in the data child's interpreter") from exc
    return pyarrow


def fragment_name(frag: Fragment | str) -> str:
    uri = frag.uri if isinstance(frag, Fragment) else str(frag)
    return uri[len("file://"):] if uri.startswith("file://") else uri


def unreadable(frag: Fragment, exc: BaseException | None, what: str) -> FormatError:
    """A :class:`FormatError` naming the fragment and its partition."""
    detail = f" ({type(exc).__name__}: {exc})" if exc is not None else ""
    return FormatError(f"{fragment_name(frag)}: {what}{detail}", fragment=frag.uri, partition=frag.partition)


def fragment_identity(frag: Fragment) -> tuple[Any, ...]:
    """What identifies a fragment's bytes for caches keyed per fingerprint."""
    return (frag.uri, frag.size, frag.mtime_ns, frag.sha256)


def open_bytes(frag: Fragment) -> Any:
    """A binary file for a fragment (local, ``.gz``, archive member or URL), decompressed when gzipped."""
    from ..layouts.zip_member import open_fragment

    try:
        fh = open_fragment(frag)
    except OSError as exc:
        raise unreadable(frag, exc, "cannot be opened") from exc
    if fragment_name(frag).endswith(".gz"):
        return gzip.GzipFile(fileobj=fh, mode="rb")
    return fh


# ---------------------------------------------------------------------------
# Shared helpers: native rows, residual compilation, leaf paths
# ---------------------------------------------------------------------------

def native_rows(table: Any) -> list[dict[str, Any]]:
    """Plain Python rows: lists, dicts, NaN as None at every level, never numpy (F-1)."""
    from .parquet import _converter

    rows = table.to_pylist()
    convs = {f.name: c for f in table.schema if (c := _converter(f.type)) is not None}
    if convs:
        for row in rows:
            for name, conv in convs.items():
                row[name] = conv(row.get(name))
    return rows


def storage_type_of(schema: Any) -> Callable[[str], str | None]:
    from .parquet import type_at

    def type_of(path: str) -> str | None:
        try:
            t = type_at(schema, path)
        except ValueError:
            return None
        return None if t is None else str(t)

    return type_of


def residual_compile(predicate: Predicate | None, schema: Any) -> tuple[Any, Predicate | None]:
    """No pushdown: the whole predicate is the storage-typed residual."""
    if predicate is None:
        return None, None
    return None, storage_typed(predicate, storage_type_of(schema))


def generic_leaf_path(path: str, schema: Any) -> str:
    """§6.4 path -> the leaf path the way Parquet would write it (stats keys of non-Parquet formats)."""
    from .parquet import leaf_index, modern_paths

    parsed = parse_path(path)
    if parsed.axis or parsed.up:
        raise ValueError(f"{path!r} is not a table column path")
    want: list[str] = []
    for seg in parsed.segments:
        if seg.name:
            want.append(seg.name)
        want.extend("[]" for _ in seg.brackets)
    leaves = leaf_index(schema, modern_paths(schema))
    tokens = tuple(want)
    if tokens in leaves:
        return leaves[tokens][0]
    for k in range(1, 4):
        if tokens + ("[]",) * k in leaves:
            return leaves[tokens + ("[]",) * k][0]
    n = len(tokens)
    for toks, (leaf, ends, _t) in leaves.items():
        if toks[:n] == tokens and len(toks) > n:
            return ".".join(leaf.split(".")[:ends[n - 1]])
    if len(tokens) == 1 and tokens[0] in schema.names:
        return tokens[0]
    raise ValueError(f"{path!r} is not a column path of this schema")


def projection(columns: Sequence[str] | None, residual: Predicate | None, names: Sequence[str]) -> list[str]:
    """Top-level columns a scan reads: the requested ones plus those the residual needs."""
    wanted = list(columns) if columns else list(names)
    extra = sorted({top_column(c) for c in predicate_columns(residual)}) if residual is not None else []
    out: dict[str, None] = {}
    for c in [*wanted, *extra]:
        head = top_column(c) if c not in names else c
        if head not in names:
            raise ValueError(f"unknown column {c!r} (columns: {', '.join(names)})")
        out.setdefault(head, None)
    return list(out)


# ---------------------------------------------------------------------------
# stats_scan: one bounded pass per fragment identity
# ---------------------------------------------------------------------------

class ScanStatsCache:
    """FragmentStats of ``stats_scan`` formats by ``(plugin, configuration, fragment identity)``; ``passes``
    counts the scans actually made (one per fingerprint)."""

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[tuple[Any, ...], FragmentStats] = OrderedDict()
        self._lock = threading.Lock()
        self.passes: dict[tuple[Any, ...], int] = {}

    def get_or_scan(self, key: tuple[Any, ...], scan: Callable[[], FragmentStats]) -> FragmentStats:
        with self._lock:
            hit = self._data.get(key)
            if hit is not None:
                self._data.move_to_end(key)
                return hit
        st = scan()
        with self._lock:
            self.passes[key] = self.passes.get(key, 0) + 1
            self._data[key] = st
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
        return st

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.passes.clear()

    def __len__(self) -> int:
        return len(self._data)


STATS_CACHE = ScanStatsCache()


class Memo:
    """A small thread-safe LRU of values computed once per key (headers and inferred types per fragment)."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...], compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        value = compute()
        with self._lock:
            self._data[key] = value
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
        return value


_HEADERS = Memo()
_TAILS = Memo()
_TYPES = Memo()


def _kind(t: Any) -> str:
    pa = _arrow()
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_struct(t) or pa.types.is_fixed_size_list(t):
        return "nested"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t):
        return "string"
    return "flat"


def stats_from_batches(batches: Iterable[Any], *, shape: tuple[int, int] | None = None) -> FragmentStats:
    """Row count and per-column null counts, min/max (flat columns; NaN skipped), bytes and storage types,
    accumulated one batch at a time."""
    pa = _arrow()
    import pyarrow.compute as pc

    rows = 0
    acc: dict[str, dict[str, Any]] = {}
    for batch in batches:
        rows += batch.num_rows
        for name, col in zip(batch.schema.names, batch.columns):
            a = acc.setdefault(name, {"bytes": 0, "nulls": 0, "values": 0, "min": None, "max": None,
                                      "type": col.type, "minmax": True})
            a["bytes"] += int(col.nbytes)
            a["nulls"] += int(col.null_count)
            if isinstance(col, (pa.ListArray, pa.LargeListArray)):
                a["values"] += len(col.flatten())
            else:
                a["values"] += len(col)
            if a["minmax"] and _kind(col.type) != "nested" and not pa.types.is_null(col.type) \
                    and not pa.types.is_boolean(col.type):
                try:
                    mm = pc.min_max(col).as_py()
                except (pa.ArrowNotImplementedError, pa.ArrowInvalid, TypeError):
                    a["minmax"] = False
                    continue
                for k, better in (("min", lambda x, y: x < y), ("max", lambda x, y: x > y)):
                    v = mm.get(k)
                    if v is not None and (a[k] is None or better(v, a[k])):
                        a[k] = v
    cols = {name: ColumnStats(uncompressed_bytes=a["bytes"], null_count=a["nulls"], num_values=a["values"],
                              min=a["min"] if a["minmax"] else None, max=a["max"] if a["minmax"] else None,
                              storage_type=str(a["type"]), kind=_kind(a["type"]))  # type: ignore[arg-type]
            for name, a in acc.items()}
    return FragmentStats(rows=rows, row_groups=1, columns=cols, method="scan", shape=shape)


# ---------------------------------------------------------------------------
# Matrices: the logical long view (§6.7), shared by csv, h5ad, zarr and npy
# ---------------------------------------------------------------------------

class SliceBudgetExceeded(Exception):
    """A matrix slice would emit more than ``budget_bytes`` of long-view cells (the reader maps it to
    ``too_large``); nothing beyond the budget is decoded."""

    def __init__(self, message: str, *, emitted_bytes: int, budget_bytes: int) -> None:
        super().__init__(message)
        self.emitted_bytes = emitted_bytes
        self.budget_bytes = budget_bytes


@dataclass(frozen=True)
class MatrixAxis:
    """One axis of a matrix as a format needs it (from ``AxisSpec``)."""

    name: str                                          # long-view name ("model", "gene")
    source: str                                        # column | header | index | table | positional | file
    key: tuple[str, ...]                               # axis key fields ("ModelID",) / ("entrez_id",)
    column: str | int | None = None                    # row-ID column name or position
    aliases: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    pattern: str | None = None                         # ParseSpec.pattern
    parse_fields: tuple[str, ...] = ()
    on_mismatch: str = "error"
    index_name: str | None = None
    ids_from: str | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        out = list(self.parse_fields) if self.parse_fields else list(self.key)
        for k in self.key:
            if k not in out:
                out.append(k)
        return tuple(out)


@dataclass(frozen=True)
class MatrixLayout:
    row: MatrixAxis
    col: MatrixAxis
    values: tuple[str, ...] = ("value",)
    implicit: str = "none"
    storage: str = "dense"
    value_types: Mapping[str, str] = field(default_factory=dict)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        if name == "from_":
            return obj.get("from", obj.get("from_", default))
        return obj.get(name, default)
    return getattr(obj, name, default)


def _axis(name: str, ax: Any) -> MatrixAxis:
    key = _get(ax, "key") or {}
    key_cols = tuple(_get(key, "columns") or ())
    column = _get(ax, "column")
    if isinstance(column, Mapping):
        column = int(column.get("index", 0))
    parse = _get(ax, "parse")
    pattern = _get(parse, "pattern") if parse is not None else None
    pfields = tuple((_get(parse, "fields") or {}).keys()) if parse is not None else ()
    return MatrixAxis(name=str(_get(ax, "name") or name), source=str(_get(ax, "from_") or "column"),
                      key=key_cols, column=column, aliases=tuple(_get(ax, "aliases") or ()),
                      exclude=tuple(_get(ax, "exclude") or ()), pattern=pattern, parse_fields=pfields,
                      on_mismatch=str(_get(parse, "on_mismatch") or "error") if parse is not None else "error",
                      index_name=_get(ax, "index_name"), ids_from=_get(ax, "ids_from"))


def matrix_layout(spec: Any) -> MatrixLayout:
    """A :class:`MatrixLayout` from a ``MatrixSpec`` model or its dict (``obs``/``var`` = ``row``/``col``)."""
    if isinstance(spec, MatrixLayout):
        return spec
    axes = _get(spec, "axes") or {}
    if isinstance(axes, Mapping):
        axes = {{"obs": "row", "var": "col"}.get(str(k), str(k)): v for k, v in axes.items()}
    if "row" not in axes or "col" not in axes:
        raise ValueError("a matrix declares both axes (row/obs and col/var)")
    values = _get(spec, "values") or {"value": {}}
    names = tuple(values.keys()) if isinstance(values, Mapping) else tuple(values)
    return MatrixLayout(row=_axis("row", axes["row"]), col=_axis("col", axes["col"]), values=names,
                        implicit=str(_get(spec, "implicit") or "none"), storage=str(_get(spec, "storage") or "dense"))


def long_schema(layout: MatrixLayout, value: str, attributes: Sequence[str] = ()) -> Any:
    """``(<row key fields>, <col key fields>, <attributes>, <value>)``: strings for keys, float64 values."""
    pa = _arrow()
    fields = [pa.field(k, pa.string()) for k in layout.row.key]
    fields += [pa.field(k, pa.string()) for k in layout.col.key if k not in layout.row.key]
    fields += [pa.field(a, pa.string()) for a in attributes if a not in layout.row.key + layout.col.key]
    fields.append(pa.field(value, pa.float64()))
    return pa.schema(fields)


def _strings(values: Sequence[Any]) -> Any:
    import numpy as np

    return np.asarray([None if v is None else str(v) for v in values], dtype=object)


def long_batch(layout: MatrixLayout, value: str, row_keys: Mapping[str, Sequence[Any]],
               col_keys: Mapping[str, Sequence[Any]], block: Any, *,
               row_attributes: Mapping[str, Sequence[Any]] | None = None,
               col_attributes: Mapping[str, Sequence[Any]] | None = None, missing: Any = None) -> Any:
    """A row-major long-view batch from an ``R x C`` float block (numpy; NaN = missing). ``row_keys``
    and ``col_keys`` map each key field to its ``R`` / ``C`` values; ``row_attributes`` and
    ``col_attributes`` add axis fields. ``missing`` is an ``R x C`` boolean mask of cells that are null
    for another reason than NaN (an unmeasured sparse cell)."""
    pa = _arrow()
    import numpy as np

    block = np.asarray(block, dtype="float64")
    n_rows, n_cols = block.shape if block.ndim == 2 else (0, 0)
    arrays: dict[str, Any] = {}
    for k in layout.row.key:
        arrays[k] = pa.array(np.repeat(_strings(row_keys[k]), n_cols), pa.string())
    for k in layout.col.key:
        if k not in arrays:
            arrays[k] = pa.array(np.tile(_strings(col_keys[k]), n_rows), pa.string())
    names: list[str] = []
    for attrs, repeat in ((row_attributes, True), (col_attributes, False)):
        for name, vals in (attrs or {}).items():
            if name in arrays:
                continue
            arr = np.repeat(_strings(vals), n_cols) if repeat else np.tile(_strings(vals), n_rows)
            arrays[name] = pa.array(arr, pa.string())
            names.append(name)
    flat = block.reshape(-1)
    mask = np.isnan(flat)
    if missing is not None:
        mask = mask | np.asarray(missing, dtype=bool).reshape(-1)
    arrays[value] = pa.array(flat, pa.float64(), mask=mask)
    return pa.RecordBatch.from_pydict(arrays, schema=long_schema(layout, value, names))


def charge(emitted: int, cells: int, row_key_bytes: int, col_key_bytes: int, budget_bytes: int | None,
           where: str) -> int:
    """Add a block's long-view bytes to ``emitted`` and raise :class:`SliceBudgetExceeded` past the budget."""
    emitted += cells * 8 + row_key_bytes + col_key_bytes
    if budget_bytes is not None and emitted > budget_bytes:
        raise SliceBudgetExceeded(f"{where}: the slice exceeds its budget ({emitted} > {budget_bytes} bytes); "
                                  "restrict the rows or columns", emitted_bytes=emitted, budget_bytes=budget_bytes)
    return emitted


def row_filter(layout: MatrixLayout, row_predicate: Predicate | None, rows: Sequence[Mapping[str, Any]]
               ) -> list[int]:
    """Indices of the axis rows a row predicate keeps (Kleene: unknown is dropped)."""
    if row_predicate is None:
        return list(range(len(rows)))
    return [i for i, r in enumerate(rows) if evaluate(row_predicate, r) is True]


# ---------------------------------------------------------------------------
# The plugin
# ---------------------------------------------------------------------------

@dataclass
class _Header:
    names: list[str]                                   # raw header cells, positional
    skip: int                                          # lines before the first data row
    dims: tuple[int, int] | None = None                # GCT dimension line (rows, data columns)
    preamble: list[str] = field(default_factory=list)


@register
class CsvFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "csv"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"tabular", "stats_scan", "matrix"})
    requires: ClassVar[tuple[str, ...]] = ("pyarrow",)
    delimiter: ClassVar[str] = ","

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}
        self.matrix: MatrixLayout | None = None

    # -- configuration ------------------------------------------------------------

    def configure(self, options: Mapping[str, Any] | None = None, matrix: Any = None) -> Self:
        """A configured copy: ``options`` (``delimiter``, ``preamble: gct | <n lines>``, ``null_values``,
        ``column_types``, ``allow_unterminated``, ``newlines_in_values``, ``encoding``) and an optional
        matrix layout (``MatrixSpec`` or dict)."""
        other = copy.copy(self)
        other.options = dict(options or {})
        other.matrix = matrix_layout(matrix) if matrix is not None else None
        return other

    def _config_key(self) -> str:
        data = {"o": self.options, "m": repr(self.matrix)}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @property
    def _delimiter(self) -> str:
        d = self.options.get("delimiter") or self.delimiter
        return "\t" if d in ("\\t", "tab") else str(d)

    def _gct(self, frag: Fragment) -> bool:
        pre = self.options.get("preamble")
        return pre == "gct" or (pre is None and fragment_name(frag).lower().removesuffix(".gz").endswith(".gct"))

    # -- header ---------------------------------------------------------------------

    def _head(self, frag: Fragment, n: int = INFER_BYTES) -> bytes:
        try:
            with open_bytes(frag) as fh:
                data = fh.read(n)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise unreadable(frag, exc, "cannot be read") from exc
        if not data:
            raise unreadable(frag, None, "is empty (no header): not a CSV file")
        if b"\x00" in data:
            raise unreadable(frag, None, "holds binary data (NUL bytes): not a text table")
        return data

    def _tail_ok(self, frag: Fragment) -> None:
        """FormatError unless the file ends with a newline (checked once per fragment identity)."""
        if self.options.get("allow_unterminated"):
            return
        _TAILS.get(self._key(frag), lambda: self._check_tail(frag))

    def _check_tail(self, frag: Fragment) -> bool:
        try:
            with open_bytes(frag) as fh:
                if fh.seekable() and not fragment_name(frag).endswith(".gz"):
                    fh.seek(0, io.SEEK_END)
                    size = fh.tell()
                    if size == 0:
                        return True                     # an empty file fails its header check instead
                    fh.seek(size - 1)
                    last = fh.read(1)
                else:
                    last = b""
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        last = chunk[-1:]
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise unreadable(frag, exc, "cannot be read") from exc
        if last not in (b"\n", b"\r"):
            raise unreadable(frag, None, "does not end with a newline: the file was cut short "
                             "(options.allow_unterminated accepts it)")
        return True

    def _key(self, frag: Fragment) -> tuple[Any, ...]:
        return (self.name, self._config_key(), *fragment_identity(frag))

    def header(self, frag: Fragment) -> _Header:
        """The header row (after any preamble), read once per fragment identity."""
        return _HEADERS.get(self._key(frag), lambda: self._read_header(frag))

    def _read_header(self, frag: Fragment) -> _Header:
        data = self._head(frag)
        text = data.decode(self.options.get("encoding") or "utf-8", errors="replace")
        lines = text.splitlines()
        skip = 0
        dims = None
        preamble: list[str] = []
        pre = self.options.get("preamble")
        if self._gct(frag):
            if len(lines) < GCT_PREAMBLE + 1 or not lines[0].startswith("#1."):
                raise unreadable(frag, None, "is not a GCT file (no '#1.x' version line)")
            parts = re.split(r"\s+", lines[1].strip())
            try:
                dims = (int(parts[0]), int(parts[1]))
            except (IndexError, ValueError) as exc:
                raise unreadable(frag, exc, "has no GCT dimension line") from exc
            preamble = lines[:GCT_PREAMBLE]
            skip = GCT_PREAMBLE
        elif isinstance(pre, int) and not isinstance(pre, bool):
            preamble = lines[:pre]
            skip = pre
        if len(lines) <= skip:
            raise unreadable(frag, None, "has no header row")
        delimiter = "\t" if self._gct(frag) else self._delimiter
        names = next(_csv.reader([lines[skip]], delimiter=delimiter))
        if not names or (len(names) == 1 and not names[0]):
            raise unreadable(frag, None, "has an empty header row")
        return _Header(names=names, skip=skip + 1, dims=dims, preamble=preamble)

    @staticmethod
    def unique_names(names: Sequence[str]) -> list[str]:
        """Header names made unique by position (``name#<position>`` for repeats and blanks)."""
        seen: dict[str, int] = {}
        for n in names:
            seen[n] = seen.get(n, 0) + 1
        return [n if n and seen[n] == 1 else f"{n}#{i}" for i, n in enumerate(names)]

    # -- reading ------------------------------------------------------------------------

    def _source(self, frag: Fragment) -> Any:
        pa = _arrow()
        from ..layouts.zip_member import local_path

        path = local_path(frag)
        if path is not None:
            try:
                return pa.input_stream(path, compression="detect")
            except (pa.ArrowException, OSError) as exc:
                raise unreadable(frag, exc, "cannot be opened") from exc
        return pa.PythonFile(open_bytes(frag), mode="r")

    def _types(self, frag: Fragment, hdr: _Header) -> dict[str, Any]:
        """Positional column name -> Arrow type: inferred from the first block (once per fragment identity),
        overridden by ``options.column_types`` (header names)."""
        return dict(_TYPES.get(self._key(frag), lambda: self._infer_types(frag, hdr)))

    def _infer_types(self, frag: Fragment, hdr: _Header) -> dict[str, Any]:
        pa = _arrow()
        import pyarrow.csv as pcsv

        n = len(hdr.names)
        positional = [f"c{i}" for i in range(n)]
        try:
            reader = pcsv.open_csv(self._source(frag), read_options=self._read_options(hdr, INFER_BYTES),
                                   parse_options=self._parse_options(frag),
                                   convert_options=self._convert_options(None, None))
            first = reader.read_next_batch()
            types = {f.name: f.type for f in first.schema}
        except StopIteration:
            types = {c: pa.string() for c in positional}
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise unreadable(frag, exc, "is not a readable CSV file") from exc
        for i, c in enumerate(positional):
            if pa.types.is_null(types.get(c, pa.null())):
                types[c] = pa.string()
        declared = self.options.get("column_types") or {}
        for i, name in enumerate(hdr.names):
            if name in declared:
                types[positional[i]] = pa.type_for_alias(str(declared[name]))
        return types

    def _read_options(self, hdr: _Header, block_size: int | None = None) -> Any:
        import pyarrow.csv as pcsv

        return pcsv.ReadOptions(skip_rows=hdr.skip, column_names=[f"c{i}" for i in range(len(hdr.names))],
                                block_size=int(block_size or INFER_BYTES), use_threads=False,
                                encoding=self.options.get("encoding") or "utf8")

    def _parse_options(self, frag: Fragment) -> Any:
        import pyarrow.csv as pcsv

        delimiter = "\t" if self._gct(frag) else self._delimiter
        return pcsv.ParseOptions(delimiter=delimiter, newlines_in_values=bool(self.options.get("newlines_in_values")))

    def _convert_options(self, types: Mapping[str, Any] | None, include: Sequence[str] | None) -> Any:
        import pyarrow.csv as pcsv

        nulls = list(self.options.get("null_values") or [""])
        kwargs: dict[str, Any] = {"strings_can_be_null": True, "quoted_strings_can_be_null": False,
                                  "null_values": nulls}
        if types is not None:
            kwargs["column_types"] = dict(types)
        if include is not None:
            kwargs["include_columns"] = list(include)
        return pcsv.ConvertOptions(**kwargs)

    def _batches(self, frag: Fragment, positions: Sequence[int] | None, *, block_size: int | None = None,
                 memory_pool: Any = None, types: Mapping[str, Any] | None = None) -> Iterator[Any]:
        """Record batches under positional names ``c<i>`` (only ``positions`` when given)."""
        pa = _arrow()
        import pyarrow.csv as pcsv

        hdr = self.header(frag)
        self._tail_ok(frag)
        all_types = dict(types) if types is not None else self._types(frag, hdr)
        include = None if positions is None else [f"c{i}" for i in positions]
        wanted = {c: all_types[c] for c in (include or all_types)}
        try:
            reader = pcsv.open_csv(self._source(frag), read_options=self._read_options(hdr, block_size),
                                   parse_options=self._parse_options(frag),
                                   convert_options=self._convert_options(wanted, include), memory_pool=memory_pool)
            for batch in reader:
                yield batch
                del batch
        except (pa.ArrowException, OSError, ValueError, UnicodeDecodeError) as exc:
            raise unreadable(frag, exc, "is not a readable CSV file") from exc

    def logical_schema(self, frag: Fragment) -> Any:
        pa = _arrow()
        hdr = self.header(frag)                             # an unreadable file raises here, never later
        self._tail_ok(frag)
        if self.matrix is not None:
            return long_schema(self.matrix, self.matrix.values[0])
        types = self._types(frag, hdr)
        names = self.unique_names(hdr.names)
        fields = [pa.field(n, types[f"c{i}"]) for i, n in enumerate(names)]
        from .parquet import _infer_type

        for key, value in (frag.partition or {}).items():
            if key not in names:
                fields.append(pa.field(key, _infer_type(value)))
        return pa.schema(fields)

    def leaf_path(self, path: str, schema: Any) -> str:
        return generic_leaf_path(path, schema)

    def stats(self, frag: Fragment) -> FragmentStats:
        return STATS_CACHE.get_or_scan(self._key(frag), lambda: self._scan_stats(frag))

    def _scan_stats(self, frag: Fragment) -> FragmentStats:
        hdr = self.header(frag)
        names = self.unique_names(hdr.names)
        rename = {f"c{i}": n for i, n in enumerate(names)}

        # blocks of many rows: with 1 MiB blocks a wide matrix (DepMap 24Q4 CRISPRGeneEffect.csv, 17,917 columns of
        # about 360 KB per line) decoded two rows per batch and the per-column statistics of 1,178 rows took 182 s;
        # 32 MiB blocks take 7.7 s at 674 MB peak (64 MiB: 5.9 s, 735 MB)
        block = self._block_size(frag, hdr, STATS_BATCH_ROWS, cap=STATS_BLOCK_BYTES)

        def renamed() -> Iterator[Any]:
            for b in self._batches(frag, None, block_size=block):
                yield b.rename_columns([rename[c] for c in b.schema.names])

        st = stats_from_batches(renamed())
        shape = None
        if self.matrix is not None:
            n_cols = len(self._col_positions(hdr))
            shape = (int(st.rows or 0), n_cols)
        return FragmentStats(rows=st.rows, row_groups=st.row_groups, columns=st.columns, method="scan", shape=shape)

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        hdr = self.header(frag)
        out = {"delimiter": "\t" if self._gct(frag) else self._delimiter, "columns": str(len(hdr.names))}
        if hdr.dims is not None:
            out.update({"gct_version": hdr.preamble[0].lstrip("#"), "gct_rows": str(hdr.dims[0]),
                        "gct_columns": str(hdr.dims[1])})
        return out

    def storage_type_of(self, schema: Any) -> Callable[[str], str | None]:
        return storage_type_of(schema)

    def compile(self, predicate: Predicate, schema: Any) -> tuple[Any, Predicate | None]:
        return residual_compile(predicate, schema)

    def _block_size(self, frag: Fragment, hdr: _Header, batch_rows: int, *, cap: int = INFER_BYTES * 16) -> int:
        """Bytes of ``batch_rows`` lines (estimated from the head), so a block decodes about one batch (at most
        ``cap`` bytes of rows)."""
        head = self._head(frag)
        lines = head.split(b"\n")
        header_bytes = sum(len(line) + 1 for line in lines[:hdr.skip])
        complete = lines[hdr.skip:-1]
        per_line = sum(len(line) + 1 for line in complete) // len(complete) if complete else \
            max(64, len(head) - header_bytes)
        # a block holds the skipped header lines plus about one batch of rows (never less than one row)
        return header_bytes + max(1 << 14, per_line, min(int(cap), per_line * max(1, int(batch_rows))))

    def scan(self, frags: Sequence[Fragment], *, columns: list[str] | None, predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None, memory_pool: Any = None) -> Iterator[Any]:
        """Batches of rows (the caller applies the residual); ``memory_pool`` receives the allocations."""
        frags = list(frags)
        if not frags:
            return iter(())
        if self.matrix is not None:
            return self._scan_matrix(frags, columns, batch_rows)
        return self._scan(frags, columns, predicate, batch_rows, memory_pool)

    def _scan(self, frags: list[Fragment], columns: list[str] | None, predicate: Predicate | None, batch_rows: int,
              memory_pool: Any) -> Iterator[Any]:
        pa = _arrow()
        for frag in frags:
            schema = self.logical_schema(frag)
            _, residual = self.compile(predicate, schema) if predicate is not None else (None, None)
            wanted = projection(columns, residual, schema.names)
            hdr = self.header(frag)
            names = self.unique_names(hdr.names)
            positions = [names.index(c) for c in wanted if c in names]
            types = {f"c{i}": schema.field(n).type for i, n in enumerate(names)}
            parts = {k: v for k, v in (frag.partition or {}).items() if k in wanted and k not in names}
            for batch in self._batches(frag, positions, block_size=self._block_size(frag, hdr, batch_rows),
                                       memory_pool=memory_pool, types=types):
                batch = batch.rename_columns([names[int(c[1:])] for c in batch.schema.names])
                for key, value in parts.items():
                    batch = batch.append_column(schema.field(key), pa.array([value] * batch.num_rows,
                                                                            schema.field(key).type))
                batch = batch.select([c for c in wanted if c in batch.schema.names])
                for start in range(0, batch.num_rows, max(1, int(batch_rows))):
                    yield batch.slice(start, int(batch_rows))
                del batch

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return native_rows(table)

    # -- matrix capability ------------------------------------------------------------

    def _require_matrix(self) -> MatrixLayout:
        if self.matrix is None:
            raise ValueError(f"{self.name}: configure(matrix=...) before reading a matrix")
        return self.matrix

    def _row_position(self, hdr: _Header) -> int:
        m = self._require_matrix()
        col = m.row.column
        if isinstance(col, int):
            return col
        names = [col] if col else []
        names += list(m.row.aliases) + list(m.row.key)
        for n in names:
            if n in hdr.names:
                return hdr.names.index(n)
        return 0 if not col else -1

    def _col_positions(self, hdr: _Header) -> list[int]:
        m = self._require_matrix()
        row = self._row_position(hdr)
        excluded = set(m.col.exclude)
        return [i for i, n in enumerate(hdr.names) if i != row and n not in excluded]

    def _parse_header(self, name: str) -> dict[str, str] | None:
        m = self._require_matrix()
        if not m.col.pattern:
            return {m.col.key[0] if m.col.key else "column": name}
        hit = re.fullmatch(m.col.pattern, name)
        return None if hit is None else {k: v for k, v in hit.groupdict().items() if v is not None}

    def _col_axis(self, frag: Fragment) -> tuple[list[int], list[dict[str, str]], list[str]]:
        """``(positions, parsed fields, unparsed headers)`` of the column axis, in header order."""
        hdr = self.header(frag)
        positions, parsed, bad = [], [], []
        for i in self._col_positions(hdr):
            fields = self._parse_header(hdr.names[i])
            if fields is None:
                bad.append(hdr.names[i])                    # never read as a value column
                continue
            positions.append(i)
            parsed.append(fields)
        return positions, parsed, bad

    def axis_values(self, frag: Fragment, axis: str) -> Any:
        """The axis members in file order, exposed under the declared key names, with their ``position``."""
        pa = _arrow()
        m = self._require_matrix()
        axis = {"obs": "row", "var": "col"}.get(axis, axis)
        if axis == "col":
            positions, parsed, bad = self._col_axis(frag)
            if bad and m.col.on_mismatch == "error":
                raise unreadable(frag, None, f"{len(bad)} header(s) do not match the column axis pattern "
                                 f"{m.col.pattern!r}: {', '.join(bad[:5])}")
            fields = m.col.fields
            data = {f: pa.array([p.get(f) for p in parsed], pa.string()) for f in fields}
            data["position"] = pa.array(positions, pa.int32())
            return pa.table(data)
        if axis != "row":
            raise ValueError(f"unknown axis {axis!r} (row/obs or col/var)")
        hdr = self.header(frag)
        pos = self._row_position(hdr)
        if pos < 0:
            raise unreadable(frag, None, f"no row-ID column {m.row.column!r} (aliases {list(m.row.aliases)})")
        values: list[Any] = []
        for batch in self._batches(frag, [pos], types={f"c{pos}": pa.string()}):
            values.extend(batch.column(0).to_pylist())
        key = m.row.key[0] if m.row.key else (m.row.index_name or "id")
        return pa.table({key: pa.array(values, pa.string()), "position": pa.array(range(len(values)), pa.int32())})

    def matrix_checks(self, frag: Fragment) -> list[CheckItem]:
        """Header-axis findings (R4/R5 for matrices): unparseable headers, duplicate headers and axis keys,
        all-empty columns and a GCT dimension line that disagrees with the data."""
        m = self._require_matrix()
        hdr = self.header(frag)
        items: list[CheckItem] = []
        positions, parsed, bad = self._col_axis(frag)
        if bad:
            level = "error" if m.col.on_mismatch == "error" else "warning"
            items.append(CheckItem("header_parse", False, f"{len(bad)} header(s) do not match {m.col.pattern!r}: "
                                   f"{', '.join(bad[:10])}", level=level))
        counts: dict[str, list[int]] = {}
        for i, n in enumerate(hdr.names):
            counts.setdefault(n, []).append(i)
        dup = {n: p for n, p in counts.items() if len(p) > 1}
        if dup:
            text = "; ".join(f"{n!r} at {p}" for n, p in list(dup.items())[:10])
            items.append(CheckItem("duplicate_header", False, f"duplicate header(s) {text}: values are read by "
                                   "position, the axis key must still be unique"))
        keys: dict[tuple[str, ...], list[int]] = {}
        for pos, p in zip(positions, parsed):
            keys.setdefault(tuple(p.get(k, "") for k in m.col.key), []).append(pos)
        dup_keys = {k: p for k, p in keys.items() if len(p) > 1}
        if dup_keys:
            text = "; ".join(f"{'/'.join(k)} at {p}" for k, p in list(dup_keys.items())[:10])
            items.append(CheckItem("duplicate_key", False, f"column axis key not unique: {text}"))
        st = self.stats(frag)
        names = self.unique_names(hdr.names)
        empty = [names[i] for i in positions
                 if st.rows and (cs := st.columns.get(names[i])) is not None and cs.null_count == st.rows]
        if empty:
            items.append(CheckItem("all_null_column", False, f"{len(empty)} column(s) without a value (inferred as "
                                   f"null): {', '.join(empty[:10])}", level="warning"))
        rows = self.axis_values(frag, "row").column(0).to_pylist()
        seen: set[Any] = set()
        dup_rows = sorted({r for r in rows if r in seen or seen.add(r)}, key=str)   # type: ignore[func-returns-value]
        if dup_rows:
            items.append(CheckItem("duplicate_key", False, f"row axis key not unique: {dup_rows[:10]}"))
        if None in rows:
            items.append(CheckItem("row_key_null", False, f"{rows.count(None)} row(s) without a row ID"))
        if hdr.dims is not None:
            n_cols = len(positions)
            ok = hdr.dims == (len(rows), n_cols)
            items.append(CheckItem("shape", ok, f"GCT dimension line says {hdr.dims[0]} x {hdr.dims[1]}, the data "
                                   f"holds {len(rows)} x {n_cols}" + ("" if ok else ": truncated or edited"),
                                   level="info" if ok else "error"))
        return items

    def slice(self, frag: Fragment, value: str, *, row_predicate: Predicate | None, col_keys: Sequence[Any] | None,
              budget_bytes: int | None, attributes: Sequence[str] = (), batch_rows: int = 1024) -> Iterator[Any]:
        """Long-view batches of ``value`` for the rows ``row_predicate`` keeps and the columns whose axis key
        is in ``col_keys`` (all when None), read by position. Raises :class:`SliceBudgetExceeded`."""
        pa = _arrow()
        import numpy as np

        m = self._require_matrix()
        hdr = self.header(frag)
        rpos = self._row_position(hdr)
        if rpos < 0:
            raise unreadable(frag, None, f"no row-ID column {m.row.column!r} (aliases {list(m.row.aliases)})")
        positions, parsed, _bad = self._col_axis(frag)
        want = None if col_keys is None else {str(k) for k in col_keys}
        keep = [(pos, p) for pos, p in zip(positions, parsed)
                if want is None or "\x1f".join(p.get(k, "") for k in m.col.key) in want]
        cols = {k: [p.get(k) for _pos, p in keep] for k in (*m.col.key, *[a for a in attributes
                                                                          if a in m.col.fields])}
        col_key_bytes = sum(len(str(v)) for v in cols[m.col.key[0]]) if m.col.key and keep else 0
        rkey = m.row.key[0] if m.row.key else "id"
        types = {f"c{rpos}": pa.string(), **{f"c{pos}": pa.float64() for pos, _p in keep}}
        emitted = 0
        where = fragment_name(frag)
        for batch in self._batches(frag, [rpos] + [pos for pos, _p in keep], types=types,
                                   block_size=self._block_size(frag, hdr, batch_rows)):
            ids = batch.column(f"c{rpos}").to_pylist()
            idx = row_filter(m, row_predicate, [{rkey: r} for r in ids])
            if not idx or not keep:
                continue
            block = np.column_stack([batch.column(f"c{pos}").to_numpy(zero_copy_only=False)[idx]
                                     for pos, _p in keep]) if keep else np.empty((len(idx), 0))
            row_ids = [ids[i] for i in idx]
            emitted = charge(emitted, block.size, len(keep) * sum(len(str(r)) for r in row_ids),
                             len(row_ids) * col_key_bytes, budget_bytes, where)
            yield long_batch(m, value, {rkey: row_ids}, cols, block,
                             col_attributes={a: cols[a] for a in attributes if a in cols and a not in m.col.key})

    def _scan_matrix(self, frags: list[Fragment], columns: list[str] | None, batch_rows: int) -> Iterator[Any]:
        m = self._require_matrix()
        for frag in frags:
            for batch in self.slice(frag, m.values[0], row_predicate=None, col_keys=None, budget_bytes=None,
                                    batch_rows=batch_rows):
                if columns:
                    batch = batch.select([c for c in columns if c in batch.schema.names])
                yield batch

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".csv", write=write_csv, scalars="text_scalars",
                           skip={"scalars": "CSV stores no float32, int32, timestamp or NaN-versus-null distinction "
                                            "(text_scalars covers the same rows)"},
                           write_matrix=write_matrix_csv, matrix_extension=".csv",
                           matrix_features=frozenset({"header", "duplicate_header", "all_empty", "truncation",
                                                      "shape_line", "wide"}))


@register
class TsvFormat(CsvFormat):
    name: ClassVar[str] = "tsv"
    delimiter: ClassVar[str] = "\t"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".tsv", write=write_tsv, scalars="text_scalars",
                           skip={"scalars": "TSV stores no float32, int32, timestamp or NaN-versus-null distinction "
                                            "(text_scalars covers the same rows)"},
                           write_matrix=write_gct, matrix_extension=".gct",
                           matrix_features=frozenset({"header", "duplicate_header", "all_empty", "truncation",
                                                      "shape_line", "gct"}))


# ---------------------------------------------------------------------------
# Writers (goldens)
# ---------------------------------------------------------------------------

def write_csv(table: Any, path: str, row_group_size: int | None = None, *, delimiter: str = ",") -> None:
    """Write a flat table as CSV (null as an empty cell, NaN as ``nan``, strings quoted)."""
    import os

    import pyarrow.csv as pcsv

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pcsv.write_csv(table, path, write_options=pcsv.WriteOptions(delimiter=delimiter, quoting_style="needed"))


def write_tsv(table: Any, path: str, row_group_size: int | None = None) -> None:
    write_csv(table, path, row_group_size, delimiter="\t")


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return "nan"
    return repr(float(v)) if isinstance(v, float) else str(v)


def write_matrix_csv(golden: Any, path: str, *, delimiter: str = ",") -> dict[str, Any]:
    """A header-axis matrix golden as a DepMap-style CSV: first column the row IDs (header
    ``golden.row_header``), one column per header label. Returns the ``configure`` arguments."""
    lines = [delimiter.join([golden.row_header, *golden.headers])]
    for rid, row in zip(golden.row_ids, golden.values):
        lines.append(delimiter.join([rid, *(_cell(v) for v in row)]))
    text = "\n".join(lines) + "\n"
    if golden.truncate_rows:
        text = "\n".join(lines[: 1 + len(golden.row_ids) - golden.truncate_rows]) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return {"options": {"delimiter": delimiter}, "matrix": golden.matrix_spec("header")}


def write_gct(golden: Any, path: str) -> dict[str, Any]:
    """A matrix golden as GCT 1.2 (``#1.2``, the dimension line, ``Name``/``Description`` columns); the
    dimension line always states the full shape, so a truncated golden disagrees with it."""
    n_rows = len(golden.row_ids)
    lines = ["#1.2", f"{n_rows}\t{len(golden.headers)}", "\t".join(["Name", "Description", *golden.headers])]
    for rid, row in zip(golden.row_ids, golden.values):
        lines.append("\t".join([rid, "na", *(_cell(v) for v in row)]))
    keep = len(lines) - golden.truncate_rows if golden.truncate_rows else len(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines[:keep]) + "\n")
    spec = golden.matrix_spec("header")
    spec["axes"]["row"]["column"] = "Name"
    spec["axes"]["row"]["aliases"] = []
    spec["axes"]["col"]["exclude"] = ["Name", "Description"]
    return {"options": {"preamble": "gct", "delimiter": "\t"}, "matrix": spec}

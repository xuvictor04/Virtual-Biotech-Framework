"""The ``jsonl`` format plugin: one JSON object per line (§9.4; phase 2, F12). pyarrow is imported inside
methods only.

Capabilities: ``tabular``, ``nested``, ``stats_scan``.

* Lines are parsed with :mod:`json` (``NaN`` and ``Infinity`` accepted as Python reads them; NaN is
  null at every level, I6) and converted with one Arrow schema per fragment, inferred from the whole
  file in the cached stats pass (``options.schema`` overrides it): a field that is ``null`` in the
  first rows and a list later is typed by its values, never by its first row.
* A JSON value never distinguishes ``int32``/``float32``/``large_*`` storage, so to_native rows
  equal a golden's whatever its storage widths; timestamps are strings (the ``scalars`` golden is
  written as ``text_scalars``).
* A line that is not a JSON object, a zero-length file or a file cut mid-line raises
  :class:`FormatError` naming the fragment (F-8, I14). Blank lines are skipped, so an intentionally
  empty table is a file holding one newline (what :func:`write_jsonl` writes), never zero bytes.
* Predicates are never pushed down (the reader evaluates the storage-typed residual).
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from typing import Any, Callable, ClassVar, Iterator, Mapping, Self, Sequence

from ...predicate import Predicate
from ..base import FormatError, Fragment, FragmentStats, PluginBase
from ..registry import register
from .csv import (
    STATS_CACHE,
    Memo,
    fragment_identity,
    generic_leaf_path,
    native_rows,
    open_bytes,
    projection,
    residual_compile,
    stats_from_batches,
    storage_type_of,
    unreadable,
)

__all__ = ["JsonlFormat", "write_jsonl"]

_SCHEMAS = Memo()


def _arrow() -> Any:
    import pyarrow

    return pyarrow


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None if math.isnan(value) else value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


@register
class JsonlFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "jsonl"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"tabular", "nested", "stats_scan"})
    requires: ClassVar[tuple[str, ...]] = ("pyarrow",)
    stats_cache = STATS_CACHE

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def configure(self, options: Mapping[str, Any] | None = None, matrix: Any = None) -> Self:
        """A configured copy: ``options.schema`` ({column: Arrow type alias}) fixes column types."""
        other = copy.copy(self)
        other.options = dict(options or {})
        return other

    def _key(self, frag: Fragment) -> tuple[Any, ...]:
        cfg = hashlib.sha256(json.dumps(self.options, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return (self.name, cfg, *fragment_identity(frag))

    # -- parsing ---------------------------------------------------------------------

    def _lines(self, frag: Fragment) -> Iterator[tuple[int, dict[str, Any]]]:
        """``(line number, object)`` for every non-blank line; FormatError on anything else."""
        try:
            fh = open_bytes(frag)
        except FormatError:
            raise
        with fh:
            text = io.TextIOWrapper(fh, encoding=self.options.get("encoding") or "utf-8", newline="")
            seen = False
            last = ""
            n = 0
            try:
                for n, line in enumerate(text, 1):
                    last = line
                    if not line.strip():
                        continue
                    if "\x00" in line:
                        raise unreadable(frag, None, f"line {n} holds binary data (NUL bytes)")
                    try:
                        obj = json.loads(line)
                    except ValueError as exc:
                        raise unreadable(frag, exc, f"line {n} is not JSON") from exc
                    if not isinstance(obj, dict):
                        raise unreadable(frag, None, f"line {n} is a JSON {type(obj).__name__}, not an object")
                    seen = True
                    yield n, _clean(obj)
            except UnicodeDecodeError as exc:
                raise unreadable(frag, exc, "is not UTF-8 text") from exc
            if not seen and n == 0:
                raise unreadable(frag, None, "is empty: not a JSON-lines file")
            if last and not last.endswith(("\n", "\r")) and not self.options.get("allow_unterminated"):
                raise unreadable(frag, None, "does not end with a newline: the file was cut short")

    def _infer(self, frag: Fragment) -> Any:
        pa = _arrow()
        declared = {k: pa.type_for_alias(str(v)) for k, v in (self.options.get("schema") or {}).items()}
        names: dict[str, None] = {}
        types: dict[str, Any] = {}
        for _n, obj in self._lines(frag):
            for k, v in obj.items():
                names.setdefault(k, None)
                if k in declared or v is None:
                    continue
                t = pa.array([v]).type
                prev = types.get(k)
                types[k] = t if prev is None else _unify(prev, t)
        fields = [pa.field(k, declared.get(k) or types.get(k) or pa.string()) for k in names]
        return pa.schema(fields)

    def _schema(self, frag: Fragment) -> Any:
        return _SCHEMAS.get(self._key(frag), lambda: self._infer(frag))

    # -- protocol --------------------------------------------------------------------

    def logical_schema(self, frag: Fragment) -> Any:
        pa = _arrow()
        schema = self._schema(frag)
        from .parquet import _infer_type

        for key, value in (frag.partition or {}).items():
            if key not in schema.names:
                schema = schema.append(pa.field(key, _infer_type(value)))
        return schema

    def leaf_path(self, path: str, schema: Any) -> str:
        return generic_leaf_path(path, schema)

    def stats(self, frag: Fragment) -> FragmentStats:
        return STATS_CACHE.get_or_scan(self._key(frag), lambda: stats_from_batches(
            self._batches(frag, None, 4096)))

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        return {"columns": str(len(self._schema(frag).names))}

    def storage_type_of(self, schema: Any) -> Callable[[str], str | None]:
        return storage_type_of(schema)

    def compile(self, predicate: Predicate, schema: Any) -> tuple[Any, Predicate | None]:
        return residual_compile(predicate, schema)

    def _batches(self, frag: Fragment, columns: Sequence[str] | None, batch_rows: int,
                 memory_pool: Any = None) -> Iterator[Any]:
        pa = _arrow()
        schema = self._schema(frag)
        if columns is not None:
            schema = pa.schema([schema.field(c) for c in columns if c in schema.names])
        rows: list[dict[str, Any]] = []
        line = 0

        def flush() -> Any:
            try:
                return pa.RecordBatch.from_pylist(rows, schema=schema)
            except (pa.ArrowException, TypeError, ValueError) as exc:
                raise unreadable(frag, exc, f"rows up to line {line} do not fit the inferred schema") from exc

        for line, obj in self._lines(frag):
            rows.append(obj)
            if len(rows) >= batch_rows:
                yield flush()
                rows = []
        if rows:
            yield flush()

    def scan(self, frags: Sequence[Fragment], *, columns: list[str] | None, predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None, memory_pool: Any = None) -> Iterator[Any]:
        return self._scan(list(frags), columns, predicate, max(1, int(batch_rows)))

    def _scan(self, frags: list[Fragment], columns: list[str] | None, predicate: Predicate | None,
              batch_rows: int) -> Iterator[Any]:
        pa = _arrow()
        for frag in frags:
            schema = self.logical_schema(frag)
            _, residual = self.compile(predicate, schema) if predicate is not None else (None, None)
            wanted = projection(columns, residual, schema.names)
            stored = [c for c in wanted if c in self._schema(frag).names]
            parts = {k: v for k, v in (frag.partition or {}).items() if k in wanted and k not in stored}
            for batch in self._batches(frag, stored, batch_rows):
                for key, value in parts.items():
                    f = schema.field(key)
                    batch = batch.append_column(f, pa.array([value] * batch.num_rows, f.type))
                yield batch.select([c for c in wanted if c in batch.schema.names])

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return native_rows(table)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".jsonl", write=write_jsonl, scalars="text_scalars",
                           skip={"scalars": "JSON has no timestamp type (text_scalars covers the same rows)"})


def _unify(a: Any, b: Any) -> Any:
    """The Arrow type holding values of types ``a`` and ``b`` (ints widen to double, structs merge fields)."""
    pa = _arrow()
    if a == b or pa.types.is_null(b):
        return a
    if pa.types.is_null(a):
        return b
    if pa.types.is_integer(a) and pa.types.is_floating(b) or pa.types.is_floating(a) and pa.types.is_integer(b):
        return pa.float64()
    if pa.types.is_list(a) and pa.types.is_list(b):
        return pa.list_(_unify(a.value_type, b.value_type))
    if pa.types.is_struct(a) and pa.types.is_struct(b):
        fields: dict[str, Any] = {f.name: f.type for f in a}
        for f in b:
            fields[f.name] = _unify(fields[f.name], f.type) if f.name in fields else f.type
        return pa.struct([(k, v) for k, v in fields.items()])
    raise FormatError(f"values of types {a} and {b} in one column")


def write_jsonl(table: Any, path: str, row_group_size: int | None = None) -> None:
    """Write a table as JSON lines (NaN as null; datetimes as ISO strings)."""
    import os

    from .parquet import _converter

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    convs = {f.name: c for f in table.schema if (c := _converter(f.type)) is not None}
    with open(path, "w", encoding="utf-8") as fh:
        if table.num_rows == 0:
            fh.write("\n")                                 # an empty table, never a zero-length file
        for row in table.to_pylist():
            for name, conv in convs.items():
                row[name] = conv(row.get(name))
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


"""Row-group metadata and row-group value indexes for ``access_paths`` (§6.3, §11.6, §14.6).

Two things the reader needs below the fragment level live here:

* :func:`read_footer` parses one fragment's footer into :class:`FooterInfo`: rows, and per row
  group and leaf (``path_in_schema``) the decoded bytes, value count, null count and min/max.
  The reader prunes row groups with it (``via: row_group_stats``), estimates scan bytes before
  reading, and the data child caches it per fingerprint (a multi-GB single file has ~62,600 row
  groups, so the footer is parsed once). Only formats whose fragments carry such footers have
  one; :func:`footer_reader` returns None for the others and the reader treats each fragment
  as one row group.
* A **sidecar index** (``via: sidecar_index``) maps each value of an indexed column to the
  ``(fragment, row group)`` pairs that hold it. It is built by streaming the indexed leaf one
  row group at a time (:func:`build_access_index`), stored as a sorted gzip TSV under
  ``<cache_dir>/<source>/<fingerprint>/access/<table>.<column>.idx`` and rebuilt only when the
  table's fingerprint changes. Values are keyed by their canonical rendering in the column's
  storage type (:func:`vbt.datalayer.rowkey.render_value`), fragments by their path relative to
  the table location, so moving a release keeps the index valid. A build over the scan budget
  fails with an error naming ``vbt ds index build --access-paths``.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..plugins.base import Fragment, FormatError
from ..rowkey import render_value

__all__ = [
    "ChunkInfo", "RowGroupInfo", "FooterInfo", "footer_reader", "read_footer", "AccessIndex", "access_index_path",
    "build_access_index", "load_access_index", "BUILD_HINT",
]

BUILD_HINT = "vbt ds index build --access-paths"
_HEADER = ("value", "fragment", "row_groups")


@dataclass(frozen=True)
class ChunkInfo:
    bytes: int
    num_values: int
    null_count: int | None
    min: Any = None
    max: Any = None
    has_minmax: bool = False


@dataclass(frozen=True)
class RowGroupInfo:
    rows: int
    chunks: Mapping[str, ChunkInfo]

    def leaf_bytes(self, leaf: str) -> int:
        """Decoded bytes of a leaf or of every leaf under a group prefix."""
        if leaf in self.chunks:
            return self.chunks[leaf].bytes
        prefix = leaf + "."
        return sum(c.bytes for p, c in self.chunks.items() if p.startswith(prefix))


@dataclass(frozen=True)
class FooterInfo:
    uri: str
    row_groups: tuple[RowGroupInfo, ...]
    leaves: tuple[str, ...] = ()

    @property
    def rows(self) -> int:
        return sum(rg.rows for rg in self.row_groups)

    def bytes(self, leaves: Iterable[str], row_groups: Iterable[int] | None = None) -> int:
        ids = range(len(self.row_groups)) if row_groups is None else row_groups
        wanted = list(leaves)
        return sum(self.row_groups[i].leaf_bytes(leaf) for i in ids for leaf in wanted)


def _decode(v: Any) -> Any:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else v


def read_footer(frag: Fragment) -> FooterInfo:
    """Parse one Parquet footer; an unreadable file raises :class:`FormatError` naming it and its partition."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = frag.uri[len("file://"):] if frag.uri.startswith("file://") else frag.uri
    try:
        md = pq.ParquetFile(path).metadata
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise FormatError(f"{path}: not a readable data file ({type(exc).__name__}: {exc})", fragment=frag.uri,
                          partition=frag.partition) from exc
    leaves = tuple(md.schema.column(i).path for i in range(md.num_columns))
    groups = []
    for r in range(md.num_row_groups):
        rg = md.row_group(r)
        chunks: dict[str, ChunkInfo] = {}
        for i, leaf in enumerate(leaves):
            cc = rg.column(i)
            st = cc.statistics
            has = bool(st is not None and st.has_min_max)
            try:
                mn, mx = (_decode(st.min), _decode(st.max)) if has else (None, None)
            except (pa.ArrowException, ValueError, TypeError):    # logical types without a Python min/max
                mn = mx = None
                has = False
            nulls = int(st.null_count) if st is not None and st.has_null_count else None
            chunks[leaf] = ChunkInfo(int(cc.total_uncompressed_size or 0), int(cc.num_values or 0), nulls, mn, mx, has)
        groups.append(RowGroupInfo(int(rg.num_rows), chunks))
    return FooterInfo(frag.uri, tuple(groups), leaves)


def footer_reader(fmt: Any) -> Callable[[Fragment], FooterInfo] | None:
    """The footer parser for a format plugin (``parquet``), None for formats without row-group footers."""
    if getattr(fmt, "name", None) == "parquet" and "leaf_projection" in (getattr(fmt, "capabilities", ()) or ()):
        return read_footer
    return None


# ---------------------------------------------------------------------------
# Row-group value indexes (access_paths: via sidecar_index)
# ---------------------------------------------------------------------------

@dataclass
class AccessIndex:
    """``value -> {(fragment relpath, row group)}`` for one indexed column of one table fingerprint."""

    path: Path
    column: str
    storage_type: str | None = None
    entries: dict[str, set[tuple[str, int]]] = field(default_factory=dict)

    def lookup(self, values: Iterable[Any]) -> set[tuple[str, int]]:
        out: set[tuple[str, int]] = set()
        for v in values:
            out |= self.entries.get(render_value(v, self.storage_type), set())
        return out

    def __len__(self) -> int:
        return len(self.entries)


class _TSV(csv.Dialect):
    delimiter = "\t"
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = "\n"
    quoting = csv.QUOTE_MINIMAL


def access_index_path(cache_dir: str | Path, source: str, fingerprint: str, table: str, column: str) -> Path:
    safe = fingerprint.replace(":", "_").replace(os.sep, "_")
    name = column.replace("/", "_").replace(os.sep, "_")
    return Path(cache_dir) / source / safe / "access" / f"{table}.{name}.idx"


def write_access_index(path: Path, column: str, storage_type: str | None,
                       entries: Mapping[str, Iterable[tuple[str, int]]]) -> int:
    """Write a sorted gzip TSV atomically; returns the number of distinct values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            w = csv.writer(text, dialect=_TSV)
            w.writerow(("#column", column, storage_type or ""))
            w.writerow(_HEADER)
            for value in sorted(entries):
                by_frag: dict[str, list[int]] = {}
                for frag, rg in sorted(set(entries[value])):
                    by_frag.setdefault(frag, []).append(rg)
                for frag, rgs in by_frag.items():
                    w.writerow((value, frag, ",".join(str(r) for r in rgs)))
            text.flush()
            text.detach()
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(entries)


def load_access_index(path: str | Path) -> AccessIndex:
    p = Path(path)
    with gzip.open(p, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, dialect=_TSV)
        meta = next(reader, None)
        header = next(reader, None)
        if not meta or meta[0] != "#column" or tuple(header or ()) != _HEADER:
            raise ValueError(f"{p} is not a row-group value index")
        idx = AccessIndex(p, meta[1], meta[2] or None)
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"{p}: short row {row!r}")
            value, frag, rgs = row
            bucket = idx.entries.setdefault(value, set())
            for r in rgs.split(","):
                if r:
                    bucket.add((frag, int(r)))
    return idx


def build_access_index(reader: Any, column: str, *, budget_bytes: int | None = None, force: bool = False
                       ) -> tuple[Path, int]:
    """Build (or reuse) the sidecar of ``column`` for the reader's current fingerprint: ``(path, rows)``.

    Streams the column's leaf one row group at a time (list columns index each element). Raises
    the reader's ``BudgetExceeded`` when the leaf's decoded bytes exceed ``budget_bytes``."""
    from .reader import BudgetExceeded

    fp = reader.fingerprint()
    t = reader.table
    path = access_index_path(reader.ctx.settings.cache_dir, t.physical.source, fp, t.physical.table, column)
    if path.exists() and not force:
        idx = reader.ctx.sidecars.get(str(path)) or load_access_index(path)
        reader.ctx.sidecars[str(path)] = idx
        return path, len(idx)
    leaf = reader.physical_path(column)
    need = reader.estimate_scan_bytes([leaf], None, use_sidecars=False)
    limit = budget_bytes if budget_bytes is not None else reader.ctx.scan_budget(t)
    if need > limit:
        raise BudgetExceeded(f"building the sidecar index of {t.physical}.{column} needs {need} decoded bytes, over "
                             f"the {limit}-byte budget; build it offline with `{BUILD_HINT}`",
                             need_bytes=need, limit_bytes=limit)
    storage = reader.storage_type(leaf)
    entries: dict[str, set[tuple[str, int]]] = {}
    for frag, rg, values in reader.leaf_values(leaf):
        name = reader.fragment_name(frag)
        for v in values:
            if v is None:
                continue
            entries.setdefault(render_value(v, storage), set()).add((name, rg))
    rows = write_access_index(path, column, storage, entries)
    reader.ctx.sidecars[str(path)] = load_access_index(path)
    return path, rows

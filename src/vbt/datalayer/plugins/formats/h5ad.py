"""The ``h5ad`` format plugin: AnnData files read backed, as the logical long view of §6.7 (phase 2, F12).
h5py, numpy and pyarrow are imported inside methods only.

Capabilities: ``matrix``, ``stats``. Requires ``h5py``, ``anndata`` and ``scipy`` (preflight probes
them; a missing module makes the tables that use the plugin ``plugin_unavailable``).

* **Backed reads.** The file is opened read-only with :mod:`h5py` (the ``backed='r'`` access of
  AnnData without loading ``X``): ``obs``/``var`` frames and the requested rows and columns of ``X``
  or ``layers/<name>`` are read, nothing else. Dense ``X`` is read in row chunks; CSR by row
  ranges of ``indptr``; CSC by column; gzip-compressed (chunked) datasets decompress transparently.
  Archive members and remote files are read through
  :func:`~vbt.datalayer.plugins.layouts.zip_member.open_fragment`.
* **Axes.** ``axis_values(frag, "row"|"col")`` returns the ``obs``/``var`` frame with the index exposed
  under the declared ``index_name`` (default: the frame's own ``_index`` name) and every column
  decoded: a categorical code ``-1`` is null while the category string ``'nan'`` stays a string;
  nullable integer and boolean arrays keep their mask. An axis keyed by a declared column (``from:
  column``) uses that column, never the index.
* **Positional indexes.** An index whose values are exactly ``"0" .. "n-1"`` carries no identity (a
  ``var_names`` lost on write). :meth:`H5adFormat.matrix_checks` reports ``positional_index``: an error
  when the axis key is that index, a warning when a declared key column is used instead.
* **Long view.** :meth:`H5adFormat.slice` returns row-major batches ``(<row key>, <col key>,
  <value>)``. Stored NaN is null. Cells a sparse matrix does not store follow ``implicit``:
  ``zero`` -> 0, ``none`` -> null, ``zero_if_measured`` -> 0 only where the ``measured(row, col)``
  callback says the feature was measured, else null (I6).
* ``stats`` reads only metadata: ``shape``, ``nnz`` and the value dtype; ``rows`` counts logical cells.

:class:`AnnDataStore` adapts h5py files and zarr groups to one interface, so the ``zarr`` plugin
reuses this module.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, ClassVar, Iterator, Mapping, Self, Sequence

from ...predicate import Predicate
from ..base import CheckItem, ColumnStats, FormatError, Fragment, FragmentStats, PluginBase
from ..registry import register
from .csv import (
    MatrixLayout,
    charge,
    fragment_name,
    long_batch,
    long_schema,
    matrix_layout,
    native_rows,
    residual_compile,
    row_filter,
    storage_type_of,
    unreadable,
)

__all__ = ["H5adFormat", "AnnDataStore", "is_positional", "write_h5ad", "write_matrix_anndata"]

_ROW_CHUNK = 1024


def is_positional(values: Sequence[Any]) -> bool:
    """True when an index is the positions ``"0" .. "n-1"`` (no identity)."""
    return bool(values) and all(str(v) == str(i) for i, v in enumerate(values))


def _decode(values: Any) -> list[Any]:
    out = []
    for v in values:
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        elif hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and v != v:
            v = None
        out.append(v)
    return out


class AnnDataStore:
    """The AnnData on-disk encoding (``encoding-type`` attributes) over an h5py file or a zarr group."""

    def __init__(self, root: Any, closer: Callable[[], None] | None = None) -> None:
        self.root = root
        self._closer = closer

    def close(self) -> None:
        if self._closer is not None:
            self._closer()

    def __enter__(self) -> "AnnDataStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- nodes ------------------------------------------------------------------------

    def has(self, path: str) -> bool:
        try:
            self.root[path]
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def node(self, path: str) -> Any:
        return self.root[path]

    def attrs(self, path: str) -> dict[str, Any]:
        node = self.root[path] if path else self.root
        return {k: (v.item() if hasattr(v, "item") and getattr(v, "shape", None) == () else v)
                for k, v in dict(node.attrs).items()}

    def encoding(self, path: str) -> str:
        enc = self.attrs(path).get("encoding-type")
        if isinstance(enc, bytes):
            enc = enc.decode()
        if enc:
            return str(enc)
        return "array" if not self.is_group(path) else "dict"

    def is_group(self, path: str) -> bool:
        node = self.root[path]
        return not hasattr(node, "dtype")

    def read(self, path: str, sel: Any = slice(None)) -> Any:
        import numpy as np

        node = self.root[path]
        data = node[sel]
        return np.asarray(data)

    # -- frames -----------------------------------------------------------------------

    def column(self, frame: str, name: str) -> list[Any]:
        """One decoded frame column (categorical code -1 -> None; 'nan' strings kept)."""
        path = f"{frame}/{name}"
        enc = self.encoding(path)
        if enc == "categorical":
            cats = _decode(self.read(f"{path}/categories"))
            codes = self.read(f"{path}/codes").tolist()
            return [cats[c] if c is not None and 0 <= c < len(cats) else None for c in codes]
        if enc in ("nullable-integer", "nullable-boolean", "nullable-string-array"):
            vals = _decode(self.read(f"{path}/values"))
            mask = self.read(f"{path}/mask").tolist()
            return [None if m else v for v, m in zip(vals, mask)]
        legacy = self.attrs(frame).get("__categories")
        values = _decode(self.read(path))
        if legacy is None and self.has(f"{frame}/__categories/{name}"):
            cats = _decode(self.read(f"{frame}/__categories/{name}"))
            return [cats[c] if c is not None and 0 <= c < len(cats) else None for c in values]
        return values

    def index(self, frame: str) -> tuple[str, list[Any]]:
        name = self.attrs(frame).get("_index") or "_index"
        name = name.decode() if isinstance(name, bytes) else str(name)
        return name, self.column(frame, name)

    def columns(self, frame: str) -> list[str]:
        order = self.attrs(frame).get("column-order")
        if order is None:
            return []
        if isinstance(order, (str, bytes)):
            order = [order]
        return [o.decode() if isinstance(o, bytes) else str(o) for o in list(order)]

    def shape(self, value: str) -> tuple[int, int]:
        path = self.value_path(value)
        enc = self.encoding(path)
        if enc in ("csr_matrix", "csc_matrix"):
            shape = self.attrs(path).get("shape")
            return int(shape[0]), int(shape[1])
        node = self.root[path]
        return int(node.shape[0]), int(node.shape[1])

    @staticmethod
    def value_path(value: str) -> str:
        if value in ("X", "value"):
            return "X"
        return value.replace(".", "/", 1) if value.startswith("layers.") else value


@register
class H5adFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "h5ad"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"matrix", "stats"})
    requires: ClassVar[tuple[str, ...]] = ("h5py", "anndata", "scipy")

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}
        self.matrix: MatrixLayout | None = None

    def configure(self, options: Mapping[str, Any] | None = None, matrix: Any = None) -> Self:
        """A configured copy: ``options`` (``backed: r`` accepted) and the table's ``MatrixSpec``."""
        other = copy.copy(self)
        other.options = dict(options or {})
        other.matrix = matrix_layout(matrix) if matrix is not None else None
        return other

    # -- opening -------------------------------------------------------------------------

    def open(self, frag: Fragment) -> AnnDataStore:
        try:
            import h5py
        except ImportError as exc:
            raise FormatError(f"{self.name} needs h5py (pip install h5py)", fragment=frag.uri) from exc
        from ..layouts.zip_member import local_path, open_fragment

        path = local_path(frag)
        try:
            if path is not None:
                fh = h5py.File(path, "r")
                return AnnDataStore(fh, fh.close)
            raw = open_fragment(frag)
            fh = h5py.File(raw, "r")

            def close() -> None:
                fh.close()
                raw.close()

            return AnnDataStore(fh, close)
        except (OSError, ValueError) as exc:
            raise unreadable(frag, exc, "is not a readable HDF5/AnnData file") from exc

    def _store(self, frag: Fragment) -> AnnDataStore:
        store = self.open(frag)
        if not store.has("obs") or not store.has("var") or not store.has("X"):
            store.close()
            raise unreadable(frag, None, "is not an AnnData file (no obs, var or X)")
        return store

    def _layout(self, store: AnnDataStore) -> MatrixLayout:
        if self.matrix is not None:
            return self.matrix
        row_name, _ = store.index("obs")
        col_name, _ = store.index("var")
        return matrix_layout({"axes": {"row": {"from": "index", "index_name": row_name,
                                               "key": {"columns": [row_name]}},
                                       "col": {"from": "index", "index_name": col_name,
                                               "key": {"columns": [col_name]}}},
                              "values": {"X": {}}})

    # -- axes ------------------------------------------------------------------------------

    def _frame(self, store: AnnDataStore, axis: str, layout: MatrixLayout) -> tuple[list[str], dict[str, list[Any]]]:
        """``(axis key values, decoded columns incl. the exposed index)`` of ``obs`` or ``var``."""
        frame = "obs" if axis == "row" else "var"
        ax = layout.row if axis == "row" else layout.col
        index_name, index = store.index(frame)
        exposed = ax.index_name or index_name
        cols: dict[str, list[Any]] = {exposed: [None if v is None else str(v) for v in index]}
        for name in store.columns(frame):
            if name not in cols:
                cols[name] = store.column(frame, name)
        if ax.source in ("column", "table") and ax.column is not None:
            key_col = str(ax.column)
            if key_col not in cols:
                raise FormatError(f"{frame} has no column {key_col!r} (the declared {ax.name} key)")
            keys = [None if v is None else str(v) for v in cols[key_col]]
        else:
            keys = cols[exposed]
        return keys, cols

    def axis_values(self, frag: Fragment, axis: str) -> Any:
        pa = _arrow()
        axis = {"obs": "row", "var": "col"}.get(axis, axis)
        if axis not in ("row", "col"):
            raise ValueError(f"unknown axis {axis!r} (row/obs or col/var)")
        with self._store(frag) as store:
            layout = self._layout(store)
            try:
                keys, cols = self._frame(store, axis, layout)
            except FormatError as exc:
                raise unreadable(frag, exc, "has no declared axis key") from exc
        ax = layout.row if axis == "row" else layout.col
        data: dict[str, Any] = {}
        for k in ax.key:
            data[k] = pa.array(keys, pa.string())
        for name, values in cols.items():
            if name in data:
                continue
            data[name] = _as_arrow(values)
        data["position"] = pa.array(range(len(keys)), pa.int32())
        return pa.table(data)

    # -- protocol ---------------------------------------------------------------------------

    def logical_schema(self, frag: Fragment) -> Any:
        with self._store(frag) as store:
            layout = self._layout(store)
        return long_schema(layout, layout.values[0])

    def leaf_path(self, path: str, schema: Any) -> str:
        if path.startswith(("@row.", "@col.", "@obs.", "@var.")):
            path = path.split(".", 1)[1]
        if path not in schema.names:
            raise ValueError(f"{path!r} is not a column of the long view")
        return path

    def stats(self, frag: Fragment) -> FragmentStats:
        with self._store(frag) as store:
            layout = self._layout(store)
            cols: dict[str, ColumnStats] = {}
            shape = None
            for value in layout.values:
                path = store.value_path(value)
                if not store.has(path):
                    continue
                enc = store.encoding(path)
                n_obs, n_var = store.shape(value)
                shape = shape or (n_obs, n_var)
                if enc in ("csr_matrix", "csc_matrix"):
                    data = store.node(f"{path}/data")
                    nnz = int(data.shape[0])
                    cols[value] = ColumnStats(uncompressed_bytes=nnz * (data.dtype.itemsize + 4), null_count=None,
                                              num_values=nnz, storage_type=str(data.dtype), kind="sparse_matrix")
                else:
                    node = store.node(path)
                    cols[value] = ColumnStats(uncompressed_bytes=n_obs * n_var * node.dtype.itemsize,
                                              null_count=None, num_values=n_obs * n_var,
                                              storage_type=str(node.dtype), kind="dense_matrix")
        if shape is None:
            raise unreadable(frag, None, f"holds none of the declared values {list(layout.values)}")
        return FragmentStats(rows=shape[0] * shape[1], row_groups=1, columns=cols, method="footer", shape=shape)

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        with self._store(frag) as store:
            n_obs, n_var = store.shape("X")
            out = {"n_obs": str(n_obs), "n_vars": str(n_var), "X_encoding": store.encoding("X"),
                   "encoding-version": str(store.attrs("").get("encoding-version", ""))}
            if store.has("layers"):
                out["layers"] = ",".join(sorted(store.node("layers").keys()))
        return out

    def storage_type_of(self, schema: Any) -> Callable[[str], str | None]:
        return storage_type_of(schema)

    def compile(self, predicate: Predicate, schema: Any) -> tuple[Any, Predicate | None]:
        return residual_compile(predicate, schema)

    def scan(self, frags: Sequence[Fragment], *, columns: list[str] | None, predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None, memory_pool: Any = None) -> Iterator[Any]:
        """The long view of the first declared value (every cell); the caller applies the residual."""
        return self._scan(list(frags), columns, batch_rows)

    def _scan(self, frags: list[Fragment], columns: list[str] | None, batch_rows: int) -> Iterator[Any]:
        for frag in frags:
            with self._store(frag) as store:
                value = self._layout(store).values[0]
            for batch in self.slice(frag, value, row_predicate=None, col_keys=None, budget_bytes=None,
                                    batch_rows=batch_rows):
                yield batch.select([c for c in columns if c in batch.schema.names]) if columns else batch

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return native_rows(table)

    # -- matrix capability -------------------------------------------------------------------

    def matrix_checks(self, frag: Fragment) -> list[CheckItem]:
        """Positional or ``-N``-uniquified indexes, duplicate axis keys and missing declared values."""
        items: list[CheckItem] = []
        with self._store(frag) as store:
            layout = self._layout(store)
            for axis, frame in (("row", "obs"), ("col", "var")):
                ax = layout.row if axis == "row" else layout.col
                _name, index = store.index(frame)
                keyed_by_index = not (ax.source in ("column", "table") and ax.column is not None)
                if is_positional(index):
                    items.append(CheckItem(
                        "positional_index", False,
                        f"{frame} index is positional ('0'..'{len(index) - 1}'): " + (
                            f"the declared {ax.name} key is that index" if keyed_by_index else
                            f"the declared key column {ax.column!r} is used instead"),
                        level="error" if keyed_by_index else "warning", column=f"@{axis}"))
                try:
                    keys, _cols = self._frame(store, axis, layout)
                except FormatError as exc:
                    items.append(CheckItem("axis_key", False, str(exc), column=f"@{axis}"))
                    continue
                seen: set[Any] = set()
                dup = sorted({k for k in keys if k in seen or seen.add(k)}, key=str)   # type: ignore[func-returns-value]
                if dup:
                    items.append(CheckItem("duplicate_key", False, f"{frame} key not unique: {dup[:10]}",
                                           column=f"@{axis}"))
                uniq = [k for k in keys if k is not None and re.fullmatch(r".+-\d+", str(k))
                        and str(k).rsplit("-", 1)[0] in seen]
                if uniq:
                    items.append(CheckItem("uniquified_index", False, f"{frame} keys look '-N' uniquified: "
                                           f"{uniq[:10]}", level="warning", column=f"@{axis}"))
                if None in keys:
                    items.append(CheckItem("null_key", False, f"{keys.count(None)} {frame} key(s) are null",
                                           column=f"@{axis}"))
            for value in layout.values:
                if not store.has(store.value_path(value)):
                    items.append(CheckItem("value_missing", False, f"declared value {value!r} is not in the file"))
        return items

    def slice(self, frag: Fragment, value: str, *, row_predicate: Predicate | None, col_keys: Sequence[Any] | None,
              budget_bytes: int | None, attributes: Sequence[str] = (), batch_rows: int = _ROW_CHUNK,
              measured: Callable[[str, str], bool] | None = None) -> Iterator[Any]:
        """Long-view batches of ``value`` (``X`` or ``layers.<name>``) for the rows ``row_predicate`` keeps and
        the columns whose key is in ``col_keys``. Raises :class:`~.csv.SliceBudgetExceeded`."""
        import numpy as np

        with self._store(frag) as store:
            layout = self._layout(store)
            path = store.value_path(value)
            if not store.has(path):
                raise unreadable(frag, None, f"has no value {value!r}")
            row_keys, row_cols = self._frame(store, "row", layout)
            col_keys_all, col_cols = self._frame(store, "col", layout)
            rkey, ckey = layout.row.key[0], layout.col.key[0]
            axis_rows = [{**{n: v[i] for n, v in row_cols.items()}, rkey: row_keys[i]} for i in range(len(row_keys))]
            rows = row_filter(layout, row_predicate, axis_rows)
            want = None if col_keys is None else {str(k) for k in col_keys}
            cols = [j for j, k in enumerate(col_keys_all) if want is None or k in want]
            if not rows or not cols:
                return
            enc = store.encoding(path)
            row_attrs = [a for a in attributes if a in row_cols and a != rkey]
            col_attrs = [a for a in attributes if a in col_cols and a != ckey]
            col_bytes = sum(len(str(col_keys_all[j])) for j in cols)
            emitted = 0
            where = fragment_name(frag)
            step = max(1, int(batch_rows))
            reader = _CsrReader(store, path) if enc == "csr_matrix" else (
                _CscReader(store, path, cols) if enc == "csc_matrix" else None)
            for start in range(0, len(rows), step):
                chunk = rows[start:start + step]
                if reader is None:
                    block = store.read(path, (np.asarray(chunk),))[:, cols].astype("float64")
                    absent = None
                else:
                    block, absent = reader.block(chunk, cols)
                    if layout.implicit == "zero":
                        absent = None
                    elif layout.implicit == "zero_if_measured" and measured is not None:
                        rk = [row_keys[i] for i in chunk]
                        ck = [col_keys_all[j] for j in cols]
                        meas = np.array([[measured(r, c) for c in ck] for r in rk], dtype=bool)
                        absent = absent & ~meas
                ids = [row_keys[i] for i in chunk]
                emitted = charge(emitted, block.size, len(cols) * sum(len(str(r)) for r in ids),
                                 len(ids) * col_bytes, budget_bytes, where)
                yield long_batch(layout, value, {rkey: ids}, {ckey: [col_keys_all[j] for j in cols]}, block,
                                 row_attributes={a: [row_cols[a][i] for i in chunk] for a in row_attrs},
                                 col_attributes={a: [col_cols[a][j] for j in cols] for a in col_attrs},
                                 missing=absent)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".h5ad", write=write_h5ad, write_matrix=write_matrix_anndata,
                           matrix_extension=".h5ad",
                           matrix_features=frozenset({"sparse", "gzip", "categorical", "positional_index"}))


class _CsrReader:
    """Rows of a CSR matrix: ``indptr`` read once, ``data``/``indices`` per contiguous row range."""

    def __init__(self, store: AnnDataStore, path: str) -> None:
        self.store = store
        self.path = path
        self.indptr = store.read(f"{path}/indptr")

    def block(self, rows: Sequence[int], cols: Sequence[int]) -> tuple[Any, Any]:
        import numpy as np

        pos = {c: i for i, c in enumerate(cols)}
        out = np.zeros((len(rows), len(cols)), dtype="float64")
        absent = np.ones((len(rows), len(cols)), dtype=bool)
        for r_i, r in enumerate(rows):
            lo, hi = int(self.indptr[r]), int(self.indptr[r + 1])
            if hi <= lo:
                continue
            idx = self.store.read(f"{self.path}/indices", slice(lo, hi))
            data = self.store.read(f"{self.path}/data", slice(lo, hi))
            for c, v in zip(idx.tolist(), data.tolist()):
                j = pos.get(c)
                if j is not None:
                    out[r_i, j] = v
                    absent[r_i, j] = False
        return out, absent


class _CscReader:
    """Columns of a CSC matrix, read once for the selected columns."""

    def __init__(self, store: AnnDataStore, path: str, cols: Sequence[int]) -> None:
        import numpy as np

        indptr = store.read(f"{path}/indptr")
        self.cols: dict[int, tuple[Any, Any]] = {}
        for c in cols:
            lo, hi = int(indptr[c]), int(indptr[c + 1])
            self.cols[c] = (store.read(f"{path}/indices", slice(lo, hi)) if hi > lo else np.empty(0, int),
                            store.read(f"{path}/data", slice(lo, hi)) if hi > lo else np.empty(0))

    def block(self, rows: Sequence[int], cols: Sequence[int]) -> tuple[Any, Any]:
        import numpy as np

        pos = {r: i for i, r in enumerate(rows)}
        out = np.zeros((len(rows), len(cols)), dtype="float64")
        absent = np.ones((len(rows), len(cols)), dtype=bool)
        for j, c in enumerate(cols):
            idx, data = self.cols[c]
            for r, v in zip(idx.tolist(), data.tolist()):
                i = pos.get(r)
                if i is not None:
                    out[i, j] = v
                    absent[i, j] = False
        return out, absent


def _arrow() -> Any:
    import pyarrow

    return pyarrow


def _as_arrow(values: Sequence[Any]) -> Any:
    """A frame column as Arrow: numbers stay numeric, everything else is a string column."""
    pa = _arrow()
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, bool) for v in present):
        return pa.array(values, pa.bool_())
    if present and all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return pa.array(values, pa.int64())
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return pa.array([None if v is None else float(v) for v in values], pa.float64())
    return pa.array([None if v is None else str(v) for v in values], pa.string())


# ---------------------------------------------------------------------------
# Writers (goldens)
# ---------------------------------------------------------------------------

def write_h5ad(table: Any, path: str, row_group_size: int | None = None) -> None:  # pragma: no cover - unused
    raise NotImplementedError("h5ad holds matrices: use write_matrix_anndata")


def anndata_of(golden: Any) -> Any:
    """An AnnData object of a :class:`~vbt.datalayer.plugins.conformance.golden.MatrixGolden`."""
    import anndata
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    rows = list(golden.written_rows)
    dense = np.array([[np.nan if v is None else v for v in golden.values[i]] for i in range(len(rows))],
                     dtype="float64")
    obs = pd.DataFrame(index=pd.Index(rows, name="sample_id"))
    for name, values in (golden.obs or {}).items():
        vals = list(values)[: len(rows)]
        obs[name] = pd.Categorical(vals) if name in golden.categorical else pd.array(vals, dtype="string")
    if golden.positional_var:
        var = pd.DataFrame({"feature_id": list(golden.col_ids), "symbol": list(golden.symbols)},
                           index=pd.Index([str(i) for i in range(len(golden.col_ids))]))
    else:
        var = pd.DataFrame({"symbol": list(golden.symbols)}, index=pd.Index(list(golden.col_ids), name="entrez_id"))
    if golden.storage == "csr":
        x = sp.csr_matrix(np.nan_to_num(dense, nan=0.0))
        x.eliminate_zeros()
    elif golden.storage == "csc":
        x = sp.csc_matrix(np.nan_to_num(dense, nan=0.0))
        x.eliminate_zeros()
    else:
        x = dense
    return anndata.AnnData(X=x, obs=obs, var=var)


def write_matrix_anndata(golden: Any, path: str) -> dict[str, Any]:
    """Write a matrix golden as ``.h5ad`` (gzip-compressed when the golden asks for it); returns the
    ``configure`` arguments."""
    import warnings

    adata = anndata_of(golden)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata.write_h5ad(path, compression="gzip" if golden.gzip else None)
    return {"options": {"backed": "r"}, "matrix": golden.matrix_spec("index")}

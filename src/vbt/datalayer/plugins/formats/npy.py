"""The ``npy`` and ``safetensors`` format plugins: embedding matrices with a vocabulary file (§6.7, §9.4;
phase 2, F12). numpy and pyarrow are imported inside methods only.

Capabilities: ``tabular``, ``vectors``.

An embedding file holds one 2-D float array (``n x dim``); its row identities come from a
**vocabulary file** (``AxisSpec.ids_from``): one ID per line, in row order. The logical schema is
``id`` (string) and ``vector`` (``fixed_size_list<float32|float64, dim>``), so a ``vectors`` table
(``key: [id]``, ``vector`` role with ``dim``) reads it like the Parquet ``literature_vector``.

* ``ids_from``: ``options.ids_from`` (relative to the embedding file, or absolute), else
  ``<stem>.ids.txt`` next to the file. The vocabulary must have exactly ``n`` non-empty lines and
  no duplicate; otherwise :class:`FormatError` (a misaligned vocabulary would attach every vector to
  the wrong ID).
* ``npy``: NumPy ``.npy`` (memory-mapped for local files); ``safetensors``: the safetensors layout
  (an 8-byte little-endian header length, a JSON header ``{name: {dtype, shape, data_offsets}}``,
  then the raw little-endian tensors) parsed directly, without the ``safetensors`` package;
  ``options.tensor`` names the tensor (default: the only 2-D tensor).
* A zero-length, truncated (data shorter than the header says) or corrupt file raises
  :class:`FormatError` naming the fragment (I14). NaN elements are null.
"""

from __future__ import annotations

import copy
import io
import json
import os
import struct
from typing import Any, Callable, ClassVar, Iterator, Mapping, Self, Sequence

from ...predicate import Predicate
from ..base import ColumnStats, FormatError, Fragment, FragmentStats, PluginBase
from ..registry import register
from .csv import (
    Memo,
    fragment_identity,
    generic_leaf_path,
    matrix_layout,
    native_rows,
    open_bytes,
    projection,
    residual_compile,
    storage_type_of,
    unreadable,
)

__all__ = ["NpyFormat", "SafetensorsFormat", "ids_uri", "write_npy", "write_safetensors",
           "embedding_projection_golden"]

_ST_DTYPES = {"F32": "<f4", "F64": "<f8", "F16": "<f2", "BF16": None}
_ARRAYS = Memo(64)


def ids_uri(frag: Fragment, ids_from: str | None) -> str:
    """The vocabulary file of an embedding fragment (same scheme: a sibling path, archive member or URL)."""
    uri = frag.uri
    base, _sep, name = uri.rpartition("/")
    if ids_from:
        if os.path.isabs(ids_from) or ids_from.startswith(("http://", "https://", "zip://")):
            return ids_from
        return f"{base}/{ids_from}" if base else ids_from
    stem = name
    for suffix in (".npy", ".safetensors"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return f"{base}/{stem}.ids.txt" if base else f"{stem}.ids.txt"


class _EmbeddingFormat(PluginBase):
    kind: ClassVar[str] = "format"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"tabular", "vectors"})
    requires: ClassVar[tuple[str, ...]] = ("numpy", "pyarrow")

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def configure(self, options: Mapping[str, Any] | None = None, matrix: Any = None) -> Self:
        """A configured copy: ``options.ids_from`` (or the matrix row axis's ``ids_from``), ``options.tensor``."""
        other = copy.copy(self)
        other.options = dict(options or {})
        if matrix is not None and "ids_from" not in other.options:
            ids = matrix_layout(matrix).row.ids_from
            if ids:
                other.options["ids_from"] = ids
        return other

    # -- reading (subclasses implement _array) -----------------------------------------------

    def _array(self, frag: Fragment) -> Any:
        raise NotImplementedError

    def _load(self, frag: Fragment) -> tuple[Any, list[str]]:
        key = (self.name, json.dumps(self.options, sort_keys=True, default=str), *fragment_identity(frag))
        return _ARRAYS.get(key, lambda: self._load_now(frag))

    def _load_now(self, frag: Fragment) -> tuple[Any, list[str]]:
        arr = self._array(frag)
        if arr.ndim != 2:
            raise unreadable(frag, None, f"holds a {arr.ndim}-D array, not an (n x dim) embedding matrix")
        ids = self._ids(frag, arr.shape[0])
        return arr, ids

    def _ids(self, frag: Fragment, n: int) -> list[str]:
        uri = ids_uri(frag, self.options.get("ids_from"))
        try:
            fh = open_bytes(Fragment(uri=uri, size=None, mtime_ns=None))
            with fh:
                text = fh.read().decode("utf-8")
        except (FormatError, OSError, UnicodeDecodeError) as exc:
            raise unreadable(frag, exc, f"has no readable vocabulary file {uri}") from exc
        ids = [line.strip() for line in text.splitlines() if line.strip()]
        if len(ids) != n:
            raise unreadable(frag, None, f"has {n} vectors but its vocabulary {uri} lists {len(ids)} IDs")
        if len(set(ids)) != len(ids):
            raise unreadable(frag, None, f"vocabulary {uri} repeats an ID")
        return ids

    def _schema(self, arr: Any) -> Any:
        import pyarrow as pa

        elem = pa.float32() if str(arr.dtype) in ("float32", "float16") else pa.float64()
        return pa.schema([("id", pa.string()), ("vector", pa.list_(pa.field("element", elem), int(arr.shape[1])))])

    # -- protocol ----------------------------------------------------------------------------

    def logical_schema(self, frag: Fragment) -> Any:
        arr, _ids = self._load(frag)
        return self._schema(arr)

    def leaf_path(self, path: str, schema: Any) -> str:
        return generic_leaf_path(path, schema)

    def stats(self, frag: Fragment) -> FragmentStats:
        arr, ids = self._load(frag)
        n, dim = int(arr.shape[0]), int(arr.shape[1])
        cols = {"id": ColumnStats(uncompressed_bytes=sum(len(i) for i in ids), null_count=0, num_values=n,
                                  storage_type="string", kind="string"),
                "vector": ColumnStats(uncompressed_bytes=n * dim * arr.dtype.itemsize, null_count=0,
                                      num_values=n * dim, storage_type=f"fixed_size_list<{arr.dtype}>[{dim}]",
                                      kind="nested")}
        return FragmentStats(rows=n, row_groups=1, columns=cols, method="footer", shape=(n, dim))

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        arr, _ids = self._load(frag)
        return {"rows": str(arr.shape[0]), "dim": str(arr.shape[1]), "dtype": str(arr.dtype),
                "ids_from": ids_uri(frag, self.options.get("ids_from"))}

    def storage_type_of(self, schema: Any) -> Callable[[str], str | None]:
        return storage_type_of(schema)

    def compile(self, predicate: Predicate, schema: Any) -> tuple[Any, Predicate | None]:
        return residual_compile(predicate, schema)

    def scan(self, frags: Sequence[Fragment], *, columns: list[str] | None, predicate: Predicate | None,
             partitions: Mapping[str, str], batch_rows: int = 1024,
             row_groups: Mapping[str, Sequence[int]] | None = None, memory_pool: Any = None) -> Iterator[Any]:
        return self._scan(list(frags), columns, predicate, max(1, int(batch_rows)))

    def _scan(self, frags: list[Fragment], columns: list[str] | None, predicate: Predicate | None,
              batch_rows: int) -> Iterator[Any]:
        import numpy as np
        import pyarrow as pa

        for frag in frags:
            arr, ids = self._load(frag)
            schema = self._schema(arr)
            _, residual = self.compile(predicate, schema) if predicate is not None else (None, None)
            wanted = projection(columns, residual, schema.names)
            vtype = schema.field("vector").type
            for start in range(0, len(ids), batch_rows):
                stop = min(len(ids), start + batch_rows)
                data: dict[str, Any] = {}
                if "id" in wanted:
                    data["id"] = pa.array(ids[start:stop], pa.string())
                if "vector" in wanted:
                    block = np.ascontiguousarray(arr[start:stop], dtype=vtype.value_type.to_pandas_dtype())
                    flat = block.reshape(-1)
                    values = pa.array(flat, vtype.value_type, mask=np.isnan(flat))
                    data["vector"] = pa.FixedSizeListArray.from_arrays(values, vtype.list_size)
                yield pa.RecordBatch.from_pydict(data, schema=pa.schema([schema.field(c) for c in wanted]))

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return native_rows(table)


@register
class NpyFormat(_EmbeddingFormat):
    name: ClassVar[str] = "npy"

    def _array(self, frag: Fragment) -> Any:
        import numpy as np

        from ..layouts.zip_member import local_path

        path = local_path(frag)
        try:
            if path is not None:
                if os.path.getsize(path) == 0:
                    raise unreadable(frag, None, "is empty: not a .npy file")
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
            else:
                with open_bytes(frag) as fh:
                    arr = np.load(io.BytesIO(fh.read()), allow_pickle=False)
        except FormatError:
            raise
        except (OSError, ValueError, EOFError) as exc:
            raise unreadable(frag, exc, "is not a readable .npy array") from exc
        if arr.dtype.kind != "f":
            raise unreadable(frag, None, f"holds {arr.dtype} values, not floats")
        return arr

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".npy", write=write_npy, projection=embedding_projection_golden)


@register
class SafetensorsFormat(_EmbeddingFormat):
    name: ClassVar[str] = "safetensors"

    def _array(self, frag: Fragment) -> Any:
        import numpy as np

        try:
            with open_bytes(frag) as fh:
                data = fh.read()
        except OSError as exc:
            raise unreadable(frag, exc, "cannot be read") from exc
        if len(data) < 8:
            raise unreadable(frag, None, "is too short for a safetensors header")
        (n,) = struct.unpack("<Q", data[:8])
        if n <= 0 or 8 + n > len(data):
            raise unreadable(frag, None, "has no complete safetensors header (truncated or corrupt)")
        try:
            header = json.loads(data[8:8 + n].decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise unreadable(frag, exc, "has an unreadable safetensors header") from exc
        tensors = {k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}
        name = self.options.get("tensor") or next((k for k, v in tensors.items() if len(v.get("shape", ())) == 2),
                                                  None)
        if name is None or name not in tensors:
            raise unreadable(frag, None, f"has no 2-D tensor {name or ''} (tensors: {sorted(tensors)})")
        t = tensors[name]
        dtype = _ST_DTYPES.get(str(t.get("dtype")))
        if dtype is None:
            raise unreadable(frag, None, f"tensor {name} has unsupported dtype {t.get('dtype')}")
        lo, hi = (int(x) for x in t["data_offsets"])
        start = 8 + n
        if start + hi > len(data):
            raise unreadable(frag, None, f"tensor {name} ends past the end of the file (truncated)")
        shape = tuple(int(x) for x in t["shape"])
        buf = data[start + lo:start + hi]
        if len(buf) != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise unreadable(frag, None, f"tensor {name} has {len(buf)} bytes for shape {shape}")
        return np.frombuffer(buf, dtype=dtype).reshape(shape)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        # Truncation inside the tensor data is detected; a cut that keeps the header and data intact is
        # impossible (the header gives exact offsets).
        return FormatCases(extension=".safetensors", write=write_safetensors, projection=embedding_projection_golden)


# ---------------------------------------------------------------------------
# Writers (goldens)
# ---------------------------------------------------------------------------

def _write_ids(path: str, ids: Sequence[str]) -> None:
    base = path
    for suffix in (".npy", ".safetensors"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    with open(base + ".ids.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(ids) + ("\n" if ids else ""))


def _matrix(table: Any) -> tuple[list[str], Any]:
    import numpy as np

    vtype = table.schema.field("vector").type
    dim = vtype.list_size
    ids = table.column("id").to_pylist()
    values = table.column("vector").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    dtype = "float32" if str(vtype.value_type) == "float" else "float64"
    return ids, np.asarray(values, dtype=dtype).reshape(len(ids), dim)


def write_npy(table: Any, path: str, row_group_size: int | None = None) -> None:
    import numpy as np

    ids, arr = _matrix(table)
    with open(path, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    _write_ids(path, ids)


def write_safetensors(table: Any, path: str, row_group_size: int | None = None) -> None:
    ids, arr = _matrix(table)
    dtype = "F32" if str(arr.dtype) == "float32" else "F64"
    raw = arr.astype("<f4" if dtype == "F32" else "<f8").tobytes()
    header = json.dumps({"embeddings": {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [0, len(raw)]},
                         "__metadata__": {"format": "pt"}}).encode()
    header += b" " * (-len(header) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)) + header + raw)
    _write_ids(path, ids)


def embedding_projection_golden() -> Any:
    """Five 4-d float32 embeddings of mixed entity IDs (one with a NaN element)."""
    import pyarrow as pa

    ids = ["ENSG00000169174", "CHEMBL25", "MONDO_0005148", "EFO_0000685", "HP_0001250"]
    vectors = [[0.5, -0.25, 1.0, 0.0], [0.125, 0.5, -1.5, 2.0], [1.0, 1.0, 1.0, 1.0], [0.0, float("nan"), 0.75, -2.0],
               [3.0, -0.5, 0.25, 0.5]]
    vtype = pa.list_(pa.field("element", pa.float32()), 4)
    return pa.table({"id": pa.array(ids, pa.string()), "vector": pa.array(vectors, vtype)})


"""The ``zarr`` format plugin: AnnData stores in Zarr (``*.zarr`` directories or ``*.zarr.zip`` archives)
read as the logical long view of §6.7 (phase 2, F12). zarr and numpy are imported inside methods only.

Capabilities: ``matrix``, ``stats``. Requires ``zarr`` (and ``anndata`` for the conformance writer).

The on-disk encoding is AnnData's (``encoding-type`` attributes on ``obs``, ``var``, ``X``,
``layers``), so everything but opening is :class:`~vbt.datalayer.plugins.formats.h5ad.H5adFormat`:
axes with the index exposed under ``index_name``, categorical code -1 as null and the string
``'nan'`` kept, positional-index detection, dense and CSR/CSC reads with the ``implicit`` policy,
and the long-view ``slice``. A store is opened read-only (``mode="r"``); a ``.zip`` store through a
read-only zip store. A directory that is not a Zarr group, or an archive that is not a zip, raises
:class:`~vbt.datalayer.plugins.base.FormatError`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import FormatError, Fragment
from ..registry import register
from .csv import unreadable
from .h5ad import AnnDataStore, H5adFormat, anndata_of

__all__ = ["ZarrFormat", "write_matrix_zarr"]


@register
class ZarrFormat(H5adFormat):
    name: ClassVar[str] = "zarr"
    version: ClassVar[str] = "1.0"
    requires: ClassVar[tuple[str, ...]] = ("zarr",)

    def open(self, frag: Fragment) -> AnnDataStore:
        try:
            import zarr
        except ImportError as exc:
            raise FormatError("the zarr format needs zarr (pip install zarr)", fragment=frag.uri) from exc
        from ..layouts.zip_member import local_path

        path = local_path(frag)
        if path is None:
            raise unreadable(frag, None, "zarr stores are read from local paths (directories or .zarr.zip)")
        try:
            if path.endswith(".zip"):
                from zarr.storage import ZipStore

                store = ZipStore(path, mode="r")
                group = zarr.open_group(store=store, mode="r")
                return AnnDataStore(group, store.close)
            group = zarr.open_group(store=path, mode="r")
            return AnnDataStore(group)
        except Exception as exc:  # noqa: BLE001 - zarr raises many kinds for a store that is not one
            raise unreadable(frag, exc, "is not a readable Zarr store") from exc

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".zarr", write=_write_table, write_matrix=write_matrix_zarr,
                           matrix_extension=".zarr",
                           matrix_features=frozenset({"sparse", "categorical", "positional_index"}))


def _write_table(table: Any, path: str, row_group_size: int | None = None) -> None:  # pragma: no cover - unused
    raise NotImplementedError("zarr holds matrices: use write_matrix_zarr")


def write_matrix_zarr(golden: Any, path: str) -> dict[str, Any]:
    """Write a matrix golden as an AnnData Zarr store; returns the ``configure`` arguments."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anndata_of(golden).write_zarr(path)
    return {"options": {}, "matrix": golden.matrix_spec("index")}

"""The ``gmt`` format plugin: gene-set collections (MSigDB GMT) as the published projection (§6.8 S9; phase 2,
F12). pyarrow is imported inside methods only.

Capabilities: ``tabular``, ``nested``.

A GMT line is ``<set name> TAB <description> TAB <member> TAB <member> ...``. The logical schema is
fixed: ``set_id`` (string), ``description`` (string; MSigDB puts the set's URL there) and
``members`` (``list<string>``, in file order, duplicates kept as stored). A set with no members has
``members: []``. Blank lines are skipped; a line with fewer than two fields, binary content, a
zero-length file or a file cut mid-line raises :class:`FormatError` (I14).

**Fragment key.** A GMT file is one collection; its fragment key defaults to the file stem
(``h.all.v2024.1.Hs.symbols`` from ``h.all.v2024.1.Hs.symbols.gmt``) unless the table's
``fragment_key`` names one: :func:`collection_of` and ``metadata()["fragment_key"]``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, ClassVar, Iterator, Mapping, Sequence

from ...predicate import Predicate
from ..base import Fragment, FragmentStats, PluginBase
from ..registry import register
from .csv import (
    Memo,
    fragment_identity,
    fragment_name,
    generic_leaf_path,
    native_rows,
    open_bytes,
    projection,
    residual_compile,
    stats_from_batches,
    storage_type_of,
    unreadable,
)

__all__ = ["GmtFormat", "gmt_schema", "collection_of", "write_gmt", "gmt_projection_golden"]

_PARSED = Memo()


def gmt_schema() -> Any:
    import pyarrow as pa

    return pa.schema([("set_id", pa.string()), ("description", pa.string()), ("members", pa.list_(pa.string()))])


def collection_of(frag: Fragment) -> str:
    """The fragment key of a GMT file: the layout's ``fragment_key``, else the file stem."""
    if frag.fragment_key:
        return frag.fragment_key
    base = os.path.basename(fragment_name(frag).split("!/")[-1])
    for suffix in (".gz", ".gmt", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


@register
class GmtFormat(PluginBase):
    kind: ClassVar[str] = "format"
    name: ClassVar[str] = "gmt"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"tabular", "nested"})
    requires: ClassVar[tuple[str, ...]] = ("pyarrow",)

    def _rows(self, frag: Fragment) -> list[dict[str, Any]]:
        return _PARSED.get((self.name, *fragment_identity(frag)), lambda: self._read(frag))

    def _read(self, frag: Fragment) -> list[dict[str, Any]]:
        fh = open_bytes(frag)
        with fh:
            try:
                data = fh.read()
            except (OSError, EOFError) as exc:
                raise unreadable(frag, exc, "cannot be read") from exc
        if not data:
            raise unreadable(frag, None, "is empty: not a GMT file")
        if b"\x00" in data:
            raise unreadable(frag, None, "holds binary data (NUL bytes): not a GMT file")
        if not data.endswith((b"\n", b"\r")):
            raise unreadable(frag, None, "does not end with a newline: the file was cut short")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise unreadable(frag, exc, "is not UTF-8 text") from exc
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.rstrip("\r").split("\t")
            if len(parts) < 2 or not parts[0].strip():
                raise unreadable(frag, None, f"line {n} is not '<set>\\t<description>\\t<members...>'")
            rows.append({"set_id": parts[0].strip(), "description": parts[1].strip() or None,
                         "members": [m.strip() for m in parts[2:] if m.strip()]})
        return rows

    def logical_schema(self, frag: Fragment) -> Any:
        self._rows(frag)
        return gmt_schema()

    def leaf_path(self, path: str, schema: Any) -> str:
        return generic_leaf_path(path, schema)

    def stats(self, frag: Fragment) -> FragmentStats:
        return stats_from_batches(self.scan([frag], columns=None, predicate=None, partitions={}))

    def metadata(self, frag: Fragment) -> Mapping[str, str]:
        rows = self._rows(frag)
        return {"fragment_key": collection_of(frag), "sets": str(len(rows)),
                "members": str(sum(len(r["members"]) for r in rows))}

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
        import pyarrow as pa

        schema = gmt_schema()
        for frag in frags:
            rows = self._rows(frag)
            _, residual = self.compile(predicate, schema) if predicate is not None else (None, None)
            wanted = projection(columns, residual, schema.names)
            sub = pa.schema([schema.field(c) for c in wanted])
            for start in range(0, len(rows), batch_rows):
                yield pa.RecordBatch.from_pylist([{c: r[c] for c in wanted} for r in rows[start:start + batch_rows]],
                                                 schema=sub)

    def to_native(self, table: Any) -> list[dict[str, Any]]:
        return native_rows(table)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import FormatCases

        return FormatCases(extension=".gmt", write=write_gmt, projection=gmt_projection_golden,
                           projection_key=("set_id",))


def write_gmt(table: Any, path: str, row_group_size: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in table.to_pylist():
            fh.write("\t".join([row["set_id"], row.get("description") or "", *(row.get("members") or [])]) + "\n")


def gmt_projection_golden() -> Any:
    import pyarrow as pa

    rows = [
        {"set_id": "HALLMARK_APOPTOSIS",
         "description": "https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_APOPTOSIS",
         "members": ["CASP3", "CASP8", "TP53", "BAX", "BCL2"]},
        {"set_id": "HALLMARK_P53_PATHWAY", "description": "https://www.gsea-msigdb.org/x",
         "members": ["TP53", "MDM2", "CDKN1A", "TP53"]},
        {"set_id": "HALLMARK_EMPTY", "description": "a set listed without members", "members": []},
        {"set_id": "HALLMARK_UNICODE", "description": None, "members": ["SEPTIN9", "HLA-A", "C1orf112"]},
    ]
    return pa.Table.from_pylist(rows, schema=gmt_schema())

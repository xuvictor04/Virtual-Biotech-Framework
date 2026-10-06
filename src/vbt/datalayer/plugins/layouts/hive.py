"""``hive``: ``key=value`` partition directories over sharded files (§6.3, §9.4).

Fragments are listed like ``sharded_dir`` (by name, never skipping an unreadable file; no
``exclude_invalid_files``), and every ``key=value`` directory between the table location and the
file becomes ``Fragment.partition[key]``: URL-unquoted, converted to the declared
``partitions.<key>.type`` (``string``, ``int64``, ``date``), and ``__HIVE_DEFAULT_PARTITION__``
as ``None`` (null: three-valued pruning never selects it for ``Eq``). Partition columns are
logical columns of the table; the format restores them with their declared types.

The probe adds R1 partition findings: a declared vocabulary (``expect: declared``) value without
a directory (``partition_missing``, that partition not ready), a directory outside it
(``partition_extra``, ``schema_drift``), a value not of the declared type (``partition_type``)
and a partition directory without data files (``partition_empty``).
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, ClassVar, Sequence
from urllib.parse import unquote

from ..base import CheckItem, Fragment, LayoutSpec
from ..registry import register
from . import is_data_name, partition_label, table_location
from .sharded_dir import ShardedDirLayout

HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"


def convert_partition(raw: str, declared: str | None) -> Any:
    """A directory value as its declared type (the raw string when it does not convert)."""
    value = unquote(raw)
    if value == HIVE_NULL:
        return None
    try:
        if declared in ("int64", "int32", "int"):
            return int(value)
        if declared == "date":
            return date.fromisoformat(value)
    except ValueError:
        return value
    return value


def _segments(path: str, location: str) -> list[tuple[str, str]]:
    rel = os.path.relpath(os.path.dirname(path), location)
    out = []
    for part in rel.replace(os.sep, "/").split("/"):
        key, sep, raw = part.partition("=")
        if sep and key:
            out.append((key, raw))
    return out


@register
class HiveLayout(ShardedDirLayout):
    name: ClassVar[str] = "hive"

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {str(k): str(v) for k, v in (spec.partitions or {}).items()}

    def partition_of(self, path: str, location: str, spec: LayoutSpec) -> dict[str, Any]:
        declared = spec.partitions or {}
        return {key: convert_partition(raw, declared.get(key)) for key, raw in _segments(path, location)}

    def _partition_dir(self, rel: str) -> str:
        parts = [p for p in rel.split("/")[:-1] if "=" in p]
        return "/".join(parts)

    def partition_items(self, root: str, spec: LayoutSpec, frags: Sequence[Fragment]) -> list[CheckItem]:
        location = table_location(root, spec)
        declared = self.partition_columns(spec)
        items: list[CheckItem] = []
        for key, typ in declared.items():
            values = [f.partition.get(key) for f in frags if key in f.partition]
            if frags and len(values) < len(frags):
                items.append(CheckItem("partition_missing", False, f"{len(frags) - len(values)} file(s) outside any "
                                       f"{key}=... directory", column=key))
            if typ in ("int64", "int32", "int", "date"):
                bad = sorted({str(v) for v in values if isinstance(v, str)})
                if bad:
                    items.append(CheckItem("partition_type", False, f"{key} values not {typ}: {', '.join(bad[:10])}",
                                           column=key))
            expect = (spec.partition_expect or {}).get(key)
            if expect is None:
                continue
            seen = {HIVE_NULL if v is None else str(v) for v in values}
            want = {HIVE_NULL if v is None else str(v) for v in expect}
            for v in sorted(want - seen):
                items.append(CheckItem("partition_missing", False, f"declared partition {key}={v} has no data",
                                       hint="download the partition or narrow the declared vocabulary", column=key,
                                       partition=f"{key}={v}"))
            for v in sorted(seen - want):
                items.append(CheckItem("partition_extra", False, f"partition {key}={v} is not in the declared "
                                       f"vocabulary", column=key, partition=f"{key}={v}"))
        items.extend(self._empty_partitions(location, spec))
        if declared and not items:
            labels = {partition_label(f.partition) for f in frags}
            items.append(CheckItem("partitions", True, f"{len(labels)} partition(s)", level="info"))
        return items

    def _empty_partitions(self, location: str, spec: LayoutSpec) -> list[CheckItem]:
        pattern = self.pattern(spec)
        items = []
        if not os.path.isdir(location):
            return items
        for dirpath, dirnames, filenames in os.walk(location):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "_")))
            rel = os.path.relpath(dirpath, location).replace(os.sep, "/")
            if rel == "." or "=" not in rel.split("/")[-1] or dirnames:
                continue
            if not any(is_data_name(f, pattern) for f in filenames):
                items.append(CheckItem("partition_empty", False, f"partition directory {rel} has no data files",
                                       partition=rel))
        return items

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        return LayoutCases(tree="hive", path="evidence", partitions={"sourceId": "string", "year": "int64"})

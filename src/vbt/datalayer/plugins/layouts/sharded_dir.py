"""``sharded_dir``: a directory of data files (Spark/Open Targets ``part-*.parquet``) (§9.4).

Lists files matching ``options.pattern`` (default by format, ``*.parquet``) **by name**,
recursively and sorted by relative path. ``*.part``, ``*.part.json``, ``_SUCCESS``, dotfiles
and files under ``.``/``_``-prefixed directories (``_temporary``) are never fragments; a listed
file is never dropped for being unreadable (VERIFIED upstream failure: one stray ``*.part``
breaks ``ds.dataset``, and ``exclude_invalid_files`` silently returned 4,800 of 22,521 rows).
A ``spec.path`` that is a file is a single fragment. For ``zarr`` a ``*.zarr`` directory store is one
fragment (never walked into), and ``*.zarr.zip`` archives are listed too.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..base import LayoutSpec
from ..registry import register
from . import FileLayout, is_data_name, is_store_format, table_location, walk_files, walk_stores


@register
class ShardedDirLayout(FileLayout):
    name: ClassVar[str] = "sharded_dir"

    def files(self, root: str, spec: LayoutSpec) -> list[str]:
        location = table_location(root, spec)
        if os.path.isfile(location):
            return [location] if is_data_name(os.path.basename(location), "*") else []
        pattern = self.pattern(spec)
        if is_store_format(spec):                       # *.zarr directory stores and *.zarr.zip archives
            return [os.path.join(location, rel) for rel, _entry in sorted(walk_stores(location, pattern),
                                                                           key=lambda x: x[0])
                    if is_data_name(os.path.basename(rel), pattern)
                    or is_data_name(os.path.basename(rel), pattern + ".zip")]
        return [os.path.join(location, rel) for rel, _entry in sorted(walk_files(location), key=lambda x: x[0])
                if is_data_name(os.path.basename(rel), pattern)]

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        return LayoutCases(tree="sharded", path="target")

"""``single_file``: one file, or one fragment per file matched by a glob (§9.4).

``TableSpec.path`` names the file (``tahoe_permissive_padj010.parquet``) or a glob
(``GSE*.h5ad``, ``**`` recursive); without a path the source root is the file. Each match is one
fragment; ``TableSpec.fragment_key`` (``{name, from: filename_regex | path_regex | directory,
pattern}``) names it (``GSE12251``). Hidden, partial (``*.part``, ``*.part.json``) and
``_SUCCESS`` files never match. The stat-only signature covers the matched files and their
partial siblings.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any, ClassVar

from ..base import CheckItem, LayoutSpec
from ..registry import register
from . import (PARTIAL_SUFFIXES, FileLayout, glob_files, is_data_name, is_hidden, is_store_format, store_files,
               table_location, walk_files)

_GLOB_CHARS = "*?["


def _is_glob(path: str) -> bool:
    return any(ch in path for ch in _GLOB_CHARS)


@register
class SingleFileLayout(FileLayout):
    name: ClassVar[str] = "single_file"

    def pattern(self, spec: LayoutSpec) -> str:
        return os.path.basename(spec.path) if spec.path and _is_glob(spec.path) else super().pattern(spec)

    def files(self, root: str, spec: LayoutSpec) -> list[str]:
        location = table_location(root, spec)
        store = is_store_format(spec)
        if _is_glob(location):
            return glob_files(location, stores=store)
        if (os.path.isfile(location) or (store and os.path.isdir(location))) and \
                is_data_name(os.path.basename(location), "*"):
            return [location]                           # a file, or one directory store (``x.zarr``)
        return []

    def location_exists(self, root: str, spec: LayoutSpec) -> bool:
        location = table_location(root, spec)
        if _is_glob(location):
            return os.path.isdir(_glob_base(location))
        return os.path.exists(location)

    def signature_files(self, root: str, spec: LayoutSpec) -> list[tuple[str, int, int]]:
        location = table_location(root, spec)
        if is_store_format(spec) and os.path.isdir(location):
            return super().signature_files(root, spec)  # a directory store: every file in it
        if is_store_format(spec) and _is_glob(location):
            out = []
            for path in glob_files(location, stores=True):
                rel = os.path.relpath(path, _glob_base(location)).replace(os.sep, "/")
                if os.path.isdir(path):
                    out.extend((f"{rel}/{r}", size, mtime) for r, size, mtime in store_files(path))
                else:
                    st = os.stat(path)
                    out.append((rel, st.st_size, st.st_mtime_ns))
            return out
        if not _is_glob(location):
            out = []
            for path in (location, *(location + s for s in PARTIAL_SUFFIXES)):
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if not os.path.isdir(path):
                    out.append((os.path.basename(path), st.st_size, st.st_mtime_ns))
            return out
        base = _glob_base(location)
        rel_pattern = os.path.relpath(location, base).replace(os.sep, "/")
        patterns = [rel_pattern, *(rel_pattern + s for s in PARTIAL_SUFFIXES)]
        out = []
        for rel, entry in walk_files(base, recursive="/" in rel_pattern or "**" in rel_pattern):
            if is_hidden(os.path.basename(rel)) or not any(_match(rel, p) for p in patterns):
                continue
            try:
                st = entry.stat(follow_symlinks=True)
            except OSError:
                continue
            out.append((rel, st.st_size, st.st_mtime_ns))
        return out

    def partial_items(self, root: str, spec: LayoutSpec) -> list[CheckItem]:
        if not _is_glob(table_location(root, spec)):
            return super().partial_items(root, spec)
        partial = sorted(rel for rel, _size, _mtime in self.signature_files(root, spec)
                         if rel.endswith(PARTIAL_SUFFIXES))
        if not partial:
            return []
        return [CheckItem("partial_files", False, f"partial download(s): {', '.join(partial[:10])}",
                          hint="finish or remove the partial files")]

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        return LayoutCases(tree="single", path="cohorts/GSE*.parquet",
                           fragment_key={"name": "cohort", "from": "filename_regex", "pattern": r"(GSE\d+)"})


def _glob_base(pattern_path: str) -> str:
    """The longest leading directory of a glob without glob characters."""
    parts = pattern_path.split(os.sep)
    base: list[str] = []
    for part in parts[:-1]:
        if _is_glob(part):
            break
        base.append(part)
    return os.sep.join(base) or (os.sep if pattern_path.startswith(os.sep) else ".")


def _match(rel: str, pattern: str) -> bool:
    if "**" in pattern:
        return fnmatch.fnmatchcase(rel, pattern.replace("**/", "*")) or fnmatch.fnmatchcase(rel, pattern)
    if rel.count("/") != pattern.count("/"):
        return False
    return fnmatch.fnmatchcase(rel, pattern)

"""Builtin layout plugins (phase 1) and what they share. Stdlib only (no pyarrow, I12).

Layouts list fragments **by name** (§9.2, I14): data files are never dropped for being
unreadable (the format raises :class:`~vbt.datalayer.plugins.base.FormatError` when a listed
fragment is read); ``*.part`` and ``*.part.json`` (partial downloads), ``_SUCCESS``, dotfiles,
and directories whose name starts with ``.`` or ``_`` (``.cache``, Spark ``_temporary``) are
never data.

* :meth:`FileLayout.signature` is stat-only (``os.scandir``/``os.stat``): sha256 of the sorted
  ``(relpath, size, mtime_ns)`` of every non-hidden file under the table location, so the
  harness can run it every turn without opening a file.
* :meth:`FileLayout.fingerprint` identifies the bytes: ``fp1:manifest:`` when the manifest gives
  a sha256 for every fragment; else ``fp1:sha256:`` (content hashes; every fragment at most
  :data:`CONTENT_HASH_MAX_BYTES`, 64 MiB) or ``fp1:stat:`` when a larger fragment is
  identified by size, mtime and the sha256 of its footer (the Parquet footer, else the last
  64 KiB). Names are relative to the fragments' common directory, so moving a release keeps its
  fingerprint and renaming a file changes it. :meth:`FileLayout.partition_fingerprints` does the
  same per partition (``"sourceId=chembl/year=2020"``), so touching one partition changes only
  its fingerprint.
* :meth:`FileLayout.probe` reports R1/R2 findings as :class:`~vbt.datalayer.plugins.base.CheckItem`
  whose ``name`` maps to a readiness status through :data:`PROBE_STATUS`.
* :func:`prune_fragments` drops fragments whose partition values make a partition-only conjunct
  false **or unknown** (three-valued: a ``__HIVE_DEFAULT_PARTITION__`` value is null).
"""

from __future__ import annotations

import fnmatch
import glob as _glob
import hashlib
import os
import re
from typing import Any, ClassVar, Iterator, Mapping, Sequence

from ...predicate import PredicateError, columns as predicate_columns, evaluate
from ..base import CheckItem, Fragment, LayoutSpec, Manifest, PluginBase
from ..formats import conjuncts, top_column

__all__ = [
    "FileLayout", "PROBE_STATUS", "PARTIAL_SUFFIXES", "JUNK_NAMES", "CONTENT_HASH_MAX_BYTES", "FORMAT_PATTERNS",
    "is_partial", "is_hidden", "is_data_name", "table_location", "prune_fragments", "partition_label",
    "manifest_entry", "attributed_entries", "file_sha256", "footer_sha256", "glob_files", "walk_files",
    "DIRECTORY_FORMATS", "is_store_format", "walk_stores", "store_files", "tree_sha256",
]

PARTIAL_SUFFIXES = (".part", ".part.json")
JUNK_NAMES = frozenset({"_SUCCESS"})
CONTENT_HASH_MAX_BYTES = 64 * 1024 * 1024
FOOTER_TAIL_BYTES = 64 * 1024
#: Data file pattern by format when ``options.pattern`` is not given.
FORMAT_PATTERNS = {"parquet": "*.parquet", "csv": "*.csv", "tsv": "*.tsv", "jsonl": "*.jsonl", "h5ad": "*.h5ad",
                   "zarr": "*.zarr", "obo": "*.obo", "gmt": "*.gmt", "npy": "*.npy", "safetensors": "*.safetensors"}
#: Formats whose fragment may be a directory store (a ``*.zarr`` directory is one fragment, never walked into);
#: their single-file form (``*.zarr.zip``) is listed as well.
DIRECTORY_FORMATS = frozenset({"zarr"})

#: Probe finding name -> the readiness status a failed finding implies (§13).
PROBE_STATUS = {
    "location": "missing",                 # the table path does not exist
    "fragments": "missing",                # no data file under it (L-3: never an empty table)
    "partial_files": "partial",            # *.part / *.part.json / _temporary: the partition is not servable
    "manifest_missing": "missing",         # a manifest entry of this table has no file
    "manifest_bytes": "partial",           # a file's size differs from its manifest entry
    "manifest_complete": "partial",        # the manifest says the download is incomplete (source level)
    "manifest_unlisted": "partial",        # warning: a data file the manifest does not list
    "partition_missing": "partial",        # a declared partition value has no directory
    "partition_extra": "schema_drift",     # a partition directory outside the declared vocabulary
    "partition_type": "schema_drift",      # a partition value that is not of the declared type
    "partition_empty": "partial",          # a partition directory without data files
    "fragment_key": "schema_drift",        # a fragment whose name does not match TableSpec.fragment_key
    "upstream_only": "ready",              # served by upstream tools only
}


def is_partial(name: str) -> bool:
    return name.endswith(PARTIAL_SUFFIXES)


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def is_data_name(name: str, pattern: str = "*.parquet") -> bool:
    """A data file by name: matches ``pattern``, not hidden, not partial, not ``_SUCCESS``."""
    if is_hidden(name) or is_partial(name) or name in JUNK_NAMES:
        return False
    return fnmatch.fnmatchcase(name, pattern)


def is_store_format(spec: LayoutSpec) -> bool:
    return str(spec.format or "") in DIRECTORY_FORMATS


def table_location(root: str | None, spec: LayoutSpec) -> str:
    """The table's path: ``spec.path`` under ``root`` (an absolute ``spec.path`` stands alone)."""
    path = spec.path or ""
    if not root:
        return path
    return os.path.join(root, path) if path else root


def _skip_dir(name: str) -> bool:
    return name.startswith((".", "_"))


def _walk(base: str, *, recursive: bool = True, stores: str | None = None) -> Iterator[tuple[str, os.DirEntry]]:
    """``(relpath, entry)`` of files under ``base``, hidden and ``_``-prefixed directories skipped. With
    ``stores`` (a name pattern), directories matching it are yielded as entries and not walked into."""
    stack = [""]
    while stack:
        rel = stack.pop()
        try:
            with os.scandir(os.path.join(base, rel) if rel else base) as it:
                entries = sorted(it, key=lambda e: e.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for e in entries:
            relpath = f"{rel}/{e.name}" if rel else e.name
            try:
                is_dir = e.is_dir(follow_symlinks=True)
            except OSError:
                is_dir = False
            if is_dir:
                if stores is not None and not _skip_dir(e.name) and fnmatch.fnmatchcase(e.name, stores):
                    yield relpath, e
                elif recursive and not _skip_dir(e.name):
                    stack.append(relpath)
                continue
            yield relpath, e


def partition_label(partition: Mapping[str, Any]) -> str:
    """``"k1=v1/k2=v2"`` (null as ``__HIVE_DEFAULT_PARTITION__``); ``""`` without partitions."""
    return "/".join(f"{k}={'__HIVE_DEFAULT_PARTITION__' if v is None else v}" for k, v in partition.items())


def prune_fragments(frags: Sequence[Fragment], predicate: Any, params: Mapping[str, Any] | None = None
                    ) -> list[Fragment]:
    """Fragments whose partition values do not make a partition-only conjunct of ``predicate``
    false or unknown (unknown never satisfies a filter, I6)."""
    if predicate is None:
        return list(frags)
    parts = [(c, {top_column(x) for x in predicate_columns(c)}) for c in conjuncts(predicate)]
    out = []
    for frag in frags:
        keys = set(frag.partition or {})
        keep = True
        for c, cols in parts:
            if not cols or not cols <= keys:
                continue
            try:
                if evaluate(c, dict(frag.partition), params) is not True:
                    keep = False
                    break
            except PredicateError:
                continue                                   # a call-time parameter: decided per row
        if keep:
            out.append(frag)
    return out


def manifest_entry(uri: str, manifest: Manifest | None) -> Mapping[str, Any] | None:
    """The manifest entry of a fragment: the longest entry key that is a path suffix of ``uri``."""
    if manifest is None:
        return None
    path = uri.replace(os.sep, "/")
    best: tuple[int, Mapping[str, Any]] | None = None
    for key, entry in manifest.entries.items():
        k = str(key).replace(os.sep, "/").lstrip("./")
        if path == k or path.endswith("/" + k):
            if best is None or len(k) > best[0]:
                best = (len(k), entry)
    return best[1] if best else None


def attributed_entries(spec: LayoutSpec, manifest: Manifest | None) -> dict[str, Mapping[str, Any]]:
    """Manifest entries of this table, by path prefix under ``spec.path`` (R2)."""
    if manifest is None:
        return {}
    prefix = (spec.path or "").replace(os.sep, "/").strip("/")
    out = {}
    for key, entry in manifest.entries.items():
        k = str(key).replace(os.sep, "/").lstrip("./")
        if not prefix or k == prefix or k.startswith(prefix + "/") or \
                (any(ch in prefix for ch in "*?[") and fnmatch.fnmatchcase(k, prefix)):
            out[k] = entry
    return out


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def footer_sha256(path: str) -> str:
    """sha256 of a Parquet footer (length-prefixed before the trailing ``PAR1``), else of the last 64 KiB."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        tail_len = 0
        if size >= 12:
            fh.seek(size - 8)
            tail = fh.read(8)
            if tail[4:] == b"PAR1":
                n = int.from_bytes(tail[:4], "little")
                if 0 < n <= size - 12:
                    tail_len = n + 8
        tail_len = tail_len or min(size, FOOTER_TAIL_BYTES)
        fh.seek(size - tail_len)
        return hashlib.sha256(fh.read(tail_len)).hexdigest()


def _local(uri: str) -> str:
    return uri[len("file://"):] if uri.startswith("file://") else uri


def _relnames(frags: Sequence[Fragment]) -> dict[str, str]:
    paths = [_local(f.uri) for f in frags]
    if not paths:
        return {}
    common = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for p in paths])
    return {f.uri: os.path.relpath(os.path.abspath(p), common).replace(os.sep, "/") for f, p in zip(frags, paths)}


class FileLayout(PluginBase):
    """Shared behaviour of the file-based layouts (``single_file``, ``sharded_dir``, ``hive``)."""

    kind: ClassVar[str] = "layout"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"scan"})
    content_hash_max_bytes: ClassVar[int] = CONTENT_HASH_MAX_BYTES

    # -- listing (subclasses) -----------------------------------------------------

    def pattern(self, spec: LayoutSpec) -> str:
        opts = spec.options or {}
        return str(opts.get("pattern") or FORMAT_PATTERNS.get(str(spec.format or "parquet"), "*.parquet"))

    def files(self, root: str, spec: LayoutSpec) -> list[str]:
        """Absolute paths of the data files, sorted (subclasses)."""
        raise NotImplementedError

    def partition_of(self, path: str, location: str, spec: LayoutSpec) -> dict[str, Any]:
        return {}

    def fragment_key(self, path: str, location: str, spec: LayoutSpec) -> str | None:
        fk = spec.fragment_key
        if not fk:
            return None
        source = fk.get("from") or fk.get("from_") or "filename_regex"
        pattern = fk.get("pattern")
        if source == "directory":
            text = os.path.basename(os.path.dirname(path))
        elif source == "path_regex":
            text = os.path.relpath(path, location if os.path.isdir(location) else os.path.dirname(location))
            text = text.replace(os.sep, "/")
        else:
            text = os.path.basename(path)
        if not pattern:
            return text
        m = re.search(str(pattern), text)
        if not m:
            return None
        return m.group(1) if m.groups() else m.group(0)

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        location = table_location(root, spec)
        out = []
        for path in self.files(root, spec):
            try:
                st = os.stat(path)
                size, mtime = st.st_size, st.st_mtime_ns
            except OSError:
                size = mtime = None                         # listed but unreadable: the format reports it
            out.append(Fragment(uri=path, size=size, mtime_ns=mtime, partition=self.partition_of(path, location, spec),
                                fragment_key=self.fragment_key(path, location, spec)))
        return out

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {}

    # -- harness-side identity -----------------------------------------------------

    def signature_files(self, root: str, spec: LayoutSpec) -> list[tuple[str, int, int]]:
        """``(relpath, size, mtime_ns)`` of every non-hidden file the table location holds (stat only)."""
        location = table_location(root, spec)
        out: list[tuple[str, int, int]] = []
        try:
            st = os.stat(location)
        except OSError:
            return out
        if not os.path.isdir(location):
            return [(os.path.basename(location), st.st_size, st.st_mtime_ns)]
        if is_store_format(spec) and fnmatch.fnmatchcase(os.path.basename(location), self.pattern(spec)):
            return store_files(location)                # the location is one directory store
        stores = self.pattern(spec) if is_store_format(spec) else None
        for rel, entry in _walk(location, stores=stores):
            if entry.is_dir():
                out.extend((f"{rel}/{r}", size, mtime) for r, size, mtime in store_files(os.path.join(location, rel)))
                continue
            if is_hidden(os.path.basename(rel)):
                continue
            try:
                est = entry.stat(follow_symlinks=True)
            except OSError:
                continue
            out.append((rel, est.st_size, est.st_mtime_ns))
        return out

    def signature(self, root: str, spec: LayoutSpec) -> str:
        entries = sorted(self.signature_files(root, spec))
        h = hashlib.sha256()
        h.update(f"{self.name}\n{spec.path or ''}\n".encode())
        if not entries and not os.path.exists(table_location(root, spec)):
            h.update(b"missing\n")
        for rel, size, mtime in entries:
            h.update(f"{rel}\t{size}\t{mtime}\n".encode())
        return "sig1:" + h.hexdigest()

    def _token(self, frag: Fragment, manifest: Manifest | None, max_bytes: int) -> tuple[str, str]:
        """``(method, token)`` identifying one fragment's bytes."""
        entry = manifest_entry(frag.uri, manifest) or {}
        sha = frag.sha256 or entry.get("sha256")
        path = _local(frag.uri)
        try:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime_ns
        except OSError:
            return "stat", "absent"
        if sha:
            return "manifest", f"{sha}:{size}"
        if os.path.isdir(path):                             # a directory store (zarr)
            return tree_sha256(path, max_bytes)
        if size <= max_bytes:
            return "sha256", file_sha256(path)
        return "stat", f"{size}:{mtime}:{footer_sha256(path)}"

    def _fingerprint(self, frags: Sequence[Fragment], manifest: Manifest | None, max_bytes: int) -> str:
        names = _relnames(frags)
        lines = []
        methods = set()
        for frag in sorted(frags, key=lambda f: names[f.uri]):
            method, token = self._token(frag, manifest, max_bytes)
            methods.add(method)
            lines.append(f"{names[frag.uri]}\t{token}")
        prefix = "manifest" if methods == {"manifest"} else ("stat" if "stat" in methods else "sha256")
        digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        return f"fp1:{prefix}:{digest}"

    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str:
        return self._fingerprint(frags, manifest, self.content_hash_max_bytes)

    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]:
        groups: dict[str, list[Fragment]] = {}
        for frag in frags:
            if frag.partition:
                groups.setdefault(partition_label(frag.partition), []).append(frag)
        return {label: self._fingerprint(group, manifest, self.content_hash_max_bytes)
                for label, group in sorted(groups.items())}

    def as_of(self, root: str, spec: LayoutSpec) -> str | None:
        return None                                         # local files: the release says when

    # -- probe (R1, R2) -------------------------------------------------------------

    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]:
        location = table_location(root, spec)
        items: list[CheckItem] = []
        exists = self.location_exists(root, spec)
        items.append(CheckItem("location", exists, f"{location} {'exists' if exists else 'does not exist'}",
                               hint="" if exists else "download or prepare the table, or fix data.sources paths"))
        frags = self.fragments(root, spec) if exists else []
        if exists:
            items.append(CheckItem("fragments", bool(frags),
                                   f"{len(frags)} data file(s) matching {self.pattern(spec)}" if frags else
                                   f"no data files matching {self.pattern(spec)} under {location}",
                                   hint="" if frags else "an empty location is not an empty table"))
        items.extend(self.partial_items(root, spec))
        if spec.fragment_key:
            unmatched = [os.path.basename(f.uri) for f in frags if f.fragment_key is None]
            items.append(CheckItem("fragment_key", not unmatched,
                                   f"fragment_key did not match: {', '.join(unmatched[:10])}" if unmatched else
                                   "every fragment has a fragment_key", level="error" if unmatched else "info"))
        items.extend(self.partition_items(root, spec, frags))
        items.extend(self.manifest_items(root, spec, manifest, frags))
        return items

    def location_exists(self, root: str, spec: LayoutSpec) -> bool:
        return os.path.exists(table_location(root, spec))

    def partial_items(self, root: str, spec: LayoutSpec) -> list[CheckItem]:
        location = table_location(root, spec)
        if not os.path.isdir(location):
            partial = [p for p in (location + s for s in PARTIAL_SUFFIXES) if os.path.exists(p)]
            by_part: dict[str, list[str]] = {"": partial} if partial else {}
        else:
            by_part = {}
            for rel, _entry in _walk(location):
                name = os.path.basename(rel)
                if is_partial(name) and not is_hidden(name):
                    by_part.setdefault(self._partition_dir(rel), []).append(rel)
            try:
                with os.scandir(location) as it:
                    for e in it:
                        if e.is_dir() and e.name == "_temporary":
                            by_part.setdefault("", []).append("_temporary/")
            except OSError:
                pass
        return [CheckItem("partial_files", False, f"partial download(s): {', '.join(sorted(files)[:10])}",
                          hint="finish or remove the partial files; only this partition is affected" if part else
                          "finish or remove the partial files", partition=part or None)
                for part, files in sorted(by_part.items())]

    def _partition_dir(self, rel: str) -> str:
        return ""

    def partition_items(self, root: str, spec: LayoutSpec, frags: Sequence[Fragment]) -> list[CheckItem]:
        return []

    def manifest_items(self, root: str, spec: LayoutSpec, manifest: Manifest | None,
                       frags: Sequence[Fragment]) -> list[CheckItem]:
        if manifest is None:
            return []
        items: list[CheckItem] = []
        if manifest.complete is False:
            items.append(CheckItem("manifest_complete", False, f"manifest {manifest.path or ''} says complete: false",
                                   hint="the download or preparation did not finish"))
        entries = attributed_entries(spec, manifest)
        base = root or ""
        missing, mismatched = [], []
        for key, entry in sorted(entries.items()):
            path = os.path.join(base, key) if base else key
            try:
                size = os.stat(path).st_size
            except OSError:
                missing.append(key)
                continue
            expected = entry.get("bytes", entry.get("size"))
            if expected is not None and int(expected) != size:
                mismatched.append(f"{key} ({size} bytes, manifest {expected})")
        if missing:
            items.append(CheckItem("manifest_missing", False, f"manifest entries without a file: "
                                   f"{', '.join(missing[:10])}" + (f" (+{len(missing) - 10})" if len(missing) > 10
                                                                    else "")))
        if mismatched:
            items.append(CheckItem("manifest_bytes", False, f"size differs from the manifest: "
                                   f"{', '.join(mismatched[:10])}"))
        if entries:
            unlisted = [os.path.basename(f.uri) for f in frags if manifest_entry(f.uri, manifest) is None]
            if unlisted:
                items.append(CheckItem("manifest_unlisted", False, f"data files the manifest does not list: "
                                       f"{', '.join(unlisted[:10])}", level="warning"))
        if not items:
            items.append(CheckItem("manifest", True, f"{len(entries)} manifest entr{'y' if len(entries) == 1 else 'ies'}"
                                   f" match the files", level="info"))
        return items


def glob_files(pattern_path: str, *, stores: bool = False) -> list[str]:
    """Data files matching a glob (``**`` recursive), sorted; hidden, partial and junk names excluded.
    With ``stores``, matching directories (``*.zarr`` stores) are data as well."""
    return sorted(p for p in _glob.glob(pattern_path, recursive=True)
                  if (os.path.isfile(p) or (stores and os.path.isdir(p))) and is_data_name(os.path.basename(p), "*"))


def walk_files(base: str, *, recursive: bool = True) -> Iterator[tuple[str, os.DirEntry]]:
    """``(relpath, DirEntry)`` of the files under ``base`` (stat-only listing)."""
    return _walk(base, recursive=recursive)


def walk_stores(base: str, pattern: str, *, recursive: bool = True) -> Iterator[tuple[str, os.DirEntry]]:
    """``(relpath, DirEntry)`` of the files under ``base`` plus the directory stores matching ``pattern``
    (``*.zarr``), which are listed as one entry each and not walked into."""
    return _walk(base, recursive=recursive, stores=pattern)


def store_files(path: str) -> list[tuple[str, int, int]]:
    """``(relpath, size, mtime_ns)`` of every file of a directory store. Store metadata is hidden
    (``.zattrs``) or ``_``-prefixed (``_index``), so nothing inside a store is skipped."""
    out = []
    for dirpath, _dirs, names in os.walk(path, followlinks=True):
        for name in names:
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append((os.path.relpath(full, path).replace(os.sep, "/"), st.st_size, st.st_mtime_ns))
    return sorted(out)


def tree_sha256(path: str, max_bytes: int = CONTENT_HASH_MAX_BYTES) -> tuple[str, str]:
    """``(method, token)`` of a directory store: content hashes of every file when the store holds at most
    ``max_bytes``, else each file's size, mtime and footer hash (``stat``)."""
    files = store_files(path)
    content = sum(size for _rel, size, _mtime in files) <= max_bytes
    h = hashlib.sha256()
    for rel, size, mtime in files:
        full = os.path.join(path, rel)
        try:
            token = file_sha256(full) if content else f"{size}:{mtime}:{footer_sha256(full)}"
        except OSError:
            token = "absent"
        h.update(f"{rel}\t{token}\n".encode())
    return ("sha256" if content else "stat"), h.hexdigest()

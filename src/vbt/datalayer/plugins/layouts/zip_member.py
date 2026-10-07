"""``zip_member``: members of a zip archive read in place, locally or over HTTP ranges (§9.4, phase 2).

The paper's Zenodo deposit (record 22259123) is one 2.9 GB zip. This layout lists the members of
an archive that match the table's member glob **by name** (directories, ``*.part``, dotfiles and
``_SUCCESS`` are never members) and reads them without extracting the archive: a local archive
through :mod:`zipfile`, a remote one (``https://...``) through the existing range reader
:class:`vbt.data.zenodo.HTTPRangeFile`, so only the central directory and the members read are
transferred.

Where the archive is:

* ``layout: {plugin: zip_member, options: {archive: virtualbiotech_submission.zip}}`` with
  ``path`` the member glob (``virtualbiotech_submission/osmr/code/data/GSE*.h5ad``), or
* ``path: "virtualbiotech_submission.zip!osmr/code/data/GSE*.h5ad"`` (archive ``!`` member glob).

A relative archive is resolved under the source root; ``options.prefix`` is prepended to the member
glob (the archive's top folder). Fragment URIs are ``zip://<archive>!/<member>``;
:func:`open_fragment` opens them (and plain paths, ``file://`` and ``http(s)://`` URIs), so every
phase-2 format plugin reads archive members, remote files and local files alike.

* ``signature`` is stat-only: the archive's ``(size, mtime_ns)`` (the URL for a remote archive,
  which cannot be stat'ed).
* ``fingerprint`` identifies the members' bytes from the central directory (``fp1:zipcrc:``: CRC-32
  and size per member, names relative to the archive), or ``fp1:manifest:`` when a manifest gives
  a sha256 for every member.
* ``probe`` reports the archive's existence (R1), matched members, a partial download next to it
  (``<archive>.part``), fragment keys and manifest entries (R2).

Stdlib only at import; the remote reader imports ``httpx`` when a remote archive is opened.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import os
import threading
import zipfile
from datetime import datetime
from typing import Any, BinaryIO, ClassVar

from ..base import CheckItem, Fragment, LayoutSpec, Manifest, PluginBase
from ..registry import register
from . import JUNK_NAMES, PARTIAL_SUFFIXES, is_hidden, is_partial, manifest_entry

__all__ = ["ZipMemberLayout", "open_fragment", "local_path", "split_uri", "member_uri", "is_remote",
           "archive_members", "clear_archive_cache"]

ZIP_SCHEME = "zip://"
_BUFFER = 1 << 20

_LOCK = threading.Lock()
_ARCHIVES: dict[str, tuple[tuple[Any, ...], zipfile.ZipFile]] = {}
_MAX_OPEN = 8


def is_remote(location: str) -> bool:
    return location.startswith(("http://", "https://"))


def member_uri(archive: str, member: str) -> str:
    """``zip://<archive>!/<member>``."""
    return f"{ZIP_SCHEME}{archive}!/{member}"


def split_uri(uri: str) -> tuple[str, str] | None:
    """``(archive, member)`` of a ``zip://`` URI, None for any other URI."""
    if not uri.startswith(ZIP_SCHEME):
        return None
    rest = uri[len(ZIP_SCHEME):]
    archive, sep, member = rest.rpartition("!/")
    if not sep:
        raise ValueError(f"{uri!r} is not zip://<archive>!/<member>")
    return archive, member


def local_path(frag: Fragment | str) -> str | None:
    """The local filesystem path of a fragment, None for archive members and remote files."""
    uri = frag.uri if isinstance(frag, Fragment) else str(frag)
    if uri.startswith(ZIP_SCHEME) or is_remote(uri):
        return None
    return uri[len("file://"):] if uri.startswith("file://") else uri


def _archive_key(archive: str) -> tuple[Any, ...]:
    if is_remote(archive):
        return (archive,)
    st = os.stat(archive)
    return (archive, st.st_size, st.st_mtime_ns)


def _open_archive(archive: str) -> zipfile.ZipFile:
    """A cached :class:`zipfile.ZipFile` of a local or remote archive (reopened when the file changes)."""
    key = _archive_key(archive)
    with _LOCK:
        hit = _ARCHIVES.get(archive)
        if hit is not None and hit[0] == key:
            return hit[1]
        if hit is not None:
            hit[1].close()
        if is_remote(archive):
            from ....data.zenodo import HTTPRangeFile

            raw: Any = io.BufferedReader(HTTPRangeFile(archive), buffer_size=_BUFFER)
            zf = zipfile.ZipFile(raw)
        else:
            zf = zipfile.ZipFile(archive)
        if len(_ARCHIVES) >= _MAX_OPEN:
            oldest = next(iter(_ARCHIVES))
            _ARCHIVES.pop(oldest)[1].close()
        _ARCHIVES[archive] = (key, zf)
        return zf


def clear_archive_cache() -> None:
    """Close every cached archive (tests; long-lived data children call it on reload)."""
    with _LOCK:
        for _key, zf in _ARCHIVES.values():
            zf.close()
        _ARCHIVES.clear()


def archive_members(archive: str) -> list[zipfile.ZipInfo]:
    """The file members of an archive, in central-directory order."""
    return [i for i in _open_archive(archive).infolist() if not i.is_dir()]


def open_fragment(frag: Fragment | str) -> BinaryIO:
    """A readable, seekable binary file for a fragment: a local path, ``file://``, ``zip://`` member or
    ``http(s)://`` URI. The caller closes it. Raises ``OSError`` when it cannot be opened."""
    uri = frag.uri if isinstance(frag, Fragment) else str(frag)
    parts = split_uri(uri)
    if parts is not None:
        archive, member = parts
        try:
            zf = _open_archive(archive)
            return zf.open(member)                     # type: ignore[return-value]
        except KeyError as exc:
            raise FileNotFoundError(f"{member} is not a member of {archive}") from exc
        except zipfile.BadZipFile as exc:
            raise OSError(f"{archive} is not a readable zip archive: {exc}") from exc
    if is_remote(uri):
        from ....data.zenodo import HTTPRangeFile

        return io.BufferedReader(HTTPRangeFile(uri), buffer_size=_BUFFER)   # type: ignore[return-value]
    path = uri[len("file://"):] if uri.startswith("file://") else uri
    return open(path, "rb")


def _is_member_name(name: str, pattern: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    if not base or is_hidden(base) or is_partial(base) or base in JUNK_NAMES:
        return False
    if any(is_hidden(p) or p.startswith("_") for p in name.split("/")[:-1]):
        return False
    if "**" in pattern:
        return fnmatch.fnmatchcase(name, pattern.replace("**/", "*")) or fnmatch.fnmatchcase(name, pattern)
    return name.count("/") == pattern.count("/") and fnmatch.fnmatchcase(name, pattern)


@register
class ZipMemberLayout(PluginBase):
    kind: ClassVar[str] = "layout"
    name: ClassVar[str] = "zip_member"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"scan"})

    # -- where -----------------------------------------------------------------------

    def location(self, root: str | None, spec: LayoutSpec) -> tuple[str, str]:
        """``(archive, member glob)`` of a table."""
        opts = dict(spec.options or {})
        path = spec.path or ""
        archive = opts.get("archive")
        if archive is None:
            archive, sep, path = path.partition("!")
            if not sep:
                raise ValueError(f"{spec.table}: zip_member needs options.archive or path '<archive>!<member glob>'")
        archive = str(archive)
        if not is_remote(archive) and not os.path.isabs(archive) and root:
            archive = os.path.join(root, archive)
        prefix = str(opts.get("prefix") or "").strip("/")
        member = path.lstrip("/")
        return archive, f"{prefix}/{member}" if prefix else member

    # -- listing ---------------------------------------------------------------------

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        try:
            archive, pattern = self.location(root, spec)
        except ValueError:
            return []
        if not is_remote(archive) and not os.path.isfile(archive):
            return []
        try:
            infos = archive_members(archive)
        except (OSError, zipfile.BadZipFile):
            return []
        out = []
        for info in sorted(infos, key=lambda i: i.filename):
            if not _is_member_name(info.filename, pattern):
                continue
            mtime = int(datetime(*info.date_time).timestamp() * 1e9)
            out.append(Fragment(uri=member_uri(archive, info.filename), size=info.file_size or None,
                                mtime_ns=mtime, fragment_key=self._fragment_key(info.filename, spec)))
        return out

    def _fragment_key(self, member: str, spec: LayoutSpec) -> str | None:
        fk = spec.fragment_key
        if not fk:
            return None
        import re

        source = fk.get("from") or fk.get("from_") or "filename_regex"
        text = member.rsplit("/", 1)[-1] if source == "filename_regex" else (
            member.rsplit("/", 2)[-2] if source == "directory" and "/" in member else member)
        pattern = fk.get("pattern")
        if not pattern:
            return text
        m = re.search(str(pattern), text)
        if not m:
            return None
        return m.group(1) if m.groups() else m.group(0)

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {}

    # -- identity ----------------------------------------------------------------------

    def signature(self, root: str, spec: LayoutSpec) -> str:
        h = hashlib.sha256(f"{self.name}\n{spec.path or ''}\n".encode())
        try:
            archive, pattern = self.location(root, spec)
        except ValueError:
            return "sig1:" + h.hexdigest()
        h.update(f"{archive}\n{pattern}\n".encode())
        if not is_remote(archive):
            for path in (archive, *(archive + s for s in PARTIAL_SUFFIXES)):
                try:
                    st = os.stat(path)
                except OSError:
                    h.update(f"{os.path.basename(path)}\tmissing\n".encode())
                    continue
                h.update(f"{os.path.basename(path)}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
        return "sig1:" + h.hexdigest()

    def _crc(self, frag: Fragment) -> tuple[str, str]:
        parts = split_uri(frag.uri)
        if parts is None:
            return frag.uri, "absent"
        archive, member = parts
        try:
            info = _open_archive(archive).getinfo(member)
        except (KeyError, OSError, zipfile.BadZipFile):
            return member, "absent"
        return member, f"{info.CRC:08x}:{info.file_size}"

    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str:
        lines = []
        shas = []
        for frag in sorted(frags, key=lambda f: f.uri):
            member, token = self._crc(frag)
            entry = manifest_entry(member, manifest) or {}
            sha = frag.sha256 or entry.get("sha256")
            shas.append(sha)
            lines.append(f"{member}\t{sha or token}")
        prefix = "manifest" if frags and all(shas) else "zipcrc"
        return f"fp1:{prefix}:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()

    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]:
        return {}

    def as_of(self, root: str, spec: LayoutSpec) -> str | None:
        return None

    # -- probe (R1, R2) ------------------------------------------------------------------

    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]:
        try:
            archive, pattern = self.location(root, spec)
        except ValueError as exc:
            return [CheckItem("location", False, str(exc), hint="name the archive in options.archive")]
        items: list[CheckItem] = []
        if is_remote(archive):
            try:
                archive_members(archive)
                ok, detail = True, f"{archive} is reachable"
            except (OSError, zipfile.BadZipFile) as exc:
                ok, detail = False, f"{archive} cannot be read: {exc}"
        else:
            ok = os.path.isfile(archive)
            detail = f"{archive} {'exists' if ok else 'is not a file'}"
        items.append(CheckItem("location", ok, detail, hint="" if ok else "fetch the archive or fix options.archive"))
        if not ok:
            return items
        partial = [archive + s for s in PARTIAL_SUFFIXES if not is_remote(archive) and os.path.exists(archive + s)]
        if partial:
            items.append(CheckItem("partial_files", False, f"partial download(s): {', '.join(partial)}",
                                   hint="finish or remove the partial download"))
        try:
            frags = self.fragments(root, spec)
        except (OSError, zipfile.BadZipFile) as exc:
            items.append(CheckItem("fragments", False, f"{archive}: {exc}"))
            return items
        items.append(CheckItem("fragments", bool(frags),
                               f"{len(frags)} member(s) matching {pattern}" if frags else
                               f"no member of {archive} matches {pattern}",
                               hint="" if frags else "an empty selection is not an empty table"))
        if spec.fragment_key:
            unmatched = [split_uri(f.uri)[1] for f in frags if f.fragment_key is None]   # type: ignore[index]
            items.append(CheckItem("fragment_key", not unmatched,
                                   f"fragment_key did not match: {', '.join(unmatched[:10])}" if unmatched else
                                   "every member has a fragment_key", level="error" if unmatched else "info"))
        if manifest is not None:
            mismatched = []
            for frag in frags:
                member = split_uri(frag.uri)[1]                                     # type: ignore[index]
                entry = manifest_entry(member, manifest)
                expected = None if entry is None else entry.get("bytes", entry.get("size"))
                if expected is not None and int(expected) != (frag.size or 0):
                    mismatched.append(f"{member} ({frag.size} bytes, manifest {expected})")
            if mismatched:
                items.append(CheckItem("manifest_bytes", False, f"size differs from the manifest: "
                                       f"{', '.join(mismatched[:10])}"))
        return items

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        # The file-tree suite (L-1, L-2, L-4, L-5, L-7) damages and renames files on disk, which archive
        # members are not; test_dl_formats_v2.py runs the same properties on a golden archive.
        return LayoutCases(tree="none", path="golden.zip!data/*.csv", format="csv")

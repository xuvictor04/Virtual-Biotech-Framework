"""``zip_member``: members of a remote zip archive, read with HTTP range requests.

Options::

    archive:  https://example.org/archive.zip       the archive (or find it in a JSON index:)
    archive_index:                                   a json_index spec whose files include the archive
      index:  https://zenodo.org/api/records/22259123
      files:  "$.files[*]"
      path:   "$.key"
      url:    "$.links.self"
      size:   "$.size"
      checksums: {md5: "$.checksum"}
      select: virtualbiotech_submission.zip          the archive's path in that index
      record: {doi: "$.doi", version_record: "$.id"}

Only the central directory and the selected members are transferred. A member keeps its archive path (a member
that would land outside the destination is dropped). Its checksum is the CRC-32 the central directory records,
which :mod:`zipfile` also checks while it inflates the member, so a corrupt transfer never takes the file's name.
Members are fetched whole (no range resume).
"""

from __future__ import annotations

import io
import threading
import zipfile
import zlib
from dataclasses import replace
from typing import Any, Iterator, Mapping

from ..base import AcquisitionBase, AcquisitionError, RemoteFile
from ..registry import register
from . import canonical_path, sorted_files

__all__ = ["ZipMemberTransport"]

_BUFFER = 1 << 20


@register
class ZipMemberTransport(AcquisitionBase):
    name = "zip_member"
    version = "1.0"
    capabilities = frozenset({"members", "sizes", "checksums"})

    def describe(self, options: Mapping[str, Any]) -> str:
        if options.get("archive"):
            return f"members of the zip {options['archive']} (range requests)"[:200]
        idx = dict(options.get("archive_index") or {})
        return f"members of {idx.get('select', '?')} listed by {idx.get('index', '?')} (range requests)"[:200]

    def _archive(self, options: Mapping[str, Any], session: Any) -> tuple[str, int | None, dict[str, Any]]:
        if options.get("archive"):
            return str(options["archive"]), None, {"url": str(options["archive"])}
        spec = dict(options.get("archive_index") or {})
        if not spec.get("index") or not spec.get("select"):
            raise AcquisitionError("zip_member: give `archive`, or `archive_index` with `index` and `select`")
        from ...gateway.fields import jp_first
        from .json_index import index_files

        doc = session.get_json(str(spec["index"]))
        hit = [f for f in index_files(doc, spec, url=str(spec["index"])) if f.path == spec["select"]]
        if not hit:
            raise AcquisitionError(f"{spec['index']}: lists no file {spec['select']!r}", url=str(spec["index"]))
        about = {"url": str(spec["index"]), "archive": hit[0].url, "archive_size": hit[0].size,
                 "archive_checksums": dict(hit[0].checksums),
                 **{str(k): jp_first(doc, str(v)) for k, v in dict(spec.get("record") or {}).items()}}
        return hit[0].url, hit[0].size, about

    def _open(self, url: str, size: int | None, session: Any) -> zipfile.ZipFile:
        """The archive opened once per thread of this session (a ZipFile is not shared across threads); the
        session closes it."""
        key = ("zip_member", threading.get_ident(), url)
        zf = session.cache.get(key)
        if zf is None:
            from ....data.zenodo import HTTPRangeFile

            try:
                raw = HTTPRangeFile(url, client=session.client, size=size, retries=session.retries,
                                    backoff=max(session.retry_wait, 0.0))
                zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=_BUFFER))
            except (zipfile.BadZipFile, OSError) as exc:
                raise AcquisitionError(f"{url}: not a readable zip archive ({exc})", url=url) from exc
            session.cache[key] = zf
        return zf

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        url, size, about = self._archive(options, session)
        zf = self._open(url, size, session)
        files = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            try:
                rel = canonical_path(info.filename)
            except AcquisitionError:
                continue                               # a member that would land outside the destination
            files.append(RemoteFile(rel, url, info.file_size, {"crc32": f"{info.CRC & 0xFFFFFFFF:08x}"},
                                    member=info.filename, extra={"archive_size": size}))
        return sorted_files(replace(f, extra={**f.extra, "index": about}) for f in files)

    def fetch(self, file: RemoteFile, session: Any, *, offset: int = 0) -> Iterator[bytes]:
        if offset:
            raise AcquisitionError(f"{file.path}: zip members are fetched whole (no range resume)")
        zf = self._open(file.url, file.extra.get("archive_size"), session)
        try:
            with zf.open(file.member or file.path) as src:
                for chunk in iter(lambda: src.read(_BUFFER), b""):
                    session.bytes += len(chunk)
                    yield chunk
        except (zipfile.BadZipFile, KeyError, OSError, EOFError, zlib.error) as exc:
            raise AcquisitionError(f"{file.path}: {type(exc).__name__}: {exc}", url=file.url,
                                   transient=isinstance(exc, OSError)) from exc

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.acquisition import AcquisitionCases

        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("top/", b"")
                for rel, body in sorted(files.items()):
                    zf.writestr(rel, body)
            return {"/a.zip": (buf.getvalue(), "application/zip")}

        return AcquisitionCases(publish=publish, options=lambda root: {"archive": f"{root}/a.zip"})

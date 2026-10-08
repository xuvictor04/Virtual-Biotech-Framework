"""``http_range``: files served over HTTP with ``Range`` support, read in place (§9.4, phase 2).

A table names its files by URL: ``options.urls`` (a list), or ``path`` (one URL, or a URL relative to
a ``root`` that is itself a URL). Each URL is one fragment, listed in the declared order and never
dropped for being unreachable (the format raises when it reads it, I14). Reads go through the
existing range reader :class:`vbt.data.zenodo.HTTPRangeFile` (via
:func:`~vbt.datalayer.plugins.layouts.zip_member.open_fragment`), so a scan transfers only the bytes
it reads.

:meth:`HttpRangeLayout.footer_stats` gives **footer-only** statistics of a remote Parquet file: the
size comes from one ``HEAD``, the tail and the footer from range requests (one when the footer fits
in pyarrow's first 64 KiB read), parsed with ``pyarrow.parquet.ParquetFile``; no row group is
transferred. The statistics are the parquet format's own
(:func:`~vbt.datalayer.plugins.formats.parquet.footer_stats`: Arrow storage types, null counts,
min/max), so a remote fragment reports what the same file reports locally. Formats without footers
fall back to their own ``stats``.

* ``signature`` cannot ``stat`` a URL: it hashes the URL list (stdlib only, no request), so the
  harness's per-turn check is free; the content is identified by ``fingerprint``.
* ``fingerprint``: the manifest sha256 when every URL has one, else ``size`` + ``ETag`` +
  ``Last-Modified`` from one ``HEAD`` per URL (``fp1:http:``).
* ``probe``: one ``HEAD`` per URL (reachable, size, ``Accept-Ranges``) and the manifest byte counts.

``httpx`` is imported when a request is made.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Mapping

from ..base import CheckItem, FormatError, Fragment, FragmentStats, LayoutSpec, Manifest, PluginBase
from ..registry import register
from . import manifest_entry

__all__ = ["HttpRangeLayout", "head", "read_footer", "footer_stats"]

_TIMEOUT = 60.0


def head(url: str, *, client: Any = None, timeout: float = _TIMEOUT) -> dict[str, Any]:
    """``{status, size, etag, last_modified, ranges}`` of one ``HEAD`` request (redirects followed)."""
    import httpx

    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
        r = client.head(url)
    finally:
        if own:
            client.close()
    size = r.headers.get("content-length")
    return {"status": r.status_code, "size": int(size) if size and size.isdigit() else None,
            "etag": r.headers.get("etag"), "last_modified": r.headers.get("last-modified"),
            "ranges": r.headers.get("accept-ranges", "").lower() == "bytes"}


def read_footer(frag: Fragment | str, *, size: int | None = None) -> tuple[Any, Any]:
    """``(arrow schema, FileMetaData)`` of a remote Parquet file, read with range requests only (the
    size, the 8-byte tail and the footer; no row group is transferred). Raises :class:`FormatError`
    when the file has no readable footer."""
    from ....data.zenodo import HTTPRangeFile

    uri = frag.uri if isinstance(frag, Fragment) else str(frag)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - the data child has pyarrow
        raise FormatError("footer statistics need pyarrow", fragment=uri) from exc
    raw = None
    try:
        raw = HTTPRangeFile(uri, size=size if size is not None else (frag.size if isinstance(frag, Fragment)
                                                                      else None))
        pf = pq.ParquetFile(raw)
        return pf.schema_arrow, pf.metadata
    except Exception as exc:  # noqa: BLE001 - any transport or footer error: never an empty table
        raise FormatError(f"{uri}: no readable Parquet footer over HTTP ranges ({type(exc).__name__}: {exc})",
                          fragment=uri) from exc
    finally:
        if raw is not None:
            raw.close()


def footer_stats(frag: Fragment | str, *, size: int | None = None) -> FragmentStats:
    """Parquet footer statistics of a remote file read with range requests only (no row group is
    transferred): the same per-leaf Arrow storage types, null counts and min/max as a local file's
    :meth:`~vbt.datalayer.plugins.formats.parquet.ParquetFormat.stats`. Raises :class:`FormatError`
    when the file has no readable footer."""
    from ..formats.parquet import footer_stats as stats_of

    schema, md = read_footer(frag, size=size)
    return stats_of(schema, md)


@register
class HttpRangeLayout(PluginBase):
    kind: ClassVar[str] = "layout"
    name: ClassVar[str] = "http_range"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"scan"})

    def urls(self, root: str | None, spec: LayoutSpec) -> list[str]:
        """The table's URLs in declared order."""
        opts: Mapping[str, Any] = spec.options or {}
        listed = opts.get("urls")
        if listed:
            raw = [str(u) for u in (listed if isinstance(listed, (list, tuple)) else [listed])]
        elif spec.path:
            raw = [spec.path]
        else:
            raw = [root] if root else []
        out = []
        for u in raw:
            if u.startswith(("http://", "https://")):
                out.append(u)
            elif root and str(root).startswith(("http://", "https://")):
                out.append(str(root).rstrip("/") + "/" + u.lstrip("/"))
        return out

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        # Listing makes no request: sizes come from the manifest or the first read.
        return [Fragment(uri=u, size=None, mtime_ns=None) for u in self.urls(root, spec)]

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {}

    def signature(self, root: str, spec: LayoutSpec) -> str:
        text = "\n".join([self.name, *self.urls(root, spec)])
        return "sig1:" + hashlib.sha256(text.encode()).hexdigest()

    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str:
        lines = []
        shas = []
        for frag in sorted(frags, key=lambda f: f.uri):
            entry = manifest_entry(frag.uri, manifest) or {}
            sha = frag.sha256 or entry.get("sha256")
            shas.append(sha)
            if sha:
                lines.append(f"{frag.uri}\t{sha}")
                continue
            try:
                h = head(frag.uri)
                token = f"{h['status']}:{h['size']}:{h['etag'] or ''}:{h['last_modified'] or ''}"
            except Exception as exc:  # noqa: BLE001 - an unreachable URL still gets a (distinct) fingerprint
                token = f"unreachable:{type(exc).__name__}"
            lines.append(f"{frag.uri}\t{token}")
        prefix = "manifest" if frags and all(shas) else "http"
        return f"fp1:{prefix}:" + hashlib.sha256("\n".join(lines).encode()).hexdigest()

    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]:
        return {}

    def as_of(self, root: str, spec: LayoutSpec) -> str | None:
        return None

    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]:
        urls = self.urls(root, spec)
        if not urls:
            return [CheckItem("location", False, f"{spec.table}: no URL (options.urls, an http(s) path or root)",
                              hint="declare the files' URLs")]
        items: list[CheckItem] = []
        unreachable, no_ranges, mismatched = [], [], []
        for u in urls:
            try:
                h = head(u)
            except Exception as exc:  # noqa: BLE001
                unreachable.append(f"{u} ({type(exc).__name__})")
                continue
            if h["status"] >= 400:
                unreachable.append(f"{u} (HTTP {h['status']})")
                continue
            if not h["ranges"]:
                no_ranges.append(u)
            entry = manifest_entry(u, manifest)
            expected = None if entry is None else entry.get("bytes", entry.get("size"))
            if expected is not None and h["size"] is not None and int(expected) != h["size"]:
                mismatched.append(f"{u} ({h['size']} bytes, manifest {expected})")
        items.append(CheckItem("location", not unreachable,
                               f"unreachable: {', '.join(unreachable[:10])}" if unreachable else
                               f"{len(urls)} URL(s) reachable", hint="check the URLs and the network"
                               if unreachable else ""))
        items.append(CheckItem("fragments", bool(urls), f"{len(urls)} file(s) by URL"))
        if no_ranges:
            items.append(CheckItem("ranges", False, f"no Accept-Ranges: bytes on {', '.join(no_ranges[:10])}",
                                   level="warning", hint="reads will download whole files"))
        if mismatched:
            items.append(CheckItem("manifest_bytes", False, f"size differs from the manifest: "
                                   f"{', '.join(mismatched[:10])}"))
        return items

    def footer_stats(self, frag: Fragment) -> FragmentStats:
        """Footer-only statistics of a remote Parquet fragment (see :func:`footer_stats`)."""
        return footer_stats(frag)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        # Remote files cannot be damaged or renamed by the file-tree suite; test_dl_formats_v2.py
        # serves a golden over a local Range-capable server instead.
        return LayoutCases(tree="none", path=None, format="csv")

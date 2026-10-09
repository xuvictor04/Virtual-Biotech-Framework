"""``json_index``: files listed by a JSON document (a Zenodo record, a figshare article, any REST listing).

Options::

    index:     https://api.figshare.com/v2/articles/27993248/files?page_size=1000
    files:     "$[*]"                    the file objects (JSONPath, the overlay subset)
    path:      "$.name"                  each file's relative path (where it lands; what patterns match)
    url:       "$.download_url"          where it is read from
    size:      "$.size"
    checksums: {md5: "$.supplied_md5"}   {algo: JSONPath}; a value "md5:<hex>" (Zenodo) names its own algorithm
    record:    {version: "$.version", doi: "$.doi"}   facts of the index recorded with the acquisition (optional)

Zenodo answers a concept record with a redirect to its latest version (followed); pin a version record when the
release must not move. Files are read with range requests (Zenodo and figshare both honour them).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..base import AcquisitionBase, AcquisitionError, RemoteFile
from ..registry import register
from . import canonical_path, sorted_files

__all__ = ["JsonIndexTransport", "index_files"]

_ALGOS = ("sha256", "sha1", "md5")


def index_files(doc: Any, options: Mapping[str, Any], *, url: str) -> list[RemoteFile]:
    """The files a JSON index document lists, read with the options' JSONPaths."""
    from ...gateway.fields import jp_first, jp_get

    items = jp_get(doc, str(options.get("files") or "$[*]"))
    files: list[RemoteFile] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        path = jp_first(item, str(options.get("path") or "$.name"))
        link = jp_first(item, str(options.get("url") or "$.url"))
        if not path or not link:
            raise AcquisitionError(f"{url}: a listed file has no path or url ({str(item)[:120]})", url=url)
        size = jp_first(item, str(options.get("size") or "$.size"))
        sums: dict[str, str] = {}
        for algo, jp in dict(options.get("checksums") or {}).items():
            value = jp_first(item, str(jp))
            if not value:
                continue
            text = str(value)
            named, sep, hexd = text.partition(":")
            if sep and named in _ALGOS:
                sums[named] = hexd.lower()
            elif algo in _ALGOS:
                sums[str(algo)] = text.lower()
        try:
            n = int(size) if size is not None else None
        except (TypeError, ValueError):
            n = None
        files.append(RemoteFile(canonical_path(str(path)), str(link), n, sums))
    return files


@register
class JsonIndexTransport(AcquisitionBase):
    name = "json_index"
    version = "1.0"
    capabilities = frozenset({"range", "sizes", "checksums"})

    def describe(self, options: Mapping[str, Any]) -> str:
        return f"files listed by the JSON index {options.get('index', '')}"[:200]

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        url = str(options.get("index") or "")
        if not url:
            raise AcquisitionError("json_index: `index` is required")
        from dataclasses import replace

        from ...gateway.fields import jp_first

        doc = session.get_json(url)
        files = index_files(doc, options, url=url)
        if not files:
            raise AcquisitionError(f"{url}: the index lists no file at {options.get('files') or '$[*]'}", url=url)
        about = {"url": url, **{str(k): jp_first(doc, str(v)) for k, v in dict(options.get("record") or {}).items()}}
        return sorted_files(replace(f, extra={"index": about}) for f in files)

    @classmethod
    def conformance_cases(cls) -> Any:
        import hashlib

        from ..conformance.acquisition import AcquisitionCases

        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            out: dict[str, tuple[bytes, str]] = {}
            entries = []
            for i, (rel, body) in enumerate(sorted(files.items())):
                out[f"/files/{i}"] = (body, "application/octet-stream")
                entries.append({"key": rel, "size": len(body), "checksum": f"md5:{hashlib.md5(body).hexdigest()}",
                                "links": {"self": f"{root}/files/{i}"}})
            out["/api/records/1"] = (json.dumps({"id": 1, "files": entries}).encode(), "application/json")
            return out

        return AcquisitionCases(publish=publish, options=lambda root: {
            "index": f"{root}/api/records/1", "files": "$.files[*]", "path": "$.key", "url": "$.links.self",
            "size": "$.size", "checksums": {"md5": "$.checksum"}})

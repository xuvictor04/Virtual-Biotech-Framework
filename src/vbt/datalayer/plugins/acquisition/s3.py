"""``s3`` and ``gcs``: objects of a public cloud bucket, listed and read anonymously over HTTPS.

Options (``s3``)::

    bucket:    cellxgene-census-public-us-west-2
    region:    us-west-2
    prefix:    cell-census/{release}/           listed objects land at their key below the prefix
    endpoint:  https://<bucket>.s3.<region>.amazonaws.com   (default; another S3-compatible endpoint works)

Options (``gcs``)::

    bucket:    <bucket>
    prefix:    <prefix>/
    endpoint:  https://storage.googleapis.com

S3 is listed with ListObjectsV2 (XML, ``continuation-token`` paging); a single-part object's ETag is its md5
(a multi-part ETag, ``<hex>-<n>``, is not a checksum and is dropped). GCS is listed with the JSON API
(``pageToken`` paging) and reports ``md5Hash`` (base64). Objects are read with range requests. An XML listing that
declares a DTD or an entity is refused before it is parsed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET

from ..base import AcquisitionBase, AcquisitionError, RemoteFile
from ..registry import register
from . import b64_to_hex, canonical_path, sorted_files

__all__ = ["S3Transport", "GcsTransport"]

_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _relative(key: str, prefix: str) -> str | None:
    if not key.startswith(prefix) or key.endswith("/"):
        return None
    try:
        return canonical_path(key[len(prefix):])
    except AcquisitionError:
        return None


@register
class S3Transport(AcquisitionBase):
    name = "s3"
    version = "1.0"
    capabilities = frozenset({"range", "sizes", "checksums"})

    @staticmethod
    def _endpoint(options: Mapping[str, Any]) -> tuple[str, str, str]:
        bucket = str(options.get("bucket") or "")
        if not bucket:
            raise AcquisitionError("s3: `bucket` is required")
        region = str(options.get("region") or "us-east-1")
        endpoint = str(options.get("endpoint") or f"https://{bucket}.s3.{region}.amazonaws.com").rstrip("/")
        return bucket, endpoint, str(options.get("prefix") or "")

    def describe(self, options: Mapping[str, Any]) -> str:
        bucket, endpoint, prefix = self._endpoint(options)
        return f"public S3 bucket s3://{bucket}/{prefix} ({endpoint}, anonymous)"[:200]

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        _bucket, endpoint, prefix = self._endpoint(options)
        files: list[RemoteFile] = []
        token: str | None = None
        while True:
            q = {"list-type": "2", "prefix": prefix}
            if token:
                q["continuation-token"] = token
            url = f"{endpoint}/?{urlencode(q)}"
            raw = session.get_bytes(url)
            head = raw[:2048].upper()
            if b"<!DOCTYPE" in head or b"<!ENTITY" in raw.upper():
                raise AcquisitionError(f"GET {url}: refusing an XML listing with a DTD or entities", url=url)
            try:
                root = ET.fromstring(raw)
            except ET.ParseError as exc:
                raise AcquisitionError(f"GET {url}: not an S3 listing ({exc})", url=url) from exc
            ns = _NS if root.tag.startswith(_NS) else ""
            if root.tag != f"{ns}ListBucketResult":
                raise AcquisitionError(f"GET {url}: not an S3 listing (<{root.tag}>)", url=url)
            for c in root.findall(f"{ns}Contents"):
                key = c.findtext(f"{ns}Key") or ""
                rel = _relative(key, prefix)
                if rel is None:
                    continue
                size = c.findtext(f"{ns}Size")
                etag = (c.findtext(f"{ns}ETag") or "").strip('"').lower()
                sums = {"md5": etag} if len(etag) == 32 and "-" not in etag else {}
                files.append(RemoteFile(rel, f"{endpoint}/{quote(key)}", int(size) if size else None, sums))
            if (root.findtext(f"{ns}IsTruncated") or "").lower() == "true":
                token = root.findtext(f"{ns}NextContinuationToken")
                if not token:
                    raise AcquisitionError(f"GET {url}: truncated listing without a continuation token", url=url)
                continue
            return sorted_files(files)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.acquisition import AcquisitionCases

        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            out: dict[str, tuple[bytes, str]] = {}
            items = sorted(files.items())
            pages = [items[: len(items) // 2], items[len(items) // 2:]]
            for n, page in enumerate(pages):
                contents = "".join(
                    f"<Contents><Key>pre/{rel}</Key><Size>{len(body)}</Size>"
                    f"<ETag>&quot;{hashlib.md5(body).hexdigest()}&quot;</ETag></Contents>" for rel, body in page)
                more = "<IsTruncated>true</IsTruncated><NextContinuationToken>t2</NextContinuationToken>" if n == 0 \
                    else "<IsTruncated>false</IsTruncated>"
                doc = (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="{_NS[1:-1]}">'
                       f"<Name>b</Name><Prefix>pre/</Prefix>{contents}{more}</ListBucketResult>")
                q = {"list-type": "2", "prefix": "pre/", **({"continuation-token": "t2"} if n else {})}
                out[f"/?{urlencode(q)}"] = (doc.encode(), "application/xml")
            for rel, body in items:
                out[f"/{quote('pre/' + rel)}"] = (body, "application/octet-stream")
            return out

        return AcquisitionCases(publish=publish, options=lambda root: {"bucket": "b", "prefix": "pre/",
                                                                       "endpoint": root})


@register
class GcsTransport(AcquisitionBase):
    name = "gcs"
    version = "1.0"
    capabilities = frozenset({"range", "sizes", "checksums"})

    @staticmethod
    def _parts(options: Mapping[str, Any]) -> tuple[str, str, str]:
        bucket = str(options.get("bucket") or "")
        if not bucket:
            raise AcquisitionError("gcs: `bucket` is required")
        endpoint = str(options.get("endpoint") or "https://storage.googleapis.com").rstrip("/")
        return bucket, endpoint, str(options.get("prefix") or "")

    def describe(self, options: Mapping[str, Any]) -> str:
        bucket, endpoint, prefix = self._parts(options)
        return f"public GCS bucket gs://{bucket}/{prefix} ({endpoint}, anonymous)"[:200]

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        bucket, endpoint, prefix = self._parts(options)
        files: list[RemoteFile] = []
        token: str | None = None
        while True:
            q = {"prefix": prefix, **({"pageToken": token} if token else {})}
            url = f"{endpoint}/storage/v1/b/{quote(bucket, safe='')}/o?{urlencode(q)}"
            page = session.get_json(url)
            if not isinstance(page, Mapping):
                raise AcquisitionError(f"GET {url}: not a GCS listing", url=url)
            for item in page.get("items") or []:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name") or "")
                rel = _relative(name, prefix)
                if rel is None:
                    continue
                sums = {"md5": b64_to_hex(str(item["md5Hash"]))} if item.get("md5Hash") else {}
                size = item.get("size")
                files.append(RemoteFile(rel, f"{endpoint}/{quote(bucket, safe='')}/{quote(name)}",
                                        int(size) if size is not None else None, sums))
            token = page.get("nextPageToken")
            if not token:
                return sorted_files(files)

    @classmethod
    def conformance_cases(cls) -> Any:
        import base64

        from ..conformance.acquisition import AcquisitionCases

        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            out: dict[str, tuple[bytes, str]] = {}
            items = sorted(files.items())
            pages = [items[: len(items) // 2], items[len(items) // 2:]]
            for n, page in enumerate(pages):
                body = {"items": [{"name": f"pre/{rel}", "size": str(len(b)),
                                   "md5Hash": base64.b64encode(hashlib.md5(b).digest()).decode()} for rel, b in page]}
                if n == 0:
                    body["nextPageToken"] = "p2"
                q = {"prefix": "pre/", **({"pageToken": "p2"} if n else {})}
                out[f"/storage/v1/b/b/o?{urlencode(q)}"] = (json.dumps(body).encode(), "application/json")
            for rel, b in items:
                out[f"/b/{quote('pre/' + rel)}"] = (b, "application/octet-stream")
            return out

        return AcquisitionCases(publish=publish, options=lambda root: {"bucket": "b", "prefix": "pre/",
                                                                       "endpoint": root})

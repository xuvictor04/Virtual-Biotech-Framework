"""``huggingface``: files of a Hugging Face repository at a pinned revision.

Options::

    repo:       tahoebio/Tahoe-100M
    repo_type:  dataset                  dataset | model | space (default dataset)
    revision:   "{release}"              a commit hash pins the files; a branch name does not
    paths:      [metadata]               directories listed (recursively); default: the whole repository
    endpoint:   https://huggingface.co
    token_env:  HF_TOKEN                 a gated repository: the token is read from this variable, never stored

The listing is the tree API (``/api/<type>s/<repo>/tree/<revision>/<path>?recursive=true``, paged through the
``Link`` header). Files stored with LFS carry their sha256 (``lfs.oid``); others their git blob id
(``git_sha1``). Files are read from ``/<type prefix><repo>/resolve/<revision>/<path>`` with range requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping
from urllib.parse import quote

from ..base import AcquisitionBase, AcquisitionError, RemoteFile
from ..registry import register
from . import canonical_path, sorted_files

__all__ = ["HuggingFaceTransport"]

_PREFIX = {"dataset": "datasets/", "model": "", "space": "spaces/"}


def _next_link(header: str | None) -> str | None:
    for part in (header or "").split(","):
        m = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part)
        if m:
            return m.group(1)
    return None


@register
class HuggingFaceTransport(AcquisitionBase):
    name = "huggingface"
    version = "1.0"
    capabilities = frozenset({"range", "sizes", "checksums"})

    @staticmethod
    def _parts(options: Mapping[str, Any]) -> tuple[str, str, str, str]:
        repo = str(options.get("repo") or "")
        kind = str(options.get("repo_type") or "dataset")
        revision = str(options.get("revision") or "main")
        endpoint = str(options.get("endpoint") or "https://huggingface.co").rstrip("/")
        if not repo or "/" not in repo or kind not in _PREFIX:
            raise AcquisitionError(f"huggingface: `repo` is owner/name and repo_type one of {sorted(_PREFIX)}")
        return repo, kind, revision, endpoint

    def describe(self, options: Mapping[str, Any]) -> str:
        repo, kind, revision, endpoint = self._parts(options)
        paths = ", ".join(options.get("paths") or []) or "/"
        return f"Hugging Face {kind} {repo} at {revision} ({paths}; {endpoint})"[:200]

    def _headers(self, options: Mapping[str, Any]) -> dict[str, str]:
        env = options.get("token_env")
        token = os.environ.get(str(env)) if env else None
        return {"Authorization": f"Bearer {token}"} if token else {}

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        repo, kind, revision, endpoint = self._parts(options)
        headers = self._headers(options)
        rev = quote(revision, safe="")
        files: list[RemoteFile] = []
        for top in options.get("paths") or [""]:
            top = str(top).strip("/")
            url = f"{endpoint}/api/{kind}s/{repo}/tree/{rev}" + (f"/{quote(top)}" if top else "") + "?recursive=true"
            while url:
                r = session.request("GET", url, headers=headers)
                try:
                    page = r.json()
                except ValueError as exc:
                    raise AcquisitionError(f"GET {url}: not JSON ({exc})", url=url) from exc
                if not isinstance(page, list):
                    raise AcquisitionError(f"GET {url}: expected a list of tree entries", url=url)
                for e in page:
                    if not isinstance(e, Mapping) or e.get("type") != "file":
                        continue
                    path = canonical_path(str(e.get("path") or ""))
                    lfs = e.get("lfs") if isinstance(e.get("lfs"), Mapping) else None
                    sums = {"sha256": str(lfs["oid"]).lower()} if lfs and lfs.get("oid") else \
                        ({"git_sha1": str(e["oid"]).lower()} if e.get("oid") else {})
                    size = e.get("size")
                    files.append(RemoteFile(
                        path, f"{endpoint}/{_PREFIX[kind]}{repo}/resolve/{rev}/{quote(path)}",
                        int(size) if isinstance(size, int) else None, sums))
                nxt = _next_link(r.headers.get("link"))
                url = nxt if nxt and nxt != url else ""
        return sorted_files(files)

    def fetch(self, file: RemoteFile, session: Any, *, offset: int = 0) -> Any:
        return session.stream_bytes(file.url, offset=offset)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.acquisition import AcquisitionCases

        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            out: dict[str, tuple[bytes, str]] = {}
            entries = []
            for i, (rel, body) in enumerate(sorted(files.items())):
                out[f"/datasets/org/repo/resolve/abc123/{quote(rel)}"] = (body, "application/octet-stream")
                if i % 2:                                      # half stored with LFS, half as git blobs
                    entries.append({"type": "file", "path": rel, "size": len(body), "oid": "0" * 40,
                                    "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}})
                else:
                    blob = hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()
                    entries.append({"type": "file", "path": rel, "size": len(body), "oid": blob})
            entries.append({"type": "directory", "path": "sub", "size": 0, "oid": "1" * 40})
            half = len(entries) // 2
            first = "/api/datasets/org/repo/tree/abc123?recursive=true"
            second = "/api/datasets/org/repo/tree/abc123?recursive=true&cursor=p2"
            out[first] = (json.dumps(entries[:half]).encode(), "application/json",
                          {"Link": f'<{root}{second}>; rel="next"'})
            out[second] = (json.dumps(entries[half:]).encode(), "application/json")
            return out

        return AcquisitionCases(publish=publish, options=lambda root: {
            "repo": "org/repo", "revision": "abc123", "endpoint": root})

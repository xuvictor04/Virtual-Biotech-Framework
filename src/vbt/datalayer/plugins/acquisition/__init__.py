"""Acquisition plugins (the ``acquisition`` kind): transports of ``vbt data acquire``. Stdlib and httpx only.

Builtins, each generic (a protocol or a hosting platform; what is specific to a source is its descriptor's
``acquisition`` section):

==============  =========================================================================================
``http``        files under a base URL: an explicit list, the server's HTML directory index, or a checksum list
                (``<hex>  <path>`` lines, e.g. a release's ``release_data_integrity``) as the inventory
``huggingface`` a Hugging Face repository at a pinned revision (tree API; LFS sha256, git blob ids)
``s3``          a public S3 bucket (ListObjectsV2, anonymous; single-part ETag = md5)
``gcs``         a public Google Cloud Storage bucket (JSON API; md5Hash)
``json_index``  files listed by a JSON document (a Zenodo record, a figshare article, any REST listing),
                read with JSONPaths
``zip_member``  members of a remote zip read with HTTP range requests (only the central directory and the
                members selected are transferred)
==============  =========================================================================================

This module holds what they share: :class:`HttpSession` (the one HTTP client of an acquisition, with retries;
``HTTPS_PROXY`` and ``SSL_CERT_FILE`` of the environment apply), path safety, glob matching and checksums.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import re
import time
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..base import CHECKSUM_ALGOS, AcquisitionError, RemoteFile

__all__ = [
    "HttpSession", "AcquisitionError", "RemoteFile", "CHECKSUM_ALGOS", "canonical_path", "safe_join", "glob_match",
    "matches", "Hashers", "file_digests", "strongest", "b64_to_hex", "substitute", "sorted_files", "USER_AGENT",
]

USER_AGENT = "vbt-data-acquire/1"
CHUNK = 1 << 20
TRANSIENT = frozenset({408, 425, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------- paths and patterns


def canonical_path(path: str) -> str:
    """``path`` when it is a canonical relative POSIX path (no ``..``, no absolute or empty parts, no
    backslash or NUL), else :class:`AcquisitionError`: a listing never places a file outside its directory."""
    p = PurePosixPath(path)
    if (not path or p.is_absolute() or ".." in p.parts or "\\" in path or "\x00" in path or p.as_posix() != path
            or path.startswith("./")):
        raise AcquisitionError(f"not a canonical relative path: {path!r}")
    return path


def safe_join(root: Path, rel: str) -> Path:
    """``root / rel`` for a canonical ``rel`` whose directories are not symlinks."""
    target = Path(root) / canonical_path(rel)
    cur = Path(root)
    for part in PurePosixPath(rel).parts[:-1]:
        cur = cur / part
        if cur.is_symlink():
            raise AcquisitionError(f"refusing a symlink in the destination path: {cur}")
    return target


_GLOB_CACHE: dict[str, re.Pattern[str]] = {}


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """A path glob: ``**`` spans directories, ``*`` and ``?`` do not, ``[...]`` is a character class."""
    rx = _GLOB_CACHE.get(pattern)
    if rx is not None:
        return rx
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pattern.find("]", i)
            if j < 0:
                out.append(re.escape(c))
                i += 1
            else:
                out.append(fnmatch.translate(pattern[i:j + 1])[4:-3])  # strip (?s:...)\Z
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    rx = re.compile("".join(out) + r"\Z", re.S)
    _GLOB_CACHE[pattern] = rx
    return rx


def glob_match(path: str, pattern: str) -> bool:
    return bool(_glob_regex(pattern).match(path))


def matches(path: str, include: Sequence[str], exclude: Sequence[str] = ()) -> bool:
    """``path`` matches one ``include`` glob and no ``exclude`` glob."""
    return any(glob_match(path, p) for p in include) and not any(glob_match(path, p) for p in exclude)


def sorted_files(files: Iterable[RemoteFile]) -> list[RemoteFile]:
    """Files by path, each path once (the first listed wins), every path canonical."""
    seen: dict[str, RemoteFile] = {}
    for f in files:
        canonical_path(f.path)
        seen.setdefault(f.path, f)
    return [seen[k] for k in sorted(seen)]


def substitute(value: Any, variables: Mapping[str, str]) -> Any:
    """``{name}`` placeholders of strings (in lists and mappings too) replaced from ``variables``; unknown
    names stay as they are."""
    if isinstance(value, str):
        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                      lambda m: str(variables[m.group(1)]) if m.group(1) in variables else m.group(0), value)
    if isinstance(value, list):
        return [substitute(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, variables) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------- checksums


def b64_to_hex(value: str) -> str:
    return base64.b64decode(value).hex()


class Hashers:
    """Incremental digests of one byte stream for the given algorithms (``git_sha1`` needs the size)."""

    def __init__(self, algos: Iterable[str], *, size: int | None = None) -> None:
        self.algos = [a for a in dict.fromkeys(algos)]
        unknown = [a for a in self.algos if a not in CHECKSUM_ALGOS]
        if unknown:
            raise ValueError(f"unknown checksum algorithm(s) {unknown} (known: {', '.join(CHECKSUM_ALGOS)})")
        self._h: dict[str, Any] = {}
        for a in self.algos:
            if a == "git_sha1":
                if size is None:
                    raise ValueError("git_sha1 needs the size before the content")
                h = hashlib.sha1()
                h.update(b"blob %d\0" % size)
                self._h[a] = h
            elif a != "crc32":
                self._h[a] = hashlib.new(a)
        self._crc = 0
        self.size = 0

    def update(self, chunk: bytes) -> None:
        for h in self._h.values():
            h.update(chunk)
        if "crc32" in self.algos:
            self._crc = zlib.crc32(chunk, self._crc)
        self.size += len(chunk)

    def hexdigests(self) -> dict[str, str]:
        out = {a: h.hexdigest() for a, h in self._h.items()}
        if "crc32" in self.algos:
            out["crc32"] = f"{self._crc & 0xFFFFFFFF:08x}"
        return out


def file_digests(path: Path, algos: Iterable[str]) -> tuple[int, dict[str, str]]:
    """``(bytes, {algo: hex})`` of a file, read once."""
    path = Path(path)
    h = Hashers(algos, size=path.stat().st_size)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * CHUNK), b""):
            h.update(chunk)
    return h.size, h.hexdigests()


def strongest(checksums: Mapping[str, str]) -> tuple[str, str] | None:
    """The strongest ``(algo, hex)`` a listing reported (:data:`CHECKSUM_ALGOS` order)."""
    for a in CHECKSUM_ALGOS:
        if checksums.get(a):
            return a, str(checksums[a]).lower()
    return None


# ---------------------------------------------------------------------------- HTTP


class HttpSession:
    """The HTTP client of one acquisition: redirects followed, ``identity`` encoding (byte counts and ranges are
    the file's), transient failures (connection errors, HTTP 408/425/429/5xx) retried with exponential backoff.
    ``client`` is an ``httpx.Client`` (tests pass their own); ``headers`` are added to every request (a token
    read from the environment for a gated repository, never stored)."""

    def __init__(self, client: Any = None, *, timeout: float = 120.0, retries: int = 4, retry_wait: float = 1.0,
                 headers: Mapping[str, str] | None = None) -> None:
        self._own = client is None
        self._client = client
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.retry_wait = retry_wait
        self.headers = dict(headers or {})
        self.requests = 0
        self.bytes = 0
        #: per-session state a plugin keeps between calls (an open remote archive), closed with the session
        self.cache: dict[Any, Any] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(follow_redirects=True, timeout=self.timeout,
                                        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        return self._client

    def close(self) -> None:
        for obj in list(self.cache.values()):
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing is best effort
                    pass
        self.cache.clear()
        if self._own and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "HttpSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _sleep(self, attempt: int) -> None:
        if self.retry_wait > 0:
            time.sleep(self.retry_wait * min(2 ** attempt, 16))

    def request(self, method: str, url: str, *, headers: Mapping[str, str] | None = None,
                ok: Sequence[int] = (200,)) -> Any:
        """One request with retries; a status outside ``ok`` is :class:`AcquisitionError` (transient for
        408/425/429/5xx after the retries)."""
        import httpx

        last: AcquisitionError | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.client.request(method, url, headers={**self.headers, **dict(headers or {})})
                self.requests += 1
            except httpx.TransportError as exc:
                last = AcquisitionError(f"{method} {url}: {type(exc).__name__}: {exc}", url=url, transient=True)
            else:
                if r.status_code in ok:
                    return r
                last = AcquisitionError(f"{method} {url}: HTTP {r.status_code}", url=url, status=r.status_code,
                                        transient=r.status_code in TRANSIENT)
                if not last.transient:
                    raise last
            if attempt < self.retries:
                self._sleep(attempt)
        assert last is not None
        raise last

    def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> Any:
        r = self.request("GET", url, headers=headers)
        self.bytes += len(r.content)
        return r

    def get_bytes(self, url: str) -> bytes:
        return bytes(self.get(url).content)

    def get_text(self, url: str) -> str:
        return self.get(url).text

    def get_json(self, url: str, *, headers: Mapping[str, str] | None = None) -> Any:
        r = self.get(url, headers=headers)
        try:
            return r.json()
        except ValueError as exc:
            raise AcquisitionError(f"GET {url}: not JSON ({exc})", url=url) from exc

    def head_size(self, url: str) -> int | None:
        """The content length of ``url`` (HEAD; a server that refuses HEAD is asked for byte 0)."""
        try:
            r = self.request("HEAD", url)
            n = r.headers.get("content-length")
            if n is not None:
                return int(n)
        except AcquisitionError as exc:
            if exc.status not in (403, 405, 501):
                raise
        r = self.request("GET", url, headers={"Range": "bytes=0-0"}, ok=(200, 206))
        cr = r.headers.get("content-range", "")
        if r.status_code == 206 and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            return int(cr.rsplit("/", 1)[1])
        n = r.headers.get("content-length")
        return int(n) if n is not None and r.status_code == 200 else None

    def stream_bytes(self, url: str, *, offset: int = 0, chunk: int = CHUNK) -> Iterator[bytes]:
        """The body of ``url`` from ``offset``: a ``Range`` request answered 206 with the matching
        ``Content-Range``; a server that answers 200 to a range is :class:`AcquisitionError` (the caller
        restarts the file). Connection failures before the first byte are retried."""
        import httpx

        headers = {**self.headers, "Range": f"bytes={offset}-"} if offset else dict(self.headers)
        for attempt in range(self.retries + 1):
            try:
                with self.client.stream("GET", url, headers=headers) as r:
                    self.requests += 1
                    if r.status_code == 416 and offset:
                        raise AcquisitionError(f"GET {url}: range from {offset} not satisfiable", url=url,
                                               status=416)
                    if r.status_code not in (200, 206):
                        err = AcquisitionError(f"GET {url}: HTTP {r.status_code}", url=url, status=r.status_code,
                                               transient=r.status_code in TRANSIENT)
                        if not err.transient or attempt >= self.retries:
                            raise err
                        self._sleep(attempt)
                        continue
                    if offset and (r.status_code != 206
                                   or not r.headers.get("content-range", "").startswith(f"bytes {offset}-")):
                        raise AcquisitionError(f"GET {url}: the server ignored the range from {offset} (HTTP "
                                               f"{r.status_code}, Content-Range {r.headers.get('content-range')!r})",
                                               url=url, status=r.status_code)
                    for block in r.iter_bytes(chunk):
                        self.bytes += len(block)
                        yield block
                    return
            except httpx.TransportError as exc:
                raise AcquisitionError(f"GET {url}: {type(exc).__name__}: {exc}", url=url, transient=True) from exc

"""Acquisition conformance suite (the harness-side kind), parametrized over the registered acquisition plugins.

Each plugin's ``conformance_cases()`` returns :class:`AcquisitionCases`: ``publish(files, root)`` renders a set of
files the way the plugin's platform publishes them (index documents and file bodies, keyed by request path and
query) and ``options(root)`` the transport options that point at it. The suite serves them from a local HTTP
server (:class:`FixtureSite`: ``GET``/``HEAD``, byte ranges, a request log), so every case runs offline. A case
runs only when the plugin declares the capabilities it needs; ``variants`` are further publish/options pairs
(``http`` with an HTML index besides its checksum list).

- A-1 the listing is exactly the published files: canonical paths, sorted, no directories, deterministic.
- A-2 ``sizes``: every listed size is the file's; ``checksums``: every file carries a checksum and it matches.
- A-3 ``fetch`` returns the file's bytes; with ``range`` a fetch from an offset returns the rest.
- A-4 a file that is gone, or whose bytes are damaged, raises :class:`~vbt.datalayer.plugins.base.AcquisitionError`
  (a member whose CRC fails included): never an empty or short file read as complete.
- A-5 an index that keeps failing raises ``AcquisitionError``: never an empty listing.
- A-6 a published name that would land outside the destination (``../x``) is dropped or refused.
- A-7 ``describe`` is one non-empty line of at most 200 characters.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator, Mapping

import pytest

from ..base import AcquisitionError
from . import applicable, selected_plugins

__all__ = ["AcquisitionCases", "FixtureSite", "SAMPLE", "site_for", "check_listing", "fetch_all"]

Published = Mapping[str, tuple[Any, ...]]


@dataclass(frozen=True)
class AcquisitionCases:
    publish: Callable[[Mapping[str, bytes], str], Published]
    options: Callable[[str], Mapping[str, Any]]
    variants: Mapping[str, tuple[Callable[[Mapping[str, bytes], str], Published],
                                 Callable[[str], Mapping[str, Any]]]] = field(default_factory=dict)


def _blob(n: int, seed: int) -> bytes:
    out = bytearray()
    h = hashlib.sha256(str(seed).encode()).digest()
    while len(out) < n:
        out.extend(h)
        h = hashlib.sha256(h).digest()
    return bytes(out[:n])


#: The files every plugin publishes: nested directories, an empty file, a binary file over the read chunk.
SAMPLE: dict[str, bytes] = {
    "a.txt": b"alpha\n",
    "dir/b.bin": _blob(1_300_000, 1),
    "dir/sub/c.csv": b"id,value\n1,2\n",
    "dir/sub/empty.dat": b"",
    "z=1/part-0.parquet": _blob(5000, 2),
}


class FixtureSite:
    """A local HTTP server serving ``{path?query: (body, content type[, headers])}`` with byte ranges.
    ``broken`` answers every request 500; ``gone`` paths answer 404; ``requests`` logs ``(method, path, range)``."""

    def __init__(self) -> None:
        self.docs: dict[str, tuple[bytes, str, dict[str, str]]] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        self.broken = False
        self.gone: set[str] = set()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _serve(self, head: bool) -> None:
                rng = self.headers.get("Range")
                outer.requests.append((self.command, self.path, rng))
                doc = outer.docs.get(self.path)
                if outer.broken or doc is None or self.path in outer.gone:
                    code = 500 if outer.broken else 404
                    self.send_response(code)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body, ctype, extra = doc
                start, end = 0, len(body) - 1
                if rng and rng.startswith("bytes="):
                    a, _, b = rng[6:].partition("-")
                    start = int(a)
                    end = min(int(b), len(body) - 1) if b else len(body) - 1
                    if start >= len(body) and len(body):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(body)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                part = body[start:end + 1]
                self.send_response(206 if rng else 200)
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{max(end, start)}/{len(body)}")
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(part)))
                self.send_header("Accept-Ranges", "bytes")
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                if not head:
                    self.wfile.write(part)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                self._serve(False)

            def do_HEAD(self) -> None:  # noqa: N802 - http.server API
                self._serve(True)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.root = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def publish(self, docs: Published) -> None:
        self.docs = {}
        for key, value in docs.items():
            body, ctype, *rest = value
            self.docs[key] = (bytes(body), str(ctype), dict(rest[0]) if rest else {})

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def site_for(cases: AcquisitionCases, files: Mapping[str, bytes], variant: str | None = None
             ) -> tuple[FixtureSite, dict[str, Any]]:
    """A running site publishing ``files`` the plugin's way, and the options that point at it."""
    publish, options = (cases.publish, cases.options) if variant is None else cases.variants[variant]
    site = FixtureSite()
    site.publish(publish(files, site.root))
    return site, dict(options(site.root))


def _session() -> Any:
    from ..acquisition import HttpSession

    return HttpSession(retries=2, retry_wait=0)


def fetch_all(plugin: Any, f: Any, session: Any, offset: int = 0) -> bytes:
    return b"".join(plugin.fetch(f, session, offset=offset))


def check_listing(plugin: Any, files: list[Any], published: Mapping[str, bytes]) -> None:
    """A-1 and A-2 for one listing."""
    from ..acquisition import Hashers, canonical_path

    paths = [f.path for f in files]
    assert paths == sorted(published), f"{plugin.name}: listed {paths}, published {sorted(published)}"
    for f in files:
        assert canonical_path(f.path) == f.path
        body = published[f.path]
        if applicable(plugin, "sizes"):
            assert f.size == len(body), f"{plugin.name}: {f.path} size {f.size} != {len(body)}"
        if applicable(plugin, "checksums"):
            assert f.checksums, f"{plugin.name}: {f.path} has no checksum"
            h = Hashers(f.checksums, size=len(body))
            h.update(body)
            assert h.hexdigests() == {k: str(v).lower() for k, v in f.checksums.items()}, f.path


ACQUISITION_PLUGINS = selected_plugins("acquisition")


def _cases(plugin: Any) -> AcquisitionCases:
    cases = plugin.conformance_cases()
    if not isinstance(cases, AcquisitionCases):
        pytest.fail(f"{plugin.name}: conformance_cases() must return AcquisitionCases")
    return cases


def _variant_params() -> list[Any]:
    out = []
    for p in ACQUISITION_PLUGINS:
        out.append(pytest.param(p, None, id=p.name))
        out.extend(pytest.param(p, v, id=f"{p.name}-{v}") for v in _cases(p).variants)
    return out


@pytest.fixture
def _site_cleanup() -> Iterator[list[FixtureSite]]:
    sites: list[FixtureSite] = []
    yield sites
    for s in sites:
        s.close()


@pytest.mark.parametrize("plugin, variant", _variant_params())
def test_a1_a2_listing_is_exactly_the_published_files(plugin: Any, variant: str | None,
                                                      _site_cleanup: list[FixtureSite]) -> None:
    site, options = site_for(_cases(plugin), SAMPLE, variant)
    _site_cleanup.append(site)
    with _session() as session:
        first = plugin.listing(options, session)
        check_listing(plugin, first, SAMPLE)
        assert [(f.path, f.size, dict(f.checksums)) for f in plugin.listing(options, session)] == \
            [(f.path, f.size, dict(f.checksums)) for f in first], "listing is not deterministic"


@pytest.mark.parametrize("plugin, variant", _variant_params())
def test_a3_fetch_returns_the_bytes(plugin: Any, variant: str | None, _site_cleanup: list[FixtureSite]) -> None:
    site, options = site_for(_cases(plugin), SAMPLE, variant)
    _site_cleanup.append(site)
    with _session() as session:
        for f in plugin.listing(options, session):
            assert fetch_all(plugin, f, session) == SAMPLE[f.path], f.path
            if applicable(plugin, "range") and len(SAMPLE[f.path]) > 3:
                k = len(SAMPLE[f.path]) // 3
                assert fetch_all(plugin, f, session, k) == SAMPLE[f.path][k:], f"{f.path} from {k}"


@pytest.mark.parametrize("plugin", [pytest.param(p, id=p.name) for p in ACQUISITION_PLUGINS])
def test_a4_a_missing_or_damaged_file_raises(plugin: Any, _site_cleanup: list[FixtureSite]) -> None:
    cases = _cases(plugin)
    site, options = site_for(cases, SAMPLE)
    _site_cleanup.append(site)
    with _session() as session:
        files = {f.path: f for f in plugin.listing(options, session)}
    target = files["dir/b.bin"]
    if applicable(plugin, "members"):
        damaged = {k: (v[0], *v[1:]) for k, v in cases.publish(SAMPLE, site.root).items()}
        key = next(iter(damaged))
        body = bytearray(damaged[key][0])
        at = bytes(body).find(SAMPLE["dir/b.bin"][:64])
        at = at if at >= 0 else len(body) // 2
        body[at + 100] ^= 0xFF
        site.publish({k: ((bytes(body), *v[1:]) if k == key else v) for k, v in damaged.items()})
    else:
        site.gone = {path for path, _ in site.docs.items() if site.docs[path][0] == SAMPLE["dir/b.bin"]}
        assert site.gone, "the fixture does not serve dir/b.bin as its own document"
    with _session() as session, pytest.raises(AcquisitionError):
        data = fetch_all(plugin, target, session)
        assert data == SAMPLE["dir/b.bin"], "a damaged file read as different bytes without an error"


@pytest.mark.parametrize("plugin", [pytest.param(p, id=p.name) for p in ACQUISITION_PLUGINS])
def test_a5_an_index_that_fails_raises(plugin: Any, _site_cleanup: list[FixtureSite]) -> None:
    site, options = site_for(_cases(plugin), SAMPLE)
    _site_cleanup.append(site)
    site.broken = True
    with _session() as session, pytest.raises(AcquisitionError):
        plugin.listing(options, session)


@pytest.mark.parametrize("plugin, variant", _variant_params())
def test_a6_names_outside_the_destination_never_list(plugin: Any, variant: str | None,
                                                      _site_cleanup: list[FixtureSite]) -> None:
    hostile = {"ok.txt": b"fine", "../evil.txt": b"escape", "sub/../../evil2.txt": b"escape"}
    site, options = site_for(_cases(plugin), hostile, variant)
    _site_cleanup.append(site)
    with _session() as session:
        try:
            files = plugin.listing(options, session)
        except AcquisitionError:
            return                                     # refused: acceptable
    assert [f.path for f in files] == ["ok.txt"], [f.path for f in files]


@pytest.mark.parametrize("plugin, variant", _variant_params())
def test_a7_describe(plugin: Any, variant: str | None, _site_cleanup: list[FixtureSite]) -> None:
    site, options = site_for(_cases(plugin), {"a.txt": b"x"}, variant)
    _site_cleanup.append(site)
    text = plugin.describe(options)
    assert text and "\n" not in text and len(text) <= 200

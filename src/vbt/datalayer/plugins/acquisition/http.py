"""``http``: files under a base URL, listed explicitly, by the server's HTML directory index, or by a checksum list.

Options (``acquisition.transport.options``)::

    base:      https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{release}/output/
    files:     [go-basic.obo, ...] | {name: {bytes: 32227785, sha256: ..., url: ...}}    an explicit list
               (a file's `url` overrides base + name; `sha256`/`sha1`/`md5` pin its content)
    listing:   checksums | html | none          default: checksums with `checksums`, none with `files`, else html
    checksums:                                  a list of "<hex>  <path>" lines (sha1sum/sha256sum/md5sum format)
      url:     https://.../{release}/release_data_integrity
      algo:    sha1
      prefix:  ./output/                        list paths under this prefix are files under `base` (stripped)
      verify:  https://.../{release}/release_data_integrity.sha1    the list's own checksum (first token, `algo`)

With ``listing: checksums`` the list is the inventory (no directory index is walked) and the checksum source; with
``html`` it only adds checksums to the files the index lists. The list and its ``verify`` file are kept in the
``index_cache`` directory (``acquisition.index_cache``, e.g. ``_release/``) and read back while they verify, so a
manifest can be rewritten offline. A list that does not match its ``verify`` checksum is refused, as is an index
link that leaves ``base``.
"""

from __future__ import annotations

import hashlib
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urljoin, urlparse

from ..base import AcquisitionBase, AcquisitionError, RemoteFile
from ..registry import register
from . import canonical_path, sorted_files

__all__ = ["HttpTransport", "parse_checksum_list", "load_checksum_list"]


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href" and v)


def parse_checksum_list(text: str, *, prefix: str = "", algo: str = "sha1") -> dict[str, str]:
    """``{path below prefix: hex}`` of a ``<hex>  <path>`` list (``*`` binary markers and ``./`` allowed);
    lines of other lengths or outside ``prefix`` are skipped, a path that is not canonical is refused."""
    width = {"sha1": 40, "sha256": 64, "md5": 32}.get(algo)
    if width is None:
        raise ValueError(f"a checksum list holds sha1, sha256 or md5, not {algo!r}")
    out: dict[str, str] = {}
    for line in text.splitlines():
        digest, _, path = line.strip().partition(" ")
        path = path.lstrip(" ").lstrip("*")
        if len(digest) != width or not path:
            continue
        if prefix:
            if not path.startswith(prefix):
                continue
            path = path[len(prefix):]
        elif path.startswith("./"):
            path = path[2:]
        out[canonical_path(path)] = digest.lower()
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise AcquisitionError(f"refusing to write through a symlink: {path}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def load_checksum_list(spec: Mapping[str, Any], session: Any, index_cache: Path | None = None
                       ) -> tuple[bytes, dict[str, Any]]:
    """``(raw list, about)``: the list from ``index_cache`` while it matches its kept ``verify`` checksum, else
    downloaded, checked against ``verify`` and kept there. ``about`` = ``{url, <algo>, from}``."""
    url = str(spec["url"])
    algo = str(spec.get("algo") or "sha1")
    verify = spec.get("verify")
    name = url.rstrip("/").rsplit("/", 1)[-1] or "checksums"
    vname = str(verify).rstrip("/").rsplit("/", 1)[-1] if verify else None
    cache = Path(index_cache) if index_cache is not None else None
    if cache is not None and (cache / name).is_file() and (vname is None or (cache / vname).is_file()):
        raw = (cache / name).read_bytes()
        got = hashlib.new(algo, raw).hexdigest()
        want = (cache / vname).read_text().split()[0].lower() if vname else got
        if got == want:
            return raw, {"url": url, algo: got, "from": str(cache / name)}
    want = session.get_text(str(verify)).split()[0].lower() if verify else None
    raw = session.get_bytes(url)
    got = hashlib.new(algo, raw).hexdigest()
    if want is not None and got != want:
        raise AcquisitionError(f"{url}: {algo} {got} differs from {verify} ({want})", url=url)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        _atomic_write(cache / name, raw)
        if vname:
            _atomic_write(cache / vname, f"{got}  {name}\n".encode())
    return raw, {"url": url, algo: got, "from": url}


@register
class HttpTransport(AcquisitionBase):
    name = "http"
    version = "1.0"
    capabilities = frozenset({"range", "checksums", "index"})

    def describe(self, options: Mapping[str, Any]) -> str:
        how = self._mode(options)
        extra = f" (inventory and checksums: {options['checksums'].get('url')})" if how == "checksums" else \
            (" (HTML directory index)" if how == "html" else "")
        where = str(options.get("base") or "")
        if not where:
            listed = options.get("files") or {}
            urls = [str((v or {}).get("url") or "") for v in listed.values()] if isinstance(listed, Mapping) else []
            hosts = sorted({urlparse(u).netloc for u in urls if u})
            where = f"{len(urls)} file URL(s) on {', '.join(hosts) or '?'}"
        return f"https {where}{extra}"[:200]

    @staticmethod
    def _mode(options: Mapping[str, Any]) -> str:
        mode = options.get("listing")
        if mode:
            if mode not in ("checksums", "html", "none"):
                raise AcquisitionError(f"http listing must be checksums, html or none, not {mode!r}")
            return str(mode)
        if options.get("checksums") and not options.get("files"):
            return "checksums"
        return "none" if options.get("files") else "html"

    def listing(self, options: Mapping[str, Any], session: Any, *, index_cache: Any = None) -> list[RemoteFile]:
        base = str(options.get("base") or "")
        mode = self._mode(options)
        if not base and mode != "none":
            raise AcquisitionError("http transport: `base` is required to list a directory or a checksum list")
        base = base if not base or base.endswith("/") else base + "/"
        sums: dict[str, str] = {}
        algo = None
        about: dict[str, Any] = {}
        if options.get("checksums"):
            spec = dict(options["checksums"])
            algo = str(spec.get("algo") or "sha1")
            raw, about = load_checksum_list(spec, session, Path(index_cache) if index_cache else None)
            sums = parse_checksum_list(raw.decode("utf-8"), prefix=str(spec.get("prefix") or ""), algo=algo)
        files: dict[str, RemoteFile] = {}
        extra = {"index": about} if about else {}
        if mode == "checksums":
            if not options.get("checksums"):
                raise AcquisitionError("http listing: checksums needs a `checksums` list")
            for rel, digest in sums.items():
                files[rel] = RemoteFile(rel, base + rel, None, {str(algo): digest}, extra=extra)
        elif mode == "html":
            for rel in self._walk(base, session):
                files[rel] = RemoteFile(rel, base + rel, None, {str(algo): sums[rel]} if rel in sums else {},
                                        extra=extra)
        listed = options.get("files") or []
        entries = listed.items() if isinstance(listed, Mapping) else ((f, {}) for f in listed)
        for rel, info in entries:
            rel = canonical_path(str(rel))
            info = dict(info or {})
            known = {a: str(v).lower() for a, v in info.items() if a in ("sha256", "sha1", "md5")}
            if rel in sums and algo:
                known.setdefault(algo, sums[rel])
            size = info.get("bytes")
            url = str(info.get("url") or "") or (base + rel if base else "")
            if not url:
                raise AcquisitionError(f"http transport: {rel} has no `url` and there is no `base`")
            files[rel] = RemoteFile(rel, url, int(size) if size is not None else None, known, extra=extra)
        return sorted_files(files.values())

    def _walk(self, base: str, session: Any) -> list[str]:
        """Files under ``base`` reachable through its HTML index pages (directories end in ``/``)."""
        pending, seen, found = [base], set(), set()
        while pending:
            page = pending.pop()
            if page in seen:
                continue
            seen.add(page)
            parser = _Links()
            parser.feed(session.get_text(page))
            for href in parser.links:
                url = urljoin(page, href)
                parsed = urlparse(url)
                if not url.startswith(base) or parsed.query or parsed.fragment or url == page:
                    continue
                rel = unquote(url[len(base):])
                if url.endswith("/"):
                    if url not in seen:
                        pending.append(url)
                    continue
                try:
                    found.add(canonical_path(rel))
                except AcquisitionError:
                    continue
        return sorted(found)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.acquisition import AcquisitionCases

        def publish(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            out: dict[str, tuple[bytes, str]] = {}
            lines = []
            for rel, body in files.items():
                out[f"/out/{rel}"] = (body, "application/octet-stream")
                lines.append(f"{hashlib.sha1(body).hexdigest()}  ./out/{rel}")
            text = ("\n".join(lines) + "\n").encode()
            out["/list.txt"] = (text, "text/plain")
            out["/list.txt.sha1"] = (f"{hashlib.sha1(text).hexdigest()}  list.txt\n".encode(), "text/plain")
            return out

        def options(root: str) -> dict[str, Any]:
            return {"base": f"{root}/out/", "listing": "checksums",
                    "checksums": {"url": f"{root}/list.txt", "algo": "sha1", "prefix": "./out/",
                                  "verify": f"{root}/list.txt.sha1"}}

        def publish_html(files: Mapping[str, bytes], root: str) -> dict[str, tuple[bytes, str]]:
            out: dict[str, tuple[bytes, str]] = {}
            dirs: dict[str, set[str]] = {"": set()}
            sums = "".join(f"{hashlib.sha256(b).hexdigest()} *./web/{rel}\n" for rel, b in files.items()
                           if ".." not in rel)
            out["/SHA256SUMS"] = (sums.encode(), "text/plain")
            for rel, body in files.items():
                out[f"/web/{rel}"] = (body, "application/octet-stream")
                parts = rel.split("/")
                for i in range(len(parts)):
                    d = "/".join(parts[:i])
                    dirs.setdefault(d, set()).add(parts[i] + ("/" if i < len(parts) - 1 else ""))
            for d, entries in dirs.items():
                links = "".join(f'<a href="{e}">{e}</a>\n' for e in sorted(entries))
                page = f'<html><body><a href="../">Parent</a>\n<a href="?C=M;O=A">sort</a>\n{links}</body></html>'
                out[f"/web/{d}/" if d else "/web/"] = (page.encode(), "text/html")
            return out

        return AcquisitionCases(publish=publish, options=options, variants={"html": (publish_html, lambda root: {
            "base": f"{root}/web/", "listing": "html",
            "checksums": {"url": f"{root}/SHA256SUMS", "algo": "sha256", "prefix": "./web/"}})})

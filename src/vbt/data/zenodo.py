"""The paper's Zenodo case-study archive (doi:10.5281/zenodo.22259123).

The GitHub repository of *The Virtual Biotech* ships the agentic system but
not the case-study analysis code; the authors deposited that (code, result
tables, agent reports, data subsets) as a single 2.9 GB zip on Zenodo
(record 22259123, ``virtualbiotech_submission.zip``, CC-BY-4.0).  Zenodo
serves the file with HTTP ``Range`` support, so :class:`HTTPRangeFile` lets
:mod:`zipfile` read the central directory and extract individual members
without downloading the whole archive.

* :class:`ZenodoArchive` — resolve the file link from the records API,
  :meth:`~ZenodoArchive.list` members, :meth:`~ZenodoArchive.fetch` a preset
  or glob selection (skipping files already present with the same size and
  zip CRC32; atomic writes), or :meth:`~ZenodoArchive.download` the whole zip
  with md5 verification against the record checksum.
* :data:`PRESETS` — named selections (``case1``, ``b7h3-code`` ...).
* :func:`zenodo_root` — where the extracted archive lives
  (``${VBT_ZENODO_DIR:-<project>/data/zenodo}/virtualbiotech_submission``).

Networking uses plain ``httpx`` defaults, so ``HTTPS_PROXY`` / ``SSL_CERT_FILE``
from the environment are honoured.  Tests inject a local server via
``api_base``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import os
import re
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

RECORD_ID = 22259123
DOI = "10.5281/zenodo.22259123"
FILE_KEY = "virtualbiotech_submission.zip"
API_BASE = "https://zenodo.org/api"
ARCHIVE_TOP = "virtualbiotech_submission"

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".tif", ".tiff", ".webp", ".bmp", ".eps")
SMALL_TEXT_EXT = (".csv", ".tsv", ".json", ".md", ".txt", ".yaml", ".yml")

__all__ = ["RECORD_ID", "DOI", "PRESETS", "HTTPRangeFile", "ZenodoArchive", "FetchReport",
           "zenodo_root", "select_members", "glob_match", "preset_names"]


# ---------------------------------------------------------------------------
# Presets and glob matching
# ---------------------------------------------------------------------------

#: name -> selection. Patterns are matched against member paths *relative to
#: the archive's top folder* (``clinical_trials/data/...``). ``exclude_ext``
#: drops file extensions; ``max_bytes`` drops larger members.
PRESETS: dict[str, dict[str, Any]] = {
    "case1": {"include": ["clinical_trials/code/**", "clinical_trials/results/**", "clinical_trials/data/**",
                          "clinical_trials/benchmarks/*/*.csv", "README.md"],
              "help": "Case 1: authors' code, result tables, data subsets, competitor annotations"},
    "b7h3-code": {"include": ["b7-h3/code/**", "b7-h3/run_pipeline.sh", "README.md"],
                  "help": "B7-H3 analysis scripts"},
    "b7h3-results": {"include": ["b7-h3/data/**"], "only_ext": SMALL_TEXT_EXT, "exclude": ["**/*.h5ad"],
                     "help": "B7-H3 small result tables (csv/json/md) incl. TCGA inputs; not the 2.5 GB h5ad"},
    "b7h3-scrnaseq": {"include": ["b7-h3/data/inputs/*.h5ad"],
                      "help": "the 2.5 GB B7-H3 lung scRNA-seq h5ad (explicit opt-in)"},
    "osmr-code": {"include": ["osmr/code/*.py", "osmr/run_pipeline.sh", "README.md"],
                  "help": "OSMR analysis scripts"},
    "osmr-results": {"include": ["osmr/code/*.tsv", "osmr/code/data/**", "osmr/data/**"],
                     "help": "OSMR result tables, processed bulk cohorts (h5ad), inputs"},
    "traces": {"include": ["*/traces/**"], "help": "specialist agent reports"},
    "benchmarks": {"include": ["*/benchmarks/**"], "exclude_ext": IMAGE_EXT,
                   "help": "Biomni / Kosmos / PantheonOS comparison runs (no images)"},
    "system-snapshot": {"include": ["TheVirtualBiotech_github/**"],
                        "help": "frozen snapshot of the agentic-system source"},
    "small": {"include": ["**"], "max_bytes": 5 * 1024 * 1024, "exclude_ext": IMAGE_EXT,
              "help": "every non-image member under 5 MB"},
    "all": {"include": ["**"], "help": "the whole archive"},
}


def preset_names() -> list[str]:
    return list(PRESETS)


_GLOB_CACHE: dict[str, re.Pattern] = {}


def _glob_regex(pattern: str) -> re.Pattern:
    """Translate a path glob: ``**`` spans directories, ``*``/``?`` do not."""
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


def _rel(name: str) -> str:
    """Member path relative to the archive top folder."""
    top = ARCHIVE_TOP + "/"
    return name[len(top):] if name.startswith(top) else name


def select_members(infos: Iterable[zipfile.ZipInfo], patterns: Sequence[str] | None = None,
                   preset: str | None = None, max_member_mb: float | None = None) -> list[zipfile.ZipInfo]:
    """Members (files only) matching ``patterns`` and/or ``preset``.

    Patterns match the path relative to the top folder or the full member
    name. With neither patterns nor preset, nothing is selected.
    """
    if preset is not None and preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {', '.join(PRESETS)}")
    spec = PRESETS.get(preset or "", {})
    include = list(spec.get("include", [])) + list(patterns or [])
    if not include:
        return []
    exclude = list(spec.get("exclude", []))
    excl_ext = tuple(spec.get("exclude_ext", ()))
    only_ext = tuple(spec.get("only_ext", ()))
    max_bytes = spec.get("max_bytes")
    if max_member_mb is not None:
        cap = int(max_member_mb * 1024 * 1024)
        max_bytes = cap if max_bytes is None else min(max_bytes, cap)
    preset_inc = list(spec.get("include", []))
    out = []
    for info in infos:
        if info.is_dir():
            continue
        rel = _rel(info.filename)
        by_pattern = any(glob_match(rel, p) or glob_match(info.filename, p) for p in (patterns or []))
        by_preset = any(glob_match(rel, p) for p in preset_inc)
        if by_preset and not by_pattern:
            low = rel.lower()
            if any(glob_match(rel, p) for p in exclude):
                continue
            if excl_ext and low.endswith(excl_ext):
                continue
            if only_ext and not low.endswith(only_ext):
                continue
        if not (by_pattern or by_preset):
            continue
        if max_bytes is not None and info.file_size > max_bytes:
            continue
        out.append(info)
    return out


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def zenodo_root(config: dict[str, Any] | None = None) -> Path:
    """Directory of the extracted archive (``.../virtualbiotech_submission``).

    ``$VBT_ZENODO_DIR`` (the directory *containing* ``virtualbiotech_submission``)
    wins, then ``config['zenodo']['dir']``, then ``<project>/data/zenodo``.
    """
    from ..config import PROJECT_ROOT

    base = os.environ.get("VBT_ZENODO_DIR") or ((config or {}).get("zenodo") or {}).get("dir")
    base_p = Path(base).expanduser() if base else PROJECT_ROOT / "data" / "zenodo"
    if not base_p.is_absolute():
        base_p = PROJECT_ROOT / base_p
    if base_p.name == ARCHIVE_TOP:
        return base_p
    return base_p / ARCHIVE_TOP


# ---------------------------------------------------------------------------
# HTTP range file
# ---------------------------------------------------------------------------


class RangeNotSupported(IOError):
    pass


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP ``Range`` requests.

    Wrap in :class:`io.BufferedReader` (``ZenodoArchive`` does) so small reads
    are served from a buffer. Transient failures (transport errors, 429, 5xx)
    are retried with exponential backoff.
    """

    def __init__(self, url: str, *, client=None, size: int | None = None, retries: int = 4,
                 backoff: float = 0.5, timeout: float = 120.0):
        import httpx

        self._own = client is None
        self.client = client or httpx.Client(follow_redirects=True, timeout=timeout)
        self.url = url
        self.retries = retries
        self.backoff = backoff
        self.pos = 0
        self.requests = 0
        self.bytes_read = 0
        self.size = size if size is not None else self._probe_size()

    # -- helpers
    def _request(self, method: str, headers: dict[str, str] | None = None):
        import httpx

        delay = self.backoff
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.client.request(method, self.url, headers=headers or {})
                self.requests += 1
                if r.status_code in (429, 500, 502, 503, 504):
                    last = IOError(f"HTTP {r.status_code}")
                else:
                    return r
            except httpx.TransportError as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(delay)
                delay *= 2
        raise IOError(f"request failed after {self.retries + 1} attempts: {last}")

    def _probe_size(self) -> int:
        r = self._request("HEAD")
        if r.status_code < 400 and r.headers.get("content-length"):
            self.url = str(r.url)
            return int(r.headers["content-length"])
        r = self._request("GET", {"Range": "bytes=0-0"})
        cr = r.headers.get("content-range", "")
        if r.status_code == 206 and "/" in cr:
            self.url = str(r.url)
            return int(cr.rsplit("/", 1)[1])
        raise RangeNotSupported(f"cannot determine size of {self.url} (HTTP {r.status_code})")

    # -- io API
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        if self.pos < 0:
            raise ValueError("negative seek position")
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = self._request("GET", {"Range": f"bytes={self.pos}-{end}"})
        if r.status_code != 206:
            raise RangeNotSupported(f"server ignored Range (HTTP {r.status_code}) for {self.url}")
        data = r.content
        self.pos += len(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self) -> None:
        if self._own and not self.closed:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
        super().close()


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


@dataclass
class FetchReport:
    dest: Path
    fetched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    bytes_fetched: int = 0
    bytes_transferred: int = 0  # HTTP payload incl. zip directory / headers
    selected: int = 0

    def summary(self) -> str:
        return (f"{self.selected} members selected: {len(self.fetched)} fetched "
                f"({self.bytes_fetched / 1e6:.1f} MB uncompressed, {self.bytes_transferred / 1e6:.1f} MB "
                f"transferred), {len(self.skipped)} already present -> {self.dest}")

    def to_dict(self) -> dict[str, Any]:
        return dict(dest=str(self.dest), fetched=self.fetched, skipped=self.skipped,
                    bytes_fetched=self.bytes_fetched, bytes_transferred=self.bytes_transferred,
                    selected=self.selected)


def _crc32_file(path: Path) -> int:
    crc = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def _safe_target(dest: Path, name: str) -> Path:
    parts = Path(name).parts
    if not name or name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
        raise ValueError(f"unsafe member path in archive: {name!r}")
    return dest / Path(*parts)


class ZenodoArchive:
    """One file of a Zenodo record, read remotely as a zip."""

    def __init__(self, record_id: int = RECORD_ID, *, file_key: str | None = FILE_KEY,
                 api_base: str = API_BASE, client=None, timeout: float = 120.0, retries: int = 4,
                 backoff: float = 0.5, buffer_size: int = 1 << 20):
        self.record_id = record_id
        self.file_key = file_key
        self.api_base = api_base.rstrip("/")
        self._client = client
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.buffer_size = buffer_size
        self._file: dict[str, Any] | None = None
        self._zip: zipfile.ZipFile | None = None
        self._raw: HTTPRangeFile | None = None

    # -- resolution
    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(follow_redirects=True, timeout=self.timeout)
        return self._client

    def resolve(self) -> dict[str, Any]:
        """``{key, url, size, md5}`` of the record file (records API)."""
        if self._file is not None:
            return self._file
        url = f"{self.api_base}/records/{self.record_id}"
        r = self.client.get(url)
        r.raise_for_status()
        rec = r.json()
        files = rec.get("files") or []
        if isinstance(files, dict):  # newer API shape {"entries": {...}}
            files = list((files.get("entries") or {}).values())
        if not files:
            raise IOError(f"Zenodo record {self.record_id} lists no files")
        f = next((x for x in files if x.get("key") == self.file_key), None) if self.file_key else None
        f = f or files[0]
        links = f.get("links") or {}
        link = links.get("self") or links.get("content") or links.get("download")
        if not link:
            raise IOError(f"no download link for {f.get('key')} in record {self.record_id}")
        md5 = None
        cs = f.get("checksum") or ""
        if cs.startswith("md5:"):
            md5 = cs[4:]
        self._file = dict(key=f.get("key"), url=link, size=f.get("size"), md5=md5)
        return self._file

    # -- zip access
    def open_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            info = self.resolve()
            self._raw = HTTPRangeFile(info["url"], client=self.client, size=info.get("size"),
                                      retries=self.retries, backoff=self.backoff)
            self._zip = zipfile.ZipFile(io.BufferedReader(self._raw, buffer_size=self.buffer_size))
        return self._zip

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def list(self, pattern: str | None = None) -> list[zipfile.ZipInfo]:
        infos = [i for i in self.open_zip().infolist() if not i.is_dir()]
        if pattern:
            infos = [i for i in infos if glob_match(_rel(i.filename), pattern) or glob_match(i.filename, pattern)]
        return infos

    def fetch(self, patterns: Sequence[str] | str | None = None, preset: str | None = None,
              dest: str | Path = Path("data/zenodo"), max_member_mb: float | None = None,
              dry_run: bool = False, on_progress=None) -> FetchReport:
        """Extract the selected members under ``dest`` (keeping their archive paths).

        Members already present with the same size and CRC32 are skipped;
        each file is written to a temporary name and renamed into place.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        dest = Path(dest)
        zf = self.open_zip()
        start = self._raw.bytes_read if self._raw else 0
        sel = select_members(zf.infolist(), patterns, preset, max_member_mb)
        rep = FetchReport(dest=dest, selected=len(sel))
        for info in sel:
            target = _safe_target(dest, info.filename)
            if target.exists() and target.stat().st_size == info.file_size and _crc32_file(target) == info.CRC:
                rep.skipped.append(info.filename)
                continue
            if dry_run:
                rep.fetched.append(info.filename)
                rep.bytes_fetched += info.file_size
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.part{os.getpid()}")
            crc = 0
            try:
                with zf.open(info) as src, open(tmp, "wb") as out:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        crc = zlib.crc32(chunk, crc)
                        out.write(chunk)
                if (crc & 0xFFFFFFFF) != info.CRC:
                    raise IOError(f"CRC mismatch for {info.filename}")
                os.replace(tmp, target)
            finally:
                if tmp.exists():
                    tmp.unlink()
            rep.fetched.append(info.filename)
            rep.bytes_fetched += info.file_size
            if on_progress:
                on_progress(info.filename, info.file_size)
        rep.bytes_transferred = (self._raw.bytes_read - start) if self._raw else 0
        return rep

    def download(self, dest: str | Path, *, verify_md5: bool = True, chunk: int = 1 << 22) -> Path:
        """Stream the whole zip to ``dest`` (a directory or file path) and verify its md5."""
        info = self.resolve()
        dest = Path(dest)
        target = dest / info["key"] if dest.is_dir() or not dest.suffix else dest
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.part")
        h = hashlib.md5()
        with self.client.stream("GET", info["url"]) as r:
            r.raise_for_status()
            with open(tmp, "wb") as out:
                for block in r.iter_bytes(chunk):
                    h.update(block)
                    out.write(block)
        if verify_md5 and info.get("md5") and h.hexdigest() != info["md5"]:
            tmp.unlink()
            raise IOError(f"md5 mismatch: got {h.hexdigest()}, record says {info['md5']}")
        os.replace(tmp, target)
        return target


# ---------------------------------------------------------------------------
# CLI (`vbt data zenodo ...`)
# ---------------------------------------------------------------------------


def add_data_parsers(sub) -> None:
    d = sub.add_parser("data", help="fetch external data (the paper's Zenodo case-study archive, Open Targets "
                                    "release tables)")
    ds = d.add_subparsers(dest="data_source", required=True)
    from .opentargets import add_ot_parser

    add_ot_parser(ds)                    # vbt data ot list|fetch|manifest
    z = ds.add_parser("zenodo", help=f"Zenodo record {RECORD_ID} ({FILE_KEY}, doi:{DOI})")
    zs = z.add_subparsers(dest="zenodo_action", required=True)

    def common(p):
        p.add_argument("--record", type=int, default=RECORD_ID)
        p.add_argument("--api-base", default=API_BASE, help=argparse_suppress())

    lp = zs.add_parser("list", help="list archive members (central directory only, a few hundred KB)")
    lp.add_argument("--pattern", help="glob on member paths, e.g. 'clinical_trials/**'")
    lp.add_argument("--preset", choices=list(PRESETS))
    common(lp)
    fp = zs.add_parser("fetch", help="extract selected members via HTTP range requests")
    fp.add_argument("--preset", choices=list(PRESETS))
    fp.add_argument("--pattern", action="append", default=[], help="glob (repeatable)")
    fp.add_argument("--dest", default=None, help="directory (default: $VBT_ZENODO_DIR or data/zenodo)")
    fp.add_argument("--max-member-mb", type=float)
    fp.add_argument("--dry-run", action="store_true")
    common(fp)
    dp = zs.add_parser("download", help="download the whole 2.9 GB zip and verify its md5")
    dp.add_argument("--dest", default=None)
    dp.add_argument("--no-verify", action="store_true")
    common(dp)
    pp = zs.add_parser("presets", help="list the named selections")
    del pp
    z.set_defaults(handler=_cmd_zenodo)


def argparse_suppress():
    import argparse

    return argparse.SUPPRESS


def _cmd_zenodo(args, config: dict[str, Any]) -> int:
    if args.zenodo_action == "presets":
        for k, v in PRESETS.items():
            print(f"{k:16s} {v.get('help', '')}")
        return 0
    dest = Path(args.dest) if getattr(args, "dest", None) else zenodo_root(config).parent
    with ZenodoArchive(args.record, api_base=args.api_base) as arch:
        if args.zenodo_action == "list":
            infos = arch.list(args.pattern)
            if args.preset:
                infos = select_members(infos, None, args.preset)
            total = 0
            for i in infos:
                total += i.file_size
                print(f"{i.file_size}\t{i.filename}")
            print(f"# {len(infos)} members, {total / 1e6:.1f} MB")
            return 0
        if args.zenodo_action == "fetch":
            if not args.preset and not args.pattern:
                print("error: give --preset or --pattern", flush=True)
                return 2
            rep = arch.fetch(args.pattern or None, preset=args.preset, dest=dest,
                             max_member_mb=args.max_member_mb, dry_run=args.dry_run,
                             on_progress=lambda name, n: print(f"  {n:>12,d}  {name}"))
            print(rep.summary())
            return 0
        if args.zenodo_action == "download":
            path = arch.download(dest, verify_md5=not args.no_verify)
            print(f"downloaded {path}")
            return 0
    return 2

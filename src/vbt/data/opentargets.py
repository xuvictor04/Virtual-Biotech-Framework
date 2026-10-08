"""Open Targets Platform release tables: fetch selected tables, verified against the publisher's sha1 list.

A release directory (``https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/<release>/``) holds the Parquet
tables under ``output/<table>/`` (``evidence`` is hive-partitioned by ``sourceId=``) and ``release_data_integrity``:
one ``<sha1>  ./<path>`` line for every file of the release (25.09: 22,557 lines, 3,508 of them output Parquet
files in 38 tables), itself checked by ``release_data_integrity.sha1``. That list is the inventory and the
checksum source here, so no directory listing is walked.

* :func:`load_integrity` reads the list (from ``<dest>/_release/`` when present, else from the release) and checks
  it against its ``.sha1``; :func:`inventory` groups its Parquet files by table.
* :class:`OpenTargetsRelease.fetch` downloads the files of the selected tables into ``<dest>/<table>/`` (resumable
  ``.part`` files with ``Range``; a file takes its name only after its sha1 matches the list and its Parquet
  framing is intact; files already present with the listed sha1 are kept) and then writes the manifest.
* :func:`write_manifest` writes ``<dest>/.download-manifest.json`` for the tables whose every file is present and
  matches the list, without downloading anything (a table missing a file, or holding a file the release does
  not list, is left out and reported).

The manifest has the format of the upstream downloader (``third_party/TheVirtualBiotech/tools/
download_open_targets.py``): ``release``, ``base`` (the ``output/`` URL), ``expected_files``, ``complete`` and
``files`` (``{"<table>/<file>.parquet": {"bytes", "sha256", "url"}}``), so the upstream doctor and downloader
accept it and the data layer's readiness check (R2) finds bytes and sha256 for every file. Each entry also keeps
the publisher's ``sha1``. A manifest written here covers the tables it lists in ``tables``: ``complete`` means
every file of those tables was verified and ``expected_files`` counts them (``archive_files`` is the release's
count). The upstream doctor still reports the tables that are missing from the directory.

Networking uses ``httpx`` defaults, so ``HTTPS_PROXY`` / ``SSL_CERT_FILE`` from the environment apply.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

RELEASE = "25.09"
SITE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
MANIFEST = ".download-manifest.json"
INTEGRITY = "release_data_integrity"
RELEASE_DIR = "_release"                 # where the integrity list is kept under the destination
ATTEMPTS = 4
CHUNK = 1 << 20

__all__ = [
    "RELEASE", "SITE", "MANIFEST", "IntegrityError", "FileResult", "FetchReport", "ManifestReport",
    "OpenTargetsRelease", "release_base", "output_base", "parse_integrity", "load_integrity", "inventory",
    "write_manifest", "file_hashes", "add_ot_parser",
]


class IntegrityError(ValueError):
    """A file or list that does not match the publisher's checksum."""


def release_base(release: str = RELEASE, site: str = SITE) -> str:
    return f"{site.rstrip('/')}/{release}/"


def output_base(release: str = RELEASE, site: str = SITE) -> str:
    """The ``output/`` URL; the manifest's ``base`` (the upstream downloader's ``BASE``)."""
    return release_base(release, site) + "output/"


def _relative(path: str) -> str:
    """A canonical relative path below ``output/`` (no ``..``, no absolute or empty parts)."""
    p = PurePosixPath(path)
    if not path or p.is_absolute() or ".." in p.parts or "\\" in path or "\x00" in path or p.as_posix() != path:
        raise IntegrityError(f"not a canonical relative path: {path!r}")
    return path


def parse_integrity(text: str) -> dict[str, str]:
    """``{path below output/: sha1}`` of the output files in a ``release_data_integrity`` text."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        sha1, _, path = line.strip().partition("  ")
        if not path.startswith("./output/") or len(sha1) != 40:
            continue
        out[_relative(path[len("./output/"):])] = sha1.lower()
    return out


def inventory(integrity: Mapping[str, str]) -> dict[str, list[str]]:
    """``{table: [its Parquet files below output/]}`` (sorted)."""
    tables: dict[str, list[str]] = {}
    for rel in integrity:
        if rel.endswith(".parquet") and "/" in rel:
            tables.setdefault(rel.split("/", 1)[0], []).append(rel)
    return {t: sorted(files) for t, files in sorted(tables.items())}


def file_hashes(path: Path) -> tuple[int, str, str]:
    """``(bytes, sha1, sha256)`` of a file, read once."""
    h1, h256 = hashlib.sha1(), hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * CHUNK), b""):
            h1.update(chunk)
            h256.update(chunk)
            size += len(chunk)
    return size, h1.hexdigest(), h256.hexdigest()


def _parquet_framed(path: Path) -> bool:
    """``PAR1`` at both ends and a footer length that fits (the upstream downloader's check)."""
    size = path.stat().st_size
    if size < 12:
        return False
    with path.open("rb") as fh:
        if fh.read(4) != b"PAR1":
            return False
        fh.seek(-8, os.SEEK_END)
        footer = int.from_bytes(fh.read(4), "little")
        return 0 < footer <= size - 12 and fh.read(4) == b"PAR1"


def _client(timeout: float) -> Any:
    import httpx

    return httpx.Client(follow_redirects=True, timeout=timeout,
                        headers={"User-Agent": "vbt-data-ot/1", "Accept-Encoding": "identity"})


def load_integrity(release: str = RELEASE, *, dest: Path | None = None, site: str = SITE, client: Any = None,
                   timeout: float = 120.0) -> tuple[dict[str, str], dict[str, Any]]:
    """``(integrity, about)``: the release's ``release_data_integrity`` (``<dest>/_release/`` copy when present and
    valid, else downloaded and kept there) checked against ``release_data_integrity.sha1``. ``about`` records where
    it came from and its sha1."""
    local = Path(dest) / RELEASE_DIR if dest is not None else None
    url = release_base(release, site) + INTEGRITY
    if local is not None and (local / INTEGRITY).is_file() and (local / f"{INTEGRITY}.sha1").is_file():
        raw = (local / INTEGRITY).read_bytes()
        want = (local / f"{INTEGRITY}.sha1").read_text().split()[0].lower()
        got = hashlib.sha1(raw).hexdigest()
        if got == want:
            return parse_integrity(raw.decode("utf-8")), {"url": url, "sha1": got, "from": str(local / INTEGRITY)}
    own = client is None
    client = client or _client(timeout)
    try:
        r = client.get(url + ".sha1")
        r.raise_for_status()
        want = r.text.split()[0].lower()
        r = client.get(url)
        r.raise_for_status()
        raw = r.content
    finally:
        if own:
            client.close()
    got = hashlib.sha1(raw).hexdigest()
    if got != want:
        raise IntegrityError(f"{url}: sha1 {got} differs from {INTEGRITY}.sha1 ({want})")
    if local is not None:
        local.mkdir(parents=True, exist_ok=True)
        _atomic_write(local / INTEGRITY, raw)
        _atomic_write(local / f"{INTEGRITY}.sha1", f"{want}  {INTEGRITY}\n".encode())
    return parse_integrity(raw.decode("utf-8")), {"url": url, "sha1": got, "from": url}


def _atomic_write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {path}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class FileResult:
    rel: str
    status: str                       # downloaded | present | failed
    bytes: int = 0
    sha1: str = ""
    sha256: str = ""
    seconds: float = 0.0
    error: str = ""


@dataclass
class ManifestReport:
    path: Path
    tables: dict[str, int] = field(default_factory=dict)          # verified tables -> files
    skipped: dict[str, str] = field(default_factory=dict)         # table -> why it was left out
    files: int = 0
    bytes: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [f"manifest {self.path}: {len(self.tables)} table(s), {self.files} file(s), "
                 f"{self.bytes / 1e9:.2f} GB verified against {INTEGRITY} in {self.seconds:.1f} s"]
        lines += [f"  left out {t}: {why}" for t, why in sorted(self.skipped.items())]
        return "\n".join(lines)


@dataclass
class FetchReport:
    results: list[FileResult] = field(default_factory=list)
    manifest: ManifestReport | None = None
    seconds: float = 0.0

    @property
    def failed(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "failed"]

    def summary(self) -> str:
        got = [r for r in self.results if r.status == "downloaded"]
        kept = [r for r in self.results if r.status == "present"]
        lines = [f"{len(got)} file(s) downloaded ({sum(r.bytes for r in got) / 1e6:.1f} MB), {len(kept)} already "
                 f"present, {len(self.failed)} failed, {self.seconds:.1f} s"]
        lines += [f"  FAILED {r.rel}: {r.error}" for r in self.failed]
        if self.manifest is not None:
            lines.append(self.manifest.summary())
        return "\n".join(lines)


def write_manifest(dest: Path, integrity: Mapping[str, str], *, release: str = RELEASE, site: str = SITE,
                   tables: Iterable[str] | None = None, known: Mapping[str, Mapping[str, Any]] | None = None,
                   about: Mapping[str, Any] | None = None) -> ManifestReport:
    """Write ``<dest>/.download-manifest.json`` for the tables (default: every table directory under ``dest`` that
    the release lists) whose files are all present and match the integrity list. ``known`` (``{rel: {bytes,
    sha1, sha256}}``, e.g. just downloaded) spares hashing a file again when its size is unchanged."""
    t0 = time.monotonic()
    dest = Path(dest)
    inv = inventory(integrity)
    names = sorted(tables) if tables is not None else sorted(p.name for p in dest.iterdir() if p.is_dir()
                                                              and p.name in inv)
    report = ManifestReport(path=dest / MANIFEST)
    previous = _load_manifest(dest / MANIFEST)
    old_files = previous.get("files") if previous.get("release") == release and \
        previous.get("base") == output_base(release, site) else None
    old_files = old_files if isinstance(old_files, dict) else {}
    base = output_base(release, site)
    entries: dict[str, dict[str, Any]] = {}
    for table in names:
        if table not in inv:
            report.skipped[table] = "not a table of this release"
            continue
        listed = set(inv[table])
        present = {p.relative_to(dest).as_posix() for p in (dest / table).rglob("*.parquet")} \
            if (dest / table).is_dir() else set()
        if not present:
            report.skipped[table] = "no Parquet files"
            continue
        extra = sorted(present - listed)
        missing = sorted(listed - present)
        if extra or missing:
            report.skipped[table] = (f"{len(missing)} of {len(listed)} file(s) missing" if missing else "") + \
                ("; " if extra and missing else "") + (f"{len(extra)} file(s) not in the release: {extra[0]}"
                                                       if extra else "")
            continue
        mine: dict[str, dict[str, Any]] = {}
        bad = ""
        for rel in sorted(listed):
            path = dest / rel
            size = path.stat().st_size
            cached = (known or {}).get(rel) or {}
            if cached.get("bytes") == size and cached.get("sha1") and cached.get("sha256"):
                sha1, sha256 = str(cached["sha1"]), str(cached["sha256"])
            else:
                size, sha1, sha256 = file_hashes(path)
            if sha1 != integrity[rel]:
                bad = f"{rel}: sha1 {sha1} differs from {INTEGRITY} ({integrity[rel]})"
                break
            mine[rel] = {"bytes": size, "sha256": sha256, "url": base + rel, "sha1": sha1}
        if bad:
            report.skipped[table] = bad
            continue
        entries.update(mine)
        report.tables[table] = len(mine)
        report.bytes += sum(e["bytes"] for e in mine.values())
    # entries of tables not looked at this time stay when their files are unchanged
    for rel, entry in old_files.items():
        table = rel.split("/", 1)[0]
        if table in report.tables or table in report.skipped or rel in entries or rel not in integrity:
            continue
        path = dest / rel
        if isinstance(entry, dict) and path.is_file() and path.stat().st_size == entry.get("bytes"):
            entries[rel] = dict(entry)
            report.tables.setdefault(table, 0)
            report.tables[table] += 1
    covered = sorted({rel.split("/", 1)[0] for rel in entries})
    data = {"release": release, "base": base, "expected_files": len(entries), "complete": bool(entries),
            "files": dict(sorted(entries.items())), "tables": covered,
            "archive_files": sum(len(v) for v in inv.values()),
            "verified": f"sha1 of every file against {INTEGRITY}",
            "integrity": dict(about or {}), "written_by": "vbt data ot",
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic_write(dest / MANIFEST, (json.dumps(data, indent=2) + "\n").encode())
    report.files = len(entries)
    report.seconds = time.monotonic() - t0
    return report


class OpenTargetsRelease:
    """Downloads of one release's tables (``site`` and ``client`` are replaced by tests)."""

    def __init__(self, release: str = RELEASE, *, site: str = SITE, client: Any = None, timeout: float = 120.0,
                 retry_wait: float = 1.0) -> None:
        self.release = release
        self.site = site
        self.timeout = timeout
        self.retry_wait = retry_wait
        self._client = client
        self._own = client is None

    def __enter__(self) -> "OpenTargetsRelease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _client(self.timeout)
        return self._client

    def close(self) -> None:
        if self._own and self._client is not None:
            self._client.close()
            self._client = None

    def integrity(self, dest: Path | None = None) -> tuple[dict[str, str], dict[str, Any]]:
        return load_integrity(self.release, dest=dest, site=self.site, client=self.client, timeout=self.timeout)

    def sizes(self, rels: Sequence[str], workers: int = 8) -> dict[str, int]:
        """Content lengths of files (HEAD requests)."""
        base = output_base(self.release, self.site)

        def head(rel: str) -> tuple[str, int]:
            r = self.client.head(base + rel)
            r.raise_for_status()
            return rel, int(r.headers.get("content-length", 0))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return dict(pool.map(head, rels))

    def fetch(self, tables: Sequence[str], dest: Path, *, workers: int = 4, max_bytes: int | None = None,
              dry_run: bool = False, on_file: Callable[[FileResult], None] | None = None) -> FetchReport:
        """Download the files of ``tables`` into ``dest`` and write the manifest for them. Raises ValueError for a
        table the release does not have, or when the files to download exceed ``max_bytes``."""
        t0 = time.monotonic()
        dest = Path(dest)
        integrity, about = self.integrity(dest if not dry_run else None)
        inv = inventory(integrity)
        unknown = [t for t in tables if t not in inv]
        if unknown:
            raise ValueError(f"not a table of release {self.release}: {', '.join(unknown)} "
                             f"(tables: {', '.join(inv)})")
        rels = [rel for t in tables for rel in inv[t]]
        report = FetchReport()
        present = {rel: got for rel in rels if (got := self._present(rel, dest, integrity[rel])) is not None}
        todo = [rel for rel in rels if rel not in present]
        if max_bytes is not None or dry_run:
            sizes = self.sizes(todo)
            need = sum(sizes.values())
            if dry_run:
                report.results = [FileResult(rel, "planned", bytes=sizes.get(rel, 0)) for rel in todo]
                report.seconds = time.monotonic() - t0
                return report
            if max_bytes is not None and need > max_bytes:
                raise ValueError(f"{len(todo)} file(s) to download hold {need / 1e9:.2f} GB, over the "
                                 f"{max_bytes / 1e9:.2f} GB allowed (--max-gb)")
        report.results = list(present.values())
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._download, rel, dest, integrity[rel]): rel for rel in todo}
            for fut in as_completed(futures):
                res = fut.result()
                report.results.append(res)
                if on_file is not None:
                    on_file(res)
        known = {r.rel: {"bytes": r.bytes, "sha1": r.sha1, "sha256": r.sha256} for r in report.results
                 if r.status in ("downloaded", "present")}
        good = [t for t in tables if not any(r.status == "failed" and r.rel.split("/", 1)[0] == t
                                             for r in report.results)]
        report.manifest = write_manifest(dest, integrity, release=self.release, site=self.site, tables=good,
                                         known=known, about=about)
        report.seconds = time.monotonic() - t0
        return report

    @staticmethod
    def _present(rel: str, dest: Path, sha1: str) -> FileResult | None:
        """The file already in place with the listed sha1 and intact framing (hashed once), else None."""
        path = dest / rel
        if not path.is_file():
            return None
        size, got1, got256 = file_hashes(path)
        if got1 != sha1 or not _parquet_framed(path):
            return None
        return FileResult(rel, "present", bytes=size, sha1=got1, sha256=got256)

    def _download(self, rel: str, dest: Path, sha1: str) -> FileResult:
        url = output_base(self.release, self.site) + rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        t0 = time.monotonic()
        error = ""
        for attempt in range(ATTEMPTS):
            try:
                self._get(url, part)
                size, got1, got256 = file_hashes(part)
                if got1 != sha1:
                    part.unlink(missing_ok=True)
                    raise IntegrityError(f"sha1 {got1} differs from {INTEGRITY} ({sha1})")
                if not _parquet_framed(part):
                    part.unlink(missing_ok=True)
                    raise IntegrityError("not a complete Parquet file")
                part.replace(target)
                return FileResult(rel, "downloaded", bytes=size, sha1=got1, sha256=got256,
                                  seconds=round(time.monotonic() - t0, 2))
            except Exception as exc:  # noqa: BLE001 - retried, then reported per file
                error = f"{type(exc).__name__}: {exc}"
                if attempt < ATTEMPTS - 1:
                    time.sleep(self.retry_wait * min(2 ** attempt, 8))
        return FileResult(rel, "failed", error=error, seconds=round(time.monotonic() - t0, 2))

    def _get(self, url: str, part: Path) -> None:
        """Download ``url`` into ``part``, resuming from its current length with a ``Range`` request (a server that
        answers 200 restarts the file)."""
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with self.client.stream("GET", url, headers=headers) as r:
            if r.status_code == 416 and offset:                 # complete already, or changed: start again
                part.unlink(missing_ok=True)
                raise IOError("range not satisfiable; restarting the file")
            r.raise_for_status()
            mode = "ab" if offset and r.status_code == 206 else "wb"
            if mode == "ab" and not r.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                raise IOError(f"unexpected Content-Range {r.headers.get('content-range')!r}")
            with part.open(mode) as fh:
                for chunk in r.iter_bytes(CHUNK):
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())


# ---------------------------------------------------------------------------- CLI: vbt data ot


def add_ot_parser(sources: Any) -> None:
    """``vbt data ot list|fetch|manifest`` under the ``vbt data`` subcommands."""
    o = sources.add_parser("ot", help=f"Open Targets Platform release tables ({SITE})")
    os_ = o.add_subparsers(dest="ot_action", required=True)

    def common(p: Any, dest: bool = True) -> None:
        p.add_argument("--release", default=RELEASE)
        p.add_argument("--site", default=SITE, help=_suppress())
        if dest:
            p.add_argument("--dest", default=None,
                           help="the release's output directory (default: $OPEN_TARGETS_DATA_PATH)")

    lp = os_.add_parser("list", help="tables of the release and their file counts (from release_data_integrity)")
    common(lp, dest=False)
    fp = os_.add_parser("fetch", help="download tables, verify each file's sha1, write .download-manifest.json")
    fp.add_argument("tables", nargs="+", help="table names (e.g. target go reactome)")
    fp.add_argument("--workers", type=int, default=4)
    fp.add_argument("--max-gb", type=float, help="refuse when the files to download exceed this size")
    fp.add_argument("--dry-run", action="store_true", help="list the files and sizes to download")
    common(fp)
    mp = os_.add_parser("manifest", help="verify the tables already under --dest and write the manifest "
                                         "(nothing is downloaded)")
    mp.add_argument("tables", nargs="*", help="tables to include (default: every table directory)")
    common(mp)
    o.set_defaults(handler=_cmd_ot)


def _suppress() -> Any:
    import argparse

    return argparse.SUPPRESS


def _dest(args: Any) -> Path | None:
    raw = args.dest or os.environ.get("OPEN_TARGETS_DATA_PATH", "")
    return Path(raw).expanduser() if raw.strip() else None


def _cmd_ot(args: Any, config: Mapping[str, Any]) -> int:
    del config
    if args.ot_action == "list":
        integrity, about = load_integrity(args.release, site=args.site)
        inv = inventory(integrity)
        for table, files in inv.items():
            print(f"{len(files):6d}  {table}")
        print(f"# {len(inv)} tables, {sum(len(f) for f in inv.values())} Parquet files ({about['url']}, "
              f"sha1 {about['sha1']})")
        return 0
    dest = _dest(args)
    if dest is None:
        print("error: give --dest or set OPEN_TARGETS_DATA_PATH")
        return 2
    if args.ot_action == "manifest":
        if not dest.is_dir():
            print(f"error: {dest} is not a directory")
            return 2
        integrity, about = load_integrity(args.release, dest=dest, site=args.site)
        rep = write_manifest(dest, integrity, release=args.release, site=args.site,
                             tables=args.tables or None, about=about)
        print(rep.summary())
        return 0 if rep.tables else 1
    dest.mkdir(parents=True, exist_ok=True)
    with OpenTargetsRelease(args.release, site=args.site) as rel:
        try:
            rep = rel.fetch(args.tables, dest, workers=args.workers, dry_run=args.dry_run,
                            max_bytes=int(args.max_gb * 1e9) if args.max_gb is not None else None,
                            on_file=lambda r: print(f"  {r.status:10s} {r.bytes:>13,d}  {r.rel}", flush=True))
        except (ValueError, IntegrityError) as exc:
            print(f"error: {exc}")
            return 2
    if args.dry_run:
        for r in rep.results:
            print(f"  {r.bytes:>13,d}  {r.rel}")
        print(f"# {len(rep.results)} file(s), {sum(r.bytes for r in rep.results) / 1e9:.2f} GB to download")
        return 0
    print(rep.summary())
    return 1 if rep.failed else 0

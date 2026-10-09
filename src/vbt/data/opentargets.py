"""``vbt data ot``: the Open Targets release tables, through the generic acquisition engine.

Everything specific to Open Targets is the ``acquisition`` section of ``configs/data/sources/open_targets.yaml``
(the ``http`` transport with the release's ``release_data_integrity`` as inventory and sha1 source, kept in
``_release/``; ``parquet_framing``; the upstream ``.download-manifest.json``). This module keeps the
``vbt data ot list|fetch|manifest`` commands and the API S2 introduced (:class:`OpenTargetsRelease`,
:func:`load_integrity`, :func:`write_manifest`, :func:`inventory`), reading that section and re-rooting its URLs at
``site`` (tests serve a release locally). ``vbt data acquire open_targets[.<table>]`` does the same with the
acquisition root's directory layout.

* :func:`load_integrity` reads the list (from ``<dest>/_release/`` when present and valid, else from the release)
  and checks it against its ``.sha1``; :func:`inventory` groups its Parquet files by table.
* :meth:`OpenTargetsRelease.fetch` downloads the files of the selected tables into ``<dest>/<table>/``
  (resumable ``.part`` files; a file takes its name only after its sha1 matches the list and its Parquet framing is
  intact; files already present are kept) and then writes the manifest.
* :func:`write_manifest` writes ``<dest>/.download-manifest.json`` for the tables whose every file is present and
  matches the list, without downloading anything.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..datalayer.plugins.acquisition import HttpSession, file_digests
from ..datalayer.plugins.acquisition.http import load_checksum_list, parse_checksum_list
from ..datalayer.plugins.base import AcquisitionError, RemoteFile
from . import acquire as A
from .manifest import MANIFEST, ManifestReport
from .manifest import write_manifest as _write_manifest

RELEASE = "25.09"
SITE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
INTEGRITY = "release_data_integrity"
RELEASE_DIR = "_release"                 # where the integrity list is kept under the destination
SOURCE = "open_targets"

__all__ = [
    "RELEASE", "SITE", "MANIFEST", "IntegrityError", "FileResult", "FetchReport", "ManifestReport",
    "OpenTargetsRelease", "release_base", "output_base", "parse_integrity", "load_integrity", "inventory",
    "write_manifest", "file_hashes", "add_ot_parser", "descriptor",
]

IntegrityError = A.IntegrityError


def release_base(release: str = RELEASE, site: str = SITE) -> str:
    return f"{site.rstrip('/')}/{release}/"


def output_base(release: str = RELEASE, site: str = SITE) -> str:
    """The ``output/`` URL; the manifest's ``base`` (the upstream downloader's ``BASE``)."""
    return release_base(release, site) + "output/"


def file_hashes(path: Path) -> tuple[int, str, str]:
    """``(bytes, sha1, sha256)`` of a file, read once."""
    size, d = file_digests(path, ("sha1", "sha256"))
    return size, d["sha1"], d["sha256"]


def descriptor(release: str = RELEASE, site: str = SITE) -> Any:
    """The open_targets descriptor with its acquisition pinned to ``release`` and re-rooted at ``site``."""
    from ..datalayer.descriptor.load import load_descriptor
    from ..datalayer.settings import DataSettings

    desc = load_descriptor(DataSettings.from_dict({}).descriptors_dir / f"{SOURCE}.yaml")
    acq = desc.acquisition
    rooted = SITE.rstrip("/")

    def reroot(v: Any) -> Any:
        if isinstance(v, str):
            return site.rstrip("/") + v[len(rooted):] if v.startswith(rooted) else v
        if isinstance(v, list):
            return [reroot(x) for x in v]
        if isinstance(v, dict):
            return {k: reroot(x) for k, x in v.items()}
        return v

    transport = acq.transport.model_copy(update={"options": reroot(dict(acq.transport.options))})
    return desc.model_copy(update={"acquisition": acq.model_copy(update={"release": release,
                                                                         "transport": transport})})


def _checksum_spec(release: str, site: str) -> dict[str, Any]:
    desc = descriptor(release, site)
    return dict(A.transport_options(desc, release)["checksums"])


def parse_integrity(text: str) -> dict[str, str]:
    """``{path below output/: sha1}`` of the output files in a ``release_data_integrity`` text."""
    return parse_checksum_list(text, prefix="./output/", algo="sha1")


def inventory(integrity: Mapping[str, str]) -> dict[str, list[str]]:
    """``{table: [its Parquet files below output/]}`` (sorted)."""
    tables: dict[str, list[str]] = {}
    for rel in integrity:
        if rel.endswith(".parquet") and "/" in rel:
            tables.setdefault(rel.split("/", 1)[0], []).append(rel)
    return {t: sorted(files) for t, files in sorted(tables.items())}


def load_integrity(release: str = RELEASE, *, dest: Path | None = None, site: str = SITE, client: Any = None,
                   timeout: float = 120.0) -> tuple[dict[str, str], dict[str, Any]]:
    """``(integrity, about)``: the release's ``release_data_integrity`` (the ``<dest>/_release/`` copy when present
    and valid, else downloaded and kept there) checked against ``release_data_integrity.sha1``."""
    with HttpSession(client, timeout=timeout) as session:
        try:
            raw, about = load_checksum_list(_checksum_spec(release, site), session,
                                            Path(dest) / RELEASE_DIR if dest is not None else None)
        except AcquisitionError as exc:
            raise IntegrityError(str(exc)) from exc
    return parse_integrity(raw.decode("utf-8")), about


def _listing(integrity: Mapping[str, str], release: str, site: str) -> list[RemoteFile]:
    base = output_base(release, site)
    return [RemoteFile(rel, base + rel, None, {"sha1": sha1}) for rel, sha1 in sorted(integrity.items())]


def write_manifest(dest: Path, integrity: Mapping[str, str], *, release: str = RELEASE, site: str = SITE,
                   tables: Iterable[str] | None = None, known: Mapping[str, Mapping[str, Any]] | None = None,
                   about: Mapping[str, Any] | None = None) -> ManifestReport:
    """Write ``<dest>/.download-manifest.json`` for ``tables`` (default: every table with files under ``dest``)
    whose files are all present and match the integrity list."""
    desc = descriptor(release, site)
    return _write_manifest(desc.acquisition, Path(dest), _listing(integrity, release, site), release=release,
                           base=output_base(release, site), groups=tables, known=known, about=about,
                           skip_dirs=[RELEASE_DIR], written_by="vbt data ot")


@dataclass
class FileResult:
    rel: str
    status: str                       # downloaded | present | failed | planned
    bytes: int = 0
    sha1: str = ""
    sha256: str = ""
    seconds: float = 0.0
    error: str = ""


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


class OpenTargetsRelease:
    """Downloads of one release's tables into one directory (``site`` and ``client`` are replaced by tests)."""

    def __init__(self, release: str = RELEASE, *, site: str = SITE, client: Any = None, timeout: float = 120.0,
                 retry_wait: float = 1.0) -> None:
        self.release = release
        self.site = site
        self.timeout = timeout
        self.retry_wait = retry_wait
        self.session = HttpSession(client, timeout=timeout, retry_wait=retry_wait)

    def __enter__(self) -> "OpenTargetsRelease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    def integrity(self, dest: Path | None = None) -> tuple[dict[str, str], dict[str, Any]]:
        raw, about = load_checksum_list(_checksum_spec(self.release, self.site), self.session,
                                        Path(dest) / RELEASE_DIR if dest is not None else None)
        return parse_integrity(raw.decode("utf-8")), about

    def sizes(self, rels: Sequence[str], workers: int = 8) -> dict[str, int]:
        """Content lengths of files (HEAD requests)."""
        base = output_base(self.release, self.site)
        return {rel: int(self.session.head_size(base + rel) or 0) for rel in rels}

    def fetch(self, tables: Sequence[str], dest: Path, *, workers: int = 4, max_bytes: int | None = None,
              dry_run: bool = False, on_file: Callable[[FileResult], None] | None = None) -> FetchReport:
        """Download the files of ``tables`` into ``dest`` and write the manifest for them. Raises ValueError for a
        table the release does not have, or when the files to download exceed ``max_bytes``."""
        t0 = time.monotonic()
        dest = Path(dest)
        desc = descriptor(self.release, self.site)
        settings = A.AcquisitionSettings(root=dest, workers=max(1, workers), retries=self.session.retries,
                                         timeout_s=self.timeout, reserve_bytes=0)
        try:
            integrity, _about = self.integrity(dest if not dry_run else None)
        except AcquisitionError as exc:
            raise IntegrityError(str(exc)) from exc
        inv = inventory(integrity)
        unknown = [t for t in tables if t not in inv or t not in desc.acquisition.tables]
        if unknown:
            raise ValueError(f"not a table of release {self.release}: {', '.join(unknown)} "
                             f"(tables: {', '.join(inv)})")
        sp = A.plan_source(desc, list(tables), settings, home=dest, session=self.session,
                           sizes=dry_run or max_bytes is not None, index_cache=not dry_run)
        if sp.listing_error:
            raise ValueError(sp.listing_error)
        report = FetchReport()
        if dry_run:
            report.results = [FileResult(pf.remote.path, "planned", bytes=int(pf.size or 0))
                              for pf in sp.files if pf.state != "verified"]
            report.seconds = time.monotonic() - t0
            return report
        plan = A.AcquisitionPlan(sources=[sp], root=dest)

        def event(kind: str, info: dict[str, Any]) -> None:
            if kind == "file" and on_file is not None:
                on_file(FileResult(info["path"], info["status"], bytes=int(info.get("bytes") or 0),
                                   error=str(info.get("error") or "")))

        res = A.execute(plan, settings, session=self.session, workers=workers, max_bytes=max_bytes,
                        retry_wait=self.retry_wait, on_event=event, progress_s=3600)
        src = res.sources[0]
        if src.error:
            raise ValueError(src.error)
        report.results = [FileResult(r.rel, r.status, bytes=r.bytes, sha1=r.sha1, sha256=r.sha256, seconds=r.seconds,
                                     error=r.error) for r in src.files]
        report.manifest = src.manifest
        report.seconds = time.monotonic() - t0
        return report


# ---------------------------------------------------------------------------- CLI: vbt data ot


def add_ot_parser(sources: Any) -> None:
    """``vbt data ot list|fetch|manifest`` under the ``vbt data`` subcommands."""
    o = sources.add_parser("ot", help=f"Open Targets Platform release tables ({SITE}); "
                                      "`vbt data acquire open_targets.<table>` is the general form")
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

"""Generic, declarative acquisition: ``vbt data acquire`` from the descriptors' ``acquisition`` sections.

Nothing here knows a source. A descriptor declares how its files are acquired (an acquisition plugin and its
options, the files of each table, checksums, sizes, prepare steps that run unmodified upstream scripts, where the
files land, the environment variables that point the data layer at them, licence and login notes); this module:

1. :func:`plan_acquisition`: lists each source once through its transport (or uses the declared sizes offline),
   classifies every file (``verified`` by the manifest, ``present`` unverified, ``partial`` ``.part``, ``missing``)
   and totals what is left to transfer, the disk it needs and the time it takes at the measured rate;
2. :func:`execute`: downloads the missing files in parallel, resuming ``.part`` files with range requests; a file
   takes its name only after its size, its publisher checksum and (``parquet_framing``) its Parquet framing match;
   then it writes the manifest (:mod:`vbt.data.manifest`, the upstream downloader's format, which the readiness R2
   check and the upstream doctor read), runs the prepare steps whose inputs are all verified, records the
   acquisition in ``<home>/.vbt-acquisition.json`` (the pinned release) and appends a provenance record.

The acquisition root (``data.acquisition.root``, default ``${VBT_DATA_DIR:-data}/sources``) holds one home per
source: ``<root>/<acquisition.dir>`` (default ``{source}/{release}``). :class:`AcquisitionSettings` holds the
``data.acquisition`` policy (``auto: off | ask | under_budget``, ``budget_bytes``, ``workers``, ``rate_mbps``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..datalayer.plugins.acquisition import (
    HttpSession,
    file_digests,
    matches,
    safe_join,
    strongest,
    substitute,
)
from ..datalayer.plugins.base import AcquisitionError, RemoteFile
from .manifest import MANIFEST, ManifestReport, atomic_write, load_manifest, parquet_framed, write_manifest

__all__ = [
    "AcquisitionSettings", "SourcePlan", "PlannedFile", "AcquisitionPlan", "FileResult", "SourceResult",
    "AcquisitionReport", "plan_acquisition", "execute", "source_release", "source_home", "source_env", "source_root",
    "acquisition_registry", "LOCK", "write_env_file", "record_provenance", "fmt_bytes", "fmt_seconds",
    "groups_for", "IntegrityError", "transport_options",
]

LOCK = ".vbt-acquisition.json"
ATTEMPTS = 4


class IntegrityError(ValueError):
    """A downloaded file that does not match its size, checksum or framing."""


# ---------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class AcquisitionSettings:
    """``data.acquisition``: where acquisitions land and what the system may acquire by itself.

    ``auto``: ``off`` (a refused call says how to acquire; nothing more), ``ask`` (it also says the operator must
    approve), ``under_budget`` (tables whose acquisition fits ``budget_bytes`` are acquired between turns and
    recorded in provenance). ``workers``: parallel transfers (``auto``: 4 per CPU, at most 32); ``rate_mbps``: the
    transfer rate the plan assumes until one was measured on this host."""

    root: Path
    auto: str = "off"
    budget_bytes: int = 0
    workers: int = 8
    rate_mbps: float = 50.0
    retries: int = 4
    timeout_s: float = 120.0
    reserve_bytes: int = 1 << 30
    provenance_dir: Path | None = None
    cache_dir: Path | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "AcquisitionSettings":
        """From the typed ``data.acquisition`` section (:class:`vbt.datalayer.settings.AcquisitionSettings`, which
        merges the defaults, expands variables and refuses an unknown ``auto``). The acquisition log
        (``acquisitions.jsonl``) lives in the root, next to the homes it describes."""
        from .. import config as _config
        from ..datalayer.settings import DataSettings

        config = config or {}
        variables = {str(k): "" if v is None else str(v) for k, v in (config.get("vars") or {}).items()}
        root_project = Path(variables.get("project_root") or _config.PROJECT_ROOT)
        settings = DataSettings.from_config(config)
        acq = settings.acquisition

        def path(value: Any) -> Path:
            p = Path(str(value)).expanduser()
            return p if p.is_absolute() else root_project / p

        workers = acq.workers
        if workers in (None, "auto"):
            workers = max(1, min(32, 4 * (os.cpu_count() or 1)))
        root = path(acq.root or _config._expand("${VBT_DATA_DIR:-data}/sources", variables))
        return cls(root=root, auto=acq.auto, budget_bytes=_bytes(acq.budget_bytes), workers=int(workers),
                   rate_mbps=float(acq.rate_mbps), retries=int(acq.retries), timeout_s=float(acq.timeout_s),
                   reserve_bytes=_bytes(acq.reserve_bytes), provenance_dir=root, cache_dir=settings.cache_dir)


def _bytes(value: Any) -> int:
    """``2000000000``, ``"2 GB"``, ``"500MiB"`` -> bytes."""
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip().upper().replace(" ", "")
    units = {"TIB": 1 << 40, "GIB": 1 << 30, "MIB": 1 << 20, "KIB": 1 << 10, "TB": 10**12, "GB": 10**9,
             "MB": 10**6, "KB": 10**3, "B": 1}
    for unit, mult in units.items():
        if text.endswith(unit):
            return int(float(text[: -len(unit)]) * mult)
    return int(float(text))


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "size unknown"
    n = float(n)
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{int(n)} B"


def fmt_seconds(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# ---------------------------------------------------------------------------- per-source helpers


def source_release(desc: Any) -> str:
    acq = desc.acquisition
    return str((acq.release if acq is not None else None) or desc.release.expect or "current")


def source_root(catalog: Any, source: str, default: Path) -> Path:
    """The acquisition root of ``source``: a source the active project added (``catalog.project_sources``) is acquired
    into ``<project>/data``, where its descriptor's root reads (``${VBT_PROJECT_DIR}/data/<source>`` with ``dir:
    <source>``; docs/PROJECTS.md); every other source into ``default`` (``data.acquisition.root``)."""
    project = getattr(catalog, "project_dir", None)
    if project is not None and source in (getattr(catalog, "project_sources", None) or ()):
        return Path(project) / "data"
    return Path(default)


def source_home(desc: Any, root: Path, release: str | None = None) -> Path:
    acq = desc.acquisition
    rel = release or source_release(desc)
    sub = substitute(acq.dir if acq is not None else "{source}/{release}", {"source": desc.source, "release": rel})
    return Path(root) / sub


def _downloads(desc: Any, home: Path) -> Path:
    return home / (desc.acquisition.downloads if desc.acquisition is not None else ".")


def source_env(desc: Any, home: Path, release: str | None = None) -> dict[str, str]:
    acq = desc.acquisition
    if acq is None:
        return {}
    rel = release or source_release(desc)
    v = {"home": str(home), "downloads": str(_downloads(desc, home)), "release": rel}
    return {k: os.path.normpath(substitute(val, v)) for k, val in acq.env.items()}


def transport_options(desc: Any, release: str) -> dict[str, Any]:
    acq = desc.acquisition
    return substitute(dict(acq.transport.options), {"release": release}) if acq and acq.transport else {}


def groups_for(desc: Any, names: Iterable[str]) -> tuple[list[str], list[str]]:
    """``(download groups, prepare steps)`` that acquiring ``names`` (tables or extra groups) needs."""
    acq = desc.acquisition
    groups: list[str] = []
    steps: list[str] = []
    for n in names:
        entry = acq.tables.get(n) or acq.extra.get(n)
        if entry is None:
            continue
        if entry.files and n not in groups:
            groups.append(n)
        if entry.prepared_by and entry.prepared_by not in steps:
            steps.append(entry.prepared_by)
            for need in acq.prepare[entry.prepared_by].needs:
                if need not in groups:
                    groups.append(need)
    return groups, steps


def acquisition_registry(settings: Any = None) -> Any:
    """The registry of acquisition plugins (builtins, entry points, ``data.plugins.paths``)."""
    from ..datalayer.plugins.registry import discover_harness

    return discover_harness(settings)


# ---------------------------------------------------------------------------- plan


@dataclass
class PlannedFile:
    remote: RemoteFile
    groups: list[str]
    state: str                         # verified | present | partial | missing
    local_bytes: int = 0
    est_size: float | None = None      # the group's declared bytes per file, when the listing has no size

    @property
    def size(self) -> float | None:
        return self.remote.size if self.remote.size is not None else self.est_size

    @property
    def remaining(self) -> float | None:
        if self.state in ("verified", "present"):
            return 0
        if self.size is None:
            return None
        return max(0.0, self.size - (self.local_bytes if self.state == "partial" else 0))


@dataclass
class SourcePlan:
    source: str
    title: str
    mode: str
    release: str
    home: Path
    downloads: Path
    requested: list[str]
    groups: list[str]
    prepare: list[str]
    transport: str = ""
    describe: str = ""
    files: list[PlannedFile] = field(default_factory=list)
    listed: list[RemoteFile] = field(default_factory=list)
    declared_bytes: int | None = None
    declared_files: int | None = None
    listing_error: str | None = None
    offline: bool = False
    env: dict[str, str] = field(default_factory=dict)
    licence: str | None = None
    login: str | None = None
    notes: list[str] = field(default_factory=list)
    prepared: dict[str, str] = field(default_factory=dict)   # step -> done | pending | blocked: <why>
    desc: Any = field(default=None, repr=False)

    @property
    def to_download(self) -> list[PlannedFile]:
        return [f for f in self.files if f.state != "verified"]

    @property
    def bytes_total(self) -> int | None:
        if self.files:
            sizes = [f.size for f in self.files]
            return None if any(s is None for s in sizes) else int(sum(s for s in sizes if s is not None))
        return self.declared_bytes

    @property
    def bytes_remaining(self) -> int | None:
        if self.files:
            rem = [f.remaining for f in self.files]
            return None if any(r is None for r in rem) else int(sum(r for r in rem if r is not None))
        return self.declared_bytes

    def prepare_states(self) -> list[tuple[str, str]]:
        return [(step, self.prepared.get(step, "pending")) for step in self.prepare]


@dataclass
class AcquisitionPlan:
    sources: list[SourcePlan]
    root: Path
    notes: list[str] = field(default_factory=list)
    rate_mbps: float = 50.0
    rate_measured: bool = False

    @property
    def bytes_remaining(self) -> int | None:
        rem = [s.bytes_remaining for s in self.sources if s.mode == "download"]
        return None if any(r is None for r in rem) else sum(r for r in rem if r is not None)

    def seconds(self) -> float | None:
        b = self.bytes_remaining
        return None if b is None else b / max(self.rate_mbps * 1e6, 1.0)

    def lines(self) -> list[str]:
        out: list[str] = []
        for s in self.sources:
            head = f"{s.source} {s.release} ({s.mode})"
            if s.mode == "remote":
                out.append(f"{head}: read live from {s.describe or 'the remote source'}; nothing to download")
                out.extend(f"  note: {n}" for n in s.notes)
                continue
            if s.mode == "manual":
                out.append(f"{head}: acquired by a person: {s.login or 'see the descriptor'}")
                continue
            out.append(f"{head} -> {s.home}")
            out.append(f"  from: {s.describe}" + (" (declared sizes; not listed)" if s.offline else ""))
            if s.listing_error:
                out.append(f"  listing failed: {s.listing_error}")
            want = ", ".join(s.requested)
            out.append(f"  for: {want}")
            if s.files:
                n_ok = sum(1 for f in s.files if f.state == "verified")
                n_pres = sum(1 for f in s.files if f.state == "present")
                n_part = sum(1 for f in s.files if f.state == "partial")
                out.append(f"  files: {len(s.files)} ({fmt_bytes(s.bytes_total)}); verified {n_ok}, present "
                           f"unverified {n_pres}, partial {n_part}, missing "
                           f"{len(s.files) - n_ok - n_pres - n_part}")
            else:
                out.append(f"  files: {s.declared_files if s.declared_files is not None else '?'} "
                           f"({fmt_bytes(s.declared_bytes)}, declared)")
            rem = s.bytes_remaining
            secs = None if rem is None else rem / max(self.rate_mbps * 1e6, 1.0)
            out.append(f"  to transfer: {fmt_bytes(rem)}" + ("" if secs is None else
                       f", about {fmt_seconds(secs)} at {self.rate_mbps:.0f} MB/s"
                       f" ({'measured' if self.rate_measured else 'assumed'})"))
            for step, state in s.prepare_states():
                out.append(f"  prepare {step}: {state}")
            for k, v in s.env.items():
                out.append(f"  env: {k}={v}")
            if s.licence:
                out.append(f"  licence: {s.licence}")
            if s.login:
                out.append(f"  login: {s.login}")
            out.extend(f"  note: {n}" for n in s.notes)
        b = self.bytes_remaining
        secs = self.seconds()
        out.append(f"# total to transfer: {fmt_bytes(b)}" + ("" if secs is None else
                   f", about {fmt_seconds(secs)} at {self.rate_mbps:.0f} MB/s"))
        out.extend(f"# {n}" for n in self.notes)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"root": str(self.root), "bytes_remaining": self.bytes_remaining, "seconds": self.seconds(),
                "rate_mbps": self.rate_mbps, "rate_measured": self.rate_measured, "notes": list(self.notes),
                "sources": [{
                    "source": s.source, "mode": s.mode, "release": s.release, "home": str(s.home),
                    "requested": s.requested, "groups": s.groups, "prepare": dict(s.prepare_states()),
                    "transport": s.transport, "from": s.describe, "files": len(s.files) or s.declared_files,
                    "bytes_total": s.bytes_total, "bytes_remaining": s.bytes_remaining,
                    "states": {st: sum(1 for f in s.files if f.state == st)
                               for st in ("verified", "present", "partial", "missing")},
                    "offline": s.offline, "listing_error": s.listing_error, "env": s.env, "licence": s.licence,
                    "login": s.login, "notes": s.notes} for s in self.sources]}


def _rate(settings: AcquisitionSettings) -> tuple[float, bool]:
    """The transfer rate measured by earlier acquisitions on this host (MB/s), else the configured one."""
    if settings.cache_dir is not None:
        rec = load_manifest(Path(settings.cache_dir) / "acquisition" / "throughput.json")
        mbps = rec.get("mbps")
        if isinstance(mbps, (int, float)) and mbps > 0:
            return float(mbps), True
    return settings.rate_mbps, False


def _record_rate(settings: AcquisitionSettings, nbytes: int, seconds: float) -> None:
    if settings.cache_dir is None or nbytes < 50_000_000 or seconds <= 0:
        return                                         # small transfers say little about the link
    path = Path(settings.cache_dir) / "acquisition" / "throughput.json"
    old = load_manifest(path)
    mbps = nbytes / seconds / 1e6
    if isinstance(old.get("mbps"), (int, float)):
        mbps = 0.5 * mbps + 0.5 * float(old["mbps"])
    try:
        atomic_write(path, json.dumps({"mbps": round(mbps, 2), "bytes": nbytes, "seconds": round(seconds, 2),
                                       "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode())
    except OSError:
        pass


def _manifest_entries(downloads: Path, spec: Any, release: str) -> tuple[dict[str, Any], float]:
    path = downloads / (spec.manifest or MANIFEST)
    data = load_manifest(path)
    if data.get("release") not in (None, release):
        return {}, 0.0
    files = data.get("files")
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (files if isinstance(files, dict) else {}), mtime


def _classify(downloads: Path, f: RemoteFile, entries: Mapping[str, Any], mtime: float) -> PlannedFile:
    target = downloads / f.path
    part = target.with_name(target.name + ".part")
    if target.is_file():
        st = target.stat()
        entry = entries.get(f.path)
        pub = strongest(f.checksums)
        if isinstance(entry, dict) and entry.get("bytes") == st.st_size and (f.size in (None, st.st_size)) \
                and st.st_mtime <= mtime + 1 and (pub is None or entry.get(pub[0], pub[1]) == pub[1]):
            return PlannedFile(f, [], "verified", st.st_size)
        if f.size is None or st.st_size == f.size:
            return PlannedFile(f, [], "present", st.st_size)
        return PlannedFile(f, [], "missing", 0)
    if part.is_file():
        return PlannedFile(f, [], "partial", part.stat().st_size)
    return PlannedFile(f, [], "missing", 0)


def plan_source(desc: Any, names: Sequence[str], settings: AcquisitionSettings, *, home: Path | None = None,
                root: Path | None = None, registry: Any = None, session: HttpSession | None = None,
                offline: bool = False, sizes: bool = True, index_cache: bool = True,
                write_index: bool = True) -> SourcePlan:
    """The plan of one source: ``names`` (tables or extra groups) acquired into ``home`` (default: its home under
    ``root``). The source is listed once through its transport (``offline``: declared sizes only); files without
    a listed or declared size are sized with HEAD requests when ``sizes``. The transport keeps its index in the
    ``index_cache`` directory; without ``write_index`` (a plan only) an existing one is read and none is created."""
    from .manifest import group_patterns

    acq = desc.acquisition
    if acq is None:
        raise ValueError(f"{desc.source}: the descriptor declares no acquisition section")
    registry = registry or acquisition_registry()
    release = source_release(desc)
    home = Path(home) if home is not None else source_home(desc, Path(root or settings.root), release)
    groups, steps = groups_for(desc, names)
    sp = SourcePlan(source=desc.source, title=desc.title, mode=acq.mode, release=release, home=home,
                    downloads=_downloads(desc, home), requested=list(names), groups=groups, prepare=steps,
                    env=source_env(desc, home, release), licence=acq.licence, login=acq.login,
                    notes=[n for n in [acq.notes] if n], desc=desc)
    plugin = None
    opts: dict[str, Any] = {}
    if acq.transport is not None:
        sp.transport = acq.transport.plugin
        plugin = registry.find("acquisition", acq.transport.plugin)
        opts = transport_options(desc, release)
        if plugin is None:
            sp.listing_error = (f"no acquisition plugin {acq.transport.plugin!r} (registered: "
                                f"{', '.join(registry.names('acquisition'))})")
        else:
            try:
                sp.describe = plugin.describe(opts)
            except AcquisitionError as exc:
                sp.listing_error = str(exc)
    entries_known = [acq.tables.get(g) or acq.extra.get(g) for g in groups]
    if all(e is not None and e.bytes is not None for e in entries_known):
        sp.declared_bytes = sum(int(e.bytes or 0) for e in entries_known if e is not None)
    if all(e is not None and e.count is not None for e in entries_known):
        sp.declared_files = sum(int(e.count or 0) for e in entries_known if e is not None)
    for step in steps:
        sp.prepared[step] = _prepare_state(desc, home, step)
    if acq.mode != "download" or not groups:
        return sp
    if offline or plugin is None or sp.listing_error:
        sp.offline = True
        return sp
    own = session is None
    session = session or HttpSession(timeout=settings.timeout_s, retries=settings.retries)
    try:
        cache = (sp.downloads / acq.index_cache) if acq.index_cache and index_cache else None
        if cache is not None and not write_index and not cache.is_dir():
            cache = None
        try:
            listed = plugin.listing(opts, session, index_cache=cache)
        except AcquisitionError as exc:
            sp.listing_error = str(exc)
            sp.offline = True
            return sp
        sp.listed = listed
        entries, mtime = _manifest_entries(sp.downloads, acq, release)
        pats = {g: group_patterns(acq, g, release) for g in groups}
        for f in listed:
            owners = [g for g in groups if matches(f.path, *pats[g])]
            if not owners:
                continue
            pf = _classify(sp.downloads, f, entries, mtime)
            pf.groups = owners
            sp.files.append(pf)
        empty = [g for g in groups if not any(g in pf.groups for pf in sp.files)]
        if empty:
            sp.listing_error = "; ".join(f"the listing has no file for {g} ({', '.join(pats[g][0])})" for g in empty)
        for g in groups:                               # a declared size per file stands in for an unlisted one
            entry = acq.tables.get(g) or acq.extra.get(g)
            mine = [pf for pf in sp.files if pf.groups[0] == g]
            if entry is not None and entry.bytes and entry.count and entry.count == len(mine):
                for pf in mine:
                    if pf.remote.size is None:
                        pf.est_size = entry.bytes / entry.count
        if sizes:
            unknown = [pf for pf in sp.files if pf.size is None and pf.state not in ("verified", "present")]
            _head_sizes(unknown, session, settings.workers)
        return sp
    finally:
        if own:
            session.close()


def plan_acquisition(catalog: Any, wanted: Mapping[str, Sequence[str]], settings: AcquisitionSettings, *,
                     root: Path | None = None, registry: Any = None, session: HttpSession | None = None,
                     offline: bool = False, sizes: bool = True, write_index: bool = True) -> AcquisitionPlan:
    """The plan for ``wanted`` (``{source: [tables or extra groups]}``), one :func:`plan_source` per source. Without
    ``root`` each source goes under its :func:`source_root` (a project's sources into the project)."""
    explicit = root is not None
    root = Path(root or settings.root)
    registry = registry or acquisition_registry()
    rate, measured = _rate(settings)
    plan = AcquisitionPlan(sources=[], root=root, rate_mbps=rate, rate_measured=measured)
    own = session is None
    session = session or HttpSession(timeout=settings.timeout_s, retries=settings.retries)
    try:
        for source, names in wanted.items():
            desc = catalog.source(source)
            if desc.acquisition is None:
                plan.notes.append(f"{source}: the descriptor declares no acquisition section")
                continue
            plan.sources.append(plan_source(desc, names, settings,
                                            root=root if explicit else source_root(catalog, source, root),
                                            registry=registry, session=session, offline=offline, sizes=sizes,
                                            write_index=write_index))
        return plan
    finally:
        if own:
            session.close()


def _head_sizes(files: list[PlannedFile], session: HttpSession, workers: int) -> None:
    from dataclasses import replace

    def head(pf: PlannedFile) -> None:
        try:
            n = session.head_size(pf.remote.url)
        except AcquisitionError:
            return
        if n is not None:
            pf.remote = replace(pf.remote, size=n)
            if pf.state == "present" and pf.local_bytes != n:
                pf.state = "missing"

    if not files:
        return
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as pool:
        list(pool.map(head, files))


def _prepare_state(desc: Any, home: Path, step: str) -> str:
    spec = desc.acquisition.prepare[step]
    out = home / spec.output
    if not out.exists():
        return "pending"
    if spec.manifest:
        data = load_manifest(out / spec.manifest)
        if data.get(spec.complete_key) is True:
            return "done"
        return f"blocked: {out} exists without a complete {spec.manifest} (move it aside to prepare again)"
    return "done" if not spec.fresh else f"blocked: {out} exists (the step creates it whole; move it aside)"


# ---------------------------------------------------------------------------- execute


@dataclass
class FileResult:
    rel: str
    status: str                       # downloaded | present | failed | planned
    bytes: int = 0
    digests: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""
    groups: list[str] = field(default_factory=list)

    @property
    def sha1(self) -> str:
        return self.digests.get("sha1", "")

    @property
    def sha256(self) -> str:
        return self.digests.get("sha256", "")


@dataclass
class SourceResult:
    source: str
    release: str
    home: Path
    files: list[FileResult] = field(default_factory=list)
    manifest: ManifestReport | None = None
    prepare: dict[str, dict[str, Any]] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ""

    @property
    def failed(self) -> list[FileResult]:
        return [r for r in self.files if r.status == "failed"]

    @property
    def ok(self) -> bool:
        return not self.error and not self.failed and all(p.get("status") in ("done", "ran")
                                                           for p in self.prepare.values())


@dataclass
class AcquisitionReport:
    sources: list[SourceResult] = field(default_factory=list)
    seconds: float = 0.0
    bytes_downloaded: int = 0
    transfer_seconds: float = 0.0                      # the download phases only (the measured rate)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.sources)

    def lines(self) -> list[str]:
        out = []
        for s in self.sources:
            got = [r for r in s.files if r.status == "downloaded"]
            kept = [r for r in s.files if r.status == "present"]
            out.append(f"{s.source} {s.release}: {len(got)} file(s) downloaded ({fmt_bytes(sum(r.bytes for r in got))}"
                       f"), {len(kept)} already present, {len(s.failed)} failed, {s.seconds:.1f} s")
            out.extend(f"  FAILED {r.rel}: {r.error}" for r in s.failed)
            if s.error:
                out.append(f"  ERROR {s.error}")
            if s.manifest is not None:
                out.extend("  " + line for line in s.manifest.summary().splitlines())
            for step, p in s.prepare.items():
                out.append(f"  prepare {step}: {p.get('status')}" + (f" ({p.get('detail')})" if p.get("detail") else ""))
            for k, v in s.env.items():
                out.append(f"  env: {k}={v}")
        out.append(f"# {fmt_bytes(self.bytes_downloaded)} downloaded in {self.seconds:.1f} s")
        return out


def _verify(path: Path, f: RemoteFile, checks: Sequence[str]) -> dict[str, str]:
    """Digests of a complete file (sha256, sha1 and the publisher's algorithm); IntegrityError on a mismatch."""
    pub = strongest(f.checksums)
    algos = {"sha256", "sha1"} | ({pub[0]} if pub else set())
    size, digests = file_digests(path, algos)
    if "size" in checks and f.size is not None and size != f.size:
        raise IntegrityError(f"{size} bytes, the listing says {f.size}")
    if "checksum" in checks and pub and digests[pub[0]] != pub[1]:
        raise IntegrityError(f"{pub[0]} {digests[pub[0]]} differs from the listing ({pub[1]})")
    if "parquet_framing" in checks and f.path.endswith(".parquet") and not parquet_framed(path):
        raise IntegrityError("not a complete Parquet file")
    digests["bytes"] = str(size)
    return digests


class _Transfer:
    """Downloads of one source's files (parallel, resumable, verified)."""

    def __init__(self, plugin: Any, session: HttpSession, downloads: Path, checks: Sequence[str], *,
                 retry_wait: float = 1.0, on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.plugin = plugin
        self.session = session
        self.downloads = downloads
        self.checks = list(checks)
        self.retry_wait = retry_wait
        self.on_event = on_event
        self.lock = threading.Lock()
        self.received = 0
        self.ranged = "range" in (getattr(plugin, "capabilities", ()) or ())

    def present(self, pf: PlannedFile) -> FileResult | None:
        target = self.downloads / pf.remote.path
        if not target.is_file():
            return None
        try:
            digests = _verify(target, pf.remote, self.checks)
        except IntegrityError:
            return None
        return FileResult(pf.remote.path, "present", bytes=int(digests.pop("bytes")), digests=digests,
                          groups=pf.groups)

    def download(self, pf: PlannedFile) -> FileResult:
        f = pf.remote
        target = safe_join(self.downloads, f.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        t0 = time.monotonic()
        error = ""
        for attempt in range(ATTEMPTS):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset and (not self.ranged or (f.size is not None and offset > f.size)):
                    part.unlink(missing_ok=True)
                    offset = 0
                if not (f.size is not None and offset == f.size and offset > 0):
                    with part.open("ab" if offset else "wb") as fh:
                        for chunk in self.plugin.fetch(f, self.session, offset=offset):
                            fh.write(chunk)
                            with self.lock:
                                self.received += len(chunk)
                        fh.flush()
                        os.fsync(fh.fileno())
                try:
                    digests = _verify(part, f, self.checks)
                except IntegrityError:
                    part.unlink(missing_ok=True)
                    raise
                part.replace(target)
                res = FileResult(f.path, "downloaded", bytes=int(digests.pop("bytes")), digests=digests,
                                 seconds=round(time.monotonic() - t0, 2), groups=pf.groups)
                if self.on_event:
                    self.on_event("file", {"path": f.path, "status": "downloaded", "bytes": res.bytes})
                return res
            except AcquisitionError as exc:
                error = f"{type(exc).__name__}: {exc}"
                if exc.status == 416 or "ignored the range" in str(exc):
                    part.unlink(missing_ok=True)       # complete already, changed, or no ranges: start again
            except Exception as exc:  # noqa: BLE001 - retried, then reported per file
                error = f"{type(exc).__name__}: {exc}"
            if attempt < ATTEMPTS - 1 and self.retry_wait > 0:
                time.sleep(self.retry_wait * min(2 ** attempt, 8))
        if self.on_event:
            self.on_event("file", {"path": f.path, "status": "failed", "error": error})
        return FileResult(f.path, "failed", error=error, seconds=round(time.monotonic() - t0, 2), groups=pf.groups)


def _check_space(sp: SourcePlan, settings: AcquisitionSettings) -> str | None:
    need = sp.bytes_remaining or 0
    probe = sp.downloads
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return None
    if need + settings.reserve_bytes > free:
        return (f"{fmt_bytes(need)} to download into {sp.downloads} but {fmt_bytes(free)} is free (keeping "
                f"{fmt_bytes(settings.reserve_bytes)} in reserve, data.acquisition.reserve_bytes)")
    return None


def execute(plan: AcquisitionPlan, settings: AcquisitionSettings, *, registry: Any = None,
            session: HttpSession | None = None, workers: int | None = None, max_bytes: int | None = None,
            run_prepare: bool = True, config: Mapping[str, Any] | None = None, retry_wait: float = 1.0,
            on_event: Callable[[str, dict[str, Any]], None] | None = None,
            progress_s: float = 30.0) -> AcquisitionReport:
    """Carry out ``plan`` (see the module docstring). Raises ValueError before any transfer when the remaining
    bytes exceed ``max_bytes`` or the free disk."""
    registry = registry or acquisition_registry()
    rem = plan.bytes_remaining
    if max_bytes is not None and rem is not None and rem > max_bytes:
        raise ValueError(f"the plan transfers {fmt_bytes(rem)}, over the {fmt_bytes(max_bytes)} allowed")
    for sp in plan.sources:
        if sp.mode == "download" and sp.groups:
            why = _check_space(sp, settings)
            if why:
                raise ValueError(why)
    t_all = time.monotonic()
    report = AcquisitionReport()
    own = session is None
    session = session or HttpSession(timeout=settings.timeout_s, retries=settings.retries)
    n_workers = max(1, int(workers or settings.workers))
    try:
        for sp in plan.sources:
            if sp.mode != "download" or not sp.groups:
                continue
            res = SourceResult(sp.source, sp.release, sp.home, env=dict(sp.env))
            report.sources.append(res)
            t0 = time.monotonic()
            desc = sp.desc
            acq = desc.acquisition
            if sp.listing_error or not sp.listed:
                res.error = sp.listing_error or "nothing listed"
                continue
            plugin = registry.get("acquisition", acq.transport.plugin)
            lock_err = _check_lock(sp)
            if lock_err:
                res.error = lock_err
                continue
            sp.downloads.mkdir(parents=True, exist_ok=True)
            xfer = _Transfer(plugin, session, sp.downloads, acq.verify, retry_wait=retry_wait, on_event=on_event)
            todo: list[PlannedFile] = []
            entries = _manifest_entries(sp.downloads, acq, sp.release)[0]
            for pf in sp.files:
                if pf.state == "verified":
                    entry = entries.get(pf.remote.path) or {}
                    res.files.append(FileResult(pf.remote.path, "present", bytes=pf.local_bytes, groups=pf.groups,
                                                digests={k: str(v) for k, v in entry.items()
                                                         if k in ("sha256", "sha1", "md5", "git_sha1", "crc32")}))
                elif pf.state == "present":
                    got = xfer.present(pf)
                    if got is not None:
                        res.files.append(got)
                    else:
                        todo.append(pf)
                else:
                    todo.append(pf)
            todo.sort(key=lambda p: -(p.remote.size or 0))
            t_xfer = time.monotonic()
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                pending = {pool.submit(xfer.download, pf) for pf in todo}
                last = time.monotonic()
                while pending:
                    done, pending = wait(pending, timeout=max(progress_s, 0.1), return_when=FIRST_COMPLETED)
                    for fut in done:
                        res.files.append(fut.result())
                    if on_event and time.monotonic() - last >= progress_s:
                        last = time.monotonic()
                        on_event("progress", {"source": sp.source, "done": len(res.files), "of": len(sp.files),
                                              "received": xfer.received})
            report.bytes_downloaded += xfer.received
            if todo:
                report.transfer_seconds += time.monotonic() - t_xfer
            known = {r.rel: {"bytes": r.bytes, **r.digests} for r in res.files if r.status in ("downloaded", "present")}
            failed_groups = {g for r in res.failed for g in r.groups}
            good = [g for g in sp.groups if g not in failed_groups and (acq.tables.get(g) or acq.extra.get(g)).files]
            base = _manifest_base(acq, desc, sp.release)
            index = next((dict(f.extra.get("index")) for f in sp.listed if isinstance(f.extra.get("index"), Mapping)),
                         {})
            if acq.manifest:
                skip = [acq.index_cache] if acq.index_cache else []
                res.manifest = write_manifest(acq, sp.downloads, sp.listed, release=sp.release, base=base,
                                              groups=good, known=known, about=index, skip_dirs=skip)
            if run_prepare:
                for step in sp.prepare:
                    res.prepare[step] = _run_prepare(desc, sp, step, failed_groups, config, on_event)
            _write_lock(sp, acq, res, index)
            res.seconds = round(time.monotonic() - t0, 2)
    finally:
        if own:
            session.close()
    report.seconds = round(time.monotonic() - t_all, 2)
    report.transfer_seconds = round(report.transfer_seconds, 2)
    _record_rate(settings, report.bytes_downloaded, report.transfer_seconds)
    return report


def _manifest_base(acq: Any, desc: Any, release: str) -> str:
    if acq.manifest_base:
        return substitute(acq.manifest_base, {"release": release})
    opts = transport_options(desc, release)
    return str(opts.get("base") or opts.get("index") or opts.get("archive") or "")


def _check_lock(sp: SourcePlan) -> str | None:
    lock = load_manifest(sp.home / LOCK)
    if lock and lock.get("release") not in (None, sp.release):
        return (f"{sp.home} holds release {lock.get('release')} (its {LOCK}); acquire release {sp.release} into "
                f"another directory (the acquisition `dir` names the release by default)")
    return None


def _write_lock(sp: SourcePlan, acq: Any, res: SourceResult, index: Mapping[str, Any]) -> None:
    lock = load_manifest(sp.home / LOCK)
    groups = dict(lock.get("groups") or {}) if lock.get("release") in (None, sp.release) else {}
    for g in sp.groups:
        mine = [r for r in res.files if g in r.groups]
        if mine and not any(r.status == "failed" for r in mine):
            groups[g] = {"files": len(mine), "bytes": sum(r.bytes for r in mine),
                         "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    prepared = dict(lock.get("prepare") or {})
    prepared.update({k: v for k, v in res.prepare.items() if v.get("status") in ("ran", "done")})
    data = {"source": sp.source, "release": sp.release, "transport": acq.transport.plugin if acq.transport else None,
            "from": sp.describe, "index": dict(index), "groups": dict(sorted(groups.items())), "prepare": prepared,
            "env": sp.env, "licence": acq.licence,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        atomic_write(sp.home / LOCK, (json.dumps(data, indent=2, default=str) + "\n").encode())
    except OSError as exc:
        res.error = res.error or f"could not write {sp.home / LOCK}: {exc}"


def _upstream_dir(config: Mapping[str, Any] | None) -> str:
    from .. import config as _config

    up = ((config or {}).get("vars") or {}).get("upstream") or os.environ.get("VBT_UPSTREAM") or \
        str(_config.PROJECT_ROOT / "third_party" / "TheVirtualBiotech")
    return str(_config.resolve_path(up))


def _run_prepare(desc: Any, sp: SourcePlan, step: str, failed: set[str], config: Mapping[str, Any] | None,
                 on_event: Callable[[str, dict[str, Any]], None] | None) -> dict[str, Any]:
    spec = desc.acquisition.prepare[step]
    blocked = [g for g in spec.needs if g in failed]
    if blocked:
        return {"status": "blocked", "detail": f"downloads failed for {', '.join(blocked)}"}
    state = _prepare_state(desc, sp.home, step)
    if state == "done":
        return {"status": "done", "detail": f"{sp.home / spec.output} is complete"}
    if state.startswith("blocked"):
        return {"status": "blocked", "detail": state.partition(": ")[2]}
    upstream = _upstream_dir(config)
    v = {"python": sys.executable, "upstream": upstream, "home": str(sp.home), "downloads": str(sp.downloads),
         "output": str(sp.home / spec.output), "release": sp.release}
    argv = [substitute(a, v) for a in spec.command]
    script = next((a for a in argv if a.endswith(".py")), None)
    if script is not None and not Path(script).is_file():
        return {"status": "blocked", "detail": f"{script} is missing (the upstream checkout, vars.upstream)"}
    if on_event:
        on_event("prepare", {"step": step, "argv": argv})
    t0 = time.monotonic()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run(argv, cwd=str(sp.home), env=env, capture_output=True, text=True)
    # ru_maxrss of the waited-for children: the step's peak unless an earlier child of this process was larger
    out = {"status": "ran" if proc.returncode == 0 else "failed", "argv": argv, "returncode": proc.returncode,
           "seconds": round(time.monotonic() - t0, 2), "children_max_rss_mb": _children_rss() or None,
           "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    if proc.returncode != 0:
        out["detail"] = (proc.stderr.strip() or proc.stdout.strip())[-300:]
    elif spec.manifest and load_manifest(sp.home / spec.output / spec.manifest).get(spec.complete_key) is not True:
        out["status"] = "failed"
        out["detail"] = f"{spec.manifest} is not complete after the step"
    return out


def _children_rss() -> int:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------- environment and provenance


def write_env_file(path: Path, values: Mapping[str, str]) -> list[str]:
    """Set ``KEY="value"`` lines of a dotenv file (existing keys replaced in place, others appended); returns the
    keys written."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.is_file() else []
    done: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip() if "=" in line else ""
        if key in values and not line.lstrip().startswith("#"):
            out.append(f'{key}="{values[key]}"')
            done.add(key)
        else:
            out.append(line)
    out.extend(f'{k}="{v}"' for k, v in values.items() if k not in done)
    atomic_write(path, ("\n".join(out) + "\n").encode())
    return list(values)


def record_provenance(settings: AcquisitionSettings, report: AcquisitionReport, *, by: str,
                      run_dir: Path | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Append one acquisition record to ``<data.acquisition.root>/acquisitions.jsonl`` (and to
    ``<run_dir>/data_acquisitions.jsonl`` when the acquisition happened during a run)."""
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "by": by, "policy": settings.auto,
           "budget_bytes": settings.budget_bytes, "seconds": report.seconds, "bytes": report.bytes_downloaded,
           "ok": report.ok, **dict(extra or {}),
           "sources": [{"source": s.source, "release": s.release, "home": str(s.home),
                        "downloaded": sum(1 for r in s.files if r.status == "downloaded"),
                        "present": sum(1 for r in s.files if r.status == "present"),
                        "failed": [r.rel for r in s.failed][:50], "error": s.error,
                        "manifest": str(s.manifest.path) if s.manifest else None,
                        "tables": sorted(s.manifest.tables) if s.manifest else [],
                        "prepare": {k: {kk: vv for kk, vv in v.items() if kk not in ("stdout_tail", "stderr_tail")}
                                    for k, v in s.prepare.items()}} for s in report.sources]}
    line = json.dumps(rec, default=str, sort_keys=True) + "\n"
    for d, name in ((settings.provenance_dir, "acquisitions.jsonl"), (run_dir, "data_acquisitions.jsonl")):
        if d is None:
            continue
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            with (Path(d) / name).open("a") as fh:
                fh.write(line)
        except OSError:
            pass
    return rec


def plan_dict(plan: AcquisitionPlan) -> dict[str, Any]:
    return plan.to_dict()


def report_dict(report: AcquisitionReport) -> dict[str, Any]:
    return {"ok": report.ok, "seconds": report.seconds, "bytes_downloaded": report.bytes_downloaded,
            "sources": [{"source": s.source, "release": s.release, "home": str(s.home), "error": s.error,
                         "files": [asdict(r) for r in s.files if r.status != "present"],
                         "present": sum(1 for r in s.files if r.status == "present"),
                         "manifest": None if s.manifest is None else {
                             "path": str(s.manifest.path), "tables": s.manifest.tables, "skipped": s.manifest.skipped,
                             "files": s.manifest.files, "bytes": s.manifest.bytes},
                         "prepare": s.prepare, "env": s.env} for s in report.sources]}

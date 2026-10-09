"""``vbt data status``: every declared table with present / verified / ready and the tools it unlocks. No pyarrow.

* **present**: the files the data layer would read now (the descriptor's root as the environment sets it, the
  table's layout plugin listing its fragments); a source read live is ``remote``.
* **verified**: every present file is in the acquisition manifest with its size (``yes``), some are not (``no``),
  or there is no manifest (``-``); a table written by a prepare step is verified by that step's manifest.
* **ready**: the table's status in the readiness cache (``vbt ds check``; ``--check`` runs it), else ``unchecked``.
* **unlocks**: the reviewed tools whose bindings read the table (:func:`vbt.data.targets.tools_by_table`).

When the environment does not point at a source but its acquisition home holds the files, the source line says
which variable to set (``vbt data acquire --env-file`` writes it).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .acquire import AcquisitionSettings, source_env, source_home, source_release
from .manifest import MANIFEST, load_manifest

__all__ = ["TableStatus", "SourceStatus", "collect_status", "status_lines"]


@dataclass
class TableStatus:
    table: str
    physical: str
    present: str                      # "<n> file(s)" | absent | remote | error: ...
    files: int = 0
    bytes: int = 0
    verified: str = "-"               # yes | no | -
    ready: str = "unchecked"
    tools: list[str] = field(default_factory=list)
    acquirable: bool = False


@dataclass
class SourceStatus:
    source: str
    kind: str
    release: str
    root: str | None
    home: str | None
    env: dict[str, str] = field(default_factory=dict)
    env_set: dict[str, bool] = field(default_factory=dict)
    acquired: list[str] = field(default_factory=list)
    hint: str = ""
    tables: list[TableStatus] = field(default_factory=list)


def _fragments(catalog_table: Any, registry: Any, root: str) -> tuple[int, int, list[Path], str]:
    """``(files, bytes, local paths, error)`` the table's layout lists under ``root``."""
    from ..datalayer.gateway.readiness import layout_spec

    plugin = registry.find("layout", catalog_table.layout) if registry is not None and catalog_table.layout else None
    if plugin is None or "scan" not in (getattr(plugin, "capabilities", ()) or ()):
        return 0, 0, [], "no local layout"
    try:
        frags = plugin.fragments(root, layout_spec(catalog_table))
    except Exception as exc:  # noqa: BLE001 - an unreadable location is reported, never raised
        return 0, 0, [], f"{type(exc).__name__}: {exc}"[:200]
    paths = [Path(str(f.uri)) for f in frags if "://" not in str(f.uri)]
    return len(frags), sum(int(f.size or 0) for f in frags), paths, ""


def _verified(desc: Any, table: str, root: str, paths: Sequence[Path]) -> str:
    """``yes`` / ``no`` / ``-`` (see the module docstring). The manifest is looked for in ``root``, else (a table
    whose path is a file named by a variable, root unset) in the directory of its files."""
    acq = desc.acquisition
    entry = acq.tables.get(table) if acq is not None else None
    base = Path(root) if root else (paths[0].parent if paths else None)
    if base is None:
        return "-"
    if entry is not None and entry.prepared_by:
        step = acq.prepare[entry.prepared_by]
        data = load_manifest(base / step.manifest) if step.manifest else {}
        return "yes" if data.get(step.complete_key) is True else ("no" if data else "-")
    name = (acq.manifest if acq is not None else None) or MANIFEST
    # the manifest sits in the downloads directory, which may be a parent of the root (zenodo_vbt: the root is a
    # folder of the archive)
    for _ in range(4):
        if (base / name).is_file() or base.parent == base:
            break
        base = base.parent
    files = load_manifest(base / name).get("files")
    if not isinstance(files, dict) or not paths:
        return "-"
    for p in paths:
        try:
            rel, size = p.relative_to(base).as_posix(), p.stat().st_size
        except (ValueError, OSError):
            return "no"
        got = files.get(rel)
        if not isinstance(got, dict) or got.get("bytes") != size:
            return "no"
    return "yes"


def collect_status(config: Mapping[str, Any], *, sources: Sequence[str] = (), check: bool = False,
                   depth: str = "standard") -> list[SourceStatus]:
    """The status of every declared table (of ``sources`` when given)."""
    from ..datalayer.gateway.readiness import ReadinessCache, table_status
    from ..preflight import data_catalog
    from .targets import tools_by_table

    settings, catalog, registry = data_catalog(dict(config))
    acq_settings = AcquisitionSettings.from_config(config)
    unlocks = tools_by_table(catalog)
    names = [s for s in catalog.sources if not sources or s in sources]
    out: list[SourceStatus] = []
    present_refs: list[str] = []
    for source in sorted(names):
        desc = catalog.source(source)
        acq = desc.acquisition
        release = source_release(desc)
        remote = desc.kind == "remote" or (acq is not None and acq.mode == "remote")
        home = source_home(desc, acq_settings.root, release) if acq is not None and not remote else None
        env = source_env(desc, home, release) if home is not None else {}
        st = SourceStatus(source=source, kind="remote" if remote else "local", release=release,
                          root=str(desc.root) if desc.root else None, home=str(home) if home else None, env=env,
                          env_set={k: bool(os.environ.get(k)) for k in env})
        if home is not None and (home / ".vbt-acquisition.json").is_file():
            lock = load_manifest(home / ".vbt-acquisition.json")
            st.acquired = sorted((lock.get("groups") or {}).keys())
        for tname in desc.tables:
            ref = f"{source}.{tname}"
            try:
                ct = catalog.table(ref)
            except Exception as exc:  # noqa: BLE001
                st.tables.append(TableStatus(ref, ref, f"error: {exc}"[:120]))
                continue
            phys = str(ct.physical)
            ts = TableStatus(ref, phys, "remote" if remote else "absent", tools=list(unlocks.get(phys, [])),
                             acquirable=acq is not None and (tname in acq.tables or phys.split(".")[1] in acq.tables))
            if not remote:
                root = str(ct.descriptor.root or "")
                if not root and not Path(str(ct.physical_spec.path or "")).is_absolute():
                    st.tables.append(ts)                # the root variable is unset: nothing to look at
                    continue
                n, nbytes, paths, err = _fragments(ct, registry, root)
                if err and err != "no local layout":
                    ts.present = f"error: {err}"
                elif err:
                    ts.present = "n/a"
                elif n:
                    ts.present, ts.files, ts.bytes = f"{n} file(s)", n, nbytes
                    present_refs.append(phys)
                    ts.verified = _verified(desc, phys.split(".", 1)[1], root, paths)
            st.tables.append(ts)
        missing_env = [k for k, v in st.env_set.items() if not v]
        if missing_env and home is not None and st.acquired:
            st.hint = "acquired at " + str(home) + "; point the data layer at it: " + \
                " ".join(f"{k}={env[k]}" for k in missing_env) + " (vbt data acquire --env-file .env writes it)"
        out.append(st)
    cache = ReadinessCache(settings.cache_dir, catalog, registry)
    cache.load()
    if check and present_refs:
        from ..preflight import run_data_check

        cache.load_check_results(run_data_check(dict(config), tables=sorted(set(present_refs)), depth=depth))
    for st in out:
        for ts in st.tables:
            m = cache.get(ts.physical)
            if m is None:
                continue
            item = ts.table.split(".", 1)[1] if ts.table != ts.physical else None
            ts.ready = (m.item_tables.get(item) or table_status(m)) if item else table_status(m)
    return out


def status_lines(statuses: Sequence[SourceStatus], *, verbose: bool = False) -> list[str]:
    lines: list[str] = []
    width = max((len(t.table) for s in statuses for t in s.tables), default=20)
    for s in statuses:
        head = f"{s.source} [{s.kind}] release {s.release}"
        if s.kind == "local":
            head += f"  root: {s.root or '(unset)'}"
            if s.acquired:
                head += f"  acquired: {len(s.acquired)} group(s) at {s.home}"
        lines.append(head)
        if s.hint:
            lines.append(f"  hint: {s.hint}")
        lines.append(f"  {'table':<{width}}  {'present':<12} {'verified':<8} {'ready':<16} unlocks")
        for t in s.tables:
            tools = f"{len(t.tools)} tool(s)" if not verbose else (", ".join(t.tools) or "-")
            lines.append(f"  {t.table:<{width}}  {t.present:<12} {t.verified:<8} {t.ready:<16} {tools}")
    return lines


def status_dict(statuses: Sequence[SourceStatus]) -> list[dict[str, Any]]:
    return [asdict(s) for s in statuses]

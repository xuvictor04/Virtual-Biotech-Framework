"""The data child (§4, §11.8): a harness-owned FastMCP server that reads the data. May import pyarrow lazily.

Only this package reads data files. It runs as its own process (``service/server.py``) under the
same launcher and memory limit as the upstream servers, so a heavy scan never takes down the
orchestrator. The harness reaches it through the hidden verbs of ``service/verbs/*.py``.

:class:`ServiceContext` is what every verb receives: the settings (from ``VBT_DATA_SETTINGS``),
the plugin registry, the catalog, the resolver :class:`~vbt.datalayer.resolve.index.IndexStore`
and the caches that make repeated calls cheap: one :class:`~vbt.datalayer.service.reader.TableReader`
per table, fingerprints per stat-only signature, parsed footers per fingerprint and loaded
row-group sidecar indexes.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from ..catalog import Catalog, CatalogTable, TableRef, build_catalog
from ..descriptor.models import plugin_name
from ..plugins.base import LayoutSpec, Manifest
from ..resolve.index import IndexStore
from ..settings import DataSettings

__all__ = ["ServiceContext", "ServiceError", "json_path", "load_manifest", "layout_spec"]


class ServiceError(RuntimeError):
    """A request the data child cannot serve (unknown table, unsupported table kind, bad arguments)."""


def json_path(data: Any, path: str | None) -> Any:
    """The value at a dotted JSONPath subset (``$.files``, ``$.rows.permissive``, ``$.a[0].b``); None when absent."""
    if not path:
        return None
    text = str(path).strip()
    if text.startswith("$"):
        text = text[1:]
    cur = data
    for part in [p for p in text.replace("[", ".[").split(".") if p]:
        if part.startswith("[") and part.endswith("]"):
            try:
                cur = cur[int(part[1:-1])] if isinstance(cur, list) else None
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, Mapping):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _rebased(entries: Mapping[str, Any], manifest_dir: Path, root: Path) -> dict[str, Any]:
    """Entry paths are relative to the manifest's own directory (the downloader's format); a manifest kept above
    the root (``path: ../.download-manifest.json``, the Zenodo archive's top directory) has them made relative
    to the root, and entries outside the root dropped."""
    a, b = os.path.normpath(os.path.abspath(manifest_dir)), os.path.normpath(os.path.abspath(root))
    if a == b:
        return dict(entries)
    out = {}
    for key, entry in entries.items():
        rel = os.path.relpath(os.path.normpath(os.path.join(a, str(key).lstrip("./"))), b).replace(os.sep, "/")
        if rel != ".." and not rel.startswith("../"):
            out[rel] = entry
    return out


def load_manifest(root: str | None, spec: Any) -> Manifest | None:
    """A source's manifest (``ManifestSpec``): a JSON file under ``root`` or an inline map; None when the file
    is absent or unreadable (R2 reports that)."""
    if spec is None:
        return None
    if spec.inline is not None:
        return Manifest(path=None, entries=dict(spec.inline), data={}, sha256=None, complete=None)
    path = Path(root or "") / str(spec.path) if root else Path(str(spec.path))
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        return None
    entries = json_path(data, spec.entries) or {}
    if isinstance(entries, list):                      # [{path, bytes, sha256}] lists are accepted too
        entries = {str(e.get("path") or e.get("name")): e for e in entries if isinstance(e, Mapping)}
    if isinstance(entries, Mapping) and root:
        entries = _rebased(entries, path.parent, Path(root))
    complete = data.get("complete") if isinstance(data, Mapping) else None
    return Manifest(path=str(path), entries=dict(entries) if isinstance(entries, Mapping) else {},
                    data=data if isinstance(data, Mapping) else {}, sha256=hashlib.sha256(raw).hexdigest(),
                    complete=complete if isinstance(complete, bool) else None)


def layout_spec(table: CatalogTable) -> LayoutSpec:
    """The :class:`LayoutSpec` of a table's physical table (item tables share their parent's)."""
    desc = table.descriptor
    spec = table.physical_spec
    ref = spec.layout if spec.layout is not None else desc.defaults.layout
    options = dict(getattr(ref, "options", {}) or {})
    partitions = {name: p.type for name, p in spec.partitions.items()}
    expect: dict[str, Any] = {}
    for name, p in spec.partitions.items():
        vocab = getattr(p.column, "vocab", None)
        expect[name] = list(vocab) if p.expect == "declared" and isinstance(vocab, list) else None
    fk = spec.fragment_key.model_dump(by_alias=True) if spec.fragment_key is not None else None
    return LayoutSpec(table=str(table.physical), path=spec.path, options=options, partitions=partitions,
                      partition_expect=expect, fragment_key=fk, format=table.format)


class ServiceContext:
    """Settings, registry, catalog and caches shared by every verb of one data child."""

    def __init__(self, settings: DataSettings | None = None, *, catalog: Catalog | None = None,
                 registry: Any = None) -> None:
        from ..plugins.registry import discover

        self.settings = settings or DataSettings.from_env()
        self.registry = registry if registry is not None else discover(self.settings)
        self.catalog = catalog if catalog is not None else build_catalog(self.settings, self.registry)
        if self.catalog.registry is None:
            self.catalog.registry = self.registry
        self.index_store = IndexStore(self.settings.cache_dir)
        self.readers: dict[str, Any] = {}
        self.fingerprints: dict[tuple[str, str], str] = {}     # (physical table, signature) -> fingerprint
        self.footers: dict[tuple[str, str], Any] = {}          # (fingerprint, fragment uri) -> parsed footer
        self.sidecars: dict[str, Any] = {}                     # sidecar path -> loaded row-group value index
        self.parents: dict[str, dict[str, str]] = {}           # id_type -> {key: parent key}
        self.vocab: dict[str, Any] = {}                        # snapshot id -> ValueSnapshot
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(max(1, int(self.settings.service.max_concurrency)))

    # -- catalog -------------------------------------------------------------------

    def table(self, ref: str | TableRef) -> CatalogTable:
        try:
            return self.catalog.table(ref)
        except LookupError as exc:
            raise ServiceError(str(exc)) from None

    def table_refs(self, *, item_tables: bool = True) -> list[str]:
        out = []
        for ref in self.catalog.table_refs():
            spec = self.catalog.source(ref.source).tables[ref.table]
            if spec.items_of is not None and not item_tables:
                continue
            out.append(str(ref))
        return out

    def item_tables_of(self, physical: str) -> list[str]:
        """Item tables whose rows are items of ``physical`` (``source.table``)."""
        out = []
        for ref in self.table_refs():
            t = self.table(ref)
            if t.is_item_table and str(t.physical) == physical:
                out.append(ref)
        return out

    def reader(self, ref: str | TableRef) -> Any:
        from .reader import TableReader

        key = str(TableRef.parse(ref))
        with self.lock:
            r = self.readers.get(key)
            if r is None:
                r = self.readers[key] = TableReader(self, key)
        return r

    # -- plugins ---------------------------------------------------------------------

    def plugin(self, kind: str, name: str | None) -> Any:
        if not name:
            raise ServiceError(f"no {kind} plugin named for this table")
        p = self.registry.find(kind, str(plugin_name(name)))
        if p is None:
            raise ServiceError(f"{kind} plugin {name!r} is not registered (plugin_unavailable)")
        return p

    def format_plugin(self, table: CatalogTable) -> Any:
        """The format plugin of a table, configured with its ``FormatRef.options`` and matrix spec (§6.7)
        when the plugin takes configuration (``configure(options, matrix=...)``)."""
        plugin = self.plugin("format", table.format)
        spec = table.physical_spec
        ref = spec.format if spec.format is not None else table.descriptor.defaults.format
        options = dict(getattr(ref, "options", {}) or {})
        matrix = spec.matrix
        if (options or matrix is not None) and callable(getattr(plugin, "configure", None)):
            plugin = plugin.configure(options, matrix=matrix)
        return plugin

    def identifier(self, qualified: str, sample: Any = None) -> Any:
        """The configured identifier plugin of an id_type (``source:name`` or bare with a unique source)."""
        src, spec = self.catalog.id_type(qualified)
        plugin = self.plugin("identifier", spec.plugin)
        return plugin.configure(dict(spec.options or {}), list(sample) if sample is not None else None)

    def statistic(self, name: str | None) -> Any:
        return self.registry.find("statistic", name) if name else None

    # -- budgets, manifests, paths -----------------------------------------------------

    def scan_budget(self, table: CatalogTable, requested: int | None = None) -> int:
        """Decoded bytes one request may scan: the request's, else the table's, else ``data.witness.max_scan_bytes``."""
        limits = [int(self.settings.witness.max_scan_bytes)]
        if table.physical_spec.max_scan_bytes:
            limits.append(int(table.physical_spec.max_scan_bytes))
        if requested:
            limits.append(int(requested))
        return min(limits)

    def manifest(self, source: str) -> Manifest | None:
        desc = self.catalog.source(source)
        for spec in desc.manifests:
            m = load_manifest(desc.root, spec)
            if m is not None:
                return m
        return None

    def cache_path(self, source: str, fingerprint: str, *parts: str) -> Path:
        safe = fingerprint.replace(":", "_").replace(os.sep, "_")
        return Path(self.settings.cache_dir) / source / safe / Path(*parts)

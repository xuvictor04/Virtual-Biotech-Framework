"""``soma``: TileDB-SOMA dataframes of the CELLxGENE Census, read live (phase 4, F20). Stdlib only at import.

Options (``layout: {plugin: soma, options: {...}}``)::

    uri:            census_data/homo_sapiens/obs     the dataframe inside the Census (``ms/RNA/var`` for genes)
    census_version: stable                            what the upstream server opens; resolved, never assumed
    module:         cellxgene_census                  the Census client module (imported in the data child)
    tiledb_config:  {soma.init_buffer_bytes: 134217728, py.init_buffer_bytes: 134217728}
                                                      passed to ``open_soma`` (default: 128 MiB read buffers)

Read buffers: ``cellxgene_census.open_soma`` reserves 1 GiB per column read by default. Under the data
child's ``RLIMIT_DATA`` (which counts reserved, not resident, memory) that fails with ``std::bad_alloc``
before any data arrives: on the real Census (2025-11-08) a one-column count failed under a 5000 MB limit
at 260 MB resident. With 128 MiB buffers the same count of one dataset (7,750 cells) took 2.9 s, and
counting a tissue (1.6 M cells) about 1.3 s. Counts stream the ``soma_joinid`` batches (never one
table of every id).

``stable`` is an alias that moves. :func:`resolve_version` asks the client
(``cellxgene_census.get_census_version_description``) which dated release it names, records it, and
reports **drift** when the same alias resolved to another release earlier in this process (or in the
record kept under ``data.cache_dir``): results of one session must not silently mix releases.
``release_confidence`` is ``exact`` for a dated version, ``inferred`` for an alias resolved through the
description, and ``unknown`` when the description cannot be read.

Implements ``live`` (:meth:`SomaLayout.request`: one read, the whole match; SOMA does not page) and
``count`` (:meth:`SomaLayout.count`: the count-first admission of Census pulls and the remote witness;
the predicate is compiled to a ``value_filter`` by the ``soma`` format, and a residual makes the count
unknown). ``probe``, ``signature`` and ``fingerprint`` never open the store.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Mapping

from ...predicate import Predicate, evaluate
from ...predicate import columns as _predicate_columns
from ..base import CheckItem, Fragment, LayoutSpec, Manifest, Page, PluginBase
from ..formats.soma import SomaFormat
from ..registry import register

__all__ = ["SomaLayout", "Resolved", "resolve_version", "open_dataframe", "soma_uri", "census_module",
           "count_rows", "read_columns", "VERSION_RECORD", "DEFAULT_TILEDB_CONFIG"]

DEFAULT_MODULE = "cellxgene_census"
VERSION_RECORD = "census_versions.json"
DEFAULT_TILEDB_CONFIG = {"soma.init_buffer_bytes": 128 * 1024 ** 2, "py.init_buffer_bytes": 128 * 1024 ** 2}
_DATED = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RESOLVED: dict[str, str] = {}                 # requested alias -> dated release, this process
_AS_OF: dict[str, str] = {}


def census_module(name: str | None = None) -> Any:
    """The Census client module (``cellxgene_census``; tests insert a stub in ``sys.modules``)."""
    return importlib.import_module(name or DEFAULT_MODULE)


class Resolved(dict):
    """``{requested, resolved, release_confidence, drift?, description?}``."""


def resolve_version(requested: str = "stable", *, module: Any = None, record: str | Path | None = None
                    ) -> Resolved:
    """Resolve a Census version alias to its dated release, detecting drift (module docstring)."""
    requested = str(requested or "stable")
    mod = module if module is not None else census_module()
    previous = _RESOLVED.get(requested)
    stored: dict[str, Any] = {}
    if record is not None:
        try:
            stored = json.loads(Path(record).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        previous = previous or (stored.get(requested) or {}).get("resolved")
    if _DATED.fullmatch(requested):
        out = Resolved(requested=requested, resolved=requested, release_confidence="exact")
    else:
        try:
            desc = dict(mod.get_census_version_description(requested) or {})
        except Exception as exc:  # noqa: BLE001 - an unreadable description leaves the release unknown
            return Resolved(requested=requested, resolved=None, release_confidence="unknown",
                            reason=f"{type(exc).__name__}: {exc}")
        resolved = desc.get("release_build") or desc.get("census_version") or desc.get("release_date")
        out = Resolved(requested=requested, resolved=str(resolved) if resolved else None,
                       release_confidence="inferred" if resolved else "unknown",
                       description={k: desc[k] for k in ("release_build", "release_date", "alias") if k in desc})
    if previous and out.get("resolved") and previous != out["resolved"]:
        out["drift"] = {"before": previous, "now": out["resolved"]}
    if out.get("resolved"):
        _RESOLVED[requested] = str(out["resolved"])
        if record is not None:
            stored[requested] = {"resolved": out["resolved"],
                                 "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            try:
                Path(record).parent.mkdir(parents=True, exist_ok=True)
                Path(record).write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")
            except OSError:
                pass
    return out


def soma_uri(version: str, path: str) -> str:
    return f"soma://{version}/{str(path).strip('/')}"


def _walk(root: Any, path: str) -> Any:
    node = root
    for part in [p for p in str(path).split("/") if p]:
        try:
            node = node[part]
        except (KeyError, TypeError, IndexError):
            node = getattr(node, part)
    return node


class _Opened:
    def __init__(self, census: Any, frame: Any) -> None:
        self.census = census
        self.frame = frame

    def __enter__(self) -> Any:
        return self.frame

    def __exit__(self, *exc: Any) -> None:
        close = getattr(self.census, "close", None)
        if callable(close):
            close()


def _open(version: str, path: str, module: Any = None, tiledb_config: Mapping[str, Any] | None = None) -> _Opened:
    mod = module if module is not None else census_module()
    kwargs: dict[str, Any] = {"census_version": version}
    config = dict(DEFAULT_TILEDB_CONFIG if tiledb_config is None else tiledb_config)
    if config and _accepts(mod.open_soma, "tiledb_config"):
        kwargs["tiledb_config"] = config
    census = mod.open_soma(**kwargs)
    return _Opened(census, _walk(census, path))


def _accepts(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def open_dataframe(uri: str, module: Any = None) -> Any:
    """The SOMA dataframe a ``soma://<version>/<path>`` URI names (left open; for format reads)."""
    if not str(uri).startswith("soma://"):
        raise ValueError(f"not a soma:// URI: {uri!r}")
    version, _, path = str(uri)[len("soma://"):].partition("/")
    return _open(version, path, module).frame


def count_rows(result: Any) -> int:
    """Rows of a SOMA read, batch by batch when the read iterates (``ReadIter`` of Arrow tables), else of
    its ``concat()``."""
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
        return int(sum(getattr(t, "num_rows", None) if getattr(t, "num_rows", None) is not None else len(t)
                       for t in result))
    return int(len(result.concat()))


def _rows(frame: Any, value_filter: str | None, columns: list[str] | None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if value_filter:
        kwargs["value_filter"] = value_filter
    if columns:
        kwargs["column_names"] = list(columns)
    table = frame.read(**kwargs).concat()
    if hasattr(table, "to_pylist"):
        return list(table.to_pylist())
    return list(table.to_pandas().to_dict("records"))


def read_columns(frame: Any, value_filter: str | None, columns: list[str]) -> dict[str, list[Any]]:
    """``{column: values}`` of a SOMA read (column by column: no per-row dict for millions of cells)."""
    kwargs: dict[str, Any] = {"column_names": list(columns)}
    if value_filter:
        kwargs["value_filter"] = value_filter
    table = frame.read(**kwargs).concat()
    if hasattr(table, "column") and hasattr(table, "num_rows"):
        return {c: table.column(c).to_pylist() for c in columns}
    rows = table.to_pylist() if hasattr(table, "to_pylist") else table.to_pandas().to_dict("records")
    return {c: [r.get(c) for r in rows] for c in columns}


@register
class SomaLayout(PluginBase):
    kind: ClassVar[str] = "layout"
    name: ClassVar[str] = "soma"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"live", "count"})

    def __init__(self) -> None:
        self.fmt = SomaFormat()
        self.module: Any = None                   # a seam: the Census client (default: imported by name)
        self.record: str | None = None            # where resolved versions are kept (data.cache_dir)

    def _config(self, spec: LayoutSpec) -> dict[str, Any] | None:
        cfg = self._options(spec).get("tiledb_config")
        return dict(cfg) if isinstance(cfg, Mapping) else None

    @staticmethod
    def _options(spec: LayoutSpec) -> dict[str, Any]:
        return dict(spec.options or {})

    def _module(self, spec: LayoutSpec) -> Any:
        return self.module if self.module is not None else census_module(self._options(spec).get("module"))

    def _uri(self, spec: LayoutSpec) -> str | None:
        opts = self._options(spec)
        path = opts.get("uri") or spec.path
        if not path:
            return None
        return soma_uri(str(opts.get("census_version") or "stable"), str(path))

    # ------------------------------------------------------------------ identity (no store access)

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        return []

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {}

    def signature(self, root: str, spec: LayoutSpec) -> str:
        key = json.dumps([spec.table, self._uri(spec)], sort_keys=True)
        return "sig1:soma:" + hashlib.sha256(key.encode()).hexdigest()[:16]

    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str:
        return "fp1:soma"

    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]:
        return {}

    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]:
        uri = self._uri(spec)
        if not uri:
            return [CheckItem("location", False, f"{spec.table}: soma needs options.uri", level="error",
                              hint="set layout.options.uri (e.g. census_data/homo_sapiens/obs)")]
        return [CheckItem("upstream_only", True, f"live SOMA dataframe {uri} (checked at call time)", level="info")]

    def as_of(self, root: str, spec: LayoutSpec) -> str | None:
        requested = str(self._options(spec).get("census_version") or "stable")
        return _RESOLVED.get(requested) or _AS_OF.get(self._uri(spec) or "") or \
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------ reads

    def resolve(self, spec: LayoutSpec) -> Resolved:
        requested = str(self._options(spec).get("census_version") or "stable")
        return resolve_version(requested, module=self._module(spec), record=self.record)

    def _read_target(self, spec: LayoutSpec) -> tuple[str, str, Resolved]:
        opts = self._options(spec)
        path = str(opts.get("uri") or spec.path or "")
        if not path:
            raise ValueError(f"{spec.table}: soma needs options.uri")
        res = self.resolve(spec)
        version = str(res.get("resolved") or opts.get("census_version") or "stable")
        return version, path, res

    def request(self, spec: LayoutSpec, *, predicate: Predicate | None, projection: list[str],
                page_token: str | None, budget: Any) -> Page:
        value_filter, residual = self.fmt.compile(predicate)
        version, path, res = self._read_target(spec)
        with _open(version, path, self._module(spec), self._config(spec)) as frame:
            rows = _rows(frame, value_filter, list(projection or []) or None)
        if residual is not None:
            rows = [r for r in rows if evaluate(residual, r) is True]
        as_of = str(res.get("resolved") or "") or None
        if as_of:
            _AS_OF[self._uri(spec) or ""] = as_of
        return Page(rows=rows, total=len(rows), next=None, as_of=as_of)

    def count(self, spec: LayoutSpec, *, predicate: Predicate | None, budget: Any = None) -> int | None:
        """Rows matching ``predicate`` (one read of ``soma_joinid`` only); None when it does not compile."""
        value_filter, residual = self.fmt.compile(predicate)
        if residual is not None:
            return None
        version, path, _res = self._read_target(spec)
        with _open(version, path, self._module(spec), self._config(spec)) as frame:
            kwargs: dict[str, Any] = {"column_names": ["soma_joinid"]}
            if value_filter:
                kwargs["value_filter"] = value_filter
            return count_rows(frame.read(**kwargs))

    def columns(self, spec: LayoutSpec, *, predicate: Predicate | None, columns: list[str]) -> dict[str, list[Any]]:
        """``{column: values}`` of the rows matching ``predicate`` (one read; the residual is applied)."""
        value_filter, residual = self.fmt.compile(predicate)
        version, path, _res = self._read_target(spec)
        need = list(dict.fromkeys(list(columns) + (sorted(_predicate_columns(residual)) if residual else [])))
        with _open(version, path, self._module(spec), self._config(spec)) as frame:
            got = read_columns(frame, value_filter, need)
        if residual is None:
            return {c: got[c] for c in columns}
        n = len(next(iter(got.values()))) if got else 0
        keep = [i for i in range(n) if evaluate(residual, {c: got[c][i] for c in need}) is True]
        return {c: [got[c][i] for i in keep] for c in columns}

    def describe_read(self, spec: LayoutSpec, predicate: Predicate | None) -> Mapping[str, Any]:
        """What a read would send (the compiled filter and its residual), without opening the store."""
        value_filter, residual = self.fmt.compile(predicate)
        return {"uri": self._uri(spec), "value_filter": value_filter, "residual": residual is not None}

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        return LayoutCases(tree="none", path=None, format="soma")

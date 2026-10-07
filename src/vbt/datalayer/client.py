"""``vbt.datalayer.client``: the in-process read API (§10.6, rev 2, phase 2). No pyarrow at import.

Harness modules and notebooks read reference data through the same verbs agents call, as Python
functions::

    from vbt.datalayer import client
    res = client.find("open_targets.known_drug", where={"targetId": "PCSK9"}, limit=5)
    res.rows, res.header["total"], res.prov                 # rows, honest totals, the provenance id
    frame = client.read_frame("open_targets.drug_molecule", columns=["id", "drugType"], root=ot_path)
    cohort = client.open_matrix("zenodo_vbt.ibd_cohorts", fragment="GSE12251")

Every call goes through the data child's verbs, so it gets the same identifier resolution (``not_found``,
``ambiguous`` and ``invalid_argument`` are raised as :class:`~vbt.datalayer.errors.GatewayError`), the
same limits, and a readiness check of the tables it reads (``not_ready``); and every call writes a
``vbt.dataprov/1`` record and returns its id (``dp_...``), so an artifact built from it can cite its
inputs: ``run.register_artifact(..., derived_from=[res.prov])``.

**Backends.** ``child`` starts the data child (the same launcher and memory limit as every MCP server)
through an :class:`~vbt.tools.mcp_bridge.MCPBridge`, or reuses one already started
(:func:`use_bridge`, e.g. the runtime's); ``inprocess`` runs the verbs in this process (the data child's
code, without its process boundary: for notebooks and harness readers that already hold the data in
pandas). ``auto`` (default; ``VBT_DATA_CLIENT``) reuses a registered bridge, else runs in process when
pyarrow is importable, else starts the child. ``off`` serves no verb; :meth:`DataClient.read_frame` and
:meth:`DataClient.open_matrix` then read the declared files directly, unguarded, and their provenance
records say so (``served_by: unguarded``).

**Backed reads.** :meth:`DataClient.read_frame` has the data child write the selected rows as one Parquet
file (``_materialize``) that the caller reads with pandas, so large reads never travel as JSON and the
frame has the stored types. :meth:`DataClient.open_matrix` checks a matrix fragment ready, fingerprints
it and returns its path for a backed open (AnnData ``backed="r"``).

**Who may read what.** Inside an agent's process (``VBT_AGENT`` set by the runtime) the agent's
``expose.withhold_from`` and ``expose.native: false`` apply, exactly as for ``mcp__data__*``; harness
code (no agent) may read ``native: false`` tables such as the Case 1 answer key.

**Provenance.** Records go to ``<run>/logs/data_provenance/client/<id>.json`` for the run named by
``run_dir=`` or ``VBT_RUN_DIR``, else to a standalone ``log_dir`` (``VBT_DATA_CLIENT_LOG``, default
``<data.cache_dir>/provenance/client``).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ErrorKind, GatewayError
from .record import DataProvenance, RequestInfo, ResolutionRecord, ResultInfo, SourceInfo, TableInfo

__all__ = [
    "DataClient", "Result", "MatrixHandle", "get_client", "configure", "use_bridge", "close", "find", "lookup",
    "search", "aggregate", "similar", "resolve", "describe", "vocab", "members", "neighbors", "read_frame",
    "open_matrix", "BACKENDS", "ENV_BACKEND", "ENV_LOG",
]

BACKENDS = ("auto", "inprocess", "child", "off")
ENV_BACKEND = "VBT_DATA_CLIENT"
ENV_LOG = "VBT_DATA_CLIENT_LOG"
_CLIENT_DIR = "client"


@dataclass
class Result:
    """One client call: the rows, the ``_vbt`` header, the provenance id and the whole answer."""

    rows: list[Any]
    header: dict[str, Any]
    prov: str
    body: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str | None:
        return self.header.get("status")

    def frame(self) -> Any:
        """The rows as a pandas DataFrame (pandas imported on use)."""
        import pandas as pd

        return pd.DataFrame(self.rows)


@dataclass
class MatrixHandle:
    """A matrix fragment checked ready and fingerprinted: open it with :meth:`anndata` (or your own reader)."""

    table: str
    path: str
    fingerprint: str
    prov: str
    fragment_key: str | None = None
    shape: tuple[int, int] | None = None
    guarded: bool = True

    def anndata(self, backed: str | None = None) -> Any:
        import anndata as ad

        return ad.read_h5ad(self.path, backed=backed) if backed else ad.read_h5ad(self.path)


def _error(env: Mapping[str, Any]) -> GatewayError:
    """The :class:`GatewayError` of a §12.1 envelope from the data child."""
    fixed = {"status", "contract", "kind", "subkind", "tool", "argument", "value", "message", "citable",
             "retryable", "instruction"}
    payload = {k: v for k, v in env.items() if k not in fixed}
    try:
        kind = ErrorKind(str(env.get("kind")))
    except ValueError:
        kind = ErrorKind.source_error
    return GatewayError(kind, str(env.get("message") or kind.value), tool=env.get("tool"),
                        argument=env.get("argument"), value=env.get("value"), payload=payload,
                        subkind=env.get("subkind"))


def _resolution(arg: str, summary: str) -> ResolutionRecord:
    """A header resolution (``"PCSK9 -> ENSG00000169174 (label_exact:approvedSymbol)"``) as a record."""
    raw, sep, rest = str(summary).partition(" -> ")
    if not sep:
        return ResolutionRecord(arg=arg, raw=raw, canonical=raw, rule="exact")
    canonical, _, rule = rest.partition(" (")
    return ResolutionRecord(arg=arg, raw=raw, canonical=canonical, rule=rule.rstrip(")") or None)


# ---------------------------------------------------------------------------- backends


class _InProcess:
    """The data child's verbs run in this process."""

    name = "inprocess"

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._ctx: Any = None
        self._verbs: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def call(self, verb: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._ctx is None:
                from .service import ServiceContext
                from .service.verbs import load_verbs

                self._ctx = ServiceContext(self.settings)
                self._verbs = load_verbs()
            assert self._verbs is not None
            if verb not in self._verbs:
                raise GatewayError(ErrorKind.invalid_argument, f"the data child has no verb {verb!r}")
            return self._verbs[verb](self._ctx, dict(payload))

    def close(self) -> None:
        self._ctx = None


class _Off:
    """No data child: verbs fail; backed reads fall back to direct, unguarded reads."""

    name = "off"

    def call(self, verb: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        raise GatewayError(ErrorKind.service_unavailable, f"the data client is off ({ENV_BACKEND}=off): {verb} "
                           "cannot be served")

    def close(self) -> None:
        pass


class _Child:
    """The data child as an MCP server: a bridge of our own on a private event loop, or a reused one."""

    name = "child"

    def __init__(self, config: Mapping[str, Any], bridge: Any = None, loop: asyncio.AbstractEventLoop | None = None
                 ) -> None:
        self.config = dict(config)
        self.bridge = bridge
        self.loop = loop
        self._own = bridge is None
        self._thread: threading.Thread | None = None

    def _start(self) -> None:
        if self.bridge is not None:
            return
        from ..tools.mcp_bridge import MCPBridge, MCPServerConfig
        from .gateway import build_gateway

        gateway = build_gateway(self.config)
        specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
                 for s in gateway.extra_servers()]
        if not specs:
            raise GatewayError(ErrorKind.service_unavailable, "the data child script is missing in this checkout")
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="vbt-data-client", daemon=True)
        self._thread.start()
        bridge = MCPBridge(specs, options=self.config.get("mcp") or {}, gateway=gateway)
        gateway.bind_bridge(bridge)
        asyncio.run_coroutine_threadsafe(bridge.start(), self.loop).result()
        self.bridge = bridge

    def call(self, verb: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        from .gateway.service_client import DATA_SERVER, _text_of

        self._start()
        coro = self.bridge.call_raw(DATA_SERVER, verb, {"request": dict(payload)})
        if self.loop is not None and self.loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is self.loop:
                raise RuntimeError("the client's synchronous API cannot run on the bridge's own event loop")
            result = asyncio.run_coroutine_threadsafe(coro, self.loop).result()
        else:
            result = asyncio.run(coro)
        body = _text_of(result)
        if isinstance(body, (str, bytes)):
            try:
                body = json.loads(body)
            except ValueError:
                raise GatewayError(ErrorKind.service_unavailable, f"the data child answered {verb} with "
                                   f"non-JSON text: {str(body)[:200]}") from None
        if not isinstance(body, dict):
            raise GatewayError(ErrorKind.service_unavailable, f"the data child answered {verb} with {type(body)}")
        return body

    def close(self) -> None:
        if not self._own or self.bridge is None or self.loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.bridge.aclose(), self.loop).result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.bridge = None


# ---------------------------------------------------------------------------- the client


class DataClient:
    """Verbs of the data child as Python calls, with resolution, readiness, limits and provenance."""

    def __init__(self, config: Mapping[str, Any] | None = None, *, backend: str | None = None,
                 run_dir: str | os.PathLike[str] | None = None, log_dir: str | os.PathLike[str] | None = None,
                 agent: str | None = None, bridge: Any = None, loop: asyncio.AbstractEventLoop | None = None) -> None:
        from .settings import DataSettings

        if config is None:
            from ..config import load_config

            config = load_config()
        self.config = dict(config)
        self.settings = DataSettings.from_config(self.config)
        choice = (backend or os.environ.get(ENV_BACKEND) or "auto").strip().lower()
        if choice not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)} (got {choice!r})")
        if choice == "auto":
            choice = "child" if bridge is not None else (
                "inprocess" if importlib.util.find_spec("pyarrow") is not None else "child")
        self.backend: Any = (_Child(self.config, bridge, loop) if choice == "child" else
                             _Off() if choice == "off" else _InProcess(self.settings))
        self.agent = agent if agent is not None else (os.environ.get("VBT_AGENT") or None)
        run = run_dir or os.environ.get("VBT_RUN_DIR")
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        elif run:
            self.log_dir = Path(run) / self.settings.provenance.dir / _CLIENT_DIR
        else:
            self.log_dir = Path(os.environ.get(ENV_LOG) or Path(self.settings.cache_dir) / "provenance" / _CLIENT_DIR)
        self.records: list[str] = []

    # -- plumbing -------------------------------------------------------------------------------

    def call(self, verb: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """One verb of the data child; a typed error envelope is raised as :class:`GatewayError`."""
        body = dict(payload)
        if self.agent:
            body["agent"] = self.agent
        out = self.backend.call(verb, body)
        if isinstance(out, Mapping) and out.get("status") == "tool_error":
            raise _error(out)
        return dict(out)

    def _record(self, verb: str, payload: Mapping[str, Any], out: Mapping[str, Any], *,
                tables: Sequence[tuple[str, str | None]] = (), output_sha256: str | None = None,
                served_by: str = "derived") -> str:
        hdr = dict(out.get("_vbt") or {})
        src = str(hdr.get("source") or "")
        name, _, release = src.partition("@")
        resolutions = [_resolution(k, v) for k, v in (hdr.get("resolved") or {}).items()]
        rec = DataProvenance(tool=f"client.{verb}", server="data", mode="off" if served_by == "unguarded" else "enforce",
                             served_by=served_by,
                             source=SourceInfo(name=name or None, release=release or None),
                             tables=[TableInfo(name=t, fingerprint=fp, access="client") for t, fp in tables],
                             request=RequestInfo(args_raw=dict(payload), args_sent=dict(payload),
                                                 resolutions=resolutions),
                             result=ResultInfo(status=hdr.get("status"), returned=hdr.get("returned"),
                                               total=hdr.get("total"), total_method=hdr.get("total_method"),
                                               coverage=hdr.get("coverage"),
                                               coverage_statement=hdr.get("coverage_statement"),
                                               truncated=hdr.get("truncated"), key_columns=list(hdr.get("key") or []),
                                               output_sha256=output_sha256),
                             evidence_nature={"caveat": hdr["evidence"]} if hdr.get("evidence") else None)
        keys = [[r.get(k) for k in rec.result.key_columns] for r in out.get("rows") or [] if isinstance(r, Mapping)]
        if keys:
            rec.set_row_keys(keys, int(self.settings.provenance.row_keys_max))
        rec.finalize()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{rec.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(rec.to_json(), encoding="utf-8")
        os.replace(tmp, path)
        self.records.append(str(rec.id))
        return str(rec.id)

    def _rows(self, verb: str, payload: dict[str, Any], table: str | None = None) -> Result:
        out = self.call(verb, payload)
        hdr = dict(out.get("_vbt") or {})
        fp = (hdr.get("extra") or {}).get("fingerprint") if isinstance(hdr.get("extra"), Mapping) else None
        tables = [(t, fp) for t in (hdr.get("tables") or ([table] if table else []))]
        prov = self._record(verb, payload, out, tables=tables)
        rows = out.get("rows")
        return Result(rows=list(rows) if isinstance(rows, list) else [], header=hdr, prov=prov, body=out)

    # -- verbs ----------------------------------------------------------------------------------

    def find(self, table: str, where: Mapping[str, Any] | None = None, *, rank_by: Any = None, limit: int = 50,
             distinct: Sequence[str] | None = None, group_by: Sequence[str] | None = None,
             columns: Sequence[str] | None = None) -> Result:
        payload = {"table": table, "where": dict(where) if where else None, "rank_by": rank_by, "limit": limit,
                   "distinct": list(distinct or []), "group_by": list(group_by or []), "columns": list(columns or [])}
        return self._rows("find", payload, table)

    def lookup(self, table: str, key: Mapping[str, Any]) -> Result:
        return self._rows("lookup", {"table": table, "key": dict(key)}, table)

    def search(self, table: str, text: str, *, limit: int = 20) -> Result:
        return self._rows("search", {"table": table, "text": text, "limit": limit}, table)

    def aggregate(self, table: str, group_by: Sequence[str], *, measure: str | None = None, how: str = "count",
                  min_n: int = 1, where: Mapping[str, Any] | None = None, limit: int = 100) -> Result:
        payload = {"table": table, "group_by": list(group_by), "measure": measure, "how": how, "min_n": min_n,
                   "where": dict(where) if where else None, "limit": limit}
        return self._rows("aggregate", payload, table)

    def similar(self, table: str, anchor: str, *, where: Mapping[str, Any] | None = None, top_k: int = 10) -> Result:
        return self._rows("similar", {"table": table, "anchor": anchor, "where": dict(where) if where else None,
                                      "top_k": top_k}, table)

    def resolve(self, id_type: str, values: Sequence[str]) -> Result:
        return self._rows("resolve", {"id_type": id_type, "values": list(values)})

    def describe(self, source: str, table: str | None = None) -> Result:
        return self._rows("describe", {"source": source, "table": table})

    def vocab(self, table: str, column: str, *, max_values: int | None = None) -> Result:
        return self._rows("vocab", {"table": table, "column": column, "max_values": max_values}, table)

    def members(self, table: str, set_id: str) -> Result:
        return self._rows("members", {"table": table, "set_id": set_id}, table)

    def neighbors(self, table: str, node: str, *, where: Mapping[str, Any] | None = None, limit: int = 50) -> Result:
        return self._rows("neighbors", {"table": table, "node": node, "where": dict(where) if where else None,
                                        "limit": limit}, table)

    # -- backed reads ---------------------------------------------------------------------------

    def read_frame(self, table: str, columns: Sequence[str] | None = None, *, where: Mapping[str, Any] | None = None,
                   root: str | os.PathLike[str] | None = None) -> Any:
        """The rows of ``table`` (optionally filtered and projected) as a pandas DataFrame with the stored
        types. ``root`` reads the table's source from that directory (the readers' own input paths). The
        frame's ``attrs["vbt_prov"]`` holds the provenance id."""
        import pandas as pd

        if isinstance(self.backend, _Off):
            return self._direct_frame(table, columns, where, root)
        payload: dict[str, Any] = {"table": table, "columns": list(columns or []),
                                   "where": dict(where) if where else None,
                                   "root": str(root) if root is not None else None, "native": self.agent is not None,
                                   "out_dir": str(Path(self.settings.cache_dir))}
        out = self.call("_materialize", payload)
        path = Path(str(out["path"]))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        prov = self._record("read_frame", payload, out, tables=[(str(out["table"]), out.get("fingerprint"))],
                            output_sha256=sha)
        frame = pd.read_parquet(path)
        frame.attrs["vbt_prov"] = prov
        return frame

    def _direct_frame(self, table: str, columns: Sequence[str] | None, where: Mapping[str, Any] | None,
                      root: str | os.PathLike[str] | None) -> Any:
        import pandas as pd

        from .catalog import build_catalog

        if where:
            raise GatewayError(ErrorKind.service_unavailable, "where needs the data child (the client is off)")
        t = build_catalog(self.settings).table(table)
        base = Path(str(root)) if root is not None else Path(str(t.descriptor.root or ""))
        path = base / str(t.physical_spec.path or "")
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
        payload = {"table": table, "columns": list(columns or []), "root": str(base)}
        out = {"_vbt": {"status": "ok" if len(frame) else "empty", "returned": len(frame), "total": len(frame),
                        "notes": [f"read directly from {path} ({ENV_BACKEND}=off): no readiness check"]}}
        frame.attrs["vbt_prov"] = self._record("read_frame", payload, out, tables=[(table, None)],
                                               served_by="unguarded")
        return frame

    def open_matrix(self, table: str | None = None, fragment: str | None = None, *,
                    path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None
                    ) -> MatrixHandle:
        """One matrix fragment checked ready and fingerprinted. With ``path`` (a file the caller already
        names), the matrix table whose layout lists that file is found and read from the file's root; a file
        no descriptor declares is opened unguarded, and its provenance record says so."""
        if path is not None:
            found = self._matrix_for_path(Path(path))
            if found is None or isinstance(self.backend, _Off):
                return self._unguarded(Path(path))
            table, root, fragment = found
        if not table:
            raise GatewayError(ErrorKind.invalid_argument, "name a matrix table or a path", argument="table")
        payload = {"table": table, "fragment": fragment, "root": str(root) if root is not None else None,
                   "native": self.agent is not None}
        out = self.call("_matrix", payload)
        frags = list(out.get("fragments") or [])
        if len(frags) != 1:
            raise GatewayError(ErrorKind.invalid_argument, f"{table}: name one fragment of "
                               f"{[f.get('fragment_key') for f in frags]}", argument="fragment",
                               payload={"valid_values": [f.get("fragment_key") for f in frags]})
        f = frags[0]
        prov = self._record("open_matrix", payload, out, tables=[(table, out.get("fingerprint"))])
        shape = tuple(f["shape"]) if f.get("shape") else None
        return MatrixHandle(table=str(table), path=str(f["path"]), fingerprint=str(out.get("fingerprint")),
                            prov=prov, fragment_key=f.get("fragment_key"), shape=shape)  # type: ignore[arg-type]

    def _matrix_for_path(self, path: Path) -> tuple[str, Path, str] | None:
        """``(table, source root, fragment)`` of the matrix table whose path pattern matches ``path``."""
        from fnmatch import fnmatch

        from .catalog import build_catalog

        catalog = build_catalog(self.settings)
        resolved = path.resolve()
        for ref in catalog.table_refs():
            t = catalog.table(ref)
            if t.kind != "matrix" or not t.physical_spec.path:
                continue
            pattern = Path(str(t.physical_spec.path))
            depth = len(pattern.parts)
            if len(resolved.parts) <= depth:
                continue
            tail = Path(*resolved.parts[-depth:])
            if fnmatch(str(tail), str(pattern)):
                return str(ref), Path(*resolved.parts[:-depth]), resolved.name
        return None

    def _unguarded(self, path: Path) -> MatrixHandle:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        out = {"_vbt": {"status": "ok", "notes": [f"{path} is declared by no descriptor: opened unguarded"]}}
        prov = self._record("open_matrix", {"path": str(path)}, out, tables=[(str(path), f"sha256:{sha}")],
                            output_sha256=sha, served_by="unguarded")
        return MatrixHandle(table="", path=str(path), fingerprint=f"sha256:{sha}", prov=prov, guarded=False)

    def close(self) -> None:
        self.backend.close()


# ---------------------------------------------------------------------------- module-level API

_default: DataClient | None = None
_default_lock = threading.Lock()
_bridge: tuple[Any, Any] | None = None


def use_bridge(bridge: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Reuse a started bridge's data child for the module-level client (the runtime registers its own)."""
    global _bridge, _default
    with _default_lock:
        _bridge = (bridge, loop)
        _default = None


def configure(config: Mapping[str, Any] | None = None, **kwargs: Any) -> DataClient:
    """Replace the module-level client (``backend``, ``run_dir``, ``log_dir``, ``agent``)."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
        if _bridge is not None and "bridge" not in kwargs:
            kwargs.setdefault("bridge", _bridge[0])
            kwargs.setdefault("loop", _bridge[1])
        _default = DataClient(config, **kwargs)
        return _default


def get_client() -> DataClient:
    return _default if _default is not None else configure()


def close() -> None:
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
        _default = None


def find(table: str, where: Mapping[str, Any] | None = None, **kwargs: Any) -> Result:
    return get_client().find(table, where, **kwargs)


def lookup(table: str, key: Mapping[str, Any]) -> Result:
    return get_client().lookup(table, key)


def search(table: str, text: str, **kwargs: Any) -> Result:
    return get_client().search(table, text, **kwargs)


def aggregate(table: str, group_by: Sequence[str], **kwargs: Any) -> Result:
    return get_client().aggregate(table, group_by, **kwargs)


def similar(table: str, anchor: str, **kwargs: Any) -> Result:
    return get_client().similar(table, anchor, **kwargs)


def resolve(id_type: str, values: Sequence[str]) -> Result:
    return get_client().resolve(id_type, values)


def describe(source: str, table: str | None = None) -> Result:
    return get_client().describe(source, table)


def vocab(table: str, column: str, **kwargs: Any) -> Result:
    return get_client().vocab(table, column, **kwargs)


def members(table: str, set_id: str) -> Result:
    return get_client().members(table, set_id)


def neighbors(table: str, node: str, **kwargs: Any) -> Result:
    return get_client().neighbors(table, node, **kwargs)


def read_frame(table: str, columns: Sequence[str] | None = None, **kwargs: Any) -> Any:
    return get_client().read_frame(table, columns, **kwargs)


def open_matrix(table: str | None = None, fragment: str | None = None, **kwargs: Any) -> MatrixHandle:
    return get_client().open_matrix(table, fragment, **kwargs)

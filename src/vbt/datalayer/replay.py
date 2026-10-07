"""``vbt ds replay <run> <tool_use_id>``: re-execute a recorded data call on current data (§15.1, §16, F22).

The call's ``vbt.dataprov/1`` record (``<run>/logs/data_provenance/<tool_use_id>.json``) holds the
arguments the gateway sent. :func:`replay` sends them again through a temporary gateway (and a
temporary bridge, or the data child's verbs in this process) and compares the new record with the
recorded one:

* ``row_keys_sha256``: the canonical row keys (§6.3), so float32 keys, null parts and list parts
  compare as the witness compares them;
* ``output_rows_sha256``: the returned rows. When the hashes differ, the recorded rows (from the
  trace or the output spill) are compared with the new ones value by value, with floats equal
  within :data:`REL_TOL`; enrichment rows compare as ``(set, k, K, p, q)`` only (set id, overlap,
  set size, p-value, FDR), so a column added beside them is not a mismatch;
* ``computed`` (similarity scores) and ``statistics`` (family size, universe), with the same tolerance;
* table fingerprints and, when the record names them, the fingerprints of **only the partitions
  read** (``partitions_read``): a refresh of a partition the call never read is listed under
  ``ignored``, never as drift;
* live record versions (``KeySpec.version``): records whose version is unchanged are compared; a
  record the source changed since the call is reported as ``source_updated``, not as a mismatch.

The result is ``match``, ``replay_mismatch``, ``source_updated`` or ``unavailable`` (the call could
not run here: a data outage, a missing server, no record). ``vbt verify --data`` replays every cited
call the same way. Identifier arguments are sent in their resolved form (``args_sent``) and every
other argument as the agent gave it, so a limit the gateway inflated is not inflated twice.

No pyarrow at import: the in-process backend imports the data child's verbs when it runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

__all__ = [
    "REL_TOL", "ABS_TOL", "ENRICHMENT_FIELDS", "STATUSES", "ReplayCheck", "ReplayResult", "load_record",
    "record_path", "recorded_output", "replay_args", "record_versions", "partitions_from_scope", "stamp_partitions",
    "rows_of", "rows_close", "compare", "GatewayReplayer", "InProcessDataBridge", "open_replayer", "replay",
    "replay_async", "replay_run", "data_calls", "format_result",
]

#: Floats (p-values, scores, similarities) compare equal within this relative tolerance.
REL_TOL = 1e-9
ABS_TOL = 1e-12
#: The fields an enrichment row is compared on: (set, k, K, p, q).
ENRICHMENT_FIELDS = ("set_id", "overlap", "set_size", "pvalue", "fdr")
STATUSES = ("match", "replay_mismatch", "source_updated", "unavailable")
PROV_DIR = "logs/data_provenance"
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
#: Error kinds that mean "this call cannot be replayed here", not "the answer changed".
_UNAVAILABLE_KINDS = frozenset({"not_ready", "service_unavailable", "server_crashed", "oom", "too_large",
                                "source_error", "quarantined"})


def _safe(name: str) -> str:
    return _SAFE_RE.sub("_", str(name))[:160] or "x"


# ---------------------------------------------------------------------------- results


@dataclass
class ReplayCheck:
    name: str
    ok: bool | None                                    # None: not comparable (not recorded)
    recorded: Any = None
    current: Any = None
    detail: str = ""


@dataclass
class ReplayResult:
    tool_use_id: str
    tool: str | None = None
    status: str = "unavailable"
    checks: list[ReplayCheck] = field(default_factory=list)
    drift: list[dict[str, Any]] = field(default_factory=list)       # fingerprints of data the call read that changed
    ignored: list[dict[str, Any]] = field(default_factory=list)     # changes in data the call did not read
    source_updated: list[str] = field(default_factory=list)         # live record keys with a new version
    notes: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    args: dict[str, Any] | None = None
    record: dict[str, Any] | None = None                            # the replay's own record

    @property
    def ok(self) -> bool:
        return self.status == "match"

    @property
    def mismatches(self) -> list[ReplayCheck]:
        return [c for c in self.checks if c.ok is False]

    def to_dict(self, *, with_record: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_record:
            d.pop("record", None)
        return d


def format_result(r: ReplayResult) -> list[str]:
    """Text lines of one replay."""
    lines = [f"replay {r.tool_use_id} {r.tool or ''}: {r.status}".rstrip()]
    for c in r.checks:
        mark = {True: "ok", False: "MISMATCH", None: "-"}[c.ok]
        line = f"  {c.name:<20} {mark}"
        if c.ok is False:
            line += f" (recorded {str(c.recorded)[:40]}, now {str(c.current)[:40]})"
        if c.detail:
            line += f" {c.detail}"
        lines.append(line)
    for d in r.drift:
        where = d["table"] + (f"/{d['partition']}" if d.get("partition") else "")
        lines.append(f"  data changed: {where}: {d.get('recorded')} -> {d.get('current')}")
    for d in r.ignored:
        lines.append(f"  ignored: {d['table']}: {d.get('detail', '')}")
    if r.source_updated:
        lines.append(f"  source_updated: {', '.join(r.source_updated[:10])}"
                     + (f" (+{len(r.source_updated) - 10} more)" if len(r.source_updated) > 10 else ""))
    if r.error:
        lines.append(f"  error: {r.error.get('kind')}: {str(r.error.get('message'))[:300]}")
    lines.extend(f"  note: {n}" for n in r.notes)
    return lines


# ---------------------------------------------------------------------------- the recorded call


def record_path(run_dir: str | Path, tool_use_id: str) -> Path:
    return Path(run_dir) / PROV_DIR / f"{_safe(tool_use_id)}.json"


def load_record(run_dir: str | Path, tool_use_id: str) -> dict[str, Any]:
    """The call's ``vbt.dataprov/1`` record; :class:`FileNotFoundError` when the run has none."""
    path = record_path(run_dir, tool_use_id)
    if not path.is_file():
        # a custom data.provenance.dir, or a sanitised name that differs: match on the stored id
        for p in sorted((Path(run_dir) / PROV_DIR).glob("*.json")):
            with contextlib.suppress(OSError, ValueError):
                rec = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(rec, dict) and rec.get("tool_use_id") == tool_use_id:
                    return rec
        raise FileNotFoundError(f"{path} does not exist: the call has no data provenance record")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rec, dict) or not str(rec.get("schema") or "").startswith("vbt.dataprov/"):
        raise ValueError(f"{path} is not a vbt.dataprov record")
    return rec


def data_calls(run_dir: str | Path) -> list[str]:
    """The tool_use ids of every call with a provenance record (client records excluded)."""
    out = []
    for p in sorted((Path(run_dir) / PROV_DIR).glob("*.json")):
        with contextlib.suppress(OSError, ValueError):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and rec.get("tool_use_id"):
                out.append(str(rec["tool_use_id"]))
    return out


def recorded_output(run_dir: str | Path, tool_use_id: str) -> Any:
    """The recorded result payload of a call (the output spill, else the untruncated trace output),
    parsed as JSON; None when it was truncated, is not JSON or is missing."""
    from ..audit.provenance import read_trace

    events, _bad = read_trace(Path(run_dir) / "logs" / "trace.jsonl")
    end = next((e for e in reversed(events) if e.get("type") == "tool_end" and e.get("tool_use_id") == tool_use_id
                and "output" in e), None)
    if end is None:
        return None
    text: str | None = None
    spill = end.get("output_path")
    if spill:
        with contextlib.suppress(OSError):
            text = (Path(run_dir) / str(spill)).read_text(encoding="utf-8")
    if text is None:
        out = end.get("output")
        if not isinstance(out, str):
            return out if isinstance(out, (dict, list)) else None
        chars = end.get("output_chars")
        if isinstance(chars, int) and chars > len(out):
            return None                                # truncated in the trace and not spilled
        text = out
    try:
        return json.loads(text)
    except ValueError:
        return None


def replay_args(record: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments of the replay: the agent's arguments, with every resolved identifier argument in
    the form the gateway sent (so resolution drift does not hide data drift) and nothing inflated."""
    req = record.get("request") if isinstance(record.get("request"), Mapping) else {}
    raw = req.get("args_raw") if isinstance(req.get("args_raw"), Mapping) else None
    sent = req.get("args_sent") if isinstance(req.get("args_sent"), Mapping) else {}
    if raw is None:
        return dict(sent)
    args = dict(raw)
    for r in req.get("resolutions") or []:
        name = r.get("arg") if isinstance(r, Mapping) else None
        if name in args and name in sent:
            args[name] = sent[name]
        elif name in args and r.get("canonical") is not None:
            args[name] = r["canonical"]
    return args


def record_versions(record: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Live record versions ``{table: {key: version}}`` of a record (top level, ``result`` or the
    derived handlers' records)."""
    if not isinstance(record, Mapping):
        return {}
    for holder in (record, record.get("result"), record.get("derived")):
        if isinstance(holder, Mapping) and isinstance(holder.get("record_versions"), Mapping):
            return {str(t): {str(k): str(v) for k, v in (m or {}).items()}
                    for t, m in holder["record_versions"].items() if isinstance(m, Mapping)}
    return {}


def _partition_parts(name: str) -> dict[str, str]:
    out = {}
    for part in str(name).split("/"):
        k, sep, v = part.partition("=")
        if sep:
            out[k] = v
    return out


def partitions_from_scope(record: Mapping[str, Any], available: Iterable[str]) -> list[str] | None:
    """The partitions (``col=value[/col=value]``) a call read, from the scope it fixed: the partitions
    whose values agree with every fixed scope dimension that is a partition column. None when the scope
    fixes no partition column (the call read every partition)."""
    req = record.get("request") if isinstance(record.get("request"), Mapping) else {}
    scope = req.get("scope") if isinstance(req.get("scope"), Mapping) else {}
    names = list(available)
    cols = {k for n in names for k in _partition_parts(n)}
    fixed = {str(k).split(".")[-1]: v for k, v in scope.items() if str(k).split(".")[-1] in cols}
    if not fixed:
        return None

    def agrees(name: str) -> bool:
        parts = _partition_parts(name)
        for col, value in fixed.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            if col in parts and parts[col] not in {str(v) for v in values}:
                return False
        return True

    return sorted(n for n in names if agrees(n))


def stamp_partitions(record: dict[str, Any], stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Fill ``partitions_read`` and ``partition_fingerprints`` of a record's tables from ``_stats``
    results (``{"source.table" | "table": {partition_fingerprints, ...}}``): the partitions the scope
    fixed, else every partition. Tables already stamped are kept. Returns ``record``."""
    source = (record.get("source") or {}).get("name") if isinstance(record.get("source"), Mapping) else None
    for t in record.get("tables") or []:
        if not isinstance(t, dict) or t.get("partition_fingerprints"):
            continue
        st = stats.get(f"{source}.{t.get('name')}") or stats.get(str(t.get("name"))) or {}
        pf = dict(st.get("partition_fingerprints") or {})
        if not pf:
            continue
        read = t.get("partitions_read") or partitions_from_scope(record, pf) or sorted(pf)
        t["partitions_read"] = list(read)
        t["partition_fingerprints"] = {p: pf[p] for p in read if p in pf}
    return record


# ---------------------------------------------------------------------------- comparison


def rows_of(obj: Any, paths: Sequence[str] = ()) -> list[Any] | None:
    """The result rows of a payload: at the binding's row paths, else the first list of objects."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except ValueError:
            return None
    if obj is None:
        return None
    if paths:
        from .gateway.fields import extract_rows

        with contextlib.suppress(Exception):
            found = extract_rows(obj, list(paths))
            rows = [r for _path, part in found for r in part]
            if found:
                return rows
    if isinstance(obj, list):
        return obj
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k != "_vbt" and isinstance(v, list) and (not v or isinstance(v[0], Mapping)):
                return v
    return None


def _project(row: Any) -> Any:
    if isinstance(row, Mapping) and all(f in row for f in ("set_id", "pvalue")):
        return {f: row.get(f) for f in ENRICHMENT_FIELDS}
    if isinstance(row, Mapping):
        return {k: v for k, v in row.items() if k != "_vbt"}
    return row


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def rows_close(recorded: Sequence[Any], current: Sequence[Any]) -> bool:
    """Same rows in the same order, floats equal within tolerance (enrichment rows as (set, k, K, p, q))."""
    return len(recorded) == len(current) and all(_close(_project(a), _project(b)) for a, b in zip(recorded, current))


def _belongs(row: Any, keys: set[str]) -> bool:
    """A row of a live record whose version changed: any top-level scalar equals one of its keys."""
    if not isinstance(row, Mapping):
        return False
    return any(isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) in keys for v in row.values())


def _qualified(record: Mapping[str, Any], name: str) -> str:
    source = (record.get("source") or {}).get("name") if isinstance(record.get("source"), Mapping) else None
    return f"{source}.{name}" if source else str(name)


def compare(recorded: Mapping[str, Any], current: Mapping[str, Any], *, recorded_rows: Sequence[Any] | None = None,
            current_rows: Sequence[Any] | None = None, stats: Mapping[str, Mapping[str, Any]] | None = None,
            tool_use_id: str | None = None) -> ReplayResult:
    """Compare a recorded ``vbt.dataprov/1`` record with the replay's (see the module docstring).

    ``recorded_rows``/``current_rows`` enable the value-by-value comparison when hashes differ;
    ``stats`` (``_stats`` results by ``source.table``) gives current partition fingerprints when the
    replay's own record has none."""
    res = ReplayResult(tool_use_id=str(tool_use_id or recorded.get("tool_use_id") or ""), tool=recorded.get("tool"),
                       record=dict(current))
    rr = recorded.get("result") if isinstance(recorded.get("result"), Mapping) else {}
    cr = current.get("result") if isinstance(current.get("result"), Mapping) else {}
    stats = stats or {}

    # live record versions first: rows of records the source changed are not compared
    rv, cv = record_versions(recorded), record_versions(current)
    updated = sorted({k for t, m in rv.items() for k, v in m.items() if k in cv.get(t, {}) and cv[t][k] != v})
    res.source_updated = updated
    have_rows = recorded_rows is not None and current_rows is not None

    def rows_check() -> tuple[bool | None, str]:
        if not have_rows:
            return None, ""
        a, b = list(recorded_rows or []), list(current_rows or [])
        if updated:
            keys = set(updated)
            a = [r for r in a if not _belongs(r, keys)]
            b = [r for r in b if not _belongs(r, keys)]
        if rows_close(a, b):
            return True, ("equal within tolerance" if not updated
                          else f"rows of records with unchanged versions are equal ({len(updated)} updated)")
        return False, f"{len(a)} recorded row(s) vs {len(b)} now"

    def add(name: str, a: Any, b: Any, *, tolerant: bool = False) -> None:
        if a is None:
            res.checks.append(ReplayCheck(name, None, a, b, "not recorded"))
            return
        if a == b:
            res.checks.append(ReplayCheck(name, True, a, b))
            return
        if tolerant:
            ok, detail = rows_check()
            if ok is not None:
                res.checks.append(ReplayCheck(name, ok, a, b, detail))
                return
        res.checks.append(ReplayCheck(name, False, a, b))

    add("status", rr.get("status"), cr.get("status"))
    if rr.get("total_method") != "unknown":
        add("total", rr.get("total"), cr.get("total"), tolerant=bool(updated))
    add("row_keys_sha256", rr.get("row_keys_sha256"), cr.get("row_keys_sha256"), tolerant=bool(updated))
    add("output_rows_sha256", rr.get("output_rows_sha256"), cr.get("output_rows_sha256"), tolerant=True)
    rc, cc = rr.get("computed"), cr.get("computed")
    if rc:
        add("computed", (rc or {}).get("values_sha256"), (cc or {}).get("values_sha256"), tolerant=True)
    rs, cs = rr.get("statistics"), cr.get("statistics")
    if rs:
        ok = _close({k: v for k, v in rs.items() if k != "universe"}, {k: v for k, v in (cs or {}).items()
                                                                       if k != "universe"}) \
            and _close((rs.get("universe") or {}).get("n"), ((cs or {}).get("universe") or {}).get("n"))
        res.checks.append(ReplayCheck("statistics", ok, rs, cs))

    # fingerprints of the data the call read
    cur_tables = {str(t.get("name")): t for t in current.get("tables") or [] if isinstance(t, Mapping)}
    for t in recorded.get("tables") or []:
        if not isinstance(t, Mapping):
            continue
        name = str(t.get("name"))
        qualified = _qualified(recorded, name)
        ct = cur_tables.get(name) or {}
        st = stats.get(qualified) or stats.get(name) or {}
        cur_fp = ct.get("fingerprint") or st.get("fingerprint")
        rec_pf = t.get("partition_fingerprints") or {}
        if rec_pf:
            cur_pf = ct.get("partition_fingerprints") or st.get("partition_fingerprints") or {}
            changed = [p for p, fp in sorted(rec_pf.items()) if cur_pf.get(p) != fp]
            for p in changed:
                res.drift.append({"table": qualified, "partition": p, "recorded": rec_pf[p], "current": cur_pf.get(p)})
            if not changed and t.get("fingerprint") and cur_fp and cur_fp != t.get("fingerprint"):
                res.ignored.append({"table": qualified, "recorded": t.get("fingerprint"), "current": cur_fp,
                                    "detail": "only partitions the call did not read changed "
                                              f"(read: {', '.join(sorted(rec_pf))})"})
        elif t.get("fingerprint") and cur_fp and cur_fp != t.get("fingerprint"):
            entry = {"table": qualified, "recorded": t.get("fingerprint"), "current": cur_fp}
            pinned = partitions_from_scope(recorded, st.get("partition_fingerprints") or {})
            if pinned:
                entry["detail"] = (f"the call read {', '.join(pinned)} but its record has no per-partition "
                                   "fingerprints, so a change elsewhere in the table counts")
            res.drift.append(entry)
        elif t.get("fingerprint") and cur_fp is None:
            res.notes.append(f"{qualified}: no current fingerprint (the table was not read by the replay)")

    mismatch = bool(res.mismatches)
    if updated:
        res.status = "replay_mismatch" if mismatch else "source_updated"
    else:
        res.status = "replay_mismatch" if mismatch else "match"
    if res.drift and res.status == "match":
        res.notes.append("the data changed since the call, but the answer did not")
    return res


# ---------------------------------------------------------------------------- executing the call


class InProcessDataBridge:
    """What the gateway needs from ``MCPBridge``, with the data child's verbs run in this process.
    Only the ``data`` server exists: a call routed to an upstream server raises."""

    def __init__(self, ctx: Any) -> None:
        from .service.verbs import load_verbs

        self.ctx = ctx
        self.verbs = load_verbs()
        self.failures: dict[str, str] = {}

    async def call_raw(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        from .gateway.service_client import DATA_SERVER

        if server != DATA_SERVER:
            from ..tools.base import ToolFailure

            raise ToolFailure(f"{server}.{tool} is answered by its upstream server, which the in-process replay "
                              "does not start (use --backend bridge)")
        if tool not in self.verbs:
            from ..tools.base import ToolFailure

            raise ToolFailure(f"the data child has no verb {tool!r}")
        payload = args.get("request") if isinstance(args.get("request"), Mapping) else args
        out = await asyncio.to_thread(self.verbs[tool], self.ctx, dict(payload))
        return json.dumps(out, default=str)

    def status(self) -> dict[str, Any]:
        return {}

    async def recycle(self, server: str, wait_s: float = 30.0) -> bool:
        return True

    def _emit(self, kind: str, **data: Any) -> None:
        pass


def _raw_result(out: Any) -> Any:
    """The RawResult ``MCPBridge._convert(classify_only=True)`` builds for a JSON payload."""
    from ..tools.mcp_bridge import _lookup_miss, tool_result_error
    from .api import RawResult

    text = out if isinstance(out, str) else json.dumps(out, default=str)
    try:
        structured = json.loads(text)
    except ValueError:
        structured = None
    err = tool_result_error(text)
    if err:
        return RawResult(text, structured, None, "legacy_error", err)
    if _lookup_miss(text):
        return RawResult(text, structured, None, "empty_lookup")
    return RawResult(text, structured, None, "ok")


class GatewayReplayer:
    """Runs calls through a gateway: ``MCPBridge.call`` when the bridge carries the gateway, else
    ``prepare`` -> ``call_raw`` (upstream route) -> ``finish`` as ``MCPBridge.call`` does."""

    def __init__(self, gateway: Any, *, close: Callable[[], Awaitable[None]] | None = None) -> None:
        self.gateway = gateway
        self._close = close

    @property
    def bridge(self) -> Any:
        return self.gateway.bridge

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        bridge = self.bridge
        if getattr(bridge, "gateway", None) is self.gateway and callable(getattr(bridge, "call", None)):
            return await bridge.call(server, tool, dict(args))
        plan = await self.gateway.prepare(server, tool, dict(args), None)
        raw = None
        if plan.route == "upstream":
            raw = _raw_result(await bridge.call_raw(server, tool, dict(plan.args_sent)))
        return await self.gateway.finish(plan, raw)

    async def stats(self, tables: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not tables:
            return {}
        try:
            resp = await self.gateway.service.stats(list(tables))
        except Exception:  # noqa: BLE001 - fingerprints stay unknown
            return {}
        return {k: v.model_dump() if hasattr(v, "model_dump") else dict(v) for k, v in (resp.tables or {}).items()}

    def row_paths(self, server: str, tool: str) -> list[str]:
        try:
            b = self.gateway.catalog.contract(server, tool).binding
            return list(b.result.row_paths) if b is not None and b.result is not None else []
        except Exception:  # noqa: BLE001 - unbound tools: the heuristic finds the rows
            return []

    async def aclose(self) -> None:
        if self._close is not None:
            await self._close()


def _have_arrow() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in ("pyarrow",))


async def open_replayer(config: Mapping[str, Any], server: str, *, backend: str = "auto",
                        served_by: str | None = None) -> GatewayReplayer:
    """A gateway to replay one call on: ``inprocess`` (the data child's verbs in this process; derived
    and native calls), ``bridge`` (a temporary MCPBridge with the call's server and the data child), or
    ``auto`` (in-process for derived or data-child calls when pyarrow is importable)."""
    from .gateway import build_gateway
    from .gateway.service_client import DATA_SERVER

    config = dict(config or {})
    if backend == "auto":
        derived = served_by in ("derived",) or server == DATA_SERVER
        backend = "inprocess" if derived and _have_arrow() else "bridge"
    tmp = Path(tempfile.mkdtemp(prefix="vbt-replay-"))
    (tmp / "mcp").mkdir()
    run = {"dir": str(tmp), "run_id": "replay", "mcp_output_dir": str(tmp / "mcp")}
    gateway = build_gateway(config, run)
    if backend == "inprocess":
        from .service import ServiceContext

        ctx = ServiceContext(gateway.settings, catalog=gateway.catalog, registry=gateway.registry)
        gateway.bind_bridge(InProcessDataBridge(ctx))

        async def close_inprocess() -> None:
            shutil.rmtree(tmp, ignore_errors=True)

        return GatewayReplayer(gateway, close=close_inprocess)
    if backend != "bridge":
        raise ValueError(f"unknown replay backend {backend!r} (auto, inprocess, bridge)")
    from ..config import base_tool_env
    from ..tools.mcp_bridge import MCPBridge, MCPServerConfig

    raw = [dict(s) for s in ((config.get("mcp_servers") or {}).get("servers") or [])
           if isinstance(s, Mapping) and s.get("name") == server and s.get("enabled", True)]
    if server != DATA_SERVER and not raw:
        shutil.rmtree(tmp, ignore_errors=True)
        raise LookupError(f"server {server!r} is not configured in mcp_servers")
    names = {s.get("name") for s in raw}
    raw += [s for s in gateway.extra_servers() or [] if s.get("name") not in names]
    specs = [MCPServerConfig(**{k: v for k, v in s.items() if k in MCPServerConfig.__dataclass_fields__})
             for s in raw]
    extra = base_tool_env(config)
    extra.update({"VBT_RUN_DIR": str(tmp), "MCP_OUTPUT_DIR": str(tmp / "mcp")})
    bridge = MCPBridge(specs, extra_env=extra, log_dir=tmp / "logs", options=config.get("mcp") or {},
                       gateway=gateway)
    if getattr(gateway, "bridge", None) is not bridge:
        gateway.bind_bridge(bridge)

    async def close_bridge() -> None:
        with contextlib.suppress(Exception):
            await gateway.aclose()
        with contextlib.suppress(Exception):
            await bridge.aclose()
        shutil.rmtree(tmp, ignore_errors=True)

    try:
        await bridge.start({server, DATA_SERVER})
        if server not in getattr(bridge, "sessions", {server: None}):
            raise RuntimeError(f"server {server!r} did not start: {bridge.failures.get(server)}")
    except BaseException:
        await close_bridge()
        raise
    return GatewayReplayer(gateway, close=close_bridge)


def _split(tool: str, server: str | None) -> tuple[str, str]:
    name = str(tool)
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    return str(server or ""), name


def _error_of(exc: BaseException) -> dict[str, Any]:
    kind = getattr(exc, "kind", None)
    return {"kind": str(kind) if kind is not None else type(exc).__name__,
            "message": str(getattr(exc, "message", None) or exc)[:1000]}


async def replay_async(run_dir: str | Path, tool_use_id: str, config: Mapping[str, Any] | None = None, *,
                       replayer: Any = None, backend: str = "auto") -> ReplayResult:
    """Replay one recorded call (see the module docstring). ``replayer`` (with ``call``, ``stats``,
    ``row_paths``) runs it; by default a temporary one from :func:`open_replayer`, closed afterwards."""
    run_dir = Path(run_dir)
    try:
        record = load_record(run_dir, tool_use_id)
    except (OSError, ValueError) as exc:
        return ReplayResult(tool_use_id, status="unavailable", error={"kind": "no_record", "message": str(exc)})
    server, tool = _split(str(record.get("tool") or ""), record.get("server"))
    args = replay_args(record)
    own = replayer is None
    try:
        if own:
            replayer = await open_replayer(config or {}, server, backend=backend, served_by=record.get("served_by"))
    except Exception as exc:  # noqa: BLE001 - the call cannot run here
        return ReplayResult(tool_use_id, tool=record.get("tool"), status="unavailable", args=args,
                            error=_error_of(exc))
    try:
        try:
            result = await replayer.call(server, tool, args)
        except Exception as exc:  # noqa: BLE001 - classified below
            err = _error_of(exc)
            kind = err["kind"]
            status = "unavailable" if kind in _UNAVAILABLE_KINDS or not hasattr(exc, "kind") else "replay_mismatch"
            out = ReplayResult(tool_use_id, tool=record.get("tool"), status=status, args=args, error=err)
            if status == "replay_mismatch":
                out.checks.append(ReplayCheck("status", False, (record.get("result") or {}).get("status"), kind,
                                              "the call now fails"))
            return out
        prov = getattr(result, "provenance", None)
        current = prov.to_dict() if callable(getattr(prov, "to_dict", None)) else dict(prov or {})
        if not current:
            return ReplayResult(tool_use_id, tool=record.get("tool"), status="unavailable", args=args,
                                error={"kind": "no_record", "message": "the replay produced no provenance record"})
        paths = replayer.row_paths(server, tool) if callable(getattr(replayer, "row_paths", None)) else []
        recorded_rows = rows_of(recorded_output(run_dir, tool_use_id), paths)
        current_rows = rows_of(getattr(result, "obj", None), paths)
        tables = sorted({_qualified(record, str(t.get("name"))) for t in record.get("tables") or []
                         if isinstance(t, Mapping) and t.get("name")})
        stats = await replayer.stats(tables) if callable(getattr(replayer, "stats", None)) else {}
        out = compare(record, current, recorded_rows=recorded_rows, current_rows=current_rows, stats=stats,
                      tool_use_id=tool_use_id)
        out.args = args
        if recorded_rows is None:
            out.notes.append("the recorded rows are not in the trace (truncated output without a spill): "
                             "rows compare by hash only")
        return out
    finally:
        if own and replayer is not None:
            with contextlib.suppress(Exception):
                await replayer.aclose()


def replay(run_dir: str | Path, tool_use_id: str, config: Mapping[str, Any] | None = None, *,
           replayer: Any = None, backend: str = "auto") -> ReplayResult:
    """:func:`replay_async` from synchronous code."""
    return asyncio.run(replay_async(run_dir, tool_use_id, config, replayer=replayer, backend=backend))


async def replay_run(run_dir: str | Path, tool_use_ids: Iterable[str], config: Mapping[str, Any] | None = None, *,
                     replayer: Any = None, backend: str = "auto") -> list[ReplayResult]:
    """Replay several calls of one run, one after the other."""
    return [await replay_async(run_dir, t, config, replayer=replayer, backend=backend) for t in tool_use_ids]

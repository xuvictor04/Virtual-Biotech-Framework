"""The gateway end to end (§11.3-§11.6, §11.10) against a fake data child and fake upstream tools.

The fake data child (:class:`FakeService`) answers the hidden verbs from in-memory rows with the
predicate oracle (``predicate.evaluate``), so witness counts, top-k, distinct values and derived
serving behave like the real reader on small tables. Upstream tools are plain functions of the
arguments the gateway sends. The helpers here are imported by the other gateway test modules.
"""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from vbt.datalayer.api import CallPlan, RawResult
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.gateway import DataGateway
from vbt.datalayer.gateway.service_client import ServiceClient, ServiceError
from vbt.datalayer.ipc import (
    BuildIndexResponse,
    CheckResponse,
    ServeResponse,
    StatsResponse,
    TableCheckModel,
    TableStatsModel,
    ColumnStatsModel,
    VocabResponse,
    WitnessResponse,
)
from vbt.datalayer.plugins.registry import discover
from vbt.datalayer.plugins.statistics import order_rows
from vbt.datalayer.predicate import RankKey, evaluate, from_json
from vbt.datalayer.resolve import Entry, IndexStore
from vbt.datalayer.result import DataResult
from vbt.datalayer.rowkey import canonical
from vbt.datalayer.settings import DataSettings
from vbt.tools.mcp_bridge import _lookup_miss, tool_result_error

# --------------------------------------------------------------------------- helpers (shared)

REGISTRY = discover(entry_points=False)


def f32(x: float) -> float:
    """``x`` rounded to float32 (what a float32 column holds)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def get(row: Mapping[str, Any], path: str) -> Any:
    cur: Any = row
    for p in path.split("."):
        cur = cur.get(p) if isinstance(cur, Mapping) else None
    return cur


class FakeService(ServiceClient):
    """An in-memory data child: ``tables`` maps ``source.table`` to rows."""

    def __init__(self, tables: Mapping[str, list[dict[str, Any]]], *, store: IndexStore,
                 index_rows: Mapping[str, list[Entry]] | None = None, storage: Mapping[str, Mapping[str, str]] | None = None,
                 check: Mapping[str, TableCheckModel] | None = None, fail: set[str] | None = None,
                 witness_hook: Callable[[Any, WitnessResponse], WitnessResponse] | None = None,
                 keys: Mapping[str, list[str]] | None = None,
                 codes: Mapping[str, Mapping[str, list[Any]]] | None = None) -> None:
        super().__init__(None)
        self.codes = {k: dict(v) for k, v in (codes or {}).items()}
        self.tables = {k: [dict(r) for r in v] for k, v in tables.items()}
        self.store = store
        self.index_rows = dict(index_rows or {})
        self.storage = {k: dict(v) for k, v in (storage or {}).items()}
        self.check_models = dict(check or {})
        self.fail = set(fail or ())
        self.witness_hook = witness_hook
        self.keys = dict(keys or {})
        self.log: list[tuple[str, Any]] = []

    @property
    def available(self) -> bool:
        return True

    async def call(self, verb: str, request: Any) -> Any:
        self.log.append((verb, request))
        self.calls += 1
        if verb in self.fail or "*" in self.fail:
            raise ServiceError(f"fake child: {verb} failed", verb=verb)
        return getattr(self, "_" + verb.lstrip("_"))(request)

    def verbs(self, verb: str) -> list[Any]:
        return [r for v, r in self.log if v == verb]

    # -- verbs -----------------------------------------------------------

    def _match(self, table: str, predicate: Any, params: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
        pred = from_json(predicate) if predicate else None
        hit, unknown = [], []
        codes = self.codes.get(table, {})
        for r in self.tables.get(table, []):
            view = {k: (None if k in codes and v in codes[k] else v) for k, v in r.items()}
            v = evaluate(pred, view, params)
            if v is True:
                hit.append(r)
            elif v is None:
                unknown.append(r)
        return hit, unknown

    def _order(self, rows: list[dict], order: list[Any], key: list[str]) -> list[dict]:
        keys = []
        for o in order:
            rk = RankKey(column=o.column, direction=o.direction, nulls=o.nulls, statistic=o.statistic)
            plugin = REGISTRY.find("statistic", o.statistic or "numeric") or REGISTRY.find("statistic", "numeric")
            keys.append((rk, plugin, None))
        return list(order_rows(rows, keys, tie_key=lambda r: canonical([get(r, k) for k in key])))

    def _witness(self, req: Any) -> WitnessResponse:
        hit, unknown = self._match(req.table, req.predicate, req.params)
        key = list(req.key) or self.keys.get(req.table, [])
        ordered = self._order(hit, list(req.order), key)
        topk: Any = []
        if req.k:
            topk = [[get(r, c) for c in key] for r in ordered[: req.k]]
        key_set = [[get(r, c) for c in key] for r in hit] if (req.key_set_max or 0) >= len(hit) and key else None
        distinct = {c: sorted({get(r, c) for r in hit}, key=lambda v: (v is None, str(v))) for c in req.distinct}
        excluded = {}
        codes = self.codes.get(req.table, {})
        for c in req.unknown_columns:
            n = sum(1 for r in unknown if get(r, c) is None or get(r, c) in codes.get(c, []))
            if n:
                excluded[c] = n
        if unknown:
            excluded["_rows"] = len(unknown)
        counts = {}
        for g, cols in req.grains.items():
            cols = cols if isinstance(cols, list) else cols.get("columns", [])
            counts[g] = len({canonical([get(r, c) for c in cols]) for r in hit})
        resp = WitnessResponse(total=len(hit), total_method="scan", topk=topk, key_set=key_set, distinct=distinct,
                               excluded_unknown=excluded, unknown_total=len(unknown), distinct_counts=counts)
        return self.witness_hook(req, resp) if self.witness_hook else resp

    def _serve(self, req: Any) -> ServeResponse:
        hit, unknown = self._match(req.table, req.predicate, req.params)
        key = self.keys.get(req.table, [])
        ordered = self._order(hit, list(req.order), key)
        total = len(ordered)

        def shape(r: dict) -> dict:
            out = {k: v for k, v in r.items() if not req.columns or k in req.columns}
            return {req.rename.get(k, k): v for k, v in out.items()}

        if req.split:
            col = req.split["by_sign"]
            parts: dict[str, list] = {"+": [], "-": []}
            for r in ordered:
                v = r.get(col)
                if v is None:
                    continue
                bucket = parts["+" if v > 0 else "-"]
                if req.limit is None or len(bucket) < req.limit:
                    bucket.append(shape(r))
            rows: Any = parts
            truncated = sum(len(p) for p in parts.values()) < total
        else:
            cut = ordered if req.limit is None else ordered[: req.limit]
            rows = [shape(r) for r in cut]
            truncated = len(cut) < total
        sections = {}
        for name, sec in req.sections.items():
            srows = [r for r in self.tables.get(sec["table"], [])
                     if all(get(r, c) == v for c, v in (sec.get("key") or {}).items())]
            sections[name] = srows[0] if sec.get("single") and srows else srows
        return ServeResponse(rows=rows, total=total, truncated=truncated, key_columns=key, sections=sections,
                             excluded_unknown={"_rows": len(unknown)} if unknown else {})

    def _vocab(self, req: Any) -> VocabResponse:
        values = sorted({get(r, req.column) for r in self.tables.get(req.table, [])
                         if get(r, req.column) is not None}, key=str)
        st = self.storage.get(req.table, {}).get(req.column)
        return VocabResponse(values=values, rendered=[str(v) for v in values], storage_type=st, complete=True)

    def _stats(self, req: Any) -> StatsResponse:
        out = {}
        for t in req.tables:
            cols = {c: ColumnStatsModel(storage_type=s) for c, s in self.storage.get(t, {}).items()}
            out[t] = TableStatsModel(fingerprint=f"fp1:{t}", columns=cols, rows=len(self.tables.get(t, [])))
        return StatsResponse(tables=out)

    def _check(self, req: Any) -> CheckResponse:
        tables = req.tables or list(self.tables)
        out = {t: self.check_models.get(t, TableCheckModel(status="ready", fingerprint=f"fp1:{t}")) for t in tables}
        return CheckResponse(tables=out, hash_randomization=0)

    def _census_count(self, req: Any) -> Any:
        from vbt.datalayer.ipc import CensusCountResponse

        return CensusCountResponse(table=req.table, reason="no Census in the fake child")

    def _release(self, req: Any) -> Any:
        from vbt.datalayer.ipc import ReleaseResponse

        return ReleaseResponse(table=req.table, reason="no release in the fake child")

    def _build_index(self, req: Any) -> BuildIndexResponse:
        rows = self.index_rows.get(f"{req.source}:{req.id_type}")
        if rows is None:
            raise ServiceError(f"no index for {req.source}:{req.id_type}", verb="_build_index")
        path = self.store.write_sidecar(req.source, "fp1:test", req.id_type, rows)
        return BuildIndexResponse(path=str(path), rows=len(rows), fingerprint="fp1:test")


class FakeBridge:
    """What the gateway needs from ``MCPBridge``: ``call_raw`` (re-calls, listings) and status."""

    def __init__(self, upstream: Callable[[str, dict[str, Any]], Any] | None = None) -> None:
        self.upstream = upstream
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.failures: dict[str, str] = {}

    async def call_raw(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        self.calls.append((server, tool, dict(args)))
        out = self.upstream(tool, dict(args)) if self.upstream else {}
        return out if isinstance(out, str) else json.dumps(out)

    def status(self) -> dict[str, Any]:
        return {}

    async def recycle(self, server: str, wait_s: float = 30.0) -> bool:
        return True

    def _emit(self, kind: str, **data: Any) -> None:
        self.events.append((kind, data))


def raw_of(obj: Any, *, is_error: bool = False) -> RawResult:
    """The RawResult the bridge would build for ``obj`` (``classify_only``)."""
    text = obj if isinstance(obj, str) else json.dumps(obj)
    if is_error:
        return RawResult(text, None, None, "is_error", text)
    err = tool_result_error(text)
    if err:
        return RawResult(text, None, None, "legacy_error", err)
    if _lookup_miss(text):
        return RawResult(text, None, None, "empty_lookup")
    return RawResult(text, None, None, "ok")


def make_gateway(tmp_path: Path, descriptors: list[dict[str, Any]], overlays: list[dict[str, Any]],
                 tables: Mapping[str, list[dict[str, Any]]], *, index_rows: Mapping[str, list[Entry]] | None = None,
                 data: Mapping[str, Any] | None = None, storage: Mapping[str, Mapping[str, str]] | None = None,
                 check: Mapping[str, TableCheckModel] | None = None, fail: set[str] | None = None,
                 upstream: Callable[[str, dict[str, Any]], Any] | None = None,
                 witness_hook: Any = None, keys: Mapping[str, list[str]] | None = None,
                 generic: list[dict[str, Any]] | None = None,
                 codes: Mapping[str, Mapping[str, list[Any]]] | None = None) -> DataGateway:
    settings = DataSettings.from_dict({"cache_dir": str(tmp_path / "cache"), **dict(data or {})},
                                      project_root=tmp_path)
    descs = {d["source"]: SourceDescriptor.model_validate(copy.deepcopy(d)) for d in descriptors}
    ovs = {o["server"]: Overlay.model_validate(copy.deepcopy(o)) for o in overlays}
    gens = [Overlay.model_validate(copy.deepcopy(g)) for g in (generic or [])]
    catalog = Catalog(descs, ovs, gens, registry=REGISTRY)
    store = IndexStore(settings.cache_dir)
    service = FakeService(tables, store=store, index_rows=index_rows, storage=storage, check=check, fail=fail,
                          witness_hook=witness_hook, keys=keys, codes=codes)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    gw = DataGateway(settings, catalog, REGISTRY, service=service, index_store=store,
                     run={"mcp_output_dir": str(out)})
    gw.bind_bridge(FakeBridge(upstream))
    return gw


async def call(gw: DataGateway, server: str, tool: str, args: dict[str, Any],
               upstream: Callable[[dict[str, Any]], Any] | None = None, *,
               is_error: bool = False) -> tuple[CallPlan, DataResult]:
    """prepare -> (upstream when routed) -> finish, like ``MCPBridge.call``."""
    plan = await gw.prepare(server, tool, args, None)
    raw = None
    if plan.route == "upstream":
        assert upstream is not None, "the call was routed upstream"
        gw.bridge.upstream = lambda t, a: upstream(a)
        raw = raw_of(upstream(dict(plan.args_sent)), is_error=is_error)
    return plan, await gw.finish(plan, raw)


def hdr(result: Any) -> dict[str, Any]:
    return result.header if isinstance(result, DataResult) else {}


def gene_key(text: str) -> str:
    return REGISTRY.get("identifier", "ensembl_gene").label_key(text)


# --------------------------------------------------------------------------- the test world

PCSK9, TP53, BRCA1 = "ENSG00000169174", "ENSG00000141510", "ENSG00000012048"
GENES = {PCSK9: "PCSK9", TP53: "TP53", BRCA1: "BRCA1"}

OT: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "open_targets", "title": "Open Targets",
    "release": {"expect": "25.09", "from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
    "id_types": {
        "ensembl_gene": {"plugin": "ensembl_gene", "universe": "target.id", "resolve_via": ["target.approvedSymbol"]},
        "hgnc_symbol": {"plugin": "hgnc_symbol", "label_of": "ensembl_gene"},
        "chembl_molecule": {"plugin": "chembl_molecule", "universe": "drug_molecule.id"},
    },
    "tables": {
        "target": {"kind": "entity", "grain": "one target", "key": {"columns": ["id"]},
                   "columns": {"id": {"role": "identifier", "id_type": "ensembl_gene", "self": True},
                               "approvedSymbol": {"role": "label", "of": "id"}}},
        "drug_molecule": {"kind": "entity", "grain": "one molecule", "key": {"columns": ["id"]},
                          "columns": {"id": {"role": "identifier", "id_type": "chembl_molecule", "self": True}}},
        "known_drug": {
            "kind": "fact", "grain": "one (drug, target) record", "key": {"columns": ["drugId", "targetId"]},
            "grains": {"drug": ["drugId"]},
            "coverage": {"statement": "ChEMBL-curated drugs only.", "absence_means": "absent"},
            "columns": {"drugId": {"role": "identifier", "id_type": "chembl_molecule", "ref": "drug_molecule.id"},
                        "targetId": {"role": "identifier", "id_type": "ensembl_gene", "ref": "target.id"},
                        "phase": {"role": "measure", "statistic": "numeric", "missing_values": [-1]},
                        "status": {"role": "category", "vocab": ["Completed", "Recruiting"]}}},
        "interaction": {
            "kind": "edges", "grain": "one edge from one source",
            "key": {"columns": ["sourceDatabase", "targetA", "targetB"]},
            "columns": {"sourceDatabase": {"role": "category", "scope": {"kind": "partition", "pooling": "group"}},
                        "targetA": {"role": "identifier", "id_type": "ensembl_gene"},
                        "targetB": {"role": "identifier", "id_type": "ensembl_gene"},
                        "score": {"role": "measure", "statistic": "numeric",
                                  "comparable_within": ["sourceDatabase"]}}},
    },
}

KNOWN_DRUG = [
    {"drugId": f"CHEMBL{i}", "targetId": PCSK9, "phase": p, "status": "Completed"}
    for i, p in zip(range(1, 9), [4, 1, 3, -1, 2, 4, None, 3])
] + [{"drugId": "CHEMBL99", "targetId": TP53, "phase": 2, "status": "Recruiting"}]

INTERACTION = [
    {"sourceDatabase": "intact", "targetA": PCSK9, "targetB": TP53, "score": 0.9},
    {"sourceDatabase": "intact", "targetA": PCSK9, "targetB": BRCA1, "score": 0.5},
    {"sourceDatabase": "string", "targetA": PCSK9, "targetB": TP53, "score": 950},
    {"sourceDatabase": "string", "targetA": BRCA1, "targetB": PCSK9, "score": 400},
]

DRUG_OVERLAY: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "drug", "sources": ["open_targets"],
    "tools": {
        "search_known_drugs": {
            "reads": {"open_targets.known_drug": {"access": "full_table"}},
            "args": {"target_id": {"binds": "open_targets.known_drug.targetId", "accepts": ["ensembl_gene", "hgnc_symbol"]},
                     "min_phase": {"binds": "open_targets.known_drug.phase", "op": "ge", "min": 0, "max": 4},
                     "limit": {"role": "limit", "min": 1, "max": 200}},
            "require_any": [["target_id"]],
            "result": {"rows": "$.drugs", "order": [{"column": "phase", "direction": "desc"}],
                       "count_fields": ["$.count"], "arg_echo": {"limit": "$.limit"}},
            "defects": [{"id": "OT-DRUG-003", "what": "head before sort"}],
        },
        "search_known_drugs_derived": {
            "reads": {"open_targets.known_drug": {"access": "full_table"}},
            "args": {"target_id": {"binds": "open_targets.known_drug.targetId", "accepts": ["ensembl_gene", "hgnc_symbol"]},
                     "limit": {"role": "limit", "min": 1, "max": 200}},
            "result": {"rows": "$.drugs", "order": [{"column": "phase", "direction": "desc"}]},
            "serve": "derived",
            "derived": {"verb": "find", "table": "open_targets.known_drug", "envelope": {"success": True}},
        },
        "get_target": {
            "reads": {"open_targets.target": {"access": "full_table"}},
            "args": {"target_id": {"binds": "open_targets.target.id", "accepts": ["ensembl_gene", "hgnc_symbol"]}},
            "result": {"kind": "record", "rows": "$", "echo": {"target_id": {"path": "$.id"}}},
        },
        "old_tool": {"reads": {}, "args": {}, "serve": "block",
                     "block": {"reason": "wrong answers on this data", "alternatives": ["mcp__drug__search_known_drugs"]}},
        "interactions": {
            "reads": {"open_targets.interaction": {"access": "full_table"}},
            "args": {"target_id": {"binds_any": ["open_targets.interaction.targetA", "open_targets.interaction.targetB"],
                                   "accepts": ["ensembl_gene", "hgnc_symbol"]},
                     "limit": {"role": "limit", "min": 1, "max": 100, "limit_mode": "refuse"}},
            "result": {"rows": "$.rows", "order": [{"column": "score", "direction": "desc"}]},
        },
    },
}


def index_rows() -> dict[str, list[Entry]]:
    g = gene_key
    rows = []
    for gid, sym in GENES.items():
        rows += [Entry(g(gid), gid, "exact", sym), Entry(g(sym), gid, "label_exact:approvedSymbol", sym)]
    c = REGISTRY.get("identifier", "chembl_molecule").label_key
    drugs = [Entry(c(f"CHEMBL{i}"), f"CHEMBL{i}", "exact") for i in list(range(1, 9)) + [99]]
    return {"open_targets:ensembl_gene": rows, "open_targets:chembl_molecule": drugs}


TABLES = {"open_targets.known_drug": KNOWN_DRUG, "open_targets.interaction": INTERACTION,
          "open_targets.target": [{"id": g, "approvedSymbol": s} for g, s in GENES.items()],
          "open_targets.drug_molecule": [{"id": f"CHEMBL{i}"} for i in list(range(1, 9)) + [99]]}
KEYS = {"open_targets.known_drug": ["drugId", "targetId"], "open_targets.target": ["id"],
        "open_targets.interaction": ["sourceDatabase", "targetA", "targetB"]}


GENERIC_OVERLAY: dict[str, Any] = {"schema": "vbt.overlay/1", "server": "*",
                                   "generic": {"not_found_when": ["$.status == 404"]}}


def world(tmp_path: Path, **kw: Any) -> DataGateway:
    tables = kw.pop("tables", TABLES)
    return make_gateway(tmp_path, [OT], [kw.pop("overlay", DRUG_OVERLAY)], tables, index_rows=index_rows(),
                        keys=KEYS, codes={"open_targets.known_drug": {"phase": [-1]}},
                        generic=kw.pop("generic", [GENERIC_OVERLAY]), **kw)


def head_first(n_default: int = 20) -> Callable[[dict[str, Any]], Any]:
    """Upstream ``search_known_drugs``: head(limit) BEFORE sorting by phase (OT-DRUG-003)."""
    def tool(args: dict[str, Any]) -> Any:
        rows = [r for r in KNOWN_DRUG if r["targetId"] == args.get("target_id")]
        rows = rows[: int(args.get("limit", n_default))]
        rows = sorted(rows, key=lambda r: -(r["phase"] if r["phase"] is not None else -9))
        return {"success": True, "count": len(rows), "limit": args.get("limit"), "drugs": rows}
    return tool


def sorted_first(args: dict[str, Any]) -> Any:
    rows = [r for r in KNOWN_DRUG if r["targetId"] == args.get("target_id")]
    rows = sorted(rows, key=lambda r: (-(r["phase"] if r["phase"] is not None else -9), r["drugId"]))
    rows = rows[: int(args.get("limit", 20))]
    return {"success": True, "count": len(rows), "limit": args.get("limit"), "drugs": rows}


# --------------------------------------------------------------------------- resolution and existence


async def test_resolution_rule_recorded_and_symbol_sent_as_ensembl(tmp_path):
    gw = world(tmp_path)
    seen: dict[str, Any] = {}

    def tool(args):
        seen.update(args)
        return sorted_first(args)

    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20}, tool)
    assert seen["target_id"] == PCSK9
    h = hdr(res)
    assert "label_exact:approvedSymbol" in h["resolved"]["target_id"]
    rec = plan.resolutions[0]
    assert rec["canonical"] == PCSK9 and rec["rule"] == "label_exact:approvedSymbol"
    assert rec["matched_id_type"].endswith("hgnc_symbol") and rec["canonical_id_type"].endswith("ensembl_gene")
    assert res.provenance.request.resolutions[0].rule == "label_exact:approvedSymbol"


async def test_unknown_identifier_is_not_found_with_suggestions(tmp_path):
    gw = world(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK99"}, sorted_first)
    assert e.value.kind == ErrorKind.not_found
    env = e.value.envelope()
    assert env["argument"] == "target_id" and env["value"] == "PCSK99"
    assert any(s["label"] == "PCSK9" for s in env["suggestions"])


async def test_wrong_kind_is_invalid_argument(tmp_path):
    gw = world(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "CHEMBL25"}, sorted_first)
    assert e.value.kind in (ErrorKind.invalid_argument, ErrorKind.not_found)


async def test_existence_bound_one_witness_count(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["by_drug"] = {
        "reads": {"open_targets.known_drug": {"access": "full_table"}},
        "args": {"drug_id": {"binds": "open_targets.known_drug.drugId", "accepts": ["chembl_molecule"],
                             "existence": "bound"}},
        "result": {"rows": "$.drugs"}}
    tables = dict(TABLES)
    tables["open_targets.known_drug"] = KNOWN_DRUG + [{"drugId": "CHEMBL777", "targetId": TP53, "phase": 1}]
    gw = world(tmp_path, overlay=ov, tables=tables)
    plan, res = await call(gw, "drug", "by_drug", {"drug_id": "CHEMBL777"},
                           lambda a: {"drugs": [r for r in tables["open_targets.known_drug"]
                                                if r["drugId"] == a["drug_id"]]})
    assert plan.existence["drug_id"] == "exists"
    assert any("not in the identity universe" in n for n in hdr(res)["notes"])
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "by_drug", {"drug_id": "CHEMBL778"}, lambda a: {"drugs": []})
    assert e.value.kind == ErrorKind.not_found


async def test_existence_unknown_gives_empty_unverified(tmp_path):
    ot = copy.deepcopy(OT)
    ot["id_types"]["chembl_molecule"]["index"] = "remote"       # resolved by the data child
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["by_drug"] = {"reads": {"open_targets.known_drug": {"access": "full_table"}},
                              "args": {"drug_id": {"binds": "open_targets.known_drug.drugId",
                                                   "accepts": ["chembl_molecule"]}},
                              "result": {"rows": "$.drugs"}}
    gw = make_gateway(tmp_path, [ot], [ov], TABLES, index_rows=index_rows(), keys=KEYS, fail={"_resolve_remote"})
    plan, res = await call(gw, "drug", "by_drug", {"drug_id": "CHEMBL555"}, lambda a: {"drugs": []})
    assert plan.existence["drug_id"] == "unknown"
    assert hdr(res)["status"] == "empty_unverified" and hdr(res)["cite"] == "not citable"
    assert any("could not be decided" in n for n in hdr(res)["notes"])


# --------------------------------------------------------------------------- witness, inflation, ranking


async def test_limit_inflated_by_total_plus_unknown_and_cut_after_sort(tmp_path):
    gw = world(tmp_path)
    sent: dict[str, Any] = {}

    def tool(args):
        sent.update(args)
        return head_first()(args)

    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 3}, tool)
    w = gw.service.verbs("_witness")[0]
    assert w.k == 3
    # 8 matching rows (2 with an unknown phase, ranked last): limit 3 -> 8
    assert sent["limit"] == 8
    rows = res.obj["drugs"]
    assert [r["phase"] for r in rows] == [4, 4, 3]
    h = hdr(res)
    assert h["total"] == 8 and h["returned"] == 3 and h["truncated"] is True and h["status"] == "partial"
    assert "verified" in h["order"] and h["total_method"] == "witness_scan"
    assert res.obj["limit"] == 3                      # arg_echo restored
    assert res.obj["count"] == 3                      # count_fields recomputed
    assert "limit_inflation 3->8" in res.provenance.result.transforms
    assert h["grains"]["drug"] == {"returned": 3, "total": 8}
    # with a threshold, unknown phases never pass and are counted
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "min_phase": 0, "limit": 3},
                           tool)
    assert sent["limit"] == 8 and hdr(res)["total"] == 6 and hdr(res)["excluded_unknown"]["phase"] == 2


async def test_no_inflation_with_output_path(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["args"]["out"] = {"role": "output_path"}
    ov["tools"]["search_known_drugs"]["result"]["order_source"] = "upstream_full_sort"
    gw = world(tmp_path, overlay=ov)
    sent: dict[str, Any] = {}
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": PCSK9, "limit": 3, "out": "x.csv"},
                           lambda a: sent.update(a) or sorted_first(a))
    assert sent["limit"] == 3 and hdr(res)["total"] == 8 and hdr(res)["status"] == "partial"


async def test_no_inflation_with_output_path_or_witness_false(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["witness"] = False
    gw = world(tmp_path, overlay=ov)
    sent: dict[str, Any] = {}
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 3},
                           lambda a: sent.update(a) or sorted_first(a))
    assert sent["limit"] == 3
    assert hdr(res)["total_method"] == "unknown"
    assert not gw.service.verbs("_witness")


async def test_topk_compared_when_not_inflated_and_tool_defect_on_mismatch(tmp_path):
    gw = world(tmp_path, data={"witness": {"max_inflate_rows": 2}})
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 3}, sorted_first)
    assert hdr(res)["order"].endswith("(verified)")
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 3}, head_first())
    assert e.value.kind == ErrorKind.tool_defect
    env = e.value.envelope()
    assert env["check"] == "W3" and env["witness"]["total"] == 8 and env["defect_ids"] == ["OT-DRUG-003"]


async def test_unranked_truncation_refused_when_topk_disabled(tmp_path):
    gw = world(tmp_path, data={"witness": {"max_inflate_rows": 2, "topk": False}})
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 3}, sorted_first)
    assert e.value.kind == ErrorKind.too_large and e.value.subkind == "unranked_truncation"


async def test_order_source_upstream_full_sort_is_accepted(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["result"]["order_source"] = "upstream_full_sort"
    gw = world(tmp_path, overlay=ov, data={"witness": {"max_inflate_rows": 2, "topk": False}})
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 3}, sorted_first)
    assert "upstream full sort" in hdr(res)["order"]


async def test_w1_empty_contradiction_repaired_when_derived_declared(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["derived"] = {"verb": "find", "table": "open_targets.known_drug"}
    gw = world(tmp_path, overlay=ov)
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20},
                           lambda a: {"success": True, "count": 0, "drugs": []})
    h = hdr(res)
    assert h["served_by"] == "repaired" and h["total"] == 8 and h["status"] in ("ok", "partial")
    assert len(res.obj["drugs"]) == 8


async def test_w1_without_derived_is_tool_defect(tmp_path):
    gw = world(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20},
                   lambda a: {"success": True, "count": 0, "drugs": []})
    assert e.value.kind == ErrorKind.tool_defect and e.value.envelope()["check"] == "W1"


async def test_w2_phantom_key_outside_witness(tmp_path):
    gw = world(tmp_path)

    def tool(args):
        out = sorted_first(args)
        out["drugs"].append({"drugId": "CHEMBL404", "targetId": PCSK9, "phase": 4})
        return out

    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 50}, tool)
    assert e.value.envelope()["check"] == "W2"


async def test_t4_rows_violating_bound_argument_removed_not_w2(tmp_path):
    gw = world(tmp_path)

    def tool(args):            # ignores min_phase
        return sorted_first(args)

    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "min_phase": 3, "limit": 50}, tool)
    rows = res.obj["drugs"]
    assert rows and all(r["phase"] >= 3 for r in rows)
    h = hdr(res)
    assert h["excluded"]["min_phase"] >= 1
    assert h["excluded_unknown"]["phase"] == 2


async def test_w4_echo_mismatch_and_w5_one_to_many(tmp_path):
    gw = world(tmp_path)
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "get_target", {"target_id": "PCSK9"}, lambda a: {"id": TP53, "approvedSymbol": "TP53"})
    assert e.value.envelope()["check"] == "W4"
    plan, res = await call(gw, "drug", "get_target", {"target_id": "PCSK9"},
                           lambda a: {"id": PCSK9, "approvedSymbol": "PCSK9"})
    assert hdr(res)["status"] == "ok" and res.obj["id"] == PCSK9


async def test_echo_redirect_accepted(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["get_target"]["result"]["echo"] = {"target_id": {"path": "$.id", "accept": ["canonical", "redirect"]}}
    gw = world(tmp_path, overlay=ov)
    plan, res = await call(gw, "drug", "get_target", {"target_id": "PCSK9"},
                           lambda a: {"id": TP53, "approvedSymbol": "TP53"})
    assert "alias_redirect" in hdr(res)["resolved"]["target_id"]


async def test_w6_short_page_recalled_once_with_corrected_limit(tmp_path):
    gw = world(tmp_path)
    calls = []

    def tool(args):                         # returns half a page the first time
        calls.append(dict(args))
        out = sorted_first(args)
        if len(calls) == 1:
            out["drugs"] = out["drugs"][:2]
        return out

    gw.bridge.upstream = lambda t, a: tool(a)
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20}, tool)
    assert len(calls) == 2
    assert len(res.obj["drugs"]) == 8 and hdr(res)["status"] == "ok"


async def test_per_group_ranking_and_incomparable_order(tmp_path):
    gw = world(tmp_path)

    def tool(args):
        rows = [r for r in INTERACTION if args["target_id"] in (r["targetA"], r["targetB"])]
        return {"rows": rows}

    with pytest.raises(GatewayError) as e:      # limit_mode: refuse
        await call(gw, "drug", "interactions", {"target_id": "PCSK9", "limit": 1}, tool)
    assert e.value.kind == ErrorKind.incomplete_key and e.value.subkind == "incomparable_order"
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["interactions"]["args"]["limit"]["limit_mode"] = "per_group"
    gw = world(tmp_path / "b", overlay=ov)
    plan, res = await call(gw, "drug", "interactions", {"target_id": "PCSK9", "limit": 1}, tool)
    rows = res.obj["rows"]
    assert sorted(r["sourceDatabase"] for r in rows) == ["intact", "string"]
    assert "within sourceDatabase" in hdr(res)["order"]


# --------------------------------------------------------------------------- derived, block, readiness, observe


async def test_derived_route_serves_from_data_child(tmp_path):
    gw = world(tmp_path)
    plan, res = await call(gw, "drug", "search_known_drugs_derived", {"target_id": "PCSK9", "limit": 2})
    assert plan.route == "derived"
    h = hdr(res)
    assert h["served_by"] == "derived" and h["total"] == 8 and h["returned"] == 2 and h["status"] == "partial"
    assert res.obj["success"] is True and [r["phase"] for r in res.obj["drugs"]] == [4, 4]


async def test_fidelity_profile_serves_derived_tools_pass_and_refuses_contradictions(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["derived"] = {"verb": "find", "table": "open_targets.known_drug"}
    gw = world(tmp_path, overlay=ov, data={"gateway": {"profile": "fidelity"}})
    plan, res = await call(gw, "drug", "search_known_drugs_derived", {"target_id": "PCSK9", "limit": 20},
                           sorted_first)
    assert plan.route == "upstream" and hdr(res)["served_by"] == "upstream"
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20},
                   lambda a: {"success": True, "count": 0, "drugs": []})
    assert e.value.kind == ErrorKind.tool_defect


async def test_block_is_quarantined_with_alternatives(tmp_path):
    gw = world(tmp_path)
    with pytest.raises(GatewayError) as e:
        await gw.prepare("drug", "old_tool", {}, None)
    assert e.value.kind == ErrorKind.quarantined
    assert e.value.envelope()["alternatives"] == ["mcp__drug__search_known_drugs"]


async def test_not_ready_names_table_and_check(tmp_path):
    bad = TableCheckModel(status="missing", checks=[{"name": "R1", "ok": False, "detail": "no files"}])
    gw = world(tmp_path, check={"open_targets.known_drug": bad})
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9"}, sorted_first)
    assert e.value.kind == ErrorKind.not_ready
    t = e.value.envelope()["tables"][0]
    assert t["name"] == "open_targets.known_drug" and t["check"] == "R1"
    # another tool on another table keeps working
    plan, res = await call(gw, "drug", "get_target", {"target_id": "PCSK9"},
                           lambda a: {"id": PCSK9, "approvedSymbol": "PCSK9"})
    assert hdr(res)["status"] == "ok"


async def test_unread_drifted_column_does_not_block(tmp_path):
    drift = TableCheckModel(status="schema_drift", columns={"status": "schema_drift"},
                            checks=[{"name": "R4", "ok": False, "column": "status", "detail": "str vs int"}])
    gw = world(tmp_path, check={"open_targets.known_drug": drift})
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 50}, sorted_first)
    assert hdr(res)["status"] == "ok"


async def test_index_degradation_per_accepted_kind(tmp_path):
    ot = copy.deepcopy(OT)
    ot["id_types"]["gene_alias"] = {"plugin": "hgnc_symbol", "universe": "target.approvedSymbol",
                                    "maps_to": [{"id_type": "ensembl_gene"}]}
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["args"]["target_id"]["accepts"] = ["ensembl_gene", "gene_alias"]
    gw = make_gateway(tmp_path / "a", [ot], [ov], TABLES, index_rows=index_rows(), keys=KEYS)
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": PCSK9, "limit": 50}, sorted_first)
    assert any("gene_alias not tried" in n for n in hdr(res)["notes"])    # that kind's index is missing
    assert gw.readiness_snapshot()["indexes"]["open_targets:gene_alias"]["status"] == "missing"
    gw = world(tmp_path)
    gw.readiness.set_index("open_targets:ensembl_gene", "ready")
    # a symbol needs the ensembl index (label kind) -> fails when that index cannot be built
    gw.service.index_rows.pop("open_targets:ensembl_gene")
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9"}, sorted_first)
    assert e.value.kind == ErrorKind.not_ready
    assert e.value.envelope()["tables"][0]["check"] == "index"


async def test_service_down_strict_vs_lenient(tmp_path):
    gw = world(tmp_path, fail={"*"})
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "search_known_drugs", {"target_id": PCSK9}, sorted_first)
    assert e.value.kind == ErrorKind.service_unavailable


async def test_observe_mode_never_raises_and_returns_upstream_unchanged(tmp_path):
    gw = world(tmp_path, data={"gateway": {"mode": "observe"}})
    await gw._ensure_index("open_targets:ensembl_gene")     # observe mode never builds one in the call path
    await gw._ensure_index("open_targets:hgnc_symbol")
    plan = await gw.prepare("drug", "search_known_drugs", {"target_id": "PCSK99"}, None)
    assert plan.route == "upstream" and plan.args_sent == {"target_id": "PCSK99"}
    out = await gw.finish(plan, raw_of({"success": True, "drugs": []}))
    assert json.loads(out) == {"success": True, "drugs": []}
    assert any(e[0] == "data_observe" and e[1]["decision"] == "would_not_found" for e in gw.bridge.events)


async def test_on_crash_oom_learns_refusal(tmp_path):
    gw = world(tmp_path)
    plan = await gw.prepare("drug", "search_known_drugs", {"target_id": PCSK9, "limit": 5}, None)
    plan.cold_tables = ("open_targets.known_drug",)
    d = await gw.on_crash("drug", plan, "connection closed", 'VBT_CHILD_EXIT {"signal": 9, "code": null}')
    assert d.oom and not d.retry and d.error.kind == ErrorKind.oom
    assert gw.admission.is_learned("drug", ["open_targets.known_drug"])
    d2 = await gw.on_crash("drug", plan, "connection reset", "")
    assert d2.error.kind == ErrorKind.server_crashed and not d2.retry   # second attempt


async def test_anchor_excluded_and_undefined_when_absent_from_table(tmp_path):
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["similar"] = {
        "reads": {"open_targets.target": {"access": "full_table"}},
        "args": {"entity_id": {"role": "anchor", "binds": "open_targets.target.id", "accepts": ["ensembl_gene", "hgnc_symbol"]},
                 "top_k": {"role": "limit", "min": 1, "max": 10}},
        "result": {"rows": "$.similar"}}
    gw = world(tmp_path, overlay=ov)
    plan, res = await call(gw, "drug", "similar", {"entity_id": "PCSK9", "top_k": 5},
                           lambda a: {"similar": [{"id": TP53}, {"id": BRCA1}]})
    w = gw.service.verbs("_witness")[-1]
    assert "not" in json.dumps(w.predicate)
    assert hdr(res)["total"] == 2                   # the anchor is not a candidate
    tables = dict(TABLES)
    tables["open_targets.target"] = [{"id": TP53, "approvedSymbol": "TP53"}]
    gw = world(tmp_path / "b", overlay=ov, tables=tables)
    plan, res = await call(gw, "drug", "similar", {"entity_id": "PCSK9"}, lambda a: {"similar": []})
    assert plan.route == "none"
    assert hdr(res)["undefined"]["reason"] == "not_in_table" and hdr(res)["status"] == "empty"


# --------------------------------------------------------------------------- status and coverage


async def test_empty_with_coverage_and_record_row_keys(tmp_path):
    gw = world(tmp_path)
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "BRCA1", "limit": 5},
                           lambda a: {"success": True, "count": 0, "drugs": []})
    h = hdr(res)
    assert h["status"] == "empty" and h["coverage"] == "covered"
    assert h["coverage_statement"] == "ChEMBL-curated drugs only."
    assert h["cite"].startswith("citable only as an absence")
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 50}, sorted_first)
    prov = res.provenance
    assert prov.id.startswith("dp_") and hdr(res)["prov"] == prov.id
    assert prov.result.row_keys_complete and len(prov.result.row_keys) == 8
    assert prov.result.key_columns == ["drugId", "targetId"]
    assert res.full_text == res.text and res.text.startswith('{"_vbt"')


async def test_explicit_not_found_on_other_table_is_empty_when_witness_counts_zero(tmp_path):
    gw = world(tmp_path)
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "BRCA1", "limit": 5},
                           lambda a: {"error": "Target BRCA1 not found"})
    assert hdr(res)["status"] == "empty"
    with pytest.raises(GatewayError) as e:      # witness counted rows: contradiction -> tool_defect
        await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 5},
                   lambda a: {"error": "Target PCSK9 not found"})
    assert e.value.kind == ErrorKind.tool_defect


async def test_generic_tool_structural_empty_is_empty_unverified(tmp_path):
    gw = world(tmp_path)
    plan, res = await call(gw, "other", "list_things", {"q": "x"}, lambda a: {"count": 0, "results": []})
    assert hdr(res)["status"] == "empty_unverified" and hdr(res)["cite"] == "not citable"
    with pytest.raises(GatewayError) as e:
        await call(gw, "other", "list_things", {"q": "x"}, lambda a: {"error": "Gene not found"})
    assert e.value.kind == ErrorKind.not_found


async def test_server_without_any_overlay_keeps_legacy_results(tmp_path):
    gw = world(tmp_path, generic=[])
    plan, res = await call(gw, "other", "list_things", {"q": "x"}, lambda a: {"count": 0, "results": []})
    assert json.loads(res) == {"count": 0, "results": []}
    plan, res = await call(gw, "other", "lookup", {}, lambda a: {"error": "Gene not found"})
    assert "not found" in res


async def test_pinned_and_readiness_snapshot(tmp_path):
    gw = world(tmp_path)
    await gw.refresh_readiness()
    snap = gw.readiness_snapshot()
    assert snap["tables"]["open_targets.known_drug"]["status"] == "ready"
    pin = gw.pinned()
    assert pin["mode"] == "enforce" and "open_targets" in pin["descriptors"]
    assert pin["determinism"]["hash_seed"] is None and pin["leakage"] == {"ceiling": None}
    assert pin["sources"]["open_targets"]["tables"]["known_drug"] == "fp1:open_targets.known_drug"


async def test_listing_hides_internal_verbs_and_rewrites_bound_tools(tmp_path):
    gw = world(tmp_path)
    d = gw.rewrite_listing("data", "_witness", "x", {})
    assert not d.visible
    schema = {"type": "object", "properties": {"target_id": {"type": "string"}, "limit": {"type": "integer"}}}
    d = gw.rewrite_listing("drug", "search_known_drugs", "Search drugs. Returns: stuff", schema)
    assert d.visible and d.input_schema["properties"]["target_id"]["x-vbt-id-type"] == "open_targets:ensembl_gene"
    assert d.input_schema["properties"]["limit"]["maximum"] == 200
    assert "Data: Open Targets 25.09" in d.description and "Returns" not in d.description
    d = gw.rewrite_listing("drug", "old_tool", "Old tool.", {})
    assert d.description.startswith("UNAVAILABLE: wrong answers on this data")
    d = gw.rewrite_listing("other", "list_things", "Generic.", {"type": "object"})
    assert d.description == "Generic." and d.input_schema == {"type": "object"}


@pytest.mark.parametrize("absence,universe,in_universe,excluded,expected", [
    ("unknown", False, None, 0, "unknown"),
    ("absent", False, None, 0, "covered"),
    ("absent", True, True, 0, "covered"),
    ("absent", True, False, 0, "not_covered"),
    ("absent", False, None, 2, "partial_unknown"),
    ("censored", False, None, 0, "censored"),
])
def test_coverage_table(tmp_path, absence, universe, in_universe, excluded, expected):
    from types import SimpleNamespace

    from vbt.datalayer.gateway.transforms import Counters
    gw = world(tmp_path)
    cov = SimpleNamespace(statement="s", absence_means=absence,
                          universe=SimpleNamespace(table="target", keys=["id"]) if universe else None)
    t = SimpleNamespace(spec=SimpleNamespace(coverage=cov))
    counters = Counters(excluded_unknown={"phase": excluded} if excluded else {})
    assert gw._coverage(t, "empty", counters, in_universe)[0] == expected


# --------------------------------------------------------------------------- through the real MCPBridge

FIXTURE_SERVER = Path(__file__).resolve().parent / "servers" / "gateway_fixture_server.py"
HAVE_FASTMCP = __import__("importlib").util.find_spec("fastmcp") is not None


@pytest.mark.skipif(not HAVE_FASTMCP, reason="fastmcp not installed")
async def test_through_mcp_bridge_with_stdio_server(tmp_path):
    import sys

    from vbt.tools.mcp_bridge import MCPBridge, MCPServerConfig

    ov = {"schema": "vbt.overlay/1", "server": "fix", "sources": ["open_targets"],
          "tools": {"lookup": {"reads": {"open_targets.target": {"access": "full_table"}},
                               "args": {"target_id": {"binds": "open_targets.target.id",
                                                      "accepts": ["ensembl_gene", "hgnc_symbol"]}},
                               "result": {"kind": "record", "rows": "$", "echo": {"target_id": "$.id"}}}}}
    gw = make_gateway(tmp_path, [OT], [ov], TABLES, index_rows=index_rows(), keys=KEYS)
    cfg = MCPServerConfig(name="fix", command=sys.executable, args=["-E", str(FIXTURE_SERVER), str(tmp_path / "st")])
    bridge = MCPBridge([cfg], log_dir=tmp_path / "logs", gateway=gw,
                       options={"start_backoff_s": 0.01, "start_backoff_factor": 1.0, "start_timeout_s": 60})
    try:
        tools = {t.name: t for t in await bridge.start()}
        assert tools["mcp__fix__lookup"].input_schema["properties"]["target_id"]["x-vbt-id-type"] == \
            "open_targets:ensembl_gene"
        out = await bridge.call("fix", "lookup", {"target_id": "PCSK9"})
        assert isinstance(out, DataResult) and out.status == "ok"
        assert out.obj["_vbt"]["resolved"]["target_id"].startswith("PCSK9 -> ENSG00000169174")
        assert out.obj["approvedSymbol"] == "PCSK9"
        with pytest.raises(GatewayError) as e:
            await bridge.call("fix", "lookup", {"target_id": "PCSK99"})
        assert e.value.kind == ErrorKind.not_found
        out = await bridge.call("fix", "empty", {})              # generic: uncitable structural empty
        assert out.status == "empty_unverified"
        with pytest.raises(GatewayError) as e:
            await bridge.call("fix", "legacy_error", {})
        assert e.value.kind == ErrorKind.source_error
    finally:
        await bridge.aclose()


# --------------------------------------------------------------------------- readiness cache


async def test_partial_partition_blocks_only_calls_that_can_read_it(tmp_path):
    ot = copy.deepcopy(OT)
    ot["tables"]["evidence"] = {
        "kind": "fact", "grain": "one evidence string", "key": {"columns": ["id"]}, "layout": "hive",
        "partitions": {"sourceId": {"column": {"role": "category", "vocab": ["chembl", "europepmc"],
                                               "scope": {"kind": "partition"}}}},
        "columns": {"id": {"role": "identifier"}, "targetId": {"role": "identifier", "id_type": "ensembl_gene"}}}
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["evidence"] = {"reads": {"open_targets.evidence": {"access": "bounded_scan"}},
                               "args": {"target_id": {"binds": "open_targets.evidence.targetId",
                                                      "accepts": ["ensembl_gene"]},
                                        "datasource_id": {"binds": "open_targets.evidence.sourceId"}},
                               "result": {"rows": "$.rows"}}
    check = TableCheckModel(status="partial", partitions={"sourceId=europepmc": "partial"},
                            checks=[{"name": "R1", "ok": False, "partition": "sourceId=europepmc",
                                     "detail": "a .part file"}])
    tables = dict(TABLES)
    tables["open_targets.evidence"] = [{"id": "e1", "targetId": PCSK9, "sourceId": "chembl"}]
    gw = make_gateway(tmp_path, [ot], [ov], tables, index_rows=index_rows(), keys=KEYS,
                      check={"open_targets.evidence": check})
    with pytest.raises(GatewayError) as e:
        await call(gw, "drug", "evidence", {"target_id": PCSK9}, lambda a: {"rows": []})
    assert e.value.kind == ErrorKind.not_ready and e.value.envelope()["tables"][0]["partition"] == "sourceId=europepmc"
    plan, res = await call(gw, "drug", "evidence", {"target_id": PCSK9, "datasource_id": "chembl"},
                           lambda a: {"rows": tables["open_targets.evidence"]})
    assert hdr(res)["status"] == "ok"
    assert "mcp__drug__evidence" not in gw.degraded_tools()      # a partition may be excluded by arguments


def test_readiness_cache_persists_and_shallow_refresh_detects_changes(tmp_path):
    from vbt.datalayer.gateway.readiness import ReadinessCache
    from vbt.datalayer.ipc import CheckResponse

    root = tmp_path / "ot"
    (root / "known_drug").mkdir(parents=True)
    (root / "known_drug" / "part-0.parquet").write_bytes(b"x")
    ot = copy.deepcopy(OT)
    ot["root"] = str(root)
    ot["tables"]["known_drug"]["path"] = "known_drug"
    gw = make_gateway(tmp_path, [ot], [DRUG_OVERLAY], TABLES, index_rows=index_rows(), keys=KEYS)
    cache = ReadinessCache(tmp_path / "cache", gw.catalog, REGISTRY, refresh_interval_s=0)
    cache.load_check_results(CheckResponse(tables={"open_targets.known_drug": TableCheckModel(status="ready")}))
    assert cache.signatures["open_targets.known_drug"]
    again = ReadinessCache(tmp_path / "cache", gw.catalog, REGISTRY, refresh_interval_s=0)
    assert again.load() == 1 and again.get("open_targets.known_drug").status == "ready"
    assert again.shallow_refresh(["open_targets.known_drug"]) == []
    (root / "known_drug" / "part-1.parquet").write_bytes(b"y")
    assert again.shallow_refresh(["open_targets.known_drug"]) == ["open_targets.known_drug"]
    assert again.get("open_targets.known_drug") is None
    third = ReadinessCache(tmp_path / "cache", gw.catalog, REGISTRY)
    assert third.load() == 0                                  # the persisted signature no longer matches


# --------------------------------------------------------------------------- levels and phantom rows (cBioPortal shape)

CBIO: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "cbio", "title": "cBioPortal", "kind": "remote",
    "release": {"from": "as_of"},
    "tables": {"sample": {"kind": "records", "grain": "one sample", "format": "none",
                          "key": {"columns": ["studyId", "sampleId"]}, "grains": {"patient": ["patientId"]},
                          "columns": {"studyId": {"role": "identifier"}, "sampleId": {"role": "identifier"},
                                      "patientId": {"role": "identifier"},
                                      "OS_MONTHS": {"role": "time", "level": "patient"}}}},
}
CBIO_OV: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "clinicaltrials",
    "tools": {"get_clinical_data": {
        "reads": {"cbio.sample": {"access": "upstream"}},
        "args": {"study_id": {"binds": "cbio.sample.studyId", "existence": "off"},
                 "sample_ids": {"binds": "cbio.sample.sampleId", "op": "in", "each": True}},
        "result": {"rows": "$.data", "row_key": ["sampleId"], "key_from_args": {"studyId": "study_id"},
                   "exists_when": "$.patientId != null", "on_unknown_items": "partial",
                   "levels": {"patient": ["OS_MONTHS"]}, "count_fields": ["$.sample_count"]}}},
}


async def test_levels_split_and_phantom_rows(tmp_path):
    gw = make_gateway(tmp_path, [CBIO], [CBIO_OV], {})
    data = [{"sampleId": "S-01", "patientId": "P-01", "OS_MONTHS": 12.0},
            {"sampleId": "S-02", "patientId": "P-01", "OS_MONTHS": 12.0},
            {"sampleId": "S-03", "patientId": "P-02", "OS_MONTHS": 3.5},
            {"sampleId": "S-99", "patientId": None, "OS_MONTHS": None}]
    plan, res = await call(gw, "clinicaltrials", "get_clinical_data",
                           {"study_id": "study_x", "sample_ids": ["S-01", "S-02", "S-03", "S-99"]},
                           lambda a: {"data": data, "sample_count": 4})
    h = hdr(res)
    assert [r["sampleId"] for r in res.obj["data"]] == ["S-01", "S-02", "S-03"]
    assert all("OS_MONTHS" not in r for r in res.obj["data"])
    assert res.obj["patients"] == [{"patientId": "P-01", "OS_MONTHS": 12.0}, {"patientId": "P-02", "OS_MONTHS": 3.5}]
    assert h["grains"]["patient"] == {"returned": 2, "total": 2}
    assert h["not_found_items"] == ["S-99"] and h["withheld"] == {"phantom": 1} and h["status"] == "partial"
    assert res.obj["sample_count"] == 3
    strict = copy.deepcopy(CBIO_OV)
    strict["tools"]["get_clinical_data"]["result"]["on_unknown_items"] = "not_found"
    gw = make_gateway(tmp_path / "b", [CBIO], [strict], {})
    with pytest.raises(GatewayError) as e:
        await call(gw, "clinicaltrials", "get_clinical_data", {"study_id": "study_x", "sample_ids": ["S-99"]},
                   lambda a: {"data": data[3:], "sample_count": 1})
    assert e.value.kind == ErrorKind.not_found and e.value.envelope()["items"] == ["S-99"]


async def test_witness_types_come_from_the_session_check_not_a_stats_request(tmp_path):
    """A first call waited for the session check and then for ``_stats`` of its table, only for the storage types
    of the witness keys: the data child samples the table and every item table over it (14.5 s for the 25.09
    target tables). The check read the footers already and hands the types over."""
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["reads"]["open_targets.known_drug"]["access"] = "projection"   # no admission
    types = {"drugId": "large_string", "targetId": "large_string", "phase": "int32", "status": "string"}
    checked = TableCheckModel(status="ready", fingerprint="fp1:open_targets.known_drug", storage_types=types)
    gw = world(tmp_path, overlay=ov, check={"open_targets.known_drug": checked})
    await gw.wait_readiness()
    _, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20}, sorted_first)
    assert hdr(res)["total"] == 8 and gw.service.verbs("_witness")
    assert not gw.service.verbs("_stats"), "the witness asked _stats for types the check had read"
    assert res.provenance.result.key_storage_types == ["large_string", "large_string"]
    # without types in the check (a readiness cache written before them) the witness still asks _stats
    gw2 = world(tmp_path / "old", overlay=ov,
                storage={"open_targets.known_drug": {"drugId": "string", "targetId": "string"}})
    await gw2.wait_readiness()
    _, res2 = await call(gw2, "drug", "search_known_drugs", {"target_id": "PCSK9", "limit": 20}, sorted_first)
    assert gw2.service.verbs("_stats") and res2.provenance.result.key_storage_types == ["string", "string"]

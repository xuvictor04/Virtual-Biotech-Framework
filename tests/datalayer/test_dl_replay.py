"""``vbt ds replay`` (§15.1, §16, F22) on the Open Targets fixture through the real gateway and data child.

* A recorded derived call (``association.query_evidence``, which reads the hive-partitioned
  ``evidence`` table and fixes ``sourceId`` from the data) replays to ``match``: same canonical row
  keys, same output rows, same partition fingerprints.
* Changing a row of the partition the call read gives ``replay_mismatch`` and names the partition.
* Changing a partition the call did not read is ``match``: the change is listed under ``ignored``,
  never as drift.
* A live record whose version changed is ``source_updated``; rows of records whose version did not
  change are still compared, so a changed unchanged-version row is ``replay_mismatch``.
* The pure parts: replay arguments (resolved identifiers, agent's limit), partitions from the scope,
  enrichment rows compared as ``(set, k, K, p, q)`` within tolerance, and the CLI wiring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow
from vbt.datalayer.replay import (
    GatewayReplayer,
    InProcessDataBridge,
    compare,
    partitions_from_scope,
    replay_args,
    replay_async,
    rows_close,
    stamp_partitions,
)

TUID = "toolu_replay_01"
CALL = ("association", "query_evidence", {"target_id": "PCSK9", "limit": 5})


# ---------------------------------------------------------------------------- the world


def _replayer(root: Path, cache: Path) -> GatewayReplayer:
    """A gateway over ``root`` with the data child's verbs in this process (a fresh context each time,
    as a new process would have)."""
    from vbt.datalayer.gateway import DataGateway
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(cache)}, project_root=REPO)
    ctx = ServiceContext(settings)
    out = cache.parent / "mcp-out"
    out.mkdir(parents=True, exist_ok=True)
    gw = DataGateway(ctx.settings, ctx.catalog, ctx.registry, run={"mcp_output_dir": str(out)})
    gw.bind_bridge(InProcessDataBridge(ctx))
    return GatewayReplayer(gw)


def _write_run(run: Path, record: dict[str, Any], text: str) -> None:
    (run / "logs" / "data_provenance").mkdir(parents=True, exist_ok=True)
    (run / "logs" / "data_provenance" / f"{TUID}.json").write_text(json.dumps(record))
    events = [{"type": "tool_start", "tool": record["tool"], "tool_use_id": TUID, "input": CALL[2]},
              {"type": "tool_end", "tool": record["tool"], "tool_use_id": TUID, "is_error": False,
               "output": text, "output_chars": len(text), "result_status": record["result"]["status"]}]
    (run / "logs" / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


@pytest.fixture
def world(ot_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A copy of the OT fixture and a run with one recorded ``query_evidence`` call, its record stamped
    with the partitions it read (as the runtime does once the gateway records them)."""
    import dl_fixtures as F

    root = F.copy_fixture(ot_root, tmp_path / "ot" / "25.09")
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(root))
    rp = _replayer(root, tmp_path / "cache0")

    async def record() -> tuple[dict[str, Any], str]:
        res = await rp.call(*CALL)
        rec = res.provenance.to_dict()
        rec["tool_use_id"] = TUID
        tables = [f"open_targets.{t['name']}" for t in rec["tables"]]
        stamp_partitions(rec, await rp.stats(tables))
        return rec, res.text

    import asyncio

    rec, text = asyncio.run(record())
    run = tmp_path / "run"
    _write_run(run, rec, text)
    return SimpleNamespace(root=root, run=run, record=rec, tmp=tmp_path, n=0)


def _replay(world: SimpleNamespace) -> Any:
    import asyncio

    world.n += 1
    rp = _replayer(world.root, world.tmp / f"cache{world.n}")
    return asyncio.run(replay_async(world.run, TUID, replayer=rp))


def _edit_partition(root: Path, source: str, edit: Any) -> None:
    """Rewrite one ``evidence`` partition with ``edit(rows) -> rows`` and refresh the manifest."""
    import pyarrow.parquet as pq

    import dl_fixtures as F

    for path in sorted((root / "evidence" / f"sourceId={source}").glob("*.parquet")):
        tbl = pq.read_table(path)
        rows = edit(tbl.to_pylist())
        pq.write_table(F.pa.Table.from_pylist(rows, schema=tbl.schema), path)
    F.write_manifest(root)


# ---------------------------------------------------------------------------- the fixture world


@needs_arrow
def test_the_recorded_call_reads_one_partition(world) -> None:
    rec = world.record
    assert rec["served_by"] == "derived" and rec["result"]["returned"] >= 1
    (ev,) = [t for t in rec["tables"] if t["name"] == "evidence"]
    assert ev["partitions_read"] == ["sourceId=europepmc"]
    assert set(ev["partition_fingerprints"]) == {"sourceId=europepmc"}
    assert rec["request"]["args_sent"]["target_id"] == "ENSG00000169174"


@needs_arrow
def test_replay_matches_on_unchanged_data(world) -> None:
    r = _replay(world)
    assert r.status == "match", r.to_dict()
    names = {c.name: c.ok for c in r.checks}
    assert names["row_keys_sha256"] is True and names["output_rows_sha256"] is True and names["status"] is True
    assert r.drift == [] and r.ignored == [] and r.source_updated == []
    assert r.args == {"target_id": "ENSG00000169174", "limit": 5}


@needs_arrow
def test_a_changed_row_in_the_partition_read_is_a_mismatch(world) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            if r.get("targetId") == "ENSG00000169174":
                r["score"] = 0.123456
        return rows

    _edit_partition(world.root, "europepmc", edit)
    r = _replay(world)
    assert r.status == "replay_mismatch", r.to_dict()
    assert "output_rows_sha256" in {c.name for c in r.mismatches}
    assert [d["partition"] for d in r.drift] == ["sourceId=europepmc"]


@needs_arrow
def test_a_changed_partition_the_call_did_not_read_is_ignored(world) -> None:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in rows:
            r["score"] = 0.5
        return rows

    _edit_partition(world.root, "chembl", edit)
    r = _replay(world)
    assert r.status == "match", r.to_dict()
    assert r.drift == []
    assert [d["table"] for d in r.ignored] == ["open_targets.evidence"]
    assert "did not read" in r.ignored[0]["detail"]


@needs_arrow
def test_without_partition_fingerprints_any_table_change_is_drift_but_rows_decide(world) -> None:
    rec = json.loads((world.run / "logs" / "data_provenance" / f"{TUID}.json").read_text())
    for t in rec["tables"]:
        t["partitions_read"] = t["partition_fingerprints"] = None
    (world.run / "logs" / "data_provenance" / f"{TUID}.json").write_text(json.dumps(rec))
    _edit_partition(world.root, "chembl", lambda rows: [{**r, "score": 0.5} for r in rows])
    r = _replay(world)
    assert r.status == "match"
    assert [d["table"] for d in r.drift] == ["open_targets.evidence"]
    assert "sourceId=europepmc" in r.drift[0]["detail"]


@needs_arrow
def test_a_missing_record_is_unavailable(world) -> None:
    import asyncio

    r = asyncio.run(replay_async(world.run, "toolu_unknown", replayer=_replayer(world.root, world.tmp / "c9")))
    assert r.status == "unavailable" and r.error["kind"] == "no_record"


# ---------------------------------------------------------------------------- live record versions


class _LiveReplayer:
    def __init__(self, record: dict[str, Any], obj: Any) -> None:
        self.result = SimpleNamespace(provenance=record, obj=obj)

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        return self.result

    async def stats(self, tables: list[str]) -> dict[str, Any]:
        return {}

    def row_paths(self, server: str, tool: str) -> list[str]:
        return ["$.studies"]


def _live(tmp: Path, recorded_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]],
          recorded_versions: dict[str, str], current_versions: dict[str, str]) -> Any:
    import asyncio

    def rec(rows: list[dict[str, Any]], versions: dict[str, str], digest: str) -> dict[str, Any]:
        return {"schema": "vbt.dataprov/1", "id": "dp_x", "tool_use_id": TUID, "tool": "mcp__clinicaltrials__search",
                "server": "clinicaltrials", "served_by": "upstream", "source": {"name": "clinicaltrials"},
                "tables": [{"name": "studies", "fingerprint": None}],
                "request": {"args_raw": {"query": "x"}, "args_sent": {"query": "x"}},
                "result": {"status": "ok", "returned": len(rows), "total": len(rows), "total_method": "remote",
                           "row_keys_sha256": digest, "output_rows_sha256": digest},
                "record_versions": {"clinicaltrials.studies": versions}}

    recorded = rec(recorded_rows, recorded_versions, "a")
    run = tmp / "live-run"
    (run / "logs" / "data_provenance").mkdir(parents=True)
    (run / "logs" / "data_provenance" / f"{TUID}.json").write_text(json.dumps(recorded))
    text = json.dumps({"studies": recorded_rows})
    (run / "logs" / "trace.jsonl").write_text(json.dumps(
        {"type": "tool_end", "tool_use_id": TUID, "output": text, "output_chars": len(text)}) + "\n")
    current = rec(current_rows, current_versions, "b")
    return asyncio.run(replay_async(run, TUID, replayer=_LiveReplayer(current, {"studies": current_rows})))


def test_a_changed_live_version_is_source_updated(tmp_path: Path) -> None:
    before = [{"nctId": "NCT1", "status": "RECRUITING"}, {"nctId": "NCT2", "status": "COMPLETED"}]
    after = [{"nctId": "NCT1", "status": "TERMINATED"}, {"nctId": "NCT2", "status": "COMPLETED"}]
    r = _live(tmp_path, before, after, {"NCT1": "2024-01-01", "NCT2": "2024-02-01"},
              {"NCT1": "2025-06-01", "NCT2": "2024-02-01"})
    assert r.status == "source_updated", r.to_dict()
    assert r.source_updated == ["NCT1"]
    assert all(c.ok is not False for c in r.checks)


def test_an_unchanged_live_version_with_different_rows_is_a_mismatch(tmp_path: Path) -> None:
    before = [{"nctId": "NCT1", "status": "RECRUITING"}, {"nctId": "NCT2", "status": "COMPLETED"}]
    after = [{"nctId": "NCT1", "status": "TERMINATED"}, {"nctId": "NCT2", "status": "WITHDRAWN"}]
    r = _live(tmp_path, before, after, {"NCT1": "2024-01-01", "NCT2": "2024-02-01"},
              {"NCT1": "2025-06-01", "NCT2": "2024-02-01"})
    assert r.status == "replay_mismatch" and r.source_updated == ["NCT1"]


# ---------------------------------------------------------------------------- pure parts


def test_replay_args_send_resolved_identifiers_and_the_agents_limit() -> None:
    record = {"request": {"args_raw": {"target_id": "PCSK9", "limit": 20, "min_phase": 2},
                          "args_sent": {"target_id": "ENSG00000169174", "limit": 61, "min_phase": 2},
                          "resolutions": [{"arg": "target_id", "raw": "PCSK9", "canonical": "ENSG00000169174"}]}}
    assert replay_args(record) == {"target_id": "ENSG00000169174", "limit": 20, "min_phase": 2}
    assert replay_args({"request": {"args_sent": {"a": 1}}}) == {"a": 1}


def test_partitions_from_the_fixed_scope() -> None:
    parts = ["sourceId=chembl", "sourceId=europepmc", "sourceId=eva"]
    rec = {"request": {"scope": {"sourceId": "europepmc"}}}
    assert partitions_from_scope(rec, parts) == ["sourceId=europepmc"]
    assert partitions_from_scope({"request": {"scope": {"sourceId": ["eva", "chembl"]}}}, parts) == \
        ["sourceId=chembl", "sourceId=eva"]
    assert partitions_from_scope({"request": {"scope": {"other": 1}}}, parts) is None


def test_enrichment_rows_compare_as_set_k_K_p_q_within_tolerance() -> None:
    a = [{"set_id": "GO:1", "overlap": 3, "set_size": 40, "pvalue": 1.0e-6, "fdr": 2.0e-5, "set_label": "x"}]
    b = [{"set_id": "GO:1", "overlap": 3, "set_size": 40, "pvalue": 1.0e-6 * (1 + 1e-12), "fdr": 2.0e-5,
          "set_label": "renamed", "expected": 0.4}]
    assert rows_close(a, b)
    c = [dict(b[0], overlap=4)]
    assert not rows_close(a, c)
    assert not rows_close(a, [dict(b[0], pvalue=1.1e-6)])


def test_compare_uses_rows_when_the_hashes_differ() -> None:
    rec = {"tool": "mcp__x__y", "tables": [], "result": {"status": "ok", "total": 1, "row_keys_sha256": "k",
                                                         "output_rows_sha256": "a"}}
    cur = {"tables": [], "result": {"status": "ok", "total": 1, "row_keys_sha256": "k", "output_rows_sha256": "b"}}
    rows = [{"id": "x", "score": 0.1 + 0.2}]
    assert compare(rec, cur, recorded_rows=rows, current_rows=[{"id": "x", "score": 0.3}]).status == "match"
    assert compare(rec, cur, recorded_rows=rows, current_rows=[{"id": "x", "score": 0.31}]).status == \
        "replay_mismatch"
    assert compare(rec, cur).status == "replay_mismatch"      # no rows: the hash decides


def test_replay_cli_is_wired() -> None:
    from vbt.datalayer.cli import COMMANDS, add_datasource_parsers

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    add_datasource_parsers(sub)
    args = ap.parse_args(["ds", "replay", "runs/r1", "toolu_1", "--backend", "inprocess", "--json"])
    assert args.ds_cmd == "replay" and args.tool_use_id == ["toolu_1"] and args.backend == "inprocess"
    assert "replay" in COMMANDS

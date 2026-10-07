"""Row-level citations and ``vbt verify --data`` (DATA_LAYER.md §15.3-15.4, F22).

* A cited ``row_key`` is checked against the row keys stored in the call's ``vbt.dataprov/1`` record
  in the canonical encoding: a key the call returned is accepted, also when a float32 key part is
  cited widened to float64 and when a part is null; a key it did not return, of the wrong width, or
  against a record whose keys no longer match their digest is rejected; a record that kept only a
  sample warns; a record that kept only the digest of one row is checked by recomputing the digest.
* ``refresh_claims`` marks tool_call evidence unresolved when a table the call read has a new
  fingerprint, and when the fingerprints a claim was filed with differ from the call's.
* ``data_version_drift``: pinned or cited-call fingerprints that differ from the current ones are a
  warning with cached fingerprints and a problem (INCOMPLETE) under ``--data``; ``replay_mismatch``
  is a problem and ``source_updated`` a warning; none is an integrity failure.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from vbt.audit.claims import EvidenceContext, check_evidence, load_claims, refresh_claims, row_key_problems
from vbt.datalayer.record import cap_row_keys
from vbt.datalayer.replay import ReplayCheck, ReplayResult
from vbt.datalayer.rowkey import canonical, canonical_sha256
from vbt.session import Run
from vbt.verify import DATA_KINDS, INTEGRITY_KINDS, main, pinned_fingerprints, verify_run

TOOL = "mcp__drug__search_known_drugs"
F32_005 = struct.unpack("f", struct.pack("f", 0.05))[0]           # 0.05000000074505806


def _record(keys: list[list[Any]], *, types: list[str | None] | None = None, max_n: int = 10_000,
            cols: list[str] | None = None) -> dict[str, Any]:
    kept, complete, sha = cap_row_keys(keys, max_n, types)
    return {"schema": "vbt.dataprov/1", "id": "dp_000000000001", "tool": TOOL, "server": "drug",
            "result": {"status": "ok", "returned": len(keys), "total": len(keys),
                       "key_columns": cols or ["drugId", "concentration", "diseaseId"], "row_keys": kept,
                       "row_keys_complete": complete, "row_keys_sha256": sha}}


KEYS = [["CHEMBL25", F32_005, None], ["CHEMBL25", 0.5, "EFO_0000001"], ["CHEMBL99", 1.0, None]]
TYPES = ["string", "float", "string"]


# --------------------------------------------------------------------------- row_key_problems


@pytest.mark.parametrize("cited,ok", [
    (["CHEMBL25", 0.05, None], True),                      # the canonical float32 rendering
    (["CHEMBL25", F32_005, None], True),                   # widened to float64 by the agent
    (["CHEMBL25", 0.5, "EFO_0000001"], True),
    (["CHEMBL99", 1, None], True),                         # an integer for a float part
    (["CHEMBL25", 0.05, "EFO_0000001"], False),            # null part cited as a value
    (["CHEMBL25", 0.051, None], False),
    (["CHEMBL26", 0.05, None], False),
])
def test_row_key_against_the_stored_keys(cited: list[Any], ok: bool) -> None:
    problems, warnings = row_key_problems(cited, _record(KEYS, types=TYPES))
    assert (not problems) is ok, problems
    assert not warnings
    if not ok:
        assert "is not among the 3 row(s)" in problems[0]


def test_row_key_with_recorded_storage_types_is_exact() -> None:
    rec = _record(KEYS, types=TYPES)
    rec["result"]["key_storage_types"] = TYPES
    assert row_key_problems(["CHEMBL25", F32_005, None], rec) == ([], [])
    assert row_key_problems(["CHEMBL25", 0.05, None], rec) == ([], [])


def test_row_key_width_sample_and_digest_checks() -> None:
    problems, _ = row_key_problems(["CHEMBL25"], _record(KEYS, types=TYPES))
    assert "has 1 value(s) but the result's key has 3" in problems[0]
    problems, _ = row_key_problems("CHEMBL25", _record(KEYS))
    assert "must be a list" in problems[0]
    # a sample only: a key outside it is not rejected
    sampled = _record([[f"CHEMBL{i}", 1.0, None] for i in range(50)], max_n=10)
    assert sampled["result"]["row_keys_complete"] is False
    problems, warnings = row_key_problems(["CHEMBL40", 1.0, None], sampled)
    assert not problems and "stored sample" in warnings[0]
    assert row_key_problems(["CHEMBL3", 1.0, None], sampled) == ([], [])
    # stored keys that do not match their digest: the record was altered
    altered = _record(KEYS, types=TYPES)
    altered["result"]["row_keys"].append(["CHEMBL1", 2.0, None])
    problems, _ = row_key_problems(["CHEMBL1", 2.0, None], altered)
    assert "do not match their digest" in problems[0]
    # the trace's digest disagrees with the record's
    problems, _ = row_key_problems(["CHEMBL25", 0.05, None], _record(KEYS, types=TYPES),
                                   {"row_keys_sha256": "0" * 64})
    assert "does not match the row-key digest in the trace" in problems[0]


def test_row_key_by_recomputing_the_digest_of_one_row() -> None:
    key = ["CHEMBL25", 0.05, None]
    rec = {"result": {"returned": 1, "key_columns": ["a", "b", "c"], "row_keys": [],
                      "row_keys_sha256": canonical_sha256([canonical(key, TYPES)])}}
    assert row_key_problems(key, rec) == ([], [])
    problems, _ = row_key_problems(["CHEMBL25", 0.5, None], rec)
    assert "does not match the row-key digest" in problems[0]
    problems, _ = row_key_problems(key, {"result": {"returned": 0, "row_keys": []}})
    assert "returned no rows" in problems[0]
    _p, warnings = row_key_problems(key, None)
    assert "provenance record is missing" in warnings[0]


def test_row_keys_by_grain_are_accepted() -> None:
    rec = _record([["S1", "P1"], ["S2", "P1"]], cols=["sampleId", "patientId"])
    _kept, _c, sha = cap_row_keys([["P1"]], 100)
    rec["result"]["row_keys_by_grain"] = {"patient": {"key_columns": ["patientId"], "row_keys": [["P1"]],
                                                      "row_keys_complete": True, "row_keys_sha256": sha}}
    assert row_key_problems(["P1"], rec) == ([], [])
    assert row_key_problems(["S2", "P1"], rec) == ([], [])


# --------------------------------------------------------------------------- through record_claims and verify


@pytest.fixture
def run(tmp_path: Path):
    r = Run(tmp_path / "runs")
    yield r
    if not r._closed:
        r.close()


def _data_call(run: Run, tuid: str, record: dict[str, Any] | None, *, fp: str = "fp1:sha256:old") -> None:
    run.trace("tool_start", agent="a", tool=TOOL, tool_use_id=tuid, input={"target_id": "PCSK9"})
    summary = {"prov": f"dp_{tuid:0>12}", "status": "ok", "coverage": "covered", "source": "open_targets@25.09",
               "tables": [{"name": "known_drug", "fingerprint": fp}], "returned": 3, "total": 3}
    if record is not None:
        rel = f"logs/data_provenance/{tuid}.json"
        (run.dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (run.dir / rel).write_text(json.dumps({**record, "tool_use_id": tuid}))
        summary["record"] = rel
        summary["row_keys_sha256"] = record["result"]["row_keys_sha256"]
    run.trace("tool_end", agent="a", tool=TOOL, tool_use_id=tuid, is_error=False, output="{}", output_chars=2,
              result_status="ok", data_provenance=summary)


def _claim(cid: str, tuid: str, row_key: list[Any] | None = None) -> dict[str, Any]:
    ev: dict[str, Any] = {"kind": "tool_call", "tool_use_id": tuid}
    if row_key is not None:
        ev["row_key"] = row_key
    return {"id": cid, "text": "CHEMBL25 is a known drug of PCSK9", "evidence": [ev]}


def test_record_claims_accepts_and_rejects_row_keys(run: Run) -> None:
    _data_call(run, "t1", _record(KEYS, types=TYPES))
    ok = run.record_claims([_claim("C1", "t1", ["CHEMBL25", F32_005, None])])
    assert ok["ok"], ok
    assert load_claims(run.dir)[0]["evidence"][0]["row_key"] == ["CHEMBL25", F32_005, None]
    bad = run.record_claims([_claim("C2", "t1", ["CHEMBL42", 0.05, None])])
    assert not bad["ok"] and "is not among the 3 row(s) the call returned" in bad["errors"][0]
    run.close()
    report = verify_run(run.dir)
    assert report["evidence"]["valid_claims"] == 1 and report["status"] == "COMPLETE", report["problems"]


def test_verify_rejects_a_row_key_after_the_record_changed(run: Run) -> None:
    _data_call(run, "t1", _record(KEYS, types=TYPES))
    assert run.record_claims([_claim("C1", "t1", ["CHEMBL99", 1.0, None])])["ok"]
    run.close()
    path = run.dir / "logs" / "data_provenance" / "t1.json"
    rec = json.loads(path.read_text())
    rec["result"]["row_keys"] = rec["result"]["row_keys"][:2]           # the cited row removed
    path.write_text(json.dumps(rec))
    report = verify_run(run.dir)
    assert report["status"] == "INCOMPLETE"
    assert any("digest" in p["detail"] for p in report["problems"] if p["kind"] == "unresolved_evidence")


def test_refresh_claims_marks_evidence_unresolved_on_a_fingerprint_change(run: Run) -> None:
    _data_call(run, "t1", _record(KEYS, types=TYPES), fp="fp1:sha256:old")
    assert run.record_claims([_claim("C1", "t1")])["ok"]
    stored = load_claims(run.dir)
    assert stored[0]["evidence"][0]["fingerprints"] == {"known_drug": "fp1:sha256:old"}
    calls = {"t1": run.tool_call_status("t1")}
    same = refresh_claims(stored, EvidenceContext(run.dir, {}, calls,
                                                  fingerprints={"open_targets.known_drug": "fp1:sha256:old"}))
    assert same[0]["evidence"][0]["evidence_status"] == "verified" and same[0]["n_verified"] == 1
    changed = refresh_claims(stored, EvidenceContext(run.dir, {}, calls,
                                                     fingerprints={"open_targets.known_drug": "fp1:sha256:new"}))
    ev = changed[0]["evidence"][0]
    assert ev["evidence_status"] == "unresolved" and changed[0]["n_verified"] == 0
    assert "the data changed since the call" in ev["problem"]
    # without current fingerprints nothing changes; a refiled claim with other fingerprints is unresolved
    assert refresh_claims(stored, EvidenceContext(run.dir, {}, calls))[0]["n_verified"] == 1
    tampered = [dict(stored[0], evidence=[dict(stored[0]["evidence"][0], fingerprints={"known_drug": "fp1:x"})])]
    out = refresh_claims(tampered, EvidenceContext(run.dir, {}, calls))
    assert out[0]["evidence"][0]["evidence_status"] == "unresolved"


def test_check_evidence_fingerprints_by_table_name(tmp_path: Path) -> None:
    call = {"tool": TOOL, "is_error": False, "result_status": "ok",
            "data_provenance": {"status": "ok", "source": "open_targets@25.09",
                                "tables": [{"name": "known_drug", "fingerprint": "fp1:a"}]}}
    ctx = EvidenceContext(tmp_path, {}, {"t1": call}, fingerprints={"known_drug": "fp1:b"})
    _entry, problems, _w = check_evidence({"kind": "tool_call", "tool_use_id": "t1"}, ctx)
    assert problems and "which is now fp1:b" in problems[0]


# --------------------------------------------------------------------------- data_version_drift, replays


def _pinned_run(run: Run, *, fp: str = "fp1:sha256:old") -> None:
    run.set_config({"data": {"sources": {"open_targets": {"release": "25.09",
                                                          "tables": {"known_drug": fp, "target": "fp1:t"}}}}})
    _data_call(run, "t1", _record(KEYS, types=TYPES), fp=fp)
    assert run.record_claims([_claim("C1", "t1")])["ok"]
    run.close()


def _kinds(items: list[dict[str, Any]]) -> list[str]:
    return [p["kind"] for p in items]


def test_pinned_fingerprints_come_from_the_manifest(run: Run) -> None:
    _pinned_run(run)
    manifest = json.loads((run.dir / "MANIFEST.json").read_text())
    assert pinned_fingerprints(manifest) == {"open_targets.known_drug": "fp1:sha256:old",
                                             "open_targets.target": "fp1:t"}


@pytest.mark.usefixtures("fixed_replays")
def test_data_version_drift_is_a_warning_without_data_and_incomplete_with_it(run: Run) -> None:
    _pinned_run(run)
    current = {"open_targets.known_drug": "fp1:sha256:new", "open_targets.target": "fp1:t"}
    soft = verify_run(run.dir, fingerprints=current)
    assert soft["status"] == "COMPLETE"
    (drift,) = [w for w in soft["evidence"]["warnings"] if w["kind"] == "data_version_drift"]
    assert drift["table"] == "open_targets.known_drug" and drift["claims"] == ["C1"]
    assert drift["tool_use_ids"] == ["t1"] and drift["current"] == "fp1:sha256:new"

    hard = verify_run(run.dir, fingerprints=current, data=True, config={},
                      replayer=_FixedReplayer(ReplayResult("t1", TOOL, status="match")))
    assert hard["status"] == "INCOMPLETE" and "data_version_drift" in _kinds(hard["problems"])
    assert hard["integrity"]["status"] == "passed"
    unchanged = verify_run(run.dir, fingerprints={"open_targets.known_drug": "fp1:sha256:old"}, data=True,
                           config={}, replayer=_FixedReplayer(ReplayResult("t1", TOOL, status="match")))
    assert unchanged["status"] == "COMPLETE", unchanged["problems"]
    assert unchanged["data"]["replays"] == {"t1": "match"}


def test_cached_fingerprints_feed_the_warning(run: Run, tmp_path: Path) -> None:
    _pinned_run(run)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "fingerprints.json").write_text(json.dumps({"tables": {"open_targets.known_drug": "fp1:sha256:new"}}))
    report = verify_run(run.dir, config={"data": {"cache_dir": str(cache)}})
    assert report["data"]["fingerprints_from"] == "cache" and report["data"]["drift"] == ["open_targets.known_drug"]
    assert report["status"] == "COMPLETE"


class _FixedReplayer:
    """Stands in for ``replay_async``'s replayer: every call replays to the given result."""

    def __init__(self, result: ReplayResult) -> None:
        self.result = result


@pytest.fixture
def fixed_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    import vbt.datalayer.replay as replay_mod

    async def fake(run_dir: Any, tuid: str, config: Any = None, *, replayer: Any = None, backend: str = "auto"):
        return replayer.result if isinstance(replayer, _FixedReplayer) else ReplayResult(tuid, status="unavailable")

    monkeypatch.setattr(replay_mod, "replay_async", fake)


@pytest.mark.usefixtures("fixed_replays")
def test_replay_mismatch_is_a_problem_and_source_updated_a_warning(run: Run) -> None:
    _pinned_run(run)
    fps = {"open_targets.known_drug": "fp1:sha256:old"}
    bad = ReplayResult("t1", TOOL, status="replay_mismatch",
                       checks=[ReplayCheck("row_keys_sha256", False, "a", "b")])
    report = verify_run(run.dir, fingerprints=fps, data=True, config={}, replayer=_FixedReplayer(bad))
    (p,) = [p for p in report["problems"] if p["kind"] == "replay_mismatch"]
    assert p["claims"] == ["C1"] and p["tool_use_id"] == "t1" and report["status"] == "INCOMPLETE"
    updated = ReplayResult("t1", TOOL, status="source_updated", source_updated=["NCT1"])
    report = verify_run(run.dir, fingerprints=fps, data=True, config={}, replayer=_FixedReplayer(updated))
    assert report["status"] == "COMPLETE"
    assert "source_updated" in _kinds(report["evidence"]["warnings"])
    gone = ReplayResult("t1", TOOL, status="unavailable", error={"kind": "not_ready", "message": "x"})
    report = verify_run(run.dir, fingerprints=fps, data=True, config={}, replayer=_FixedReplayer(gone))
    assert report["status"] == "COMPLETE" and "replay_unavailable" in _kinds(report["evidence"]["warnings"])


def test_data_kinds_are_not_integrity_kinds() -> None:
    assert not DATA_KINDS & INTEGRITY_KINDS
    assert {"data_version_drift", "replay_mismatch", "source_updated"} <= DATA_KINDS


def test_verify_main_takes_data(run: Run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    run.close()
    seen: dict[str, Any] = {}

    def fake_verify(run_dir: Any, **kw: Any) -> dict[str, Any]:
        seen.update(kw)
        return {"status": "COMPLETE", "problems": []}

    import vbt.verify as verify_mod

    monkeypatch.setattr(verify_mod, "verify_run", fake_verify)
    assert main([str(run.dir), "--data", "--backend", "inprocess", "--json"]) == 0
    assert seen["data"] is True and seen["backend"] == "inprocess"
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"

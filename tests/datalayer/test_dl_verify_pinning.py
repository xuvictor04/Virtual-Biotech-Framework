"""``vbt verify`` and pinning for the data layer (DATA_LAYER.md §15.2-15.5): ``empty_result_cited``
per its definition, ``degraded_run`` from ``MANIFEST.degraded``, record_claims and verify deciding
identically from the live and the rebuilt call index, the audit index's data fields, and
``pinned["data"]`` from a gateway stub and without a gateway."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vbt.audit.claims import load_claims
from vbt.audit.provenance import build_index, data_call_fields
from vbt.pinning import REDACTED, build_pinned_config, pinned_data
from vbt.session import Run, _upgrade_manifest
from vbt.verify import INTEGRITY_KINDS, degraded_run_problem, verify_run

TOOL = "mcp__target__get_target_info"
CENSOR = 'citable only as "not significant at padj 0.10, or not tested"'


def _data_call(run: Run, tuid: str, status: str | None, coverage: str | None = None, statement: str | None = None,
               **extra):
    run.trace("tool_start", agent="genomics-analyst", tool=TOOL, tool_use_id=tuid, input={"target_id": "PCSK9"})
    end = {"agent": "genomics-analyst", "tool": TOOL, "tool_use_id": tuid, "is_error": False, "duration_s": 0.1,
           "output": "{}", "output_chars": 2}
    if status is not None:
        end["result_status"] = status
        end["data_provenance"] = {"prov": f"dp_{tuid:0>12}", "status": status, "coverage": coverage,
                                  "coverage_statement": statement, "source": "open_targets@25.09",
                                  "tables": [{"name": "target", "fingerprint": "fp1:sha256:abc"}],
                                  "returned": 0, "total": 0, **extra}
    run.trace("tool_end", **end)


def _claim(cid, tuid, *, supports=None, text="No ChEMBL drug targets PCSK9"):
    ev = {"kind": "tool_call", "tool_use_id": tuid}
    if supports:
        ev["supports"] = supports
    return {"id": cid, "text": text, "evidence": [ev]}


def _kinds(report):
    return [p["kind"] for p in report["problems"]]


@pytest.fixture
def run(tmp_path):
    r = Run(tmp_path / "runs")
    yield r
    if not r._closed:
        r.close()


# --------------------------------------------------------------------------- empty_result_cited


@pytest.mark.parametrize("status,coverage,statement,supports,text,cited", [
    ("empty", "covered", None, "absence", "No drug", False),
    ("empty", "covered", None, None, "A drug", True),                       # presence on empty
    ("empty", "covered", None, "presence", "A drug", True),
    ("empty", "unknown", None, "absence", "No drug", True),                 # coverage not covered
    ("empty", "not_covered", None, "absence", "No drug", True),
    ("empty", "partial_unknown", None, "absence", "No drug", True),
    ("empty", "censored", CENSOR, "absence", "Not significant at padj 0.10, or not tested.", False),
    ("empty", "censored", CENSOR, "absence", "Not differentially expressed.", True),
    ("empty_unverified", "covered", None, "absence", "No drug", True),
    ("empty_unverified", None, None, None, "A drug", True),
    ("ok", None, None, None, "A drug", False),
    ("partial", None, None, "absence", "No drug", False),
    (None, None, None, None, "A drug", False),                               # old trace: no status
])
def test_empty_result_cited_per_definition(run, status, coverage, statement, supports, text, cited):
    _data_call(run, "t1", status, coverage, statement)
    r = run.record_claims([_claim("C1", "t1", supports=supports, text=text)], strict=False)
    assert r["ok"], r
    run.close()
    report = verify_run(run.dir)
    hits = [p for p in report["problems"] if p["kind"] == "empty_result_cited"]
    assert bool(hits) is cited, report["problems"]
    if cited:
        assert hits[0]["claim"] == "C1" and hits[0]["tool_use_id"] == "t1" and report["status"] == "INCOMPLETE"
        assert report["integrity"]["status"] == "passed"           # not an integrity failure
        assert "unresolved_evidence" not in _kinds(report)          # reported once, with its own kind
    else:
        assert "empty_result_cited" not in _kinds(report)


def test_new_kinds_are_not_integrity_kinds():
    assert "empty_result_cited" not in INTEGRITY_KINDS and "degraded_run" not in INTEGRITY_KINDS


# --------------------------------------------------------------------------- record_claims and verify agree


def test_verify_and_record_claims_agree_on_a_stored_absence_claim(run):
    _data_call(run, "t1", "empty", "covered", "every target is assessed")
    _data_call(run, "t2", "empty", "unknown")
    ok = run.record_claims([_claim("C1", "t1", supports="absence")])
    assert ok["ok"] and ok["claims"][0]["n_verified"] == 1, ok
    stored = load_claims(run.dir)[0]["evidence"][0]
    assert stored["evidence_status"] == "absence" and stored["supports"] == "absence"
    assert stored["prov"] == "dp_0000000000t1" and stored["coverage"] == "covered"
    assert stored["source"] == "open_targets" and stored["release"] == "25.09"
    assert stored["fingerprints"] == {"target": "fp1:sha256:abc"}
    # the same citation of an empty result with unknown coverage is refused by record_claims ...
    refused = run.record_claims([_claim("C2", "t2", supports="absence")])
    assert not refused["ok"] and "coverage 'unknown'" in refused["errors"][0]
    run.close()
    # ... and verify, rebuilding the index from the trace, accepts C1 exactly as record_claims did
    report = verify_run(run.dir)
    assert report["evidence"]["valid_claims"] == 1 and report["evidence"]["claims_without_verified_evidence"] == []
    assert not {"empty_result_cited", "unresolved_evidence", "invalid_claims"} & set(_kinds(report)), report
    claims = json.loads((run.dir / "evidence" / "claims.json").read_text())
    assert claims["stats"]["n_absence_evidence"] == 1


def test_live_and_rebuilt_call_indexes_carry_the_same_data_fields(run):
    _data_call(run, "t1", "partial", None, None, evidence_nature="literature_cooccurrence", leakage_risk=False)
    _data_call(run, "t2", None)
    run.trace("tool_start", agent="g", tool=TOOL, tool_use_id="t3", input={})
    run.trace("tool_end", agent="g", tool=TOOL, tool_use_id="t3", is_error=True, error_kind="not_found",
              output="Error: {}")
    prov = build_index(run.events(), run.dir)
    for tuid in ("t1", "t2", "t3"):
        live = run.tool_call_status(tuid)
        rebuilt = prov.calls[tuid]
        for key in ("error_kind", "prov", "coverage", "data_provenance"):
            assert live.get(key) == rebuilt.get(key), (tuid, key)
    assert run.tool_call_status("t1")["result_status"] == prov.calls["t1"]["result_status"] == "partial"
    # old traces and non-gateway tools read 'unknown' in the audit index (§15.2)
    assert prov.calls["t2"]["result_status"] == "unknown" and run.tool_call_status("t2").get("result_status") is None
    assert prov.calls["t3"]["error_kind"] == "not_found"
    rows = {c["tool_use_id"]: c for c in prov.to_dict()["tool_calls"]}
    assert rows["t1"]["data_provenance"]["evidence_nature"] == "literature_cooccurrence"


def test_data_call_fields():
    assert data_call_fields({"type": "tool_end"}) == {}
    got = data_call_fields({"result_status": "empty", "error_kind": None,
                            "data_provenance": {"prov": "dp_x", "coverage": "covered", "status": "empty"}})
    assert got == {"result_status": "empty", "prov": "dp_x", "coverage": "covered",
                   "data_provenance": {"prov": "dp_x", "coverage": "covered", "status": "empty"}}
    assert data_call_fields({"data_provenance": {"status": "ok"}})["result_status"] == "ok"


# --------------------------------------------------------------------------- degraded_run


def test_degraded_run_is_reported_from_the_manifest(run):
    run.mark_degraded({"open_targets": "Open Targets data unavailable"},
                      tools={"mcp__functional_genomics__query_drug_perturbation": "tahoe_100m.de_permissive missing"},
                      tables=["tahoe_100m.de_permissive"])
    degraded = json.loads((run.dir / "MANIFEST.json").read_text())["degraded"]
    assert set(degraded) == {"servers", "tools", "tables", "reason", "at"}
    assert degraded["tables"] == {"tahoe_100m.de_permissive": "missing reference data"}
    ev = [e for e in run.events() if e["type"] == "run_degraded"][-1]
    assert ev["tools"] == degraded["tools"] and ev["servers"] == degraded["servers"]
    run.close()
    report = verify_run(run.dir)
    hit = next(p for p in report["problems"] if p["kind"] == "degraded_run")
    assert hit["servers"] == ["open_targets"]
    assert hit["tools"] == ["mcp__functional_genomics__query_drug_perturbation"]
    assert report["status"] == "INCOMPLETE" and report["integrity"]["status"] == "passed"


def test_degraded_run_needs_servers_or_tools():
    assert degraded_run_problem(None) is None
    assert degraded_run_problem({"servers": {}, "tools": {}, "tables": {"t": "x"}}) is None
    assert degraded_run_problem({"servers": ["open_targets"]})["servers"] == ["open_targets"]
    assert degraded_run_problem({"tools": {"mcp__a__b": "x"}})["tools"] == ["mcp__a__b"]


def test_old_degraded_manifest_is_upgraded_with_defaults():
    m = _upgrade_manifest({"run_id": "r", "degraded": {"reason": "missing reference data",
                                                       "servers": {"open_targets": "x"}, "at": "t"}}, "r")
    assert m["degraded"] == {"reason": "missing reference data", "servers": {"open_targets": "x"}, "tools": {},
                             "tables": {}, "at": "t"}
    assert _upgrade_manifest({"run_id": "r"}, "r")["degraded"] is None
    assert _upgrade_manifest({"run_id": "r", "degraded": "garbage"}, "r")["degraded"] is None
    listed = _upgrade_manifest({"run_id": "r", "degraded": {"servers": ["a"], "reason": "why"}}, "r")["degraded"]
    assert listed["servers"] == {"a": "why"} and listed["at"] is None


def test_open_existing_upgrades_an_old_degraded_record(run):
    run.close()
    path = run.dir / "MANIFEST.json"
    m = json.loads(path.read_text())
    m["degraded"] = {"reason": "missing reference data", "servers": {"open_targets": "x"}, "at": "t"}
    path.write_text(json.dumps(m))
    again = Run.open_existing(run.dir)
    assert again.manifest["degraded"]["tools"] == {} and again.manifest["degraded"]["tables"] == {}
    again.close()


# --------------------------------------------------------------------------- pinned["data"]


class _StubGateway:
    mode = "enforce"

    def __init__(self, record=None, fail=False):
        self.record = record if record is not None else {
            "mode": "enforce", "profile": "safe", "gateway_version": "1.0", "catalog_sha256": "sha256:abc",
            "descriptors": {"open_targets": "sha256:d1"}, "overlays": {"target": "sha256:o1"},
            "plugins": {"format/parquet": "1.0"},
            "sources": {"open_targets": {"release": "25.09", "tables": {"target": "fp1:x"}}},
            "readiness": {"tools_not_ready": {}, "columns_not_ready": {}},
            "memory": {"server_limits_mb": {"target": 7168}, "containment": "rlimit_data"},
            "service": {"url": "http://user:secret@host:1/", "api_token": "tok"},
        }
        self.fail = fail

    def pinned(self):
        if self.fail:
            raise RuntimeError("catalog unreadable")
        return dict(self.record)


def _runtime(gateway=None, error=None):
    return SimpleNamespace(gateway=gateway, gateway_error=error, cso=None, agents={}, mcp=None, skill_roots=[],
                           read_roots=[], skill_hashes=None)


def test_pinned_data_from_a_gateway_stub_is_redacted_with_determinism_and_leakage(config):
    config["data"] = {"enabled": True, "leakage": {"ceiling": "2019/12/31"}}
    pinned = build_pinned_config(config, _runtime(_StubGateway()))
    data = pinned["data"]
    assert data["mode"] == "enforce" and data["catalog_sha256"] == "sha256:abc"
    assert data["sources"]["open_targets"]["tables"] == {"target": "fp1:x"}
    assert data["service"]["api_token"] == REDACTED and "secret" not in data["service"]["url"]
    assert data["determinism"] == {"hash_seed": None, "flags_stripped": []}
    assert data["leakage"] == {"ceiling": "2019/12/31"}
    # the gateway's own blocks win
    stub = _StubGateway({"mode": "observe", "determinism": {"hash_seed": 0, "flags_stripped": ["-E"]},
                         "leakage": {"ceiling": None}})
    data = pinned_data(config, _runtime(stub))
    assert data["determinism"] == {"hash_seed": 0, "flags_stripped": ["-E"]} and data["leakage"] == {"ceiling": None}
    assert data["mode"] == "observe" and data["enabled"] is True
    json.dumps(pinned)                                 # the record stays JSON


def test_pinned_data_fallbacks(config):
    config["data"] = {"enabled": True}
    assert pinned_data(config, _runtime()) == {"enabled": True, "mode": "enforce", "gateway": None,
                                               "provenance_dir": "logs/data_provenance"}
    out = pinned_data(config, _runtime(error="ImportError: gateway package not installed"))
    assert out["gateway"] is None and out["reason"].startswith("ImportError")
    out = pinned_data(config, _runtime(_StubGateway(fail=True)))
    assert out["gateway"] is None and "catalog unreadable" in out["error"] and out["mode"] == "enforce"
    config["data"] = {"enabled": False}
    assert pinned_data(config, _runtime()) == {"enabled": False, "mode": "off", "gateway": None,
                                               "provenance_dir": "logs/data_provenance"}
    config["data"] = {"gateway": {"mode": "observe"}}
    assert pinned_data(config, _runtime())["mode"] == "observe"
    assert pinned_data(config, object())["gateway"] is None    # a runtime without the attribute


async def test_session_pins_the_data_block(config, scripted_session):
    session = await scripted_session(config, {"cso": []})
    pinned = json.loads((Path(session.run.dir) / "inputs" / "config.json").read_text())
    assert pinned["data"]["gateway"] is None and "mode" in pinned["data"] and "enabled" in pinned["data"]
    await session.close()

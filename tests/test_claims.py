"""record_claims: strict validation of claim evidence (port of upstream validate_claims)."""

import copy
import json
from pathlib import Path

import pytest

from vbt.audit.claims import load_claims
from vbt.session import Run
from vbt.tools.base import ToolContext
from vbt.tools.provenance import provenance_tools

GA = "work/genomics-analyst/results/tables/l2g.csv"
SC = "work/single-cell-analyst/results/tables/celltype.csv"


def tool_call(run, agent, tool, tuid, input=None, *, is_error=False, output="ok", end=True, action=None):
    run.trace("tool_start", agent=agent, tool=tool, tool_use_id=tuid, input=input or {})
    if action:
        action()
    if end:
        run.trace("tool_end", agent=agent, tool=tool, tool_use_id=tuid, is_error=is_error,
                  duration_s=0.0, output=output)


@pytest.fixture
def run(tmp_path):
    r = Run(tmp_path / "runs", run_id="R1")
    for rel, text in ((GA, "gene,l2g\nEGFR,0.82\n"), (SC, "cell,mean\nmast,2.4\n"),
                      ("work/genomics-analyst/results/dup.csv", "a\n"),
                      ("work/single-cell-analyst/results/dup.csv", "b\n")):
        p = r.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    tool_call(r, "genomics-analyst", "mcp__genetics__l2g", "toolu_ok", {"gene": "EGFR"})
    tool_call(r, "genomics-analyst", "mcp__genetics__l2g", "toolu_err", {"gene": "X"}, is_error=True,
              output="Error: upstream 503")
    tool_call(r, "genomics-analyst", "mcp__genetics__slow", "toolu_pending", {}, end=False)
    yield r
    r.close()


def claim(**over):
    c = {"id": "C1", "text": "EGFR has L2G 0.82 in lung adenocarcinoma", "confidence": "strong",
         "evidence": [{"kind": "table", "path": GA}]}
    c.update(over)
    return c


def errors_of(r):
    assert r["ok"] is False, r
    return " | ".join(r["errors"])


def test_valid_claim_is_recorded_verified_with_hash_and_inferred_agent(run):
    r = run.record_claims([claim()], agent="cso", turn=1)
    assert r["ok"], r
    assert r["claims"] == [{"id": "C1", "n_evidence": 1, "n_verified": 1, "turn": 1}]
    data = json.loads((run.dir / "evidence" / "claims.json").read_text())
    assert set(data) == {"stats", "claims"}
    stored = data["claims"][0]
    ev = stored["evidence"][0]
    assert ev["path"] == GA and ev["verified"] is True and ev["evidence_status"] == "verified"
    assert len(ev["sha256"]) == 64
    assert stored["agent"] == "genomics-analyst"  # inferred from the evidence producer
    assert stored["n_verified"] == 1 and stored["turn"] == 1 and stored["filed_by"] == "cso"
    assert run.manifest["artifacts"][GA]["cited_by"] == ["C1"]
    assert json.loads((run.dir / "MANIFEST.json").read_text())["artifacts"][GA]["cited_by"] == ["C1"]


@pytest.mark.parametrize("path, why", [
    ("/etc/hostname", "outside this run"),
    ("logs/trace.jsonl", "harness record"),
    ("evidence/claims.json", "harness record"),
    ("MANIFEST.json", "harness record"),
    ("../../../../etc/hostname", "outside this run"),
    ("made_up_results.csv", "not a registered artifact"),
])
def test_evidence_outside_work_or_unknown_is_rejected(run, path, why):
    (run.dir / "evidence" / "claims.json").write_text("{}") if path == "evidence/claims.json" else None
    r = run.record_claims([claim(evidence=[{"kind": "artifact", "path": path}])])
    assert why in errors_of(r)
    assert not load_claims(run.dir)


def test_absolute_path_inside_the_run_is_stored_run_relative(run):
    r = run.record_claims([claim(evidence=[{"kind": "table", "path": str(run.dir / GA)}])])
    assert r["ok"], r
    assert load_claims(run.dir)[0]["evidence"][0]["path"] == GA


def test_basename_suffix_and_workspace_relative_paths_resolve(run):
    r = run.record_claims([
        claim(id="C1", evidence=[{"kind": "table", "path": "l2g.csv"}]),
        claim(id="C2", evidence=[{"kind": "table", "path": "tables/celltype.csv"}]),
        claim(id="C3", evidence=[{"kind": "table", "path": "./results/tables/celltype.csv"}]),
    ], agent="single-cell-analyst")
    assert r["ok"], r
    paths = {c["id"]: c["evidence"][0]["path"] for c in load_claims(run.dir)}
    assert paths == {"C1": GA, "C2": SC, "C3": SC}


def test_ambiguous_basename_is_rejected(run):
    msg = errors_of(run.record_claims([claim(evidence=[{"kind": "table", "path": "dup.csv"}])]))
    assert "ambiguous" in msg
    # a fully qualified path never resolves to another agent's file of the same name
    r = run.record_claims([claim(evidence=[{"kind": "table", "path": "work/genomics-analyst/missing/dup.csv"}])])
    assert r["ok"] and load_claims(run.dir)[0]["evidence"][0]["path"] == "work/genomics-analyst/results/dup.csv"


def test_tool_call_evidence_rules(run):
    ok = run.record_claims([claim(evidence=[{"kind": "tool_call", "tool_use_id": "toolu_ok"}])])
    assert ok["ok"], ok
    ev = load_claims(run.dir)[0]["evidence"][0]
    assert ev["tool_name"] == "mcp__genetics__l2g" and ev["agent"] == "genomics-analyst" and ev["verified"]
    assert "a failed query cannot support a finding" in errors_of(
        run.record_claims([claim(id="C2", evidence=[{"kind": "tool_call", "tool_use_id": "toolu_err"}])]))
    assert "unfinished tool call" in errors_of(
        run.record_claims([claim(id="C3", evidence=[{"kind": "tool_call", "tool_use_id": "toolu_pending"}])]))
    assert "does not appear in this run's trace" in errors_of(
        run.record_claims([claim(id="C4", evidence=[{"kind": "tool_call", "tool_use_id": "toolu_fake"}])]))


def test_record_claims_own_call_cannot_be_cited(run):
    """The record_claims call is pending while it runs, so it cannot cite itself."""
    run.trace("tool_start", agent="cso", tool="mcp__provenance__record_claims", tool_use_id="toolu_self", input={})
    r = run.record_claims([claim(evidence=[{"kind": "tool_call", "tool_use_id": "toolu_self"}])])
    assert "unfinished tool call 'toolu_self'" in errors_of(r)


def test_citations_are_external_and_format_checked(run):
    r = run.record_claims([
        claim(id="C1", evidence=[{"kind": "citation", "pmid": "31234567"}]),
        claim(id="C2", evidence=[{"kind": "citation", "doi": "https://doi.org/10.1038/s41586-020-2308-7"}]),
        claim(id="C3", evidence=[{"kind": "web", "url": "https://example.org/paper"}]),
        claim(id="C4", evidence=[{"pmid": "PMID: 12345"}]),  # kind inferred
    ])
    assert r["ok"], r
    assert any("not locally verifiable" in w for w in r["warnings"])
    by_id = {c["id"]: c for c in load_claims(run.dir)}
    for cid in ("C1", "C2", "C3", "C4"):
        ev = by_id[cid]["evidence"][0]
        assert ev["kind"] == "citation" and ev["verified"] is False and ev["evidence_status"] == "external"
        assert by_id[cid]["n_verified"] == 0
    assert by_id["C2"]["evidence"][0]["doi"] == "10.1038/s41586-020-2308-7"
    assert by_id["C4"]["evidence"][0]["pmid"] == "12345"
    assert "needs pmid, doi or url" in errors_of(run.record_claims([claim(id="C5", evidence=[{"kind": "citation"}])]))
    assert "not a valid PMID" in errors_of(
        run.record_claims([claim(id="C6", evidence=[{"kind": "citation", "pmid": "abc"}])]))
    assert "not a valid URL" in errors_of(
        run.record_claims([claim(id="C7", evidence=[{"kind": "citation", "url": "ftp:/x"}])]))


def test_unknown_kind_rejected_and_report_alias_accepted(run):
    assert "unknown kind" in errors_of(run.record_claims([claim(evidence=[{"kind": "vibes", "path": GA}])]))
    r = run.record_claims([claim(evidence=[{"kind": "report", "path": GA}])])
    assert r["ok"] and load_claims(run.dir)[0]["evidence"][0]["kind"] == "artifact"


def test_stale_refile_is_rejected(run):
    assert run.record_claims([claim()])["ok"]
    old_sha = load_claims(run.dir)[0]["evidence"][0]["sha256"]
    (run.dir / GA).write_text("gene,l2g\nEGFR,0.10\n")
    msg = errors_of(run.record_claims([claim(evidence=[{"kind": "table", "path": GA, "sha256": old_sha}])]))
    assert "review the updated artifact and refile" in msg
    # refiling without the stale hash re-validates against the current version
    assert run.record_claims([claim()])["ok"]
    assert load_claims(run.dir)[0]["evidence"][0]["sha256"] != old_sha


def test_duplicate_and_malformed_ids_rejected_and_nothing_recorded(run):
    assert "duplicate id" in errors_of(run.record_claims([claim(), claim()]))
    assert "must match" in errors_of(run.record_claims([claim(id="bad id!")]))
    assert "must cite at least one" in errors_of(run.record_claims([claim(evidence=[])]))
    assert "missing 'text'" in errors_of(run.record_claims([claim(text="  ")]))
    # one bad claim rejects the whole batch
    r = run.record_claims([claim(id="C1"), claim(id="C2", evidence=[{"kind": "table", "path": "/etc/passwd"}])])
    assert r["ok"] is False
    assert load_claims(run.dir) == []


def test_inputs_are_never_mutated(run):
    batch = [claim(evidence=[{"kind": "table", "path": "l2g.csv"}]),
             claim(id="C2", evidence=[{"kind": "tool_call", "tool_use_id": "toolu_ok"}])]
    before = copy.deepcopy(batch)
    assert run.record_claims(batch)["ok"]
    assert batch == before


def test_refile_replaces_and_turn_is_recorded(run):
    run.trace("turn_start", turn=2, prompt="follow-up")
    assert run.current_turn == 2
    assert run.record_claims([claim(text="v1")])["ok"]
    assert run.record_claims([claim(text="v2"), claim(id="C9")])["ok"]
    claims = load_claims(run.dir)
    assert [c["id"] for c in claims] == ["C1", "C9"]
    assert claims[0]["text"] == "v2" and claims[0]["turn"] == 2


def test_load_claims_reads_legacy_list_and_tolerates_garbage(tmp_path):
    d = tmp_path / "run"
    (d / "evidence").mkdir(parents=True)
    (d / "evidence" / "claims.json").write_text(json.dumps([{"id": "C1", "text": "t", "evidence": []}]))
    assert load_claims(d)[0]["id"] == "C1"
    (d / "evidence" / "claims.json").write_text("{not json")
    assert load_claims(d) == []
    assert load_claims(tmp_path / "nope") == []


class _FakeRuntime:
    def __init__(self):
        self.events = []
        self.agents = {"genomics-analyst": object(), "single-cell-analyst": object()}

    def emit(self, kind, **data):
        self.events.append((kind, data))


def _tool(name):
    return next(t for t in provenance_tools() if t.name == name)


async def test_provenance_tools_return_ok_false_payloads_and_emit_events(run):
    rt = _FakeRuntime()
    cso = ToolContext(agent="cso", run=run, runtime=rt, tool_call_id="toolu_rc")
    bad = await _tool("mcp__provenance__record_claims")(cso, {"claims": [claim(evidence=[{"path": "/etc/hostname"}])]})
    assert bad["ok"] is False and bad["errors"]
    good = await _tool("mcp__provenance__record_claims")(cso, {"claims": [claim()]})
    assert good["ok"], good
    assert ("claims_filed", {"ids": ["C1"], "n": 1}) in rt.events

    ga = ToolContext(agent="genomics-analyst", run=run, runtime=rt, tool_call_id="toolu_reg")
    reg = await _tool("mcp__provenance__register_artifact")(ga, {"path": "results/tables/l2g.csv",
                                                               "description": "L2G scores"})
    assert reg["ok"], reg
    assert reg["artifact"]["path"] == GA and reg["artifact"]["produced_by"] == "genomics-analyst"
    assert ("artifact_registered", {"path": GA, "agent": "genomics-analyst"}) in rt.events
    refused = await _tool("mcp__provenance__register_artifact")(ga, {"path": "logs/trace.jsonl",
                                                                   "description": "x"})
    assert refused["ok"] is False and "harness record" in refused["errors"][0]

    listed = await _tool("mcp__provenance__list_artifacts")(cso, {"agent": "genomics-analyst"})
    assert listed["ok"] and {a["path"] for a in listed["artifacts"]} == {GA, "work/genomics-analyst/results/dup.csv"}
    row = next(a for a in listed["artifacts"] if a["path"] == GA)
    assert row["cited_by"] == ["C1"] and row["description"] == "L2G scores" and row["bytes"] > 0

    plan = await _tool("mcp__provenance__write_plan")(cso, {"steps": [{"id": "s1", "agent": "nobody"}]})
    assert plan["ok"] is True and any("not in this run's roster" in w for w in plan["warnings"])
    bad_plan = await _tool("mcp__provenance__write_plan")(cso, {"steps": [{"id": "s1"}]})
    assert bad_plan == {"ok": False, "errors": ["step s1: missing 'agent'"], "warnings": []}


def test_tool_schemas_document_citation_fields_and_kinds():
    spec = _tool("mcp__provenance__record_claims").input_schema
    ev = spec["properties"]["claims"]["items"]["properties"]["evidence"]["items"]["properties"]
    assert {"pmid", "doi", "url", "tool_use_id", "path"} <= set(ev)
    assert ev["kind"]["enum"] == ["artifact", "figure", "table", "code", "tool_call", "citation"]


def test_validation_against_registry_after_files_change_on_disk(run, tmp_path):
    """A file written by a child process (no tool event) is discovered before validation."""
    p = run.dir / "work" / "genomics-analyst" / "results" / "figures" / "late.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG")
    r = run.record_claims([claim(evidence=[{"kind": "figure", "path": "late.png"}])])
    assert r["ok"], r
    assert Path(load_claims(run.dir)[0]["evidence"][0]["path"]).name == "late.png"

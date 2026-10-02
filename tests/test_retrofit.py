"""`vbt audit`: rebuild MANIFEST, provenance and reports from a run's trace and files."""

import argparse
import json
import os
import time

from vbt.audit.cli import add_audit_parsers
from vbt.audit.retrofit import audit_all, audit_run
from vbt.session import Run
from vbt.verify import verify_run

TABLE = "work/genomics-analyst/results/tables/l2g.csv"


def ev(f, **e):
    f.write(json.dumps(e) + "\n")


def crashed_run(tmp_path, run_id="CR1"):
    """A run killed mid-turn: trace and files on disk, no turn record, never closed."""
    run = Run(tmp_path / "runs", run_id=run_id)
    run.trace("turn_start", turn=1, prompt="Is EGFR a target?")
    run.trace("tool_start", agent="cso", tool="Task", tool_use_id="task1", input={"subagent_type": "genomics-analyst"})
    run.trace("delegation", agent="genomics-analyst", description="genetics", prompt="EGFR genetics",
              tool_use_id="task1")
    run.trace("agent_start", agent="genomics-analyst", depth=1)
    run.trace("tool_start", agent="genomics-analyst", tool="Write", tool_use_id="w1",
              input={"file_path": "results/tables/l2g.csv"})
    (run.dir / TABLE).parent.mkdir(parents=True, exist_ok=True)
    (run.dir / TABLE).write_text("gene,l2g\nEGFR,0.82\n")
    run.trace("tool_end", agent="genomics-analyst", tool="Write", tool_use_id="w1", is_error=False, output="ok")
    run.trace("agent_end", agent="genomics-analyst", depth=1, stop="end_turn", text="L2G 0.82")
    run.trace("tool_end", agent="cso", tool="Task", tool_use_id="task1", is_error=False, output="L2G 0.82")
    run.trace("agent_end", agent="cso", depth=0, stop="end_turn", text="EGFR has genetic support.")
    with open(run.dir / "logs" / "trace.jsonl", "a") as f:
        f.write('{"type": "turn_e')  # the process died here
    return run


def test_run_without_close_gets_reconstructed_manifest(tmp_path):
    run = crashed_run(tmp_path)
    src_manifest = (run.dir / "MANIFEST.json").read_text()
    res = audit_run(run.dir)
    out = tmp_path / "audits" / "CR1"
    assert res["run_dir"] == str(out) and not res["in_place"]
    assert (run.dir / "MANIFEST.json").read_text() == src_manifest  # the source is untouched
    assert not (run.dir / "README.md").exists()

    m = json.loads((out / "MANIFEST.json").read_text())
    assert m["status"] == "reconstructed" and m["schema"] == 2
    assert m["reconstructed"]["status_before"] == "in_progress"
    counts = m["reconstructed"]["attribution"]
    assert counts["tool_input"] + counts["tool_capture"] == 1 and sum(counts.values()) == 1
    art = m["artifacts"][TABLE]
    assert art["produced_by"] == "genomics-analyst" and art["tool_use_id"] == "w1" and art["sha256"]
    assert m["reconstructed"]["reconstructed_turns"] == [1]
    assert any(n.startswith("Attribution of 1 artifact(s)") for n in m["audit_notes"])
    assert m["execution"][0]["agent"] == "genomics-analyst"

    sr = json.loads((out / "session_report.json").read_text())
    assert sr["turns"][0]["status"] == "unfinished" and sr["turns"][0]["reconstructed"]
    assert sr["turns"][0]["response"] == "EGFR has genetic support."
    assert (out / "evidence" / "provenance.json").is_file()
    readme = (out / "README.md").read_text()
    assert "Reconstructed by `vbt audit`" in readme and "Attribution of 1 artifact(s)" in readme
    assert (out / "audit.html").is_file()

    v = verify_run(out)
    assert v["integrity"]["status"] == "passed", v["integrity"]  # hashes are consistent after the rebuild
    assert v["status"] == "INCOMPLETE" and "failed_turn" in {p["kind"] for p in v["problems"]}
    # a second retrofit replaces the earlier copy
    again = audit_run(run.dir)
    assert again["status"] == "reconstructed"


def test_in_place_and_attribution_order(tmp_path):
    root = tmp_path / "runs"
    d = root / "OLD1"
    (d / "logs").mkdir(parents=True)
    t0 = time.time() - 1000

    def touch(rel, text, at):
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        os.utime(p, (at, at))
        return p

    script = "import pandas as pd\ndf = pd.DataFrame()\ndf.to_csv('results/tables/scored.csv')\n"
    with open(d / "logs" / "trace.jsonl", "w") as f:
        ev(f, type="turn_start", turn=1, prompt="q", t=t0)
        ev(f, type="agent_start", agent="genomics-analyst", depth=1, t=t0 + 10)
        ev(f, type="agent_start", agent="single-cell-analyst", depth=1, t=t0 + 40)
        ev(f, type="tool_start", agent="genomics-analyst", tool="Write", tool_use_id="w1", t=t0 + 11,
           input={"file_path": "code/scripts/score.py", "content": script})
        ev(f, type="tool_end", agent="genomics-analyst", tool="Write", tool_use_id="w1", t=t0 + 12, is_error=False)
        ev(f, type="tool_start", agent="genomics-analyst", tool="mcp__genetics__gwas", tool_use_id="m1", t=t0 + 13,
           input={"gene": "EGFR"})
        ev(f, type="tool_end", agent="genomics-analyst", tool="mcp__genetics__gwas", tool_use_id="m1", t=t0 + 14,
           is_error=False, files_returned=[str(d / "work/_mcp/data/processed/gwas.csv")])
        ev(f, type="agent_end", agent="genomics-analyst", depth=1, t=t0 + 50, stop="end_turn")
        ev(f, type="agent_end", agent="single-cell-analyst", depth=1, t=t0 + 80, stop="end_turn")
        ev(f, type="turn_end", turn=1, t=t0 + 90, status="completed")
    touch("work/genomics-analyst/code/scripts/score.py", script, t0 + 12)
    touch("work/genomics-analyst/results/tables/scored.csv", "x\n", t0 + 20)  # script writer line
    touch("work/_mcp/data/processed/gwas.csv", "g\n", t0 + 14)                # returned by an MCP tool
    touch("work/_mcp/data/processed/alone.csv", "a\n", t0 + 20)               # only genomics was running
    touch("work/_mcp/data/processed/both.csv", "b\n", t0 + 45)                # both windows overlap
    touch("work/_mcp/data/processed/late.csv", "c\n", t0 + 500)               # nobody was running
    touch("work/single-cell-analyst/notes.txt", "n\n", t0 + 500)              # its own workspace only
    touch("stray.txt", "s\n", t0 + 20)                                        # outside work/

    res = audit_run(d, in_place=True)
    assert res["run_dir"] == str(d.resolve()) and res["status"] == "reconstructed"
    m = json.loads((d / "MANIFEST.json").read_text())
    arts = m["artifacts"]
    assert arts["work/genomics-analyst/code/scripts/score.py"]["attribution"] == "tool_input"
    scored = arts["work/genomics-analyst/results/tables/scored.csv"]
    assert scored["attribution"] == "script" and scored["created_by"] == "score.py:3"
    gwas = arts["work/_mcp/data/processed/gwas.csv"]
    assert gwas["attribution"] == "mcp_return" and gwas["produced_by"] == "genomics-analyst"
    alone = arts["work/_mcp/data/processed/alone.csv"]
    assert alone["attribution"] == "window" and alone["produced_by"] == "genomics-analyst"
    assert arts["work/_mcp/data/processed/both.csv"]["attribution"] == "ambiguous"
    assert arts["work/_mcp/data/processed/late.csv"]["attribution"] == "unattributed"
    assert arts["work/single-cell-analyst/notes.txt"]["attribution"] == "workspace"
    counts = m["reconstructed"]["attribution"]
    assert counts == {"tool_capture": 0, "tool_input": 1, "mcp_return": 1, "script": 1, "window": 1,
                      "ambiguous": 1, "workspace": 1, "unattributed": 1}
    assert [x["path"] for x in m["misplaced_files"]] == ["stray.txt"]
    assert m["completed"] and m["query"] == "q"
    notes = " ".join(m["audit_notes"])
    assert "1 ambiguous" in notes and "1 could not be attributed" in notes
    assert (d / "README.md").is_file() and "1 traced to a writer line" in (d / "README.md").read_text()


def test_claims_revalidated_non_strictly(tmp_path):
    run = crashed_run(tmp_path, "CR2")
    assert run.record_claims([{"id": "C1", "text": "EGFR L2G 0.82",
                               "evidence": [{"kind": "table", "path": TABLE}]}])["ok"]
    (run.dir / TABLE).unlink()  # a later step deleted the cited table
    res = audit_run(run.dir, in_place=True)
    claims = json.loads((run.dir / "evidence" / "claims.json").read_text())["claims"]
    assert [c["id"] for c in claims] == ["C1"]                     # kept on record...
    assert claims[0]["evidence"][0]["evidence_status"] == "unresolved"  # ...as unresolved
    m = json.loads((run.dir / "MANIFEST.json").read_text())
    assert [x["path"] for x in m["deleted_artifacts"]] == [TABLE]
    assert res["n_claims"] == 1


def test_finalised_runs_keep_their_status_and_audit_all(tmp_path, config, capsys):
    root = tmp_path / "runs"
    done = Run(root, run_id="DONE")
    done.finish_turn({"turn": 1, "prompt": "hello", "response": "hi", "status": "completed"})
    done.close()
    crashed_run(tmp_path, "CR3")
    results = audit_all(root, in_place=True)
    by_id = {r["run_id"]: r for r in results}
    assert by_id["DONE"]["status"] == "completed" and by_id["CR3"]["status"] == "reconstructed"
    assert json.loads((root / "INDEX.json").read_text())["n_runs"] == 2

    config["paths"]["runs_dir"] = str(root)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    add_audit_parsers(sub)
    args = p.parse_args(["audit", "CR3", "-o", str(tmp_path / "out")])
    assert args.handler(args, config) == 0
    out = capsys.readouterr().out
    assert "[ok] CR3: status reconstructed" in out and (tmp_path / "out" / "CR3" / "audit.html").is_file()
    args = p.parse_args(["audit"])
    assert args.handler(args, config) == 2

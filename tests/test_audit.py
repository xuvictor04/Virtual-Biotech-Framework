"""The crash-safe audit record: MANIFEST v2, incremental capture, attribution, guards."""

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import vbt.session as session_mod
from vbt.audit import storage
from vbt.providers.base import Usage
from vbt.session import CostLedger, Run
from vbt.verify import verify_run

SRC = Path(__file__).resolve().parents[1] / "src"


def tool_call(run, agent, tool, tuid, input=None, *, action=None, is_error=False, output="ok", **end):
    run.trace("tool_start", agent=agent, tool=tool, tool_use_id=tuid, input=input or {})
    if action:
        action()
    run.trace("tool_end", agent=agent, tool=tool, tool_use_id=tuid, is_error=is_error, duration_s=0.0,
              output=output, **end)


def write(run, rel, text="x\n"):
    p = run.dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def turn(n=1, **over):
    t = {"turn": n, "prompt": f"question {n}", "response": f"answer {n}", "status": "completed", "agents": [],
         "cost_usd": 0.0, "cumulative_cost_usd": 0.0}
    t.update(over)
    return t


def manifest(run):
    return json.loads((run.dir / "MANIFEST.json").read_text())


# ----------------------------------------------------------------- MANIFEST basics

def test_new_run_manifest_is_v2_and_in_progress(tmp_path):
    run = Run(tmp_path / "runs", run_id="R0", config={"provider": "mock"})
    m = manifest(run)
    assert m["schema"] == 2 and m["status"] == "in_progress" and m["completed"] is None
    for k in ("run_id", "created", "updated", "query", "agents", "turns", "cost_usd", "config", "artifacts",
              "harness_files", "execution", "interrupted_turns", "data_source_errors", "audit_errors",
              "misplaced_files"):
        assert k in m, k
    run.close()
    assert manifest(run)["status"] == "empty"  # zero turns


def test_runs_root_under_logs_dir_and_agent_logs_subdir_are_hashed(tmp_path):
    run = Run(tmp_path / "logs" / "runs", run_id="R1")
    a = "work/genomics-analyst/results/tables/x.csv"
    b = "work/x/logs/fit.csv"
    tool_call(run, "genomics-analyst", "Write", "t1", {"file_path": a}, action=lambda: write(run, a, "v1\n"))
    write(run, b, "fit\n")  # written outside any tool event: picked up by the turn rescan
    write(run, "logs/ignored.txt")
    write(run, "work/x/__pycache__/m.cpython-311.pyc")
    write(run, "work/x/.ipynb_checkpoints/n.ipynb")
    write(run, "work/x/results/out.csv.part")
    run.finish_turn(turn())
    arts = manifest(run)["artifacts"]
    assert set(arts) == {a, b}
    assert all(len(e["sha256"]) == 64 for e in arts.values())
    run.close()
    assert verify_run(run.dir)["integrity"]["status"] == "passed"
    (run.dir / a).write_text("tampered\n")
    (run.dir / b).write_text("tampered\n")
    v = verify_run(run.dir)
    assert v["status"] == "FAIL" and set(v["integrity"]["changed"]) == {a, b}


def test_unclosed_run_reports_unfinished_run(tmp_path):
    run = Run(tmp_path / "runs", run_id="R2")
    run.finish_turn(turn())
    v = verify_run(run.dir)
    assert v["status"] == "INCOMPLETE"
    assert "unfinished_run" in {p["kind"] for p in v["evidence"]["problems"]}
    run.close()
    assert verify_run(run.dir)["status"] == "COMPLETE"


def test_sigkill_after_turn_one_keeps_hashes_and_detects_tampering(tmp_path):
    if os.name != "posix":
        pytest.skip("SIGKILL simulation needs POSIX")
    runs = tmp_path / "runs"
    code = textwrap.dedent(f"""
        import os, signal, sys
        sys.path.insert(0, {str(SRC)!r})
        from vbt.session import Run
        run = Run({str(runs)!r}, run_id="K1")
        rel = "work/genomics-analyst/results/tables/t1.csv"
        run.trace("tool_start", agent="genomics-analyst", tool="Write", tool_use_id="w1", input={{"file_path": rel}})
        p = run.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("turn one\\n")
        run.trace("tool_end", agent="genomics-analyst", tool="Write", tool_use_id="w1", is_error=False)
        run.finish_turn({{"turn": 1, "prompt": "q1", "response": "a1", "status": "completed", "agents": []}})
        run.trace("turn_start", turn=2, prompt="q2")
        rel2 = "work/genomics-analyst/results/tables/t2.csv"
        run.trace("tool_start", agent="genomics-analyst", tool="Bash", tool_use_id="b2", input={{"command": "x"}})
        p2 = run.dir / rel2
        p2.write_text("turn two\\n")
        run.trace("tool_end", agent="genomics-analyst", tool="Bash", tool_use_id="b2", is_error=False)
        os.kill(os.getpid(), signal.SIGKILL)
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    run_dir = runs / "K1"
    m = json.loads((run_dir / "MANIFEST.json").read_text())
    assert m["status"] == "in_progress"
    arts = m["artifacts"]
    assert "work/genomics-analyst/results/tables/t1.csv" in arts  # saved by finish_turn
    assert "work/genomics-analyst/results/tables/t2.csv" in arts  # saved by the mid-turn capture
    v = verify_run(run_dir)
    assert v["status"] == "INCOMPLETE" and v["integrity"]["status"] == "passed"
    (run_dir / "work/genomics-analyst/results/tables/t1.csv").write_text("edited after the crash\n")
    v = verify_run(run_dir)
    assert v["status"] == "FAIL" and v["integrity"]["changed"] == ["work/genomics-analyst/results/tables/t1.csv"]


# ----------------------------------------------------------------- atomic writes

def test_atomic_writes_leave_no_tmp_files_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "d" / "f.json"
    storage.write_json_atomic(target, {"v": 1})

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(storage.os, "replace", boom)
    with pytest.raises(OSError):
        storage.write_json_atomic(target, {"v": 2})
    monkeypatch.undo()
    monkeypatch.setattr(storage.os, "fsync", boom)
    with pytest.raises(OSError):
        storage.write_text_atomic(target, "half")
    monkeypatch.undo()
    circular: list = []
    circular.append(circular)
    with pytest.raises(ValueError):
        storage.write_json_atomic(target, circular)
    assert sorted(p.name for p in target.parent.iterdir()) == ["f.json"]
    assert json.loads(target.read_text()) == {"v": 1}


def test_run_lock_is_reentrant_and_creates_lock_file(tmp_path):
    with storage.run_lock(tmp_path):
        with storage.run_lock(tmp_path):
            pass
    assert (tmp_path / ".audit.lock").exists()


# ----------------------------------------------------------------- attribution

def test_after_tool_attributes_mcp_output_to_calling_agent(tmp_path):
    run = Run(tmp_path / "runs", run_id="M1")
    rel = "work/_mcp/data/processed/2026-10-02/expression/il33_expr.parquet"
    full = run.dir / rel
    out = json.dumps({"summary": "saved", "output_path": str(full)})
    tool_call(run, "single-cell-analyst", "mcp__expression__get_expression", "toolu_mcp", {"gene": "IL33"},
              action=lambda: write(run, rel), output=out)
    e = run.manifest["artifacts"][rel]
    assert e["produced_by"] == "single-cell-analyst"
    assert e["created_by"] == "mcp__expression__get_expression" and e["tool_use_id"] == "toolu_mcp"
    # P1-style call: explicit after_tool with files_returned, capture from trace disabled
    run.audit_settings["capture_on_trace"] = False
    rel2 = "work/_mcp/data/processed/2026-10-02/genetics/l2g.parquet"
    run.trace("tool_start", agent="genomics-analyst", tool="mcp__genetics__l2g", tool_use_id="toolu_g", input={})
    write(run, rel2)
    ev = {"type": "tool_end", "agent": "genomics-analyst", "tool": "mcp__genetics__l2g", "tool_use_id": "toolu_g",
          "is_error": False, "output": "(spilled)", "files_returned": [str(run.dir / rel2)], "input": {}}
    run.trace(**ev)
    assert rel2 not in run.manifest["artifacts"]
    run.after_tool(ev)
    run.after_tool(ev)  # idempotent per tool_use_id
    assert run.manifest["artifacts"][rel2]["produced_by"] == "genomics-analyst"
    run.audit_settings["capture_on_trace"] = True
    run.finish_turn(turn())
    prov = json.loads((run.dir / "evidence" / "provenance.json").read_text())
    assert prov["attribution"][rel]["method"] == "mcp_return"
    assert prov["attribution"][rel]["agent"] == "single-cell-analyst"
    call = next(c for c in prov["tool_calls"] if c["tool_use_id"] == "toolu_mcp")
    assert call["files_returned"] == [rel] and call["agent"] == "single-cell-analyst"
    assert manifest(run)["artifacts"][rel]["produced_by"] == "single-cell-analyst"
    run.close()


def test_tool_capture_records_tool_use_id_created_by_and_script_line(tmp_path):
    run = Run(tmp_path / "runs", run_id="A1")
    ws = "work/single-cell-analyst"
    script = f"{ws}/code/scripts/01_expr.py"
    src = "import pandas as pd\ndf = make()\ndf.to_csv(f'{out}/il33_expr.csv')\n"
    tool_call(run, "single-cell-analyst", "Write", "w1", {"file_path": "code/scripts/01_expr.py", "content": src},
              action=lambda: write(run, script, src))
    tool_call(run, "single-cell-analyst", "Bash", "b1", {"command": "python code/scripts/01_expr.py"},
              action=lambda: write(run, f"{ws}/results/tables/il33_expr.csv", "gene,mean\n"))
    # a file written by another agent's concurrent process is owned by its directory, not the caller
    tool_call(run, "genomics-analyst", "Bash", "b2", {"command": "ls"},
              action=lambda: write(run, f"{ws}/results/tables/late.csv", "x\n"))
    arts = run.manifest["artifacts"]
    assert arts[script]["tool_use_id"] == "w1" and arts[script]["created_by"] == "Write"
    out = arts[f"{ws}/results/tables/il33_expr.csv"]
    assert out["tool_use_id"] == "b1" and out["created_by"] == "01_expr.py" and out["produced_by"] == "single-cell-analyst"
    late = arts[f"{ws}/results/tables/late.csv"]
    assert late["produced_by"] == "single-cell-analyst" and late["tool_use_id"] is None
    run.finish_turn(turn())
    arts = manifest(run)["artifacts"]
    assert arts[f"{ws}/results/tables/il33_expr.csv"]["created_by"] == "01_expr.py:3"
    run.close()


def test_misplaced_files_are_recorded_with_attribution(tmp_path):
    run = Run(tmp_path / "runs", run_id="X1")
    write(run, "report/chief_of_staff_brief.md", "brief")  # harness write before any call: fine
    tool_call(run, "genomics-analyst", "Bash", "b1", {"command": "echo {} > ../../evidence/claims.json"},
              action=lambda: write(run, "evidence/claims.json", "[]"))
    tool_call(run, "genomics-analyst", "Bash", "b2", {"command": "echo > junk.csv"},
              action=lambda: write(run, "junk.csv", "x"))
    mis = {m["path"]: m for m in run.manifest["misplaced_files"]}
    assert set(mis) == {"evidence/claims.json", "junk.csv"}
    assert mis["evidence/claims.json"]["reason"] == "harness_file_modified"
    assert mis["evidence/claims.json"]["agent"] == "genomics-analyst" and mis["evidence/claims.json"]["tool_use_id"] == "b1"
    assert mis["junk.csv"]["reason"] == "outside_work" and mis["junk.csv"]["tool"] == "Bash"
    write(run, "inputs/stray.txt")  # found by the turn rescan, no attribution
    run.finish_turn(turn())
    run.close()
    v = verify_run(run.dir)
    probs = [p for p in v["evidence"]["problems"] if p["kind"] == "misplaced_files"]
    assert {p["path"] for p in probs} == {"evidence/claims.json", "junk.csv", "inputs/stray.txt"}
    assert v["status"] == "INCOMPLETE"


def test_harness_record_deleted_or_rewritten_during_a_call_is_flagged(tmp_path):
    run = Run(tmp_path / "runs", run_id="X2")
    run.finish_turn(turn())
    tool_call(run, "a", "Bash", "b1", {"command": "rm ../../report/FINAL_REPORT.md"},
              action=lambda: (run.dir / "report" / "FINAL_REPORT.md").unlink())
    tool_call(run, "a", "Bash", "b2", {"command": "echo {} > ../../MANIFEST.json"},
              action=lambda: (run.dir / "MANIFEST.json").write_text("{}"))
    mis = {m["path"]: m for m in run.manifest["misplaced_files"]}
    assert mis["report/FINAL_REPORT.md"]["reason"] == "harness_file_deleted"
    assert mis["MANIFEST.json"]["reason"] == "harness_file_modified" and mis["MANIFEST.json"]["tool_use_id"] == "b2"
    assert manifest(run)["status"] == "in_progress"  # the harness rewrote its own record


def test_deleted_artifacts_leave_the_registry(tmp_path):
    run = Run(tmp_path / "runs", run_id="D1")
    rel = "work/a/results/tmp.csv"
    tool_call(run, "a", "Write", "w1", {"file_path": rel}, action=lambda: write(run, rel))
    assert rel in run.manifest["artifacts"]
    tool_call(run, "a", "Bash", "b1", {"command": "rm results/tmp.csv"}, action=lambda: (run.dir / rel).unlink())
    assert rel not in run.manifest["artifacts"]
    assert run.manifest["deleted_artifacts"][0]["path"] == rel
    run.finish_turn(turn())
    run.close()
    assert verify_run(run.dir)["integrity"]["missing"] == []


# ----------------------------------------------------------------- guards and tolerance

def test_audit_errors_are_captured_never_raised(tmp_path, monkeypatch):
    run = Run(tmp_path / "runs", run_id="G1")

    def broken(*a, **k):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(session_mod, "build_index", broken)
    run.finish_turn(turn())  # must not raise
    assert any("index exploded" in e for e in run.audit_errors)
    assert any("index exploded" in e for e in manifest(run)["audit_errors"])
    monkeypatch.undo()

    class BadFile:
        closed = False

        def write(self, s):
            raise OSError("trace disk gone")

        def flush(self):
            pass

        def close(self):
            pass

    real = run._trace
    run._trace = BadFile()
    run.trace("model_call", agent="cso")  # must not raise
    run._trace = real
    assert any("trace disk gone" in e for e in run.audit_errors)
    run.close()
    v = verify_run(run.dir)
    assert "audit_capture_error" in {p["kind"] for p in v["problems"]} and v["status"] == "INCOMPLETE"


def test_events_skip_malformed_lines(tmp_path):
    run = Run(tmp_path / "runs", run_id="T1")
    run.trace("turn_start", turn=1, prompt="q")
    with open(run.dir / "logs" / "trace.jsonl", "a") as f:
        f.write('{"type": "tool_start", "partial\n[1, 2]\n')
    run.trace("turn_end", turn=1)
    events = run.events()
    assert [e["type"] for e in events] == ["turn_start", "turn_end"]
    assert run.malformed_trace_lines == 2


# ----------------------------------------------------------------- turn records and reports

def test_finish_turn_writes_turn_records_and_reports(tmp_path):
    run = Run(tmp_path / "runs", run_id="F1")
    rel = "work/genomics-analyst/results/tables/l2g.csv"
    tool_call(run, "genomics-analyst", "Write", "w1", {"file_path": rel}, action=lambda: write(run, rel))
    assert run.record_claims([{"id": "C1", "text": "EGFR L2G 0.82",
                               "evidence": [{"kind": "table", "path": rel}]}])["ok"]
    run.finish_turn(turn(1, response="EGFR is supported[[claim:C1]] and[[claim:C7]].",
                         thinking_traces=["step one " * 400, {"text": "short thought", "chars": 13}],
                         subagent_traces=[{"agent": "genomics-analyst", "description": "genetics",
                                           "model_calls": 3, "tool_calls": 2, "cost_usd": 0.12,
                                           "transcript_path": "logs/agents/x.jsonl"}],
                         data_source_failures=[{"tool": "mcp__genetics__gwas", "tool_use_id": "t9",
                                                "error": "HTTP 503"}]))
    run.finish_turn(turn(2, prompt="follow-up\nwith two lines", response="plain", status="interrupted"))
    q = (run.dir / "inputs" / "query.txt").read_text()
    assert q == "--- turn 1 ---\nquestion 1\n\n--- turn 2 ---\nfollow-up\nwith two lines\n"
    tr = (run.dir / "logs" / "transcript.md").read_text()
    assert "### CSO Reasoning" in tr and "of 3,600 chars shown" in tr and "short thought" in tr
    assert "### Sub-agent traces" in tr and "**genomics-analyst** — genetics" in tr
    raw = (run.dir / "report" / "FINAL_REPORT.md").read_text()
    assert "[[claim:C1]]" in raw
    rendered = (run.dir / "report" / "FINAL_REPORT.rendered.md").read_text()
    assert "supported[1] and[C7?]." in rendered and "## Claims and evidence" in rendered
    m = manifest(run)
    assert m["query"] == "question 1" and m["turns"] == 2
    assert m["interrupted_turns"] == [2] and m["status"] == "interrupted"
    assert m["data_source_errors"][0]["tool_name"] == "mcp__genetics__gwas" and m["data_source_errors"][0]["turn"] == 1
    assert {"report/FINAL_REPORT.md", "session_report.json", "inputs/query.txt",
            "evidence/claims.json"} <= set(m["harness_files"])
    sr = json.loads((run.dir / "session_report.json").read_text())
    t1 = sr["turns"][0]
    assert t1["research"] is True and t1["claims_filed"] == ["C1"] and t1["dangling_claim_refs"] == ["C7"]
    run.close()
    assert manifest(run)["status"] == "interrupted"  # sticky
    assert manifest(run)["completed"]


def test_close_status_incomplete_for_failed_turn(tmp_path):
    run = Run(tmp_path / "runs", run_id="F2")
    run.finish_turn(turn(1, status="failed: RuntimeError: x"))
    run.close()
    assert manifest(run)["status"] == "incomplete"


def test_register_artifact_rules_and_list(tmp_path):
    run = Run(tmp_path / "runs", run_id="RG")
    rel = "work/genomics-analyst/results/tables/l2g.csv"
    write(run, rel)
    write(run, "work/single-cell-analyst/results/figures/umap.png")
    write(run, "logs/x.txt")
    ws = run.dir / "work" / "genomics-analyst"
    r = run.register_artifact("results/tables/l2g.csv", "L2G scores", "cso", workspace=run.dir)
    assert r["ok"] is False  # run-relative 'results/...' does not exist
    r = run.register_artifact("results/tables/l2g.csv", "L2G scores", "genomics-analyst", workspace=ws)
    assert r["ok"] and r["artifact"]["produced_by"] == "genomics-analyst"
    r = run.register_artifact(rel, "L2G (by CSO)", "cso", kind="table", workspace=run.dir)
    assert r["ok"] and r["artifact"]["produced_by"] == "genomics-analyst" and r["artifact"]["registered_by"] == "cso"
    assert "outside the run" in run.register_artifact("/etc/hostname", "x", "cso")["errors"][0]
    assert "harness record" in run.register_artifact("logs/x.txt", "x", "cso")["errors"][0]
    assert "No such file" in run.register_artifact("work/nope.csv", "x", "cso")["errors"][0]
    rows = run.list_artifacts()
    assert {r["path"] for r in rows} == {rel, "work/single-cell-analyst/results/figures/umap.png"}
    assert run.list_artifacts(kind="figure")[0]["description"] == "(unregistered)"
    assert [r["path"] for r in run.list_artifacts(agent="genomics-analyst")] == [rel]
    arts = json.loads((run.dir / "evidence" / "artifacts.json").read_text())
    assert arts[0]["path"] == rel and arts[0]["description"] == "L2G (by CSO)" and arts[0]["registered_by"] == "cso"
    run.close()


def test_set_config_and_open_existing_resume(tmp_path):
    run = Run(tmp_path / "runs", run_id="OE")
    run.set_config({"provider": "mock", "models": {"orchestrator": "m"}, "harness": {"git": "abc"}})
    assert json.loads((run.dir / "inputs" / "config.json").read_text())["harness"]["git"] == "abc"
    assert manifest(run)["config"]["provider"] == "mock"
    rel = "work/a/results/r.csv"
    tool_call(run, "a", "Write", "w1", {"file_path": rel}, action=lambda: write(run, rel))
    run.cost.add("cso", Usage(10, 5), 0.5)
    run.finish_turn(turn())
    run.close()
    with open(run.dir / "logs" / "trace.jsonl", "a") as f:
        f.write('{"type": "trunc')  # a crash left a partial last line
    again = Run.open_existing(run.dir)
    assert again.status == "in_progress" and len(again.turns) == 1 and again.current_turn == 2
    assert rel in again.manifest["artifacts"] and again.manifest["config"]["harness"]["git"] == "abc"
    assert again.cost.total_usd == pytest.approx(0.5) and again.cost.calls_by_agent["cso"] == 1
    assert again.tool_call_status("w1")["pending"] is False
    again.trace("turn_start", turn=2, prompt="q2")
    assert [e["type"] for e in again.events()][-1] == "turn_start"
    assert again.malformed_trace_lines == 1
    again.finish_turn(turn(2))
    again.close()
    assert manifest(again)["status"] == "completed" and manifest(again)["turns"] == 2


def test_open_existing_reads_v1_manifest(tmp_path):
    d = tmp_path / "old"
    (d / "work" / "a").mkdir(parents=True)
    (d / "work" / "a" / "x.csv").write_text("1")
    (d / "MANIFEST.json").write_text(json.dumps({"run_id": "old", "started": "2026-01-01",
                                                 "artifacts": {"work/a/x.csv": storage.sha256_file(d / "work/a/x.csv")}}))
    run = Run.open_existing(d, resume=False)
    assert run.manifest["schema"] == 2 and run.manifest["artifacts"]["work/a/x.csv"]["produced_by"] == "a"


def test_cost_ledger_add_extra_attributes_without_model_call():
    c = CostLedger()
    c.add("genomics-analyst", Usage(100, 10), 0.2)
    c.add_extra(agent="genomics-analyst", usd=0.03, label="web_search")
    c.add_extra(agent=None, usd=0.01, label="web_search")
    c.add_extra(agent="cso", usd=0.0, label="noop")
    c.extra_usd += 0.5  # legacy direct increments still count
    assert c.calls_by_agent["genomics-analyst"] == 1
    assert c.usd_by_agent["genomics-analyst"] == pytest.approx(0.23)
    assert c.total_usd == pytest.approx(0.74)
    r = c.report()
    assert r["extra_by_label"] == {"web_search": 0.04}
    assert r["agents"]["genomics-analyst"]["tool_usd"] == pytest.approx(0.03)
    assert "cso" not in r["agents"]
    assert CostLedger.from_report(r).total_usd == pytest.approx(0.74)

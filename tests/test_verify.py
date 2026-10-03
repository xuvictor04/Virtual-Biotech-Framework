"""verify_run: integrity vs evidence coverage, tolerant reads, and --rerun re-execution."""

import json
import subprocess
import sys

from vbt.audit.rerun import rerun_scripts
from vbt.audit.storage import sha256_file
from vbt.session import Run
from vbt.verify import CAVEAT, format_report, main, verify_run

REL = "work/genomics-analyst/results/tables/l2g.csv"


def tool_call(run, agent, tool, tuid, input=None, *, action=None, is_error=False, output="ok"):
    run.trace("tool_start", agent=agent, tool=tool, tool_use_id=tuid, input=input or {})
    if action:
        action()
    run.trace("tool_end", agent=agent, tool=tool, tool_use_id=tuid, is_error=is_error, duration_s=0.0, output=output)


def write(run, rel, text="x\n"):
    p = run.dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def dispatch(run, agent, tuid, body=None):
    run.trace("tool_start", agent="cso", tool="Task", tool_use_id=tuid, input={"subagent_type": agent})
    run.trace("delegation", parent="cso", agent=agent, description=agent, prompt="go", tool_use_id=tuid)
    run.trace("agent_start", agent=agent, depth=1)
    if body:
        body()
    run.trace("agent_end", agent=agent, depth=1, stop="end_turn")
    run.trace("tool_end", agent="cso", tool="Task", tool_use_id=tuid, is_error=False, output="report")


def research_run(tmp_path, run_id="V1", *, close=True):
    run = Run(tmp_path / "runs", run_id=run_id)
    run.trace("turn_start", turn=1, prompt="Is EGFR a target?")
    dispatch(run, "genomics-analyst", "task1", lambda: tool_call(
        run, "genomics-analyst", "Write", "w1", {"file_path": "results/tables/l2g.csv"},
        action=lambda: write(run, REL, "gene,l2g\nEGFR,0.82\n")))
    assert run.record_claims([{"id": "C1", "text": "EGFR L2G 0.82", "evidence": [{"kind": "table", "path": REL}]}])["ok"]
    run.trace("turn_end", turn=1)
    run.finish_turn({"turn": 1, "prompt": "Is EGFR a target?", "response": "Yes[[claim:C1]].",
                     "status": "completed", "agents": ["genomics-analyst"]})
    if close:
        run.close()
    return run


def kinds(report):
    return {p["kind"] for p in report["problems"]}


def test_complete_research_run_and_human_report(tmp_path, capsys):
    run = research_run(tmp_path)
    v = verify_run(run.dir)
    assert v["status"] == "COMPLETE", v["problems"]
    assert v["integrity"]["status"] == "passed" and v["integrity"]["checked"] >= 2
    assert v["evidence"]["status"] == "complete" and v["evidence"]["research_turns"] == [1]
    text = format_report(v)
    assert text.endswith(CAVEAT) and CAVEAT == "These checks do not verify scientific correctness."
    assert main([str(run.dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETE"
    assert main([str(run.dir)]) == 0
    assert capsys.readouterr().out.rstrip().endswith(CAVEAT)


def test_in_progress_and_interrupted_runs_are_never_complete(tmp_path):
    live = research_run(tmp_path, "V2", close=False)
    v = verify_run(live.dir)
    assert v["status"] == "INCOMPLETE" and "unfinished_run" in kinds(v)
    live.finish_turn({"turn": 2, "prompt": "more", "response": "partial", "status": "interrupted"})
    live.close()
    v = verify_run(live.dir)
    assert v["status"] == "INCOMPLETE" and "interrupted_turn" in kinds(v)
    assert [p["turn"] for p in v["problems"] if p["kind"] == "interrupted_turn"] == [2]


def test_forged_claim_without_hash_is_caught(tmp_path):
    run = research_run(tmp_path, "V3")
    path = run.dir / "evidence" / "claims.json"
    data = json.loads(path.read_text())
    data["claims"].append({"id": "C2", "text": "forged", "n_verified": 1,
                           "evidence": [{"kind": "table", "path": REL, "verified": True}]})
    path.write_text(json.dumps(data))
    v = verify_run(run.dir)
    assert v["status"] == "FAIL"  # claims.json no longer matches its recorded hash
    assert "evidence/claims.json" in v["integrity"]["harness_changed"]
    unresolved = [p["detail"] for p in v["problems"] if p["kind"] == "unresolved_evidence"]
    assert unresolved and "no recorded sha256" in unresolved[0]


def test_forged_tool_call_evidence_is_caught(tmp_path):
    run = research_run(tmp_path, "V3b")
    tool_call(run, "genomics-analyst", "mcp__x__q", "bad", is_error=True)
    path = run.dir / "evidence" / "claims.json"
    data = json.loads(path.read_text())
    data["claims"][0]["evidence"] = [{"kind": "tool_call", "tool_use_id": "bad", "verified": True}]
    path.write_text(json.dumps(data))
    v = verify_run(run.dir)
    assert any("failed query" in p["detail"] for p in v["problems"] if p["kind"] == "unresolved_evidence")


def test_follow_up_research_turn_without_anchors_is_flagged(tmp_path):
    run = research_run(tmp_path, "V4", close=False)
    run.trace("turn_start", turn=2, prompt="And in SCLC?")
    dispatch(run, "single-cell-analyst", "task2")
    run.trace("turn_end", turn=2)
    run.finish_turn({"turn": 2, "prompt": "And in SCLC?", "response": "Also expressed, see above.",
                     "status": "completed", "agents": ["single-cell-analyst"]})
    run.close()
    v = verify_run(run.dir)
    assert v["status"] == "INCOMPLETE"
    missing = [p for p in v["problems"] if p["kind"] == "missing_turn_claim_references"]
    assert [p["turn"] for p in missing] == [2]
    assert v["evidence"]["research_turns"] == [1, 2]


def test_chief_of_staff_only_turn_is_not_research(tmp_path):
    run = Run(tmp_path / "runs", run_id="V4b")
    run.trace("turn_start", turn=1, prompt="hello")
    run.trace("agent_start", agent="chief-of-staff", depth=1)  # orientation brief: not a Task delegation
    tool_call(run, "chief-of-staff", "WebSearch", "ws1", {"query": "x"})
    run.trace("agent_end", agent="chief-of-staff", depth=1, stop="end_turn")
    dispatch(run, "scientific-reviewer", "rev1")
    run.trace("turn_end", turn=1)
    run.finish_turn({"turn": 1, "prompt": "hello", "response": "Which subtype?", "status": "completed"})
    run.close()
    v = verify_run(run.dir)
    assert v["status"] == "COMPLETE", v["problems"]
    assert v["evidence"]["status"] == "not_required"


def test_research_without_claims_and_dangling_references(tmp_path):
    run = Run(tmp_path / "runs", run_id="V5")
    run.trace("turn_start", turn=1, prompt="q")
    tool_call(run, "cso", "mcp__genetics__l2g", "m1", {"gene": "EGFR"})
    run.finish_turn({"turn": 1, "prompt": "q", "response": "EGFR[[claim:C9]]", "status": "completed"})
    run.close()
    k = kinds(verify_run(run.dir))
    assert {"no_claims", "dangling_claim_reference", "dangling_turn_claim_references",
            "missing_turn_claim_references"} <= k


def test_data_source_failures_and_failed_turns_are_reported(tmp_path):
    run = research_run(tmp_path, "V6", close=False)
    run.finish_turn({"turn": 2, "prompt": "q2", "response": "x", "status": "budget_exceeded",
                     "data_source_failures": [{"tool": "mcp__genetics__gwas", "error": "HTTP 503"}]})
    run.close()
    v = verify_run(run.dir)
    ds = [p["detail"] for p in v["problems"] if p["kind"] == "data_source_unavailable"]
    assert ds == ["turn 2: mcp__genetics__gwas: HTTP 503"]
    assert "failed_turn" in kinds(v) and v["status"] == "INCOMPLETE"


def test_malformed_trace_line_is_tolerated(tmp_path):
    run = research_run(tmp_path, "V7")
    with open(run.dir / "logs" / "trace.jsonl", "a") as f:
        f.write('{"type": "tool_end", "tool_use_id": "w1"\n')
    v = verify_run(run.dir)
    assert v["status"] == "INCOMPLETE" and "unreadable_trace_lines" in kinds(v)


def test_missing_and_invalid_manifest_are_reported_not_raised(tmp_path):
    v = verify_run(tmp_path / "does-not-exist")
    assert v["status"] == "FAIL" and kinds(v) == {"no_manifest"}
    assert format_report(v).endswith(CAVEAT)
    d = tmp_path / "bad"
    d.mkdir()
    (d / "MANIFEST.json").write_text("{oops")
    v = verify_run(d)
    assert v["status"] == "FAIL" and kinds(v) == {"invalid_manifest"}
    (d / "MANIFEST.json").write_text(json.dumps({"run_id": "bad"}))  # no artifact map at all
    v = verify_run(d)
    assert "artifacts_not_hashed" in kinds(v) and v["status"] != "COMPLETE"


def test_v1_manifest_is_still_checked(tmp_path):
    d = tmp_path / "v1run"
    (d / "work" / "a").mkdir(parents=True)
    f = d / "work" / "a" / "t.csv"
    f.write_text("1\n")
    (d / "MANIFEST.json").write_text(json.dumps({"run_id": "v1run", "artifacts": {"work/a/t.csv": sha256_file(f)}}))
    assert verify_run(d)["integrity"]["status"] == "passed"
    f.write_text("2\n")
    v = verify_run(d)
    assert v["status"] == "FAIL" and v["integrity"]["changed"] == ["work/a/t.csv"]


def test_harness_record_tampering_fails(tmp_path):
    run = research_run(tmp_path, "V8")
    sr = run.dir / "session_report.json"
    sr.write_text(sr.read_text().replace("Yes", "No"))
    v = verify_run(run.dir)
    assert v["status"] == "FAIL" and v["integrity"]["harness_changed"] == ["session_report.json"]


def test_multi_specialist_turn_without_plan_is_a_warning(tmp_path):
    run = research_run(tmp_path, "V9", close=False)
    run.trace("turn_start", turn=2, prompt="q2")
    dispatch(run, "single-cell-analyst", "t2")
    dispatch(run, "fda-safety-officer", "t3")
    run.trace("turn_end", turn=2)
    run.finish_turn({"turn": 2, "prompt": "q2", "response": "R[[claim:C1]]", "status": "completed"})
    run.close()
    v = verify_run(run.dir)
    assert v["status"] == "COMPLETE", v["problems"]
    assert [w["kind"] for w in v["evidence"]["warnings"]] == ["no_plan_for_multi_specialist_turn"]


# ----------------------------------------------------------------- rerun

DET = "from pathlib import Path\nPath('results/tables').mkdir(parents=True, exist_ok=True)\n" \
      "Path('results/tables/det.csv').write_text('a,b\\n1,2\\n')\n"
NONDET = "import time\nfrom pathlib import Path\nPath('results/tables/nondet.csv').write_text(str(time.time_ns()))\n"


def _script_run(run, ws, name, src, tuid):
    write(run, f"{ws}/code/scripts/{name}", src)
    tool_call(run, "a", "Write", f"w_{tuid}", {"file_path": f"code/scripts/{name}"})

    def execute():
        subprocess.run([sys.executable, f"code/scripts/{name}"], cwd=run.dir / ws, check=True)

    tool_call(run, "a", "Bash", tuid, {"command": f"python code/scripts/{name}"}, action=execute)


def test_rerun_matches_deterministic_and_flags_nondeterministic_outputs(tmp_path):
    run = Run(tmp_path / "runs", run_id="RR")
    ws = "work/a"
    absolute = (f"from pathlib import Path\np = Path({str(run.dir)!r}) / 'work/a/results/tables/abs.csv'\n"
                "p.write_text('fixed\\n')\n")
    run.trace("turn_start", turn=1, prompt="q")
    _script_run(run, ws, "01_det.py", DET, "b1")
    _script_run(run, ws, "02_abs.py", absolute, "b2")
    _script_run(run, ws, "03_nondet.py", NONDET, "b3")
    run.trace("turn_end", turn=1)
    run.finish_turn({"turn": 1, "prompt": "q", "response": "r", "status": "completed"})
    run.close()
    arts = json.loads((run.dir / "MANIFEST.json").read_text())["artifacts"]
    assert arts[f"{ws}/results/tables/det.csv"]["tool_use_id"] == "b1"
    before = {p: p.stat().st_mtime_ns for p in run.dir.rglob("*") if p.is_file() and "logs" not in p.parts}

    rr = rerun_scripts(run.dir)
    assert rr["order"] == "trace" and rr["n_scripts"] == 3
    by = {s["script"].rsplit("/", 1)[-1]: s for s in rr["scripts"]}
    assert by["01_det.py"]["status"] == "matched"
    assert by["01_det.py"]["outputs_matched"] == [f"{ws}/results/tables/det.csv"]
    assert by["02_abs.py"]["status"] == "matched"  # the literal run-dir prefix was rewritten
    assert by["03_nondet.py"]["status"] == "differed"
    assert rr["original_modified"] == [] and rr["scratch_dir"] is None
    after = {p: p.stat().st_mtime_ns for p in run.dir.rglob("*") if p.is_file() and "logs" not in p.parts}
    assert after == before  # the original run was never written

    v = verify_run(run.dir, rerun=True)
    assert v["rerun"]["summary"] == {"matched": 2, "differed": 1}
    assert "rerun_differed" in kinds(v) and v["status"] == "INCOMPLETE"
    assert "Rerun:  2/3" in format_report(v)


def test_rerun_reports_failures_and_missing_outputs(tmp_path):
    run = Run(tmp_path / "runs", run_id="RF")
    ws = "work/a"
    _script_run(run, ws, "01_det.py", DET, "b1")
    # the script is later edited so that it no longer writes its recorded output, then breaks
    write(run, f"{ws}/code/scripts/01_det.py", "print('no output')\n")
    write(run, f"{ws}/code/scripts/02_fail.py", "raise SystemExit(3)\n")
    tool_call(run, "a", "Bash", "b2", {"command": "python code/scripts/02_fail.py"})
    run.finish_turn({"turn": 1, "prompt": "q", "response": "r", "status": "completed"})
    run.close()
    rr = rerun_scripts(run.dir, timeout=60)
    by = {s["script"].rsplit("/", 1)[-1]: s for s in rr["scripts"]}
    assert by["01_det.py"]["status"] == "missing"
    assert by["01_det.py"]["outputs_missing"] == [f"{ws}/results/tables/det.csv"]
    assert by["02_fail.py"]["status"] == "failed" and by["02_fail.py"]["returncode"] == 3
    assert rr["ok"] is False


def test_rerun_uses_the_bash_env_allow_list_and_redacts_failures(tmp_path, monkeypatch):
    secret = "sk-ant-test-0123456789abcdefSECRET"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-0123456789abcdef")
    run = Run(tmp_path / "runs", run_id="RS")
    ws = "work/a"
    probe = ("import os, sys\nfrom pathlib import Path\n"
             "Path('seen.txt').write_text(repr(sorted(k for k in os.environ if 'KEY' in k)))\n")
    leak = "import sys\nsys.stderr.write('token=' + sys.argv[1])\nraise SystemExit(1)\n"
    write(run, f"{ws}/code/scripts/01_probe.py", probe)
    tool_call(run, "a", "Bash", "b1", {"command": "python code/scripts/01_probe.py"})
    write(run, f"{ws}/code/scripts/02_leak.py", leak)
    tool_call(run, "a", "Bash", "b2", {"command": f"python code/scripts/02_leak.py {secret}"})
    run.finish_turn({"turn": 1, "prompt": "q", "response": "r", "status": "completed"})
    run.close()
    rr = rerun_scripts(run.dir, timeout=60, keep_scratch=True, passthrough=[])
    import shutil
    from pathlib import Path
    try:
        seen = (Path(rr["scratch_dir"]) / ws / "seen.txt").read_text()
    finally:
        shutil.rmtree(Path(rr["scratch_dir"]).parent, ignore_errors=True)
    assert seen == "[]"  # neither the provider key nor any other *_API_KEY reached the script
    by = {s["script"].rsplit("/", 1)[-1]: s for s in rr["scripts"]}
    assert by["02_leak.py"]["status"] == "failed"
    assert secret not in by["02_leak.py"]["detail"] and "[redacted:ANTHROPIC_API_KEY]" in by["02_leak.py"]["detail"]
    rr2 = rerun_scripts(run.dir, timeout=60, keep_scratch=True, passthrough=["NCBI_API_KEY"])
    try:
        seen = (Path(rr2["scratch_dir"]) / ws / "seen.txt").read_text()
    finally:
        shutil.rmtree(Path(rr2["scratch_dir"]).parent, ignore_errors=True)
    assert seen == "['NCBI_API_KEY']"

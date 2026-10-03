"""Plan validation, reconciliation against observed dispatches, and per-turn records."""

import json

from vbt.audit.plan import reconcile, render_plan_md, validate_plan
from vbt.session import Run


def _steps():
    return [
        {"id": "s1", "agent": "single-cell-analyst", "task": "expression", "depends_on": [],
         "expected_outputs": "il33_celltype_expression.csv"},
        {"id": "s2", "agent": "bio-pathways-ppi-analyst", "task": "pathways"},
        {"id": "s3", "agent": "fda-safety-officer", "task": "safety", "depends_on": "s1"},
        {"id": "s4", "agent": "scientific-reviewer", "task": "review", "depends_on": ["s1", "s2", "s3"]},
    ]


def test_valid_plan_order_parallel_groups_and_string_coercion():
    r = validate_plan(_steps(), goal="IL-33 safety")
    assert r.ok, r.errors
    assert r.order == ["s1", "s2", "s3", "s4"]
    assert r.parallel_groups == [["s1", "s2"], ["s3"], ["s4"]]
    s3 = next(s for s in r.plan["steps"] if s["id"] == "s3")
    assert s3["depends_on"] == ["s1"]  # a string is one id, not ['s', '1']
    s1 = r.plan["steps"][0]
    assert s1["expected_outputs"] == ["il33_celltype_expression.csv"]
    assert r.plan["goal"] == "IL-33 safety" and r.plan["valid_order"] == r.order
    d = r.as_dict()
    assert d == {"ok": True, "n_steps": 4, "order": r.order, "parallel_groups": r.parallel_groups, "warnings": []}


def test_plan_rejections_and_warnings():
    assert "missing 'agent'" in validate_plan([{"id": "s1"}]).errors[0]
    assert "depends on itself" in validate_plan([{"id": "s1", "agent": "a", "depends_on": "s1"}]).errors[0]
    assert "unknown step 'nope'" in validate_plan([{"id": "s1", "agent": "a", "depends_on": ["nope"]}]).errors[0]
    cyc = validate_plan([{"id": "a", "agent": "x", "depends_on": "b"}, {"id": "b", "agent": "y", "depends_on": "a"}])
    assert "cycle" in cyc.errors[0]
    assert "duplicate id" in validate_plan([{"id": "s1", "agent": "a"}, {"id": "s1", "agent": "b"}]).errors[0]
    assert "too long" in validate_plan([{"id": "s" * 33, "agent": "a"}]).errors[0]
    assert "non-empty list" in validate_plan([]).errors[0]
    w = validate_plan([{"id": "s1", "agent": "ghost"}], roster={"genomics-analyst"})
    assert w.ok and any("not in this run's roster" in x for x in w.warnings)
    assert any("no 'task'" in x for x in w.warnings)
    # a JSON-encoded steps string and a {goal, steps} dict are accepted
    assert validate_plan(json.dumps([{"id": "s1", "agent": "a", "task": "t"}])).ok
    assert validate_plan({"goal": "g", "steps": [{"id": "s1", "agent": "a", "task": "t"}]}).plan["goal"] == "g"


def test_reconcile_reports_every_deviation_kind():
    plan = validate_plan(_steps()).plan
    execution = [  # fda-safety-officer dispatched before its dependency; pathways never ran
        {"agent": "fda-safety-officer", "start_t": 1.0},
        {"agent": "single-cell-analyst", "start_t": 2.0},
        {"agent": "genomics-analyst", "start_t": 3.0},
        {"agent": "scientific-reviewer", "start_t": 4.0},
        {"agent": "chief-of-staff", "start_t": 0.5},  # orientation brief: never "unplanned"
    ]
    rec = reconcile(plan, execution, {"work/single-cell-analyst/results/tables/other.csv": {}})
    assert [d["agent"] for d in rec["not_run"]] == ["bio-pathways-ppi-analyst"]
    assert [d["agent"] for d in rec["unplanned"]] == ["genomics-analyst"]
    assert [d["step"] for d in rec["out_of_order"]] == ["s3"]
    assert [d["output"] for d in rec["missing_output"]] == ["il33_celltype_expression.csv"]
    assert rec["n_deviations"] == 4 and "4 deviation(s)" in rec["summary"]
    ok = reconcile(plan, execution, {"work/single-cell-analyst/results/tables/il33_celltype_expression.csv": {}})
    assert ok["missing_output"] == []
    assert reconcile(None, execution)["summary"] == "No plan was recorded."
    md = render_plan_md(plan, rec)
    assert "## The analysis plan" in md and "*out_of_order*" in md and "[s1, s2] → [s3] → [s4]" in md


def test_reconcile_matches_steps_to_dispatches_one_to_one():
    plan = validate_plan([
        {"id": "s1", "agent": "statistician", "task": "a"},
        {"id": "s2", "agent": "geneticist", "task": "b", "depends_on": "s1"},
        {"id": "s3", "agent": "statistician", "task": "c", "depends_on": "s2"},
    ]).plan
    seq = lambda *agents: [{"agent": a, "start_t": float(i)} for i, a in enumerate(agents)]  # noqa: E731
    exact = reconcile(plan, seq("statistician", "geneticist", "statistician"))
    assert exact["n_deviations"] == 0, exact["deviations"]
    # the repeat statistician step was skipped: s3 is not_run, nothing is out of order
    skipped = reconcile(plan, seq("statistician", "geneticist"))
    assert [d["step"] for d in skipped["not_run"]] == ["s3"] and skipped["out_of_order"] == []
    # geneticist ran before the statistician it depends on
    early = reconcile(plan, seq("geneticist", "statistician", "statistician"))
    assert [(d["step"], d["depends_on"]) for d in early["out_of_order"]] == [("s2", "s1")]
    assert early["not_run"] == []
    # a stored plan without valid_order is still matched in dependency order
    bare = {"steps": list(reversed(plan["steps"]))}
    assert reconcile(bare, seq("statistician", "geneticist", "statistician"))["n_deviations"] == 0


def test_run_write_plan_returns_order_and_keeps_history(tmp_path):
    run = Run(tmp_path / "runs", run_id="P1")
    r = run.write_plan("goal 1", _steps(), roster={"single-cell-analyst", "bio-pathways-ppi-analyst",
                                                    "fda-safety-officer", "scientific-reviewer"}, agent="cso")
    assert r["ok"] and r["n_steps"] == 4 and r["order"] == ["s1", "s2", "s3", "s4"]
    assert r["parallel_groups"] == [["s1", "s2"], ["s3"], ["s4"]] and r["warnings"] == []
    bad = run.write_plan("g", [{"id": "s1", "agent": "a", "depends_on": "s1"}])
    assert bad["ok"] is False and bad["errors"]
    r2 = run.write_plan("goal 2", [{"id": "only", "agent": "genomics-analyst", "task": "t"}])
    assert r2["ok"] and r2["version"] == 2
    doc = json.loads((run.dir / "inputs" / "plan.json").read_text())
    assert doc["goal"] == "goal 2" and [h["goal"] for h in doc["history"]] == ["goal 1", "goal 2"]
    assert doc["turn"] == 1 and doc["written_by"] is None
    run.close()


def _dispatch(run, agent, tuid, *, write=None):
    run.trace("delegation", parent="cso", agent=agent, description=agent, prompt="do it", tool_use_id=tuid)
    run.trace("agent_start", agent=agent, depth=1)
    if write:
        p = run.dir / write
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    run.trace("agent_end", agent=agent, depth=1, stop="end_turn", duration_s=0.1)


def test_finish_turn_writes_reconciliation_and_execution(tmp_path):
    run = Run(tmp_path / "runs", run_id="P2")
    run.trace("turn_start", turn=1, prompt="q")
    run.trace("tool_start", agent="cso", tool="mcp__provenance__write_plan", tool_use_id="t_plan", input={})
    assert run.write_plan("g", [
        {"id": "s1", "agent": "genomics-analyst", "task": "a", "expected_outputs": ["l2g.csv"]},
        {"id": "s2", "agent": "single-cell-analyst", "task": "b", "depends_on": ["s1"]},
    ])["ok"]
    run.trace("tool_end", agent="cso", tool="mcp__provenance__write_plan", tool_use_id="t_plan", is_error=False)
    _dispatch(run, "single-cell-analyst", "t1")
    _dispatch(run, "genomics-analyst", "t2", write="work/genomics-analyst/results/tables/other.csv")
    run.finish_turn({"turn": 1, "prompt": "q", "response": "r", "status": "completed",
                     "agents": ["single-cell-analyst", "genomics-analyst"]})
    rec = json.loads((run.dir / "report" / "plan_reconciliation.json").read_text())
    assert rec["has_plan"] and [d["kind"] for d in rec["deviations"]] == ["out_of_order", "missing_output"]
    assert (run.dir / "report" / "plan_reconciliation.md").exists()
    t = json.loads((run.dir / "session_report.json").read_text())["turns"][0]
    assert t["plan_reconciliation"]["n_deviations"] == 2
    assert t["plan_reconciliation"]["counts"]["out_of_order"] == 1
    m = json.loads((run.dir / "MANIFEST.json").read_text())
    ex = m["execution"]
    assert [e["agent"] for e in ex] == ["single-cell-analyst", "genomics-analyst"]
    assert ex[0]["tool_use_id"] == "t1" and ex[0]["status"] == "completed" and ex[0]["turn"] == 1
    assert {"agent", "agent_run_id", "tool_use_id", "start", "end", "duration_s", "status"} <= set(ex[0])
    assert m["plan_reconciliation"]["n_deviations"] == 2
    run.close()

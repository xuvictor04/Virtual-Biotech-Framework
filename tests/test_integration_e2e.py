"""End-to-end integration of every package (offline: scripted provider, mock profile).

* through the CLI entry point (vbt.cli.main): a multi-turn research run with
  parallel delegations, an enforced review, record_claims; then `vbt verify`
  (COMPLETE), README.md + audit.html, the run index, `vbt list`, `vbt export`;
* through open_session: an interrupted turn (cancel mid-delegation) recorded as
  interrupted, the next turn succeeds and verify reports the interruption;
* a CSO BulkDispatch job that runs past the per-turn cost cap;
* the preflight gate (session refusal, degraded runs, per-turn 'not_sent');
* the lead wiring: MCP bridge options, skill materialisation, ListTools, CLI
  subcommands (audit, doctor, web), web app construction.
"""

import asyncio
import importlib.util
import json
import zipfile

import pytest

from vbt import cli, preflight
from vbt.orchestrator import open_session
from vbt.preflight import TURN_NOT_SENT, CheckResult, DataReadinessError
from vbt.providers.mock import ScriptedProvider, call, reply, turn
from vbt.tools.base import ToolContext
from vbt.verify import verify_run

from conftest import open_scripted_session

TABLE = "work/genomics-analyst/results/tables/l2g.csv"
CLAIM = {"id": "C1", "text": "EGFR has an L2G score of 0.82 in lung adenocarcinoma.", "confidence": "high",
         "evidence": [{"kind": "table", "path": TABLE}]}


def research_rules():
    """Turn 1: two parallel specialists, the CSO tries to finish unreviewed (harness
    nudges), reviewer, claims, synthesis. Turn 2: a follow-up answered directly."""
    return {
        "cso": [
            reply("Dispatching genetics and expression in parallel.",
                  call("Task", subagent_type="genomics-analyst", description="genetics", prompt="EGFR genetics"),
                  call("Task", subagent_type="single-cell-analyst", description="expression",
                       prompt="EGFR expression")),
            reply("EGFR looks supported."),            # unreviewed -> review_enforced nudge
            reply(call("Task", subagent_type="scientific-reviewer", description="review",
                       prompt="Review the genetics and expression findings.")),
            reply(call("mcp__provenance__record_claims", claims=[CLAIM])),
            reply("EGFR is a well-supported target [[claim:C1]]."),
            reply("You're welcome; the evidence is in the run's audit.html."),
        ],
        "genomics-analyst": [
            reply(call("Write", file_path="results/tables/l2g.csv", content="gene,l2g\nEGFR,0.82\n")),
            reply("L2G score 0.82 for EGFR (results/tables/l2g.csv)."),
        ],
        "single-cell-analyst": [reply("EGFR is highest in alveolar type 2 cells.")],
        "scientific-reviewer": [reply("APPROVED: conclusions are supported by the L2G table.")],
    }


def _runs(tmp_path):
    return tmp_path / "runs"


def _main(tmp_path, *argv):
    return cli.main(["--profile", "mock", "--runs-dir", str(_runs(tmp_path)), "--no-mcp", *argv])


def _only_run(tmp_path):
    runs = [p for p in _runs(tmp_path).iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert len(runs) == 1, runs
    return runs[0]


# ---------------------------------------------------------------- CLI end to end

def test_cli_research_run_verify_reports_index_export(tmp_path, mock_provider, capsys):
    mock_provider.use(research_rules())
    code = _main(tmp_path, "run", "-q", "Is EGFR a good target in lung adenocarcinoma?", "Thanks!")
    out = capsys.readouterr().out
    assert code == 0, out          # strict mode: every turn completed and verify is COMPLETE
    run_dir = _only_run(tmp_path)

    # MANIFEST v2 and the session record
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["schema"] == 2 and manifest["status"] == "completed"
    assert TABLE in manifest["artifacts"]
    assert manifest["artifacts"][TABLE]["produced_by"] == "genomics-analyst"
    report = json.loads((run_dir / "session_report.json").read_text())
    t1, t2 = report["turns"]
    assert t1["status"] == t2["status"] == "completed"
    assert set(t1["agents"]) == {"genomics-analyst", "single-cell-analyst", "scientific-reviewer"}
    assert t1["reviewed"] is True
    query = (run_dir / "inputs" / "query.txt").read_text()
    assert "--- turn 1 ---" in query and "--- turn 2 ---" in query and "Thanks!" in query
    kinds = [json.loads(line)["type"] for line in (run_dir / "logs" / "trace.jsonl").read_text().splitlines()]
    assert "review_enforced" in kinds and kinds.count("delegation") == 3

    # claims, verify, reports
    claims = json.loads((run_dir / "evidence" / "claims.json").read_text())
    claims = claims.get("claims", claims) if isinstance(claims, dict) else claims
    assert [c["id"] for c in claims] == ["C1"]
    assert verify_run(run_dir)["status"] == "COMPLETE"
    readme = (run_dir / "README.md").read_text()
    assert run_dir.name in readme and "C1" in readme
    page = (run_dir / "audit.html").read_text()
    assert "<html" in page.lower() and "genomics-analyst" in page

    # run index, `vbt verify`, `vbt list`, `vbt show`, `vbt export`
    index = json.loads((_runs(tmp_path) / "INDEX.json").read_text())
    rows = index.get("runs", index) if isinstance(index, dict) else index
    assert any(r.get("run_id") == run_dir.name for r in rows)
    assert (_runs(tmp_path) / "INDEX.md").is_file()
    capsys.readouterr()
    assert _main(tmp_path, "verify", run_dir.name[-8:]) == 0
    assert "COMPLETE" in capsys.readouterr().out
    assert _main(tmp_path, "list", "--json") == 0
    assert run_dir.name in capsys.readouterr().out
    assert _main(tmp_path, "show", "latest", "--json") == 0
    capsys.readouterr()
    bundle = tmp_path / "bundle.zip"
    assert _main(tmp_path, "export", run_dir.name, "-o", str(bundle)) == 0
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
    for rel in ("MANIFEST.json", "README.md", "audit.html", TABLE, "evidence/claims.json"):
        assert f"{run_dir.name}/{rel}" in names, rel


def test_cli_strict_run_exits_nonzero_on_an_incomplete_turn(tmp_path, mock_provider, capsys):
    from vbt.providers.base import StopReason
    mock_provider.use({"cso": [turn("I can't help with that.", stop=StopReason.REFUSAL)]})
    assert _main(tmp_path, "run", "-q", "q") == 1
    out = capsys.readouterr().out
    assert "incomplete: refusal" in out and "verify INCOMPLETE" in out
    mock_provider.use({"cso": [turn("I can't help with that.", stop=StopReason.REFUSAL)]})
    assert _main(tmp_path, "run", "-q", "--no-strict", "q") == 0


# ---------------------------------------------------------------- interrupted turn

async def test_interrupted_turn_is_recorded_and_the_next_turn_succeeds(config):
    holder = {}

    def specialist(messages):
        holder["session"].cancel()  # Ctrl+C while the specialist works
        return reply(call("Read", file_path="nothing.txt"))

    session = await open_scripted_session(config, {
        "cso": [reply("Dispatching genetics.", call("Task", subagent_type="genomics-analyst",
                                                     description="genetics", prompt="EGFR genetics")),
                reply("Fresh answer after the interruption.")],
        "genomics-analyst": [specialist],
    })
    holder["session"] = session
    with pytest.raises(asyncio.CancelledError):
        await session.ask("first question")
    assert not session.busy
    assert await session.ask("second question") == "Fresh answer after the interruption."
    run_dir = session.run.dir
    await session.close()

    turns = json.loads((run_dir / "session_report.json").read_text())["turns"]
    assert [t["status"] for t in turns] == ["interrupted", "completed"]
    assert "Turn interrupted: cancelled by the user" in turns[0]["response"]
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["interrupted_turns"] == [1]
    v = verify_run(run_dir)
    assert v["status"] != "COMPLETE"
    assert "interrupted_turn" in {p["kind"] for p in v["problems"]}
    assert (run_dir / "README.md").is_file() and (run_dir / "audit.html").is_file()


# ---------------------------------------------------------------- bulk past the turn cap

def _write_bulk_inputs(run_dir):
    d = run_dir / "work" / "clinical-trialist" / "results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trials.csv").write_text("nct_id,phase\n" + "\n".join(f"NCT{k:08d},2" for k in range(6)) + "\n")
    (d / "schema.json").write_text(json.dumps({"type": "object", "properties": {
        "nct_id": {"type": "string"}, "verdict": {"type": "string", "enum": ["POSITIVE", "NEGATIVE"]}},
        "required": ["nct_id", "verdict"]}))


async def test_cso_bulk_dispatch_runs_past_the_per_turn_cap(config):
    config["bulk"] = {**(config.get("bulk") or {}), "dispatch_enabled": True}
    config["limits"]["max_turn_cost_usd"] = 1.5
    args = dict(subagent_type="clinical-trialist", items_path="work/clinical-trialist/results/trials.csv",
                prompt_template="Trial {nct_id} (phase {phase})",
                schema="work/clinical-trialist/results/schema.json", budget_usd=10, pilot_size=2, concurrency=2)
    cso = iter([reply(call("BulkDispatch", **args)),
                reply(call("BulkDispatch", **args, confirm=True)),
                reply("Bulk annotation started; I will report when it finishes.")])

    def script(agent, system, messages, tools):
        if agent == "clinical-trialist-bulk":
            nct = messages[0].text.split("Trial ")[1].split()[0]
            return turn(call("submit_result", nct_id=nct, verdict="POSITIVE"), cost_usd=0.5)
        if agent == "cso":
            return next(cso)
        return reply("Done.")

    session = await open_scripted_session(config, ScriptedProvider(script))
    assert "BulkDispatch" in [t.name for t in session.rt.tools_for(session.rt.cso)]
    _write_bulk_inputs(session.run.dir)
    out = await session.ask("Annotate the phase 2 trials.")
    assert out.startswith("Bulk annotation started")
    jobs = list(session.rt._bulk_jobs.values())
    assert len(jobs) == 1
    await asyncio.wait_for(jobs[0].task, 20)
    status = await session.rt.registry.get("BulkStatus")(
        ToolContext(agent="cso", run=session.run, runtime=session.rt), {"job_id": jobs[0].job_id})
    assert status["state"] == "completed"
    assert json.loads(jobs[0].summary_path.read_text())["state"] == "completed"
    summary = status["summary"]
    assert summary["completed"] == 4 and summary["skipped_existing"] == 2   # pilot items reused
    bulk_usd = session.run.cost.report()["agents"]["clinical-trialist-bulk"]["usd"]
    assert bulk_usd == pytest.approx(3.0) and bulk_usd > config["limits"]["max_turn_cost_usd"]
    rec = json.loads((session.run.dir / "session_report.json").read_text())["turns"][0]
    assert rec["status"] == "completed"
    await session.close()


# ---------------------------------------------------------------- preflight

def _fail_ready(monkeypatch, *, session=False, per_turn=False):
    """Make vbt.preflight.require_ready fail for the session and/or per-turn check."""
    def gate(config, *, per_turn: bool = False, provider=None, allow_missing_data: bool = False):
        if (per_turn and gate.per_turn) or (not per_turn and gate.session):
            raise DataReadinessError(f"Not ready: OPEN_TARGETS_DATA_PATH: not set. {TURN_NOT_SENT}")
        return []
    gate.per_turn, gate.session = per_turn, session
    monkeypatch.setattr(preflight, "require_ready", gate)


async def test_preflight_refuses_a_session_before_any_run_exists(config, monkeypatch):
    _fail_ready(monkeypatch, session=True)
    provider = ScriptedProvider.from_rules({"cso": [reply("never")]})
    with pytest.raises(DataReadinessError):
        await open_session(config, provider=provider, start_mcp=False)
    runs = config["paths"]["runs_dir"]
    from pathlib import Path
    assert not Path(runs).exists() or not [p for p in Path(runs).iterdir() if p.is_dir()]
    assert provider.calls == []
    # --skip-preflight (preflight.skip) bypasses the gate
    config["preflight"] = {"skip": True}
    session = await open_scripted_session(config, provider)
    await session.close()


async def test_per_turn_preflight_records_a_not_sent_turn(config, monkeypatch):
    session = await open_scripted_session(config, {"cso": [reply("Answer after the data came back.")]})
    _fail_ready(monkeypatch, per_turn=True)
    out = await session.ask("q1")
    assert TURN_NOT_SENT in out
    assert session.rt.provider.calls == []          # nothing reached the model
    rec = session.last_turn
    assert rec["status"] == "not_sent" and rec["turn"] == 1
    monkeypatch.undo()
    assert await session.ask("q2") == "Answer after the data came back."
    run_dir = session.run.dir
    await session.close()
    turns = json.loads((run_dir / "session_report.json").read_text())["turns"]
    assert [t["status"] for t in turns] == ["not_sent", "completed"]


async def test_allow_missing_data_marks_the_run_degraded_and_tells_the_agents(config, monkeypatch):
    seen = {}

    def gate(config, *, per_turn=False, provider=None, allow_missing_data=False):
        assert allow_missing_data is True
        return [CheckResult("OPEN_TARGETS_DATA_PATH set and exists", False, detail="not set", kind="data")]
    monkeypatch.setattr(preflight, "require_ready", gate)
    config["preflight"] = {"allow_missing_data": True}

    def script(agent, system, messages, tools):
        seen[agent] = system
        return reply("Answer without Open Targets.")

    session = await open_scripted_session(config, ScriptedProvider(script))
    assert session.rt.degraded_servers
    await session.ask("q")
    run_dir = session.run.dir
    await session.close()
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["degraded"]["servers"] == session.rt.degraded_servers
    assert manifest["config"]["preflight"]["allow_missing_data"] is True
    assert "(reference data)" in seen["cso"]               # the CSO is told what is missing


def test_cli_preflight_flags_and_refusal(tmp_path, mock_provider, monkeypatch, capsys):
    p = cli.build_parser()
    args = p.parse_args(["--profile", "mock", "--skip-preflight", "--allow-missing-data", "run", "q"])
    cfg = cli.build_config(args)
    assert cfg["preflight"] == {"skip": True, "allow_missing_data": True}
    assert cli.build_config(p.parse_args(["--profile", "mock", "run", "q"]))["preflight"] == \
        {"skip": False, "allow_missing_data": False}

    mock_provider.use({"cso": [reply("never")]})
    _fail_ready(monkeypatch, session=True)
    assert _main(tmp_path, "run", "q") == 2
    err = capsys.readouterr().err
    assert TURN_NOT_SENT in err and "vbt doctor" in err
    mock_provider.use({"cso": [reply("ok")]})
    assert _main(tmp_path, "--skip-preflight", "run", "-q", "q") == 0


# ---------------------------------------------------------------- lead wiring

async def test_runtime_wiring_mcp_skills_and_list_tools(config):
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=ScriptedProvider.from_rules({}), start_mcp=True)
    rt, run_dir = session.rt, session.run.dir
    # MCP bridge built with the run's log dir, the mcp.* options and the runtime's event hook
    assert rt.mcp is not None and rt.mcp.log_dir == run_dir / "logs" / "mcp"
    assert rt.mcp.options["max_restarts"] == config["mcp"]["max_restarts"]
    assert rt.mcp.on_event == rt._mcp_event
    # skills materialised into the run and pinned
    if rt.skill_hashes:
        assert (run_dir / ".claude" / "skills").is_dir()
        pinned = json.loads((run_dir / "inputs" / "config.json").read_text())
        assert pinned["skill_hashes"] == rt.skill_hashes
    # ListTools reports server status and degraded servers
    rt.set_degraded({"open_targets": "Open Targets data unavailable: not set"})
    inv = await rt.registry.get("ListTools")(ToolContext(agent="cso", run=session.run, runtime=rt), {})
    assert "servers" in inv and inv["unavailable_servers"]["open_targets"].startswith("Open Targets")
    await session.close()


def test_cli_registers_package_subcommands():
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.dest == "cmd")
    for name in ("chat", "run", "replay", "tools", "verify", "list", "index", "export", "audit", "show",
                 "doctor", "web", "bulk", "case1", "scenario"):
        assert name in sub.choices, name
    args = p.parse_args(["doctor", "--smoke", "--analysis"])
    assert args.handler is preflight._doctor_handler
    args = p.parse_args(["verify", "latest", "--rerun"])
    assert args.rerun and callable(args.handler)


@pytest.mark.skipif(importlib.util.find_spec("starlette") is None or importlib.util.find_spec("httpx") is None,
                    reason="web extra not installed")
async def test_web_app_can_be_constructed_and_serves_the_page(config):
    import httpx

    from vbt.web import create_app

    app = create_app(config, password="pw-123456", start_mcp=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.get("/")
        assert r.status_code in (200, 302, 303, 401)
    await app.state.sessions.aclose()

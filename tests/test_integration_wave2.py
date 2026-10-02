"""Cross-package contracts after merging the agent-loop core (P1) and the
reports/run-index/web package (P8) onto the audit record (P4)."""

import pytest

import vbt.config
from vbt import verify
from vbt.audit.provenance import build_index
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, reply


async def test_delegated_runs_close_in_the_audit_index_and_report(config):
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", description="genetics", prompt="EGFR genetics"),
                      call("Task", subagent_type="single-cell-analyst", description="expr", prompt="EGFR expr")),
                reply("Done.")],
        "genomics-analyst": [reply(call("Bash", command='python3 -c "print(1)"')), reply("L2G 0.8")],
        "single-cell-analyst": [reply("expression high")],
    })
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    run_dir = session.run.dir
    await session.close()

    # P1 uses the Task tool_use_id as the sub-agent's agent_run_id; P4's index joins
    # delegation -> agent_start -> agent_end/delegation_end on it.
    runs = {r["agent"]: r for r in build_index(None, run_dir).runs.values()}
    for agent, prompt in (("genomics-analyst", "EGFR genetics"), ("single-cell-analyst", "EGFR expr")):
        r = runs[agent]
        assert r["status"] == "completed" and r["end"] is not None, r
        assert r["delegation_prompt"] == prompt
        assert (run_dir / r["transcript_path"]).is_file()      # logs/agents/<agent_run_id>.jsonl
    # P8's timeline draws a dashed bar only for runs with no agent_end.
    page = (run_dir / "audit.html").read_text()
    assert page.count('<rect class="bar') == 2 and 'class="bar open"' not in page
    assert session.run.audit_errors == []


def test_verify_main_reports_an_unknown_run_instead_of_crashing(config, monkeypatch, capsys):
    runs = vbt.config.resolve_path(config["paths"]["runs_dir"])
    (runs / "20260101_000000_aaaaaaaa").mkdir(parents=True)
    (runs / "20260101_000000_aaaaaaab").mkdir()
    monkeypatch.setattr(vbt.config, "load_config", lambda *a, **k: config)
    assert verify.main(["no-such-run"]) == 2               # vbt.audit.index.RunNotFound
    assert "no-such-run" in capsys.readouterr().err
    assert verify.main(["20260101_000000_aaaaaaa"]) == 2   # AmbiguousRunError
    with pytest.raises(LookupError):
        verify._resolve_run("no-such-run")

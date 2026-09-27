"""End-to-end orchestration with the scripted provider (no network)."""

import json

from vbt.orchestrator import CSOSession, open_session
from vbt.providers.mock import ScriptedProvider, call, reply


def _results(messages):
    """Tool-result texts in the last user message."""
    return [b.content for b in messages[-1].content if b.type == "tool_result"]


async def test_delegation_review_and_claims(config):
    def genomics(messages):
        return reply("L2G score 0.82 for EGFR in lung adenocarcinoma. Evidence: strong.")

    rules = {
        "cso": [
            reply("Dispatching two specialists in parallel.",
                  call("Task", subagent_type="genomics-analyst", description="genetics", prompt="EGFR genetics"),
                  call("Task", subagent_type="single-cell-analyst", description="expr", prompt="EGFR expression")),
            reply("EGFR is supported."),  # tries to finish without review -> harness nudges
            reply(call("Task", subagent_type="scientific-reviewer", description="review", prompt="review all")),
            reply("Final synthesis: EGFR is a strong target."),
        ],
        "genomics-analyst": [
            reply(call("Write", file_path="results/tables/l2g.csv", content="gene,l2g\nEGFR,0.82\n")),
            reply(call("mcp__provenance__register_artifact", path="work/genomics-analyst/results/tables/l2g.csv",
                       description="L2G scores")),
            genomics,
        ],
        "single-cell-analyst": [reply("EGFR highest in alveolar type 2 cells.")],
        "scientific-reviewer": [reply("APPROVED: conclusions supported.")],
    }
    provider = ScriptedProvider.from_rules(rules)
    session = await open_session(config, provider=provider, start_mcp=False)
    out = await session.ask("Is EGFR a good target in lung cancer?")
    assert out == "Final synthesis: EGFR is a strong target."

    events = session.run.events()
    kinds = [e["type"] for e in events]
    assert kinds.count("delegation") == 3
    assert "review_enforced" in kinds
    arts = json.loads((session.run.dir / "evidence" / "artifacts.json").read_text())
    assert arts[0]["path"] == "work/genomics-analyst/results/tables/l2g.csv"
    report = json.loads((session.run.dir / "session_report.json").read_text())
    assert set(report["turns"][0]["agents"]) == {"genomics-analyst", "single-cell-analyst", "scientific-reviewer"}
    await session.close()
    assert (session.run.dir / "MANIFEST.json").exists()


async def test_cso_cannot_use_data_tools_and_scientists_cannot_delegate(config):
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Bash", command="echo hi"),
                      call("Task", subagent_type="genomics-analyst", prompt="x")), reply("ok")],
        "genomics-analyst": [reply(call("Task", subagent_type="single-cell-analyst", prompt="y")), reply("done")],
    })
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    ends = [e for e in session.run.events() if e["type"] == "tool_end"]
    by_tool = {(e["agent"], e["tool"]): e for e in ends}
    assert by_tool[("cso", "Bash")]["is_error"]
    assert by_tool[("genomics-analyst", "Task")]["is_error"]
    await session.close()


async def test_orientation_runs_brief_and_clarification(config):
    config["orchestration"]["strategic_orientation"] = True
    provider = ScriptedProvider.from_rules({
        "chief-of-staff": [reply("BRIEF: B7-H3 is an emerging checkpoint target.")],
        "cso": [reply("1) Which lung cancer subtypes?"),
                lambda msgs: reply("Saw brief" if "BRIEF" in msgs[-1].text else "no brief")],
    })
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    first = await session.ask("Evaluate B7-H3 in lung cancer")
    assert "BRIEF" in first and "subtypes" in first
    second = await session.ask("LUAD and SCLC")
    assert second == "Saw brief"
    await session.close()


async def test_write_outside_run_is_blocked(config, tmp_path):
    provider = ScriptedProvider.from_rules({
        "cso": [reply(call("Task", subagent_type="genomics-analyst", prompt="x")), reply("ok")],
        "genomics-analyst": [reply(call("Write", file_path=str(tmp_path / "escape.txt"), content="x")),
                             reply("done")],
    })
    config["orchestration"]["enforce_review"] = False
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("q")
    assert not (tmp_path / "escape.txt").exists()
    await session.close()

"""Scenario replay and rubric scoring with the scripted provider."""

from vbt.case_studies import scenarios as sc
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, reply


def test_scenarios_parse():
    all_ = sc.list_scenarios()
    assert {"b7h3", "osmr", "trial_curation"} <= set(all_)
    assert all_["b7h3"]["profiles"] == ["no-web"]
    assert all(f["id"] and f["text"] for s in all_.values() for f in s["expected_findings"])


async def test_run_and_score(config):
    scen = sc.load_scenario("osmr")
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({"cso": [reply("OSMR is a partial blockade of gp130 signaling.")]})
    run, replies = await sc.run_scenario(config, scen, turns=1, provider=provider, start_mcp=False)
    assert replies == ["OSMR is a partial blockade of gp130 signaling."]

    verdicts = [{"id": f["id"], "verdict": "reproduced" if i == 0 else "absent", "evidence": "x"}
                for i, f in enumerate(scen["expected_findings"])]
    judge = ScriptedProvider.from_rules({"scenario-judge": [
        reply(call("submit_result", findings=verdicts, overall_comment="ok")), reply("done")]})
    session = await open_session(config, provider=judge, start_mcp=False)
    score = await sc.score_run(session.rt, run.dir, scen)
    await session.close()
    assert score["score"] == round(1 / len(verdicts), 3)
    assert (run.dir / "report" / "scenario_score.json").exists()

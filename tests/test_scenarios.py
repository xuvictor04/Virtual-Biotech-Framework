"""Scenario replay, rubric scoring, numeric checks and CLI config with the scripted provider."""

import argparse
import json
from pathlib import Path

import pytest

from vbt.case_studies import scenario_config
from vbt.case_studies import scenarios as sc
from vbt.orchestrator import open_session
from vbt.providers.mock import ScriptedProvider, call, reply


def test_scenarios_parse():
    all_ = sc.list_scenarios()
    assert {"b7h3", "osmr", "trial_curation"} <= set(all_)
    assert all_["b7h3"]["profiles"] == ["no-web"]
    assert all(f["id"] and f["text"] for s in all_.values() for f in s["expected_findings"])
    # paper numbers are encoded as structured expected values
    vals = {v["id"]: v["value"] for s in all_.values() for v in s.get("expected_values") or []}
    assert vals["fibroblast_log2fc_sclc"] == 2.13 and vals["fibroblast_log2fc_luad"] == 1.79
    assert vals["lr_interactions_sclc"] == 180 and vals["lr_interactions_luad"] == 226
    assert vals["os_hr"] == 1.62 and vals["dfs_hr"] == 2.06
    assert vals["stat1_stromal_types"] == 9 and vals["cross_disease_n"] == 112
    assert all_["b7h3"]["paper_cost_usd"] == 50.0 and all_["osmr"]["paper_cost_usd"] == 59.0


def test_turn_sources(tmp_path):
    scen = sc.load_scenario("b7h3")
    assert sc.scenario_turns(scen, "paper") == ["Evaluate the therapeutic candidacy of B7-H3 as a target in lung cancer."]
    assert len(sc.scenario_turns(scen, "synthetic")) == len(scen["turns"])
    with pytest.raises(ValueError):
        sc.scenario_turns(sc.load_scenario("trial_curation"), "paper")
    # a supplied transcript file supersedes the quoted turns
    transcript = tmp_path / "t.yaml"
    transcript.write_text("turns: [one, two]\n")
    src = tmp_path / "s.yaml"
    src.write_text(Path(sc.HERE / "osmr.yaml").read_text() + f"\npaper_turns_file: {transcript}\n")
    assert sc.scenario_turns(sc.load_scenario(str(src)), "paper") == ["one", "two"]


def _args(tmp_path, **kw):
    base = dict(profile=["mock"], model="claude-test-model", no_web=False, runs_dir=str(tmp_path / "myruns"),
                no_mcp=True, verbose=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_scenario_config_keeps_global_overrides(tmp_path):
    cfg = scenario_config(_args(tmp_path), sc.load_scenario("b7h3"))
    assert cfg["models"]["orchestrator"]["model"] == "claude-test-model"
    assert cfg["models"]["scientist"]["model"] == "claude-test-model"
    assert str(cfg["paths"]["runs_dir"]).endswith("myruns")
    assert cfg["web"]["enabled"] is False  # the scenario's no-web profile
    assert cfg["provider"]["name"] == "mock"


def test_scenario_cli_propagates_overrides(tmp_path, monkeypatch):
    from vbt import cli

    seen = {}

    async def fake_run(config, scenario, **kw):
        seen["config"], seen["kw"] = config, kw

        class R:
            dir = tmp_path

            class cost:
                total_usd = 0.0
        return R, []

    monkeypatch.setattr(sc, "run_scenario", fake_run)
    rc = cli.main(["--profile", "mock", "--model", "claude-x", "--runs-dir", str(tmp_path / "rr"), "--no-mcp",
                   "scenario", "run", "osmr", "--turns", "1", "--turns-source", "paper"])
    assert rc == 0
    assert seen["config"]["models"]["orchestrator"]["model"] == "claude-x"
    assert str(seen["config"]["paths"]["runs_dir"]).endswith("rr")
    assert seen["kw"]["turns_source"] == "paper" and seen["kw"]["turns"] == 1


def test_live_source_warnings(monkeypatch):
    import vbt.tools.builtin as b
    monkeypatch.setattr(b, "_unshare_available", lambda: True)
    scen = {"profiles": ["no-web"]}
    cfg = {"web": {"enabled": False}, "mcp_servers": {"servers": [{"name": "clinicaltrials"}, {"name": "pubmed"}]},
           "bash": {"network": True}}
    w = sc.live_source_warnings(cfg, scen)
    assert any("ClinicalTrials" in x for x in w) and any("PubMed" in x for x in w) and any("Bash" in x for x in w)
    cfg["web"]["literature_max_date"] = "2025/01/31"
    cfg["bash"]["network"] = False
    assert any("network isolation" in x for x in sc.live_source_warnings(cfg, scen))  # pattern block only
    cfg["bash"]["network_isolation"] = "unshare"
    assert len(sc.live_source_warnings(cfg, scen)) == 1
    assert sc.live_source_warnings({"web": {"enabled": True}}, {"profiles": []}) == []
    # a tool_env ceiling alone also counts: it is what the server enforces
    cfg2 = {"web": {"enabled": False}, "mcp_servers": {"servers": [{"name": "pubmed"}]},
            "bash": {"network": False, "network_isolation": "unshare"},
            "tool_env": {"VBT_LITERATURE_MAXDATE": "2025/01/31"}}
    assert sc.live_source_warnings(cfg2, scen) == []


def test_check_expected_values(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "claims.json").write_text(json.dumps({"stats": {}, "claims": [
        {"id": "c1", "text": "In SCLC, CD276 is up-regulated in fibroblasts (log2FC 2.05, padj 1e-6)."},
        {"id": "c2", "text": "In LUAD fibroblasts log2FC was 1.10 for 3,412 cells."},
        {"id": "c3", "text": "Overall survival HR 1.71 (95% CI 1.05-2.8)."},
    ]}))
    tables = tmp_path / "work" / "clinical-trialist" / "results" / "tables"
    tables.mkdir(parents=True)
    (tables / "survival_cox.csv").write_text("endpoint,hr,p\nOS,1.60,0.03\nDFS,2.10,0.02\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "cost_report.json").write_text(json.dumps({"total_usd": 25.0}))
    scen = sc.load_scenario("b7h3")
    rows = {r["id"]: r for r in sc.check_expected_values(tmp_path, scen)}
    assert rows["fibroblast_log2fc_sclc"]["found"] == 2.05 and rows["fibroblast_log2fc_sclc"]["within_tolerance"]
    assert rows["fibroblast_log2fc_luad"]["found"] == 1.10 and rows["fibroblast_log2fc_luad"]["within_tolerance"] is False
    assert rows["os_hr"]["found"] == 1.71 and rows["os_hr"]["within_tolerance"]
    assert rows["os_hr_table"]["found"] == 1.60 and rows["os_hr_table"]["where"].endswith("survival_cox.csv:hr")
    assert rows["lr_interactions_luad"]["found"] is None and rows["lr_interactions_luad"]["within_tolerance"] is None
    assert sc.cost_check(tmp_path, scen) == {"run_cost_usd": 25.0, "paper_cost_usd": 50.0, "ratio": 0.5}


async def test_run_and_score(config):
    scen = sc.load_scenario("osmr")
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({"cso": [reply("OSMR is a partial blockade of gp130 signaling.")]})
    run, replies = await sc.run_scenario(config, scen, turns=1, provider=provider, start_mcp=False)
    assert replies == ["OSMR is a partial blockade of gp130 signaling."]

    verdicts = [{"id": f["id"], "verdict": "reproduced" if i == 0 else "absent", "evidence": "x"}
                for i, f in enumerate(scen["expected_findings"])]
    judge = ScriptedProvider.from_rules({"scenario-judge": [
        reply(call("submit_result", findings=verdicts, overall_comment="ok"))]})
    session = await open_session(config, provider=judge, start_mcp=False)
    score = await sc.score_run(session.rt, run.dir, scen)
    await session.close()
    assert score["score"] == round(1 / len(verdicts), 3)
    assert "expected_values" in score and score["cost"]["paper_cost_usd"] == 59.0
    assert (run.dir / "report" / "scenario_score.json").exists()
    assert len(judge.calls) == 1  # submit_result is terminal


async def test_score_does_not_create_a_run(config):
    scen = sc.load_scenario("osmr")
    config["orchestration"]["enforce_review"] = False
    provider = ScriptedProvider.from_rules({"cso": [reply("done")]})
    run, _ = await sc.run_scenario(config, scen, turns=1, provider=provider, start_mcp=False)
    runs_dir = run.dir.parent
    before = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    verdicts = [{"id": f["id"], "verdict": "partial", "evidence": "x"} for f in scen["expected_findings"]]
    judge = ScriptedProvider.from_rules({"scenario-judge": [
        reply(call("submit_result", findings=verdicts, overall_comment="ok"))]})
    score = await sc.score_scenario_run(config, run.dir, scen, provider=judge)
    assert score["score"] == 0.5
    assert sorted(p.name for p in runs_dir.iterdir() if p.is_dir()) == before
    assert json.loads((run.dir / "report" / "scenario_score.json").read_text())["score"] == 0.5

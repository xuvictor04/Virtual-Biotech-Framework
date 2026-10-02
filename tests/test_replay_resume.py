"""--resume and replay with the scripted provider (offline)."""

import json

from vbt import cli, providers
from vbt.orchestrator import STATE_FILE, open_session
from vbt.providers.base import Message, ToolCall, message_to_dict
from vbt.providers.mock import ScriptedProvider, reply
from vbt.replay import CAVEAT, compare_runs, format_diff, parse_turns, replay_run


async def _two_turn_run(config):
    config["orchestration"]["enforce_review"] = False
    config["profiles"] = ["mock"]
    provider = ScriptedProvider.from_rules({"cso": [reply("Answer one."), reply("Answer two.")]})
    session = await open_session(config, provider=provider, start_mcp=False)
    await session.ask("first question")
    await session.ask("second\nquestion")
    await session.close()
    return session.run.dir


async def test_resume_restores_history_and_next_ask_sees_it(config):
    config["orchestration"]["enforce_review"] = False
    p1 = ScriptedProvider.from_rules({"cso": [reply("The answer is 42.")]})
    s1 = await open_session(config, provider=p1, start_mcp=False)
    await s1.ask("What is the answer?")
    await s1.close()
    run_dir = s1.run.dir

    def check(messages):
        texts = [m.text for m in messages]
        return reply("I remember 42." if "The answer is 42." in texts and "What is the answer?" in texts
                     else "amnesia")

    p2 = ScriptedProvider.from_rules({"cso": [check]})
    s2 = await open_session(config, provider=p2, start_mcp=False, resume=run_dir)
    assert s2.run.dir == run_dir and s2.turn == 1 and len(s2.history) == 2
    assert await s2.ask("Do you remember?") == "I remember 42."
    await s2.close()
    report = json.loads((run_dir / "session_report.json").read_text())
    assert [t["turn"] for t in report["turns"]] == [1, 2]
    assert (run_dir / "inputs" / "query.txt").read_text().count("--- turn") == 2
    types = [json.loads(line)["type"] for line in (run_dir / "logs" / "trace.jsonl").read_text().splitlines()]
    assert "session_resumed" in types
    manifest = json.loads((run_dir / "MANIFEST.json").read_text())
    assert manifest["status"] == "completed" and manifest["config"]["resumes"]


async def test_resume_repairs_a_dangling_tool_call(config):
    config["orchestration"]["enforce_review"] = False
    s1 = await open_session(config, provider=ScriptedProvider.from_rules({"cso": [reply("ok")]}), start_mcp=False)
    await s1.ask("q")
    await s1.close()
    state_path = s1.run.dir / STATE_FILE
    state = json.loads(state_path.read_text())
    # a crash right after the CSO asked for a tool: the call was never answered
    state["history"].append(message_to_dict(Message("user", [])))
    state["history"].append(message_to_dict(Message("assistant", [ToolCall("call_dangling", "Task", {})])))
    state_path.write_text(json.dumps(state))
    s2 = await open_session(config, provider=ScriptedProvider.from_rules({"cso": [reply("fine")]}),
                            start_mcp=False, resume=s1.run.dir)  # the strict mock rejects unpaired histories
    assert await s2.ask("again") == "fine"
    await s2.close()


def test_parse_turns_blocks_and_fallback(tmp_path):
    assert parse_turns("--- turn 1 ---\nfirst\n\n--- turn 2 ---\nmulti\nline\n") == ["first", "multi\nline"]
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "query.txt").write_text("")
    (run / "session_report.json").write_text(json.dumps({"turns": [{"turn": 2, "prompt": "b"},
                                                                    {"turn": 1, "prompt": "a"}]}))
    assert parse_turns(run) == ["a", "b"]
    assert parse_turns("legacy single query") == ["legacy single query"]


async def test_replay_of_a_mock_run_writes_replay_diff(config):
    src = await _two_turn_run(config)
    provider = ScriptedProvider.from_rules({"cso": [reply("Answer one."), reply("A different second answer.")]})
    new_dir, diff = await replay_run(src, provider=provider, start_mcp=False, quiet=True)
    assert new_dir != src and new_dir.parent == src.parent
    saved = json.loads((new_dir / "replay_diff.json").read_text())
    assert saved["original"]["run_id"] == src.name and saved["replay"]["run_id"] == new_dir.name
    assert [t["status_b"] for t in saved["turns"]] == ["completed", "completed"]
    assert saved["turns"][1]["prompt"] == "second\nquestion"
    assert saved["verify"]["original"] == saved["verify"]["replay"] == "COMPLETE"
    assert saved["caveat"] == CAVEAT
    text = format_diff(diff)
    assert "comparison, not a reproduction" in text and "vbt verify --rerun" in text
    cfg = json.loads((new_dir / "inputs" / "config.json").read_text())
    assert cfg["interface"] == "replay"


async def test_compare_runs_artifacts_and_claims(config):
    a = await _two_turn_run(config)
    b = await _two_turn_run(config)
    for d, content in ((a, "x,1\n"), (b, "x,1\n")):
        (d / "work" / "genomics-analyst" / "results" / "tables").mkdir(parents=True, exist_ok=True)
        (d / "work" / "genomics-analyst" / "results" / "tables" / "t.csv").write_text(content)
    (b / "work" / "genomics-analyst" / "only_b.txt").write_text("b")
    diff = compare_runs(a, b, verify=False)
    assert "work/genomics-analyst/results/tables/t.csv" in diff["artifacts"]["byte_identical"]
    assert diff["artifacts"]["only_replay"] == ["work/genomics-analyst/only_b.txt"]
    assert (b / "replay_diff.json").exists()


def test_cli_replay_command(config, monkeypatch, capsys, tmp_path):
    import asyncio

    src = asyncio.run(_two_turn_run(config))
    monkeypatch.setitem(providers._FACTORIES, "mock",
                        lambda **o: ScriptedProvider.from_rules({"cso": [reply("r1"), reply("r2")]}))
    code = cli.main(["--profile", "mock", "--runs-dir", str(src.parent), "--no-mcp", "replay", src.name, "--quiet"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"Replay of {src.name}" in out and "replay_diff.json" in out and "comparison, not a reproduction" in out
    assert cli.main(["--profile", "mock", "--runs-dir", str(src.parent), "replay", "no-such-run"]) == 2

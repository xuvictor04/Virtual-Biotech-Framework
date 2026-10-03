"""CLI: `vbt chat` and `vbt run` driven end to end with the scripted provider (offline)."""

import argparse
import builtins
import io
import json
from pathlib import Path

import pytest

from vbt import cli, orchestrator, providers
from vbt.orchestrator import NO_CLARIFY
from vbt.providers.mock import ScriptedProvider, call, fail, reply

ORIENT = "orchestration:\n  strategic_orientation: true\n  enforce_review: false\n"
NO_REVIEW = "orchestration:\n  enforce_review: false\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Install a scripted provider for the 'mock' profile and scripted stdin."""
    state = {"provider": None, "sessions": []}

    def use(rules, **kw):
        state["provider"] = ScriptedProvider.from_rules(rules, **kw)
        return state["provider"]

    monkeypatch.setitem(providers._FACTORIES, "mock", lambda **opts: state["provider"])

    def feed(lines):
        it = iter(lines)

        def fake_input(prompt=""):
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None
        monkeypatch.setattr(builtins, "input", fake_input)

    real_open = orchestrator.open_session

    async def capture_open(*a, **kw):
        s = await real_open(*a, **kw)
        state["sessions"].append(s)
        return s
    monkeypatch.setattr(orchestrator, "open_session", capture_open)

    def profile(text, name="p.yaml"):
        p = tmp_path / name
        p.write_text(text)
        return str(p)

    runs = tmp_path / "runs"

    def main(*argv, profiles=("mock",)):
        args = [x for p in profiles for x in ("--profile", p)] + ["--runs-dir", str(runs), *argv]
        return cli.main(args)

    state.update(use=use, feed=feed, profile=profile, main=main, runs=runs)
    return state


def _report(env):
    run_dir = env["sessions"][-1].run.dir
    return json.loads((run_dir / "session_report.json").read_text())


def test_chat_multiline_input_then_done(env, capsys):
    env["use"]({"cso": [lambda msgs: reply("got: " + msgs[-1].text)]})
    env["feed"](['"""', "line one", "line two", '"""', "/done"])
    assert env["main"]("chat", profiles=("mock", env["profile"](NO_REVIEW))) == 0
    out = capsys.readouterr().out
    assert "got: line one\nline two" in out
    rep = _report(env)
    assert rep["turns"][0]["prompt"] == "line one\nline two"


def test_chat_without_turns_and_eof_inside_multiline_is_safe(env, capsys):
    env["use"]({"cso": [lambda msgs: reply("partial: " + msgs[-1].text)]})
    env["feed"](["/help", "/summary", "/claims", "/evidence C9", '"""', "unterminated block"])
    assert env["main"]("chat") == 0
    out = capsys.readouterr().out
    assert "/evidence <ID>" in out and "No claims filed yet." in out and "No claim with id 'C9'" in out
    assert "partial: unterminated block" in out  # EOF submitted the collected lines


def test_chat_with_no_turns_at_all(env, capsys):
    env["use"]({})
    env["feed"]([])
    assert env["main"]("chat") == 0
    assert "Records:" in capsys.readouterr().out


def test_clarification_questions_and_briefing_are_printed_on_turn_one(env, capsys):
    env["use"]({"chief-of-staff": [reply("BRIEF: B7-H3 overview.")],
                "cso": [reply("1) Which subtypes: (a) LUAD (b) SCLC?")]})
    env["feed"](["Evaluate B7-H3 in lung cancer", "/done"])
    assert env["main"]("chat", profiles=("mock", env["profile"](ORIENT))) == 0
    out = capsys.readouterr().out
    assert "Which subtypes: (a) LUAD (b) SCLC?" in out
    assert "Chief of Staff briefing" in out and "BRIEF: B7-H3 overview." in out
    assert "chief-of-staff" in out  # agents used


def test_data_warning_is_printed(env, capsys):
    p = env["use"]({"chief-of-staff": [reply(call("WebSearch", query="B7-H3")), reply("BRIEF")],
                    "cso": [reply(NO_CLARIFY), reply("Answer.")]})

    async def broken(query, **kw):
        raise RuntimeError("quota exceeded")
    p.web_search = broken
    env["feed"](["Evaluate B7-H3", "/done"])
    assert env["main"]("chat", profiles=("mock", env["profile"](ORIENT))) == 0
    out = capsys.readouterr().out
    assert "Data/evidence warning: these tools failed during this turn: WebSearch." in out
    assert "unresolved data-source failures: WebSearch" in out


def test_interrupted_turn_returns_to_the_prompt(env, capsys):
    def specialist(messages):
        env["sessions"][-1].cancel()  # what the Ctrl+C handler does during a turn
        return reply("never seen")

    env["use"]({"cso": [reply("Dispatching.", call("Task", subagent_type="genomics-analyst", description="g",
                                                    prompt="x")),
                        reply("Second answer.")],
                "genomics-analyst": [specialist]})
    env["feed"](["first question", "second question", "/done"])
    assert env["main"]("chat", profiles=("mock", env["profile"](NO_REVIEW))) == 0
    out = capsys.readouterr().out
    assert "Turn interrupted" in out and "Second answer." in out
    turns = _report(env)["turns"]
    assert [t["status"] for t in turns] == ["interrupted", "completed"]


def test_sigint_handler_semantics():
    class FakeSession:
        busy = True
        cancelled = 0

        def cancel(self):
            self.cancelled += 1
            return True

    from rich.console import Console
    s = FakeSession()
    ctl = cli._Control(s, Console(file=io.StringIO()), loop=None)
    ctl.on_sigint()
    assert s.cancelled == 1 and ctl.turn_cancelled and not ctl.exit_requested
    s.busy = False
    ctl.last_sigint = float("-inf")  # outside the force-exit window
    ctl.on_sigint()
    assert ctl.exit_requested


def test_run_file_skips_comments_and_exit_codes(env, capsys, tmp_path):
    env["use"]({"cso": [lambda msgs: reply("echo: " + msgs[-1].text)] * 2})
    f = tmp_path / "turns.txt"
    f.write_text("# header comment\n\nfirst question\n   # indented comment\nsecond question\n")
    assert env["main"]("run", "-f", str(f)) == 0
    out = capsys.readouterr().out
    assert "echo: first question" in out and "echo: second question" in out
    assert "verify COMPLETE" in out and "claims filed: 0" in out
    assert [t["prompt"] for t in _report(env)["turns"]] == ["first question", "second question"]


def test_run_failed_turn_exit_codes(env, capsys):
    env["use"]({"cso": [fail(RuntimeError("model crashed"))]})
    assert env["main"]("run", "q") == 1
    out = capsys.readouterr().out
    assert "[ERROR] Turn 1 did not complete" in out and "Run directory:" in out
    env["use"]({"cso": [fail(RuntimeError("model crashed"))]})
    assert env["main"]("run", "--no-strict", "q") == 0
    assert env["main"]("run") == 2  # no queries


def test_run_quiet(env, capsys):
    env["use"]({"cso": [reply("streamed text")]})
    assert env["main"]("run", "-q", "q") == 0
    out = capsys.readouterr().out
    assert "streamed text" not in out and "status completed" in out


def test_run_events_ndjson(env, capsys):
    env["use"]({"cso": [reply("Hello [[claim:C1]] world")]})
    code = env["main"]("run", "--events", "ndjson", "--no-strict", "q")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    events = [json.loads(ln) for ln in lines]  # every stdout line is JSON
    kinds = [e["kind"] for e in events]
    assert "turn_start" in kinds and "text" in kinds and "turn_end" in kinds and kinds[-1] == "run_summary"
    assert events[-1]["exit_code"] == code


def test_model_alias_resolution(env, capsys):
    prof = env["profile"]("provider:\n  model_pattern: '^claude-[a-z0-9.-]+$'\n"
                          "model_aliases:\n  fast: claude-haiku-4-5\n", "aliases.yaml")
    env["use"]({"cso": [reply("ok")]})
    assert env["main"]("--model", "gpt-4o", "run", "q", profiles=("mock", prof)) == 2
    err = capsys.readouterr().err
    assert "unknown model 'gpt-4o'" in err and "fast -> claude-haiku-4-5" in err
    assert env["main"]("--model", "fast", "run", "q", profiles=("mock", prof)) == 0
    cfg = json.loads((env["sessions"][-1].run.dir / "inputs" / "config.json").read_text())
    assert cfg["models"]["orchestrator"]["model"] == "claude-haiku-4-5"


def test_build_config_propagates_scenario_overrides(tmp_path):
    args = argparse.Namespace(profile=["mock"], model="claude-x-1", no_web=True, runs_dir=str(tmp_path / "r"),
                              no_clarify=True)
    cfg = cli.build_config(args, extra_profiles=["no-web", "mock"])
    assert cfg["profiles"] == ["mock", "no-web"]
    assert cfg["web"]["enabled"] is False
    assert {cfg["models"][t]["model"] for t in ("orchestrator", "scientist", "bulk")} == {"claude-x-1"}
    assert cfg["models"]["support"]["model"] != "claude-x-1"
    assert Path(cfg["paths"]["runs_dir"]) == tmp_path / "r"
    assert cfg["orchestration"]["strategic_orientation"] is False


def test_footnotes_printed_after_turn(env, capsys):
    claim = {"id": "C1", "text": "EGFR is amplified.", "confidence": "high",
             "evidence": [{"kind": "citation", "pmid": "12345678"}]}
    env["use"]({"cso": [reply("EGFR is amplified [[claim:C1]].", call("mcp__provenance__record_claims",
                                                                       claims=[claim])),
                        reply("Done.")]})
    env["feed"](["q", "/claims", "/evidence C1", "/done"])
    assert env["main"]("chat", profiles=("mock", env["profile"](NO_REVIEW))) == 0
    out = capsys.readouterr().out
    assert "[[claim:C1]]" not in out  # anchors stripped while streaming
    assert "EGFR is amplified ." in out or "EGFR is amplified." in out
    assert "References" in out and "C1" in out
    assert "PMID 12345678" in out  # /evidence

"""Harness integration of the local model (L2): default switch, runtime guards,
bulk continuation, token budgets, reasoning clearing, readiness and pinning.

Offline: scripted providers, and a fake vLLM server (the threaded server of
tests/test_openai_compat.py) where the real OpenAI-compatible adapter is needed.
"""

import argparse
import asyncio
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
from pydantic import BaseModel

from vbt import budget
from vbt.agents import AgentDefinition
from vbt.budget import BudgetExceeded, CostScope, open_scope, usage_tokens
from vbt.bulk import SUBMIT_NUDGE, BulkItem, BulkRunner, add_budget_arguments, budget_kwargs
from vbt.config import load_config
from vbt.context import CLEARED_PREFIX, ContextManager, ContextPolicy
from vbt.providers.base import (
    Message,
    ModelSettings,
    ProviderCapabilities,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    Usage,
)
from vbt.providers.mock import ScriptedProvider, call, reply, turn
from vbt.runtime import (
    EMPTY_REPLY_MSG,
    FORCE_TOOL_MSG,
    TURN_LIMIT_MSG,
    tool_argument_problems,
)
from vbt.tools.base import Tool, schema

from conftest import open_scripted_session
from test_openai_compat import Fake, Handler, _clear_env, chunk, tc_delta, text_turn, usage

MODEL = "qwen3.8-27b"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class Recording(ScriptedProvider):
    """A scripted provider that records every request's settings and declares
    the capabilities of a local OpenAI-compatible server when asked to."""

    def __init__(self, script, *, tool_choice=True, replays=False, **kw):
        caps = ProviderCapabilities(images=True, documents=True, web_search=True, tool_choice=tool_choice,
                                    replays_reasoning=replays)
        super().__init__(script, capabilities=caps, **kw)
        self.seen: list[ModelSettings] = []

    async def complete(self, *, settings, **kw):
        self.seen.append(settings)
        return await super().complete(settings=settings, **kw)


def _agent(name="probe", tools=(), tier="scientist", max_turns=None):
    return AgentDefinition(name=name, description="probe", prompt="Answer.", tier=tier, tools=list(tools),
                           memory="none", max_turns=max_turns)


def _echo_tool(seen, **kw):
    def handler(ctx, a):
        seen.append(dict(a))
        return f"echo {a.get('q')}"
    return Tool("echo", "Echo the query.", schema({"q": {"type": "string"}, "n": {"type": "integer"},
                                                   "tags": {"type": "array", "items": {"type": "string"}}},
                                                  ["q"]), handler, **kw)


def _events(session, kind):
    return [e for e in session.run.events() if e.get("type") == kind]


@pytest.fixture
def fake(monkeypatch):
    _clear_env(monkeypatch)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    srv.fake = Fake()
    srv.fake.url = f"http://127.0.0.1:{srv.server_port}"
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv.fake
    srv.shutdown()
    srv.server_close()


def _local_config(tmp_path, base_url, **overrides):
    """The DEFAULT config (local vLLM) pointed at ``base_url``, offline otherwise."""
    from vbt.config import deep_merge
    over = {"paths": {"runs_dir": str(tmp_path / "runs")},
            "provider": {"options": {"base_url": base_url}},
            "mcp_servers_file": "configs/mcp_servers.none.yaml",
            "orchestration": {"strategic_orientation": False, "enforce_review": False},
            "web": {"search": {"searxng_url": None}}}
    return load_config([], overrides=deep_merge(over, overrides))


# ---------------------------------------------------------------------------
# 1. default switch and profiles
# ---------------------------------------------------------------------------


def test_default_config_is_the_local_h100_setup(monkeypatch):
    monkeypatch.delenv("VBT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    cfg = load_config([])
    h100 = load_config(["local-h100"])
    assert cfg["provider"] == h100["provider"]
    assert cfg["provider"]["name"] == "vllm"
    assert cfg["provider"]["options"]["base_url"] == "http://localhost:8000/v1"
    assert cfg["models"] == h100["models"]
    assert {t["model"] for t in cfg["models"].values()} == {MODEL}
    for key in ("max_parallel_agents", "max_turn_tokens", "max_item_tokens", "max_turn_cost_usd"):
        assert cfg["limits"][key] == h100["limits"][key], key
    assert cfg["bulk"]["default_concurrency"] == h100["bulk"]["default_concurrency"] == 32
    assert cfg["web"]["search"] == h100["web"]["search"] == {"backend": "auto",
                                                             "searxng_url": "http://localhost:8888"}
    for key in ("default_window_tokens", "soft_ratio", "hard_ratio"):
        assert cfg["context"][key] == h100["context"][key], key
    assert (cfg["context"]["default_window_tokens"], cfg["context"]["soft_ratio"],
            cfg["context"]["hard_ratio"]) == (262144, 0.55, 0.7)
    # every other section is still there
    for key in ("paths", "bash", "mcp", "audit", "web_ui", "orchestration", "retry", "preflight"):
        assert cfg.get(key), key
    assert cfg["model_aliases"]["qwen"] == MODEL


def test_paper_profile_is_the_claude_setup_without_local_options():
    cfg = load_config(["paper"])
    prov = cfg["provider"]
    assert prov["name"] == "anthropic"
    live = {k: v for k, v in prov["options"].items() if v is not None}
    assert live == {"refusal_fallback": True, "web_search_model": "claude-sonnet-4-5", "prompt_caching": True,
                    "max_retries": 4}
    m = cfg["models"]
    assert {m[t]["model"] for t in ("orchestrator", "scientist", "bulk")} == {"claude-sonnet-4-5"}
    assert m["support"]["model"] == "claude-haiku-4-5"
    assert (m["orchestrator"]["thinking_budget"], m["scientist"]["thinking_budget"],
            m["bulk"]["thinking_budget"]) == (32000, 16000, 8000)
    assert m["support"]["thinking_budget"] is None
    assert {t["context_window_tokens"] for t in m.values()} == {200000}
    assert cfg["limits"]["max_turn_tokens"] is None and cfg["limits"]["max_item_tokens"] is None
    assert (cfg["context"]["soft_ratio"], cfg["context"]["hard_ratio"]) == (0.7, 0.85)
    assert cfg["model_aliases"]["sonnet"] == "claude-sonnet-5"
    # the paper on top of claude gives the same provider and tiers
    both = load_config(["claude", "paper"])
    assert both["provider"]["name"] == "anthropic" and both["models"] == cfg["models"]


def test_mock_and_no_web_profiles_still_work():
    mock = load_config(["mock"])
    assert mock["provider"]["name"] == "mock"
    noweb = load_config(["no-web"])
    assert noweb["web"]["enabled"] is False and noweb["web"]["search"]["backend"] == "none"


def test_null_provider_options_are_not_passed_to_the_adapter(monkeypatch):
    from vbt.preflight import create_configured_provider, provider_options

    cfg = load_config(["claude"])
    opts = provider_options(cfg)
    assert "base_url" not in opts and "served_model_name" not in opts
    seen = {}

    def factory(**options):
        seen.update(options)
        return ScriptedProvider(lambda *a: reply("x"))

    from vbt import providers
    monkeypatch.setitem(providers._FACTORIES, "anthropic", factory)
    create_configured_provider(cfg)
    assert seen == {"refusal_fallback": True, "web_search_model": "claude-sonnet-5", "prompt_caching": True,
                    "max_retries": 4}


def test_resuming_a_pre_switch_claude_run_layers_the_claude_profile(monkeypatch):
    from vbt.cli import build_config

    monkeypatch.delenv("VBT_LLM_BASE_URL", raising=False)
    args = argparse.Namespace(profile=[], model=None, no_web=False, runs_dir=None, no_clarify=False,
                              skip_preflight=False, allow_missing_data=False)
    pinned = {"profiles": [], "provider": {"name": "anthropic", "options": {"api_key": "<redacted>"}},
              "models": {"orchestrator": {"model": "claude-opus-5", "effort": "high", "max_tokens": 64000,
                                          "thinking": True}}}
    cfg = build_config(args, pinned=pinned)
    assert cfg["profiles"] == ["claude"]
    assert cfg["provider"]["name"] == "anthropic"
    assert cfg["provider"]["options"].get("base_url") is None
    assert cfg["models"]["orchestrator"]["model"] == "claude-opus-5"
    # a local run resumes on the local default unchanged
    pinned_local = {"profiles": [], "provider": {"name": "vllm"}}
    assert build_config(args, pinned=pinned_local)["profiles"] == []


# ---------------------------------------------------------------------------
# 2. model-name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, ok", [("qwen3.8-27b", True), ("Qwen/Qwen3.8-27B-FP8", True),
                                       ("dsv4-flash", True), ("model.Q4_K_M.gguf", True), ("claude-x", True),
                                       ("bad name", False), ("-x", False), ("a;rm -rf", False)])
def test_any_served_model_name_is_accepted_for_local_providers(value, ok):
    from vbt.cli import ModelResolutionError, resolve_model
    from vbt.web.server import resolve_model as web_resolve

    cfg = load_config([])
    if ok:
        assert resolve_model(cfg, value) == value
        assert web_resolve(cfg, value) == value
    else:
        with pytest.raises(ModelResolutionError):
            resolve_model(cfg, value)
        with pytest.raises(ValueError):
            web_resolve(cfg, value)
    assert resolve_model(cfg, "qwen") == MODEL


def test_claude_pattern_applies_only_to_anthropic():
    from vbt.cli import ModelResolutionError, resolve_model
    from vbt.pinning import default_model_pattern

    claude = load_config(["claude"])
    assert resolve_model(claude, "claude-opus-9") == "claude-opus-9"
    with pytest.raises(ModelResolutionError):
        resolve_model(claude, "qwen3.8-27b-other")
    assert default_model_pattern("anthropic") == r"^claude-[a-z0-9.-]+$"
    for name in ("vllm", "sglang", "openai_compat", "llamacpp"):
        assert default_model_pattern(name) is not None
    assert default_model_pattern("mock") is None


def test_local_cli_is_registered(capsys):
    from vbt import cli

    assert cli.main(["local", "profiles", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "h100" in json.dumps(out)


# ---------------------------------------------------------------------------
# 3. runtime wiring: search backend, session key, tier extras, strict tools
# ---------------------------------------------------------------------------


async def test_search_backend_comes_from_the_resolver(config, tmp_path):
    from vbt.tools.search_backends import SearxNGBackend

    session = await open_scripted_session(config, lambda *a: reply("x"))
    rt = session.rt
    assert rt.search_backend == rt.provider.web_search          # mock: native search under 'auto'
    rt.config["web"]["search"] = {"backend": "searxng", "searxng_url": "http://127.0.0.1:9"}
    assert isinstance(rt.search_backend, SearxNGBackend)
    rt.config["web"]["search"] = {"backend": "none"}
    assert rt.search_backend is None
    await session.close()


async def test_session_key_and_tier_extras(config):
    provider = Recording(lambda a, s, m, t: reply("ok"))
    config["models"]["scientist"]["thinking_budget"] = 4321
    session = await open_scripted_session(config, provider)
    rt = session.rt
    r1 = await rt.run_agent(_agent(), "go", depth=1)
    await rt.run_agent(_agent(), "go", depth=1, session_key="fixed-key")
    s1, s2 = provider.seen
    assert s1.extra["session_key"] == r1.invocation_id
    assert s2.extra["session_key"] == "fixed-key"
    assert s1.extra["thinking_budget"] == 4321 and s1.extra["agent_name"] == "probe"
    # the CSO's key is stable across turns (one conversation, one replica)
    await session.ask("first")
    await session.ask("second")
    cso_keys = {s.extra["session_key"] for s in provider.seen if s.extra.get("agent_name") == "cso"}
    assert cso_keys == {f"{session.run.run_id}:cso"}
    await session.close()


async def test_strict_tools(config, tmp_path):
    session = await open_scripted_session(config, lambda *a: reply("x"))
    rt = session.rt
    assert rt.registry.get("mcp__provenance__record_claims").spec.strict is True
    assert rt.registry.get("Read").spec.strict is False
    seen_specs = {}

    def script(agent, system, messages, tools):
        seen_specs.update({t.name: t for t in tools})
        return reply(call("submit_result", value=1))

    class Out(BaseModel):
        value: int

    s2 = await open_scripted_session(config, script)
    await BulkRunner(s2.rt, _agent("bulk-probe", tier="bulk"), Out, tmp_path / "o.jsonl",
                     concurrency=1).run([BulkItem("a", "ITEM a")])
    assert seen_specs["submit_result"].strict is True
    await session.close()
    await s2.close()


# ---------------------------------------------------------------------------
# 4. invalid / mismatched tool arguments
# ---------------------------------------------------------------------------


async def test_invalid_json_arguments_are_not_run(config):
    seen: list = []
    bad = ToolCall("call_bad", "echo", {}, native={"invalid_arguments": '{"q": "x"', "error": "Expecting ','"})

    def script(agent, system, messages, tools):
        if len(messages) == 1:
            return reply(bad)
        return reply("done")

    session = await open_scripted_session(config, script)
    res = await session.rt.run_agent(_agent(), "go", depth=1, extra_tools=[_echo_tool(seen)])
    assert seen == []                                         # the tool never ran
    result = res.messages[2].content[0]
    assert isinstance(result, ToolResult) and result.is_error
    assert result.content == ("Your arguments for echo were not valid JSON (Expecting ','). Re-issue the call "
                              "with a single JSON object matching the schema.")
    end = [e for e in _events(session, "tool_end") if e.get("tool_use_id") == "call_bad"][0]
    assert end["model_error"] == "invalid_arguments" and end["is_error"] is True
    assert res.tool_errors[0]["model_error"] == "invalid_arguments"
    await session.close()


async def test_invalid_arguments_of_a_data_tool_are_a_model_error_not_a_data_failure(config):
    from vbt import failures

    calls = [ToolCall("c1", "mcp__genetics__query", {}, native={"invalid_arguments": "{", "error": "eof"}),
             ToolCall("c2", "mcp__genetics__query", {"gene": 5})]

    def script(agent, system, messages, tools):
        return reply(*calls) if len(messages) == 1 else reply("done")

    ran = []
    tool = Tool("mcp__genetics__query", "q", schema({"gene": {"type": "string"}}, ["gene"]),
                lambda ctx, a: ran.append(a) or "ok", source="mcp:genetics")
    session = await open_scripted_session(config, script)
    res = await session.rt.run_agent(_agent(), "go", depth=1, extra_tools=[tool])
    assert ran == []
    assert [e["model_error"] for e in res.tool_errors] == ["invalid_arguments", "schema_mismatch"]
    assert res.unresolved_data_failures == [] and res.other_errors == 2
    recs = failures.records_from_trace(session.run.events())
    assert failures.summarize(recs)["unresolved_data"] == []
    assert failures.data_failure_notice(failures.unresolved_failures(recs)) == ""
    await session.close()


async def test_schema_check_before_running_a_tool(config):
    seen: list = []
    script_calls = iter([
        reply(call("echo", n=3)),                        # missing required q
        reply(call("echo", q="x", n="3")),               # wrong type
        reply(call("echo", q="x", n=None, tags=None)),   # null for optional arguments is fine
        reply("done"),
    ])
    session = await open_scripted_session(config, lambda *a: next(script_calls))
    res = await session.rt.run_agent(_agent(), "go", depth=1, extra_tools=[_echo_tool(seen)])
    errors = [m.content[0] for m in res.messages if m.role == "user" and m.tool_results]
    assert "'q' is required" in errors[0].content and errors[0].is_error
    assert "'n' must be integer, got string" in errors[1].content
    assert errors[0].content.startswith("Your arguments for echo do not match its schema")
    assert not errors[2].is_error and seen == [{"q": "x", "n": None, "tags": None}]
    await session.close()


def test_argument_check_scope_and_fallback(monkeypatch):
    s = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": ["integer", "null"]},
                                          "c": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                                          "d": {"type": "object"}, "e": {"type": "number"},
                                          "f": {"type": "boolean"}},
         "required": ["a"]}
    good = {"a": "x", "b": None, "c": [1], "d": {"nested": {"anything": 1}}, "e": 2, "f": False, "extra": 1}
    assert tool_argument_problems(s, good) == []
    assert tool_argument_problems(s, {"a": "x", "e": 1.5}) == []
    assert tool_argument_problems(s, {"a": "x", "b": 2.0}) == []           # 2.0 is an integer in JSON Schema
    bad = {"b": "two", "e": True, "f": "yes"}
    with_js = tool_argument_problems(s, bad)
    assert with_js == ["'a' is required", "'b' must be integer, got string", "'e' must be number, got boolean",
                       "'f' must be boolean, got string"]
    assert tool_argument_problems(s, "not an object") == ["arguments must be a JSON object, got string"]
    assert tool_argument_problems({}, {"x": 1}) == [] and tool_argument_problems(None, {}) == []
    # without jsonschema: the same checks and messages
    import builtins
    real_import = builtins.__import__

    def no_jsonschema(name, *a, **kw):
        if name == "jsonschema":
            raise ImportError("no jsonschema")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_jsonschema)
    assert tool_argument_problems(s, bad) == with_js
    assert tool_argument_problems(s, good) == []


# ---------------------------------------------------------------------------
# 5. empty replies
# ---------------------------------------------------------------------------


async def test_empty_reply_after_tool_results_is_nudged_once(config):
    seen: list = []
    replies = iter([reply(call("echo", q="a")), reply(), reply(call("echo", q="b")), reply()])
    session = await open_scripted_session(config, lambda *a: next(replies))
    res = await session.rt.run_agent(_agent(), "go", depth=1, extra_tools=[_echo_tool(seen)])
    nudges = [m for m in res.messages if m.role == "user" and m.text == EMPTY_REPLY_MSG]
    assert len(nudges) == 1 and res.nudges == ["empty_reply"]
    assert res.messages[4].text == EMPTY_REPLY_MSG and seen == [{"q": "a"}, {"q": "b"}]
    assert res.status == "completed" and res.model_calls == 4   # the second empty reply is accepted
    assert len(_events(session, "empty_reply_nudge")) == 1
    await session.close()


async def test_empty_reply_without_tool_results_is_final(config):
    session = await open_scripted_session(config, lambda *a: reply())
    res = await session.rt.run_agent(_agent(), "go", depth=1)
    assert res.model_calls == 1 and res.nudges == []
    await session.close()


# ---------------------------------------------------------------------------
# 6. forced tool choice / tool_choice none
# ---------------------------------------------------------------------------


def _submit(sink):
    def handler(ctx, a):
        sink.append(a)
        return "recorded"
    return Tool("submit", "Submit.", schema({"value": {"type": "integer"}}, ["value"]), handler, terminal=True)


async def test_force_tool_on_the_final_allowed_call(config):
    sink: list = []

    def script(agent, system, messages, tools):
        if messages[-1].text == FORCE_TOOL_MSG.format(tool="submit"):
            return reply(call("submit", value=7), call("echo", q="no"))
        return reply(call("echo", q="more"))

    provider = Recording(script)
    session = await open_scripted_session(config, provider)
    seen: list = []
    res = await session.rt.run_agent(_agent(), "go", depth=1, extra_tools=[_echo_tool(seen), _submit(sink)],
                                     force_tool="submit", max_turns=2)
    assert res.status == "completed" and res.stop_reason == "terminal_tool" and sink == [{"value": 7}]
    assert ["tool_choice" in s.extra for s in provider.seen] == [False, False, True]
    assert provider.seen[-1].extra["tool_choice"] == {"name": "submit"}
    last = res.messages[-1].content
    assert [r.is_error for r in last] == [False, True] and "only submit runs" in last[1].content
    assert len(seen) == 2                                      # the echo of the forced call did not run
    await session.close()


async def test_forced_call_without_provider_support_still_runs_the_tool(config):
    sink: list = []
    provider = Recording(lambda a, s, m, t: reply(call("submit", value=1)), tool_choice=False)
    session = await open_scripted_session(config, provider)
    res = await session.rt.run_agent(_agent(), "please submit", depth=1, extra_tools=[_submit(sink)],
                                     force_tool="submit", max_turns=0)
    assert res.status == "completed" and sink == [{"value": 1}]
    assert "tool_choice" not in provider.seen[0].extra
    # max_turns=0 + a task: the task is the instruction of the forced call (no extra harness message)
    assert [m.text for m in res.messages if m.role == "user" and m.text] == ["please submit"]
    await session.close()


async def test_final_no_tool_call_sends_tool_choice_none(config):
    provider = Recording(lambda a, s, m, t: reply(call("echo", q="x")) if len(m) < 4 else reply("final"))
    session = await open_scripted_session(config, provider)
    res = await session.rt.run_agent(_agent(), "go", depth=1, extra_tools=[_echo_tool([])], max_turns=1)
    assert res.status == "turn_limit"
    assert provider.seen[-1].extra["tool_choice"] == "none"
    assert TURN_LIMIT_MSG in [m.text for m in res.messages]
    await session.close()


# ---------------------------------------------------------------------------
# 7. bulk: continuation, strict submit, concurrency default, token budgets
# ---------------------------------------------------------------------------


class _Out(BaseModel):
    value: int


def _item_id(messages):
    return messages[0].text.split("ITEM ")[1].split()[0]


def _items(n):
    return [BulkItem(f"i{k:02d}", f"ITEM i{k:02d} please") for k in range(n)]


def _recs(out):
    return {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}


async def test_bulk_continues_the_same_conversation_once_to_submit(config, tmp_path):
    def script(agent, system, messages, tools):
        if messages[-1].text == SUBMIT_NUDGE:
            assert len(messages) == 3          # task, the agent's answer, the nudge: same conversation
            return reply(call("submit_result", value=5))
        return reply("I think the value is 5.")  # ends without submitting

    provider = Recording(script)
    session = await open_scripted_session(config, provider)
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent("bulk-probe", tier="bulk"), _Out, out, concurrency=1,
                               prewarm="none").run(_items(2))
    recs = _recs(out)
    assert all(r["ok"] and r["result"] == {"value": 5} and r["continued"] and r["attempts"] == 1
               for r in recs.values())
    assert all(r["model_calls"] == 2 for r in recs.values())
    assert summary["continued"] == 2 and summary["continued_submitted"] == 2
    forced = [s for s in provider.seen if s.extra.get("tool_choice")]
    assert len(forced) == 2 and all(s.extra["tool_choice"] == {"name": "submit_result"} for s in forced)
    # both calls of an item share one session key (same replica, same prefix cache)
    keys = [s.extra["session_key"] for s in provider.seen]
    assert keys[0] == keys[1] and keys[2] == keys[3] and keys[0] != keys[2]
    assert len(_events(session, "bulk_continue")) == 2
    await session.close()


async def test_bulk_item_that_ignores_the_forced_submit_fails_without_a_rerun(config, tmp_path):
    session = await open_scripted_session(config, Recording(lambda *a: reply("no.")))
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent("bulk-probe", tier="bulk"), _Out, out, concurrency=1,
                               retries=1, prewarm="none").run(_items(1))
    rec = _recs(out)["i00"]
    assert rec["ok"] is False and rec["continued"] and rec["attempts"] == 1 and rec["model_calls"] == 2
    assert "without submit_result" in rec["error"]
    assert summary["failed"] == 1
    await session.close()


async def test_bulk_turn_limit_forces_submit_result(config, tmp_path):
    def script(agent, system, messages, tools):
        if "submit_result now" in (messages[-1].text or ""):
            return reply(call("submit_result", value=9))
        return reply(call("echo", q="look"))

    provider = Recording(script)
    session = await open_scripted_session(config, provider)
    agent = _agent("bulk-probe", tier="bulk", max_turns=2)
    session.rt.registry.add(_echo_tool([]))
    agent.tools = ["echo"]
    out = tmp_path / "b.jsonl"
    await BulkRunner(session.rt, agent, _Out, out, concurrency=1, prewarm="none").run(_items(1))
    rec = _recs(out)["i00"]
    assert rec["ok"] and rec["result"] == {"value": 9} and not rec.get("continued")
    assert rec["model_calls"] == 3 and provider.seen[-1].extra["tool_choice"] == {"name": "submit_result"}
    await session.close()


async def test_bulk_concurrency_defaults_to_the_config(config, tmp_path):
    session = await open_scripted_session(config, lambda *a: reply(call("submit_result", value=1)))
    config["bulk"]["default_concurrency"] = 7
    r = BulkRunner(session.rt, _agent("b", tier="bulk"), _Out, tmp_path / "x.jsonl")
    assert r.concurrency == 7
    assert BulkRunner(session.rt, _agent("b", tier="bulk"), _Out, tmp_path / "x.jsonl", concurrency=3).concurrency == 3
    del config["bulk"]["default_concurrency"]
    assert BulkRunner(session.rt, _agent("b", tier="bulk"), _Out, tmp_path / "x.jsonl").concurrency == 32
    await session.close()


def _tok_usage(agent, messages, msg):
    return Usage(input_tokens=1000, output_tokens=100, cache_read_tokens=1000)   # 1000 + 100 + 0.1*1000 = 1200


async def test_bulk_token_budget_stops_the_run(config, tmp_path):
    session = await open_scripted_session(config, ScriptedProvider(lambda *a: reply(call("submit_result", value=1)),
                                                                   usage_fn=_tok_usage))
    out = tmp_path / "b.jsonl"
    summary = await BulkRunner(session.rt, _agent("b", tier="bulk"), _Out, out, concurrency=1,
                               budget_tokens=3000).run(_items(10))
    assert summary["budget_stop"] is True and summary["completed"] == 3
    assert summary["budget_tokens_spent"] == 3600 and summary["unrun"] == 7
    assert summary["total_cost_usd"] == 0 and summary["total_tokens"] == 3600
    assert summary["median_tokens"] == 1200
    assert all(r["budget_tokens"] == 1200 for r in _recs(out).values())
    await session.close()


async def test_max_item_tokens_fails_the_item(config, tmp_path):
    session = await open_scripted_session(config, ScriptedProvider(lambda *a: reply(call("submit_result", value="x")),
                                                                   usage_fn=_tok_usage))
    out = tmp_path / "b.jsonl"
    config["limits"]["max_item_tokens"] = 2000
    await BulkRunner(session.rt, _agent("b", tier="bulk"), _Out, out, concurrency=1,
                     prewarm="none").run(_items(1))
    rec = _recs(out)["i00"]
    assert rec["ok"] is False and rec["status"] == "item_budget" and "tokens exceeded" in rec["error"]
    await session.close()


def test_bulk_budget_cli_arguments():
    p = argparse.ArgumentParser()
    add_budget_arguments(p)
    args = p.parse_args(["--budget-tokens", "5e6", "--max-item-tokens", "1000", "--max-item-cost", "2"])
    assert args.concurrency is None
    assert budget_kwargs(args) == {"budget_tokens": 5e6, "max_item_tokens": 1000.0, "max_item_cost_usd": 2.0}
    from vbt.cli import build_parser
    a = build_parser().parse_args(["case1", "annotate", "--budget-tokens", "100"])
    assert a.budget_tokens == 100 and a.concurrency is None
    b = build_parser().parse_args(["bulk", "in.jsonl", "--agent", "x", "--schema", "m:C", "--out", "o",
                                   "--budget-tokens", "7"])
    assert b.budget_tokens == 7 and b.concurrency is None


async def test_bulk_dispatch_takes_a_token_budget(config, tmp_path):
    from vbt.bulk_dispatch import bulk_dispatch_tools
    from vbt.tools.base import ToolContext, ToolFailure

    def script(agent, system, messages, tools):
        return reply(call("submit_result", value=1))

    session = await open_scripted_session(config, ScriptedProvider(script, usage_fn=_tok_usage))
    rt = session.rt
    items = rt.run.dir / "work" / "items.csv"
    items.parent.mkdir(parents=True, exist_ok=True)
    items.write_text("id,name\n" + "".join(f"t{i},n{i}\n" for i in range(6)))
    schema_path = rt.run.dir / "work" / "schema.json"
    schema_path.write_text(json.dumps({"type": "object", "properties": {"value": {"type": "integer"}},
                                       "required": ["value"]}))
    tools = {t.name: t for t in bulk_dispatch_tools(rt)}
    ctx = ToolContext(agent="cso", run=rt.run, runtime=rt)
    name = next(iter(rt.agents))
    args = {"subagent_type": name, "items_path": "work/items.csv", "prompt_template": "Item {id}",
            "schema": str(schema_path), "pilot_size": 2, "concurrency": 1}
    with pytest.raises(ToolFailure, match="budget_tokens"):
        await tools["BulkDispatch"].handler(ctx, args)
    pilot = await tools["BulkDispatch"].handler(ctx, {**args, "budget_tokens": 3000})
    proj = pilot["projection"]
    assert proj["budget_tokens"] == 3000 and proj["job_spent_tokens"] == 2400
    assert proj["mean_tokens_per_item"] == 1200 and proj["projected_remaining_tokens"] == 4800
    assert proj["fits_budget"] is False and proj["budget_usd"] is None
    with pytest.raises(ToolFailure, match="already used"):
        await tools["BulkDispatch"].handler(ctx, {**args, "budget_tokens": 2000})
    await session.close()


# ---------------------------------------------------------------------------
# 8. token budgets
# ---------------------------------------------------------------------------


def test_cost_scope_tokens_either_limit():
    s = CostScope("turn", 10.0, limit_tokens=1000)
    s.charge(0.0, 999)
    assert not s.exceeded and not s.exhausted
    s.charge(0.0, 2)
    assert s.tokens_exceeded and s.exceeded and not s.usd_exceeded
    assert "tokens exceeded 1.0K (used 1.0K)" in s.describe()
    u = CostScope("x", 1.0)
    u.charge(2.0)
    assert u.exceeded and u.describe().startswith("cost exceeded $1.00")
    with open_scope("outer", limit_tokens=50) as outer:
        with open_scope("inner") as inner:
            budget.charge(0.0, 60)
            assert inner.tokens == 60 and outer.tokens == 60
            with pytest.raises(BudgetExceeded) as ei:
                budget.check()
            assert ei.value.scope is outer and "tokens exceeded" in str(ei.value)
    assert usage_tokens(Usage(100, 10, 1000, 5)) == pytest.approx(100 + 10 + 5 + 100)
    assert usage_tokens(Usage(100, 10, 1000), cached_weight=0.5) == pytest.approx(610)
    assert usage_tokens(None) == 0
    assert budget.cached_token_weight({"limits": {"cached_token_weight": 0.25}}) == 0.25
    assert budget.cached_token_weight({}) == 0.1


async def test_runtime_charges_tokens_and_the_turn_token_cap_stops_a_turn(config):
    rules = {"cso": [reply(call("Glob", pattern="*.none")) for _ in range(10)]}
    provider = ScriptedProvider.from_rules(rules, usage_fn=_tok_usage)
    config["limits"]["max_turn_tokens"] = 2500
    config["limits"]["cached_token_weight"] = 0.1
    session = await open_scripted_session(config, provider)
    out = await session.ask("work until the budget runs out")
    rec = session.last_turn
    # 2 calls (2400 tokens) pass the check, the 3rd charges 3600 > 2500 -> the grace (final) call
    assert "Turn incomplete" in out or rec["status"] != "completed"
    assert rec["tokens"]["total"] == 4 * 2100 and rec["tokens"]["output_tokens"] == 400
    grace = [e for e in session.run.events() if e.get("type") == "budget_grace"]
    assert grace and "tokens exceeded" in grace[0]["error"]
    await session.close()


def test_cost_display_leads_with_tokens_when_usd_is_zero():
    from vbt.cli import _spend

    assert _spend(0.0, {"total": 1_234_567, "cached": 1_000_000, "output": 45_600}) == \
        "1.23M tokens (1.00M cached, 45.6K output), $0.00"
    assert _spend(1.5, {"total": 2000, "cached": 0, "output": 10}) == "$1.50, 2.0K tokens (0 cached, 10 output)"
    assert _spend(2.0, {}) == "$2.00"


# ---------------------------------------------------------------------------
# 9. context: old reasoning cleared with old tool results
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, replays):
        self._caps = ProviderCapabilities(replays_reasoning=replays)

    def capabilities(self, model=None):
        return self._caps

    def context_window(self, model):
        return 30_000


def _long_history(n_rounds=8, think=4000, result=6000):
    msgs = [Message.user("task")]
    for i in range(n_rounds):
        msgs.append(Message("assistant", [ThinkingBlock("r" * think, "vllm", native={"field": "reasoning",
                                                                                      "family": "qwen3_8"}),
                                          ToolCall(f"c{i}", "Read", {"file_path": f"f{i}"})]))
        msgs.append(Message("user", [ToolResult(f"c{i}", "x" * result)]))
    return msgs


@pytest.mark.parametrize("replays", [True, False])
async def test_soft_clear_also_clears_old_reasoning_for_replaying_providers(tmp_path, replays):
    pol = ContextPolicy(soft_ratio=0.5, hard_ratio=10.0, keep_recent_calls=2, min_clear_fraction=0.0)
    cm = ContextManager(_Ctx(replays), pol, spill_dir=tmp_path)
    msgs = _long_history()
    settings = ModelSettings("vllm", MODEL)
    res = await cm.maybe_compact(msgs, last_usage=None, settings=settings, system="s", agent="probe")
    assert res is not None and res.strategy == "clear_tool_results"
    thinking = [b for m in msgs for b in m.content if isinstance(b, ThinkingBlock)]
    old, recent = thinking[:-2], thinking[-2:]
    assert all(b.text == "r" * 4000 for b in recent)
    if replays:
        assert all(b.text == "" and b.native["cleared_by_harness"] and b.native["cleared_chars"] == 4000
                   and b.native["family"] == "qwen3_8" for b in old)
        assert res.thinking_cleared == 6
    else:
        assert all(b.text for b in old) and res.thinking_cleared == 0
    results = [b for m in msgs for b in m.content if isinstance(b, ToolResult)]
    assert all(r.content.startswith(CLEARED_PREFIX) for r in results[:-2])


async def test_cleared_reasoning_is_not_replayed_to_the_server():
    from vbt.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(base_url="http://127.0.0.1:9/v1", family="qwen3_8")
    msgs = [Message.user("q"),
            Message("assistant", [ThinkingBlock("", "vllm", native={"cleared_by_harness": True,
                                                                     "family": "qwen3_8"}),
                                  ToolCall("c1", "Read", {"file_path": "a"})]),
            Message("user", [ToolResult("c1", "ok")])]
    body = p._request(ModelSettings("vllm", MODEL), "s", msgs, [])
    asst = [m for m in body["messages"] if m["role"] == "assistant"][0]
    assert "reasoning" not in asst and "reasoning_content" not in asst


# ---------------------------------------------------------------------------
# 10. readiness (prepare), pinning (server_info), doctor --smoke, CLI errors
# ---------------------------------------------------------------------------


async def test_open_session_prepares_the_server_and_pins_its_facts(fake, tmp_path):
    from vbt.orchestrator import open_session

    cfg = _local_config(tmp_path, fake.url + "/v1")
    fake.responses.append(("sse", text_turn("Hello from Qwen.")))
    session = await open_session(cfg, start_mcp=False)
    try:
        gets = [r["path"] for r in fake.requests if r["method"] == "GET"]
        assert "/health" in gets and "/v1/models" in gets and "/version" in gets
        pinned = json.loads((session.run.dir / "inputs" / "config.json").read_text())
        server = pinned["provider"]["server"]
        assert server["version"] == "0.31.0" and server["max_model_len"] == 262144
        assert server["models"][0]["id"] == MODEL and server["models"][0]["root"] == "Qwen/Qwen3.8-27B-FP8"
        assert server["family"] == "qwen3_8"
        assert pinned["provider"]["name"] == "vllm" and pinned["provider"]["model_pattern"]
        assert session.run.manifest["config"]["provider"]["server"]["version"] == "0.31.0"
        out = await session.ask("Say hello.")
        assert out.endswith("Hello from Qwen.")
        body = fake.posts[0]["body"]
        assert body["model"] == MODEL and body["reasoning_effort"] == "xhigh"
        assert body["thinking_token_budget"] == 24576 and body["max_tokens"] == 40960
        assert "session_key" not in json.dumps(body)                      # never forwarded
        strict = {t["function"]["name"] for t in body["tools"] if t["function"].get("strict")}
        assert "mcp__provenance__record_claims" in strict
    finally:
        await session.close()


async def test_open_session_refuses_when_the_model_is_not_served(fake, tmp_path):
    from vbt.orchestrator import open_session
    from vbt.preflight import DataReadinessError, ProviderNotReadyError

    cfg = _local_config(tmp_path, fake.url + "/v1", provider={"options": {"served_model_name": "other-model"}})
    with pytest.raises(ProviderNotReadyError) as ei:
        await open_session(cfg, start_mcp=False)
    assert "other-model" in str(ei.value) and MODEL in str(ei.value)
    assert isinstance(ei.value, DataReadinessError)
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


async def test_open_session_refuses_when_the_server_is_down(tmp_path, monkeypatch):
    from vbt.orchestrator import open_session
    from vbt.preflight import ProviderNotReadyError

    _clear_env(monkeypatch)
    cfg = _local_config(tmp_path, "http://127.0.0.1:9/v1")
    with pytest.raises(ProviderNotReadyError, match="vbt local serve"):
        await open_session(cfg, start_mcp=False)
    # --skip-preflight skips the readiness check (the first model call reports the problem instead)
    cfg["preflight"]["skip"] = True
    session = await open_session(cfg, start_mcp=False)
    assert session.run.config["provider"]["server"]["error"]
    await session.close()


def test_cli_reports_a_server_that_is_not_ready(tmp_path, monkeypatch, capsys):
    from vbt import cli

    _clear_env(monkeypatch)
    monkeypatch.setenv("VBT_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    rc = cli.main(["--runs-dir", str(tmp_path / "r"), "--no-mcp", "run", "-q", "hello"])
    err = capsys.readouterr().err
    assert rc == 2 and "model server not ready" in err and "vbt local serve" in err


def test_doctor_smoke_checks_the_model_server(fake, tmp_path, monkeypatch):
    from vbt.preflight import run_doctor

    cfg = _local_config(tmp_path, fake.url + "/v1")
    fake.models[0]["max_model_len"] = 131072
    lines: list[str] = []
    run_doctor(cfg, smoke=True, out=lines.append)
    text = "\n".join(lines)
    assert "[ok] vllm model server ready: serves qwen3.8-27b; version 0.31.0; family qwen3_8" in text
    assert "[--] model context window: server max_model_len 131,072; configured context_window_tokens 262,144" \
        in text
    assert "lower models.<tier>.context_window_tokens to 131072" in text
    lines.clear()
    run_doctor(cfg, smoke=False, out=lines.append)
    text = "\n".join(lines)
    assert "[ok] vllm credentials" in text
    assert f"[ok] vllm model server: responding at {fake.url}" in text and "vbt doctor --smoke" in text
    assert "web search backend" in text


def test_doctor_reports_a_down_server_without_smoke(tmp_path, monkeypatch):
    """Plain `vbt doctor`: one bounded /health probe says the server is not running (optional check)."""
    from vbt.preflight import run_doctor

    _clear_env(monkeypatch)
    cfg = _local_config(tmp_path, "http://127.0.0.1:9/v1")
    lines: list[str] = []
    run_doctor(cfg, smoke=False, out=lines.append)
    text = "\n".join(lines)
    assert "[--] vllm model server: not running: nothing answers at http://127.0.0.1:9" in text
    assert "vbt local serve" in text and "model server ready" not in text


def test_doctor_smoke_reports_a_down_server(tmp_path, monkeypatch):
    from vbt.preflight import run_doctor

    _clear_env(monkeypatch)
    cfg = _local_config(tmp_path, "http://127.0.0.1:9/v1")
    lines: list[str] = []
    rc = run_doctor(cfg, smoke=True, out=lines.append)
    text = "\n".join(lines)
    assert rc == 1 and "[!!] vllm model server ready" in text and "vbt local serve" in text


def test_anthropic_adapter_accepts_the_session_key():
    from vbt.providers.anthropic_provider import LOCAL_EXTRA_KEYS

    assert "session_key" in LOCAL_EXTRA_KEYS

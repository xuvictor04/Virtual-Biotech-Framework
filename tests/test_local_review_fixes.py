"""Regression tests for the review of the local-LLM build (findings P3, P4, INT-1..INT-7,
ops-1..ops-3, SEC-1..SEC-6, DT1..DT7). Offline: fake HTTP servers and scripted providers."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from vbt.config import ProfileError, load_config
from vbt.context import ContextManager, ContextPolicy
from vbt.providers import create_provider
from vbt.providers.base import (
    Message,
    ModelResponse,
    ProviderError,
    RetryableProviderError,
    StopReason,
    TextBlock,
    ThinkingBlock,
    Usage,
)
from vbt.providers.openai_compat import OpenAICompatProvider
from vbt.providers.retry import RetryPolicy, complete_with_retry

from conftest import open_scripted_session
from test_local_integration import Recording, _agent, _echo_tool, _items, _local_config, _Out, _recs, _tok_usage
import test_openai_compat
from test_openai_compat import MODEL, Q, TOOLS, chunk, settings, tc_delta, text_turn

# the fake vLLM server fixtures of test_openai_compat, bound here so pytest collects them for this module
fake = test_openai_compat.fake
make = test_openai_compat.make

ROOT_DOCS = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_local_env(monkeypatch):
    for v in ("VBT_LLM_BASE_URL", "VBT_LLM_DP_SIZE", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY", "OPENAI_API_KEY",
              "VBT_LLM_API_KEY"):
        monkeypatch.delenv(v, raising=False)


# ---------------------------------------------------------------------------
# P3: an aborted stream with half-streamed tool calls is retried, never a TOOL_USE turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("finish", ["abort", "error"])
async def test_p3_abort_with_partial_tool_call_is_retryable(fake, make, finish):
    partial = [tc_delta(0, id="c1", name="Read", args='{"file_path": "TP'), chunk(finish=finish)]
    fake.responses.append(("sse", partial))
    p = make(auto_discover=False)
    with pytest.raises(RetryableProviderError, match=finish):
        await p.complete(settings=settings(), system="s", messages=Q, tools=TOOLS)
    # through the retry layer the request is re-sent; no bogus call/result pair reaches the history
    fake.responses.append(("sse", partial))
    fake.responses.append(("sse", text_turn("ok")))

    async def no_sleep(_):
        return None

    resp = await complete_with_retry(p, policy=RetryPolicy(attempts=3, base_delay_s=0.0, jitter=0.0), sleep=no_sleep,
                                     settings=settings(), system="s", messages=Q, tools=TOOLS)
    await p.aclose()
    assert resp.stop_reason is StopReason.END_TURN and resp.message.text == "ok" and not resp.message.tool_calls
    assert resp.retries == 1


# ---------------------------------------------------------------------------
# SEC-4: OPENAI_API_KEY is never sent to the self-hosted server implicitly
# ---------------------------------------------------------------------------


async def test_sec4_openai_api_key_is_not_sent_to_the_local_server(fake, make, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-leak-123")
    p = make(auto_discover=False)
    await p.complete(settings=settings(), system="s", messages=Q, tools=[])
    await p.prepare()
    sent = [{k.lower(): v for k, v in r["headers"].items()} for r in fake.requests]
    assert sent and not any("authorization" in h for h in sent)
    # explicit opt-in for a hosted OpenAI-compatible API
    q = make(auto_discover=False, api_key_env="OPENAI_API_KEY")
    await q.complete(settings=settings(), system="s", messages=Q, tools=[])
    assert {k.lower(): v for k, v in fake.posts[-1]["headers"].items()}["authorization"] == \
        "Bearer sk-openai-should-not-leak-123"
    for prov in (p, q):
        await prov.aclose()
    # vbt local check's raw HTTP client follows the same rule
    from vbt.local import check as chk
    async with chk.LocalChecker(chk.CheckOptions(base_url=fake.url + "/v1"))._http() as http:
        assert "authorization" not in {k.lower() for k in http.headers}


# ---------------------------------------------------------------------------
# P4 / DT3: DeepSeek-V4 never gets Think-Max for a harness effort
# ---------------------------------------------------------------------------


def test_p4_single_cell_max_effort_is_high_on_deepseek_v4():
    cfg = load_config(["local-deepseek-v4"])
    p = OpenAICompatProvider(base_url="http://127.0.0.1:9/v1")
    from vbt.agents import AgentDefinition
    sc = AgentDefinition(name="single-cell-analyst", description="", prompt="", tier="scientist", tools=[],
                         memory="none", effort="max")
    s = sc.settings(cfg)
    assert s.effort == "max" and s.max_tokens == 65536
    body = p._request(s, "sys", Q, [])
    assert body["reasoning_effort"] == "high" and body["chat_template_kwargs"] == {"thinking": True}
    # on Qwen3.8 the same agent gets xhigh (never 'max')
    body = p._request(sc.settings(load_config([])), "sys", Q, [])
    assert body["reasoning_effort"] == "xhigh"


# ---------------------------------------------------------------------------
# SEC-1: the Anthropic adapter never inherits the local server's base_url
# ---------------------------------------------------------------------------


def test_sec1_anthropic_ignores_the_local_server_options():
    from vbt.preflight import provider_options
    from vbt.providers import anthropic_options

    cfg = load_config([], overrides={"provider": {"name": "anthropic", "options": {"prompt_caching": True}}})
    opts = provider_options(cfg)
    assert opts["base_url"] == "http://localhost:8000/v1"           # inherited from default.yaml ...
    clean = anthropic_options(opts)
    for key in ("base_url", "read_timeout_s", "max_concurrency", "data_parallel_size", "served_model_name"):
        assert key not in clean                                       # ... and dropped for Claude
    assert clean["prompt_caching"] is True
    gw = anthropic_options({"base_url": "http://localhost:8000/v1", "anthropic_base_url": "https://gw.example/a"})
    assert gw == {"base_url": "https://gw.example/a"}
    pytest.importorskip("anthropic")
    p = create_provider("anthropic", api_key="sk-ant-test-123456789", **opts)
    assert "localhost:8000" not in str(p.client.base_url)


# ---------------------------------------------------------------------------
# INT-1: replaying a pre-switch Claude run layers the claude profile
# ---------------------------------------------------------------------------


def test_int1_replay_of_a_pinned_anthropic_run_uses_the_claude_profile():
    from vbt.replay import _replay_config

    pinned = {"profiles": [], "provider": {"name": "anthropic", "options": {"api_key": "<redacted>"}},
              "models": {"orchestrator": {"model": "claude-opus-5", "effort": "high", "max_tokens": 64000,
                                          "thinking": True}}}
    cfg = _replay_config(pinned, model=None, runs_dir=None)
    assert cfg["profiles"] == ["claude"] and cfg["provider"]["name"] == "anthropic"
    live = {k: v for k, v in cfg["provider"]["options"].items() if v is not None}
    assert "base_url" not in live and "max_concurrency" not in live and "read_timeout_s" not in live
    orch = cfg["models"]["orchestrator"]
    assert orch["model"] == "claude-opus-5" and orch.get("thinking_budget") is None \
        and orch.get("context_window_tokens") is None
    assert (cfg["context"]["soft_ratio"], cfg["context"]["hard_ratio"]) == (0.7, 0.85)
    # explicit profiles: the pinned provider stays anthropic, so the rule still applies
    assert _replay_config(pinned, model=None, runs_dir=None, profiles=["no-web"])["profiles"] == ["claude", "no-web"]
    # paper runs and local runs are left alone
    assert _replay_config({"profiles": ["paper"], "provider": {"name": "anthropic"}}, model=None,
                          runs_dir=None)["profiles"] == ["paper"]
    assert _replay_config({"profiles": [], "provider": {"name": "vllm"}}, model=None, runs_dir=None)["profiles"] == []


# ---------------------------------------------------------------------------
# INT-2 / DT4: --model reaches the wire; served_model_name conflicts are refused
# ---------------------------------------------------------------------------


def _args(**kw):
    base = dict(profile=[], model=None, no_web=False, runs_dir=None, no_clarify=False, skip_preflight=False,
                allow_missing_data=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_int2_model_flag_is_the_model_requested_from_the_server():
    from vbt.agents import AgentDefinition
    from vbt.cli import build_config
    from vbt.preflight import provider_options

    cfg = build_config(_args(model="Qwen/Qwen3.8-27B-FP8"))
    assert cfg["provider"]["options"].get("served_model_name") is None
    assert cfg["provider"]["options"].get("family") is None
    cso = AgentDefinition(name="cso", description="", prompt="", tier="orchestrator", tools=[], memory="none")
    s = cso.settings(cfg)
    p = OpenAICompatProvider(**provider_options(cfg))
    body = p._request(s, "sys", Q, [])
    assert body["model"] == "Qwen/Qwen3.8-27B-FP8" and body["reasoning_effort"] == "xhigh"
    assert p.family_for(s.model).name == "qwen3_8"


def test_int2_served_model_name_conflicting_with_model_is_refused():
    from vbt.cli import ModelResolutionError, resolve_model
    from vbt.web.server import resolve_model as web_resolve

    cfg = load_config([], overrides={"provider": {"options": {"served_model_name": "qwen3.8-27b"}}})
    with pytest.raises(ModelResolutionError, match="served_model_name"):
        resolve_model(cfg, "Qwen/Qwen3.8-27B-FP8")
    with pytest.raises(ValueError, match="served_model_name"):
        web_resolve(cfg, "opus")
    assert resolve_model(cfg, "qwen") == "qwen3.8-27b" and web_resolve(cfg, "qwen3.8-27b") == "qwen3.8-27b"
    # without served_model_name (the shipped configs) any served-like name is accepted
    assert resolve_model(load_config([]), "Qwen/Qwen3.8-27B-FP8") == "Qwen/Qwen3.8-27B-FP8"


async def test_int2_prepare_checks_every_configured_tier_model(fake, tmp_path):
    from vbt.preflight import ProviderNotReadyError, create_configured_provider, prepare_provider

    cfg = _local_config(tmp_path, fake.url + "/v1")
    cfg["models"]["orchestrator"]["model"] = "Qwen/Qwen3.8-27B-FP8"
    p = create_configured_provider(cfg)
    with pytest.raises(ProviderNotReadyError, match="Qwen/Qwen3.8-27B-FP8"):
        await prepare_provider(p, cfg)
    await p.aclose()


# ---------------------------------------------------------------------------
# INT-6: the configured window never exceeds the server's max_model_len
# ---------------------------------------------------------------------------


async def test_int6_window_is_clamped_to_the_server_max_model_len(fake, make):
    fake.models[0]["max_model_len"] = 65536
    p = make()
    undiscovered = ContextManager(make())
    assert undiscovered.window_for(settings(context_window_tokens=262144)) == 262144
    await p.prepare()
    assert p.server_context_window(MODEL) == 65536
    cm = ContextManager(p, ContextPolicy.from_config({"default_window_tokens": 262144}))
    assert cm.window_for(settings(context_window_tokens=262144)) == 65536
    assert cm.window_for(settings(context_window_tokens=32768)) == 32768   # a smaller configured window wins
    await p.aclose()


async def test_int6_open_session_warns_about_a_smaller_server_window(fake, tmp_path):
    from vbt.orchestrator import open_session

    fake.models[0]["max_model_len"] = 65536
    events = []
    session = await open_session(_local_config(tmp_path, fake.url + "/v1"), start_mcp=False,
                                 on_event=lambda k, d=None, **kw: events.append((k, d or kw)))
    try:
        warnings = [str((d or {}).get("message", "")) for k, d in events if k == "warning"]
        assert any("max_model_len (65,536)" in w for w in warnings), warnings
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# SEC-2: config-sourced secrets never reach the agent-readable run record
# ---------------------------------------------------------------------------


async def test_sec2_web_secrets_are_redacted_in_the_run_record(config, tmp_path):
    from vbt.envpolicy import redact
    from vbt.providers.mock import reply
    from vbt.tools import search_backends as sb

    config["web"]["search"] = {"backend": "brave", "brave_api_key": "BSA-secret-123456",
                               "searxng_url": "http://user:pa55word-xyz@searx.lan:8888"}
    session = await open_scripted_session(config, lambda *a: reply("ok"))
    try:
        for f in ("inputs/config.json", "MANIFEST.json"):
            text = (session.run.dir / f).read_text()
            assert "BSA-secret-123456" not in text and "pa55word-xyz" not in text, f
        pinned = json.loads((session.run.dir / "inputs" / "config.json").read_text())
        assert pinned["web"]["search"]["brave_api_key"] == "<redacted>"
        assert pinned["web"]["search"]["searxng_url"] == "http://<redacted>@searx.lan:8888"
    finally:
        await session.close()
    # a config key is masked in tool output like an environment key
    sb.describe_search_backend(config)
    assert "BSA-secret-123456" not in redact("token BSA-secret-123456 here", {})
    # resuming / replaying never uses the placeholder literally
    from vbt.pinning import drop_redacted
    assert drop_redacted(pinned["web"])["search"] == {"backend": "brave"}


async def test_sec2_url_credentials_are_not_echoed_in_errors():
    from vbt.tools import search_backends as sb

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    backend = sb.SearxNGBackend("http://user:pa55word-xyz@127.0.0.1:9", {"retries": 0},
                                transport=httpx.MockTransport(refuse))
    with pytest.raises(sb.SearchBackendError) as ei:
        await backend("PCSK9", max_results=3)
    assert "pa55word-xyz" not in str(ei.value) and "127.0.0.1:9" in str(ei.value)
    assert "pa55word-xyz" not in repr(backend) and "pa55word-xyz" not in json.dumps(backend.describe())
    p = OpenAICompatProvider(base_url="http://u:secretpw-xyz@127.0.0.1:9/v1")
    with pytest.raises(ProviderError) as pe:
        await p.prepare()
    await p.aclose()
    assert "secretpw-xyz" not in str(pe.value)
    assert "secretpw-xyz" not in json.dumps(await p.server_info())


# ---------------------------------------------------------------------------
# INT-3: an exhausted item budget still gets one forced submit_result
# ---------------------------------------------------------------------------


async def test_int3_item_budget_forces_one_submit_and_counts_the_calls(config, tmp_path):
    def script(agent, system, messages, tools):
        if "budget for this task is exhausted" in (messages[-1].text or ""):
            from vbt.providers.mock import call, reply
            return reply(call("submit_result", value=4))
        from vbt.providers.mock import call, reply
        return reply(call("echo", q="more"))

    provider = Recording(script, usage_fn=_tok_usage)            # 1200 budget tokens per call
    session = await open_scripted_session(config, provider)
    session.rt.registry.add(_echo_tool([]))
    agent = _agent("bulk-probe", tier="bulk", max_turns=50)
    agent.tools = ["echo"]
    config["limits"]["max_item_tokens"] = 5000
    from vbt.bulk import BulkRunner
    out = tmp_path / "b.jsonl"
    await BulkRunner(session.rt, agent, _Out, out, concurrency=1, prewarm="none").run(_items(1))
    rec = _recs(out)["i00"]
    # calls 1-5 use 6000 > 5000 tokens; call 6 is the forced, budget-exempt submit
    assert rec["ok"] and rec["result"] == {"value": 4} and rec["model_calls"] == 6, rec
    assert provider.seen[-1].extra["tool_choice"] == {"name": "submit_result"}
    assert [e for e in session.run.events() if e.get("type") == "budget_force_tool"]
    await session.close()


# ---------------------------------------------------------------------------
# INT-4 / SEC-5: USD-only budgets cannot cap a 0-USD local model
# ---------------------------------------------------------------------------


def test_int4_unpriced_provider_detection_and_usd_only_budgets():
    from vbt.bulk import unpriced_provider, usd_only_budget_problem

    local, claude = load_config([]), load_config(["claude"])
    priced = load_config([], overrides={"provider": {"options": {"pricing": {"input_per_mtok": 1.0}}}})
    assert unpriced_provider(config=local) and not unpriced_provider(config=claude)
    assert not unpriced_provider(config=priced)
    assert unpriced_provider(OpenAICompatProvider(base_url="http://127.0.0.1:9/v1"))
    assert "budget_tokens" in usd_only_budget_problem(20, None, config=local)
    assert usd_only_budget_problem(20, 1e6, config=local) is None
    assert usd_only_budget_problem(20, None, config=claude) is None
    # SEC-5: a finite token cap for the CSO's own dispatches on the local default; USD for Claude
    assert int(local["bulk"]["dispatch_max_budget_tokens"]) > 0
    assert claude["bulk"]["dispatch_max_budget_tokens"] is None


def test_int4_bulk_cli_refuses_a_usd_only_budget_on_the_local_default(capsys):
    from vbt.cli import main

    assert main(["bulk", "in.jsonl", "--agent", "x", "--schema", "m:C", "--out", "o", "--budget", "50"]) == 2
    assert "--budget-tokens" in capsys.readouterr().err


async def test_int4_bulk_dispatch_needs_tokens_for_an_unpriced_model(config, tmp_path, monkeypatch):
    from vbt import bulk as bulk_mod
    from vbt import bulk_dispatch as bd
    from vbt.providers.mock import ScriptedProvider, call, reply
    from vbt.tools.base import ToolContext, ToolFailure

    for mod in (bulk_mod, bd):
        monkeypatch.setattr(mod, "unpriced_provider", lambda *a, **k: True)
    session = await open_scripted_session(config, ScriptedProvider(lambda *a: reply(call("submit_result", value=1)),
                                                                   usage_fn=_tok_usage))
    rt = session.rt
    items = rt.run.dir / "work" / "items.csv"
    items.parent.mkdir(parents=True, exist_ok=True)
    items.write_text("id,name\n" + "".join(f"t{i},n{i}\n" for i in range(6)))
    schema_path = rt.run.dir / "work" / "schema.json"
    schema_path.write_text(json.dumps({"type": "object", "properties": {"value": {"type": "integer"}},
                                       "required": ["value"]}))
    dispatch = {t.name: t for t in bd.bulk_dispatch_tools(rt)}["BulkDispatch"]
    ctx = ToolContext(agent="cso", run=rt.run, runtime=rt)
    args = {"subagent_type": next(iter(rt.agents)), "items_path": "work/items.csv", "prompt_template": "Item {id}",
            "schema": str(schema_path), "pilot_size": 2, "concurrency": 1}
    with pytest.raises(ToolFailure, match="cannot stop"):
        await dispatch.handler(ctx, {**args, "budget_usd": 20})
    pilot = await dispatch.handler(ctx, {**args, "budget_usd": 20, "budget_tokens": 100000})
    proj = pilot["projection"]
    # per-item cost is 0 USD: only the token budget decides whether the job fits
    assert proj["mean_cost_per_item_usd"] == 0 and proj["fits_budget"] is True
    pilot2 = await dispatch.handler(ctx, {**args, "budget_usd": 20, "budget_tokens": 4000, "job_id": "j2"})
    assert pilot2["projection"]["fits_budget"] is False
    await session.close()


# ---------------------------------------------------------------------------
# INT-7: --profile claude restores the per-command concurrency; upstream-web needs Claude
# ---------------------------------------------------------------------------


def test_int7_claude_and_paper_restore_per_command_concurrency():
    from vbt.bulk import ANNOTATE_CONCURRENCY, default_concurrency

    for prof in (["claude"], ["paper"]):
        cfg = load_config(prof)
        assert cfg["bulk"]["default_concurrency"] is None, prof
        assert default_concurrency(cfg) == 32 and default_concurrency(cfg, ANNOTATE_CONCURRENCY) == 64, prof
    local = load_config([])
    assert default_concurrency(local) == 32 == default_concurrency(local, ANNOTATE_CONCURRENCY)


async def test_int7_case1_annotate_defaults_to_64_on_claude(monkeypatch, tmp_path):
    import pandas as pd

    from vbt.case_studies.trial_outcomes import annotate as ann

    seen = {}

    class Runner:
        def __init__(self, rt, agent, model, out, *, concurrency=None, **kw):
            seen["concurrency"] = concurrency

        async def run(self, items):
            return {}

    monkeypatch.setattr(ann, "BulkRunner", Runner)
    monkeypatch.setattr(ann, "annotator_agent", lambda config, protocol=None: None)
    for prof, want in ((["claude"], 64), ([], 32)):
        rt = argparse.Namespace(config=load_config(prof))
        await ann.annotate(rt, pd.DataFrame(), tmp_path / "a.jsonl")
        assert seen["concurrency"] == want, prof


def test_int7_upstream_web_alone_is_refused(capsys):
    with pytest.raises(ProfileError, match="--profile claude"):
        load_config(["upstream-web"])
    cfg = load_config(["claude", "upstream-web"])
    assert cfg["provider"]["name"] == "anthropic" and cfg["models"]["orchestrator"]["effort"] == "xhigh"
    assert "requires_provider" not in cfg
    assert load_config(["paper", "upstream-web"])["models"]["scientist"]["model"] == "claude-sonnet-4-5"
    from vbt.cli import main
    assert main(["--profile", "upstream-web", "tools"]) == 2
    assert "upstream-web" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# ops-1: serving and harness data-parallel sizes agree
# ---------------------------------------------------------------------------


def _metrics(n):
    return "".join(f'vllm:num_requests_running{{engine="{k}",model_name="{MODEL}"}} 0.0\n' for k in range(n))


async def test_ops1_prepare_refuses_more_ranks_than_the_server_has(fake, make):
    fake.metrics = "# HELP vllm:num_requests_running ...\n" + _metrics(2)
    made = [make(data_parallel_size=4), make(data_parallel_size=2), make(data_parallel_size=1)]
    with pytest.raises(ProviderError, match="VBT_LLM_DP_SIZE=2"):
        await made[0].prepare()
    await made[1].prepare()          # matches
    await made[2].prepare()          # fewer ranks than engines: a warning only
    fake.metrics = None
    await made[0].prepare()          # no per-engine metrics: not checked
    for p in made:
        await p.aclose()


def test_ops1_harness_dp_size_follows_the_environment(monkeypatch):
    assert str(load_config(["local-dp"])["provider"]["options"]["data_parallel_size"]) == "4"
    monkeypatch.setenv("VBT_LLM_DP_SIZE", "2")
    cfg = load_config(["local-dp"])
    assert OpenAICompatProvider(**{k: v for k, v in cfg["provider"]["options"].items()
                                   if v is not None}).data_parallel_size == 2


def test_ops1_serve_prints_the_harness_dp_size(capsys):
    from vbt.cli import main
    from vbt.local.profiles import load_local_profiles, pick_profile, resolve_serve

    assert main(["local", "serve", "--profile", "dp", "--data-parallel", "2", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "export VBT_LLM_DP_SIZE=2; vbt --profile local-dp" in out
    assert main(["local", "serve", "--profile", "deepseek-v4", "--variant", "tep", "--dry-run"]) == 0
    assert "export VBT_LLM_DP_SIZE=1; vbt --profile local-deepseek-v4" in capsys.readouterr().out
    assert main(["local", "serve", "--profile", "dp", "--dry-run"]) == 0
    assert "VBT_LLM_DP_SIZE" not in capsys.readouterr().out          # 4 == the profile default
    profiles = load_local_profiles()
    assert resolve_serve(profiles, "deepseek-v4", variants=["tep"]).harness["data_parallel_size"] == 1
    pick = pick_profile("0, NVIDIA H200, 143771, 580.65\n1, NVIDIA H200, 143771, 580.65\n", profiles)
    assert pick.profile == "dp" and any("VBT_LLM_DP_SIZE=2" in w for w in pick.warnings)


# ---------------------------------------------------------------------------
# ops-2: vbt local check probes with the runtime's provider options and every rank
# ---------------------------------------------------------------------------


def test_ops2_probe_adapter_is_built_from_provider_options():
    from vbt.local import check as chk

    cfg = load_config(["local-dp"], overrides={"provider": {"options": {"headers": {"X-Proxy-Auth": "t0k"},
                                                                        "extra_body": {"seed": 1}}}})
    opts = chk.options_from_config(cfg)
    p = chk.default_provider_factory(opts, {"top_k": 5})
    assert p.data_parallel_size == 4 and p.routing == "header" and p.max_concurrency == 256
    assert p.extra_headers == {"X-Proxy-Auth": "t0k"} and p.extra_body == {"seed": 1, "top_k": 5}
    assert p.served_model_name == "qwen3.8-27b"
    # the report never carries credential-looking option values
    assert opts.as_dict()["provider_options"]["headers"] == {"X-Proxy-Auth": "<redacted>"}


class _DPServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    ranks = 2

    def _send(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": MODEL, "max_model_len": 262144}]})
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["content-length"]))
        rank = int(self.headers.get("X-data-parallel-rank") or 0)
        if rank >= self.ranks:
            return self._send(400, {"error": {"message": f"data_parallel_rank {rank} is out of range "
                                                         f"[0, {self.ranks}).", "type": "BadRequestError"}})
        self._send(200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "length"}]})

    def log_message(self, *a):
        pass


@pytest.fixture
def dp_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _DPServer)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()
    srv.server_close()


async def test_ops2_routing_check_probes_every_rank(dp_server):
    from vbt.local import check as chk

    def checker(dp):
        opts = chk.CheckOptions(base_url=dp_server, model=MODEL,
                                provider_options={"data_parallel_size": dp, "routing": "header"})
        return chk.LocalChecker(opts)

    bad = await checker(4).check_routing()
    assert bad.status == "fail" and "rank 2" in bad.detail and "VBT_LLM_DP_SIZE" in bad.detail
    good = await checker(2).check_routing()
    assert good.status == "pass", good.detail
    assert (await checker(1).check_routing()).status == "skip"


# ---------------------------------------------------------------------------
# ops-3: the thinking-budget probe counts the reasoning itself
# ---------------------------------------------------------------------------


class _BudgetStub:
    def __init__(self, reasoning, answer, stop):
        self.reasoning, self.answer, self.stop = reasoning, answer, stop

    async def complete(self, **kw):
        blocks = [ThinkingBlock(self.reasoning, "vllm"), TextBlock(self.answer)]
        return ModelResponse(Message("assistant", blocks), self.stop, Usage(input_tokens=60, output_tokens=1792), MODEL)


def _tokenize(count):
    def handler(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": count, "tokens": [1] * count})
        return httpx.Response(404, json={"detail": "Not Found"})
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("count, stop, status", [
    (250, StopReason.MAX_TOKENS, "warn"),   # budget enforced; the digit-heavy answer ran on
    (250, StopReason.END_TURN, "pass"),
    (2000, StopReason.END_TURN, "fail"),    # reasoning really ran past the budget
])
async def test_ops3_thinking_budget_measures_the_reasoning(count, stop, status):
    from vbt.local import check as chk

    answer = ", ".join(str(n) for n in range(2, 400))        # Qwen tokenizes digits one by one
    stub = _BudgetStub("y" * 1000, answer, stop)
    c = chk.LocalChecker(chk.CheckOptions(base_url="http://srv:8000/v1", model=MODEL, thinking_budget=256),
                         http_transport=_tokenize(count))
    c.provider, c.family, c.supports_budget = stub, "qwen3_8", True
    res = await c.check_thinking_budget()
    assert res.status == status, res.detail
    assert res.metrics["counted_with"] == "tokenize" and res.metrics["reasoning_tokens"] == count


# ---------------------------------------------------------------------------
# SEC-3 / SEC-6: vbt local serve keeps vLLM private and pins remote code
# ---------------------------------------------------------------------------


def test_sec3_serve_refuses_a_public_bind_without_the_flag(capsys):
    from vbt.cli import main

    assert main(["local", "serve", "--profile", "h100", "--host", "0.0.0.0", "--dry-run"]) == 2
    assert "/v1" in capsys.readouterr().err
    assert main(["local", "serve", "--profile", "h100", "--docker", "--driver", "580.1", "--bind", "0.0.0.0",
                 "--dry-run"]) == 2
    assert main(["local", "serve", "--profile", "h100", "--host", "0.0.0.0", "--allow-unauthenticated",
                 "--dry-run"]) == 0
    assert "protects only the /v1 routes" in capsys.readouterr().out
    assert main(["local", "serve", "--profile", "h100", "--dry-run"]) == 0


def test_sec6_remote_code_is_pinned_or_refused(capsys, monkeypatch):
    from vbt.cli import main
    from vbt.local import serve as serve_mod
    from vbt.local.profiles import docker_run_argv, load_local_profiles, resolve_serve

    profiles = load_local_profiles()
    spec = resolve_serve(profiles, "deepseek-v4")
    assert spec.unpinned_remote_code and "--revision" not in spec.server_argv
    pinned = resolve_serve(profiles, "deepseek-v4", revision="0123abc")
    argv = pinned.server_argv
    assert not pinned.unpinned_remote_code
    assert argv[argv.index("--revision") + 1] == "0123abc" and argv[argv.index("--code-revision") + 1] == "0123abc"
    assert not resolve_serve(profiles, "h100").unpinned_remote_code
    docker, _ = docker_run_argv(pinned, driver="580.1")
    assert "--shm-size=16g" in docker and "--ipc=host" not in docker
    # starting it unpinned is refused; --dry-run still prints it
    monkeypatch.setattr(serve_mod.shutil, "which", lambda exe: "/usr/bin/" + exe)
    started = []
    monkeypatch.setattr(serve_mod, "_exec", lambda argv, env: started.append(argv) or 0)
    assert main(["local", "serve", "--profile", "deepseek-v4"]) == 2
    assert "--revision" in capsys.readouterr().err and not started
    assert main(["local", "serve", "--profile", "deepseek-v4", "--revision", "0123abc"]) == 0 and started
    assert main(["local", "serve", "--profile", "deepseek-v4", "--dry-run"]) == 0


# ---------------------------------------------------------------------------
# DT1-DT7: documentation matches the shipped defaults
# ---------------------------------------------------------------------------


def test_docs_describe_the_local_default():
    readme = (ROOT_DOCS / "README.md").read_text()
    assert "Claude is the default" not in readme and "Qwen3.8-27B" in readme
    assert "Other providers must implement `web_search`" not in readme
    assert "Default profile uses current Claude models" not in (ROOT_DOCS / "docs" / "PAPER_TO_CODE.md").read_text()
    web = (ROOT_DOCS / "docs" / "WEB_SEARCH.md").read_text()
    assert "${SEARXNG_URL:-http://localhost:8888}" in web
    providers = (ROOT_DOCS / "docs" / "PROVIDERS.md").read_text()
    assert "0.55" in providers and "OPENAI_API_KEY` is never read implicitly" in providers
    deploy = (ROOT_DOCS / "deploy" / "local" / "README.md").read_text()
    assert "recipe-verified (v0.28 image)" not in deploy and "single-cell analyst" in deploy
    assert "protects only the routes" in deploy
    verification = (ROOT_DOCS / "docs" / "LOCAL_LLM_VERIFICATION.md").read_text()
    assert "pre-L2 runtime" in verification
    # DT7: the default deepseek-v4 command (vLLM 0.31) is not recipe-verified, and serve says so
    from vbt.local.profiles import load_local_profiles, resolve_serve
    profiles = load_local_profiles()
    assert profiles["deepseek-v4"]["verification"] == "supported, smoke-test"
    assert any("vbt local check" in n for n in resolve_serve(profiles, "deepseek-v4").notes)

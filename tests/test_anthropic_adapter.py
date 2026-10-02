"""Request encoding, pricing and credential checks of the Claude adapter (no network)."""

import base64
import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from vbt.agents import AgentDefinition  # noqa: E402
from vbt.config import load_config  # noqa: E402
from vbt.providers import anthropic_provider as ap  # noqa: E402
from vbt.providers.anthropic_provider import AnthropicProvider, cost_usd, price_for  # noqa: E402
from vbt.providers.base import (  # noqa: E402
    DocumentPart,
    ImagePart,
    Message,
    ModelSettings,
    StopReason,
    SystemSegment,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

TOOLS = [ToolSpec("Read", "read", {"type": "object"})]


def _p(**kw):
    return AnthropicProvider(api_key="test-key", **kw)


def test_adaptive_model_request():
    req = _p()._request(ModelSettings("anthropic", "claude-opus-5", effort="high"), "sys",
                        [Message.user("hi")], TOOLS)
    assert req["thinking"]["type"] == "adaptive"
    assert req["output_config"] == {"effort": "high"}
    assert req["tools"][0]["name"] == "Read"
    assert "temperature" not in req
    assert req["system"] == "sys" and req["cache_control"] == {"type": "ephemeral"}


def test_paper_models_use_budget_thinking_and_no_effort():
    req = _p()._request(ModelSettings("anthropic", "claude-sonnet-4-5", effort=None, thinking=True,
                                      max_tokens=64000), "sys", [Message.user("hi")], [])
    assert req["thinking"]["type"] == "enabled" and req["thinking"]["budget_tokens"] < 64000
    assert "output_config" not in req
    assert "betas" not in req  # no tools -> no interleaved-thinking beta needed
    req = _p()._request(ModelSettings("anthropic", "claude-haiku-4-5", effort=None, thinking=False,
                                      max_tokens=16000), "sys", [Message.user("hi")], [])
    assert "thinking" not in req


def test_interleaved_thinking_beta_for_budget_models_with_tools():
    s = ModelSettings("anthropic", "claude-sonnet-4-5-20250929", effort=None, thinking=True, max_tokens=64000)
    req = _p()._request(s, "sys", [Message.user("hi")], TOOLS)
    assert ap.INTERLEAVED_THINKING_BETA == "interleaved-thinking-2025-05-14"
    assert ap.INTERLEAVED_THINKING_BETA in req["betas"]
    # adaptive models interleave on their own; thinking off needs no beta either
    for s2 in (ModelSettings("anthropic", "claude-opus-4-6", effort="high"),
               ModelSettings("anthropic", "claude-haiku-4-5", effort=None, thinking=False)):
        assert ap.INTERLEAVED_THINKING_BETA not in _p()._request(s2, "sys", [Message.user("hi")], TOOLS).get("betas", [])


def test_thinking_budget_from_paper_profile():
    cfg = load_config(["paper"])
    budgets = {}
    for tier in ("orchestrator", "scientist"):
        settings = AgentDefinition(name="x", description="", prompt="", tier=tier).settings(cfg)
        req = _p()._request(settings, "sys", [Message.user("hi")], TOOLS)
        budgets[tier] = req["thinking"]["budget_tokens"]
        assert ap.INTERLEAVED_THINKING_BETA in req["betas"]
        assert settings.extra["context_window_tokens"] == 200000
    assert budgets == {"orchestrator": 32000, "scientist": 16000}
    support = AgentDefinition(name="x", description="", prompt="", tier="support").settings(cfg)
    assert "thinking" not in _p()._request(support, "sys", [Message.user("hi")], [])


def test_system_segments_get_cache_control_on_last_cached_segment():
    system = [SystemSegment("static upstream prompt", cache=True), SystemSegment("more static", cache=True),
              SystemSegment("Today's date: 2026-10-02; run dir /x")]
    req = _p()._request(ModelSettings("anthropic", "claude-opus-5-5"), system, [Message.user("hi")], [])
    blocks = req["system"]
    assert [b["text"] for b in blocks] == ["static upstream prompt", "more static", "Today's date: 2026-10-02; run dir /x"]
    assert "cache_control" not in blocks[0] and "cache_control" not in blocks[2]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert req["cache_control"] == {"type": "ephemeral"}  # automatic caching of the tail is kept
    req = _p(system_cache_ttl="1h")._request(ModelSettings("anthropic", "claude-opus-5-5"), system,
                                            [Message.user("hi")], [])
    assert req["system"][1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    req = _p(prompt_caching=False)._request(ModelSettings("anthropic", "claude-opus-5-5"), system,
                                            [Message.user("hi")], [])
    assert all("cache_control" not in b for b in req["system"]) and "cache_control" not in req


def test_history_encoding_drops_foreign_thinking_and_keeps_own():
    own = ThinkingBlock("x", "anthropic", native={"type": "thinking", "thinking": "x", "signature": "sig"})
    foreign = ThinkingBlock("y", "other", native={"type": "reasoning"})
    msgs = [Message.user("q"),
            Message("assistant", [own, foreign, TextBlock("t"), ToolCall("id1", "Read", {"file_path": "a"})]),
            Message("user", [ToolResult("id1", "", False)])]
    enc = _p()._encode_messages(msgs)
    types = [b["type"] for b in enc[1]["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert enc[2]["content"][0]["content"] == "(empty result)"


def test_image_and_document_tool_results_are_encoded_as_blocks():
    png = base64.b64encode(b"\x89PNG fake").decode()
    pdf = base64.b64encode(b"%PDF-1.4 fake").decode()
    msgs = [Message.user("q"),
            Message("assistant", [ToolCall("t1", "Read", {"file_path": "umap.png"}),
                                  ToolCall("t2", "Read", {"file_path": "label.pdf"})]),
            Message("user", [ToolResult("t1", [TextBlock("umap.png (800x600)"), ImagePart("image/png", png, "umap.png")]),
                             ToolResult("t2", [DocumentPart(data_b64=pdf, title="label.pdf")])])]
    enc = _p()._encode_messages(msgs)[2]["content"]
    assert enc[0]["content"] == [
        {"type": "text", "text": "umap.png (800x600)"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
    ]
    assert enc[1]["content"] == [{"type": "document", "title": "label.pdf",
                                  "source": {"type": "base64", "media_type": "application/pdf", "data": pdf}}]


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
                                   "claude-sonnet-5-5"])
def test_fallback_by_family(model):
    req = _p()._request(ModelSettings("anthropic", model), "sys", [Message.user("hi")], [])
    assert req["fallbacks"] == "default" and ap.FALLBACK_BETA in req["betas"]


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5", "claude-sonnet-4-5"])
def test_no_fallback_outside_families(model):
    req = _p()._request(ModelSettings("anthropic", model, effort=None), "sys", [Message.user("hi")], [])
    assert "fallbacks" not in req and ap.FALLBACK_BETA not in req.get("betas", [])
    assert "fallbacks" not in _p(refusal_fallback=False)._request(
        ModelSettings("anthropic", "claude-opus-5-5"), "sys", [Message.user("hi")], [])


def test_extra_keys_are_never_forwarded():
    s = ModelSettings("anthropic", "claude-opus-5-5", extra={
        "agent_name": "cso", "context_window_tokens": 123, "thinking_budget": 5000, "bogus_key": 1})
    req = _p()._request(s, "sys", [Message.user("hi")], [])
    flat = repr(req)
    assert "bogus_key" not in flat and "agent_name" not in flat and "context_window_tokens" not in flat
    # one-off requests (compaction summaries) opt out of caching entirely
    one_off = ModelSettings("anthropic", "claude-opus-5-5", extra={"prompt_cache": False})
    req = _p()._request(one_off, [SystemSegment("static", cache=True)], [Message.user("hi")], [])
    assert "cache_control" not in req and "cache_control" not in req["system"][0]


def test_server_side_context_editing_request_shape():
    s = ModelSettings("anthropic", "claude-opus-5", extra={"context_management": True})
    req = _p()._request(s, "sys", [Message.user("hi")], TOOLS)
    assert req["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919"}]}
    assert ap.CONTEXT_MANAGEMENT_BETA == "context-management-2025-06-27" and ap.CONTEXT_MANAGEMENT_BETA in req["betas"]
    req = _p(context_editing={"clear_tool_inputs": True})._request(
        ModelSettings("anthropic", "claude-opus-5"), "sys", [Message.user("hi")], TOOLS)
    assert req["context_management"] == {"edits": [{"type": "clear_tool_uses_20250919", "clear_tool_inputs": True}]}
    assert "context_management" not in _p()._request(ModelSettings("anthropic", "claude-opus-5"), "sys",
                                                     [Message.user("hi")], TOOLS)
    assert _p().capabilities("claude-opus-5").server_context_management


def test_capabilities_and_context_window():
    p = _p()
    assert p.capabilities().images and p.capabilities().documents
    assert p.capabilities("claude-opus-5-5").history_bound_thinking
    assert not p.capabilities("claude-sonnet-4-5").history_bound_thinking
    assert p.context_window("claude-opus-5-5") == 1_000_000
    assert p.context_window("claude-haiku-4-5-20251001") == 200_000
    assert p.context_window("claude-sonnet-4-5") is None  # not in the skill table: profiles set it


# ---------------------------------------------------------------- pricing


def test_cost():
    assert abs(cost_usd("claude-opus-5", Usage(1_000_000, 1_000_000)) - 30.0) < 1e-9
    assert abs(cost_usd("claude-haiku-4-5", Usage(0, 0, cache_read_tokens=1_000_000)) - 0.1) < 1e-9


def test_cache_read_rates_differ_by_model():
    reads = Usage(cache_read_tokens=1_000_000)
    assert cost_usd("claude-opus-5-5", reads) == pytest.approx(0.20)   # 0.05x of $4
    assert cost_usd("claude-fable-5-1", reads) == pytest.approx(0.25)  # 0.025x of $10
    assert cost_usd("claude-fable-5", reads) == pytest.approx(1.00)    # 0.1x
    assert cost_usd("claude-opus-5", reads) == pytest.approx(0.50)
    assert cost_usd("claude-mythos-5-1", Usage(1_000_000, 1_000_000)) == pytest.approx(60.0)


def test_cache_writes_by_ttl_and_web_search_fee():
    u = Usage(cache_write_tokens=1_000_000, cache_write_1h_tokens=400_000)
    # 600k at the 5-minute rate ($5/MTok) + 400k at the 1-hour rate ($8/MTok) on Opus 5.5
    assert cost_usd("claude-opus-5-5", u) == pytest.approx(0.6 * 5.0 + 0.4 * 8.0)
    assert cost_usd("claude-fable-5-1", Usage(cache_write_tokens=1_000_000, cache_write_1h_tokens=1_000_000)) \
        == pytest.approx(20.0)
    assert cost_usd("claude-sonnet-5", Usage(server_tool_requests={"web_search": 3})) == pytest.approx(0.03)


def test_unknown_model_warns_once_and_prices_conservatively(caplog):
    ap._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="vbt.providers.anthropic_provider"):
        c1 = cost_usd("claude-unknown-9", Usage(1_000_000, 0))
        cost_usd("claude-unknown-9", Usage(1_000_000, 0))
    warnings = [r for r in caplog.records if "claude-unknown-9" in r.getMessage()]
    assert len(warnings) == 1
    assert c1 == pytest.approx(10.0)  # most expensive listed input price
    # dated / platform variants resolve to the base model without warnings
    assert price_for("claude-sonnet-4-5-20250929") is ap.PRICES["claude-sonnet-4-5"]
    assert price_for("us.anthropic.claude-haiku-4-5-20251001-v1:0") is ap.PRICES["claude-haiku-4-5"]


def test_price_overrides_from_provider_options():
    p = _p(prices={"claude-opus-5-5": {"input": 3.0}, "my-local-claude": {"input": 1.0, "output": 2.0}})
    assert p.cost("claude-opus-5-5", Usage(1_000_000, 1_000_000)) == pytest.approx(3.0 + 20.0)
    assert p.cost("my-local-claude", Usage(0, 0, cache_read_tokens=1_000_000)) == pytest.approx(0.1)


def _final(*, model, content, usage, stop="end_turn"):
    return SimpleNamespace(model=model, content=content, usage=usage, stop_reason=stop, stop_details=None,
                           _request_id="req_1")


def _blk(**kw):
    ns = SimpleNamespace(**kw)
    ns.to_dict = lambda: dict(kw)
    return ns


def test_iterations_are_priced_per_model_and_fallback_flagged():
    it = [SimpleNamespace(type="message", model="claude-fable-5-1", input_tokens=1000, output_tokens=200,
                          cache_read_input_tokens=0, cache_creation_input_tokens=0, cache_creation=None),
          SimpleNamespace(type="fallback_message", model="claude-opus-5", input_tokens=1000, output_tokens=500,
                          cache_read_input_tokens=0, cache_creation_input_tokens=0, cache_creation=None)]
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0, cache_creation=None, server_tool_use=None, iterations=it)
    content = [_blk(type="text", text="partial "), _blk(type="tool_use", id="t0", name="Read", input={}),
               _blk(type="fallback", from_={"model": "claude-fable-5-1"}), _blk(type="text", text="answer")]
    resp = _p()._response(_final(model="claude-opus-5", content=content, usage=usage),
                          ModelSettings("anthropic", "claude-fable-5-1"))
    expected = (1000 * 10 + 200 * 50) / 1e6 + (1000 * 5 + 500 * 25) / 1e6
    assert resp.cost_usd == pytest.approx(expected)
    assert resp.fallback_used and resp.served_model == "claude-opus-5"
    assert [i["model"] for i in resp.iterations] == ["claude-fable-5-1", "claude-opus-5"]
    # the declined partial's tool_use is not kept (it must not be executed or replayed)
    assert not resp.message.tool_calls and resp.message.text == "partial \n\nanswer"


def test_usage_decoding_cache_ttl_and_server_tools():
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=100,
                            cache_creation_input_tokens=300,
                            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200),
                            server_tool_use=SimpleNamespace(web_search_requests=2, web_fetch_requests=0),
                            iterations=None)
    resp = _p()._response(_final(model="claude-opus-5-5", content=[_blk(type="text", text="x")], usage=usage,
                                 stop="model_context_window_exceeded"),
                          ModelSettings("anthropic", "claude-opus-5-5"))
    u = resp.usage
    assert (u.cache_write_tokens, u.cache_write_1h_tokens, u.server_tool_requests) == (300, 200, {"web_search": 2})
    assert resp.stop_reason is StopReason.CONTEXT_EXCEEDED
    assert resp.cost_usd == pytest.approx((10 * 4 + 5 * 20 + 100 * 0.2 + 100 * 5 + 200 * 8) / 1e6 + 0.02)


# ---------------------------------------------------------------- credentials


def test_blank_key_fails_check_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    p = AnthropicProvider(api_key="")
    assert "blank" in p.check_credentials()
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert "no Anthropic credentials" in p.check_credentials()
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    assert p.check_credentials() is None
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert p.check_credentials() is None
    assert _p().check_credentials() is None


# ---------------------------------------------------------------- web search


async def test_web_search_fee_cache_tokens_and_domain_filters():
    p = _p(web_search_model="claude-sonnet-5")
    seen = {}

    async def create(**kw):
        seen.update(kw)
        results = [SimpleNamespace(title="EGFR review", url="https://pubmed.ncbi.nlm.nih.gov/1", page_age="2025")]
        return SimpleNamespace(
            model="claude-sonnet-5",
            content=[SimpleNamespace(type="web_search_tool_result", content=results),
                     SimpleNamespace(type="text", text="EGFR is a validated target.")],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=2000,
                                  cache_creation_input_tokens=0, cache_creation=None,
                                  server_tool_use=SimpleNamespace(web_search_requests=3, web_fetch_requests=0)))

    p.client.messages.create = create
    out = await p.web_search("EGFR lung", max_results=5, allowed_domains=["pubmed.ncbi.nlm.nih.gov"])
    tool = seen["tools"][0]
    assert tool["type"] == "web_search_20260209" and tool["allowed_domains"] == ["pubmed.ncbi.nlm.nih.gov"]
    assert "blocked_domains" not in tool
    assert out["results"][0]["url"].startswith("https://pubmed")
    assert out["cost_usd"] == pytest.approx((1000 * 2 + 100 * 10 + 2000 * 0.2) / 1e6 + 3 * 0.01)
    with pytest.raises(Exception, match="not both"):
        await p.web_search("x", allowed_domains=["a.org"], blocked_domains=["b.org"])
    await p.aclose()

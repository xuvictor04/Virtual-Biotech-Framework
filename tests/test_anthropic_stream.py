"""Drive the real Anthropic SDK streaming path against a local fake Messages API."""

import base64
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("anthropic")

from vbt.providers.anthropic_provider import FALLBACK_BETA, INTERLEAVED_THINKING_BETA, AnthropicProvider  # noqa: E402
from vbt.providers.base import (  # noqa: E402
    ContextOverflowError,
    ImagePart,
    Message,
    ModelSettings,
    ProviderError,
    RetryableProviderError,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from vbt.providers.retry import RetryPolicy, complete_with_retry  # noqa: E402

REQUESTS = []
RESPONSES: deque = deque()


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _start(model="claude-opus-5", usage=None):
    return {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": usage or {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 40,
                           "cache_creation_input_tokens": 0}}}


def _tool_turn(model="claude-opus-5"):
    return [
        _start(model),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking genetics."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "mcp__genetics__query_l2g_predictions",
                           "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": "{\"gene\": \"EGFR\"}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
         "usage": {"output_tokens": 50}},
        {"type": "message_stop"},
    ]


def _thinking_turn():
    return [
        _start("claude-opus-5-5"),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Weigh L2G "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "vs coloc."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Done."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]


def _error_event(etype="overloaded_error", message="Overloaded"):
    return f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': etype, 'message': message}})}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        REQUESTS.append({"path": self.path, "body": body, "beta": self.headers.get("anthropic-beta") or ""})
        kind, spec = RESPONSES.popleft() if RESPONSES else ("sse", _tool_turn())
        headers = {}
        if kind == "sse":
            status, payload = 200, _sse(spec)
        elif kind == "sse_then_error":
            status, payload = 200, _sse(spec) + _error_event()
        elif kind == "drop":  # promise more bytes than we send, then close the connection
            payload = _sse(spec)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload) + 5000))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True
            return
        else:  # ("json", (status, body_dict, headers))
            status, obj, headers = spec
            payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "text/event-stream" if kind.startswith("sse") else "application/json")
        self.send_header("content-length", str(len(payload)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    RESPONSES.clear()
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    RESPONSES.clear()


@pytest.fixture
def make_provider(server, monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    def make(**kw):
        return AnthropicProvider(api_key="k", base_url=server, max_retries=0, **kw)

    return make


def _err(status, etype, message, headers=None):
    return ("json", (status, {"type": "error", "error": {"type": etype, "message": message}}, headers or {}))


SETTINGS = ModelSettings("anthropic", "claude-opus-5", effort="high")
TOOLS = [ToolSpec("mcp__genetics__query_l2g_predictions", "L2G", {"type": "object"})]


async def test_stream_decode_and_fallback_header(make_provider):
    p = make_provider()
    chunks = []
    resp = await p.complete(settings=SETTINGS, system="sys", messages=[Message.user("EGFR?")], tools=TOOLS,
                            on_text=chunks.append)
    await p.aclose()
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.message.text == "Checking genetics." and "".join(chunks) == "Checking genetics."
    call = resp.message.tool_calls[0]
    assert call.name == "mcp__genetics__query_l2g_predictions" and call.input == {"gene": "EGFR"}
    assert resp.usage.input_tokens == 100 and resp.usage.cache_read_tokens == 40 and resp.usage.output_tokens == 50
    assert resp.served_model == "claude-opus-5" and not resp.fallback_used
    req = REQUESTS[-1]
    assert req["body"]["fallbacks"] == "default" and FALLBACK_BETA in req["beta"]
    assert req["body"]["thinking"]["type"] == "adaptive" and req["body"]["stream"] is True


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5"])
async def test_every_fallback_family_sends_the_beta(make_provider, model):
    RESPONSES.append(("sse", _tool_turn(model)))
    p = make_provider()
    await p.complete(settings=ModelSettings("anthropic", model), system="s", messages=[Message.user("q")], tools=[])
    await p.aclose()
    assert FALLBACK_BETA in REQUESTS[-1]["beta"] and REQUESTS[-1]["body"]["fallbacks"] == "default"


async def test_thinking_deltas_stream_to_on_thinking(make_provider):
    RESPONSES.append(("sse", _thinking_turn()))
    p = make_provider()
    thoughts, texts = [], []
    resp = await p.complete(settings=ModelSettings("anthropic", "claude-opus-5-5"), system="s",
                            messages=[Message.user("q")], tools=[], on_text=texts.append, on_thinking=thoughts.append)
    await p.aclose()
    assert "".join(thoughts) == "Weigh L2G vs coloc." and "".join(texts) == "Done."
    block = resp.message.content[0]
    assert block.type == "thinking" and block.native["signature"] == "sig123"


async def test_interleaved_beta_and_image_tool_result_reach_the_api(make_provider):
    RESPONSES.append(("sse", _tool_turn("claude-sonnet-4-5")))
    png = base64.b64encode(b"\x89PNG fake").decode()
    history = [Message.user("Check the UMAP"),
               Message("assistant", [ToolCall("toolu_0", "Read", {"file_path": "umap.png"})]),
               Message("user", [ToolResult("toolu_0", [TextBlock("umap.png"), ImagePart("image/png", png, "umap.png")])])]
    p = make_provider()
    await p.complete(settings=ModelSettings("anthropic", "claude-sonnet-4-5", effort=None, max_tokens=64000,
                                            extra={"thinking_budget": 16000}),
                     system="s", messages=history, tools=[ToolSpec("Read", "read", {"type": "object"})])
    await p.aclose()
    req = REQUESTS[-1]
    assert INTERLEAVED_THINKING_BETA in req["beta"]
    assert req["body"]["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    result = req["body"]["messages"][2]["content"][0]
    assert result["type"] == "tool_result"
    assert result["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}}


async def test_overloaded_error_event_mid_stream_is_retryable(make_provider):
    RESPONSES.append(("sse_then_error", _tool_turn()[:3]))
    p = make_provider()
    chunks = []
    with pytest.raises(RetryableProviderError) as ei:
        await p.complete(settings=SETTINGS, system="s", messages=[Message.user("q")], tools=TOOLS,
                         on_text=chunks.append)
    await p.aclose()
    assert ei.value.status == 529 and "overloaded" in str(ei.value).lower()
    assert "".join(chunks) == "Checking genetics."  # streaming had started before the failure


async def test_prompt_too_long_is_context_overflow(make_provider):
    RESPONSES.append(_err(400, "invalid_request_error", "prompt is too long: 212034 tokens > 200000 maximum"))
    p = make_provider()
    with pytest.raises(ContextOverflowError):
        await p.complete(settings=SETTINGS, system="s", messages=[Message.user("q")], tools=[])
    await p.aclose()


async def test_rate_limit_carries_retry_after_and_auth_is_fatal(make_provider):
    RESPONSES.append(_err(429, "rate_limit_error", "slow down", {"retry-after": "7"}))
    RESPONSES.append(_err(401, "authentication_error", "invalid x-api-key"))
    RESPONSES.append(_err(400, "invalid_request_error", "messages: roles must alternate"))
    p = make_provider()
    with pytest.raises(RetryableProviderError) as ei:
        await p.complete(settings=SETTINGS, system="s", messages=[Message.user("q")], tools=[])
    assert ei.value.retry_after == 7.0 and ei.value.status == 429
    with pytest.raises(ProviderError) as ei2:
        await p.complete(settings=SETTINGS, system="s", messages=[Message.user("q")], tools=[])
    assert not isinstance(ei2.value, RetryableProviderError)
    with pytest.raises(ProviderError) as ei3:
        await p.complete(settings=SETTINGS, system="s", messages=[Message.user("q")], tools=[])
    assert not isinstance(ei3.value, (RetryableProviderError, ContextOverflowError))
    await p.aclose()


async def test_complete_with_retry_recovers_from_429_529_and_dropped_connection(make_provider):
    RESPONSES.append(_err(429, "rate_limit_error", "slow down", {"retry-after": "3"}))
    RESPONSES.append(_err(529, "overloaded_error", "Overloaded"))
    RESPONSES.append(("sse_then_error", _tool_turn()[:3]))
    RESPONSES.append(("drop", _tool_turn()[:3]))
    RESPONSES.append(("sse", _tool_turn()))
    p = make_provider()
    sleeps, retries = [], []

    async def fake_sleep(s):
        sleeps.append(s)

    resp = await complete_with_retry(
        p, policy=RetryPolicy(attempts=6, base_delay_s=1.0, jitter=0.0), sleep=fake_sleep,
        on_retry=lambda attempt, exc, delay: retries.append(type(exc).__name__),
        settings=SETTINGS, system="s", messages=[Message.user("q")], tools=TOOLS)
    await p.aclose()
    assert resp.retries == 4 and resp.stop_reason is StopReason.TOOL_USE
    assert sleeps == [3.0, 2.0, 4.0, 8.0]  # retry-after honoured, then exponential backoff
    assert retries == ["RetryableProviderError"] * 4
    # every attempt re-sent the identical request
    bodies = [json.dumps(r["body"], sort_keys=True) for r in REQUESTS[-5:]]
    assert len(set(bodies)) == 1

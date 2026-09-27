"""Drive the real Anthropic SDK streaming path against a local fake Messages API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vbt.providers.anthropic_provider import AnthropicProvider
from vbt.providers.base import Message, ModelSettings, StopReason, ToolSpec

REQUESTS = []


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _tool_turn():
    return [
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                              "model": "claude-opus-5", "content": [], "stop_reason": None,
                                              "stop_sequence": None,
                                              "usage": {"input_tokens": 100, "output_tokens": 1,
                                                        "cache_read_input_tokens": 40,
                                                        "cache_creation_input_tokens": 0}}},
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


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        REQUESTS.append({"path": self.path, "body": body, "beta": self.headers.get("anthropic-beta")})
        payload = _sse(_tool_turn())
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


async def test_stream_decode_and_fallback_header(server, monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    p = AnthropicProvider(api_key="k", base_url=server, max_retries=0)
    chunks = []
    resp = await p.complete(settings=ModelSettings("anthropic", "claude-opus-5", effort="high"),
                            system="sys", messages=[Message.user("EGFR?")],
                            tools=[ToolSpec("mcp__genetics__query_l2g_predictions", "L2G", {"type": "object"})],
                            on_text=chunks.append)
    await p.aclose()
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.message.text == "Checking genetics." and "".join(chunks) == "Checking genetics."
    call = resp.message.tool_calls[0]
    assert call.name == "mcp__genetics__query_l2g_predictions" and call.input == {"gene": "EGFR"}
    assert resp.usage.input_tokens == 100 and resp.usage.cache_read_tokens == 40 and resp.usage.output_tokens == 50
    req = REQUESTS[-1]
    assert req["body"]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["beta"]
    assert req["body"]["thinking"]["type"] == "adaptive" and req["body"]["stream"] is True

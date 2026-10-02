"""MCP bridge: environment policy, error semantics, crash recovery, timeouts, startup retries.

Uses small stdio FastMCP fixture servers (tests/fixtures); no network, no data.
"""

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from vbt.config import load_config
from vbt.envpolicy import child_env, redact
from vbt.tools.base import ToolFailure
from vbt.tools.mcp_bridge import MCPBridge, MCPServerConfig, is_transport_error, tool_result_error

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ECHO = str(FIXTURES / "echo_env_mcp_server.py")
CRASHING = str(FIXTURES / "crashing_mcp_server.py")

needs_fastmcp = pytest.mark.skipif(
    importlib.util.find_spec("fastmcp") is None or importlib.util.find_spec("mcp") is None,
    reason="fastmcp/mcp not installed (pip install -e '.[mcp]')")

FAST = {"start_backoff_s": 0.01, "start_backoff_factor": 1.0, "start_timeout_s": 60}


def server(name, script, *args, **kw):
    return MCPServerConfig(name=name, command=sys.executable, args=["-E", script, *map(str, args)], **kw)


class Events(list):
    def __call__(self, kind, **data):
        self.append((kind, data))

    def kinds(self):
        return [k for k, _ in self]


# --------------------------------------------------------------------------- pure


def test_tool_result_error_envelopes():
    assert tool_result_error({"error": "Data not loaded"}) == "Data not loaded"
    assert tool_result_error({"success": False}) == "Tool returned an unsuccessful result"
    assert tool_result_error({"ok": False, "errors": ["a", "b"]}) == "a; b"
    assert tool_result_error(json.dumps({"error": "boom"})) == "boom"
    assert tool_result_error([{"type": "text", "text": '{"error": "x"}'}]) == "x"
    # explicit empty lookups and nested error columns are not failures
    assert tool_result_error({"error": "Target FOO not found", "results": []}) is None
    assert tool_result_error({"error": "No cells found for filter: tissue == 'x'"}) is None
    assert tool_result_error({"results": [{"error": 0.1}]}) is None
    assert tool_result_error("plain text") is None
    # a missing dataset is a failure even if phrased "not found"
    assert tool_result_error({"error": "Dataset file not found: /x.parquet"})


def test_transport_error_classification():
    import anyio

    assert is_transport_error(anyio.ClosedResourceError())
    assert is_transport_error(anyio.BrokenResourceError())
    assert is_transport_error(ConnectionError("x"))

    class McpError(Exception):
        pass

    assert is_transport_error(McpError("Connection closed"))
    assert not is_transport_error(McpError("Invalid params"))
    assert not is_transport_error(ValueError("x"))


def test_child_env_strips_secrets_and_redacts(tmp_path):
    base = {"PATH": "/bin", "LANG": "C", "LC_ALL": "C", "HOME": "/root", "CONDA_PREFIX": "/c",
            "ANTHROPIC_API_KEY": "sk-ant-abcdefgh123", "OPENAI_API_KEY": "sk-openai-12345678",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-1234567", "GITHUB_TOKEN": "ghp_12345678", "DB_PASSWORD": "hunter22xx",
            "NCBI_API_KEY": "ncbi-12345678", "RANDOM_VAR": "x", "OMP_NUM_THREADS": "4", "R_LIBS": "/r",
            "CONDA_TOKEN": "conda-tok-123456"}
    env = child_env(base, extra={"OPEN_TARGETS_DATA_PATH": "/ot"})
    assert env["PATH"] == "/bin" and env["LC_ALL"] == "C" and env["OMP_NUM_THREADS"] == "4" and env["R_LIBS"] == "/r"
    assert env["OPEN_TARGETS_DATA_PATH"] == "/ot"
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "DB_PASSWORD",
              "NCBI_API_KEY", "RANDOM_VAR", "CONDA_TOKEN"):
        assert k not in env, k
    env = child_env(base, passthrough=["NCBI_API_KEY"], home=tmp_path / "home", tmp=tmp_path / "tmp")
    assert env["NCBI_API_KEY"] == "ncbi-12345678"
    assert env["HOME"] == str(tmp_path / "home") and env["TMPDIR"] == str(tmp_path / "tmp")
    assert (tmp_path / "home").is_dir() and (tmp_path / "tmp").is_dir()

    text = "key=sk-ant-abcdefgh123 and ghp_12345678 but not x"
    red = redact(text, base)
    assert "sk-ant-abcdefgh123" not in red and "ghp_12345678" not in red
    assert "[redacted:ANTHROPIC_API_KEY]" in red and red.endswith("but not x")
    assert redact(None, base) is None


def test_pubmed_literature_ceiling(monkeypatch):
    pytest.importorskip("fastmcp")
    pytest.importorskip("httpx")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vbt.mcp_servers import pubmed_server as pm

    monkeypatch.delenv(pm.MAXDATE_ENV, raising=False)
    assert pm.literature_ceiling() is None and pm._date_params() == {}
    monkeypatch.setenv(pm.MAXDATE_ENV, "2025/01/31")
    assert pm.literature_ceiling() == (2025, 1, 31)
    assert pm._date_params() == {"datetype": "pdat", "mindate": "1800", "maxdate": "2025/01/31"}

    calls = []

    class Resp:
        def __init__(self, payload=None, text=""):
            self.payload, self.text = payload, text

        def json(self):
            return self.payload

    def fake_get(endpoint, params):
        calls.append((endpoint, params))
        if endpoint == "esearch.fcgi":
            return Resp({"esearchresult": {"count": "3", "idlist": ["1", "2", "3"]}})
        return Resp({"result": {"1": {"pubdate": "2024 Mar 5"}, "2": {"pubdate": "2025 Jun", "epubdate": ""},
                                "3": {"pubdate": "2025", "epubdate": "2024 Dec 20"}}})

    monkeypatch.setattr(pm, "_get", fake_get)
    out = pm.search("B7-H3 lung")
    assert calls[0][1]["datetype"] == "pdat" and calls[0][1]["maxdate"] == "2025/01/31"
    assert [r["pmid"] for r in out["results"]] == ["1", "3"]  # 3 was e-published before the ceiling
    assert out["withheld"] == ["2"]

    xml = """<PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>11</PMID><Article><Journal><JournalIssue>
        <PubDate><Year>2024</Year><Month>Nov</Month></PubDate></JournalIssue><Title>J</Title></Journal>
        <ArticleTitle>Old</ArticleTitle></Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>12</PMID><Article><Journal><JournalIssue>
        <PubDate><Year>2025</Year><Month>Mar</Month></PubDate></JournalIssue><Title>J</Title></Journal>
        <ArticleTitle>New</ArticleTitle></Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>13</PMID><Article><Journal><JournalIssue>
        <PubDate><Year>2025</Year></PubDate></JournalIssue><Title>J</Title></Journal>
        <ArticleTitle>Year only</ArticleTitle></Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""
    arts, withheld = pm.parse_articles(xml, pm.literature_ceiling())
    assert [a["pmid"] for a in arts] == ["11"]
    assert [w["pmid"] for w in withheld] == ["12", "13"]
    arts, withheld = pm.parse_articles(xml, None)
    assert len(arts) == 3 and withheld == []
    monkeypatch.setenv(pm.MAXDATE_ENV, "last year")
    with pytest.raises(ValueError):
        pm.literature_ceiling()


# --------------------------------------------------------------------------- live fixture servers


@needs_fastmcp
async def test_runtime_env_errors_timeout_images_and_logs(tmp_path, monkeypatch):
    """Through Runtime.start_mcp: env policy plus result semantics on one echo server."""
    from vbt.runtime import Runtime
    from vbt.session import Run

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret-0000")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok-should-not-leak")
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-passthrough-1234")
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(tmp_path / "ot"))
    config = load_config(["mock"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")}})
    config["mcp_servers"] = {"servers": [{
        "name": "echo", "command": sys.executable, "args": ["-E", ECHO], "timeout_s": 1.0,
        "env_passthrough": ["NCBI_API_KEY"], "enabled": True}]}
    run = Run(tmp_path / "runs")
    rt = Runtime(config, run)
    try:
        failures = await rt.start_mcp()
        assert failures == {}
        assert rt.registry.get("mcp__echo__echo_env") is not None
        names = ["MCP_OUTPUT_DIR", "VBT_RUN_DIR", "OPEN_TARGETS_DATA_PATH", "ANTHROPIC_API_KEY",
                 "SOME_SERVICE_TOKEN", "NCBI_API_KEY", "PATH"]
        out = json.loads(await rt.mcp.call("echo", "echo_env", {"names": names}))
        vals = out["values"]
        assert vals["MCP_OUTPUT_DIR"] == str(run.mcp_output_dir)
        assert vals["VBT_RUN_DIR"] == str(run.dir)
        assert vals["OPEN_TARGETS_DATA_PATH"] == str(tmp_path / "ot")
        assert vals["NCBI_API_KEY"] == "ncbi-passthrough-1234"
        assert vals["ANTHROPIC_API_KEY"] is None and "ANTHROPIC_API_KEY" not in out["all_names"]
        assert vals["SOME_SERVICE_TOKEN"] is None
        assert vals["PATH"]

        with pytest.raises(ToolFailure, match="kaboom"):
            await rt.mcp.call("echo", "raise_error", {"message": "kaboom"})
        with pytest.raises(ToolFailure, match="Open Targets data could not be loaded") as exc:
            await rt.mcp.call("echo", "legacy_error", {})
        assert json.loads(str(exc.value))["status"] == "tool_error"
        with pytest.raises(ToolFailure, match="unsuccessful"):
            await rt.mcp.call("echo", "legacy_unsuccessful", {})
        assert "not found" in await rt.mcp.call("echo", "empty_lookup", {"symbol": "XYZ"})
        assert "EGFR" in await rt.mcp.call("echo", "ok_result", {})

        # Through the registered tool and the runtime's error handling path.
        tool = rt.registry.get("mcp__echo__legacy_error")
        from vbt.tools.base import ToolContext
        ctx = ToolContext(agent="genomics-analyst", run=run, runtime=rt)
        with pytest.raises(ToolFailure):
            await tool(ctx, {})

        t0 = time.monotonic()
        with pytest.raises(ToolFailure) as exc:
            await rt.mcp.call("echo", "slow", {"seconds": 10})
        msg = str(exc.value)
        assert "echo.slow timed out after 1s" in msg and "count_cells first" in msg
        assert time.monotonic() - t0 < 8
        assert await rt.mcp.call("echo", "say", {"text": "still alive"}) == "still alive"

        img = await rt.mcp.call("echo", "image", {})
        try:
            from vbt.providers.base import ImagePart  # noqa: F401  (content parts, provider layer)
            assert isinstance(img, list) and any(getattr(p, "type", "") == "image" for p in img)
        except ImportError:
            assert isinstance(img, str) and "[image omitted: image/png" in img

        status = rt.mcp.status()["echo"]
        assert status["state"] == "ready" and status["in_flight"] == 0 and status["restarts"] == 0
        log_text = Path(status["log"]).read_text()
        assert "echo-env fixture server starting" in log_text
        assert "say called with 'still alive'" in log_text
    finally:
        await rt.aclose()


@needs_fastmcp
async def test_crash_restart_retry_and_restart_limit(tmp_path):
    events = Events()
    bridge = MCPBridge([server("crash", CRASHING, tmp_path / "s1"),
                        server("doomed", CRASHING, tmp_path / "s2")],
                       log_dir=tmp_path / "logs", options={**FAST, "max_restarts": 1}, on_event=events)
    try:
        await bridge.start()
        assert bridge.failures == {}
        # os._exit during the call -> restart once -> the retried call succeeds
        out = await bridge.call("crash", "crash_once", {})
        assert out == "survived after restart (start #2)"
        st = bridge.status()["crash"]
        assert st["restarts"] == 1 and st["state"] == "ready"
        assert "mcp_crash" in events.kinds() and "mcp_restart" in events.kinds()
        assert await bridge.call("crash", "ping", {}) == "pong from start #2"

        # A server that crashes on every call: restart, retry fails, then the limit stops restarts.
        with pytest.raises(ToolFailure, match="failed again after a restart"):
            await bridge.call("doomed", "crash_always", {})
        with pytest.raises(ToolFailure, match="restart limit"):
            await bridge.call("doomed", "crash_always", {})
        assert bridge.status()["doomed"]["state"] == "failed"
        assert "doomed" in bridge.failures
        starts = (tmp_path / "s2" / "starts").read_text()
        with pytest.raises(ToolFailure, match="unavailable"):
            await bridge.call("doomed", "ping", {})
        assert (tmp_path / "s2" / "starts").read_text() == starts, "no restart after the limit"
    finally:
        await bridge.aclose()
    assert all(s["state"] == "closed" for s in bridge.status().values())


@needs_fastmcp
async def test_startup_retry_stderr_log_and_lazy_retry(tmp_path):
    events = Events()
    bridge = MCPBridge([
        server("flaky", CRASHING, tmp_path / "flaky", "--fail-starts", 2),       # ok on 3rd attempt
        server("broken", CRASHING, tmp_path / "broken", "--fail-starts", 99),    # never starts
        server("late", CRASHING, tmp_path / "late", "--fail-starts", 2, start_timeout_s=60),
        MCPServerConfig("empty", command=""),
        MCPServerConfig("noscript", command=sys.executable, args=[str(tmp_path / "missing.py")]),
    ], log_dir=tmp_path / "logs", options=FAST, on_event=events)
    try:
        bridge.options["start_attempts"] = 3
        await bridge.start(only={"flaky", "broken", "empty", "noscript"})
        assert bridge.status()["flaky"]["state"] == "ready"
        assert (tmp_path / "flaky" / "starts").read_text() == "3"
        log = (tmp_path / "logs" / "flaky.log").read_text()
        assert "simulated startup failure #1" in log and "simulated startup failure #2" in log
        assert "crashing fixture server start #3" in log

        msg = bridge.failures["broken"]
        assert "simulated startup failure #3" in msg and "stderr tail" in msg
        assert "#1" not in msg.split("stderr tail")[1], "tail covers the latest attempt only"
        assert "empty command" in bridge.failures["empty"]
        assert "server script not found" in bridge.failures["noscript"]
        assert (tmp_path / "broken" / "starts").read_text() == "3"
        assert [d["server"] for k, d in events if k == "mcp_start_failed"].count("flaky") == 2

        # Lazy retry: one restart attempt on the first call to a failed server, none afterwards.
        bridge.options["start_attempts"] = 1
        with pytest.raises(ToolFailure, match="restart failed"):
            await bridge.call("broken", "ping", {})
        assert (tmp_path / "broken" / "starts").read_text() == "4"
        with pytest.raises(ToolFailure, match="unavailable"):
            await bridge.call("broken", "ping", {})
        assert (tmp_path / "broken" / "starts").read_text() == "4"

        # A server whose first start failed recovers on its lazy retry.
        await bridge.start(only={"late"})          # start #1 fails (attempts=1)
        assert bridge.status()["late"]["state"] == "failed"
        bridge.options["start_attempts"] = 2       # lazy retry: #2 fails, #3 succeeds
        assert await bridge.call("late", "ping", {}) == "pong from start #3"
        assert "mcp__late__ping" in {t.name for t in bridge.tools}
    finally:
        await bridge.aclose()


@needs_fastmcp
async def test_hung_startup_times_out_with_stderr_tail(tmp_path):
    hang = ("import sys, time; print('loading a huge index', file=sys.stderr, flush=True); "
            "open(sys.argv[1], 'w').write('started'); time.sleep(120)")
    marker = tmp_path / "started"
    bridge = MCPBridge([MCPServerConfig("hung", command=sys.executable, args=["-c", hang, str(marker)],
                                        start_timeout_s=1.5)],
                       log_dir=tmp_path / "logs", options={**FAST, "start_attempts": 1})
    t0 = time.monotonic()
    try:
        await bridge.start()
        msg = bridge.failures["hung"]
        assert "startup timed out after 1.5s" in msg and "loading a huge index" in msg
        assert marker.exists()
        assert time.monotonic() - t0 < 15, "a hung start must be cancelled, not awaited for the grace period"
    finally:
        await bridge.aclose()


@needs_fastmcp
async def test_max_concurrency(tmp_path):
    bridge = MCPBridge([server("echo", ECHO, max_concurrency=1)], log_dir=tmp_path / "logs", options=FAST)
    try:
        await bridge.start()
        t0 = time.monotonic()
        a, b = await asyncio.gather(bridge.call("echo", "slow", {"seconds": 0.6}),
                                    bridge.call("echo", "slow", {"seconds": 0.6}))
        assert a == b == "slept 0.6s"
        assert time.monotonic() - t0 >= 1.15, "calls should be serialised by max_concurrency=1"
        assert bridge.status()["echo"]["calls"] == 2
    finally:
        await bridge.aclose()

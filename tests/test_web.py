"""The `vbt web` server: auth, streamed turns over SSE, busy/stop handling, downloads and file confinement.

Driven with httpx.AsyncClient over ASGITransport and the scripted mock provider
(monkeypatched into the provider factory), so nothing touches the network.
"""

import asyncio
import io
import json
import os
import zipfile

import pytest

pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")

from vbt import providers  # noqa: E402
from vbt.providers.mock import ScriptedProvider, call, reply  # noqa: E402
from vbt.web import add_web_parser  # noqa: E402
from vbt.web.server import WebAuthError, check_bind, create_app, resolve_model, serve  # noqa: E402
from vbt.web.sessions import EventLog, SessionLimitError, WebSessionManager  # noqa: E402

PASSWORD = "correct horse battery staple"
TABLE = "work/genomics-analyst/results/tables/l2g.csv"
JSON_H = {"Content-Type": "application/json"}


class GatedProvider(ScriptedProvider):
    """Scripted provider whose calls for ``blocked`` agents wait until ``gate`` is set (or are cancelled)."""

    blocked: set = set()
    gate: asyncio.Event | None = None

    async def complete(self, **kw):
        agent = (kw["settings"].extra or {}).get("agent_name")
        if agent in type(self).blocked:
            if type(self).gate is None:
                type(self).gate = asyncio.Event()
            await type(self).gate.wait()
        return await super().complete(**kw)


def research_rules():
    return {
        "cso": [
            reply("Delegating the genetics question.",
                  call("Task", subagent_type="genomics-analyst", description="genetics", prompt="EGFR genetics")),
            reply(call("mcp__provenance__record_claims", claims=[
                {"id": "C1", "text": "EGFR has an L2G score of 0.82 in lung adenocarcinoma.",
                 "evidence": [{"kind": "table", "path": TABLE}]}])),
            reply("EGFR has strong genetic support [[claim:C1]]. See the <script>alert(1)</script> table."),
            reply("Second answer: no further analysis needed."),
        ],
        "genomics-analyst": [
            reply(call("Write", file_path="results/tables/l2g.csv", content="gene,l2g\nEGFR,0.82\n")),
            reply("L2G score 0.82 for EGFR."),
        ],
    }


def install(monkeypatch, rules, cls=ScriptedProvider):
    made = []

    def factory(**options):
        p = cls.from_rules(rules())
        made.append(p)
        return p

    monkeypatch.setitem(providers._FACTORIES, "mock", factory)
    return made


def sse_events(text):
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


@pytest.fixture
def web_config(config):
    config["orchestration"]["enforce_review"] = False
    return config


@pytest.fixture
async def app_client(web_config):
    app = create_app(web_config, password=PASSWORD, start_mcp=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield app, client
    await app.state.sessions.aclose()


async def login(client):
    r = await client.post("/api/login", json={"password": PASSWORD})
    assert r.status_code == 200, r.text


async def new_session(client):
    r = await client.post("/api/sessions", json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def events_until(client, sid, kind="turn_end", after=0, timeout=20):
    r = await asyncio.wait_for(client.get(f"/api/sessions/{sid}/events",
                                          params={"after": after, "until": kind, "timeout": timeout}), timeout + 5)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    return sse_events(r.text)


# ----------------------------------------------------------------- auth


async def test_login_is_required(app_client, monkeypatch):
    install(monkeypatch, research_rules)
    app, client = app_client
    assert (await client.get("/api/auth")).json() == {"auth_required": True, "authenticated": False}
    for method, url in (("GET", "/api/config"), ("GET", "/api/runs"), ("POST", "/api/sessions")):
        r = await client.request(method, url, json={} if method == "POST" else None)
        assert r.status_code == 401, url
    assert (await client.get("/runs/whatever/download.zip")).status_code == 401
    r = await client.post("/api/login", json={"password": "wrong"})
    assert r.status_code == 401
    r = await client.post("/api/login", content=f"password={PASSWORD}",
                          headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 415  # JSON only: a cross-site form cannot log in or act
    await login(client)
    cookie = client.cookies.get("vbt_session")
    assert cookie and cookie.count(".") == 2
    cfg = (await client.get("/api/config")).json()
    assert cfg["auth_required"] and cfg["provider"] == "mock"
    assert any(a["name"] == "genomics-analyst" for a in cfg["roster"]["agents"])
    assert cfg["roster"]["divisions"]["Target Identification and Prioritization"]
    assert any("B7-H3" in e["prompt"] for e in cfg["examples"])
    # a forged cookie is rejected
    owner, expiry, _sig = cookie.split(".")
    client.cookies.clear()
    client.cookies.set("vbt_session", f"{owner}.{expiry}.{'0' * 64}")
    assert (await client.get("/api/config")).status_code == 401


async def test_sessions_are_bound_to_their_login(app_client, monkeypatch):
    install(monkeypatch, research_rules)
    app, client = app_client
    await login(client)
    sid = await new_session(client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as other:
        await login(other)
        assert (await other.get(f"/api/sessions/{sid}")).status_code == 404
        assert (await other.post(f"/api/sessions/{sid}/ask", json={"prompt": "x"})).status_code == 404


def test_password_and_bind_rules(web_config, monkeypatch):
    with pytest.raises(WebAuthError):
        check_bind("0.0.0.0", password=None, no_auth=False)
    with pytest.raises(WebAuthError):
        check_bind("0.0.0.0", password=None, no_auth=True)       # --no-auth only on localhost
    with pytest.raises(WebAuthError):
        check_bind("192.168.1.10", password=None, no_auth=True)
    with pytest.raises(WebAuthError):
        check_bind("127.0.0.1", password=None, no_auth=False)    # a password is required by default
    check_bind("127.0.0.1", password=None, no_auth=True)
    check_bind("::1", password=None, no_auth=True)
    check_bind("0.0.0.0", password="pw", no_auth=False)
    monkeypatch.delenv("VBT_WEB_PASSWORD", raising=False)
    with pytest.raises(WebAuthError):
        create_app(web_config)

    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("server must not start"))
    assert serve(web_config, host="0.0.0.0", port=0) == 2
    assert serve(web_config, host="0.0.0.0", port=0, no_auth=True) == 2

    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    add_web_parser(sub)
    args = p.parse_args(["web", "--host", "0.0.0.0", "--no-auth"])
    assert args.handler(args, web_config) == 2


def test_model_resolution(web_config):
    web_config["model_aliases"] = {"fast": "claude-haiku-4-5"}
    assert resolve_model(web_config, "fast") == "claude-haiku-4-5"
    assert resolve_model(web_config, web_config["models"]["orchestrator"]["model"])
    with pytest.raises(ValueError):
        resolve_model(web_config, "rm -rf /")


# ----------------------------------------------------------------- turns over SSE


async def test_ask_streams_text_and_turn_end(app_client, monkeypatch):
    install(monkeypatch, research_rules)
    app, client = app_client
    await login(client)
    sid = await new_session(client)
    r = await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "Is EGFR a good target?"})
    assert r.status_code == 202, r.text
    evs = await events_until(client, sid)
    kinds = [e["kind"] for e in evs]
    assert kinds[0] == "user" and kinds[-1] == "turn_end"
    assert "session_started" in kinds and "tool_start" in kinds and "delegation" in kinds
    streamed = "".join(e["data"]["text"] for e in evs if e["kind"] == "text" and e["data"]["agent"] == "cso")
    assert "EGFR has strong genetic support" in streamed
    assert "[[claim:" not in streamed  # anchors are stripped while streaming
    end = evs[-1]["data"]
    assert end["status"] == "completed"
    assert "[1]" in end["rendered"] and "[[claim:" not in end["rendered"]
    assert end["footnotes"][0]["id"] == "C1" and end["footnotes"][0]["verified"]
    assert end["claims_filed"] == ["C1"]
    # agents grouped by division travel with the events
    starts = [e["data"] for e in evs if e["kind"] == "agent_start" and e["data"].get("agent") == "genomics-analyst"]
    assert starts and starts[0]["division"] == "Target Identification and Prioritization"

    claims = (await client.get(f"/api/sessions/{sid}/claims")).json()
    assert claims["claims"][0]["id"] == "C1"
    assert claims["claims"][0]["evidence"][0]["status"] == "verified"
    assert claims["claims"][0]["evidence"][0]["url"].endswith("/files/" + TABLE)
    files = (await client.get(f"/api/sessions/{sid}/files")).json()
    assert files["counts"]["tables"] == 1
    row = files["files"]["tables"]["genomics-analyst"][0]
    assert (await client.get(row["url"])).text == "gene,l2g\nEGFR,0.82\n"

    st = (await client.get(f"/api/sessions/{sid}")).json()
    assert st["turns"] == 1 and not st["busy"] and st["turn_records"][0]["footnotes"][0]["id"] == "C1"
    # a reconnecting client resumes after the last event it saw
    again = await events_until(client, sid, after=evs[-1]["id"], timeout=0.2)
    assert again == []


async def test_overlapping_ask_gets_409(app_client, monkeypatch):
    GatedProvider.blocked = {"cso"}
    GatedProvider.gate = asyncio.Event()
    install(monkeypatch, lambda: {"cso": [reply("first done"), reply("second done")]}, GatedProvider)
    app, client = app_client
    await login(client)
    sid = await new_session(client)
    assert (await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "one"})).status_code == 202
    r = await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "two"})
    assert r.status_code == 409 and r.json()["error"] == "previous query still processing"
    GatedProvider.gate.set()
    evs = await events_until(client, sid)
    assert evs[-1]["data"]["reply"].startswith("first done")
    assert (await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "two"})).status_code == 202
    evs2 = await events_until(client, sid, after=evs[-1]["id"])
    assert evs2[-1]["kind"] == "turn_end" and "second done" in evs2[-1]["data"]["reply"]
    GatedProvider.blocked, GatedProvider.gate = set(), None


async def test_stop_records_interrupted_turn_and_next_ask_works(app_client, monkeypatch):
    GatedProvider.blocked = {"genomics-analyst"}
    GatedProvider.gate = asyncio.Event()
    install(monkeypatch, lambda: {
        "cso": [reply("Working on it.", call("Task", subagent_type="genomics-analyst", prompt="slow analysis")),
                reply("Fresh answer after the stop.")],
        "genomics-analyst": [reply("never reached")],
    }, GatedProvider)
    app, client = app_client
    await login(client)
    sid = await new_session(client)
    assert (await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "slow question"})).status_code == 202
    started = await events_until(client, sid, kind="delegation")
    assert started[-1]["kind"] == "delegation"
    await asyncio.sleep(0.05)
    r = await client.post(f"/api/sessions/{sid}/stop", json={})
    assert r.status_code == 200 and r.json()["ok"]
    evs = await events_until(client, sid, after=started[-1]["id"])
    end = evs[-1]["data"]
    assert evs[-1]["kind"] == "turn_end" and end["status"] == "interrupted"

    ws = app.state.sessions.sessions[sid]
    run = ws.cso.run
    assert run.turns[-1]["status"] == "interrupted"
    report = json.loads((run.dir / "session_report.json").read_text())
    assert report["turns"][0]["status"] == "interrupted"
    assert json.loads((run.dir / "MANIFEST.json").read_text())["interrupted_turns"] == [1]

    GatedProvider.blocked = set()
    assert (await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "try again"})).status_code == 202
    evs2 = await events_until(client, sid, after=evs[-1]["id"])
    assert evs2[-1]["data"]["status"] == "completed"
    assert "Fresh answer after the stop." in evs2[-1]["data"]["reply"]
    assert (await client.post(f"/api/sessions/{sid}/stop", json={})).json()["ok"] is False
    GatedProvider.gate = None


# ----------------------------------------------------------------- downloads and confinement


async def _finished_run(app, client, monkeypatch):
    install(monkeypatch, research_rules)
    await login(client)
    sid = await new_session(client)
    await client.post(f"/api/sessions/{sid}/ask", json={"prompt": "Is EGFR a good target?"})
    evs = await events_until(client, sid)
    return sid, evs[-1]["data"]["run_id"], app.state.sessions.sessions[sid].cso.run.dir


async def test_downloads(app_client, monkeypatch):
    app, client = app_client
    sid, run_id, run_dir = await _finished_run(app, client, monkeypatch)
    r = await client.get(f"/runs/{run_id}/download.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert f"{run_id}/MANIFEST.json" in names and f"{run_id}/README.md" in names and f"{run_id}/{TABLE}" in names
    assert all(n.startswith(f"{run_id}/") for n in names)
    chat = await client.get(f"/runs/{run_id}/chat.md")
    assert chat.status_code == 200 and "## Turn 1 — User" in chat.text and "[1]" in chat.text
    page = await client.get(f"/runs/{run_id}/audit.html")
    assert page.status_code == 200 and "sandbox" in page.headers["content-security-policy"]
    assert "<script>alert(1)</script>" not in page.text and "&lt;script&gt;alert(1)" in page.text
    runs = (await client.get("/api/runs")).json()["runs"]
    assert runs[0]["run_id"] == run_id and runs[0]["audit_url"] == f"/runs/{run_id}/audit.html"
    assert "path" not in runs[0]


async def test_path_traversal_and_dotfiles_are_refused(app_client, monkeypatch, tmp_path):
    app, client = app_client
    sid, run_id, run_dir = await _finished_run(app, client, monkeypatch)
    (run_dir / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n")
    (run_dir / "work" / ".hidden").mkdir()
    (run_dir / "work" / ".hidden" / "x.txt").write_text("sk-secret")
    outside = tmp_path / "outside.txt"
    outside.write_text("sk-secret")
    os.symlink(outside, run_dir / "work" / "genomics-analyst" / "link.txt")
    runs_dir = run_dir.parent
    (runs_dir / "secret.txt").write_text("sk-secret")
    for url in (f"/runs/{run_id}/files/.env", f"/runs/{run_id}/files/work/.hidden/x.txt",
                f"/runs/{run_id}/files/%2e%2e/secret.txt", f"/runs/{run_id}/files/work/%2e%2e/%2e%2e/secret.txt",
                f"/runs/{run_id}/files/work/genomics-analyst/link.txt", "/runs/..%2Fsecret.txt/files/x",
                "/runs/.downloads/download.zip", "/runs/nope/files/README.md"):
        r = await client.get(url)
        assert r.status_code in (403, 404), (url, r.status_code)
        assert "sk-secret" not in r.text
    ok = await client.get(f"/runs/{run_id}/files/{TABLE}")
    assert ok.status_code == 200 and "sandbox" in ok.headers["content-security-policy"]
    html = run_dir / "work" / "genomics-analyst" / "results" / "reports" / "x.html"
    html.write_text("<script>alert(1)</script>")
    r = await client.get(f"/runs/{run_id}/files/work/genomics-analyst/results/reports/x.html")
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"].startswith("attachment")
    svg = run_dir / "work" / "genomics-analyst" / "results" / "figures" / "f.svg"
    svg.parent.mkdir(parents=True, exist_ok=True)
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')
    r = await client.get(f"/runs/{run_id}/files/work/genomics-analyst/results/figures/f.svg")
    assert r.headers["content-type"].startswith("image/svg+xml")  # for <img>, which never runs it
    assert r.headers["content-disposition"].startswith("attachment")
    assert "sandbox" in r.headers["content-security-policy"]


# ----------------------------------------------------------------- session manager


async def test_session_limit_and_idle_cleanup():
    closed = []

    class FakeCSO:
        run = None

        async def close(self):
            closed.append(True)

    async def opener(cfg, on_event):
        return FakeCSO()

    mgr = WebSessionManager(opener, max_sessions=1, idle_timeout_s=10)
    s = mgr.create("owner", {})
    await s.ensure_started()
    with pytest.raises(SessionLimitError):
        mgr.create("owner", {})
    with pytest.raises(KeyError):
        mgr.get(s.id, "someone-else")
    assert await mgr.sweep(now=s.last_active + 5) == []
    assert await mgr.sweep(now=s.last_active + 11) == [s.id]
    assert closed == [True] and len(mgr) == 0
    mgr.create("owner", {})  # capacity is free again
    await mgr.aclose()


async def test_event_log_wait_and_threads():
    log = EventLog(maxlen=3)
    assert await log.wait(0, 0.01) == []
    waiter = asyncio.ensure_future(log.wait(0, 5))
    await asyncio.sleep(0)
    await asyncio.to_thread(log.publish, "text", {"text": "from a worker thread"})
    got = await asyncio.wait_for(waiter, 2)
    assert got[0]["data"]["text"] == "from a worker thread"
    for i in range(5):
        log.publish("x", {"i": i})
    assert [e["data"]["i"] for e in log.since(0)] == [2, 3, 4]  # bounded


# ----------------------------------------------------------------- Host / Origin / content-type (S3, S4)


async def test_json_check_parses_the_media_type(app_client, monkeypatch):
    install(monkeypatch, research_rules)
    app, client = app_client
    # CORS-safelisted (no preflight) type that merely contains "application/json"
    r = await client.post("/api/login", content=json.dumps({"password": PASSWORD}),
                          headers={"Content-Type": "text/plain; x=application/json"})
    assert r.status_code == 415
    r = await client.post("/api/login", content=json.dumps({"password": PASSWORD}),
                          headers={"Content-Type": "Application/JSON; charset=utf-8"})
    assert r.status_code == 200


async def _no_auth_client(web_config, base_url, **kw):
    app = create_app(web_config, no_auth=True, start_mcp=False, **kw)
    return app, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


async def test_no_auth_rejects_foreign_host_header(web_config, monkeypatch):
    """DNS rebinding: attacker.example resolving to 127.0.0.1 must not reach the API."""
    install(monkeypatch, research_rules)
    app, client = await _no_auth_client(web_config, "http://127.0.0.1:7860", bind_host="127.0.0.1")
    try:
        r = await client.post("/api/sessions", json={}, headers={"Host": "attacker.example:7860"})
        assert r.status_code == 400
        assert (await client.get("/api/runs", headers={"Host": "attacker.example"})).status_code == 400
        assert (await client.get("/runs/x/download.zip", headers={"Host": "attacker.example"})).status_code == 400
        for host in ("127.0.0.1:7860", "localhost:7860", "[::1]:7860", "LOCALHOST"):
            assert (await client.get("/api/auth", headers={"Host": host})).status_code == 200, host
        r = await client.post("/api/sessions", json={}, headers={"Origin": "http://127.0.0.1:7860"})
        assert r.status_code == 201, r.text
    finally:
        await client.aclose()
        await app.state.sessions.aclose()


async def test_cross_origin_state_change_is_refused(web_config, monkeypatch):
    install(monkeypatch, research_rules)
    app, client = await _no_auth_client(web_config, "http://localhost:7860")
    try:
        for origin in ("http://attacker.example", "null", "http://localhost:9999"):
            r = await client.post("/api/sessions", json={}, headers={"Origin": origin})
            assert r.status_code == 403, origin
        r = await client.request("DELETE", "/api/sessions/abc", headers={"Origin": "http://evil.test"})
        assert r.status_code == 403
        # GETs are not blocked by Origin (the Host check covers them)
        assert (await client.get("/api/auth", headers={"Origin": "http://evil.test"})).status_code == 200
    finally:
        await client.aclose()
        await app.state.sessions.aclose()


def test_allowed_host_set_rules():
    from vbt.web.server import allowed_host_set
    assert allowed_host_set(no_auth=False) is None            # password mode: any host by default
    s = allowed_host_set(no_auth=False, extra=["vbt.example.org:443"])
    assert "vbt.example.org" in s and "localhost" in s
    s = allowed_host_set(no_auth=True, bind_host="::1", extra=["[::2]", "::3"])
    assert {"localhost", "127.0.0.1", "::1", "::2", "::3"} <= s and "attacker.example" not in s
    assert "0.0.0.0" not in allowed_host_set(no_auth=True, bind_host="0.0.0.0")


async def test_password_mode_honours_configured_allowed_hosts(web_config, monkeypatch):
    install(monkeypatch, research_rules)
    web_config.setdefault("web_ui", {})["allowed_hosts"] = ["vbt.example.org"]
    app = create_app(web_config, password=PASSWORD, start_mcp=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://vbt.example.org") as c:
        assert (await c.get("/api/auth")).status_code == 200
        assert (await c.get("/api/auth", headers={"Host": "rebind.example"})).status_code == 400
    await app.state.sessions.aclose()

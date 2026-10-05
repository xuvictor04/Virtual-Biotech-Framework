"""Pluggable WebSearch backends: SearxNG, Brave, provider-native and ``auto`` resolution.

Offline: fake SearxNG / Brave APIs served by ``http.server`` on 127.0.0.1.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

httpx = pytest.importorskip("httpx")

from vbt.tools import search_backends as sb  # noqa: E402
from vbt.tools import web  # noqa: E402
from vbt.tools.base import ToolContext, ToolFailure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEY = "brv-test-key-0123456789"


# ---------------------------------------------------------------- fake HTTP APIs


class FakeAPI:
    """A threaded HTTP server; ``respond(req) -> (status, headers, body)`` answers every GET."""

    def __init__(self, respond: Callable[[dict[str, Any]], tuple[int, dict[str, str], Any]]) -> None:
        self.respond = respond
        self.requests: list[dict[str, Any]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                parts = urlsplit(self.path)
                req = {"path": parts.path, "params": {k: v[0] for k, v in parse_qs(parts.query).items()},
                       "headers": {k.lower(): v for k, v in self.headers.items()}}
                api.requests.append(req)
                status, headers, body = api.respond(req)
                if not isinstance(body, (bytes, str)):
                    body = json.dumps(body)
                    headers = {"Content-Type": "application/json", **headers}
                data = body.encode() if isinstance(body, str) else body
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):  # silence
                return

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_api():
    servers: list[FakeAPI] = []

    def make(respond):
        api = FakeAPI(respond)
        servers.append(api)
        return api

    yield make
    for s in servers:
        s.close()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(sb.SEARXNG_ENV, raising=False)
    monkeypatch.delenv(sb.BRAVE_KEY_ENV, raising=False)
    monkeypatch.setattr(sb, "TRANSPORT", None)


def _cfg(enabled: bool = True, **search: Any) -> dict[str, Any]:
    return {"web": {"enabled": enabled, "search": {"retry_backoff_s": 0, **search}}}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class NoSearchProvider:
    name = "vllm"

    def supports_web_search(self) -> bool:
        return False


class NativeProvider:
    name = "anthropic"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def supports_web_search(self) -> bool:
        return True

    async def web_search(self, query, *, max_results=8, allowed_domains=None, blocked_domains=None):
        self.calls.append(query)
        return {"results": [{"title": "native", "url": "https://example.org"}], "summary": "native",
                "cost_usd": 0.01}


SX_ROWS = [
    {"title": "PCSK9 inhibitors: a <b>review</b>", "url": "https://www.nature.com/articles/nrd.2017.1",
     "content": "Evolocumab &amp; alirocumab   lower <em>LDL-C</em> by ~60%.", "engine": "duckduckgo",
     "engines": ["duckduckgo", "brave"], "publishedDate": "2017-03-05T00:00:00", "score": 2.0},
    {"title": "PCSK9 - Wikipedia", "url": "https://en.wikipedia.org/wiki/PCSK9",
     "content": "Proprotein convertase subtilisin/kexin type 9.", "engine": "wikipedia", "publishedDate": None},
    {"title": "an answer without a link", "url": None, "content": "42"},
    {"title": "FOURIER trial", "url": "https://pubmed.ncbi.nlm.nih.gov/28304224/",
     "content": "Evolocumab and clinical outcomes.", "engines": ["pubmed"], "publishedDate": "2017-05-04"},
    {"title": "Not Nature", "url": "https://notnature.com/pcsk9", "content": "lookalike domain",
     "engine": "brave"},
    {"title": "Nature news", "url": "https://news.nature.com/pcsk9", "content": "subdomain", "engine": "brave"},
]


def _searx(rows=None, unresponsive=None):
    def respond(req):
        return 200, {}, {"query": req["params"].get("q"), "results": list(SX_ROWS if rows is None else rows),
                         "answers": [], "infoboxes": [], "suggestions": [],
                         "unresponsive_engines": unresponsive or []}
    return respond


# ---------------------------------------------------------------- resolution


def test_no_web_resolves_to_none_even_when_configured(monkeypatch):
    monkeypatch.setenv(sb.SEARXNG_ENV, "http://localhost:8888")
    monkeypatch.setenv(sb.BRAVE_KEY_ENV, KEY)
    for backend in ("auto", "searxng", "brave", "provider"):
        cfg = _cfg(False, backend=backend, searxng_url="http://localhost:8888")
        assert sb.resolve_search_backend(cfg, NativeProvider()) is None
        info = sb.describe_search_backend(cfg, NativeProvider())
        assert info["backend"] is None and "no-web" in info["reason"]


def test_auto_resolution_order(monkeypatch):
    native = NativeProvider()
    plain = NoSearchProvider()
    monkeypatch.setenv(sb.SEARXNG_ENV, "http://searx.env:8888/")
    monkeypatch.setenv(sb.BRAVE_KEY_ENV, KEY)
    # 1. provider-native search wins when the provider has it
    assert sb.resolve_search_backend(_cfg(), native) == native.web_search
    # 2. else SearxNG: config URL first, then $SEARXNG_URL
    b = sb.resolve_search_backend(_cfg(searxng_url="http://cfg-host:9999/search"), plain)
    assert isinstance(b, sb.SearxNGBackend) and b.url == "http://cfg-host:9999"
    b = sb.resolve_search_backend(_cfg(), plain)
    assert isinstance(b, sb.SearxNGBackend) and b.url == "http://searx.env:8888"
    assert sb.describe_search_backend(_cfg(), plain)["reason"] == sb.SEARXNG_ENV
    # 3. else Brave when a key is set
    monkeypatch.delenv(sb.SEARXNG_ENV)
    b = sb.resolve_search_backend(_cfg(searxng_url=""), plain)  # blank ${SEARXNG_URL:-} = unset
    assert isinstance(b, sb.BraveSearchBackend) and KEY not in repr(b)
    # 4. else nothing
    monkeypatch.delenv(sb.BRAVE_KEY_ENV)
    assert sb.resolve_search_backend(_cfg(), plain) is None
    assert sb.resolve_search_backend(_cfg(), None) is None
    assert "no search backend configured" in sb.describe_search_backend(_cfg(), plain)["reason"]
    # no web.search block at all behaves as auto
    assert sb.resolve_search_backend({"web": {"enabled": True}}, native) == native.web_search
    assert sb.resolve_search_backend({}, plain) is None


def test_auto_keeps_mock_profile_on_mock_search(config):
    from vbt.providers.mock import ScriptedProvider

    provider = ScriptedProvider.from_rules({})
    assert sb.resolve_search_backend(config, provider) == provider.web_search


def test_explicit_backends(monkeypatch):
    native = NativeProvider()
    monkeypatch.setenv(sb.SEARXNG_ENV, "http://localhost:8888")
    # explicit searxng beats provider-native search
    assert isinstance(sb.resolve_search_backend(_cfg(backend="searxng"), native), sb.SearxNGBackend)
    assert sb.resolve_search_backend(_cfg(backend="provider"), native) == native.web_search
    assert sb.resolve_search_backend(_cfg(backend="none"), native) is None
    assert sb.resolve_search_backend(_cfg(backend=False), native) is None
    assert sb.resolve_search_backend({"web": {"search": "none"}}, native) is None
    assert isinstance(sb.resolve_search_backend({"web": {"search": "searx"}}, native), sb.SearxNGBackend)


@pytest.mark.parametrize("backend, needle", [
    ("provider", "has no native web search"),
    ("searxng", "SEARXNG_URL"),
    ("brave", "BRAVE_SEARCH_API_KEY"),
    ("bing", "unknown web.search.backend"),
])
async def test_explicit_backend_that_cannot_work_explains_the_fix(backend, needle):
    b = sb.resolve_search_backend(_cfg(backend=backend), NoSearchProvider())
    assert isinstance(b, sb.MisconfiguredSearchBackend)
    with pytest.raises(sb.SearchBackendError, match=needle) as err:
        await b("PCSK9", max_results=3, allowed_domains=None, blocked_domains=None)
    assert isinstance(err.value, RuntimeError) and isinstance(err.value, ToolFailure)
    info = sb.describe_search_backend(_cfg(backend=backend), NoSearchProvider())
    assert info["backend"] is None and needle in info["problem"]


def test_search_settings_and_url_normalisation():
    s = sb.search_settings({"web": {"search": {"timeout_s": None, "retries": 5}}})
    assert s["timeout_s"] == sb.SEARCH_DEFAULTS["timeout_s"] and s["retries"] == 5
    assert sb.search_settings({"web": {"search": "brave"}})["backend"] == "brave"
    assert sb.searxng_base_url("localhost:8888") == "http://localhost:8888"
    assert sb.searxng_base_url("https://h.example/searxng/search/") == "https://h.example/searxng"
    assert sb.normalize_domain("https://www.Nature.com/articles/") == "nature.com/articles"
    assert sb.normalize_domain("*.nih.gov") == "nih.gov"
    assert sb.normalize_domain("example.org:443") == "example.org"
    assert sb.normalize_domain("  ") is None


def test_clean_text_keeps_comparisons():
    assert sb.clean_text("HR <b>0.85</b> (p < 0.05; n > 1000) &amp; <span class='x'>more</span>") == (
        "HR 0.85 (p < 0.05; n > 1000) & more")
    assert sb.clean_text("a" * 50, 10) == "a" * 9 + "…"


def test_domain_filter_matching():
    f = sb.DomainFilter(["nature.com", "fda.gov/drugs"])
    assert f.allows("https://www.nature.com/x") and f.allows("https://news.nature.com/y")
    assert not f.allows("https://notnature.com/x")
    assert f.allows("https://www.fda.gov/drugs/abc") and f.allows("https://fda.gov/drugs")
    assert not f.allows("https://www.fda.gov/drugsafety") and not f.allows("https://fda.gov/news")
    assert f.query_terms(3) == "(site:nature.com OR site:fda.gov/drugs)"
    assert f.query_terms(1) == ""
    blk = sb.DomainFilter(None, "wikipedia.org, reddit.com")
    assert not blk.allows("https://en.wikipedia.org/wiki/X") and blk.allows("https://nih.gov/")
    assert blk.query_terms(3) == "-site:wikipedia.org -site:reddit.com"
    assert sb.DomainFilter(["a.org"], ["b.org"]).blocked == []  # an allow-list wins
    assert not sb.DomainFilter().active


# ---------------------------------------------------------------- SearxNG


async def test_searxng_query_mapping_and_summary(fake_api):
    api = fake_api(_searx(unresponsive=[["google scholar", "timeout"]]))
    backend = sb.resolve_search_backend(_cfg(backend="searxng", searxng_url=api.url + "/"), NoSearchProvider())
    res = await backend("PCSK9   inhibitors", max_results=3)
    req = api.requests[0]
    assert req["path"] == "/search"
    assert req["params"] == {"q": "PCSK9 inhibitors", "format": "json", "language": "en", "safesearch": "0",
                             "pageno": "1", "categories": "general,science"}
    assert req["headers"]["accept"] == "application/json" and "vbt-harness" in req["headers"]["user-agent"]
    assert len(api.requests) == 1
    assert res["cost_usd"] == 0.0 and res["backend"] == "searxng"
    assert [r["url"] for r in res["results"]] == ["https://www.nature.com/articles/nrd.2017.1",
                                                  "https://en.wikipedia.org/wiki/PCSK9",
                                                  "https://pubmed.ncbi.nlm.nih.gov/28304224/"]
    first = res["results"][0]
    assert first == {"title": "PCSK9 inhibitors: a review", "url": "https://www.nature.com/articles/nrd.2017.1",
                     "snippet": "Evolocumab & alirocumab lower LDL-C by ~60%.", "engine": "duckduckgo",
                     "published": "2017-03-05"}
    assert res["results"][1]["published"] is None and res["results"][2]["engine"] == "pubmed"
    lines = res["summary"].splitlines()
    assert lines[0] == "1. PCSK9 inhibitors: a review — https://www.nature.com/articles/nrd.2017.1 (2017-03-05)"
    assert lines[1] == "   Evolocumab & alirocumab lower LDL-C by ~60%."
    assert lines[2].startswith("2. PCSK9 - Wikipedia — https://en.wikipedia.org/wiki/PCSK9")
    assert "engines that did not answer: google scholar: timeout" in res["summary"]
    assert res["notes"] == ["engines that did not answer: google scholar: timeout"]


async def test_searxng_options_passthrough(fake_api):
    api = fake_api(_searx())
    backend = sb.SearxNGBackend(api.url, {"categories": "science", "engines": ["pubmed", "arxiv"],
                                          "language": "all", "safesearch": "strict", "time_range": "year"})
    await backend("q", max_results=2)
    p = api.requests[0]["params"]
    assert p["categories"] == "science" and p["engines"] == "pubmed,arxiv" and p["language"] == "all"
    assert p["safesearch"] == "2" and p["time_range"] == "year"
    # values SearxNG does not support are not sent
    api.requests.clear()
    await sb.SearxNGBackend(api.url, {"time_range": "2020-01-01to2021-01-01", "categories": []})("q")
    assert "time_range" not in api.requests[0]["params"] and "categories" not in api.requests[0]["params"]


async def test_searxng_allowed_domains_site_terms_filter_and_follow_up(fake_api):
    api = fake_api(_searx())
    backend = sb.SearxNGBackend(api.url, {"retry_backoff_s": 0})
    res = await backend("PCSK9", max_results=5, allowed_domains=["https://www.nature.com/"])
    queries = [(r["params"]["q"], r["params"]["pageno"]) for r in api.requests]
    # site: query first; too few matches -> plain query, then its next page
    assert queries == [("PCSK9 site:nature.com", "1"), ("PCSK9", "1"), ("PCSK9", "2")]
    urls = [r["url"] for r in res["results"]]
    assert urls == ["https://www.nature.com/articles/nrd.2017.1", "https://news.nature.com/pcsk9"]
    assert any("outside the domain filter were dropped" in n for n in res["notes"])
    # enough matches on the first request -> no follow-up
    api.requests.clear()
    res = await backend("PCSK9", max_results=2, allowed_domains=["nature.com"])
    assert len(api.requests) == 1 and len(res["results"]) == 2


async def test_searxng_blocked_domains(fake_api):
    api = fake_api(_searx())
    backend = sb.SearxNGBackend(api.url, {"max_requests": 1})
    res = await backend("PCSK9", max_results=8, blocked_domains=["wikipedia.org", "nature.com"])
    assert api.requests[0]["params"]["q"] == "PCSK9 -site:wikipedia.org -site:nature.com"
    assert len(api.requests) == 1  # max_requests caps the follow-ups
    urls = [r["url"] for r in res["results"]]
    assert "https://en.wikipedia.org/wiki/PCSK9" not in urls and "https://pubmed.ncbi.nlm.nih.gov/28304224/" in urls
    assert all("nature.com/" not in u or "notnature" in u for u in urls)


async def test_long_domain_lists_skip_site_terms_but_still_filter(fake_api):
    api = fake_api(_searx())
    allowed = ["nih.gov", "fda.gov", "ema.europa.eu", "who.int"]
    res = await sb.SearxNGBackend(api.url)("PCSK9", max_results=1, allowed_domains=allowed)
    assert api.requests[0]["params"]["q"] == "PCSK9"
    assert [r["url"] for r in res["results"]] == ["https://pubmed.ncbi.nlm.nih.gov/28304224/"]
    api.requests.clear()
    await sb.SearxNGBackend(api.url, {"site_operators": False})("PCSK9", max_results=1, allowed_domains=["nih.gov"])
    assert api.requests[0]["params"]["q"] == "PCSK9"


async def test_searxng_no_results(fake_api):
    api = fake_api(_searx(rows=[]))
    res = await sb.SearxNGBackend(api.url)("zzzz", max_results=4)
    assert res["results"] == [] and res["summary"] == "No results."
    assert "notes" not in res


async def test_searxng_retries_5xx_then_succeeds(fake_api):
    seq = iter([(503, {}, "busy"), (502, {}, "bad gateway")])

    def respond(req):
        return next(seq, None) or _searx()(req)

    api = fake_api(respond)
    res = await sb.SearxNGBackend(api.url, {"retry_backoff_s": 0})("PCSK9", max_results=1)
    assert len(api.requests) == 3 and len(res["results"]) == 1


async def test_searxng_5xx_exhausts_retries(fake_api):
    api = fake_api(lambda req: (500, {}, "boom"))
    with pytest.raises(sb.SearchBackendError) as err:
        await sb.SearxNGBackend(api.url, {"retry_backoff_s": 0, "retries": 2})("PCSK9")
    assert len(api.requests) == 3
    msg = str(err.value)
    assert api.url in msg and "HTTP 500" in msg and "3 attempt(s)" in msg and "logs searxng" in msg


async def test_retry_after_is_honoured_up_to_the_cap(fake_api):
    seq = iter([(429, {"Retry-After": "120"}, "slow down")])

    def respond(req):
        return next(seq, None) or _searx()(req)

    api = fake_api(respond)
    t0 = time.monotonic()
    res = await sb.SearxNGBackend(api.url, {"retry_after_max_s": 0.05})("PCSK9", max_results=1)
    assert time.monotonic() - t0 < 5 and len(api.requests) == 2 and res["results"]


@pytest.mark.parametrize("status, needle", [
    (403, "search.formats"),
    (429, "server.limiter: false"),
    (404, "instance root"),
    (400, "HTTP 400"),
])
async def test_searxng_http_errors_name_the_fix(fake_api, status, needle):
    api = fake_api(lambda req: (status, {"Content-Type": "text/html"}, "<html>nope</html>"))
    with pytest.raises(sb.SearchBackendError, match=needle) as err:
        await sb.SearxNGBackend(api.url, {"retries": 0})("PCSK9")
    assert api.url in str(err.value)


async def test_searxng_non_json_answer(fake_api):
    api = fake_api(lambda req: (200, {"Content-Type": "text/html"}, "<html>search page</html>"))
    with pytest.raises(sb.SearchBackendError, match="non-JSON"):
        await sb.SearxNGBackend(api.url)("PCSK9")
    api2 = fake_api(lambda req: (200, {}, {"error": "x"}))
    with pytest.raises(sb.SearchBackendError, match="unexpected JSON"):
        await sb.SearxNGBackend(api2.url)("PCSK9")


async def test_searxng_unreachable_names_url_and_compose_hint():
    url = f"http://127.0.0.1:{_free_port()}"
    backend = sb.resolve_search_backend(_cfg(backend="searxng", searxng_url=url, retries=1), NoSearchProvider())
    with pytest.raises(sb.SearchBackendError) as err:
        await backend("PCSK9")
    msg = str(err.value)
    assert url in msg and "unreachable" in msg and "docker compose" in msg and "up -d searxng" in msg
    assert isinstance(err.value, RuntimeError)


async def test_searxng_read_timeout(fake_api):
    def slow(req):
        time.sleep(0.6)
        return _searx()(req)

    api = fake_api(slow)
    with pytest.raises(sb.SearchBackendError, match="did not answer within"):
        await sb.SearxNGBackend(api.url, {"timeout_s": 0.2})("PCSK9")
    assert len(api.requests) == 1  # a slow server is not hammered with retries


async def test_follow_up_failure_keeps_first_page(fake_api):
    def respond(req):
        if req["params"]["q"] == "PCSK9":  # the plain follow-up query fails
            return 500, {}, "boom"
        return _searx()(req)

    api = fake_api(respond)
    res = await sb.SearxNGBackend(api.url, {"retries": 0})("PCSK9", max_results=5, allowed_domains=["nature.com"])
    assert len(res["results"]) == 2 and any("follow-up request failed" in n for n in res["notes"])


async def test_query_validation():
    backend = sb.SearxNGBackend("http://127.0.0.1:9")
    with pytest.raises(sb.SearchBackendError, match="query is required"):
        await backend("   ")
    with pytest.raises(sb.SearchBackendError, match="not both"):
        await backend("x", allowed_domains=["a.org"], blocked_domains=["b.org"])


async def test_transport_override_and_max_results_cap():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        rows = [{"title": f"r{i}", "url": f"https://site{i}.org/", "content": "x"} for i in range(40)]
        return httpx.Response(200, json={"results": rows})

    cfg = _cfg(backend="searxng", searxng_url="http://searxng:8080")
    backend = sb.resolve_search_backend(cfg, None, transport=httpx.MockTransport(handler))
    res = await backend("q", max_results=100)
    assert len(res["results"]) == sb.SEARCH_DEFAULTS["max_results_cap"]
    assert str(seen[0].url).startswith("http://searxng:8080/search?")


# ---------------------------------------------------------------- Brave


BRAVE_ROWS = [
    {"title": "FDA approves <strong>evolocumab</strong>", "url": "https://www.fda.gov/news/evolocumab",
     "description": "The <strong>FDA</strong> approved Repatha.", "page_age": "2015-08-27T10:00:00",
     "age": "August 27, 2015"},
    {"title": "Repatha label", "url": "https://www.accessdata.fda.gov/label.pdf", "description": "Label.",
     "age": "March 2, 2024"},
    {"title": "Evolocumab - Wikipedia", "url": "https://en.wikipedia.org/wiki/Evolocumab", "description": "Drug."},
]


def _brave(rows=None):
    def respond(req):
        if req["headers"].get("x-subscription-token") != KEY:
            return 422, {}, {"type": "ErrorResponse", "error": {
                "status": 422, "code": "SUBSCRIPTION_TOKEN_INVALID",
                "detail": "The provided subscription token is invalid."}}
        return 200, {}, {"type": "search", "query": {"original": req["params"].get("q")},
                         "web": {"type": "search", "results": list(BRAVE_ROWS if rows is None else rows)}}
    return respond


def _brave_cfg(api, **extra):
    return _cfg(backend="brave", brave_url=api.url + "/res/v1/web/search", **extra)


async def test_brave_request_mapping_and_cost(fake_api, monkeypatch):
    monkeypatch.setenv(sb.BRAVE_KEY_ENV, KEY)
    api = fake_api(_brave())
    backend = sb.resolve_search_backend(_brave_cfg(api, brave_cost_per_query=0.005, time_range="month"),
                                        NoSearchProvider())
    assert isinstance(backend, sb.BraveSearchBackend)
    res = await backend("evolocumab approval", max_results=2)
    req = api.requests[0]
    assert req["path"] == "/res/v1/web/search" and req["headers"]["x-subscription-token"] == KEY
    assert req["params"] == {"q": "evolocumab approval", "count": "2", "offset": "0", "safesearch": "off",
                             "text_decorations": "false", "result_filter": "web", "search_lang": "en",
                             "freshness": "pm"}
    assert res["cost_usd"] == pytest.approx(0.005) and res["backend"] == "brave"
    assert res["results"][0] == {"title": "FDA approves evolocumab", "url": "https://www.fda.gov/news/evolocumab",
                                 "snippet": "The FDA approved Repatha.", "engine": "brave",
                                 "published": "2015-08-27"}
    assert res["results"][1]["published"] == "March 2, 2024"
    assert res["summary"].splitlines()[0] == (
        "1. FDA approves evolocumab — https://www.fda.gov/news/evolocumab (2015-08-27)")


async def test_brave_domain_filter_and_billing_per_request(fake_api, monkeypatch):
    monkeypatch.setenv(sb.BRAVE_KEY_ENV, KEY)
    api = fake_api(_brave())
    backend = sb.resolve_search_backend(_brave_cfg(api, brave_cost_per_query=0.01), None)
    res = await backend("evolocumab", max_results=5, allowed_domains=["fda.gov"])
    assert [r["params"]["q"] for r in api.requests] == ["evolocumab site:fda.gov", "evolocumab"]
    assert api.requests[0]["params"]["count"] == "20"  # over-fetch when filtering
    assert [r["url"] for r in res["results"]] == ["https://www.fda.gov/news/evolocumab",
                                                  "https://www.accessdata.fda.gov/label.pdf"]
    assert res["cost_usd"] == pytest.approx(0.02)  # brave_max_requests = 2, each billed


async def test_brave_rejected_key_never_echoes_it(fake_api):
    api = fake_api(_brave())
    backend = sb.BraveSearchBackend("wrong-key-secret", {"brave_url": api.url + "/res/v1/web/search"})
    with pytest.raises(sb.SearchBackendError) as err:
        await backend("evolocumab")
    msg = str(err.value)
    assert "BRAVE_SEARCH_API_KEY" in msg and "SUBSCRIPTION_TOKEN_INVALID" in msg and "wrong-key-secret" not in msg


async def test_brave_rate_limit(fake_api, monkeypatch):
    api = fake_api(lambda req: (429, {"Retry-After": "0"}, {"error": {"code": "RATE_LIMITED", "detail": "1 qps"}}))
    backend = sb.BraveSearchBackend(KEY, {"brave_url": api.url, "retries": 1, "retry_backoff_s": 0})
    with pytest.raises(sb.SearchBackendError, match="rate limit"):
        await backend("x")
    assert len(api.requests) == 2


async def test_brave_does_not_follow_redirects_with_the_key(fake_api):
    other = fake_api(_brave())
    api = fake_api(lambda req: (302, {"Location": other.url + "/res/v1/web/search"}, ""))
    backend = sb.BraveSearchBackend(KEY, {"brave_url": api.url})
    with pytest.raises(sb.SearchBackendError, match="HTTP 302"):
        await backend("x")
    assert other.requests == []


# ---------------------------------------------------------------- WebSearch tool + checks


class _Runtime:
    """The parts of Runtime the WebSearch tool touches."""

    def __init__(self, config, provider=None):
        self.config = config
        self.provider = provider
        self.charges: list[tuple] = []
        self.traces: list[tuple] = []

    @property
    def search_backend(self):
        return sb.resolve_search_backend(self.config, self.provider)

    def _charge(self, agent, usage, usd, label=""):
        self.charges.append((agent, usd, label))

    def _trace(self, type, **data):
        self.traces.append((type, data))


async def test_websearch_tool_through_searxng(fake_api):
    api = fake_api(_searx())
    rt = _Runtime(_cfg(searxng_url=api.url), NoSearchProvider())
    ctx = ToolContext("chief-of-staff", None, rt, tool_call_id="s1")
    res = await web.web_search(ctx, {"query": "PCSK9", "allowed_domains": ["nature.com"], "max_results": 2})
    assert api.requests[0]["params"]["q"] == "PCSK9 site:nature.com"
    assert res["summary"].startswith("<untrusted-web-content") and "1. PCSK9 inhibitors" in res["summary"]
    assert "cost_usd" not in res and len(res["results"]) == 2
    assert rt.charges == [("chief-of-staff", 0.0, "web_search")]
    assert rt.traces[-1][0] == "web_search" and rt.traces[-1][1]["allowed_domains"] == ["nature.com"]


async def test_websearch_tool_errors(fake_api):
    url = f"http://127.0.0.1:{_free_port()}"
    rt = _Runtime(_cfg(searxng_url=url, retries=0), NoSearchProvider())
    ctx = ToolContext("chief-of-staff", None, rt, tool_call_id="s2")
    with pytest.raises(ToolFailure, match="docker compose"):
        await web.web_search(ctx, {"query": "PCSK9"})
    rt.config = _cfg()
    with pytest.raises(ToolFailure, match="no web search backend is configured"):
        await web.web_search(ctx, {"query": "PCSK9"})
    rt.config = _cfg(False, searxng_url=url)
    with pytest.raises(ToolFailure, match="disabled"):
        await web.web_search(ctx, {"query": "PCSK9"})


async def test_check_search_backend(fake_api, monkeypatch):
    api = fake_api(_searx())
    ok = await sb.check_search_backend(_cfg(searxng_url=api.url), NoSearchProvider())
    assert ok["ok"] and ok["backend"] == "searxng" and ok["n_results"] == 3 and ok["error"] is None
    assert ok["sample"]["url"].startswith("https://www.nature.com/")
    empty = fake_api(_searx(rows=[], unresponsive=[["pubmed", "HTTP error"]]))
    res = await sb.check_search_backend(_cfg(searxng_url=empty.url), NoSearchProvider())
    assert not res["ok"] and "pubmed: HTTP error" in res["error"]
    down = await sb.check_search_backend(_cfg(searxng_url=f"http://127.0.0.1:{_free_port()}", retries=0), None)
    assert not down["ok"] and "unreachable" in down["error"]
    none = await sb.check_search_backend(_cfg(), NoSearchProvider())
    assert not none["ok"] and none["backend"] is None and "no search backend" in none["error"]
    native = await sb.check_search_backend(_cfg(), NativeProvider())
    assert native["ok"] and native["backend"] == "provider"


# ---------------------------------------------------------------- deploy file


def test_searxng_settings_file():
    path = ROOT / "deploy" / "local" / "searxng" / "settings.yml"
    s = yaml.safe_load(path.read_text())
    assert "json" in s["search"]["formats"] and "html" in s["search"]["formats"]
    assert s["search"]["safe_search"] == 0 and s["search"]["default_lang"] == "en"
    assert s["server"]["limiter"] is False and s["server"]["public_instance"] is False
    assert s["server"]["secret_key"] and s["server"]["secret_key"] != "ultrasecretkey"
    keep = s["use_default_settings"]["engines"]["keep_only"]
    for name in ("duckduckgo", "brave", "wikipedia", "pubmed", "arxiv", "google scholar", "semantic scholar"):
        assert name in keep
    # an override of an engine that is not kept would be appended as a broken engine definition
    assert all(e["name"] in keep for e in s["engines"])
    assert s["outgoing"]["max_request_timeout"] < sb.SEARCH_DEFAULTS["timeout_s"]


def test_docs_mention_every_config_key():
    text = (ROOT / "docs" / "WEB_SEARCH.md").read_text()
    for key in sb.SEARCH_DEFAULTS:
        assert key in text, key
    for name in sb.BACKENDS:
        assert f"`{name}`" in text

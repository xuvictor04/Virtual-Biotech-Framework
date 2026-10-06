"""Pluggable web search backends for the WebSearch tool.

Local open-weight models (vLLM / SGLang) have no provider-native web search,
so the harness queries a search engine itself. ``web.search.backend`` picks it:

``searxng``
    A self-hosted SearxNG metasearch instance through its JSON API
    (``GET {url}/search?q=...&format=json``). URL from ``web.search.searxng_url``
    or ``SEARXNG_URL``. ``deploy/local/searxng/settings.yml`` is a ready config
    (JSON format on, limiter off, general + science engines).
``brave``
    The Brave Search API (``BRAVE_SEARCH_API_KEY`` or ``web.search.brave_api_key``);
    ``web.search.brave_cost_per_query`` (default 0.0) is charged per request.
``provider``
    The provider's native search (Anthropic server-side search; the mock provider).
``none``
    No search; WebSearch reports that no backend is configured.
``auto`` (default)
    Provider-native when the provider supports it, else SearxNG when a URL is
    configured, else Brave when a key is set, else none.

``web.enabled: false`` (the no-web profile) always resolves to None.

:func:`resolve_search_backend` is pure (no network, no state): it returns an
async callable ``(query, *, max_results, allowed_domains, blocked_domains) ->
{"results": [{"title", "url", "snippet", "engine", "published"}], "summary": str,
"cost_usd": float}`` or None. A backend that is named explicitly but cannot
work (no URL, no key, unknown name) resolves to a callable that raises
:class:`SearchBackendError` naming the fix, so the agent and the trace see why.

Domain filters are applied client-side on every result (a domain matches itself
and its subdomains; ``host/path`` entries also match the path prefix) and, when
short, also as ``site:`` / ``-site:`` query terms. A follow-up request with the
plain query (and one more page) fills up the list when the filtered first
request returned fewer results than asked.

The summary is a compact numbered list built without any model call::

    1. Title — https://url (2024-03-05)
       snippet

Errors (unreachable server, HTTP 403/429/5xx, non-JSON answers, rejected API
keys) raise :class:`SearchBackendError` -- a ``ToolFailure`` and a
``RuntimeError`` -- whose message names the URL and the fix. Connection errors
and HTTP 429/5xx are retried ``web.search.retries`` times (default 2). The
literature date ceiling (``web.literature_max_date``) is not applied here;
``web.search.time_range`` (day|week|month|year) is passed to SearxNG's
``time_range`` and Brave's ``freshness``.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

import httpx

from ..envpolicy import redact_url, register_secret, url_secrets
from .base import ToolFailure

log = logging.getLogger(__name__)

#: httpx transport override for every backend built by :func:`resolve_search_backend` (tests).
TRANSPORT: Any = None

USER_AGENT = "vbt-harness/0.2 (research search client)"
SEARXNG_ENV = "SEARXNG_URL"
BRAVE_KEY_ENV = "BRAVE_SEARCH_API_KEY"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
COMPOSE_HINT = "docker compose -f deploy/local/docker-compose.yml up -d searxng"
BACKENDS = ("auto", "searxng", "brave", "provider", "none")

#: ``web.search`` defaults; every key may be overridden in the config.
SEARCH_DEFAULTS: dict[str, Any] = {
    "backend": "auto",              # auto | searxng | brave | provider | none
    "searxng_url": None,            # e.g. http://localhost:8888 (else $SEARXNG_URL)
    "categories": ["general", "science"],
    "engines": [],                  # optional SearxNG engine names (comma list or YAML list)
    "language": "en",
    "safesearch": 0,                # SearxNG 0/1/2; Brave off/moderate/strict
    "time_range": None,             # day | week | month | year (Brave also YYYY-MM-DDtoYYYY-MM-DD)
    "brave_url": BRAVE_URL,
    "brave_api_key": None,          # else $BRAVE_SEARCH_API_KEY
    "brave_cost_per_query": 0.0,    # USD charged per successful Brave request
    "brave_country": None,          # e.g. US
    "brave_max_requests": 2,        # per WebSearch call (each one is billed)
    "timeout_s": 30.0,              # per HTTP request (SearxNG waits for its slowest engine)
    "connect_timeout_s": 10.0,
    "retries": 2,                   # extra attempts on connection errors, HTTP 429 and 5xx
    "retry_backoff_s": 1.0,         # doubled per attempt
    "retry_after_max_s": 10.0,      # cap on a server's Retry-After
    "max_results_cap": 20,
    "max_requests": 3,              # SearxNG requests per WebSearch call (filter follow-ups)
    "site_operators": True,         # add site:/-site: terms for short domain filters
    "max_site_terms": 3,
    "max_query_chars": 380,         # Brave rejects queries over 400 characters
    "snippet_max_chars": 400,       # per result
    "summary_snippet_chars": 240,   # per result in the summary
}

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
#: Markup that search APIs put in titles/snippets. Deliberately narrow: "p < 0.05 and HR > 1" is text.
_TAG = re.compile(r"</?(?:a|b|br|cite|code|div|em|font|i|mark|p|small|span|strong|sub|sup|u)(?:\s[^<>]*)?/?>",
                  re.IGNORECASE)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DATE_RANGE = re.compile(r"^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$")
_ALIASES = {"searx": "searxng", "native": "provider", "off": "none", "disabled": "none", "false": "none",
            "brave_search": "brave", "": "auto", "default": "auto"}
_TIME_RANGES = ("day", "week", "month", "year")
_BRAVE_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}
_SAFESEARCH = {"off": 0, "none": 0, "moderate": 1, "strict": 2}
_BRAVE_SAFESEARCH = ("off", "moderate", "strict")
_WARNED: set[str] = set()


class SearchBackendError(ToolFailure, RuntimeError):
    """A search backend failed or is misconfigured; the message says how to fix it.

    A ``ToolFailure`` (WebSearch returns the message to the model unchanged)
    and a ``RuntimeError``.
    """


# ---------------------------------------------------------------- config


def search_settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """``web.search`` merged over :data:`SEARCH_DEFAULTS` (None values keep the default).

    ``web.search`` may also be a bare backend name (``search: searxng``).
    """
    web = (config or {}).get("web") or {}
    raw = web.get("search") if isinstance(web, Mapping) else None
    if raw is None:
        raw = {}
    elif isinstance(raw, (str, bool)):
        raw = {"backend": raw}
    elif not isinstance(raw, Mapping):
        raw = {}
    return {**SEARCH_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _backend_name(value: Any) -> str:
    if value is False:
        return "none"
    if value is True:
        return "auto"
    name = str(value).strip().lower()
    return _ALIASES.get(name, name)


def searxng_base_url(value: str) -> str:
    """Normalise a SearxNG URL to the instance root (no trailing ``/`` or ``/search``)."""
    url = str(value).strip()
    if "://" not in url:
        url = "http://" + url
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if path.endswith("/search"):
        path = path[: -len("/search")]
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _searxng_url(s: Mapping[str, Any]) -> tuple[str | None, str]:
    if not _blank(s.get("searxng_url")):
        return searxng_base_url(str(s["searxng_url"])), "web.search.searxng_url"
    env = os.environ.get(SEARXNG_ENV, "")
    if env.strip():
        return searxng_base_url(env), SEARXNG_ENV
    return None, ""


def _brave_key(s: Mapping[str, Any]) -> tuple[str | None, str]:
    if not _blank(s.get("brave_api_key")):
        key = str(s["brave_api_key"]).strip()
        register_secret(key, "web.search.brave_api_key")  # a config key is masked in tool output like an env key
        return key, "web.search.brave_api_key"
    env = os.environ.get(BRAVE_KEY_ENV, "")
    if env.strip():
        return env.strip(), BRAVE_KEY_ENV
    return None, ""


def _safesearch_level(value: Any) -> int:
    """0 (off), 1 (moderate) or 2 (strict) from an int or a Brave-style name."""
    if isinstance(value, str) and value.strip().lower() in _SAFESEARCH:
        return _SAFESEARCH[value.strip().lower()]
    try:
        return max(0, min(int(value or 0), 2))
    except (TypeError, ValueError):
        return 0


def _native(provider: Any) -> bool:
    if provider is None:
        return False
    try:
        return bool(provider.supports_web_search())
    except Exception:  # noqa: BLE001 - a broken probe means "no native search"
        return False


@dataclass
class _Choice:
    requested: str
    kind: str | None = None          # searxng | brave | provider | None
    reason: str = ""
    problem: str | None = None
    url: str | None = None
    key: str | None = field(default=None, repr=False)
    settings: dict[str, Any] = field(default_factory=dict, repr=False)


def _choose(config: Mapping[str, Any] | None, provider: Any) -> _Choice:
    web = (config or {}).get("web") or {}
    s = search_settings(config)
    requested = _backend_name(s.get("backend"))
    if isinstance(web, Mapping) and web.get("enabled", True) is False:
        return _Choice(requested, reason="web.enabled is false (no-web run)", settings=s)
    pname = getattr(provider, "name", None) or type(provider).__name__
    if requested == "none":
        return _Choice(requested, reason="web.search.backend is 'none'", settings=s)
    if requested == "provider":
        if _native(provider):
            return _Choice(requested, "provider", reason=f"native search of provider {pname!r}", settings=s)
        return _Choice(requested, problem=(
            f"web.search.backend is 'provider' but provider {pname!r} has no native web search; "
            "set web.search.backend to 'searxng' (self-hosted, see docs/WEB_SEARCH.md) or 'brave'"), settings=s)
    if requested == "searxng":
        url, src = _searxng_url(s)
        if url:
            return _Choice(requested, "searxng", reason=src, url=url, settings=s)
        return _Choice(requested, problem=(
            f"web.search.backend is 'searxng' but no SearxNG URL is configured: set {SEARXNG_ENV} or "
            f"web.search.searxng_url (e.g. http://localhost:8888) and start it with `{COMPOSE_HINT}`"), settings=s)
    if requested == "brave":
        key, src = _brave_key(s)
        if key:
            return _Choice(requested, "brave", reason=src, url=str(s.get("brave_url") or BRAVE_URL), key=key,
                           settings=s)
        return _Choice(requested, problem=(
            f"web.search.backend is 'brave' but no API key is configured: set {BRAVE_KEY_ENV} "
            "(https://api-dashboard.search.brave.com) or web.search.brave_api_key"), settings=s)
    if requested == "auto":
        if _native(provider):
            return _Choice(requested, "provider", reason=f"native search of provider {pname!r}", settings=s)
        url, src = _searxng_url(s)
        if url:
            return _Choice(requested, "searxng", reason=src, url=url, settings=s)
        key, src = _brave_key(s)
        if key:
            return _Choice(requested, "brave", reason=src, url=str(s.get("brave_url") or BRAVE_URL), key=key,
                           settings=s)
        return _Choice(requested, reason=(
            f"no search backend configured: provider {pname!r} has no native search, and neither "
            f"{SEARXNG_ENV} / web.search.searxng_url nor {BRAVE_KEY_ENV} is set"), settings=s)
    return _Choice(requested, problem=(
        f"unknown web.search.backend {s.get('backend')!r}; expected one of {', '.join(BACKENDS)}"), settings=s)


def resolve_search_backend(config: Mapping[str, Any] | None, provider: Any = None, *,
                           transport: Any = None):
    """The WebSearch backend for ``config`` and ``provider``, or None (see module doc).

    Pure: builds a small callable object and touches no network. ``transport``
    (or the module-level :data:`TRANSPORT`) overrides the httpx transport.
    """
    c = _choose(config, provider)
    if c.problem:
        if c.problem not in _WARNED:  # the runtime resolves on every WebSearch call: warn once
            _WARNED.add(c.problem)
            log.warning("web search misconfigured: %s", c.problem)
        return MisconfiguredSearchBackend(c.problem)
    if c.kind is None:
        return None
    if c.kind == "provider":
        return provider.web_search
    tr = transport if transport is not None else TRANSPORT
    if c.kind == "searxng":
        return SearxNGBackend(c.url or "", c.settings, transport=tr)
    return BraveSearchBackend(c.key or "", c.settings, transport=tr)


def describe_search_backend(config: Mapping[str, Any] | None, provider: Any = None) -> dict[str, Any]:
    """What :func:`resolve_search_backend` would pick, without secrets (for doctor/check output).

    ``{"requested", "backend", "url", "reason", "problem"}``; ``backend`` is None
    when there is no usable backend.
    """
    c = _choose(config, provider)
    return {"requested": c.requested, "backend": None if c.problem else c.kind, "url": redact_url(c.url),
            "reason": c.reason, "problem": c.problem}


async def check_search_backend(config: Mapping[str, Any] | None, provider: Any = None, *,
                               query: str = "PCSK9 hypercholesterolemia", max_results: int = 3,
                               transport: Any = None) -> dict[str, Any]:
    """Run one live query through the resolved backend and report ``ok``, ``n_results``,
    ``elapsed_s`` and ``error`` alongside :func:`describe_search_backend`. Never raises
    (except on cancellation)."""
    info = describe_search_backend(config, provider)
    out: dict[str, Any] = {**info, "ok": False, "n_results": 0, "elapsed_s": 0.0, "error": None}
    if info["problem"] or info["backend"] is None:
        out["error"] = info["problem"] or info["reason"]
        return out
    backend = resolve_search_backend(config, provider, transport=transport)
    t0 = time.monotonic()
    try:
        res = await backend(query, max_results=max_results)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        out.update(elapsed_s=round(time.monotonic() - t0, 3), error=f"{type(exc).__name__}: {exc}")
        return out
    results = list((res or {}).get("results") or [])
    out.update(elapsed_s=round(time.monotonic() - t0, 3), n_results=len(results), ok=bool(results))
    if results:
        out["sample"] = {"title": results[0].get("title"), "url": results[0].get("url")}
    else:
        notes = (res or {}).get("notes") or []
        out["error"] = "the backend answered but returned no results" + (f" ({'; '.join(notes)})" if notes else "")
    return out


# ---------------------------------------------------------------- results


def normalize_domain(value: Any) -> str | None:
    """``https://www.Nature.com/articles/`` -> ``nature.com/articles``; None when empty."""
    s = str(value or "").strip().lower()
    if not s:
        return None
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("?", 1)[0].split("#", 1)[0]
    host, _, path = s.partition("/")
    host = host.rsplit("@", 1)[-1]
    if not host.startswith("[") and host.count(":") == 1:
        host = host.split(":", 1)[0]
    host = host.strip(".")
    for prefix in ("*.", "www."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    if not host:
        return None
    path = path.strip("/")
    return host + ("/" + path if path else "")


def _domains(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [x for x in re.split(r"[,\s]+", values) if x]
    out: list[str] = []
    for v in values:
        d = normalize_domain(v)
        if d and d not in out:
            out.append(d)
    return out


def _domain_match(host: str, path: str, entry: str) -> bool:
    dom, _, prefix = entry.partition("/")
    host = host.lower().rstrip(".")
    if not (host == dom or host.endswith("." + dom)):
        return False
    if not prefix:
        return True
    p = path.lstrip("/").lower()
    return p == prefix or p.startswith(prefix + "/")


class DomainFilter:
    """Client-side allow/block filter (a domain covers its subdomains)."""

    def __init__(self, allowed: Any = None, blocked: Any = None) -> None:
        self.allowed = _domains(allowed)
        self.blocked = [] if self.allowed else _domains(blocked)  # an allow-list excludes everything else

    @property
    def active(self) -> bool:
        return bool(self.allowed or self.blocked)

    def allows(self, url: str) -> bool:
        try:
            parts = urlsplit(url)
        except ValueError:
            return False
        host = parts.hostname or ""
        if self.allowed:
            return any(_domain_match(host, parts.path, d) for d in self.allowed)
        return not any(_domain_match(host, parts.path, d) for d in self.blocked)

    def query_terms(self, max_terms: int) -> str:
        """``site:``/``-site:`` terms, or "" when the lists are too long to add."""
        if self.allowed and len(self.allowed) <= max_terms:
            if len(self.allowed) == 1:
                return f"site:{self.allowed[0]}"
            return "(" + " OR ".join(f"site:{d}" for d in self.allowed) + ")"
        if self.blocked and len(self.blocked) <= max_terms:
            return " ".join(f"-site:{d}" for d in self.blocked)
        return ""


def clean_text(value: Any, limit: int = 0) -> str:
    """Strip tags, unescape entities, collapse whitespace; cut at ``limit`` chars (0 = no cut)."""
    s = html.unescape(_TAG.sub(" ", str(value or "")))
    s = " ".join(s.split())
    if limit and len(s) > limit:
        s = s[: max(1, limit - 1)].rstrip() + "…"
    return s


def _published(value: Any) -> str | None:
    if _blank(value):
        return None
    s = str(value).strip()
    m = _ISO_DATE.match(s)
    return m.group(0) if m else s[:40]


def _url_key(url: str) -> str:
    try:
        p = urlsplit(url.strip())
    except ValueError:
        return url
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return f"{host}{p.path.rstrip('/')}?{p.query}"


def format_summary(results: list[dict[str, Any]], *, snippet_chars: int = 240,
                   notes: list[str] | None = None) -> str:
    """Numbered ``title — url (date)`` lines with an indented snippet each; no model call."""
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        head = f"{i}. {r.get('title') or r.get('url')} — {r.get('url')}"
        if r.get("published"):
            head += f" ({r['published']})"
        lines.append(head)
        snippet = clean_text(r.get("snippet"), snippet_chars)
        if snippet:
            lines.append(f"   {snippet}")
    if not results:
        lines.append("No results.")
    for n in notes or []:
        lines.append(f"[{n}]")
    return "\n".join(lines)


def _is_local(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if host in ("localhost", "searxng") or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # docker service names (searxng, search, ...)
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _retry_after(resp: httpx.Response | None) -> float | None:
    if resp is None:
        return None
    for name in ("retry-after", "x-ratelimit-reset"):
        raw = resp.headers.get(name)
        if not raw:
            continue
        try:
            return float(str(raw).split(",")[0].strip())
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------- backends


class SearchBackend:
    """Shared request plan, retries, filtering and result shaping for HTTP backends."""

    name = "base"
    label = "search backend"

    def __init__(self, url: str, settings: Mapping[str, Any] | None = None, *, transport: Any = None) -> None:
        self.url = url
        # Shown in errors the model sees, describe() and logs: never the user:password@ part
        # (a SearxNG behind basic auth); the password is also masked in tool output.
        self.shown_url = redact_url(url)
        register_secret(url_secrets(url), "web.search URL credentials")
        self.settings = {**SEARCH_DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None}}
        self.transport = transport

    # -- subclass hooks

    async def _page(self, client: httpx.AsyncClient, query: str, page: int,
                    count: int) -> tuple[list[dict[str, Any]], list[str]]:
        """One request: (results, notes)."""
        raise NotImplementedError

    def _max_requests(self) -> int:
        return max(1, int(self.settings.get("max_requests") or 1))

    def _request_cost(self) -> float:
        return 0.0

    def _unreachable(self, exc: Exception) -> SearchBackendError:
        return SearchBackendError(f"{self.label} at {self.shown_url} is unreachable ({type(exc).__name__}: {exc})")

    def _timed_out(self, exc: Exception) -> SearchBackendError:
        return SearchBackendError(
            f"{self.label} at {self.shown_url} did not answer within {self.settings['timeout_s']} s "
            f"({type(exc).__name__}); raise web.search.timeout_s or check the server")

    def _follow_redirects(self) -> bool:
        return True

    # -- plumbing

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "url": self.shown_url}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={self.shown_url!r})"

    def _client(self) -> httpx.AsyncClient:
        timeout = float(self.settings.get("timeout_s") or 30.0)
        connect = min(timeout, float(self.settings.get("connect_timeout_s") or timeout))
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(timeout, connect=connect),
            "follow_redirects": self._follow_redirects(),
            "headers": {"User-Agent": USER_AGENT, "Accept": "application/json",
                        "Accept-Language": "en-US,en;q=0.9"},
            # A local SearxNG is reached directly even when HTTP(S)_PROXY is set.
            "trust_env": not _is_local(self.url),
        }
        if self.transport is not None:
            kwargs["transport"] = self.transport
        return httpx.AsyncClient(**kwargs)

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> float:
        cap = float(self.settings.get("retry_after_max_s") or 0.0)
        after = _retry_after(resp)
        if after is not None:
            return max(0.0, min(after, cap))
        return max(0.0, float(self.settings.get("retry_backoff_s") or 0.0)) * (2 ** attempt)

    async def _get(self, client: httpx.AsyncClient, url: str, *, params: Mapping[str, Any],
                   headers: Mapping[str, str] | None = None) -> tuple[httpx.Response, int]:
        """GET with retries on connection errors and HTTP 429/5xx; returns (response, attempts)."""
        attempts = 1 + max(0, int(self.settings.get("retries") or 0))
        for i in range(attempts):
            last = i + 1 >= attempts
            try:
                resp = await client.get(url, params=dict(params), headers=dict(headers or {}))
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                raise self._timed_out(exc) from None
            except httpx.TransportError as exc:
                if last:
                    raise self._unreachable(exc) from None
                log.info("%s request failed (%s); retrying", self.label, type(exc).__name__)
                await asyncio.sleep(self._backoff(i, None))
                continue
            if resp.status_code in _RETRY_STATUS and not last:
                log.info("%s answered HTTP %s; retrying", self.label, resp.status_code)
                await asyncio.sleep(self._backoff(i, resp))
                continue
            return resp, i + 1
        raise AssertionError("unreachable")  # pragma: no cover

    async def __call__(self, query: str, *, max_results: int = 8, allowed_domains: list[str] | None = None,
                       blocked_domains: list[str] | None = None) -> dict[str, Any]:
        query = " ".join(str(query or "").split())
        if not query:
            raise SearchBackendError("query is required")
        if allowed_domains and blocked_domains:
            raise SearchBackendError("pass allowed_domains or blocked_domains, not both")
        cap = max(1, int(self.settings.get("max_results_cap") or 20))
        n = max(1, min(int(max_results or 8), cap))
        filt = DomainFilter(allowed_domains, blocked_domains)
        plan = self._plan(query, filt)
        count = cap if filt.active else n
        results: list[dict[str, Any]] = []
        notes: list[str] = []
        seen: set[str] = set()
        removed = 0
        cost = 0.0
        async with self._client() as client:
            for i, (q, page) in enumerate(plan):
                try:
                    items, page_notes = await self._page(client, q, page, count)
                except SearchBackendError as exc:
                    if i == 0:
                        raise
                    notes.append(f"a follow-up request failed: {exc}")
                    break
                cost += self._request_cost()
                for note in page_notes:
                    if note not in notes:
                        notes.append(note)
                for it in items:
                    key = _url_key(it["url"])
                    if key in seen:
                        continue
                    seen.add(key)
                    if not filt.allows(it["url"]):
                        removed += 1
                        continue
                    results.append(it)
                if len(results) >= n or (not items and page > 1):
                    break
        if removed:
            notes.append(f"{removed} result(s) outside the domain filter were dropped")
        results = results[:n]
        out: dict[str, Any] = {
            "results": results,
            "summary": format_summary(results, snippet_chars=int(self.settings.get("summary_snippet_chars") or 240),
                                      notes=notes),
            "cost_usd": round(cost, 6),
            "backend": self.name,
        }
        if notes:
            out["notes"] = notes
        return out

    def _plan(self, query: str, filt: DomainFilter) -> list[tuple[str, int]]:
        """(query, page) requests: site-term query first, then the plain query (+ page 2) to fill up."""
        plan: list[tuple[str, int]] = []
        if filt.active and self.settings.get("site_operators", True):
            terms = filt.query_terms(int(self.settings.get("max_site_terms") or 0))
            augmented = f"{query} {terms}" if terms else query
            if terms and len(augmented) <= int(self.settings.get("max_query_chars") or 380):
                plan.append((augmented, 1))
        plan.append((query, 1))
        if filt.active:
            plan.append((query, 2))
        return plan[: self._max_requests()]

    def _item(self, *, title: Any, url: Any, snippet: Any, engine: Any, published: Any) -> dict[str, Any] | None:
        u = str(url or "").strip()
        if not u.lower().startswith(("http://", "https://")):
            return None
        return {"title": clean_text(title, 300) or u, "url": u,
                "snippet": clean_text(snippet, int(self.settings.get("snippet_max_chars") or 400)),
                "engine": str(engine or self.name), "published": _published(published)}


class SearxNGBackend(SearchBackend):
    """A SearxNG instance's JSON API (``search.formats`` must include ``json``)."""

    name = "searxng"
    label = "SearxNG"

    def __init__(self, url: str, settings: Mapping[str, Any] | None = None, *, transport: Any = None) -> None:
        super().__init__(searxng_base_url(url), settings, transport=transport)

    def _unreachable(self, exc: Exception) -> SearchBackendError:
        return SearchBackendError(
            f"SearxNG at {self.shown_url} is unreachable ({type(exc).__name__}: {exc}). Start it with "
            f"`{COMPOSE_HINT}` (see docs/WEB_SEARCH.md) or point {SEARXNG_ENV} / web.search.searxng_url "
            "at a running instance")

    def _params(self, query: str, page: int) -> dict[str, Any]:
        s = self.settings
        params: dict[str, Any] = {"q": query, "format": "json", "language": s.get("language") or "en",
                                  "safesearch": _safesearch_level(s.get("safesearch")), "pageno": page}
        for key in ("categories", "engines"):
            v = s.get(key)
            if v:
                params[key] = v if isinstance(v, str) else ",".join(str(x) for x in v)
        tr = str(s.get("time_range") or "").strip().lower()
        if tr in _TIME_RANGES:
            params["time_range"] = tr
        return params

    def _http_error(self, resp: httpx.Response, attempts: int) -> SearchBackendError:
        code = resp.status_code
        where = f"SearxNG at {self.shown_url}"
        if code == 403:
            return SearchBackendError(
                f"{where} refused the JSON API (HTTP 403): add json to search.formats in its settings.yml "
                "(deploy/local/searxng/settings.yml has `formats: [html, json]`) and restart it")
        if code == 429:
            return SearchBackendError(
                f"{where} rate-limited the request (HTTP 429) after {attempts} attempt(s): its bot limiter is on; "
                "set server.limiter: false for a private instance (deploy/local/searxng/settings.yml does)")
        if code == 404:
            return SearchBackendError(
                f"no SearxNG search endpoint at {self.shown_url}/search (HTTP 404); web.search.searxng_url / "
                f"{SEARXNG_ENV} must be the instance root, e.g. http://localhost:8888")
        if code >= 500:
            return SearchBackendError(
                f"{where} failed with HTTP {code} after {attempts} attempt(s); check its logs "
                "(`docker compose -f deploy/local/docker-compose.yml logs searxng`)")
        return SearchBackendError(f"{where} answered HTTP {code}: {clean_text(resp.text, 200)}")

    async def _page(self, client: httpx.AsyncClient, query: str, page: int,
                    count: int) -> tuple[list[dict[str, Any]], list[str]]:
        resp, attempts = await self._get(client, f"{self.url}/search", params=self._params(query, page))
        if resp.status_code != 200:
            raise self._http_error(resp, attempts)
        try:
            data = resp.json()
        except ValueError:
            ctype = resp.headers.get("content-type", "?")
            raise SearchBackendError(
                f"SearxNG at {self.shown_url} returned non-JSON ({ctype}); web.search.searxng_url must be the "
                "instance root and search.formats must include json") from None
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchBackendError(f"SearxNG at {self.shown_url} returned an unexpected JSON shape (no results list)")
        items: list[dict[str, Any]] = []
        for r in data["results"]:
            if not isinstance(r, dict):
                continue
            engines = r.get("engines") or []
            item = self._item(title=r.get("title"), url=r.get("url"), snippet=r.get("content"),
                              engine=r.get("engine") or (engines[0] if engines else None),
                              published=r.get("publishedDate"))
            if item is not None:
                items.append(item)
        notes: list[str] = []
        failed = []
        for e in data.get("unresponsive_engines") or []:
            if isinstance(e, (list, tuple)) and e:
                failed.append(f"{e[0]}: {e[1]}" if len(e) > 1 else str(e[0]))
            elif isinstance(e, dict):
                failed.append(f"{e.get('engine') or e.get('name')}: {e.get('error') or e.get('message')}")
            elif e:
                failed.append(str(e))
        if failed:
            notes.append("engines that did not answer: " + ", ".join(failed))
        return items, notes


class BraveSearchBackend(SearchBackend):
    """The Brave Search web API (``X-Subscription-Token``)."""

    name = "brave"
    label = "Brave Search API"

    def __init__(self, api_key: str, settings: Mapping[str, Any] | None = None, *, transport: Any = None) -> None:
        merged = {**SEARCH_DEFAULTS, **{k: v for k, v in (settings or {}).items() if v is not None}}
        super().__init__(str(merged.get("brave_url") or BRAVE_URL), merged, transport=transport)
        self._key = api_key

    def _max_requests(self) -> int:
        return max(1, int(self.settings.get("brave_max_requests") or 1))

    def _request_cost(self) -> float:
        return float(self.settings.get("brave_cost_per_query") or 0.0)

    def _follow_redirects(self) -> bool:
        return False  # never forward the subscription token to another host

    def _unreachable(self, exc: Exception) -> SearchBackendError:
        return SearchBackendError(
            f"Brave Search API at {self.shown_url} is unreachable ({type(exc).__name__}: {exc}); check network "
            "access (HTTPS proxy) or use the self-hosted SearxNG backend")

    def _params(self, query: str, page: int, count: int) -> dict[str, Any]:
        s = self.settings
        q = query if len(query) <= 400 else query[:400].rsplit(" ", 1)[0]
        params: dict[str, Any] = {
            "q": q, "count": max(1, min(int(count), 20)), "offset": max(0, min(page - 1, 9)),
            "safesearch": _BRAVE_SAFESEARCH[_safesearch_level(s.get("safesearch"))],
            "text_decorations": "false", "result_filter": "web",
        }
        lang = str(s.get("language") or "").strip()
        if lang and lang != "all":
            params["search_lang"] = lang
        if s.get("brave_country"):
            params["country"] = str(s["brave_country"])
        tr = str(s.get("time_range") or "").strip().lower()
        if tr in _BRAVE_FRESHNESS:
            params["freshness"] = _BRAVE_FRESHNESS[tr]
        elif _DATE_RANGE.match(tr):
            params["freshness"] = tr
        return params

    def _http_error(self, resp: httpx.Response, attempts: int) -> SearchBackendError:
        code = resp.status_code
        detail = ""
        try:
            body = resp.json()
        except ValueError:
            detail = clean_text(resp.text, 200)
        else:
            err = body.get("error") if isinstance(body, dict) else None
            if isinstance(err, dict):
                detail = " ".join(str(x) for x in (err.get("code"), err.get("detail")) if x)
            elif err:
                detail = str(err)
        detail = clean_text(detail, 300)
        if self._key:
            detail = detail.replace(self._key, "***")  # never echo the key
        tail = f": {detail}" if detail else ""
        if code in (401, 403) or "TOKEN" in detail.upper():
            return SearchBackendError(
                f"Brave Search API rejected the API key (HTTP {code}{tail}); check {BRAVE_KEY_ENV} "
                "(or web.search.brave_api_key)")
        if code == 429:
            return SearchBackendError(
                f"Brave Search API rate limit or quota exceeded (HTTP 429{tail}) after {attempts} attempt(s)")
        if code >= 500:
            return SearchBackendError(f"Brave Search API failed with HTTP {code} after {attempts} attempt(s){tail}")
        return SearchBackendError(f"Brave Search API rejected the request (HTTP {code}{tail})")

    async def _page(self, client: httpx.AsyncClient, query: str, page: int,
                    count: int) -> tuple[list[dict[str, Any]], list[str]]:
        resp, attempts = await self._get(client, self.url, params=self._params(query, page, count),
                                         headers={"X-Subscription-Token": self._key, "Accept-Encoding": "gzip"})
        if resp.status_code != 200:
            raise self._http_error(resp, attempts)
        try:
            data = resp.json()
        except ValueError:
            raise SearchBackendError(f"Brave Search API at {self.shown_url} returned non-JSON") from None
        web = (data or {}).get("web") if isinstance(data, dict) else None
        rows = (web or {}).get("results") if isinstance(web, dict) else None
        items: list[dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            item = self._item(title=r.get("title"), url=r.get("url"), snippet=r.get("description"),
                              engine="brave", published=r.get("page_age") or r.get("age"))
            if item is not None:
                items.append(item)
        return items, []


class MisconfiguredSearchBackend:
    """Stands in for a backend that was named explicitly but cannot work; every call
    raises :class:`SearchBackendError` with ``problem`` (so the agent sees the fix)."""

    name = "misconfigured"

    def __init__(self, problem: str) -> None:
        self.problem = problem

    def describe(self) -> dict[str, Any]:
        return {"backend": None, "problem": self.problem}

    def __repr__(self) -> str:
        return f"MisconfiguredSearchBackend({self.problem!r})"

    async def __call__(self, query: str, *, max_results: int = 8, allowed_domains: list[str] | None = None,
                       blocked_domains: list[str] | None = None) -> dict[str, Any]:
        raise SearchBackendError(self.problem)


__all__ = [
    "BACKENDS", "SEARCH_DEFAULTS", "BraveSearchBackend", "DomainFilter", "MisconfiguredSearchBackend",
    "SearchBackend", "SearchBackendError", "SearxNGBackend", "check_search_backend", "clean_text",
    "describe_search_backend", "format_summary", "normalize_domain", "resolve_search_backend",
    "search_settings", "searxng_base_url",
]

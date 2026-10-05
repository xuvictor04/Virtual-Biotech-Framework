# Web search backends

Agents' `WebSearch` tool needs a search engine. With Claude, the Anthropic provider runs
searches server-side. Local open-weight models (vLLM / SGLang serving Qwen3.8) have no native
search, so the harness calls a search API itself. The code is in `src/vbt/tools/search_backends.py`.

| backend | what it is | needs | cost |
|---|---|---|---|
| `searxng` | self-hosted [SearxNG](https://docs.searxng.org/) metasearch, JSON API | `SEARXNG_URL` or `web.search.searxng_url` | free |
| `brave` | [Brave Search API](https://api-dashboard.search.brave.com) | `BRAVE_SEARCH_API_KEY` | `web.search.brave_cost_per_query` per request |
| `provider` | the provider's own search (Anthropic server tool; the mock provider) | a provider with native search | provider-billed |
| `none` | no search; `WebSearch` reports "no web search backend is configured" | | |
| `auto` | the default; see the order below | | |

`auto` picks the first that applies:

1. the provider's native search, when the provider has one (Claude profiles, the mock provider);
2. SearxNG, when `web.search.searxng_url` or `SEARXNG_URL` is set;
3. Brave, when `BRAVE_SEARCH_API_KEY` (or `web.search.brave_api_key`) is set;
4. none.

With `web.enabled: false` (the `no-web` profile of case study 2), every setting resolves to no
search: `WebSearch` and `WebFetch` are removed from the agents and the tool refuses to run.

To make a Claude run use SearxNG instead of Anthropic's search, set `web.search.backend: searxng`.

## Quick start (SearxNG)

```bash
# from the repository root
docker compose -f deploy/local/docker-compose.yml up -d searxng
export SEARXNG_URL=http://localhost:8888          # or put it in .env
curl -s "$SEARXNG_URL/search?q=PCSK9&format=json" | head -c 300   # JSON, not HTML or HTTP 403
```

`deploy/local/searxng/settings.yml` is a ready configuration. It is mounted read-only at
`/etc/searxng/settings.yml`. The container listens on 8080, published on the host as 8888.
It keeps SearxNG's defaults and overrides only these settings:

- **JSON API on** (`search.formats: [html, json]`). Without it SearxNG answers HTTP 403.
- **Bot limiter off** (`server.limiter: false`). Scripted clients are not answered with HTTP 429,
  and no valkey/redis container is needed.
- **`safe_search: 0`, English results** (`default_lang: en`).
- **A small engine set** (`use_default_settings.engines.keep_only`):
  - general web: DuckDuckGo, Brave, Wikipedia (Wikipedia pages are returned as results, not only
    as infoboxes);
  - literature: PubMed, arXiv, Google Scholar, Semantic Scholar, OpenAIRE, OpenAlex, PDBe.
- **Engines kept but disabled**: Google, Bing and Qwant are disabled upstream because they block
  scrapers. Crossref is slow. To try any of them, set `disabled: false` for it in the
  `engines:` list.
- **`secret_key`**: `$SEARXNG_SECRET` overrides it. Any value other than the stock
  `ultrasecretkey` lets the instance start.

Bare metal (no Docker), following the SearxNG install docs:

```bash
SEARXNG_SETTINGS_PATH=$PWD/deploy/local/searxng/settings.yml \
SEARXNG_PORT=8888 SEARXNG_BIND_ADDRESS=127.0.0.1 python -m searx.webapp
```

Do not publish this instance on the internet. With the limiter off, anyone who can reach it can
use it, and upstream engines will then block your IP. Bind it to localhost or a private network.

SearxNG answers when its slowest engine answers or times out (`outgoing.max_request_timeout`,
15 s here). Keep `web.search.timeout_s` above that value. Engines that time out or are blocked are
listed in the result's `notes` and summary as `engines that did not answer: ...`. They are not
errors.

## Brave Search API

```bash
export BRAVE_SEARCH_API_KEY=...        # https://api-dashboard.search.brave.com
```

Set `web.search.backend: brave`, or leave it on `auto` with no SearxNG URL configured. Each HTTP
request is charged to the calling agent at `web.search.brave_cost_per_query` USD (default 0.0; set
it to your plan's price). A filtered search can make up to `brave_max_requests` requests. The key
is sent only as the `X-Subscription-Token` header to `brave_url`. Redirects are never followed, and
error messages never repeat the key.

## Configuration (`web.search`)

```yaml
web:
  enabled: true
  search:
    backend: auto                         # auto | searxng | brave | provider | none
    searxng_url: ${SEARXNG_URL:-}         # e.g. http://localhost:8888 (instance root)
```

Every key is optional. A missing key or a `null` value uses the default. `web.search: searxng`
(a bare string) is shorthand for `web.search.backend: searxng`.

| key | default | meaning |
|---|---|---|
| `backend` | `auto` | `auto`, `searxng`, `brave`, `provider` or `none` |
| `searxng_url` | unset → `$SEARXNG_URL` | SearxNG instance root. A trailing `/search` is accepted. |
| `categories` | `[general, science]` | SearxNG categories (list or comma string) |
| `engines` | `[]` | SearxNG engine names to query, in addition to the categories |
| `language` | `en` | SearxNG `language`, Brave `search_lang` (`all` = no language filter) |
| `safesearch` | `0` | 0/1/2 or off/moderate/strict (SearxNG `safesearch`, Brave `safesearch`) |
| `time_range` | `null` | `day`/`week`/`month`/`year` → SearxNG `time_range`, Brave `freshness`. Brave also accepts `YYYY-MM-DDtoYYYY-MM-DD`. |
| `brave_url` | `https://api.search.brave.com/res/v1/web/search` | Brave endpoint |
| `brave_api_key` | unset → `$BRAVE_SEARCH_API_KEY` | prefer the environment variable |
| `brave_cost_per_query` | `0.0` | USD charged per successful Brave request |
| `brave_country` | `null` | Brave `country` (e.g. `US`) |
| `brave_max_requests` | `2` | Brave requests per `WebSearch` call (filter follow-ups) |
| `timeout_s` | `30` | per HTTP request |
| `connect_timeout_s` | `10` | TCP/TLS connect timeout |
| `retries` | `2` | extra attempts on connection errors, HTTP 429 and 5xx (a read timeout is not retried) |
| `retry_backoff_s` | `1.0` | first retry delay, doubled per attempt |
| `retry_after_max_s` | `10` | cap on the delay a server asks for in its `Retry-After` header |
| `max_results_cap` | `20` | upper bound on the tool's `max_results` |
| `max_requests` | `3` | SearxNG requests per `WebSearch` call (filter follow-ups) |
| `site_operators` | `true` | add `site:` / `-site:` terms to the query for short domain lists |
| `max_site_terms` | `3` | longer domain lists are filtered client-side only |
| `max_query_chars` | `380` | no site terms when the query would exceed this length (Brave rejects more than 400 characters) |
| `snippet_max_chars` | `400` | snippet length in `results` |
| `summary_snippet_chars` | `240` | snippet length in `summary` |

A local SearxNG URL (loopback, private address, `localhost`, a Docker service name) is called
directly, even when `HTTP(S)_PROXY` is set. Brave goes through the environment's proxy and CA
settings.

The literature date ceiling (`web.literature_max_date`) is not applied here. PubMed enforces it.

## What the tool returns

```json
{
  "results": [
    {"title": "...", "url": "https://...", "snippet": "...", "engine": "pubmed", "published": "2017-05-04"}
  ],
  "summary": "1. Title — https://url (2017-05-04)\n   snippet\n2. ...",
  "backend": "searxng",
  "notes": ["engines that did not answer: google scholar: timeout"]
}
```

The summary is built without a model call. `WebSearch` wraps it in
`<untrusted-web-content source="web_search:...">` delimiters and charges `cost_usd` (removed from
the result) to the calling agent.

## Domain filters

`allowed_domains` and `blocked_domains` come from the tool arguments or from
`web.allowed_domains` / `web.blocked_domains`. Pass one or the other, not both. Filters are applied
twice:

- **Client-side, on every result.** A domain matches itself and its subdomains, so `nature.com`
  matches `www.nature.com` and `news.nature.com` but not `notnature.com`. An entry with a path
  (`fda.gov/drugs`) also matches that path prefix. Schemes, `www.` and `*.` prefixes are ignored.
- **In the query, when the list is short.** One allowed domain adds `site:d`. Several add
  `(site:a OR site:b)`. Blocked domains add `-site:d` terms.

When the first, site-restricted request returns fewer matches than asked, one or two follow-up
requests fill the list: the plain query (literature engines such as PubMed ignore `site:`), then
its next page. The number of requests is capped by `max_requests` / `brave_max_requests`. Results
are deduplicated by URL.

## Errors

Failures raise `SearchBackendError`. It is both a `ToolFailure` and a `RuntimeError`, so the agent
receives the message unchanged. Each message names the URL and the fix:

| situation | message says |
|---|---|
| SearxNG down / wrong port | unreachable; start it with `docker compose -f deploy/local/docker-compose.yml up -d searxng` or fix `SEARXNG_URL` |
| HTTP 403 | add `json` to `search.formats` |
| HTTP 429 | the bot limiter is on: set `server.limiter: false` |
| HTTP 404 / HTML answer | the URL must be the instance root (e.g. `http://localhost:8888`) |
| HTTP 5xx after retries | check `docker compose ... logs searxng` |
| timeout | raise `web.search.timeout_s` or check the server |
| Brave 401/403/422 token | check `BRAVE_SEARCH_API_KEY` |
| Brave 429 | rate limit or quota exceeded |
| `backend: searxng` without a URL, `brave` without a key, `provider` without native search, unknown name | how to configure it |

An explicitly named backend that cannot work does not silently fall back. It resolves to a stub
whose every call raises the configuration problem, so both the agent and the trace show it.

## Programmatic use

```python
from vbt.tools.search_backends import resolve_search_backend, describe_search_backend, check_search_backend

backend = resolve_search_backend(config, provider)    # pure: async callable or None
info = describe_search_backend(config, provider)      # {"requested", "backend", "url", "reason", "problem"}
report = await check_search_backend(config, provider) # one live query: {"ok", "n_results", "elapsed_s", "error", ...}
res = await backend("PCSK9 loss of function", max_results=5, allowed_domains=["nih.gov"])
```

`Runtime.search_backend` returns `resolve_search_backend(runtime.config, runtime.provider)`, so
`WebSearch` picks up configuration changes on its next call.

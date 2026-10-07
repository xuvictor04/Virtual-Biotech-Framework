"""``live_api``: tables read page by page from a REST API (phase 4, F20). Stdlib only (no pyarrow).

Implements the phase-1 ``live`` capability (:meth:`LiveApiLayout.request` -> ``Page(rows, total, next,
as_of)``) and ``count`` (:meth:`LiveApiLayout.count`: the **remote witness**, an independent count request
compiled from the bound predicates, §11.6). Options (``layout: {plugin: live_api, options: {...}}``)::

    base_url:        https://clinicaltrials.gov/api/v2       (an ${ENV} override points tests at a stub)
    endpoint:        studies                                  (default: the table's path; {column}
                                                              placeholders are filled from Eq conjuncts)
    params:          {format: json}                           static parameters of every request
    rows_path / total_path / next_path / as_of_path           how a page is decoded (rest_json)
    page_size_param: pageSize      page_token_param: pageToken      offset_param: retstart
    filters / remote_names / essie_param / text_params        how predicates compile (rest_json)
    count:           {endpoint, params: {countTotal: "true", pageSize: "0"}, total_path: $.totalCount}
                     (or header: total-count, for APIs that report the total in a response header)
    release:         {endpoint: version, path: $.dataTimestamp}   (the source's data release)

Requests honour the descriptor's ``budget`` (:class:`~vbt.datalayer.descriptor.models.RemoteBudget`):
``requests_per_min`` spaces requests per base URL, ``max_requests_per_call`` and ``max_pages`` bound one
call, ``page_size`` sets the page parameter and ``timeout_s`` each request. Nothing here touches the
network at import, in ``signature``, ``fingerprint`` or ``probe``: readiness of a live table is "checked at
call time" (the probe is an info finding), and ``as_of`` is the release the last request saw, else the
call time.

Also here, for the remote sources' descriptors (§6.1, §13): :func:`release_from_payload` (``release.resolve:
{result: <JSONPath>}``), :class:`RemoteVocab` (vocabularies with a TTL and drift detection) and
:class:`RecordVersions` (``key.version``: a record whose version changed since it was last seen is
``source_updated``, which provenance records).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping

from ...predicate import And, Eq, Predicate, evaluate
from ..base import CheckItem, Fragment, LayoutSpec, Manifest, Page, PluginBase
from ..formats.rest_json import RestJsonFormat, jp_values
from ..registry import register

__all__ = ["LiveApiLayout", "RemoteError", "Budget", "http_get", "release_from_payload", "RemoteVocab",
           "RecordVersions", "fetch_all", "LOCAL_HOSTS"]

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_PAGE_SIZE = 100

_LOCK = threading.Lock()
_LAST_REQUEST: dict[str, float] = {}
_AS_OF: dict[str, str] = {}


class RemoteError(Exception):
    """A request that failed: ``status`` is the HTTP status (None for a transport error)."""

    def __init__(self, message: str, *, status: int | None = None, url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.url = url


@dataclass
class Budget:
    """The per-call request budget (from ``RemoteBudget``) and what a call has used."""

    requests_per_min: int | None = None
    max_pages: int | None = None
    page_size: int | None = None
    max_requests_per_call: int | None = None
    timeout_s: float | None = None
    used: int = 0

    @classmethod
    def of(cls, budget: Any) -> "Budget":
        if isinstance(budget, Budget):
            return budget
        if budget is None:
            return cls()
        get = (lambda k: budget.get(k)) if isinstance(budget, Mapping) else (lambda k: getattr(budget, k, None))
        return cls(requests_per_min=get("requests_per_min"), max_pages=get("max_pages"),
                   page_size=get("page_size"), max_requests_per_call=get("max_requests_per_call"),
                   timeout_s=get("timeout_s"))

    def spend(self) -> None:
        if self.max_requests_per_call is not None and self.used >= int(self.max_requests_per_call):
            raise RemoteError(f"the request budget of {self.max_requests_per_call} request(s) per call is used up")
        self.used += 1


def _wait_turn(base: str, requests_per_min: int | None) -> None:
    if not requests_per_min:
        return
    gap = 60.0 / float(requests_per_min)
    with _LOCK:
        now = time.monotonic()
        last = _LAST_REQUEST.get(base, 0.0)
        delay = last + gap - now
        _LAST_REQUEST[base] = max(now, last + gap)
    if delay > 0:
        time.sleep(delay)


def http_get(url: str, params: Mapping[str, Any] | None = None, *, timeout: float = DEFAULT_TIMEOUT_S,
             headers: Mapping[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    """``(status, headers, body)`` of a GET. Local hosts bypass any proxy; HTTP errors return their status."""
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
    full = f"{url}?{query}" if query else url
    req = urllib.request.Request(full, headers={"Accept": "application/json", "User-Agent": "vbt-datalayer",
                                                **dict(headers or {})})
    host = urllib.parse.urlparse(full).hostname or ""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if host in LOCAL_HOSTS else \
        urllib.request.build_opener()
    try:
        with opener.open(req, timeout=timeout) as resp:
            return int(resp.status), {k: v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        return int(exc.code), {k: v for k, v in (exc.headers or {}).items()}, body
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RemoteError(f"request to {url} failed: {exc}", url=url) from exc


def release_from_payload(resolve: Mapping[str, Any] | None, payload: Any) -> str | None:
    """The release a result names (``release.resolve: {result: "$.census_version"}``), else None."""
    path = (resolve or {}).get("result") if isinstance(resolve, Mapping) else None
    if not path:
        return None
    got = [v for v in jp_values(payload, str(path)) if v not in (None, "")]
    return str(got[0]) if got else None


@register
class LiveApiLayout(PluginBase):
    kind: ClassVar[str] = "layout"
    name: ClassVar[str] = "live_api"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"live", "count"})

    #: The HTTP transport (a seam: tests may replace it on an instance).
    transport: Callable[..., tuple[int, dict[str, str], bytes]] = staticmethod(http_get)

    def __init__(self) -> None:
        self.fmt = RestJsonFormat()

    # ------------------------------------------------------------------ identity (no network)

    @staticmethod
    def _options(spec: LayoutSpec) -> dict[str, Any]:
        return dict(spec.options or {})

    def _url(self, spec: LayoutSpec, endpoint: str | None = None) -> str | None:
        opts = self._options(spec)
        base = str(opts.get("base_url") or "").rstrip("/")
        if not base:
            return None
        ep = endpoint if endpoint is not None else (opts.get("endpoint") or spec.path or "")
        ep = str(ep).lstrip("/")
        return f"{base}/{ep}" if ep else base

    def fragments(self, root: str, spec: LayoutSpec) -> list[Fragment]:
        return []

    def partition_columns(self, spec: LayoutSpec) -> dict[str, str]:
        return {}

    def signature(self, root: str, spec: LayoutSpec) -> str:
        key = json.dumps([spec.table, self._url(spec)], sort_keys=True)
        return "sig1:live_api:" + hashlib.sha256(key.encode()).hexdigest()[:16]

    def fingerprint(self, frags: list[Fragment], manifest: Manifest | None) -> str:
        return "fp1:live_api"

    def table_fingerprint(self, spec: LayoutSpec) -> str:
        """``fp1:live:<url hash>[:<as_of>]``: a live table is identified by its endpoint and the release
        the last request saw."""
        url = self._url(spec) or ""
        as_of = _AS_OF.get(url)
        return "fp1:live:" + hashlib.sha256(url.encode()).hexdigest()[:16] + (f":{as_of}" if as_of else "")

    def partition_fingerprints(self, frags: list[Fragment], manifest: Manifest | None) -> dict[str, str]:
        return {}

    def probe(self, root: str, spec: LayoutSpec, manifest: Manifest | None) -> list[CheckItem]:
        url = self._url(spec)
        if not url:
            return [CheckItem("location", False, f"{spec.table}: live_api needs options.base_url",
                              hint="set layout.options.base_url in the descriptor", level="error")]
        return [CheckItem("upstream_only", True, f"live source {url} (checked at call time)", level="info")]

    def as_of(self, root: str, spec: LayoutSpec) -> str | None:
        return _AS_OF.get(self._url(spec) or "") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------ requests

    def request_map(self, spec: LayoutSpec, remote_names: Mapping[str, str] | None = None) -> dict[str, Any]:
        """The compile map of ``rest_json`` from the options (and the descriptor's ``remote_name`` facets)."""
        opts = self._options(spec)
        return {"filters": dict(opts.get("filters") or {}),
                "remote_names": {**dict(remote_names or {}), **dict(opts.get("remote_names") or {})},
                "essie_param": opts.get("essie_param"), "text_params": dict(opts.get("text_params") or {})}

    @staticmethod
    def fill_path(endpoint: str, predicate: Predicate | None) -> tuple[str, Predicate | None]:
        """Fill ``{column}`` placeholders of an endpoint (``studies/{studyId}/samples``) from ``Eq``
        conjuncts of the predicate, which are then removed from it. A placeholder without an ``Eq`` stays
        unfilled (the request is not made)."""
        names = re.findall(r"\{([^{}]+)\}", endpoint)
        if not names or predicate is None:
            return endpoint, predicate
        parts = list(predicate.preds) if isinstance(predicate, And) else [predicate]
        rest = []
        for p in parts:
            if isinstance(p, Eq) and p.column in names and "{" + p.column + "}" in endpoint:
                endpoint = endpoint.replace("{" + p.column + "}", urllib.parse.quote(str(p.value), safe=""))
            else:
                rest.append(p)
        pred = None if not rest else (rest[0] if len(rest) == 1 else And(tuple(rest)))
        return endpoint, pred

    def _target(self, spec: LayoutSpec, predicate: Predicate | None, endpoint: str | None = None
                ) -> tuple[str | None, dict[str, str], Predicate | None]:
        """``(url, query parameters of the endpoint, predicate left)`` with path placeholders filled."""
        opts = self._options(spec)
        ep = str(endpoint if endpoint is not None else (opts.get("endpoint") or spec.path or ""))
        ep, predicate = self.fill_path(ep, predicate)
        if "{" in ep:
            missing = re.findall(r"\{([^{}]+)\}", ep)
            raise RemoteError(f"{spec.table}: the request needs {missing} fixed by the filter")
        path, _, query = ep.partition("?")
        return self._url(spec, path), dict(urllib.parse.parse_qsl(query)), predicate

    def _get(self, url: str, params: Mapping[str, Any], budget: Budget) -> tuple[Any, dict[str, str]]:
        budget.spend()
        base = urllib.parse.urlparse(url)
        _wait_turn(f"{base.scheme}://{base.netloc}", budget.requests_per_min)
        status, headers, body = self.transport(url, params, timeout=float(budget.timeout_s or DEFAULT_TIMEOUT_S))
        if status >= 400:
            text = body.decode("utf-8", "replace")[:500] if body else ""
            raise RemoteError(f"HTTP {status} from {url}: {text}", status=status, url=url)
        try:
            return json.loads(body.decode("utf-8") or "null"), headers
        except ValueError as exc:
            raise RemoteError(f"{url} did not return JSON: {exc}", status=status, url=url) from exc

    def request(self, spec: LayoutSpec, *, predicate: Predicate | None, projection: list[str],
                page_token: str | None, budget: Any, remote_names: Mapping[str, str] | None = None) -> Page:
        """One page of rows matching ``predicate`` (its residual is applied to the rows; with a residual
        the page's ``total`` is None: the source's total counts the pushed-down part only)."""
        b = Budget.of(budget)
        opts = self._options(spec)
        url, query, predicate = self._target(spec, predicate)
        if not url:
            raise RemoteError(f"{spec.table}: live_api needs options.base_url")
        params, residual = self.fmt.compile(predicate, self.request_map(spec, remote_names))
        params = {**dict(opts.get("params") or {}), **query, **params}
        size = int(b.page_size or opts.get("page_size") or DEFAULT_PAGE_SIZE)
        offset = 0
        if opts.get("page_size_param"):
            params[str(opts["page_size_param"])] = str(size)
        if page_token:
            if opts.get("page_token_param"):
                params[str(opts["page_token_param"])] = page_token
            elif opts.get("offset_param"):
                offset = int(page_token)
                params[str(opts["offset_param"])] = page_token
        payload, headers = self._get(url, params, b)
        page = self.fmt.decode_page(payload, opts, headers, offset=offset, page_size=size)
        rows = list(page.rows)
        if residual is not None:
            rows = [r for r in rows if evaluate(residual, r) is True]
        if projection:
            rows = [{k: v for k, v in r.items() if k in projection} if isinstance(r, Mapping) else r for r in rows]
        if page.as_of:
            _AS_OF[url] = page.as_of
        return Page(rows=rows, total=None if residual is not None else page.total, next=page.next, as_of=page.as_of)

    def count(self, spec: LayoutSpec, *, predicate: Predicate | None, budget: Any,
              remote_names: Mapping[str, str] | None = None) -> int | None:
        """An independent count request (the remote witness): None when the predicate does not compile
        completely or the source does not report a total."""
        opts = self._options(spec)
        count = dict(opts.get("count") or {})
        if not count:
            return None
        url, query, predicate = self._target(spec, predicate, count.get("endpoint"))
        params, residual = self.fmt.compile(predicate, self.request_map(spec, remote_names))
        if residual is not None or not url:
            return None
        params = {**dict(opts.get("params") or {}), **query, **params,
                  **{str(k): str(v) for k, v in (count.get("params") or {}).items()}}
        payload, headers = self._get(url, params, Budget.of(budget))
        if count.get("header"):
            hdr = {str(k).lower(): v for k, v in headers.items()}
            value = hdr.get(str(count["header"]).lower())
            return int(value) if value is not None and str(value).strip().isdigit() else None
        page = self.fmt.decode_page(payload, {**opts, "total_path": count.get("total_path") or opts.get("total_path"),
                                              "rows_path": None}, headers)
        if page.as_of:
            _AS_OF[self._url(spec) or url] = page.as_of
        return page.total

    def release(self, spec: LayoutSpec, budget: Any = None) -> str | None:
        """The source's data release (``options.release: {endpoint, path}``), e.g. CT.gov ``dataTimestamp``."""
        rel = dict(self._options(spec).get("release") or {})
        if not rel.get("path"):
            return None
        url = self._url(spec, rel.get("endpoint"))
        if not url:
            return None
        payload, _headers = self._get(url, dict(rel.get("params") or {}), Budget.of(budget))
        got = [v for v in jp_values(payload, str(rel["path"])) if v not in (None, "")]
        return str(got[0]) if got else None

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import LayoutCases

        # a live table has no files to damage or rename; test_dl_live_sources.py runs it against a stub
        return LayoutCases(tree="none", path=None, format="rest_json")


def fetch_all(layout: LiveApiLayout, spec: LayoutSpec, predicate: Predicate | None, budget: Any, *,
              projection: list[str] | None = None, remote_names: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every page within the budget's ``max_pages``: ``{rows, total, as_of, truncated, pages}``."""
    b = Budget.of(budget)
    rows: list[Any] = []
    token = None
    total = None
    as_of = None
    pages = 0
    while True:
        extra = {"remote_names": remote_names} if remote_names is not None else {}
        page = layout.request(spec, predicate=predicate, projection=list(projection or []), page_token=token,
                              budget=b, **extra)
        pages += 1
        rows.extend(page.rows)
        total = page.total if page.total is not None else total
        as_of = page.as_of or as_of
        token = page.next
        if not token:
            return {"rows": rows, "total": total, "as_of": as_of, "truncated": False, "pages": pages}
        if b.max_pages is not None and pages >= int(b.max_pages):
            return {"rows": rows, "total": total, "as_of": as_of, "truncated": True, "pages": pages}


# --------------------------------------------------------------------------- vocabularies and versions


@dataclass
class _Vocab:
    values: tuple[Any, ...]
    at: float


@dataclass
class RemoteVocab:
    """Remote vocabularies (``universe_via``, ``vocab: data`` of a remote table) kept for ``ttl_s`` seconds.
    A refresh compares the new values with the previous ones: ``drift`` lists what appeared and vanished
    (§13: a vocabulary that changed under a running session is a readiness warning, not silent)."""

    ttl_s: float = 3600.0
    clock: Callable[[], float] = time.time
    _cache: dict[str, _Vocab] = field(default_factory=dict)
    drift: dict[str, dict[str, list[Any]]] = field(default_factory=dict)

    def get(self, key: str, fetch: Callable[[], Iterable[Any]], *, ttl_s: float | None = None) -> tuple[Any, ...]:
        ttl = self.ttl_s if ttl_s is None else float(ttl_s)
        now = self.clock()
        hit = self._cache.get(key)
        if hit is not None and now - hit.at <= ttl:
            return hit.values
        values = tuple(sorted({v for v in fetch() if v is not None}, key=lambda v: (str(type(v)), str(v))))
        if hit is not None and set(hit.values) != set(values):
            self.drift[key] = {"added": sorted(set(values) - set(hit.values), key=str),
                               "removed": sorted(set(hit.values) - set(values), key=str)}
        self._cache[key] = _Vocab(values, now)
        return values

    def drifted(self, key: str) -> dict[str, list[Any]] | None:
        return self.drift.get(key)


class RecordVersions:
    """The last seen ``key.version`` of live records (e.g. CT.gov ``lastUpdatePostDate``), per table and key,
    stored as JSON (``<cache>/<source>/record_versions.json``). :meth:`observe` returns ``new``, ``same``
    or ``source_updated`` (the source changed the record since a previous call cited it)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, dict[str, str]] = {}
        if self.path is not None:
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.data = {}

    def observe(self, table: str, key: str, version: Any) -> str:
        if version in (None, ""):
            return "same"
        seen = self.data.setdefault(table, {})
        before = seen.get(str(key))
        seen[str(key)] = str(version)
        if before is None:
            return "new"
        return "same" if before == str(version) else "source_updated"

    def observe_rows(self, table: str, rows: Iterable[Mapping[str, Any]], key_path: str,
                     version_path: str) -> dict[str, Any]:
        """Observe every row; returns the provenance flags ``{source_updated: [keys], versions: {key: v}}``."""
        updated: list[str] = []
        versions: dict[str, str] = {}
        for row in rows:
            key = _dig(row, key_path)
            version = _dig(row, version_path)
            if key in (None, ""):
                continue
            if self.observe(table, str(key), version) == "source_updated":
                updated.append(str(key))
            if version not in (None, ""):
                versions[str(key)] = str(version)
        return {"source_updated": sorted(updated), "versions": versions}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def _dig(row: Mapping[str, Any], path: str) -> Any:
    cur: Any = row
    for part in str(path).split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur

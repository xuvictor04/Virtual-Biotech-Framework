"""The live step of ``vbt validate``: the remote sources the descriptors declare, from this host.

1. Endpoints: every ``base_url`` a remote source's table layouts name (``kind: remote`` descriptors) is asked
   once; any HTTP answer is reachable. Without one reachable endpoint the step is skipped (no network here).
2. Servers: the enabled servers whose bound tables live in a remote source start behind the gateway and make the
   doctor's smoke calls (``vbt doctor --smoke``: sentinel controls where a table declares sentinels, else one
   call per server), with their release and API versions recorded by the gateway.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .report import FAIL, PASS, StepResult

__all__ = ["run_live", "remote_endpoints"]

TITLE = "Live sources"


def remote_endpoints(catalog: Any) -> dict[str, list[str]]:
    """``{source: [base_url, ...]}`` of the remote descriptors' table layouts."""
    out: dict[str, list[str]] = {}
    for name, desc in sorted(catalog.sources.items()):
        if getattr(desc, "kind", "local") != "remote":
            continue
        urls: list[str] = []
        for spec in desc.tables.values():
            layout = spec.layout
            opts = getattr(layout, "options", None) if layout is not None and not isinstance(layout, str) else None
            url = (opts or {}).get("base_url") if isinstance(opts, dict) else None
            if url and url not in urls:
                urls.append(str(url))
        out[name] = urls
    return out


def _probe(url: str, timeout: float = 15.0) -> tuple[bool, str, float]:
    import urllib.error
    import urllib.request

    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "vbt-validate"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}", time.monotonic() - t0
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}", time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:120]}", time.monotonic() - t0


def _remote_servers(ctx: Any) -> list[str]:
    remote = {n for n, d in ctx.catalog.sources.items() if getattr(d, "kind", "local") == "remote"}
    out = []
    for server in ctx.servers():
        for tool in ctx.catalog.tools(server):
            try:
                tables = ctx.catalog.contract(server, tool).tables
            except Exception:  # noqa: BLE001
                continue
            if any(str(ref).split(".", 1)[0] in remote for ref in tables):
                out.append(server)
                break
    return out


def run_live(ctx: Any) -> StepResult:
    from ..preflight import smoke_mcp

    endpoints = remote_endpoints(ctx.catalog)
    if not endpoints:
        return StepResult.skipped("live", TITLE, "no remote source is declared")
    rows = []
    reachable = 0
    for source, urls in endpoints.items():
        for url in urls:
            ok, detail, seconds = _probe(url)
            reachable += ok
            rows.append({"what": f"{source} endpoint", "target": url, "ok": ok, "detail": detail,
                         "ms": round(1000 * seconds)})
    if endpoints and not reachable and any(endpoints.values()):
        return StepResult.skipped("live", TITLE, "no remote endpoint answers from this host (no network?): "
                                  + "; ".join(f"{r['target']}: {r['detail']}" for r in rows[:4]))
    servers = _remote_servers(ctx)
    t0 = time.monotonic()
    results = asyncio.run(smoke_mcp(ctx.config, servers=servers, mode="gateway")) if servers else []
    for r in results:
        rows.append({"what": "server", "target": r.label, "ok": r.ok, "detail": (r.detail or r.hint or "")[:300],
                     "ms": None, "required": r.required})
    failed = [r for r in rows if r["ok"] is False and r["what"] == "server" and r.get("required", True)]
    return StepResult("live", TITLE, FAIL if failed else PASS,
                      f"{reachable} of {sum(len(u) for u in endpoints.values())} endpoint(s) reachable; "
                      f"{len(results) - len(failed)} of {len(results)} server check(s) passed on "
                      f"{', '.join(servers) or 'no server'}", rows=rows, seconds=time.monotonic() - t0,
                      details={"endpoints": endpoints, "servers": servers})

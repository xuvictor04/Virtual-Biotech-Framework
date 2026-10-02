"""The ``vbt web`` server (fig. S1): a password-protected browser UI over CSO sessions.

Starlette + uvicorn (``pip install 'vbt-harness[web]'``). Everything the page
shows comes from the runtime event stream and the run's on-disk records, so the
server is provider-neutral.

Security model
--------------
* ``VBT_WEB_PASSWORD`` is required. ``POST /api/login`` compares it with
  ``hmac.compare_digest`` and sets an HMAC-signed, HttpOnly, SameSite=Strict
  session cookie; repeated failures are slowed down and then refused for a while.
  ``--no-auth`` is accepted only when binding a loopback address, and binding
  any other address without a password refuses to start.
* State-changing requests must be JSON (``Content-Type: application/json``),
  which, with the SameSite cookie, keeps other sites from driving a session.
* Browser sessions are bound to the login cookie that created them.
* Files are served only from ``runs_dir/<run_id>/``: the run id must be a plain
  directory name, the resolved path must stay inside it, and dot-files and
  dot-directories (``.env``, ``.downloads``) are refused. Served files carry a
  sandboxing Content-Security-Policy, and only raster images and plain text are
  shown inline, so an agent-written HTML or SVG file cannot run script with the
  UI's cookie. ``audit.html`` is served with its hash-pinned CSP.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hmac
import inspect
import ipaddress
import json
import logging
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("The web UI needs Starlette and uvicorn: pip install 'vbt-harness[web]' "
                      "(or: pip install starlette uvicorn)") from exc

from ..audit.index import runs_dir_from_config
from .sessions import BUSY_MESSAGE, SessionBusyError, SessionLimitError, WebSessionManager

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
COOKIE = "vbt_session"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}$")
MAX_PROMPT_CHARS = 100_000

#: In-code defaults for the ``web_ui`` config section.
WEB_UI_DEFAULTS: dict[str, Any] = {
    "max_sessions": 20,             # concurrent browser sessions
    "idle_timeout_s": 8 * 3600,     # close (and finalise) sessions idle this long
    "session_max_age_s": 12 * 3600, # login cookie lifetime
    "start_mcp": True,              # start MCP data servers for each session
    "event_buffer": 10000,          # events kept per session (for reconnects mid-turn)
    "login_max_failures": 10,       # per client, within login_lockout_s
    "login_lockout_s": 600,
}

INLINE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                   ".webp": "image/webp", ".pdf": "application/pdf"}
TEXT_SUFFIXES = (".txt", ".csv", ".tsv", ".md", ".json", ".jsonl", ".log", ".py", ".r", ".sh", ".yaml", ".yml")

FILE_CSP = "sandbox; default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
PAGE_CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class WebAuthError(RuntimeError):
    """The server would start without the required protection."""


# ----------------------------------------------------------------- binding / auth checks

def is_loopback(host: str) -> bool:
    h = str(host or "").strip().strip("[]")
    if h in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def check_bind(host: str, *, password: str | None, no_auth: bool) -> None:
    """Refuse configurations that would expose sessions without a password."""
    if no_auth:
        if not is_loopback(host):
            raise WebAuthError(f"--no-auth is only allowed on localhost; refusing to serve {host} without a "
                               "password. Set VBT_WEB_PASSWORD instead.")
        return
    if not password:
        if not is_loopback(host):
            raise WebAuthError(f"refusing to bind {host} without a password: set VBT_WEB_PASSWORD.")
        raise WebAuthError("VBT_WEB_PASSWORD is not set. Set it, or pass --no-auth to serve only on localhost "
                           "without a password.")


# ----------------------------------------------------------------- config helpers

def _web_ui(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**WEB_UI_DEFAULTS, **((config or {}).get("web_ui") or {})}


def available_profiles() -> list[str]:
    from ..config import CONFIG_DIR
    return sorted(p.stem for p in (CONFIG_DIR / "profiles").glob("*.yaml"))


def _roster(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = (config or {}).get("agents") or {}
    agents = []
    for name, d in (spec.get("agents") or {}).items():
        if not isinstance(d, Mapping) or d.get("disabled"):
            continue
        divs = [x.strip() for x in str(d.get("division") or "Other").split(";") if x.strip()]
        agents.append({"name": name, "division": divs[0] if divs else "Other", "divisions": divs,
                       "role": d.get("role") or "", "description": d.get("description") or "",
                       "tier": d.get("tier") or ""})
    cso = spec.get("cso") or {}
    divisions: dict[str, list[str]] = {}
    for a in agents:
        divisions.setdefault(a["division"], []).append(a["name"])
    return {"cso": {"name": "cso", "division": cso.get("division") or "Office of the CSO",
                    "description": cso.get("description") or ""},
            "agents": agents, "divisions": divisions}


def _examples() -> list[dict[str, Any]]:
    try:
        from ..case_studies.scenarios import list_scenarios
        out = []
        for sid, s in list_scenarios().items():
            turns = s.get("turns") or []
            if turns:
                out.append({"id": sid, "title": str(s.get("title") or sid),
                            "prompt": " ".join(str(turns[0]).split())})
        return out
    except Exception:  # noqa: BLE001 - examples are optional
        log.debug("scenario examples unavailable", exc_info=True)
        return []


def _model_pattern(config: Mapping[str, Any]) -> str | None:
    prov = (config or {}).get("provider") or {}
    pat = prov.get("model_pattern")
    if pat:
        return str(pat)
    return r"^claude-[a-z0-9.\-]+$" if prov.get("name") == "anthropic" else None


def resolve_model(config: Mapping[str, Any], value: str | None) -> str | None:
    """A model alias (``model_aliases``), a configured model id, or an id matching provider.model_pattern."""
    if not value:
        return None
    value = str(value).strip()
    aliases = (config or {}).get("model_aliases") or {}
    if value in aliases:
        target = aliases[value]
        return str(target.get("model") if isinstance(target, Mapping) else target)
    configured = {str((t or {}).get("model")) for t in ((config or {}).get("models") or {}).values()
                  if isinstance(t, Mapping)}
    if value in configured:
        return value
    pat = _model_pattern(config)
    if pat and re.fullmatch(pat, value):
        return value
    raise ValueError(f"unknown model {value!r}; choose one of: {', '.join(sorted(set(aliases) | configured))}")


def _apply_profile(base: dict[str, Any], profile: str) -> dict[str, Any]:
    from ..config import CONFIG_DIR, _expand, _load_yaml, deep_merge, resolve_path

    path = CONFIG_DIR / "profiles" / f"{profile}.yaml"
    raw = _load_yaml(path)
    variables = base.get("vars") or {}
    cfg = deep_merge(base, _expand(raw, variables))
    for key in ("agents_file", "mcp_servers_file"):
        if raw.get(key):
            cfg[key.replace("_file", "")] = _expand(_load_yaml(resolve_path(cfg[key])), variables)
    return cfg


# ----------------------------------------------------------------- the app

class WebApp:
    """Holds the server state; ``create_app`` wraps it in a Starlette application."""

    def __init__(self, config: dict[str, Any], *, provider: Any = None, password: str | None = None,
                 no_auth: bool = False, profiles: tuple[str, ...] | list[str] = (), start_mcp: bool | None = None,
                 max_sessions: int | None = None, idle_timeout_s: float | None = None, secure_cookie: bool = False):
        self.config = config
        self.ui = _web_ui(config)
        self.password = password if password is not None else (os.environ.get("VBT_WEB_PASSWORD") or None)
        self.no_auth = bool(no_auth)
        if not self.no_auth and not self.password:
            raise WebAuthError("VBT_WEB_PASSWORD is not set; the web UI never runs without a password "
                               "(use --no-auth only on localhost).")
        self.provider = provider
        self.base_profiles = [p for p in profiles or []]
        self.start_mcp = bool(self.ui["start_mcp"] if start_mcp is None else start_mcp)
        self.runs_dir = runs_dir_from_config(config)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.secret = secrets.token_bytes(32)
        self.secure_cookie = secure_cookie
        self.roster = _roster(config)
        divisions = {a["name"]: a["division"] for a in self.roster["agents"]}
        divisions["cso"] = self.roster["cso"]["division"]
        self.sessions = WebSessionManager(
            self._open_session, max_sessions=max_sessions or int(self.ui["max_sessions"]),
            idle_timeout_s=idle_timeout_s or float(self.ui["idle_timeout_s"]), divisions=divisions,
            event_buffer=int(self.ui["event_buffer"]))
        self._failures: dict[str, list[float]] = {}
        self._examples: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------ sessions

    async def _open_session(self, cfg: dict[str, Any], on_event) -> Any:
        from ..orchestrator import open_session
        from ..providers.base import LLMProvider

        kw: dict[str, Any] = {"on_event": on_event, "start_mcp": self.start_mcp}
        prov = self.provider
        if prov is not None and not isinstance(prov, LLMProvider) and callable(prov):
            prov = prov()
        if prov is not None:
            kw["provider"] = prov
        params = inspect.signature(open_session).parameters
        if "interface" in params:
            kw["interface"] = "web"
        if "profiles" in params and self.base_profiles:
            kw["profiles"] = tuple(self.base_profiles)
        return await open_session(cfg, **kw)

    def session_config(self, profile: str | None, model: str | None) -> tuple[dict[str, Any], str | None]:
        cfg = self.config
        if profile and profile not in self.base_profiles:
            if profile not in available_profiles():
                raise ValueError(f"unknown profile {profile!r}")
            cfg = _apply_profile(cfg, profile)
        else:
            cfg = copy.deepcopy(cfg)
        cfg.setdefault("paths", {})["runs_dir"] = str(self.runs_dir)
        resolved = resolve_model(cfg, model) if model else None
        if resolved:
            for tier in ("orchestrator", "scientist", "bulk"):
                if tier in cfg.get("models", {}):
                    cfg["models"][tier] = {**cfg["models"][tier], "model": resolved}
        return cfg, resolved

    # ------------------------------------------------------------ auth

    def _sign(self, payload: str) -> str:
        return hmac.new(self.secret, payload.encode(), "sha256").hexdigest()

    def make_cookie(self) -> tuple[str, str]:
        owner = secrets.token_urlsafe(18)
        expiry = int(time.time() + float(self.ui["session_max_age_s"]))
        payload = f"{owner}.{expiry}"
        return owner, f"{payload}.{self._sign(payload)}"

    def owner(self, request: Request) -> str | None:
        if self.no_auth:
            return "local"
        raw = request.cookies.get(COOKIE) or ""
        parts = raw.split(".")
        if len(parts) != 3:
            return None
        owner, expiry, sig = parts
        if not hmac.compare_digest(sig, self._sign(f"{owner}.{expiry}")):
            return None
        try:
            if int(expiry) < time.time():
                return None
        except ValueError:
            return None
        return owner

    def _client(self, request: Request) -> str:
        return request.client.host if request.client else "?"

    def _locked_out(self, client: str) -> bool:
        now = time.time()
        window = float(self.ui["login_lockout_s"])
        recent = [t for t in self._failures.get(client, []) if now - t < window]
        self._failures[client] = recent
        return len(recent) >= int(self.ui["login_max_failures"])

    def check_password(self, given: Any) -> bool:
        if not isinstance(given, str) or not self.password:
            return False
        return hmac.compare_digest(given.encode("utf-8"), self.password.encode("utf-8"))

    # ------------------------------------------------------------ paths

    def run_path(self, run_id: str) -> Path | None:
        if not RUN_ID_RE.match(run_id or "") or run_id in (".", ".."):
            return None
        root = self.runs_dir.resolve()
        d = (root / run_id).resolve()
        if d == root or root not in d.parents or not d.is_dir():
            return None
        return d

    def examples(self) -> list[dict[str, Any]]:
        if self._examples is None:
            self._examples = _examples()
        return self._examples


# ----------------------------------------------------------------- request helpers

def _err(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message, **extra}, status_code=status)


async def _json_body(request: Request, *, limit: int = 1_000_000) -> dict[str, Any] | JSONResponse:
    ctype = request.headers.get("content-type", "")
    if "application/json" not in ctype:
        return _err(415, "send application/json")
    body = await request.body()
    if len(body) > limit:
        return _err(413, "request too large")
    if not body:
        return {}
    try:
        data = json.loads(body)
    except ValueError:
        return _err(400, "invalid JSON")
    return data if isinstance(data, dict) else _err(400, "expected a JSON object")


def _sse(ev: Mapping[str, Any]) -> str:
    data = json.dumps(ev, default=str, ensure_ascii=False)
    return f"id: {ev['id']}\ndata: {data}\n\n"


def _file_url(run_id: str, rel: str) -> str:
    return f"/runs/{quote(run_id)}/files/" + quote(rel)


def create_app(config: dict[str, Any], *, provider: Any = None, password: str | None = None, no_auth: bool = False,
               profiles: tuple[str, ...] | list[str] = (), start_mcp: bool | None = None,
               max_sessions: int | None = None, idle_timeout_s: float | None = None,
               secure_cookie: bool = False) -> Starlette:
    """Build the ASGI app. ``provider`` is an LLMProvider instance or a zero-argument factory
    (default: the configured provider, one per session). ``password`` defaults to
    ``VBT_WEB_PASSWORD``; without one the app refuses to build unless ``no_auth``."""
    app_state = WebApp(config, provider=provider, password=password, no_auth=no_auth, profiles=profiles,
                       start_mcp=start_mcp, max_sessions=max_sessions, idle_timeout_s=idle_timeout_s,
                       secure_cookie=secure_cookie)
    W = app_state

    def authed(handler):
        async def wrapped(request: Request):
            owner = W.owner(request)
            if owner is None:
                return _err(401, "login required")
            request.state.owner = owner
            W.sessions.start()
            return await handler(request)
        wrapped.__name__ = handler.__name__
        return wrapped

    def session_of(request: Request):
        try:
            return W.sessions.get(request.path_params["sid"], request.state.owner)
        except KeyError:
            return None

    # ------------------------------------------------------------ pages

    async def index(request: Request):
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html",
                            headers={"Content-Security-Policy": PAGE_CSP, "Cache-Control": "no-store",
                                     "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    # ------------------------------------------------------------ auth

    async def auth_status(request: Request):
        return JSONResponse({"auth_required": not W.no_auth, "authenticated": W.owner(request) is not None})

    async def login(request: Request):
        if W.no_auth:
            return JSONResponse({"ok": True, "auth_required": False})
        client = W._client(request)
        if W._locked_out(client):
            return _err(429, "too many failed logins; try again later")
        data = await _json_body(request, limit=10_000)
        if isinstance(data, JSONResponse):
            return data
        if not W.check_password(data.get("password")):
            W._failures.setdefault(client, []).append(time.time())
            await asyncio.sleep(min(0.25 * len(W._failures[client]), 2.0))
            return _err(401, "wrong password")
        W._failures.pop(client, None)
        _owner, cookie = W.make_cookie()
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, cookie, max_age=int(W.ui["session_max_age_s"]), httponly=True, samesite="strict",
                        secure=W.secure_cookie or request.url.scheme == "https", path="/")
        return resp

    async def logout(request: Request):
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    # ------------------------------------------------------------ config and runs

    @authed
    async def api_config(request: Request):
        cfg = W.config
        models = {t: (v or {}).get("model") for t, v in (cfg.get("models") or {}).items() if isinstance(v, Mapping)}
        aliases = {k: (v.get("model") if isinstance(v, Mapping) else v)
                   for k, v in (cfg.get("model_aliases") or {}).items()}
        return JSONResponse({
            "auth_required": not W.no_auth, "provider": (cfg.get("provider") or {}).get("name"),
            "profiles": available_profiles(), "base_profiles": W.base_profiles, "models": models,
            "model_aliases": aliases, "examples": W.examples(), "roster": W.roster,
            "max_sessions": W.sessions.max_sessions, "web_search": bool((cfg.get("web") or {}).get("enabled", True)),
        })

    @authed
    async def api_runs(request: Request):
        from ..audit.index import scan_runs
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except ValueError:
            limit = 100
        rows = (await asyncio.to_thread(scan_runs, W.runs_dir))[:limit]
        for r in rows:
            rid = r.get("dir") or r["run_id"]
            r.pop("path", None)
            r["audit_url"] = f"/runs/{quote(rid)}/audit.html" if r.get("has_manifest") else None
            r["zip_url"] = f"/runs/{quote(rid)}/download.zip"
            r["chat_url"] = f"/runs/{quote(rid)}/chat.md"
        return JSONResponse({"runs": rows})

    # ------------------------------------------------------------ sessions

    @authed
    async def create_session(request: Request):
        data = await _json_body(request)
        if isinstance(data, JSONResponse):
            return data
        profile = data.get("profile") or None
        model = data.get("model") or None
        try:
            cfg, resolved = W.session_config(profile, model)
        except ValueError as exc:
            return _err(400, str(exc))
        try:
            ws = W.sessions.create(request.state.owner, cfg, profile=profile, model=resolved or model)
        except SessionLimitError as exc:
            return _err(503, str(exc))
        return JSONResponse({"ok": True, **ws.status()}, status_code=201)

    def _turn_rows(ws) -> list[dict[str, Any]]:
        from ..audit.claims import load_claims
        from ..audit.render import number_refs
        run = getattr(ws.cso, "run", None)
        if run is None:
            return []
        claims = load_claims(run.dir)
        rows = []
        for t in run.turns:
            rendered, footnotes = number_refs(str(t.get("response") or ""), claims)
            rows.append({"turn": t.get("turn"), "prompt": t.get("prompt"), "status": t.get("status"),
                         "rendered": rendered, "footnotes": footnotes, "cost_usd": t.get("cost_usd"),
                         "claims_filed": t.get("claims_filed") or [],
                         "claims_unresolved": t.get("claims_unresolved") or []})
        return rows

    @authed
    async def get_session(request: Request):
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        st = ws.status()
        resume = ws.events.last_id
        if ws.busy:
            uid = ws.events.first_id_of_last("user")
            resume = (uid - 1) if uid else 0
        return JSONResponse({"ok": True, **st, "resume_from": resume, "turn_records": _turn_rows(ws)})

    @authed
    async def delete_session(request: Request):
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        await W.sessions.close(ws.id)
        return JSONResponse({"ok": True})

    @authed
    async def ask(request: Request):
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        data = await _json_body(request, limit=4 * MAX_PROMPT_CHARS)
        if isinstance(data, JSONResponse):
            return data
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            return _err(400, "empty prompt")
        if len(prompt) > MAX_PROMPT_CHARS:
            return _err(413, f"prompt longer than {MAX_PROMPT_CHARS} characters")
        try:
            n = ws.ask(prompt)
        except SessionBusyError:
            return _err(409, BUSY_MESSAGE)
        return JSONResponse({"ok": True, "n": n}, status_code=202)

    @authed
    async def stop(request: Request):
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        if not request.headers.get("content-type", "").startswith("application/json"):
            return _err(415, "send application/json")
        return JSONResponse(await ws.stop())

    @authed
    async def events(request: Request):
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        q = request.query_params
        try:
            after = int(request.headers.get("last-event-id") or q.get("after") or 0)
        except ValueError:
            after = 0
        until = {k for k in (q.get("until") or "").split(",") if k}
        try:
            timeout = float(q["timeout"]) if q.get("timeout") else None
        except ValueError:
            timeout = None
        deadline = time.monotonic() + timeout if timeout else None

        async def gen():
            last = after
            yield ": connected\n\n"
            while True:
                wait_s = 15.0
                if deadline is not None:
                    wait_s = min(wait_s, max(0.0, deadline - time.monotonic()))
                evs = await ws.events.wait(last, wait_s)
                for ev in evs:
                    last = ev["id"]
                    yield _sse(ev)
                    if ev["kind"] in until:
                        return
                if ws.events.closed and not ws.events.since(last):
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if not evs:
                    yield ": keepalive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @authed
    async def claims(request: Request):
        from ..audit.claims import claim_stats, load_claims
        from ..audit.render import EVIDENCE_STATUS_LABELS, describe_evidence, evidence_status
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        run_dir = ws.run_dir
        if run_dir is None:
            return JSONResponse({"run_id": None, "claims": [], "stats": claim_stats([])})
        items = await asyncio.to_thread(load_claims, run_dir)
        out = []
        for c in items:
            evs = []
            for ev in c.get("evidence") or []:
                if not isinstance(ev, Mapping):
                    continue
                st = evidence_status(ev)
                e = {"kind": ev.get("kind"), "status": st, "label": EVIDENCE_STATUS_LABELS[st],
                     "text": describe_evidence(ev), "note": ev.get("note"), "sha256": ev.get("sha256")}
                if ev.get("path") and W.run_path(run_dir.name) is not None:
                    e["url"] = _file_url(run_dir.name, str(ev["path"]))
                evs.append(e)
            out.append({"id": c.get("id"), "text": c.get("text"), "agent": c.get("agent"),
                        "confidence": c.get("confidence"), "turn": c.get("turn"),
                        "n_verified": c.get("n_verified") or 0, "evidence": evs})
        return JSONResponse({"run_id": ws.run_id, "claims": out, "stats": claim_stats(items)})

    @authed
    async def files(request: Request):
        from ..audit.export import list_session_files
        ws = session_of(request)
        if ws is None:
            return _err(404, "no such session")
        run_dir = ws.run_dir
        if run_dir is None:
            return JSONResponse({"run_id": None, "files": {}, "counts": {}})
        listing = await asyncio.to_thread(list_session_files, run_dir)
        counts = listing.pop("counts", {})
        for group in listing.values():
            for rows in group.values():
                for r in rows:
                    r["url"] = _file_url(run_dir.name, r["path"])
        return JSONResponse({"run_id": ws.run_id, "files": listing, "counts": counts})

    # ------------------------------------------------------------ run downloads

    def _run_or_404(request: Request) -> Path | Response:
        d = W.run_path(request.path_params["run_id"])
        return d if d is not None else _err(404, "no such run")

    @authed
    async def download_zip(request: Request):
        from ..audit.export import ExportError, export_run
        d = _run_or_404(request)
        if isinstance(d, Response):
            return d
        no_data = request.query_params.get("no_data") in ("1", "true", "yes")
        try:
            path = await asyncio.to_thread(export_run, d, None, no_data=no_data)
        except ExportError as exc:
            return _err(400, str(exc))
        return FileResponse(path, media_type="application/zip", filename=f"{d.name}{'-nodata' if no_data else ''}.zip",
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @authed
    async def chat_md(request: Request):
        from ..audit.export import export_chat_markdown
        d = _run_or_404(request)
        if isinstance(d, Response):
            return d
        text = await asyncio.to_thread(export_chat_markdown, d)
        return Response(text, media_type="text/markdown; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="{d.name}.chat.md"', "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store"})

    @authed
    async def audit_html(request: Request):
        from ..audit.report import AUDIT_CSP, ensure_reports
        d = _run_or_404(request)
        if isinstance(d, Response):
            return d
        if not (d / "MANIFEST.json").is_file() and not (d / "audit.html").is_file():
            return _err(404, "this run has no audit record")
        paths = await asyncio.to_thread(ensure_reports, d)
        return FileResponse(paths["html"], media_type="text/html; charset=utf-8", headers={
            "Content-Security-Policy": (AUDIT_CSP + "; sandbox allow-scripts allow-popups "
                                        "allow-popups-to-escape-sandbox"),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

    @authed
    async def run_file(request: Request):
        d = _run_or_404(request)
        if isinstance(d, Response):
            return d
        rel = str(request.path_params.get("path") or "")
        parts = [p for p in rel.replace("\\", "/").split("/")]
        if not rel or any(p in ("", ".", "..") or p.startswith(".") for p in parts):
            return _err(403, "path not allowed")
        target = (d / "/".join(parts)).resolve()
        if target == d or d not in target.parents:
            return _err(403, "path not allowed")
        if not target.is_file():
            return _err(404, "no such file")
        suffix = target.suffix.lower()
        headers = {"Content-Security-Policy": FILE_CSP, "X-Content-Type-Options": "nosniff",
                   "Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
        if suffix in INLINE_SUFFIXES:
            return FileResponse(target, media_type=INLINE_SUFFIXES[suffix], headers=headers)
        if suffix == ".svg":
            # Renders in <img> (which never runs SVG script); opened directly it downloads.
            return FileResponse(target, media_type="image/svg+xml", filename=target.name, headers=headers)
        if suffix in TEXT_SUFFIXES and request.query_params.get("download") not in ("1", "true"):
            return FileResponse(target, media_type="text/plain; charset=utf-8", headers=headers)
        return FileResponse(target, media_type="application/octet-stream", filename=target.name, headers=headers)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        W.sessions.start()
        try:
            yield
        finally:
            await W.sessions.aclose()

    routes = [
        Route("/", index),
        Route("/api/auth", auth_status),
        Route("/api/login", login, methods=["POST"]),
        Route("/api/logout", logout, methods=["POST"]),
        Route("/api/config", api_config),
        Route("/api/runs", api_runs),
        Route("/api/sessions", create_session, methods=["POST"]),
        Route("/api/sessions/{sid}", get_session),
        Route("/api/sessions/{sid}", delete_session, methods=["DELETE"]),
        Route("/api/sessions/{sid}/ask", ask, methods=["POST"]),
        Route("/api/sessions/{sid}/stop", stop, methods=["POST"]),
        Route("/api/sessions/{sid}/events", events),
        Route("/api/sessions/{sid}/claims", claims),
        Route("/api/sessions/{sid}/files", files),
        Route("/runs/{run_id}/download.zip", download_zip),
        Route("/runs/{run_id}/chat.md", chat_md),
        Route("/runs/{run_id}/audit.html", audit_html),
        Route("/runs/{run_id}/files/{path:path}", run_file),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.web = W
    app.state.sessions = W.sessions
    return app


# ----------------------------------------------------------------- CLI

def serve(config: dict[str, Any], *, host: str = "127.0.0.1", port: int = 7860, no_auth: bool = False,
          profiles: tuple[str, ...] | list[str] = (), start_mcp: bool | None = None) -> int:
    password = os.environ.get("VBT_WEB_PASSWORD") or None
    try:
        check_bind(host, password=password, no_auth=no_auth)
    except WebAuthError as exc:
        print(f"error: {exc}")
        return 2
    try:
        import uvicorn
    except ImportError:
        print("error: the web UI needs uvicorn: pip install 'vbt-harness[web]'")
        return 2
    app = create_app(config, password=password, no_auth=no_auth, profiles=profiles, start_mcp=start_mcp)
    shown = f"[{host}]" if ":" in host else host
    print(f"The Virtual Biotech web UI: http://{shown}:{port}/" + ("  (no password: localhost only)" if no_auth
                                                                  else "  (password: VBT_WEB_PASSWORD)"))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


__all__ = ["create_app", "serve", "check_bind", "is_loopback", "resolve_model", "WebAuthError", "WebApp",
           "WEB_UI_DEFAULTS", "available_profiles"]

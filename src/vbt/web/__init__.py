"""``vbt web``: the browser interface (paper fig. S1).

Importing this package needs nothing beyond the core install; the server
(``vbt.web.server``) imports Starlette and uvicorn lazily and names the
``web`` extra when they are missing.

    vbt web [--host 127.0.0.1] [--port 7860] [--no-auth]

``VBT_WEB_PASSWORD`` must be set; ``--no-auth`` is accepted only on localhost.
"""

from __future__ import annotations

from typing import Any


def create_app(config: dict[str, Any], **kwargs: Any):
    """Build the ASGI application (see ``vbt.web.server.create_app``)."""
    from .server import create_app as _create_app
    return _create_app(config, **kwargs)


def _web_handler(args, config: dict[str, Any]) -> int:
    try:
        from .server import serve
    except ImportError as exc:
        print(f"error: {exc}")
        return 2
    return serve(config, host=args.host, port=args.port, no_auth=args.no_auth,
                 profiles=list(getattr(args, "profile", None) or []),
                 start_mcp=False if getattr(args, "no_mcp", False) else None)


def add_web_parser(sub) -> None:
    """Register ``vbt web`` on the main parser's subcommands (handler(args, config))."""
    w = sub.add_parser("web", help="browser UI: streamed CSO answers, live agent/tool activity, claims, downloads",
                       description="Serve the web UI. Requires VBT_WEB_PASSWORD unless --no-auth (localhost only).")
    w.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    w.add_argument("--port", type=int, default=7860)
    w.add_argument("--no-auth", action="store_true", help="no password; only allowed when --host is a loopback address")
    w.set_defaults(handler=_web_handler)


__all__ = ["create_app", "add_web_parser"]

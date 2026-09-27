"""Bridge MCP servers (stdio or HTTP) into the harness tool registry.

Each server's tools are exposed as ``mcp__<server>__<tool>`` -- the naming the
upstream prompts use -- with their MCP input schemas passed through, so any
provider can call them. Servers are any MCP implementation (the upstream
FastMCP servers, your own, or third-party ones), configured in YAML.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from .base import Tool, ToolContext, ToolFailure, inline_refs

log = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None  # streamable-HTTP servers
    enabled: bool = True
    timeout_s: float = 600.0
    inherit_env: bool = True


class MCPBridge:
    def __init__(self, servers: list[MCPServerConfig], *, extra_env: dict[str, str] | None = None):
        self.servers = [s for s in servers if s.enabled]
        self.extra_env = extra_env or {}
        self._stack = AsyncExitStack()
        self.sessions: dict[str, Any] = {}
        self.failures: dict[str, str] = {}
        self.tools: list[Tool] = []

    async def start(self, only: set[str] | None = None, connect_timeout: float = 120.0) -> list[Tool]:
        wanted = [s for s in self.servers if only is None or s.name in only]
        await asyncio.gather(*(self._start_one(s, connect_timeout) for s in wanted))
        return self.tools

    async def _start_one(self, cfg: MCPServerConfig, timeout: float) -> None:
        try:
            session = await asyncio.wait_for(self._connect(cfg), timeout)
            listed = await session.list_tools()
        except Exception as exc:  # noqa: BLE001 - report and continue without this server
            self.failures[cfg.name] = f"{type(exc).__name__}: {exc}"
            log.warning("MCP server %s failed to start: %s", cfg.name, exc)
            return
        self.sessions[cfg.name] = session
        for t in listed.tools:
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object"}
            self.tools.append(Tool(
                name=f"mcp__{cfg.name}__{t.name}",
                description=(t.description or t.name).strip(),
                input_schema=inline_refs(dict(schema)),
                handler=self._make_handler(cfg, t.name),
                source=f"mcp:{cfg.name}",
            ))

    async def _connect(self, cfg: MCPServerConfig):
        from mcp import ClientSession

        if cfg.url:
            try:
                from mcp.client.streamable_http import streamable_http_client as http_client
            except ImportError:  # mcp 1.x
                from mcp.client.streamable_http import streamablehttp_client as http_client  # type: ignore

            streams = await self._stack.enter_async_context(http_client(cfg.url))
            read, write = streams[0], streams[1]
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            env = {**(os.environ if cfg.inherit_env else {}), **self.extra_env, **cfg.env}
            params = StdioServerParameters(command=cfg.command, args=cfg.args, env=env, cwd=cfg.cwd)
            errlog = open(os.devnull, "w") if not log.isEnabledFor(logging.DEBUG) else None
            if errlog is not None:
                self._stack.callback(errlog.close)
                read, write = await self._stack.enter_async_context(stdio_client(params, errlog=errlog))
            else:
                read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def _make_handler(self, cfg: MCPServerConfig, tool_name: str):
        async def handler(ctx: ToolContext, args: dict[str, Any]) -> str:
            session = self.sessions[cfg.name]
            result = await asyncio.wait_for(session.call_tool(tool_name, args), cfg.timeout_s)
            parts = []
            for c in getattr(result, "content", []) or []:
                parts.append(getattr(c, "text", None) or f"[{getattr(c, 'type', 'content')} omitted]")
            text = "\n".join(parts)
            if not text and getattr(result, "structured_content", None) is not None:
                text = str(result.structured_content)
            if getattr(result, "is_error", False) or getattr(result, "isError", False):
                raise ToolFailure(text or "MCP tool reported an error")
            return text
        return handler

    async def aclose(self) -> None:
        try:
            await self._stack.aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown noise from child processes
            log.debug("MCP shutdown: %s", exc)

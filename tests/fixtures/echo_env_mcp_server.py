#!/usr/bin/env python3
"""Test fixture: a small stdio FastMCP server for tests/test_mcp_bridge.py.

Tools cover the bridge's environment policy (echo_env), error semantics
(raised errors, legacy envelopes, empty lookups), timeouts (slow), image
content and stderr capture. No network, no data.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

print("echo-env fixture server starting", file=sys.stderr, flush=True)

mcp = FastMCP("echo_env")

# 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


@mcp.tool()
def echo_env(names: list[str]) -> dict:
    """Return the values of the named environment variables and all variable names."""
    return {"values": {n: os.environ.get(n) for n in names}, "all_names": sorted(os.environ)}


@mcp.tool()
def raise_error(message: str = "boom") -> dict:
    """Fail with an MCP tool error (isError=true)."""
    raise ToolError(message)


@mcp.tool()
def legacy_error() -> dict:
    """Return a legacy failure envelope (transport-level success)."""
    return {"error": "Open Targets data could not be loaded", "results": []}


@mcp.tool()
def legacy_unsuccessful() -> dict:
    """Return a {success: false} envelope with no error text."""
    return {"success": False, "results": []}


@mcp.tool()
def empty_lookup(symbol: str = "NOTAGENE") -> dict:
    """An explicit empty lookup in the upstream servers' wording (not a failure)."""
    return {"error": f"Target {symbol} not found", "results": []}


@mcp.tool()
def ok_result() -> dict:
    """A successful result whose rows contain an 'error' column."""
    return {"results": [{"gene": "EGFR", "error": 0.1}], "count": 1}


@mcp.tool()
async def slow(seconds: float = 5.0) -> str:
    """Sleep, then answer."""
    await asyncio.sleep(seconds)
    return f"slept {seconds}s"


@mcp.tool()
def image():
    """Return a tiny PNG image."""
    from fastmcp.utilities.types import Image
    return Image(data=_PNG, format="png")


@mcp.tool()
def say(text: str) -> str:
    """Echo text, and write a line to stderr."""
    print(f"say called with {text!r}", file=sys.stderr, flush=True)
    return text


if __name__ == "__main__":
    mcp.run()

#!/usr/bin/env python3
"""Test fixture: a stdio FastMCP server behind the data gateway seam (test_dl_bridge_seam.py).

Usage: gateway_fixture_server.py STATE_DIR [--variant TEXT]

* ``lookup(target_id)`` returns a record for known IDs and the upstream servers' explicit
  ``{"error": "Target <id> not found"}`` envelope for unknown ones (an empty lookup).
* ``rows(limit)`` returns the first ``limit`` rows in fixed file order, with a total.
* ``legacy_error`` returns a legacy failure envelope; ``raise_error`` fails with isError.
* ``crash_once`` exits hard the first time per STATE_DIR; ``crash_always`` on every call.
* ``slow(seconds)`` sleeps; ``ping`` answers with the process start number.
* ``_stats(request)`` is an internal verb (hidden from agents when the server is ``data``).
* The description of ``lookup`` names the variant: ``--variant`` or the text of
  STATE_DIR/variant (read at start, so a restart picks up a change). With a variant file,
  the server also lists ``late_tool``.

Each start increments STATE_DIR/starts; every call is appended to STATE_DIR/calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

state = Path(sys.argv[1])
state.mkdir(parents=True, exist_ok=True)
counter = state / "starts"
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
variant_file = state / "variant"
variant = variant_file.read_text().strip() if variant_file.exists() else None
if variant is None and "--variant" in sys.argv:
    variant = sys.argv[sys.argv.index("--variant") + 1]
print(f"gateway fixture server start #{n} (variant {variant})", file=sys.stderr, flush=True)

from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402

mcp = FastMCP("gateway_fixture")

TARGETS = {"ENSG00000169174": "PCSK9", "ENSG00000141510": "TP53"}
FILE_ROWS = [{"id": f"r{i:02d}", "score": round(1.0 - i * 0.07, 2)} for i in range(12)]


def _log_call(name: str, args: dict) -> None:
    with open(state / "calls", "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": name, "args": args, "start": n}) + "\n")


@mcp.tool(description=f"Look up a target by Ensembl gene ID (variant {variant or 'base'}).")
def lookup(target_id: str) -> dict:
    _log_call("lookup", {"target_id": target_id})
    if target_id not in TARGETS:
        return {"error": f"Target {target_id} not found"}
    return {"id": target_id, "approvedSymbol": TARGETS[target_id]}


@mcp.tool()
def rows(limit: int = 5) -> dict:
    """Return the first `limit` rows in file order."""
    _log_call("rows", {"limit": limit})
    out = FILE_ROWS[:limit]
    return {"success": True, "count": len(out), "total": len(FILE_ROWS), "rows": out}


@mcp.tool()
def empty() -> dict:
    """A structurally empty success."""
    _log_call("empty", {})
    return {"success": True, "count": 0, "results": []}


@mcp.tool()
def legacy_error() -> dict:
    """Return a legacy failure envelope."""
    _log_call("legacy_error", {})
    return {"error": "Open Targets data could not be loaded"}


@mcp.tool()
def raise_error(message: str = "boom") -> dict:
    """Fail with an MCP tool error (isError=true)."""
    _log_call("raise_error", {"message": message})
    raise ToolError(message)


@mcp.tool()
def crash_once() -> str:
    """Crash the first time (per state dir); succeed after a restart."""
    _log_call("crash_once", {})
    marker = state / "crashed"
    if not marker.exists():
        marker.write_text("1")
        print("crash_once: exiting hard", file=sys.stderr, flush=True)
        os._exit(1)
    return f"survived after restart (start #{n})"


@mcp.tool()
def crash_always() -> str:
    """Crash on every call."""
    _log_call("crash_always", {})
    os._exit(1)


@mcp.tool()
async def slow(seconds: float = 0.5) -> str:
    """Sleep, then answer."""
    _log_call("slow", {"seconds": seconds})
    await asyncio.sleep(seconds)
    return f"slept {seconds}s on start #{n}"


@mcp.tool()
def ping() -> str:
    """Answer with the process start number."""
    return f"pong from start #{n}"


@mcp.tool(name="_stats")
def stats(request: str = "{}") -> dict:
    """Internal verb: echo the request (stands in for the data child's _stats)."""
    _log_call("_stats", {"request": request})
    return {"tables": {}, "echo": json.loads(request or "{}")}


if variant_file.exists():
    @mcp.tool()
    def late_tool() -> str:
        """A tool that only exists in the variant."""
        return f"late tool on start #{n}"


if __name__ == "__main__":
    mcp.run()

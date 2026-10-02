#!/usr/bin/env python3
"""Test fixture: a stdio FastMCP server that crashes or fails to start on demand.

Usage: crashing_mcp_server.py STATE_DIR [--fail-starts N]

* Each process start increments STATE_DIR/starts. With ``--fail-starts N`` the
  first N starts exit immediately with a message on stderr (startup retry tests).
* ``crash_once`` kills the process (``os._exit``) the first time it is called
  in STATE_DIR and succeeds afterwards (crash-restart-retry tests).
* ``crash_always`` kills the process on every call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

state = Path(sys.argv[1])
state.mkdir(parents=True, exist_ok=True)
fail_starts = int(sys.argv[sys.argv.index("--fail-starts") + 1]) if "--fail-starts" in sys.argv else 0
counter = state / "starts"
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
if n <= fail_starts:
    print(f"simulated startup failure #{n}: reference data missing", file=sys.stderr, flush=True)
    sys.exit(3)
print(f"crashing fixture server start #{n}", file=sys.stderr, flush=True)

from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("crashing")


@mcp.tool()
def ping() -> str:
    """Answer with the process start number."""
    return f"pong from start #{n}"


@mcp.tool()
def crash_once() -> str:
    """Crash the first time (per state dir); succeed after a restart."""
    marker = state / "crashed"
    if not marker.exists():
        marker.write_text("1")
        print("crash_once: exiting hard", file=sys.stderr, flush=True)
        os._exit(1)
    return f"survived after restart (start #{n})"


@mcp.tool()
def crash_always() -> str:
    """Crash on every call."""
    os._exit(1)


if __name__ == "__main__":
    mcp.run()

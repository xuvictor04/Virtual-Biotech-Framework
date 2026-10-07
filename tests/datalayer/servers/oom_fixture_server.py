#!/usr/bin/env python3
"""Test fixture: a stdio FastMCP server that runs out of memory on demand (test_dl_launcher.py).

Usage: oom_fixture_server.py STATE_DIR

* ``allocate(mb)`` allocates and touches ``mb`` MiB. Under the launcher's RLIMIT_DATA the
  allocation fails inside the tool, which reports it the way numpy does ("Unable to allocate
  ..."), so FastMCP answers isError and the process survives.
* ``sigkill_self`` kills the process with SIGKILL (what the kernel OOM killer does).
* ``ping`` answers with the process start number and the RLIMIT_DATA soft limit.

Each start increments STATE_DIR/starts.
"""

from __future__ import annotations

import os
import resource
import signal
import sys
from pathlib import Path

state = Path(sys.argv[1])
state.mkdir(parents=True, exist_ok=True)
counter = state / "starts"
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
print(f"oom fixture server start #{n}", file=sys.stderr, flush=True)

from fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("oom_fixture")


@mcp.tool()
def allocate(mb: int) -> str:
    """Allocate and touch `mb` MiB."""
    try:
        block = bytearray(mb * 1024 * 1024)
    except MemoryError:
        raise MemoryError(f"Unable to allocate {mb} MiB for a bytearray") from None
    block[::4096] = b"\x01" * len(block[::4096])
    return f"allocated {len(block) // (1024 * 1024)} MiB"


@mcp.tool()
def sigkill_self() -> str:
    """Die by SIGKILL."""
    print("sigkill_self: killing myself", file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
    return "unreachable"


@mcp.tool()
def ping() -> dict:
    """Answer with the start number and the data-segment limit."""
    soft, _ = resource.getrlimit(resource.RLIMIT_DATA)
    return {"start": n, "rlimit_data": None if soft == resource.RLIM_INFINITY else soft, "pid": os.getpid()}


if __name__ == "__main__":
    mcp.run()

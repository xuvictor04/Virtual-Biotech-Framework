"""Run the unmodified upstream clinicaltrials MCP server with the network-free pybioportal stub.

Usage: ``python -B -E clinicaltrials_stubbed_server.py`` (``VBT_UPSTREAM`` names the upstream
checkout; default ``third_party/TheVirtualBiotech`` of this repository). This directory goes
first on ``sys.path`` so ``from pybioportal import ...`` in the upstream tools resolves to
``stubs/pybioportal``; the upstream ``server.py`` itself runs unchanged through ``runpy``.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main() -> None:
    sys.dont_write_bytecode = True
    upstream = Path(os.environ.get("VBT_UPSTREAM") or REPO / "third_party" / "TheVirtualBiotech")
    server = upstream / "src" / "mcp_servers" / "clinicaltrials_mcp" / "server.py"
    if str(HERE) in sys.path:
        sys.path.remove(str(HERE))
    sys.path.insert(0, str(HERE))
    runpy.run_path(str(server), run_name="__main__")


if __name__ == "__main__":
    main()

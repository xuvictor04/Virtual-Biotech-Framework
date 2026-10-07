#!/usr/bin/env python3
"""Test fixture: a stdio FastMCP server that shows whether the hash seed took effect
(test_dl_launcher.py::test_hash_seed_effective).

``set_order()`` returns the iteration order of ``set('abc...z')`` (string hashes are salted
per process unless PYTHONHASHSEED takes effect), ``sys.flags.hash_randomization`` and what
the process sees of its ``PYTHON*`` environment, flags and ``sys.path``.
"""

from __future__ import annotations

import os
import string
import sys

from fastmcp import FastMCP

mcp = FastMCP("hashseed_fixture")


@mcp.tool()
def set_order() -> dict:
    """The iteration order of a set of letters, plus the interpreter's hash and environment flags."""
    return {
        "order": "".join(list(set(string.ascii_lowercase))),
        "hash_randomization": sys.flags.hash_randomization,
        "ignore_environment": sys.flags.ignore_environment,
        "isolated": sys.flags.isolated,
        "no_user_site": sys.flags.no_user_site,
        "python_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("PYTHON")},
        "sys_path": list(sys.path),
        "hostile_loaded": "vbt_hostile_probe" in sys.modules,
    }


if __name__ == "__main__":
    mcp.run()

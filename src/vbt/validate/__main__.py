"""``python -m vbt.validate [--profile P ...] [validate options]``: the same as ``vbt [--profile P ...] validate``."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from ..cli import main as vbt_main

    args = list(sys.argv[1:] if argv is None else argv)
    profiles: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--profile" and i + 1 < len(args):
            profiles += ["--profile", args[i + 1]]
            i += 2
            continue
        rest.append(args[i])
        i += 1
    return vbt_main([*profiles, "validate", *rest])


if __name__ == "__main__":
    sys.exit(main())

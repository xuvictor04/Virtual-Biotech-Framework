"""Local inference-server operations: ``vbt local profiles | serve | check | bench``.

* serving profiles (``configs/local_models.yaml``): :mod:`vbt.local.profiles`
  (load/validate, render ``vllm serve`` / ``docker run`` lines, pick a profile
  from ``nvidia-smi`` output);
* ``vbt local serve``: :mod:`vbt.local.serve`;
* ``vbt local check`` / ``bench``: :mod:`vbt.local.check` (capability probe and
  throughput sweep through the harness's OpenAI-compatible adapter).

The matching harness configurations are ``configs/profiles/local-*.yaml``
(``vbt --profile local-h100 ...``); deployment files are in ``deploy/local/``.
"""

from __future__ import annotations

from typing import Any

from .profiles import LocalProfileError, load_local_profiles


def add_local_parsers(sub: Any) -> Any:
    """Register ``vbt local {profiles,serve,check,bench}`` on an argparse
    subparsers object. Every subcommand sets ``handler(args, config)``."""
    from .check import add_bench_parser, add_check_parser
    from .serve import add_profiles_parser, add_serve_parser

    local = sub.add_parser("local", help="local LLM server: serving profiles, start, capability check, benchmark")
    lsub = local.add_subparsers(dest="local_cmd", required=True, metavar="{profiles,serve,check,bench}")
    add_profiles_parser(lsub)
    add_serve_parser(lsub)
    add_check_parser(lsub)
    add_bench_parser(lsub)
    return local


__all__ = ["LocalProfileError", "add_local_parsers", "load_local_profiles"]

"""Data-layer plugins (§9). No pyarrow at import.

Adding a plugin of an existing kind is a new module (decorated with
:func:`~vbt.datalayer.plugins.registry.register`, or an entry point in group
``vbt.datalayer.<kind>``). Adding a **kind** is the only core change: one entry here, a
protocol in ``base.py``, a conformance suite module and the descriptor field that names it.
"""

from __future__ import annotations

from .base import API_VERSION, FormatPlugin, IdentifierPlugin, LayoutPlugin, StatisticPlugin

__all__ = ["KINDS", "KIND_PACKAGES", "API_VERSION", "entry_point_group"]

KINDS: dict[str, type] = {"format": FormatPlugin, "layout": LayoutPlugin,
                          "statistic": StatisticPlugin, "identifier": IdentifierPlugin}
# phase 4 adds "envelope": EnvelopePlugin (the worked example of adding a kind)

#: In-tree builtins live in ``vbt.datalayer.plugins.<kind>s``.
KIND_PACKAGES: dict[str, str] = {kind: f"{kind}s" for kind in KINDS}


def entry_point_group(kind: str) -> str:
    return f"vbt.datalayer.{kind}"

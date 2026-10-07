"""Data-layer plugins (§9). No pyarrow at import.

Adding a plugin of an existing kind is a new module (decorated with
:func:`~vbt.datalayer.plugins.registry.register`, or an entry point in group
``vbt.datalayer.<kind>``). Adding a **kind** is the only core change: one entry here, a
protocol in ``base.py``, a conformance suite module and the descriptor field that names it.
Phase 4 did this for ``envelope`` (``base.EnvelopePlugin``, ``conformance/envelope.py``, builtins
in ``envelopes/``, named by an overlay's ``result.codec``).
"""

from __future__ import annotations

from .base import API_VERSION, EnvelopePlugin, FormatPlugin, IdentifierPlugin, LayoutPlugin, StatisticPlugin

__all__ = ["KINDS", "KIND_PACKAGES", "API_VERSION", "entry_point_group"]

KINDS: dict[str, type] = {"format": FormatPlugin, "layout": LayoutPlugin,
                          "statistic": StatisticPlugin, "identifier": IdentifierPlugin,
                          "envelope": EnvelopePlugin}

#: In-tree builtins live in ``vbt.datalayer.plugins.<kind>s``.
KIND_PACKAGES: dict[str, str] = {kind: f"{kind}s" for kind in KINDS}


def entry_point_group(kind: str) -> str:
    return f"vbt.datalayer.{kind}"

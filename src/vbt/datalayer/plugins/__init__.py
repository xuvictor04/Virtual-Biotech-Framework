"""Data-layer plugins (§9). No pyarrow at import.

Adding a plugin of an existing kind is a new module (decorated with
:func:`~vbt.datalayer.plugins.registry.register`, or an entry point in group
``vbt.datalayer.<kind>``). Adding a **kind** is the only core change: one entry here, a
protocol in ``base.py``, a conformance suite module and the descriptor field that names it.
Phase 4 did this for ``envelope`` (``base.EnvelopePlugin``, ``conformance/envelope.py``, builtins
in ``envelopes/``, named by an overlay's ``result.codec``). ASN-5 added ``derived`` the same way
(``base.DerivedPlugin``, ``conformance/derived.py``, builtins in ``derived/``, named by a derived binding's
``split.<name>``): the grouped computations a derived serve answers with, which a project can add.

The ``acquisition`` kind (``base.AcquisitionPlugin``, ``conformance/acquisition.py``, builtins in
``acquisition/``, named by a descriptor's ``acquisition.transport``) is the transports of ``vbt data acquire``.
It runs on the harness side only, so it is a :data:`HARNESS_KINDS` entry: discovered the same way
(``registry.discover(settings, kinds=HARNESS_KINDS)``), but not part of the data child's registry.
"""

from __future__ import annotations

from .base import (
    API_VERSION,
    AcquisitionPlugin,
    DerivedPlugin,
    EnvelopePlugin,
    FormatPlugin,
    IdentifierPlugin,
    LayoutPlugin,
    StatisticPlugin,
)

__all__ = ["KINDS", "KIND_PACKAGES", "HARNESS_KINDS", "HARNESS_KIND_PACKAGES", "API_VERSION", "entry_point_group"]

KINDS: dict[str, type] = {"format": FormatPlugin, "layout": LayoutPlugin,
                          "statistic": StatisticPlugin, "identifier": IdentifierPlugin,
                          "envelope": EnvelopePlugin, "derived": DerivedPlugin}

#: In-tree builtins live in ``vbt.datalayer.plugins.<kind>s`` (``derived``: ``vbt.datalayer.plugins.derived``).
KIND_PACKAGES: dict[str, str] = {kind: ("derived" if kind == "derived" else f"{kind}s") for kind in KINDS}

#: Kinds the harness uses outside the data child: ``acquisition`` (transports of ``vbt data acquire``).
HARNESS_KINDS: dict[str, type] = {"acquisition": AcquisitionPlugin}

#: Builtins of the harness-side kinds (``vbt.datalayer.plugins.acquisition``).
HARNESS_KIND_PACKAGES: dict[str, str] = {"acquisition": "acquisition"}


def entry_point_group(kind: str) -> str:
    return f"vbt.datalayer.{kind}"

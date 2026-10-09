"""The one reading of ``VBT_DL_NETWORK`` every opt-in network test shares (RR-6).

``1``, ``true``, ``yes``, ``on`` and ``full`` (case and surrounding blanks ignored) turn the live tests on; anything
else, ``0``, ``false``, ``no`` and the empty string included, keeps them off. ``full`` additionally asks the Open
Targets schema test to read every shard (:func:`network_mode`)."""
from __future__ import annotations

import os

ENABLED = frozenset({"1", "true", "yes", "on", "full"})


def network_mode() -> str:
    """The normalised ``VBT_DL_NETWORK`` value when it enables the network, else ``""``."""
    value = os.environ.get("VBT_DL_NETWORK", "").strip().lower()
    return value if value in ENABLED else ""


def network_enabled() -> bool:
    """Whether ``VBT_DL_NETWORK`` asks for the live tests."""
    return bool(network_mode())

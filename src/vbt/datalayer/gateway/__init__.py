"""The harness-side gateway (§11). No pyarrow.

* ``gateway.py``: :class:`DataGateway` (``prepare``/``finish``/``on_crash``, listings, readiness,
  pinning) and :func:`build_gateway`;
* ``classify.py``: envelope and structural classification; ``contracts.py``: argument contracts
  and predicates; ``scope.py``: the scope-completeness rule; ``fields.py``: result field maps;
  ``transforms.py``: T1-T14; ``leakage.py``: the evidence-date ceiling; ``soma_filter.py``: SOMA
  value filters; ``files.py``: output files; ``readiness.py``: call-scoped readiness;
  ``service_client.py``: typed calls to the data child.
"""

from __future__ import annotations

from .gateway import GATEWAY_VERSION, DataGateway, build_gateway

__all__ = ["DataGateway", "build_gateway", "GATEWAY_VERSION"]

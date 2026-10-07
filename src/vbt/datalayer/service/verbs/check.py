"""``_check``: readiness R1-R10 per table, column, container, item table and partition (§13).

``depth``: ``shallow`` (layout and manifest probes only), ``standard`` (every check, keys streamed
in full only up to ``data.readiness.key_check_full_max_rows``, samples elsewhere), ``deep`` (full
uniqueness and referential passes). The response also reports the child's
``sys.flags.hash_randomization``, so the gateway can verify that the launcher's hash seed took effect.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping

from ...ipc import VERB_CHECK, CheckRequest, CheckResponse
from .. import ServiceContext
from ..checks import check_tables

__all__ = ["check", "VERBS"]


def check(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = CheckRequest.model_validate(dict(payload))
    tables, errors = check_tables(ctx, req.tables, req.depth)
    resp = CheckResponse(tables=tables, depth=req.depth, hash_randomization=int(sys.flags.hash_randomization),
                         errors=errors)
    return resp.model_dump(mode="json", by_alias=True)


VERBS = {VERB_CHECK: check}

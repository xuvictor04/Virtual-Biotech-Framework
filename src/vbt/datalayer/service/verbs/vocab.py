"""``_vocab``: the distinct values of a column in its storage type (§11.8, R7).

Values come from row-group ``min == max`` statistics when every row group is single-valued, else
from a bounded scan (``data.readiness.vocab_budget_bytes``); placeholders and ``missing_values``
of the column are left out. ``values`` are the storage-typed values (a float32 ``0.05`` is
``0.05``), ``rendered`` their canonical text (strings as themselves), ``counts`` per rendered value
when the scan counted them. ``complete`` is false when the budget or ``max_values`` cut the scan.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...ipc import VERB_VOCAB, VocabRequest, VocabResponse
from ...rowkey import render_value
from .. import ServiceContext
from ..reader import rendered

__all__ = ["vocab", "VERBS"]


def vocab(ctx: ServiceContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    req = VocabRequest.model_validate(dict(payload))
    reader = ctx.reader(req.table)
    path = reader.physical_path(req.column)
    snap = reader.snapshot(path, max_values=req.max_values)
    spec = reader.column_spec(path)
    excluded = {render_value(x) for x in [*(getattr(spec, "placeholders", None) or []),
                                          *(getattr(spec, "missing_values", None) or [])]}
    st = snap.storage_type
    values, texts, counts = [], [], {}
    for v in snap.values:
        r = render_value(v, st)
        if r in excluded:
            continue
        value = rendered(v, st)
        text = value if isinstance(value, str) else r
        values.append(value)
        texts.append(text)
        if snap.counts is not None and v in snap.counts:
            counts[text] = int(snap.counts[v])
    resp = VocabResponse(values=values, rendered=texts, fingerprint=reader.fingerprint(), complete=snap.complete,
                         storage_type=st, counts=counts if snap.counts is not None else None)
    return resp.model_dump(mode="json")


VERBS = {VERB_VOCAB: vocab}

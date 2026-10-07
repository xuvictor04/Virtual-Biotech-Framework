"""``cosine``: cosine similarity of embedding vectors (§9.4, phase 3; capability ``veto_labels``).

The value lives on ``[-1, 1]`` and higher is stronger. Upstream tools attach a string label that
bins the similarity (``interpretation``: "highly similar", "moderately similar", ...) with
thresholds the data does not declare. Capability ``veto_labels``: :meth:`vetoed_companions` names
those sibling fields, which the gateway drops from served rows (the number is the evidence; a bin
label is a claim nobody declared). A spec may add more companions with ``veto: [field, ...]`` in its
``columns`` facet; ``interpretation`` is always vetoed.

:func:`cosine` computes the similarity of two vectors in float64 (``None`` when either vector has
zero norm or a non-finite element: the similarity is undefined, never 0).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import Predicate
from ..base import UnsupportedFilter
from ..registry import register
from . import spec_get
from .score import Score01Statistic

__all__ = ["CosineStatistic", "cosine", "VETOED"]

#: Companion fields every cosine vetoes.
VETOED = ("interpretation",)


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    """The cosine of ``a`` and ``b`` (equal lengths), or ``None`` when it is undefined."""
    if len(a) != len(b):
        raise ValueError(f"vectors of different lengths ({len(a)} and {len(b)})")
    xs = [float(x) for x in a]
    ys = [float(y) for y in b]
    if not all(math.isfinite(v) for v in xs + ys):
        return None
    na = math.sqrt(math.fsum(x * x for x in xs))
    nb = math.sqrt(math.fsum(y * y for y in ys))
    if na == 0 or nb == 0:
        return None
    value = math.fsum(x * y for x, y in zip(xs, ys)) / (na * nb)
    return max(-1.0, min(1.0, value))


@register
class CosineStatistic(Score01Statistic):
    name: ClassVar[str] = "cosine"
    version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset({"veto_labels"})
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (-1.0, 1.0)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"mean", "median", "min", "max", "count"})

    def vetoed_companions(self, spec: Any) -> tuple[str, ...]:
        extra = (spec_get(spec, "columns", None) or {}).get("veto")
        more = [extra] if isinstance(extra, str) else list(extra or [])
        return tuple(dict.fromkeys([*VETOED, *(str(x) for x in more)]))

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op == "significant":
            raise UnsupportedFilter(f"{column}: a cosine has no significance rule; pass a threshold",
                                    reason="op", column=column)
        return super().compare(column, op, value, spec, confirmed)

    def scale_text(self, spec: Any) -> str:
        return (f"{super(Score01Statistic, self).scale_text(spec)}; cosine similarity "
                f"(bin labels such as {', '.join(self.vetoed_companions(spec))} are not served)")

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("cosine", {"role": "measure", "statistic": cls.name},
                          values=(0.12, 0.97, -0.4, 0.97, 0.0), best=0.97,
                          thresholds=(("ge", 0.5), ("lt", 0.0), ("gt", -0.4), ("ge_abs", 0.9)),
                          refused=(("gt", 1.5), ("lt", -2.0), ("significant", None)),
                          agg=("mean", "max")),
        )

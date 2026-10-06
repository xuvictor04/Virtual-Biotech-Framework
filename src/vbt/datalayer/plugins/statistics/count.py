"""``count``: non-negative counts of declared things (§7 count role, §9.4).

Counts live on ``[0, inf)``; a negative threshold raises ``UnsupportedFilter``. A count stored
as ``float64`` (``stored_as: float64``, pandas round trips) is accepted when it is integral:
:meth:`CountStatistic.validate` confirms integrality only from a complete distinct-value
snapshot, and refutes it when a fractional value is observed. With level ``keys``
(:meth:`aggregate`) each level key is counted once (I15).
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import ColumnStats, ValueSnapshot
from ..registry import register
from . import StatisticBase, is_number, spec_get

_FLOAT_TYPES = frozenset({"float", "double", "float32", "float64", "halffloat", "float16"})


@register
class CountStatistic(StatisticBase):
    name: ClassVar[str] = "count"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, None)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"sum", "mean", "median", "min", "max", "count"})

    def check_facts(self, stats: ColumnStats | None, snapshot: ValueSnapshot | None, spec: Any,
                    known: list[Any] | None, mn: float | None, mx: float | None, facts: dict[str, Any],
                    problems: list[str], undecided: list[str]) -> None:
        super().check_facts(stats, snapshot, spec, known, mn, mx, facts, problems, undecided)
        storage = str(getattr(stats, "storage_type", None) or spec_get(spec, "stored_as", "") or "").lower()
        if storage not in _FLOAT_TYPES:
            return
        if known is None:
            undecided.append(f"a {storage} count is accepted only when integral; integrality needs a complete "
                             f"distinct-value snapshot")
            return
        fractional = [v for v in known if is_number(v) and not float(v).is_integer()]
        if fractional:
            problems.append(f"fractional count value(s) {', '.join(map(repr, fractional[:5]))}")
        else:
            facts["integral"] = True

    def describe(self, spec: Any) -> str:
        counts = spec_get(spec, "counts", None)
        text = super().describe(spec)
        return f"{text}; counts {counts}" if counts else text

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("count", {"role": "count", "counts": "distinct publications"},
                          values=(0, 3, 12, 3, 1), best=12,
                          thresholds=(("ge", 3), ("gt", 0), ("le", 12), ("lt", 1)),
                          refused=(("gt", -1), ("lt", -5)), agg=("sum", "max", "count"),
                          group=("datasourceId", "europepmc", "chembl")),
            StatisticCase("count_float_missing_zero", {"role": "count", "missing": "zero", "stored_as": "float64",
                                                       "missing_values": [-1]},
                          values=(0.0, 2.0, 5.0, None), best=5.0, unknown=(-1,),
                          thresholds=(("ge", 2.0), ("lt", 2.0)), refused=(("ge", -0.5),), agg=("sum",)),
        )

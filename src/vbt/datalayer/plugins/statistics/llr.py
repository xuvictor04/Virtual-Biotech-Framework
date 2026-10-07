"""``llr_critval``: FAERS log-likelihood ratios judged against a per-drug critical value (§9.4).

Open Targets ``adverse_events`` rows carry ``llr`` and the drug's ``critval``; an event is
significant when ``llr > critval`` (upstream's strict comparison). The predicate IR has no
column-to-column comparison, so ``op="significant"`` takes the critical value as its value (the
caller reads it for the fixed drug) or falls back to ``spec.significant_above``; without either
it raises ``UnsupportedFilter`` (``reason="unbound_argument"``). The LLR is non-negative and
higher is stronger; descriptors normally declare ``comparable_within: [chemblId]`` because the
critical value is per drug.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from ...predicate import Cmp, Predicate
from ..base import UnsupportedFilter
from ..registry import register
from . import StatisticBase, spec_get


@register
class LlrCritvalStatistic(StatisticBase):
    name: ClassVar[str] = "llr_critval"
    version: ClassVar[str] = "1.0"
    default_scale: ClassVar[tuple[float | None, float | None] | None] = (0.0, None)
    default_direction: ClassVar[str | None] = "higher_is_stronger"
    aggregations = frozenset({"min", "max", "median", "count"})

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op == "significant":
            crit = value if value is not None else spec_get(spec, "significant_above", None)
            if crit is None:
                raise UnsupportedFilter(f"{column}: significance needs the critical value of the fixed drug",
                                        reason="unbound_argument", column=column)
            return Cmp(column, ">", crit)
        return super().compare(column, op, value, spec, confirmed)

    def scale_text(self, spec: Any) -> str:
        crit = spec_get(spec, "significant_above", None)
        rule = f"significant when > {crit}" if crit is not None else "significant when llr > the drug's critval"
        return f"{super().scale_text(spec)}; {rule} (strict)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("llr_critval", {"role": "measure", "statistic": cls.name,
                                          "comparable_within": ["chemblId"]},
                          values=(12.5, 300.2, 7.0, 300.2, 0.0), best=300.2,
                          thresholds=(("significant", 12.5), ("ge", 7.0), ("lt", 12.5)),
                          refused=(("gt", -1.0),), group=("chemblId", "CHEMBL25", "CHEMBL1201"),
                          fixed_scope={"chemblId": "CHEMBL25"}),
        )

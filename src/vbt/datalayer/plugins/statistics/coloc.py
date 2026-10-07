"""Colocalisation posteriors (§9.4, phase 3): ``clpp`` (eCAVIAR) and ``coloc_h4`` (coloc).

Both are probabilities on ``[0, 1]`` where higher is stronger, but they are different quantities:
eCAVIAR's colocalisation posterior probability sums per-variant products, coloc's H4 is the
posterior of one shared causal variant. A row of one method is never ranked against, thresholded
with or aggregated with a row of the other (C7: upstream fell back to eCAVIAR for an unknown
method). Each plugin therefore refuses aggregation over ``groups`` whose ``method`` (or the
descriptor's ``comparable_within`` columns) differ, and its description names the method.
The conventional evidence rules are disclosed, not applied: H4 >= 0.8 and CLPP >= 0.01 are what the
literature calls colocalised; ``op="significant"`` uses ``significant_above`` or these defaults
(inclusive).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import Cmp, Predicate
from ..base import AggResult, UnsupportedFilter
from ..registry import register
from . import _num, spec_get
from .score import Score01Statistic

__all__ = ["ClppStatistic", "ColocH4Statistic"]


class _Posterior(Score01Statistic):
    method: ClassVar[str] = ""
    conventional: ClassVar[float] = 0.8
    aggregations = frozenset({"median", "min", "max", "count"})

    def threshold(self, spec: Any, value: Any) -> float:
        if value is not None and not isinstance(value, bool):
            return float(value)
        return float(spec_get(spec, "significant_above", None) or self.conventional)

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op == "significant":
            return Cmp(column, ">=", self.threshold(spec, value))
        if op.endswith("_abs"):
            raise UnsupportedFilter(f"{op} is not defined for a posterior probability", reason="op", column=column)
        return super().compare(column, op, value, spec, confirmed)

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if op == "significant":
            values = [v for v in values if v is not None and not isinstance(v, bool)]
            op = "ge"
        super().check_values(column, op, values, spec, confirmed)

    def aggregate(self, values: Sequence[Any], how: str, spec: Any, keys: Sequence[Any] | None = None, *,
                  groups: Sequence[Mapping[str, Any]] | None = None) -> AggResult:
        if groups is not None:
            methods = {str(g.get("method")).lower() for g in groups if isinstance(g, Mapping) and g.get("method")}
            if len(methods) > 1 or (methods and self.method not in methods):
                raise UnsupportedFilter(f"{self.name} values cannot be pooled with other colocalisation methods "
                                        f"({', '.join(sorted(methods))})", reason="group", group=["method"])
        return super().aggregate(values, how, spec, keys, groups=groups)

    def scale_text(self, spec: Any) -> str:
        rule = _num(self.threshold(spec, None))
        return f"{super().scale_text(spec)}; {self.method} posterior (colocalised when ≥ {rule}, inclusive, by convention)"


@register
class ClppStatistic(_Posterior):
    name: ClassVar[str] = "clpp"
    version: ClassVar[str] = "1.0"
    method: ClassVar[str] = "ecaviar"
    conventional: ClassVar[float] = 0.01

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("clpp", {"role": "measure", "statistic": cls.name, "comparable_within": ["rightStudyType"]},
                          values=(0.002, 0.31, 0.01, 0.31), best=0.31,
                          thresholds=(("significant", None), ("ge", 0.01), ("lt", 0.002)),
                          refused=(("ge", 1.5), ("gt_abs", 0.1)), agg=("max", "median"),
                          group=("rightStudyType", "eqtl", "pqtl"), fixed_scope={"rightStudyType": "eqtl"}),
        )


@register
class ColocH4Statistic(_Posterior):
    name: ClassVar[str] = "coloc_h4"
    version: ClassVar[str] = "1.0"
    method: ClassVar[str] = "coloc"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        return (
            StatisticCase("coloc_h4", {"role": "measure", "statistic": cls.name,
                                       "comparable_within": ["rightStudyType"]},
                          values=(0.95, 0.12, 0.8, 0.95), best=0.95,
                          thresholds=(("significant", None), ("ge", 0.8), ("lt", 0.12)),
                          refused=(("gt", 1.2), ("lt", -0.1)), agg=("max", "count"),
                          group=("rightStudyType", "eqtl", "sqtl"), fixed_scope={"rightStudyType": "eqtl"}),
        )

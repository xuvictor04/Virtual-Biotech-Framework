"""``gene_effect``: CRISPR gene effects (DepMap Chronos) with a declared dependency cutoff (§9.4, phase 3).

Lower is stronger (a more negative effect is a stronger dependency) and the scale is unbounded. The
dependency rule is the measure's declared ``cutoff`` (DepMap convention: ``geneEffect <= -0.5``,
inclusive), never a constant in code:

* ``op="significant"`` (also spelled ``essential``) compiles to the cutoff's own op and value, so
  the boundary value is a dependency exactly when the cutoff is inclusive;
* :meth:`GeneEffectStatistic.describe` names the cutoff and its inclusivity;
* aggregation ``essential_fraction`` is the fraction of known effects that meet the cutoff, over the
  known effects only (an unscreened or null effect is excluded and counted, never a non-dependency);
  with no known effect it is ``None``, never 0.

Without a declared cutoff, ``significant`` and ``essential_fraction`` raise ``UnsupportedFilter``
(``reason="unbound_argument"``): the rule is not guessed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import Cmp, Predicate, evaluate
from ..base import UnsupportedFilter
from ..registry import register
from . import _CMP, StatisticBase, spec_get

__all__ = ["GeneEffectStatistic", "meets_cutoff"]


def meets_cutoff(value: Any, spec: Any) -> bool | None:
    """Whether ``value`` meets the measure's declared cutoff (``None`` for unknown values)."""
    cutoff = spec_get(spec, "cutoff", None)
    if cutoff is None:
        raise UnsupportedFilter("no cutoff is declared for this gene effect", reason="unbound_argument")
    return evaluate(Cmp("v", _CMP[spec_get(cutoff, "op")], float(spec_get(cutoff, "value"))), {"v": value})


@register
class GeneEffectStatistic(StatisticBase):
    name: ClassVar[str] = "gene_effect"
    version: ClassVar[str] = "1.0"
    default_direction: ClassVar[str | None] = "lower_is_stronger"
    aggregations = frozenset({"mean", "median", "min", "max", "count", "essential_fraction"})

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op == "significant":
            cutoff = spec_get(spec, "cutoff", None)
            if cutoff is None:
                raise UnsupportedFilter(f"{column}: a dependency filter needs the declared cutoff",
                                        reason="unbound_argument", column=column)
            return Cmp(column, _CMP[spec_get(cutoff, "op")], float(spec_get(cutoff, "value")))
        return super().compare(column, op, value, spec, confirmed)

    def predicate(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None,
                  fixed_scope: Mapping[str, Any]) -> Predicate:
        if str(op).strip().lower() == "essential":
            op = "significant"
        return super().predicate(column, op, value, spec, confirmed, fixed_scope)

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if op == "significant":
            return
        super().check_values(column, op, values, spec, confirmed)

    def combine(self, known: list[Any], how: str, spec: Any) -> Any:
        if how != "essential_fraction":
            return super().combine(known, how, spec)
        flags = [meets_cutoff(v, spec) for v in known]
        flags = [f for f in flags if f is not None]
        return (sum(1 for f in flags if f) / len(flags)) if flags else None

    def aggregate(self, values: Sequence[Any], how: str, spec: Any, keys: Sequence[Any] | None = None, *,
                  groups: Sequence[Mapping[str, Any]] | None = None) -> Any:
        if str(how).lower() == "essential_fraction" and spec_get(spec, "cutoff", None) is None:
            raise UnsupportedFilter("essential_fraction needs the declared cutoff", reason="unbound_argument")
        return super().aggregate(values, how, spec, keys, groups=groups)

    def scale_text(self, spec: Any) -> str:
        text = super().scale_text(spec)
        if spec_get(spec, "cutoff", None) is None:
            text += "; no dependency cutoff declared"
        return text

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase

        spec = {"role": "measure", "statistic": cls.name, "missing": "unknown",
                "cutoff": {"value": -0.5, "op": "le", "meaning": "dependency", "origin": "DepMap convention"},
                "comparable_within": ["release"]}
        return (
            StatisticCase("gene_effect", spec, values=(-0.5, -1.8, 0.2, -0.49, -1.8), best=-1.8,
                          thresholds=(("le", -0.5), ("lt", -0.5), ("ge", 0.0), ("significant", None)),
                          group=("release", "24Q4", "23Q2"), fixed_scope={"release": "24Q4"},
                          agg=("mean", "median", "essential_fraction")),
            StatisticCase("gene_effect_exclusive_cutoff",
                          {"role": "measure", "statistic": cls.name,
                           "cutoff": {"value": -1.0, "op": "lt", "meaning": "strong dependency"}},
                          values=(-1.0, -2.0, 0.0), best=-2.0,
                          thresholds=(("significant", None), ("le", -1.0)), agg=("essential_fraction",)),
        )

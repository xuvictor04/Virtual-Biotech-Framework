"""``fdr_bh``: Benjamini-Hochberg q-values over a declared multiple-testing family (§9.4, phase 3).

A q-value is a probability on ``[0, 1]`` where lower is stronger, like :mod:`.pvalue`, but it is
only meaningful within the family it was adjusted over. The measure's ``family`` facet names the
columns that define one family (Tahoe ``padj``: drug, cell line, concentration and plate); this
plugin:

* ranks within the family (``RankKey.within`` lists the family columns after any
  ``comparable_within`` ones), so a top-k never interleaves q-values of different families (C30);
* adjusts with :meth:`FdrBhStatistic.adjust` over the **full declared family** ``m``, which counts
  members that were tested but have no p-value worth reporting (a gene set with zero overlap has
  p = 1 and still counts, §6.8 S9), so ``m`` may exceed the number of p-values given but never fall
  below it (that would make every q-value too small);
* reads ``op="significant"`` as ``q <= alpha`` (inclusive, the BH convention) with ``alpha`` from the
  call or from ``significant_above``; without either it raises ``UnsupportedFilter``
  (``reason="unbound_argument"``).

The S-7 conformance check runs on plugins whose cases include a ``FamilyCase``: it compares
:meth:`adjust` with a reference BH (statsmodels when installed, else a local implementation).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

from ...predicate import Cmp, Predicate, RankKey, is_null
from ..base import FamilySpec, UnsupportedFilter
from ..registry import register
from . import DIRECTION_ORDER, spec_get
from .pvalue import PValueStatistic, fdr_bh

__all__ = ["FdrBhStatistic", "bh_adjust"]


def bh_adjust(pvalues: Sequence[float | None], m: int | None = None) -> list[float | None]:
    """BH q-values in input order over a family of ``m`` tests (default: the known p-values).

    ``None`` and NaN stay ``None``; ``m`` smaller than the number of known p-values raises
    ``ValueError`` (an undersized family understates every q-value)."""
    return fdr_bh(pvalues, m)


@register
class FdrBhStatistic(PValueStatistic):
    name: ClassVar[str] = "fdr_bh"
    version: ClassVar[str] = "1.0"
    aggregations = frozenset({"min", "max", "median", "count"})

    def family_columns(self, spec: Any) -> tuple[str, ...]:
        return tuple(str(c) for c in spec_get(spec, "family", []) or [])

    def family(self, spec: Any) -> FamilySpec | None:
        cols = self.family_columns(spec)
        if not cols:
            return None
        return FamilySpec(columns=cols, method="bh", origin=spec_get(spec, "verified_by", None))

    def sort_key(self, column: str, spec: Any, direction: str | None) -> RankKey:
        if direction is None:
            direction = DIRECTION_ORDER.get(self.direction(spec)) or "asc"
        within = [str(c) for c in spec_get(spec, "comparable_within", []) or []]
        within += [c for c in self.family_columns(spec) if c not in within]
        return RankKey(column=column, direction=direction, nulls="last", statistic=self.name,  # type: ignore[arg-type]
                       within=tuple(within))

    def adjust(self, pvalues: Sequence[float | None], m: int | None = None) -> list[float | None]:
        """BH over the full family: ``m`` counts every member of the family, tested or not."""
        known = sum(1 for p in pvalues if p is not None and not is_null(p))
        if m is not None and m < known:
            raise UnsupportedFilter(f"family size m={m} is smaller than the {known} p-values given; BH over a "
                                    f"partial family understates every q-value", reason="family")
        for p in pvalues:
            if p is not None and not is_null(p) and not 0.0 <= float(p) <= 1.0:
                raise UnsupportedFilter(f"p-value {p!r} is outside [0, 1]", reason="scale")
        return bh_adjust(pvalues, m)

    def compare(self, column: str, op: str, value: Any, spec: Any, confirmed: Mapping[str, Any] | None) -> Predicate:
        if op == "significant":
            alpha = value if value is not None and not isinstance(value, bool) else spec_get(spec, "significant_above")
            if alpha is None:
                raise UnsupportedFilter(f"{column}: significance needs an FDR level (alpha)", reason="unbound_argument",
                                        column=column)
            return Cmp(column, "<=", float(alpha))
        return super().compare(column, op, value, spec, confirmed)

    def check_values(self, column: str, op: str, values: Sequence[Any], spec: Any,
                     confirmed: Mapping[str, Any] | None) -> None:
        if op == "significant":
            values = [v for v in values if v is not None and not isinstance(v, bool)]
        if op.endswith("_abs"):
            raise UnsupportedFilter(f"{op} is not defined for q-values", reason="op", column=column)
        self.check_scale(column, op if op != "significant" else "le", values, spec, confirmed)

    def scale_text(self, spec: Any) -> str:
        text = super(PValueStatistic, self).scale_text(spec)
        cols = self.family_columns(spec)
        fam = f"BH family: {', '.join(cols)}" if cols else "BH family not declared"
        return f"{text}; q-values ({fam}; significant when q ≤ alpha, inclusive)"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase
        from ..conformance.statistic import FamilyCase

        spec = {"role": "measure", "statistic": cls.name, "family": ["drug", "cell_line"]}
        return (
            StatisticCase("fdr_bh", spec, values=(0.04, 1e-6, 0.5, 1e-6, 1.0), best=1e-6,
                          thresholds=(("le", 0.05), ("lt", 0.5), ("ge", 0.04), ("significant", 0.1)),
                          refused=(("le", 2.0), ("lt", -0.1), ("gt_abs", 0.1)),
                          confirmed={"confirmed": True, "scale": [0.0, 1.0]}, agg=("min", "count"),
                          group=("drug", "Bortezomib", "Erdafitinib"),
                          fixed_scope={}),
            FamilyCase("full_family_with_zero_overlap", (0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205),
                       m=12),
            FamilyCase("known_only", (0.01, None, 0.04, 0.03, 0.005, float("nan")), m=None),
            FamilyCase("ties_and_ones", (0.02, 0.02, 1.0, 1.0, 0.5), m=7),
        )

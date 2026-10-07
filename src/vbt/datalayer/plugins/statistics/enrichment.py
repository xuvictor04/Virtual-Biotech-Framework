"""Set-overlap tests (§9.4, phase 3; capability ``test``): ``hypergeom_enrichment`` and ``fisher``.

``test(overlap, set_n, query_n, universe_n, spec)`` returns the p-value of seeing ``overlap`` query
members in a set of ``set_n`` members, when ``query_n`` items are drawn from a universe of
``universe_n`` (every count taken **within the universe**: a set member or query item outside the
universe is not counted, §6.8 S9):

* ``hypergeom_enrichment``: the upper tail ``P(X >= overlap)`` of the hypergeometric distribution
  (scipy's ``hypergeom.sf(overlap - 1, universe_n, set_n, query_n)``); ``overlap = 0`` gives 1;
* ``fisher``: Fisher's exact test on the 2x2 table ``[[k, n - k], [K - k, N - K - n + k]]``; the
  alternative is the measure's ``direction`` facet (``higher_is_stronger`` -> ``greater``, which
  equals the hypergeometric tail; ``none`` -> ``two-sided``, scipy's default, summing every table at
  most as likely as the observed one with scipy's relative tolerance ``1 + 1e-7``).

The p-values are computed in float64 from log-gamma terms summed by ratio recurrence, without scipy
(the S-9 conformance check compares them with scipy when it is installed). The result of a test is a
p-value, so ranking, thresholds and aggregation follow :mod:`.pvalue`. Inconsistent counts (overlap
larger than the set or the query, a set larger than the universe) raise ``UnsupportedFilter``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from ..base import UnsupportedFilter
from ..registry import register
from . import spec_get
from .pvalue import PValueStatistic

__all__ = ["HypergeomEnrichmentStatistic", "FisherStatistic", "hypergeom_sf", "hypergeom_pmf", "fisher_exact",
           "fold_enrichment"]

_REL = 1 + 1e-7                                       # scipy's tolerance for "as likely as observed"


def _check(k: int, K: int, n: int, N: int) -> None:
    for name, v in (("overlap", k), ("set_n", K), ("query_n", n), ("universe_n", N)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise UnsupportedFilter(f"{name} must be a non-negative integer, got {v!r}", reason="invalid_value")
    if K > N or n > N:
        raise UnsupportedFilter(f"set_n={K} and query_n={n} must be counted within the universe (N={N})",
                                reason="invalid_value")
    if k > min(K, n):
        raise UnsupportedFilter(f"overlap={k} exceeds the set ({K}) or the query ({n})", reason="invalid_value")


def _logpmf(k: int, N: int, K: int, n: int) -> float:
    lg = math.lgamma
    return (lg(K + 1) - lg(k + 1) - lg(K - k + 1) + lg(N - K + 1) - lg(n - k + 1) - lg(N - K - n + k + 1)
            - lg(N + 1) + lg(n + 1) + lg(N - n + 1))


def hypergeom_pmf(k: int, N: int, K: int, n: int) -> float:
    lo, hi = max(0, n - (N - K)), min(K, n)
    if k < lo or k > hi:
        return 0.0
    return math.exp(_logpmf(k, N, K, n))


def _support(N: int, K: int, n: int) -> tuple[int, int]:
    return max(0, n - (N - K)), min(K, n)


def _pmfs(N: int, K: int, n: int) -> tuple[int, list[float]]:
    """``(lo, [pmf(lo), ..., pmf(hi)])`` from the mode outwards by ratio recurrence (no underflow at the mode)."""
    lo, hi = _support(N, K, n)
    mode = min(hi, max(lo, int((n + 1) * (K + 1) / (N + 2))))
    out = [0.0] * (hi - lo + 1)
    out[mode - lo] = math.exp(_logpmf(mode, N, K, n))
    for i in range(mode, hi):                         # pmf(i + 1) = pmf(i) (K - i)(n - i) / ((i + 1)(N - K - n + i + 1))
        out[i + 1 - lo] = out[i - lo] * (K - i) * (n - i) / ((i + 1) * (N - K - n + i + 1))
    for i in range(mode, lo, -1):                     # pmf(i - 1) = pmf(i) i (N - K - n + i) / ((K - i + 1)(n - i + 1))
        out[i - 1 - lo] = out[i - lo] * i * (N - K - n + i) / ((K - i + 1) * (n - i + 1))
    return lo, out


def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """``P(X >= k)`` for ``X ~ Hypergeom(N, K, n)`` (scipy: ``hypergeom.sf(k - 1, N, K, n)``)."""
    _check(k, K, n, N)
    lo, hi = _support(N, K, n)
    if k <= lo:
        return 1.0
    if k > hi:
        return 0.0
    first, pm = _pmfs(N, K, n)
    upper = math.fsum(pm[k - first:])
    lower = math.fsum(pm[: k - first])
    return min(1.0, max(0.0, upper / (upper + lower))) if upper + lower > 0 else 0.0


def fisher_exact(k: int, N: int, K: int, n: int, alternative: str = "two-sided") -> float:
    """Fisher's exact p-value of the 2x2 table of ``k`` (scipy ``fisher_exact`` semantics)."""
    _check(k, K, n, N)
    if alternative == "greater":
        return hypergeom_sf(k, N, K, n)
    first, pm = _pmfs(N, K, n)
    total = math.fsum(pm)
    if alternative == "less":
        return min(1.0, math.fsum(pm[: k - first + 1]) / total)
    if alternative != "two-sided":
        raise UnsupportedFilter(f"unknown alternative {alternative!r}", reason="invalid_value")
    observed = pm[k - first]
    p = math.fsum(x for x in pm if x <= observed * _REL) / total
    return min(1.0, p)


def fold_enrichment(k: int, K: int, n: int, N: int) -> float | None:
    """``(k / n) / (K / N)``: observed over expected overlap (``None`` when undefined)."""
    if n == 0 or K == 0 or N == 0:
        return None
    return (k / n) / (K / N)


class _SetTest(PValueStatistic):
    capabilities: ClassVar[frozenset[str]] = frozenset({"test"})
    aggregations = frozenset({"min", "max", "median", "count"})
    description: ClassVar[str] = ""

    def scale_text(self, spec: Any) -> str:
        return f"{super().scale_text(spec)}; p-value of {self.description}"


@register
class HypergeomEnrichmentStatistic(_SetTest):
    name: ClassVar[str] = "hypergeom_enrichment"
    version: ClassVar[str] = "1.0"
    description: ClassVar[str] = "the hypergeometric upper tail P(X >= overlap) within the declared universe"

    def test(self, overlap: int, set_n: int, query_n: int, universe_n: int, spec: Any = None) -> float:
        return hypergeom_sf(overlap, universe_n, set_n, query_n)

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase
        from ..conformance.statistic import SetTestCase

        return (
            StatisticCase(cls.name, {"role": "measure", "statistic": cls.name},
                          values=(0.03, 1e-12, 1.0, 1e-12), best=1e-12,
                          thresholds=(("lt", 0.05), ("le", 1e-12)), refused=(("lt", 2.0),), agg=("min", "count")),
            # hand-computed: N=20, K=5, n=4, k=3 -> (C(5,3)C(15,1) + C(5,4)C(15,0)) / C(20,4) = 155/4845
            SetTestCase("hand_k3", overlap=3, set_n=5, query_n=4, universe_n=20, expected=155 / 4845),
            SetTestCase("zero_overlap", overlap=0, set_n=40, query_n=10, universe_n=1000, expected=1.0),
            SetTestCase("full_overlap", overlap=4, set_n=4, query_n=4, universe_n=10, expected=1 / 210),
            SetTestCase("large", overlap=12, set_n=200, query_n=150, universe_n=19000, expected=None),
            SetTestCase("tiny_tail", overlap=40, set_n=60, query_n=50, universe_n=20000, expected=None),
            SetTestCase("inconsistent", overlap=6, set_n=5, query_n=10, universe_n=100, expected="refused"),
        )


@register
class FisherStatistic(_SetTest):
    name: ClassVar[str] = "fisher"
    version: ClassVar[str] = "1.0"
    description: ClassVar[str] = "Fisher's exact test on the 2x2 overlap table"

    @staticmethod
    def alternative(spec: Any) -> str:
        return {"higher_is_stronger": "greater", "lower_is_stronger": "less"}.get(
            str(spec_get(spec, "direction", None) or "none"), "two-sided")

    def test(self, overlap: int, set_n: int, query_n: int, universe_n: int, spec: Any = None) -> float:
        return fisher_exact(overlap, universe_n, set_n, query_n, self.alternative(spec))

    def direction(self, spec: Any) -> str:
        return "lower_is_stronger"                  # the facet picks the test's alternative; p-values rank ascending

    def scale_text(self, spec: Any) -> str:
        return f"{super().scale_text(spec)} ({self.alternative(spec)})"

    @classmethod
    def conformance_cases(cls) -> Any:
        from ..conformance.golden import StatisticCase
        from ..conformance.statistic import SetTestCase

        return (
            StatisticCase(cls.name, {"role": "measure", "statistic": cls.name},
                          values=(0.2, 0.004, 1.0), best=0.004, thresholds=(("lt", 0.05),), agg=("min",)),
            SetTestCase("two_sided_hand", overlap=3, set_n=5, query_n=4, universe_n=20, expected=None),
            SetTestCase("greater", overlap=3, set_n=5, query_n=4, universe_n=20, expected=155 / 4845,
                        alternative="greater"),
            SetTestCase("less", overlap=0, set_n=8, query_n=6, universe_n=30, expected=None, alternative="less"),
            SetTestCase("depleted_two_sided", overlap=0, set_n=50, query_n=50, universe_n=200, expected=None),
        )

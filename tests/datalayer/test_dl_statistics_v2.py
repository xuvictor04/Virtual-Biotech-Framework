"""Statistic plugins v2 (§9.4 phase 3, F15) and the phase-3 statistic suite checks S-7 and S-9.

The suite (S-1..S-9) runs over every registered statistic plugin from
``vbt.datalayer.plugins.conformance.statistic``; S-7 and S-9 are collected here as well. The explicit
cases pin each new plugin's semantics:

* ``fdr_bh``: BH over the full declared family (m counts untested members), ranking within the family,
  ``significant`` inclusive at alpha;
* ``log2fc`` / ``wald`` / ``effect_beta`` rank by magnitude; ``std_error`` refuses ``*_abs``; ``odds_ratio``
  ranks by |log OR| and averages geometrically;
* ``gene_effect``: the declared cutoff and its inclusivity drive ``significant`` and ``essential_fraction``;
* ``clpp`` / ``coloc_h4``: never pooled across colocalisation methods;
* ``tpm``: not comparable across units, blank units refused when pooling; ``log2_intensity`` comparable
  only within a cohort;
* ``cosine``: ``veto_labels`` names ``interpretation``; range [-1, 1];
* ``hypergeom_enrichment`` / ``fisher``: ``test`` matches scipy on hand-computed tables;
* ``survival_time``: thresholds are ``CensoredCmp``; Kaplan-Meier medians; means refused.
"""

from __future__ import annotations

import math

import pytest

from vbt.datalayer.descriptor.columns import validate_column
from vbt.datalayer.plugins.base import UnsupportedFilter
from vbt.datalayer.plugins.conformance.statistic import (  # noqa: F401  (collects S-7 and S-9 here too)
    test_s7_bh_over_the_full_family,
    test_s9_paired_survival,
    test_s9_set_test_matches_reference,
)
from vbt.datalayer.plugins.registry import discover
from vbt.datalayer.predicate import CensoredCmp, Cmp, CmpAbs, evaluate

PHASE3 = ("fdr_bh", "log2fc", "std_error", "wald", "effect_beta", "odds_ratio", "gene_effect", "clpp", "coloc_h4",
          "tpm", "log2_intensity", "cosine", "hypergeom_enrichment", "fisher", "survival_time")


@pytest.fixture(scope="module")
def reg():
    return discover(entry_points=False)


def plugin(reg, name):
    p = reg.find("statistic", name)
    assert p is not None, name
    return p


def measure(**facets):
    return validate_column({"role": "measure", **facets})


def test_phase3_statistics_are_registered_with_their_capabilities(reg) -> None:
    names = reg.names("statistic")
    assert set(PHASE3) <= set(names)
    assert "test" in plugin(reg, "hypergeom_enrichment").capabilities and "test" in plugin(reg, "fisher").capabilities
    assert plugin(reg, "survival_time").capabilities == frozenset({"paired"})
    assert plugin(reg, "cosine").capabilities == frozenset({"veto_labels"})


# ---------------------------------------------------------------------------- fdr_bh


def test_fdr_bh_uses_the_full_family_and_ranks_within_it(reg) -> None:
    p = plugin(reg, "fdr_bh")
    spec = measure(statistic="fdr_bh", family=["drug", "plate"])
    assert p.family(spec).columns == ("drug", "plate") and p.family(spec).method == "bh"
    assert p.sort_key("padj", spec, None).within == ("drug", "plate")
    got = p.adjust([0.01, 0.02, None], 2)
    assert got[:2] == pytest.approx([0.02, 0.02]) and got[2] is None
    full = p.adjust([0.01, 0.02], 10)                  # eight untested members in the family
    assert full == pytest.approx([0.1, 0.1])
    with pytest.raises(UnsupportedFilter):
        p.adjust([0.01, 0.02, 0.03], 2)
    with pytest.raises(UnsupportedFilter):
        p.adjust([1.5], 1)
    pred = p.predicate("padj", "significant", 0.05, spec, {"confirmed": True}, {})
    assert evaluate(pred, {"padj": 0.05}) is True and evaluate(pred, {"padj": 0.0500001}) is False
    assert "BH family: drug, plate" in p.describe(spec)


# ---------------------------------------------------------------------------- effects


def test_effects_rank_by_magnitude_and_refuse_meaningless_ops(reg) -> None:
    from vbt.datalayer.plugins.statistics import order_rows, rank_value_of

    for name in ("log2fc", "wald", "effect_beta"):
        p = plugin(reg, name)
        spec = measure(statistic=name)
        rk = p.sort_key("v", spec, None)
        assert rk.direction == "desc_abs"
        rows = order_rows([{"v": 1.0, "k": "a"}, {"v": -3.0, "k": "b"}, {"v": None, "k": "c"}], [(rk, p, spec)],
                          tie_key=lambda r: r["k"])
        assert [r["k"] for r in rows] == ["b", "a", "c"]
    wald = plugin(reg, "wald")
    pred = wald.predicate("z", "significant", None, measure(statistic="wald"), None, {})
    assert pred == CmpAbs("z", "ge_abs", pytest.approx(1.959963984540054))
    assert evaluate(pred, {"z": -1.96}) is True and evaluate(pred, {"z": 1.9}) is False
    se = plugin(reg, "std_error")
    with pytest.raises(UnsupportedFilter):
        se.predicate("se", "gt_abs", 0.1, measure(statistic="std_error"), None, {})
    orr = plugin(reg, "odds_ratio")
    spec = measure(statistic="odds_ratio")
    assert rank_value_of(orr, {"v": 0.5}, "v", spec) == pytest.approx(-math.log(2))
    assert orr.aggregate([2.0, 8.0], "mean", spec).value == pytest.approx(4.0), "geometric mean"
    with pytest.raises(UnsupportedFilter):
        orr.predicate("v", "gt_abs", 2.0, spec, None, {})


# ---------------------------------------------------------------------------- gene_effect


def test_gene_effect_cutoff_and_inclusivity(reg) -> None:
    p = plugin(reg, "gene_effect")
    inclusive = measure(statistic="gene_effect", cutoff={"value": -0.5, "op": "le", "meaning": "dependency"})
    exclusive = measure(statistic="gene_effect", cutoff={"value": -0.5, "op": "lt"})
    pi = p.predicate("ge", "significant", None, inclusive, None, {})
    pe = p.predicate("ge", "essential", None, exclusive, None, {})
    assert pi == Cmp("ge", "<=", -0.5) and pe == Cmp("ge", "<", -0.5)
    assert evaluate(pi, {"ge": -0.5}) is True and evaluate(pe, {"ge": -0.5}) is False
    assert "cutoff ≤ -0.5 (inclusive): dependency" in p.describe(inclusive)
    assert "(exclusive)" in p.describe(exclusive)
    res = p.aggregate([-1.0, -0.5, 0.2, None], "essential_fraction", inclusive)
    assert res.value == pytest.approx(2 / 3) and res.n == 3 and res.n_excluded == 1
    assert p.aggregate([None, None], "essential_fraction", inclusive).value is None, "no effects: None, never 0"
    with pytest.raises(UnsupportedFilter):
        p.aggregate([-1.0], "essential_fraction", measure(statistic="gene_effect"))
    with pytest.raises(UnsupportedFilter):
        p.predicate("ge", "significant", None, measure(statistic="gene_effect"), None, {})


# ---------------------------------------------------------------------------- colocalisation


def test_coloc_methods_are_never_pooled(reg) -> None:
    h4, clpp = plugin(reg, "coloc_h4"), plugin(reg, "clpp")
    spec = measure(statistic="coloc_h4")
    assert h4.aggregate([0.9, 0.8], "max", spec, groups=[{"method": "coloc"}, {"method": "coloc"}]).value == 0.9
    with pytest.raises(UnsupportedFilter):
        h4.aggregate([0.9, 0.02], "max", spec, groups=[{"method": "coloc"}, {"method": "ecaviar"}])
    with pytest.raises(UnsupportedFilter):
        clpp.aggregate([0.02], "max", measure(statistic="clpp"), groups=[{"method": "coloc"}])
    assert evaluate(h4.predicate("h4", "significant", None, spec, None, {}), {"h4": 0.8}) is True
    assert "coloc posterior" in h4.describe(spec) and "ecaviar posterior" in clpp.describe(measure(statistic="clpp"))


# ---------------------------------------------------------------------------- expression


def test_tpm_unit_guard_and_log2_intensity_cohort(reg) -> None:
    tpm = plugin(reg, "tpm")
    spec = measure(statistic="tpm", unit_from="unit", missing_values=[999.9])
    assert tpm.comparable({"unit": "TPM"}, {"unit": "TPM"}, spec) is True
    assert tpm.comparable({"unit": "TPM"}, {"unit": "FPKM"}, spec) is False
    assert tpm.comparable({"unit": ""}, {"unit": ""}, spec) is False, "a blank unit is not 'the same unit'"
    with pytest.raises(UnsupportedFilter):
        tpm.aggregate([1.0, 2.0], "mean", spec, groups=[{"unit": "TPM"}, {"unit": ""}])
    with pytest.raises(UnsupportedFilter):
        tpm.aggregate([1.0, 2.0], "mean", spec, groups=[{"unit": "TPM"}, {"unit": "FPKM"}])
    assert tpm.aggregate([1.0, 999.9, 3.0], "mean", spec, groups=[{"unit": "TPM"}] * 3).value == 2.0
    with pytest.raises(UnsupportedFilter):
        tpm.predicate("v", "ge", 1.0, spec, None, {})       # the unit is not fixed
    pred = tpm.predicate("v", "ge", 1.0, spec, None, {"unit": "TPM"})
    assert evaluate(pred, {"v": 999.9}) is not True, "the in-band code never passes"
    li = plugin(reg, "log2_intensity")
    plain = measure(statistic="log2_intensity")
    assert li.sort_key("x", plain, None).within == ("cohort",)
    with pytest.raises(UnsupportedFilter):
        li.predicate("x", "ge", 8.0, plain, None, {})
    li.predicate("x", "ge", 8.0, plain, None, {"cohort": "GSE12251"})
    assert li.comparable({"cohort": "A"}, {"cohort": "B"}, plain) is False


# ---------------------------------------------------------------------------- cosine


def test_cosine_vetoes_interpretation_labels(reg) -> None:
    from vbt.datalayer.plugins.statistics.similarity import cosine

    p = plugin(reg, "cosine")
    assert p.vetoed_companions(measure(statistic="cosine")) == ("interpretation",)
    with pytest.raises(UnsupportedFilter):
        p.predicate("s", "gt", 1.5, measure(statistic="cosine"), None, {})
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0 and cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) is None and cosine([float("nan"), 1.0], [1.0, 0.0]) is None


# ---------------------------------------------------------------------------- set tests


def test_set_tests_match_scipy_on_tables(reg) -> None:
    sps = pytest.importorskip("scipy.stats")
    hg, fi = plugin(reg, "hypergeom_enrichment"), plugin(reg, "fisher")
    for k, K, n, N in [(3, 5, 4, 20), (0, 3, 2, 10), (7, 30, 12, 400), (2, 2, 2, 2), (1, 1, 1, 5000)]:
        assert hg.test(k, K, n, N) == pytest.approx(float(sps.hypergeom.sf(k - 1, N, K, n)), rel=1e-9)
        table = [[k, n - k], [K - k, N - K - n + k]]
        for alt, direction in (("two-sided", "none"), ("greater", "higher_is_stronger"), ("less", "lower_is_stronger")):
            got = fi.test(k, K, n, N, {"direction": direction})
            assert got == pytest.approx(float(sps.fisher_exact(table, alternative=alt)[1]), rel=1e-7), (k, K, n, N, alt)
    with pytest.raises(UnsupportedFilter):
        hg.test(3, 2, 5, 10)


# ---------------------------------------------------------------------------- survival


def test_survival_thresholds_are_censored_comparisons(reg) -> None:
    p = plugin(reg, "survival_time")
    spec = measure(statistic="survival_time", columns={"event": "os_event"})
    pred = p.predicate("os_months", "le", 12.0, spec, None, {})
    assert pred == CensoredCmp("os_months", "os_event", "<=", 12.0)
    assert evaluate(pred, {"os_months": 10.0, "os_event": True}) is True
    assert evaluate(pred, {"os_months": 10.0, "os_event": False}) is None, "censored before t: unknown"
    assert evaluate(pred, {"os_months": 20.0, "os_event": False}) is False
    with pytest.raises(UnsupportedFilter):
        p.predicate("os_months", "le", 12.0, measure(statistic="survival_time"), None, {})
    with pytest.raises(UnsupportedFilter):
        p.aggregate([1.0, 2.0], "mean", spec)
    km = p.aggregate_pair([1, 2, 3, 4], [1, 0, 1, 1], "km", spec)
    assert km.value == [[1.0, 0.75], [3.0, 0.375], [4.0, 0.0]]
    assert p.aggregate_pair([1, 2, 3, 4], ["1:DECEASED", "0:LIVING", "1:DECEASED", "1:DECEASED"], "median",
                            spec).value == 3.0

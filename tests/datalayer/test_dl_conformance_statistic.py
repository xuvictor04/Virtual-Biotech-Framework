"""Statistic conformance suite (§9.3 S-1..S-6, S-8) over the registered statistic plugins, plus
explicit cases: encoding validation from distinct-value snapshots, the clinical phase scale, factor
labels in positive form only (CT-6), composite p-values, bounds, missing zero and sibling guards."""

from __future__ import annotations

import math
import random
from decimal import Decimal

import pytest

from vbt.datalayer.descriptor.columns import validate_column
from vbt.datalayer.plugins.base import ColumnStats, UnsupportedFilter, ValueSnapshot
from vbt.datalayer.plugins.conformance.statistic import *  # noqa: F401,F403  (collects S-1..S-6, S-8)
from vbt.datalayer.plugins.registry import discover
from vbt.datalayer.plugins.statistics import order_rows
from vbt.datalayer.plugins.statistics.count import CountStatistic
from vbt.datalayer.plugins.statistics.factor import BinaryFactorStatistic, SignedFactorStatistic
from vbt.datalayer.plugins.statistics.numeric import NumericStatistic, PercentStatistic
from vbt.datalayer.plugins.statistics.ordinal import ClinicalPhaseStatistic, OrdinalStatistic
from vbt.datalayer.plugins.statistics.pvalue import PValueMantissaExponentStatistic, fdr_bh, split_pvalue
from vbt.datalayer.plugins.statistics.score import Score01Statistic
from vbt.datalayer.predicate import And, Any, Cmp, In, evaluate, merge_container_predicates
from vbt.datalayer.rowkey import canonical


def measure(**facets):
    return validate_column({"role": "measure", **facets})


def stats(mn=None, mx=None, storage="double"):
    return ColumnStats(uncompressed_bytes=0, null_count=0, min=mn, max=mx, storage_type=storage)


def test_phase1_statistics_are_registered():
    names = discover(entry_points=False).names("statistic")
    for name in ("numeric", "percent", "score_0_1", "score_0_100", "ordinal", "clinical_phase", "signed_factor",
                 "binary_factor", "pvalue", "pvalue_mantissa_exponent", "llr_critval", "count"):
        assert name in names
    assert PercentStatistic().declared_scale(measure(statistic="percent")) == (0.0, 100.0)
    assert PercentStatistic().declared_scale(measure(statistic="percent", scale=[0, 1])) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Encodings are confirmed only from a distinct-value snapshot
# ---------------------------------------------------------------------------

def test_encoding_validation_fails_when_a_code_is_never_observed():
    plugin = SignedFactorStatistic()
    spec = measure(statistic="signed_factor", encoding={-1: "has_safety_event", 0: "none_recorded", 1: None},
                   verified=False)
    never = plugin.validate(stats(-1, 0), ValueSnapshot(values=(0, None), null_count=4), spec)
    assert never.confirmed is False and "-1" in never.detail and "never observed" in never.detail
    assert never.facts["confirmed"] is False
    ok = plugin.validate(stats(-1, 1), ValueSnapshot(values=(-1, 0, 1, None)), spec)
    assert ok.confirmed is True and ok.facts["codes"] == [-1, 0]
    extra = plugin.validate(stats(-1, 2), ValueSnapshot(values=(-1, 0, 2)), spec)
    assert extra.confirmed is False and "not declared codes" in extra.detail
    # min/max alone never confirm an encoding; neither does a partial snapshot
    assert plugin.validate(stats(-1, 0), None, spec).confirmed is None
    assert plugin.validate(stats(-1, 0), ValueSnapshot(values=(-1, 0), complete=False), spec).confirmed is None
    # NaN is counted, not mistaken for a code
    nan = plugin.validate(stats(-1, 0), ValueSnapshot(values=(-1, 0, float("nan")), nan_count=2), spec)
    assert nan.confirmed is True and nan.facts["nan_count"] == 3
    with pytest.raises(UnsupportedFilter) as err:
        plugin.predicate("hasSafetyEvent", "in", ["none_recorded"], spec, never.facts, {})
    assert err.value.reason == "unconfirmed_encoding"
    plugin.predicate("hasSafetyEvent", "in", ["none_recorded"], spec, ok.facts, {})


def test_ordinal_encoding_and_levels_validation():
    plugin = OrdinalStatistic()
    enc = measure(statistic="ordinal", encoding={0: "none", 1: "low", 2: "high"})
    assert plugin.validate(stats(0, 1), ValueSnapshot(values=(0, 1)), enc).confirmed is False
    assert plugin.validate(stats(0, 2), ValueSnapshot(values=(0, 1, 2)), enc).confirmed is True
    levels = measure(statistic="ordinal", levels=["1A", "1B", "2A"])
    assert plugin.validate(stats(), ValueSnapshot(values=("1A", "2A")), levels).confirmed is True
    assert plugin.validate(stats(), ValueSnapshot(values=("1A", "3")), levels).confirmed is False
    assert plugin.validate(stats(), None, levels).confirmed is None


# ---------------------------------------------------------------------------
# Factors: positive forms only (CT-6)
# ---------------------------------------------------------------------------

def test_factor_labels_accept_only_positive_forms():
    plugin = SignedFactorStatistic()
    spec = measure(statistic="signed_factor", encoding={-1: "has_safety_event", 0: "none_recorded"})
    rows = [{"t": "A", "hasSafetyEvent": -1}, {"t": "B", "hasSafetyEvent": None}, {"t": "C", "hasSafetyEvent": 0},
            {"t": "D", "hasSafetyEvent": float("nan")}]
    pred = plugin.predicate("hasSafetyEvent", "in", ["none_recorded"], spec, None, {})
    assert [r["t"] for r in rows if evaluate(pred, r) is True] == ["C"]       # upstream `!= 1` returned A, B, C
    for op, value in (("ne", "has_safety_event"), ("lt", 0), ("in", ["unknown"])):
        with pytest.raises(UnsupportedFilter) as err:
            plugin.predicate("hasSafetyEvent", op, value, spec, None, {})
    assert err.value.reason == "invalid_value"
    with pytest.raises(UnsupportedFilter) as err:
        plugin.predicate("hasSafetyEvent", "ne", "has_safety_event", spec, None, {})
    assert err.value.reason == "negated_form"
    with pytest.raises(UnsupportedFilter):                  # a refuted encoding never filters
        plugin.predicate("hasSafetyEvent", "in", [0], spec, {"confirmed": False}, {})


def test_binary_factor_matches_numeric_and_boolean_spellings():
    plugin = BinaryFactorStatistic()
    spec = measure(statistic="binary_factor")
    pred = plugin.predicate("isInMembrane", "eq", "yes", spec, None, {})
    assert evaluate(pred, {"isInMembrane": 1}) and evaluate(pred, {"isInMembrane": True})
    assert evaluate(pred, {"isInMembrane": 1.0}) and not evaluate(pred, {"isInMembrane": 0})
    assert evaluate(pred, {"isInMembrane": None}) is None
    assert evaluate(plugin.predicate("isInMembrane", "eq", False, spec, None, {}), {"isInMembrane": 0}) is True


# ---------------------------------------------------------------------------
# Scales: clinical phase, bounds, comparable_within
# ---------------------------------------------------------------------------

def test_clinical_phase_scale_from_confirmed_range():
    plugin = ClinicalPhaseStatistic()
    spec = measure(statistic="clinical_phase", missing_values=[-1])
    with pytest.raises(UnsupportedFilter) as err:
        plugin.predicate("phase", "ge", 3, spec, None, {})
    assert err.value.reason == "scale"
    ot = plugin.validate(stats(0.5, 4.0), None, spec)
    assert ot.confirmed is True and ot.facts["scale"] == [0.0, 4.0]
    norm = plugin.validate(stats(0.0, 1.0), None, spec)
    assert norm.facts["scale"] == [0.0, 1.0]
    with pytest.raises(UnsupportedFilter):                  # phase 3 on a normalised 0-1 column
        plugin.predicate("phase", "ge", 3, spec, norm.facts, {})
    pred = plugin.predicate("phase", "ge", 3, spec, ot.facts, {})
    assert [evaluate(pred, {"phase": v}) for v in (4.0, 3.0, 2.0, -1, None)] == [True, True, False, False, None]
    # ChEMBL -1 hides the real minimum in min/max; a snapshot decides
    hidden = plugin.validate(stats(-1, 4.0), None, spec)
    assert hidden.confirmed is None
    snap = plugin.validate(stats(-1, 4.0), ValueSnapshot(values=(-1, 0.5, 2.0, 4.0)), spec)
    assert snap.confirmed is True and snap.facts["scale"] == [0.0, 4.0]
    declared = measure(statistic="clinical_phase", scale=[0, 4])
    plugin.predicate("phase", "ge", 3, declared, None, {})


def test_bounds_from_scale_tightened_by_constraints():
    from vbt.datalayer.descriptor.models import ConstraintSpec

    padj = ConstraintSpec(column="padj", op="<", value=0.10, origin="tools/prepare_tahoe.py:72")
    assert NumericStatistic().bounds(measure(scale=[0, 1]), [padj]) == {
        "minimum": 0.0, "exclusiveMaximum": 0.1, "constraints": ["tools/prepare_tahoe.py:72"]}
    assert Score01Statistic().bounds(measure(statistic="score_0_1"), []) == {"minimum": 0.0, "maximum": 1.0}
    assert CountStatistic().bounds(validate_column({"role": "count"}), [{"op": ">=", "value": 1, "origin": "x"}]) \
        == {"minimum": 1.0, "constraints": ["x"]}
    assert NumericStatistic().bounds(measure(), []) == {}


def test_comparable_within_guard_accepts_scope_reference_forms():
    plugin = NumericStatistic()
    spec = measure(comparable_within=["^.unit"], direction="higher_is_stronger")
    with pytest.raises(UnsupportedFilter) as err:
        plugin.predicate("value", "gt", 1, spec, None, {"unit": None})
    assert err.value.group == ("^.unit",)
    plugin.predicate("value", "gt", 1, spec, None, {"/unit": "uM"})
    assert plugin.sort_key("value", spec, None).within == ("^.unit",)


# ---------------------------------------------------------------------------
# Unknown handling: missing zero, sibling unknown_when, signed order
# ---------------------------------------------------------------------------

def test_missing_zero_includes_nulls_only_when_zero_passes():
    plugin = CountStatistic()
    spec = validate_column({"role": "count", "missing": "zero"})
    le = plugin.predicate("n", "le", 2, spec, None, {})
    ge = plugin.predicate("n", "ge", 1, spec, None, {})
    assert evaluate(le, {"n": None}) is True and evaluate(ge, {"n": None}) is False
    assert plugin.aggregate([None, 3], "sum", spec).value == 3 and plugin.aggregate([None], "sum", spec).n == 1


def test_unknown_when_on_a_nested_measure_is_one_correlated_guard():
    plugin = NumericStatistic()
    spec = measure(unknown_when=[{"column": "unit", "eq": ""}], missing_values=[-1])
    pred = plugin.predicate("tissues[].value", "gt", 5, spec, None, {})
    assert isinstance(pred, Any) and pred.path == "tissues[]"
    assert pred.pred == And((Cmp("value", ">", 5), Cmp("value", "!=", -1), Cmp("unit", "!=", "")))
    row = {"tissues": [{"value": 9, "unit": ""}, {"value": 1, "unit": "TPM"}]}
    assert evaluate(pred, row) is not True                  # the strong value has no unit: unknown
    assert evaluate(pred, {"tissues": [{"value": 9, "unit": "TPM"}]}) is True
    assert merge_container_predicates([pred]) == [pred]


def test_signed_measures_rank_by_magnitude_ties_by_key():
    plugin = NumericStatistic()
    spec = measure(direction="signed")
    rk = plugin.sort_key("lfc", spec, None)
    assert rk.direction == "desc_abs"
    rows = [{"k": "b", "lfc": 2.0}, {"k": "a", "lfc": -2.0}, {"k": "c", "lfc": float("nan")}, {"k": "d", "lfc": 0.1}]
    ordered = order_rows(rows, [(rk, plugin, spec)], tie_key=lambda r: canonical([r["k"]]))
    assert [r["k"] for r in ordered] == ["a", "b", "d", "c"]
    asc = order_rows(rows, [(plugin.sort_key("lfc", spec, "asc"), plugin, spec)], tie_key=lambda r: r["k"])
    assert [r["k"] for r in asc] == ["a", "d", "b", "c"]


def test_count_integrality_from_snapshots():
    plugin = CountStatistic()
    spec = validate_column({"role": "count", "stored_as": "float64"})
    assert plugin.validate(stats(0.0, 7.0), None, spec).confirmed is None
    assert plugin.validate(stats(0.0, 7.0), ValueSnapshot(values=(0.0, 3.0, 7.0)), spec).facts["integral"] is True
    bad = plugin.validate(stats(0.0, 7.0), ValueSnapshot(values=(0.0, 2.5)), spec)
    assert bad.confirmed is False and "fractional" in bad.detail
    assert plugin.validate(stats(-1, 3, "int64"), None, validate_column({"role": "count"})).confirmed is False
    keys = plugin.aggregate([2, 2, 5], "sum", spec, keys=["drugA", "drugA", "drugB"])
    assert (keys.value, keys.n) == (7.0, 2)


# ---------------------------------------------------------------------------
# P-values
# ---------------------------------------------------------------------------

def test_composite_pvalue_thresholds_match_exact_arithmetic():
    plugin = PValueMantissaExponentStatistic()
    spec = measure(statistic="pvalue_mantissa_exponent", columns={"mantissa": "m", "exponent": "e"})
    rng = random.Random(3)
    rows = [{"m": round(rng.uniform(1, 9.99), 2), "e": rng.randint(-330, 0)} for _ in range(300)]
    rows += [{"m": 0.0, "e": 0}, {"m": 5.0, "e": -8}, {"m": None, "e": -8}, {"m": 5.0, "e": None},
             {"m": 50.0, "e": -9}]
    for t in (5e-8, 1e-300, 0.05, 1.0, (2.5, -320), 5e-8):
        tm, te = split_pvalue(t)
        limit = Decimal(repr(tm)).scaleb(te)
        for op in ("lt", "le", "gt", "ge", "eq"):
            pred = plugin.predicate("p", op, t, spec, None, {})
            for r in rows:
                got = evaluate(pred, r)
                if r["m"] is None or r["e"] is None or not (r["m"] == 0 or 1 <= r["m"] < 10):
                    assert got is not True, (op, t, r)
                    continue
                p = Decimal(repr(r["m"])).scaleb(r["e"])
                want = {"lt": p < limit, "le": p <= limit, "gt": p > limit, "ge": p >= limit, "eq": p == limit}[op]
                assert got is want, (op, t, r)
    assert split_pvalue(5e-8) == (5.0, -8) and split_pvalue({"mantissa": 25, "exponent": -9}) == (2.5, -8)
    assert plugin.rank_value((3.0, -400), spec) < plugin.rank_value((1.0, -300), spec)


def test_fdr_bh_reference_values():
    q = fdr_bh([0.01, 0.04, 0.03, None, 0.20])
    assert q[3] is None
    assert [round(x, 6) for x in (q[0], q[1], q[2], q[4])] == [0.04, 0.053333, 0.053333, 0.2]
    larger = fdr_bh([0.01, 0.04], m=4)                       # the declared family includes untested members
    assert [round(x, 6) for x in larger] == [0.04, 0.08]
    with pytest.raises(ValueError):
        fdr_bh([0.1, 0.2, 0.3], m=2)
    assert all(math.isclose(a, b) for a, b in zip(fdr_bh([0.5]), [0.5]))


def test_aggregation_refuses_meaningless_hows_and_counts_levels_once():
    with pytest.raises(UnsupportedFilter):
        PValueMantissaExponentStatistic().aggregate([0.1], "mean", measure(statistic="pvalue"))
    with pytest.raises(UnsupportedFilter):
        OrdinalStatistic().aggregate(["1A"], "mean", measure(statistic="ordinal", levels=["1A", "1B"]))
    res = OrdinalStatistic().aggregate(["1B", "1A", "1B", None], "max", measure(statistic="ordinal",
                                                                               levels=["1A", "1B"]),
                                       keys=["x", "y", "x", "z"])
    assert (res.value, res.n, res.n_excluded) == ("1B", 2, 1)
    conflicted = NumericStatistic().aggregate([1.0, 2.0], "max", measure(), keys=["k", "k"])
    assert conflicted.n == 1 and "conflicting" in conflicted.detail
    assert In("x", ()) == In("x", ())

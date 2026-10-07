"""Statistic conformance suite (§9.3, S-1..S-9), parametrized over the registered statistic plugins.

Each plugin's ``conformance_cases()`` returns :class:`~.golden.StatisticCase` objects (a plugin
without cases is run on a plain numeric spec) plus, for the phase-3 checks, the case kinds defined
here: :class:`FamilyCase` (S-7: ``adjust`` over the full declared family equals a reference BH,
statsmodels when installed, else :func:`reference_bh`), :class:`SetTestCase` (S-9, capability
``test``: hand-computed set tests, and scipy when installed) and :class:`PairedCase` (S-9,
capability ``paired``: Kaplan-Meier medians and event counts against :func:`reference_km_median`,
means of censored times refused). The suite never needs pyarrow.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import pytest

from ...predicate import evaluate
from ...rowkey import canonical
from ..base import UnsupportedFilter
from ..statistics import order_rows, rank_value_of, spec_get
from . import for_plugin, selected_plugins
from .golden import StatisticCase

STATISTIC_PLUGINS = selected_plugins("statistic")


@dataclass(frozen=True)
class FamilyCase:
    """S-7: p-values of one family (``None``/NaN: not tested) and the declared family size ``m``
    (``None``: the known p-values), which counts zero-overlap members that have no p-value here."""

    name: str
    pvalues: tuple[Any, ...]
    m: int | None = None


@dataclass(frozen=True)
class SetTestCase:
    """S-9 (capability ``test``): ``expected`` is a hand-computed p-value, ``None`` to compare with
    scipy only, or ``"refused"`` when the counts are inconsistent."""

    name: str
    overlap: int
    set_n: int
    query_n: int
    universe_n: int
    expected: Any = None
    alternative: str = "greater"
    requires: tuple[str, ...] = ("test",)


@dataclass(frozen=True)
class PairedCase:
    """S-9 (capability ``paired``): times and events (``keys``: the level key of each row)."""

    name: str
    times: tuple[Any, ...]
    events: tuple[Any, ...]
    keys: tuple[Any, ...] | None = None
    expected_n: int | None = None
    requires: tuple[str, ...] = ("paired",)
BOUNDARY_OPS = {"lt": False, "le": True, "gt": False, "ge": True}
DIRECTION_WORDS = ("higher is stronger", "lower is stronger", "signed", "no direction")


def statistic_cases(plugin: Any) -> list[StatisticCase]:
    cases = [c for c in (plugin.conformance_cases() or ()) if isinstance(c, StatisticCase)]
    if not cases:
        cases = [StatisticCase(plugin.name, {"role": "measure", "statistic": plugin.name}, values=(1.0, 3.0, 2.0),
                               thresholds=(("ge", 2.0),))]
    return for_plugin(plugin, cases)


CASE_PARAMS = [pytest.param(p, c, id=f"{p.name}-{c.name}") for p in STATISTIC_PLUGINS for c in statistic_cases(p)]


def _unknowns(plugin: Any, case: StatisticCase, spec: Any) -> list[dict[str, Any]]:
    codes = list(spec_get(spec, "missing_values", []) or []) + list(getattr(plugin, "default_missing", ()) or ())
    rows = [case.make_row(u) for u in (*case.unknown, *codes)]
    if spec_get(spec, "missing", None) == "zero":
        rows = [r for r in rows if not all(v is None or (isinstance(v, float) and math.isnan(v)) for v in r.values())]
    return rows + [dict(r) for r in case.unknown_rows]


def _pred(plugin: Any, case: StatisticCase, op: str, value: Any, spec: Any = None, *, confirmed: Any = "case",
          fixed: Any = None) -> Any:
    spec = spec if spec is not None else case.column_spec()
    return plugin.predicate(case.column, op, value, spec, case.confirmed if confirmed == "case" else confirmed,
                            case.fixed_scope if fixed is None else fixed)


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "key"}


# ---------------------------------------------------------------------------
# S-1 total, deterministic order; nulls last
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin,case", CASE_PARAMS)
def test_s1_sort_key_total_order(plugin: Any, case: StatisticCase) -> None:
    spec = case.column_spec()
    rk = plugin.sort_key(case.column, spec, None)
    assert rk.nulls == "last" and rk.column == case.column
    rows = [case.make_row(v, key=f"k{i:02d}") for i, v in enumerate(case.values)]
    unknown = [dict(r, key=f"u{i:02d}") for i, r in enumerate(_unknowns(plugin, case, spec))]
    keys = [(rk, plugin, spec)]
    tie = lambda r: canonical([r["key"]])  # noqa: E731
    reference = order_rows(rows + unknown, keys, tie_key=tie)
    rng = random.Random(5)
    for _ in range(20):
        shuffled = rows + unknown
        rng.shuffle(shuffled)
        assert order_rows(shuffled, keys, tie_key=tie) == reference
    ranks = [rank_value_of(plugin, r, case.column, spec) for r in reference]
    known = [r for r in ranks if r is not None]
    assert ranks[: len(known)] == known and all(r is None for r in ranks[len(known):]), "unknown values sort last"
    assert len(known) == len(rows), "every known value ranks"
    if case.best is not None:
        assert _strip(reference[0]) == _strip(case.make_row(case.best)), "the strongest value comes first"


# ---------------------------------------------------------------------------
# S-2 unknown never passes; boundaries follow cutoff inclusivity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin,case", CASE_PARAMS)
def test_s2_unknown_never_passes(plugin: Any, case: StatisticCase) -> None:
    spec = case.column_spec()
    ops = list(case.thresholds)
    cutoff = spec_get(spec, "cutoff", None)
    if cutoff is not None:
        ops.append((cutoff.op, cutoff.value))
    assert ops, f"{case.name}: no thresholds to check"
    unknown = _unknowns(plugin, case, spec)
    for op, value in ops:
        pred = _pred(plugin, case, op, value, spec)
        for row in unknown:
            assert evaluate(pred, row) is not True, f"{op} {value!r} passed unknown row {row}"
        if op in BOUNDARY_OPS:
            assert evaluate(pred, case.make_row(value)) is BOUNDARY_OPS[op], f"boundary {op} {value!r}"
    if cutoff is not None:
        text = plugin.describe(spec)
        assert ("inclusive" if cutoff.op in ("le", "ge") else "exclusive") in text, text


# ---------------------------------------------------------------------------
# S-3 out-of-scale thresholds and unconfirmed facts are refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin,case", CASE_PARAMS)
def test_s3_refused_thresholds(plugin: Any, case: StatisticCase) -> None:
    spec = case.column_spec()
    for op, value in case.refused:
        with pytest.raises(UnsupportedFilter):
            _pred(plugin, case, op, value, spec)
    if not case.unconfirmed:
        return
    unverified = case.column_spec(verified=False)
    for op, value in case.unconfirmed:
        with pytest.raises(UnsupportedFilter) as err:
            _pred(plugin, case, op, value, unverified, confirmed=None)
        assert err.value.reason in ("scale", "unconfirmed_encoding"), err.value.reason
        _pred(plugin, case, op, value, unverified)          # confirmed facts lift the refusal


# ---------------------------------------------------------------------------
# S-4 aggregation: empty -> None, level keys counted once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin,case", CASE_PARAMS)
def test_s4_aggregate_empty_and_level_keys(plugin: Any, case: StatisticCase) -> None:
    spec = case.column_spec()
    known = [v for v in case.values if v is not None]
    for how in case.agg:
        assert plugin.aggregate([], how, spec).value is None
        if spec_get(spec, "missing", None) != "zero":
            assert plugin.aggregate([None], how, spec).value is None
            assert plugin.aggregate([None, float("nan")], how, spec).value is None
        a, b = known[0], known[-1]
        res = plugin.aggregate([a, a, b, None], how, spec, keys=["k1", "k1", "k2", "k3"])
        zero = spec_get(spec, "missing", None) == "zero"
        assert res.n == (3 if zero else 2), "each level key counts once"
        assert res.n_excluded == (0 if zero else 1)
        assert res.value is not None


# ---------------------------------------------------------------------------
# S-5 comparable_within groups; S-8 thresholds need the group fixed
# ---------------------------------------------------------------------------

def _group_spec(case: StatisticCase) -> Any:
    assert case.group is not None
    within = list(case.spec.get("comparable_within") or [])
    if case.group[0] not in within:
        within.append(case.group[0])
    return case.column_spec(comparable_within=within)


@pytest.mark.parametrize("plugin,case", [p for p in CASE_PARAMS if p.values[1].group])
def test_s5_comparable_within(plugin: Any, case: StatisticCase) -> None:
    col, a, b = case.group  # type: ignore[misc]
    spec = _group_spec(case)
    assert plugin.comparable({col: a}, {col: a}, spec) is True
    assert plugin.comparable({col: a}, {col: b}, spec) is False
    assert plugin.comparable({col: a}, {col: None}, spec) is False
    known = [v for v in case.values if v is not None]
    values = [known[0], known[-1]]
    how = case.agg[0]
    assert plugin.aggregate(values, how, spec, groups=[{col: a}, {col: a}]).n == 2
    with pytest.raises(UnsupportedFilter):
        plugin.aggregate(values, how, spec, groups=[{col: a}, {col: b}])


@pytest.mark.parametrize("plugin,case", [p for p in CASE_PARAMS if p.values[1].group])
def test_s8_threshold_needs_group_fixed(plugin: Any, case: StatisticCase) -> None:
    col, a, _b = case.group  # type: ignore[misc]
    spec = _group_spec(case)
    op, value = case.thresholds[0]
    with pytest.raises(UnsupportedFilter) as err:
        _pred(plugin, case, op, value, spec, fixed={})
    assert err.value.reason == "group" and col in err.value.group
    with pytest.raises(UnsupportedFilter):
        _pred(plugin, case, op, value, spec, fixed={col: [a, _b]})   # two groups are not one group
    _pred(plugin, case, op, value, spec, fixed={**case.fixed_scope, col: a})


# ---------------------------------------------------------------------------
# S-6 describe names direction and scale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin,case", CASE_PARAMS)
def test_s6_describe(plugin: Any, case: StatisticCase) -> None:
    text = plugin.describe(case.column_spec())
    assert isinstance(text, str) and text
    assert any(w in text for w in DIRECTION_WORDS), text
    assert "scale" in text, text


# ---------------------------------------------------------------------------
# S-7 BH over the full declared family
# ---------------------------------------------------------------------------

def reference_bh(pvalues: list[float], m: int) -> list[float]:
    """Textbook BH (step-up, monotone, capped at 1) over ``m`` tests, of which ``pvalues`` are given."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [0.0] * len(pvalues)
    running = 1.0
    for rank in range(len(order), 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        out[i] = min(1.0, running)
    return out


def _reference_family(known: list[float], m: int) -> list[float]:
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:                                # statsmodels is optional
        return reference_bh(known, m)
    padded = list(known) + [1.0] * (m - len(known))    # untested members of the family: p = 1
    return [float(q) for q in multipletests(padded, method="fdr_bh")[1][: len(known)]]


def _cases_of(plugin: Any, kind: type) -> list[Any]:
    return [c for c in for_plugin(plugin, plugin.conformance_cases() or ()) if isinstance(c, kind)]


FAMILY_PARAMS = [pytest.param(p, c, id=f"{p.name}-{c.name}") for p in STATISTIC_PLUGINS for c in _cases_of(p, FamilyCase)]
SET_TEST_PARAMS = [pytest.param(p, c, id=f"{p.name}-{c.name}") for p in STATISTIC_PLUGINS
                   for c in _cases_of(p, SetTestCase)]
PAIRED_PARAMS = [pytest.param(p, c, id=f"{p.name}-{c.name}") for p in STATISTIC_PLUGINS for c in _cases_of(p, PairedCase)]


@pytest.mark.parametrize("plugin,case", FAMILY_PARAMS)
def test_s7_bh_over_the_full_family(plugin: Any, case: FamilyCase) -> None:
    pvalues = list(case.pvalues)
    known_idx = [i for i, p in enumerate(pvalues) if p is not None and not (isinstance(p, float) and math.isnan(p))]
    known = [float(pvalues[i]) for i in known_idx]
    m = case.m if case.m is not None else len(known)
    got = plugin.adjust(pvalues, case.m)
    assert len(got) == len(pvalues)
    assert all(got[i] is None for i in range(len(pvalues)) if i not in known_idx), "untested members stay None"
    want = _reference_family(known, m)
    for i, q in zip(known_idx, want):
        assert got[i] == pytest.approx(q, rel=1e-12, abs=1e-15), (case.name, i)
    if case.m is not None and case.m > len(known):
        smaller = plugin.adjust(pvalues, None)
        assert all(smaller[i] <= got[i] for i in known_idx), "a larger family never lowers a q-value"
        assert any(smaller[i] < got[i] for i in known_idx if got[i] < 1.0) or all(got[i] == 1.0 for i in known_idx)
    with pytest.raises((UnsupportedFilter, ValueError)):
        plugin.adjust(pvalues, max(0, len(known) - 1))  # a family smaller than the tests given is refused


# ---------------------------------------------------------------------------
# S-9 set tests match scipy; paired survival matches Kaplan-Meier and refuses means
# ---------------------------------------------------------------------------

_ALTERNATIVE_DIRECTION = {"greater": "higher_is_stronger", "less": "lower_is_stronger", "two-sided": "none"}


@pytest.mark.parametrize("plugin,case", SET_TEST_PARAMS)
def test_s9_set_test_matches_reference(plugin: Any, case: SetTestCase) -> None:
    spec = {"role": "measure", "statistic": plugin.name, "direction": _ALTERNATIVE_DIRECTION[case.alternative]}
    args = (case.overlap, case.set_n, case.query_n, case.universe_n, spec)
    if case.expected == "refused":
        with pytest.raises(UnsupportedFilter):
            plugin.test(*args)
        return
    got = plugin.test(*args)
    assert isinstance(got, float) and 0.0 <= got <= 1.0
    if case.expected is not None:
        assert got == pytest.approx(case.expected, rel=1e-9, abs=1e-300)
    try:
        from scipy import stats as sps
    except ImportError:                                # scipy is optional; hand-computed cases still ran
        return
    k, K, n, N = case.overlap, case.set_n, case.query_n, case.universe_n
    if plugin.name == "fisher":
        table = [[k, n - k], [K - k, N - K - n + k]]
        ref = float(sps.fisher_exact(table, alternative=case.alternative)[1])
    else:
        ref = float(sps.hypergeom.sf(k - 1, N, K, n))
    assert got == pytest.approx(min(1.0, ref), rel=1e-7, abs=1e-300), (case.name, got, ref)


def reference_km_median(times: list[float], events: list[bool]) -> float | None:
    """A direct product-limit median: S(t) recomputed from scratch at every distinct event time."""
    for t in sorted({tt for tt, e in zip(times, events) if e}):
        s = 1.0
        for u in sorted({tt for tt, e in zip(times, events) if e and tt <= t}):
            at_risk = sum(1 for tt in times if tt >= u)
            deaths = sum(1 for tt, e in zip(times, events) if e and tt == u)
            s *= 1 - deaths / at_risk
        if s <= 0.5 + 1e-12:
            return t
    return None


@pytest.mark.parametrize("plugin,case", PAIRED_PARAMS)
def test_s9_paired_survival(plugin: Any, case: PairedCase) -> None:
    spec = {"role": "measure", "statistic": plugin.name}
    times = [float(t) for t in case.times]
    events = [bool(e) for e in case.events]
    keys = list(case.keys) if case.keys is not None else None
    if keys is not None:                               # one case per level key (its first row)
        first: dict[Any, int] = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        idx = sorted(first.values())
        rt, re_ = [times[i] for i in idx], [events[i] for i in idx]
    else:
        rt, re_ = times, events
    med = plugin.aggregate_pair(times, events, "median", spec, keys=keys)
    assert med.value == reference_km_median(rt, re_)
    assert med.n == (case.expected_n if case.expected_n is not None else len(rt))
    assert plugin.aggregate_pair(times, events, "events", spec, keys=keys).value == sum(re_)
    for how in ("mean", "sum"):
        with pytest.raises(UnsupportedFilter):
            plugin.aggregate_pair(times, events, how, spec)
        with pytest.raises(UnsupportedFilter):
            plugin.aggregate(times, how, spec)
    unknown = plugin.aggregate_pair([None, times[0]], [True, None], "median", spec)
    assert unknown.value is None and unknown.n == 0 and unknown.n_excluded == 2
    assert plugin.aggregate_pair([], [], "median", spec).value is None

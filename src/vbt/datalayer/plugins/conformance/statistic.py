"""Statistic conformance suite (§9.3, S-1..S-6 and S-8), parametrized over the registered statistic plugins.

Each plugin's ``conformance_cases()`` returns :class:`~.golden.StatisticCase` objects (a plugin
without cases is run on a plain numeric spec). The suite never needs pyarrow. Phase 3 adds
S-7 (family-based BH) and S-9 (``test``/``paired`` capabilities) to this module.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from ...predicate import evaluate
from ...rowkey import canonical
from ..base import UnsupportedFilter
from ..statistics import order_rows, rank_value_of, spec_get
from . import for_plugin, selected_plugins
from .golden import StatisticCase

STATISTIC_PLUGINS = selected_plugins("statistic")
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

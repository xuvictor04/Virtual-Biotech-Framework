"""Derived conformance suite (ASN-5), parametrized over the registered derived plugins.

Each plugin's ``conformance_cases()`` returns :class:`DerivedCases`: options it must accept and options it must
reject, recorded cases (in-memory rows of the request's table, the overlay options, the call's arguments and what
the answer must hold) and refusals (calls it must refuse). The cases run against :class:`MemoryView`, an in-memory
stand-in for the data child's long view with the same ``rows``/``column`` interface, so a plugin is checked without
any data on disk.

- D-1 ``validate_options`` accepts the valid options (no problem listed) and lists a problem for each invalid one;
  ``serve`` with invalid options raises ``DerivedOptionsError`` and never answers.
- D-2 recorded cases answer with the expected total, first row (the keys given), number of rows returned under the
  case's limit, key columns and ``truncated``; every answer is ``served_by: derived``, and every ``_``-section it
  returns is one of the plugin's ``records`` (which the gateway keeps in provenance).
- D-3 serving is pure: deterministic, the input rows are not modified and their order does not change the answer.
- D-4 the plugin reads only the columns ``columns(options)`` declares, through the view, with the request's byte
  budget (never an unbounded read).
- D-5 refusals raise a typed error (``GatewayError`` or ``DerivedOptionsError``), never a partial answer.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from ..base import DerivedOptionsError, derived_records
from . import selected_plugins

__all__ = ["DerivedCase", "DerivedCases", "MemoryView", "run_case"]

_UNSET: Any = object()
BUDGET = 1 << 20


@dataclass(frozen=True)
class DerivedCase:
    """One call: the table's rows, the overlay options, the call's arguments and what the answer must hold (unset
    fields are not checked; ``first`` lists keys of the first row)."""

    name: str
    rows: Sequence[Mapping[str, Any]]
    options: Mapping[str, Any]
    params: Mapping[str, Any]
    total: Any = _UNSET
    first: Mapping[str, Any] | None = None
    limit: int | None = None
    returned: Any = _UNSET
    key_columns: Any = _UNSET
    specs: Mapping[str, Any] = field(default_factory=dict)      # column path -> spec (``cutoff`` for a measure)
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedCases:
    valid_options: tuple[Mapping[str, Any], ...] = ()
    invalid_options: tuple[Mapping[str, Any], ...] = ()
    cases: tuple[DerivedCase, ...] = ()
    #: (rows, options, params) calls the plugin must refuse with a typed error
    refusals: tuple[tuple[Sequence[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]], ...] = ()


class MemoryView:
    """The long view's interface over in-memory rows: ``rows(predicate, columns, order, limit, budget)`` filters
    with the predicate (three-valued: only rows it holds for), projects to ``columns`` and records what was asked;
    ``column(path)`` returns the case's spec for that path."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], specs: Mapping[str, Any] | None = None) -> None:
        self._rows = rows
        self._specs = dict(specs or {})
        self.reads: list[dict[str, Any]] = []

    def column(self, path: str) -> Any:
        return self._specs.get(path)

    def rows(self, predicate: Any = None, *, columns: Sequence[str] | None = None, order: Sequence[Any] = (),
             limit: int | None = None, budget: Any = None) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        from ...predicate import evaluate

        self.reads.append({"columns": list(columns) if columns is not None else None, "budget": budget,
                           "limit": limit})
        out = []
        for r in self._rows:
            if evaluate(predicate, r) is True:
                out.append({c: copy.deepcopy(r.get(c)) for c in columns} if columns is not None else copy.deepcopy(
                    dict(r)))
        total = len(out)
        return (out[:limit] if limit is not None else out), total, {}


def _request(case: DerivedCase) -> dict[str, Any]:
    return {"table": "conformance.table", "params": dict(case.params), "limit": case.limit, "budget_bytes": BUDGET}


def run_case(plugin: Any, case: DerivedCase, rows: Sequence[Mapping[str, Any]] | None = None
             ) -> tuple[dict[str, Any], MemoryView]:
    view = MemoryView(case.rows if rows is None else rows, case.specs)
    got = plugin.serve(view, _request(case), copy.deepcopy(dict(case.options)))
    assert isinstance(got, Mapping), f"{plugin.name}.serve returned {type(got).__name__}"
    return dict(got), view


DERIVED_PLUGINS = selected_plugins("derived")


def _cases(plugin: Any) -> DerivedCases:
    cases = plugin.conformance_cases()
    if not isinstance(cases, DerivedCases):
        pytest.fail(f"{plugin.name}: conformance_cases() must return DerivedCases")
    return cases


def _plugin_params() -> list[Any]:
    return [pytest.param(p, id=p.name) for p in DERIVED_PLUGINS]


def _case_params() -> list[Any]:
    out = []
    for p in DERIVED_PLUGINS:
        for case in _cases(p).cases:
            if all(c in (getattr(p, "capabilities", ()) or ()) for c in case.requires):
                out.append(pytest.param(p, case, id=f"{p.name}-{case.name}"))
    return out


@pytest.mark.parametrize("plugin", _plugin_params())
def test_d1_options_are_validated(plugin: Any) -> None:
    cases = _cases(plugin)
    assert cases.valid_options and cases.invalid_options, f"{plugin.name}: give valid and invalid options"
    for options in cases.valid_options:
        assert plugin.validate_options(copy.deepcopy(dict(options))) == [], f"{plugin.name}: rejects {options}"
    for options in cases.invalid_options:
        problems = plugin.validate_options(copy.deepcopy(dict(options)))
        assert problems and all(isinstance(p, str) and p for p in problems), f"{plugin.name}: accepts {options}"
        view = MemoryView([{"x": 1}])
        with pytest.raises(DerivedOptionsError):
            plugin.serve(view, {"params": {}, "budget_bytes": BUDGET}, copy.deepcopy(dict(options)))


@pytest.mark.parametrize("plugin, case", _case_params())
def test_d2_recorded_cases(plugin: Any, case: DerivedCase) -> None:
    got, _view = run_case(plugin, case)
    assert got.get("served_by") == "derived" and not got.get("error"), f"{case.name}: {got}"
    rows = got.get("rows")
    assert isinstance(rows, list), f"{case.name}: rows is {type(rows).__name__}"
    if case.total is not _UNSET:
        assert got.get("total") == case.total, f"{case.name}: total {got.get('total')} != {case.total}"
    if case.returned is not _UNSET:
        assert len(rows) == case.returned, f"{case.name}: {len(rows)} rows returned != {case.returned}"
    if case.limit is not None:
        assert len(rows) <= case.limit
    if isinstance(got.get("total"), int):
        assert bool(got.get("truncated")) == (len(rows) < got["total"]), f"{case.name}: truncated is wrong"
    if case.key_columns is not _UNSET:
        assert list(got.get("key_columns") or []) == list(case.key_columns)
    unrecorded = [k for k in (got.get("sections") or {}) if str(k).startswith("_") and
                  k not in derived_records(plugin)]
    assert not unrecorded, f"{case.name}: sections {unrecorded} are not in the plugin's records (lost to provenance)"
    if case.first is not None:
        assert rows, f"{case.name}: no row"
        for k, v in case.first.items():
            assert rows[0].get(k) == v, f"{case.name}: first row {k}={rows[0].get(k)!r} != {v!r}"


def _round(x: Any, digits: int | None) -> Any:
    if isinstance(x, float) and digits is not None:
        return round(x, digits)
    if isinstance(x, Mapping):
        return {k: _round(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, digits) for v in x]
    return x


def _dump(x: Any, digits: int | None = None) -> str:
    return json.dumps(_round(x, digits), sort_keys=True, default=str)


@pytest.mark.parametrize("plugin", _plugin_params())
def test_d3_serving_is_pure(plugin: Any) -> None:
    for case in _cases(plugin).cases:
        rows = [copy.deepcopy(dict(r)) for r in case.rows]
        before = json.dumps(rows, sort_keys=True, default=str)
        a, _ = run_case(plugin, case, rows)
        assert json.dumps(rows, sort_keys=True, default=str) == before, f"{case.name}: input rows modified"
        b, _ = run_case(plugin, case, rows)
        c, _ = run_case(plugin, case, list(reversed(rows)))
        assert _dump(a) == _dump(b), f"{case.name}: not deterministic"
        # summing in another order may move the last bit of a mean: floats compare to 9 decimals
        assert _dump(a, 9) == _dump(c, 9), f"{case.name}: the answer depends on the row order"


@pytest.mark.parametrize("plugin", _plugin_params())
def test_d4_reads_only_its_columns_within_the_budget(plugin: Any) -> None:
    for case in _cases(plugin).cases:
        _got, view = run_case(plugin, case)
        declared = set(plugin.columns(dict(case.options)))
        assert view.reads, f"{case.name}: the table was not read through the view"
        for read in view.reads:
            assert read["columns"] is not None, f"{case.name}: read every column"
            assert set(read["columns"]) <= declared, f"{case.name}: read {set(read['columns']) - declared}"
            assert read["budget"] == BUDGET, f"{case.name}: the request's byte budget was not passed to the view"


@pytest.mark.parametrize("plugin", _plugin_params())
def test_d5_refusals_are_typed(plugin: Any) -> None:
    from ...errors import GatewayError

    for rows, options, params in _cases(plugin).refusals:
        view = MemoryView(rows)
        with pytest.raises((GatewayError, DerivedOptionsError)):
            plugin.serve(view, {"params": dict(params), "budget_bytes": BUDGET}, copy.deepcopy(dict(options)))

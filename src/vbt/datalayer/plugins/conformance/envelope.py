"""Envelope conformance suite (phase 4, F21), parametrized over the registered envelope plugins.

Each plugin's ``conformance_cases()`` returns :class:`EnvelopeCases`: recorded outputs (text or
structured content, the codec options, and what decoding must find) and the ``spec`` the
unknown-shape cases are decoded with. A case runs only when the plugin declares every capability
in its ``requires``.

- E-1 recorded outputs decode to the expected rows, total, found verdict, message and HTTP status.
- E-2 nested errors carried by a success envelope are surfaced in ``errors``, never dropped.
- E-3 a shape the spec does not describe (other keys, plain text, a scalar, an empty object) is
  ``unparsed`` with no rows: a decoder never invents rows.
- E-4 decoding is pure: deterministic, and the structured input is not modified.
- E-5 decoding never raises, whatever the payload (garbage text, deep nesting, wrong types).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from ..base import ParsedResult
from . import applicable, selected_plugins

__all__ = ["EnvelopeCase", "EnvelopeCases", "UNKNOWN_SHAPES", "GARBAGE", "check_case"]

_UNSET: Any = object()


@dataclass(frozen=True)
class EnvelopeCase:
    """One recorded output and what decoding it must find (unset fields are not checked).
    ``rows`` is the number of rows (None: no row list)."""

    name: str
    raw_text: str | None
    structured: Any
    spec: Mapping[str, Any] = field(default_factory=dict)
    rows: Any = _UNSET
    total: Any = _UNSET
    found: Any = _UNSET
    message: Any = _UNSET
    unparsed: Any = _UNSET
    errors: Any = _UNSET
    http_status: Any = _UNSET
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvelopeCases:
    recorded: tuple[EnvelopeCase, ...] = ()
    #: the codec options the unknown-shape cases (E-3) are decoded with; it should name row paths
    spec: Mapping[str, Any] = field(default_factory=dict)


#: Payloads whose shape no row-naming spec describes (E-3).
UNKNOWN_SHAPES: tuple[tuple[str, str | None, Any], ...] = (
    ("other_keys", '{"hits": {"hit": [{"id": 1}]}, "took": 3}', None),
    ("plain_text", "Service temporarily rerouted; try again", None),
    ("scalar", "42", None),
    ("empty_object", "{}", None),
    ("structured_other", None, {"payload": {"items": [1, 2, 3]}}),
)

#: Payloads that must not make a decoder raise (E-5).
GARBAGE: tuple[tuple[str | None, Any], ...] = (
    (None, None), ("", None), ("\x00\xff{", None), ("[" * 200 + "]" * 200, None), ("null", None),
    ('{"results": "not a list", "count": "many"}', None), (None, {"results": None, "count": {"x": 1}}),
    (None, [1, "two", None]), ('{"error": null, "message": 7}', None),
)

ENVELOPE_PLUGINS = selected_plugins("envelope")


def _cases(plugin: Any) -> EnvelopeCases:
    cases = plugin.conformance_cases()
    if not isinstance(cases, EnvelopeCases):
        pytest.fail(f"{plugin.name}: conformance_cases() must return EnvelopeCases")
    return cases


def _recorded_params() -> list[Any]:
    out = []
    for p in ENVELOPE_PLUGINS:
        for case in _cases(p).recorded:
            if applicable(p, case.requires or None):
                out.append(pytest.param(p, case, id=f"{p.name}-{case.name}"))
    return out


def _plugin_params() -> list[Any]:
    return [pytest.param(p, id=p.name) for p in ENVELOPE_PLUGINS]


def check_case(plugin: Any, case: EnvelopeCase) -> ParsedResult:
    """Decode ``case`` with ``plugin`` and assert every field the case sets (E-1, E-2)."""
    got = plugin.decode(case.raw_text, copy.deepcopy(case.structured), dict(case.spec))
    assert isinstance(got, ParsedResult), f"{plugin.name}.decode returned {type(got).__name__}"
    if case.rows is not _UNSET:
        n = None if got.rows is None else len(got.rows)
        assert n == case.rows, f"{case.name}: rows {n} != {case.rows}"
    for name in ("total", "found", "message", "unparsed", "http_status"):
        want = getattr(case, name)
        if want is not _UNSET:
            assert getattr(got, name) == want, f"{case.name}: {name} {getattr(got, name)!r} != {want!r}"
    if case.errors is not _UNSET:
        assert tuple(got.errors) == tuple(case.errors), f"{case.name}: errors {got.errors!r}"
    if got.unparsed:
        assert not got.rows, f"{case.name}: an unparsed result carries rows"
    return got


@pytest.mark.parametrize("plugin, case", _recorded_params())
def test_e1_recorded_outputs_decode(plugin: Any, case: EnvelopeCase) -> None:
    check_case(plugin, case)


@pytest.mark.parametrize("plugin", _plugin_params())
def test_e2_nested_errors_surface(plugin: Any) -> None:
    if not applicable(plugin, "nested_errors"):
        pytest.skip(f"{plugin.name} does not declare nested_errors")
    cases = [c for c in _cases(plugin).recorded if c.errors is not _UNSET and c.errors]
    assert cases, f"{plugin.name} declares nested_errors but records no case with one"
    for case in cases:
        assert check_case(plugin, case).errors


@pytest.mark.parametrize("plugin", _plugin_params())
def test_e3_unknown_shapes_are_unparsed(plugin: Any) -> None:
    spec = dict(_cases(plugin).spec)
    if not spec:
        pytest.skip(f"{plugin.name} gives no spec for unknown shapes")
    for name, text, structured in UNKNOWN_SHAPES:
        got = plugin.decode(text, copy.deepcopy(structured), spec)
        assert got.unparsed, f"{name}: decoded as a known shape: {got!r}"
        assert not got.rows, f"{name}: rows invented from an unknown shape"
        assert got.total is None, f"{name}: total invented"


@pytest.mark.parametrize("plugin", _plugin_params())
def test_e4_decode_is_pure(plugin: Any) -> None:
    for case in _cases(plugin).recorded:
        if case.structured is None:
            continue
        before = json.dumps(case.structured, sort_keys=True, default=str)
        data = copy.deepcopy(case.structured)
        a = plugin.decode(case.raw_text, data, dict(case.spec))
        assert json.dumps(data, sort_keys=True, default=str) == before, f"{case.name}: input modified"
        assert plugin.decode(case.raw_text, data, dict(case.spec)) == a, f"{case.name}: not deterministic"


@pytest.mark.parametrize("plugin", _plugin_params())
def test_e5_never_raises(plugin: Any) -> None:
    spec = dict(_cases(plugin).spec)
    for text, structured in GARBAGE:
        got = plugin.decode(text, structured, spec)
        assert isinstance(got, ParsedResult)
        if got.rows is not None:
            assert isinstance(got.rows, (list, tuple))

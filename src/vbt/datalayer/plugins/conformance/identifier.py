"""Identifier conformance suite I-1..I-7 (§9.3), parametrized over every registered identifier plugin.

Run it with pytest (``tests/datalayer/test_dl_conformance_identifier.py`` collects it) or with
``vbt datasource conformance --plugin <name>`` (``VBT_CONFORMANCE_PLUGIN`` restricts the subjects;
the confusion matrix still pairs the subject with every registered plugin).

Cases (``conformance_cases()``) are mappings; untagged cases apply to every plugin, cases with
``requires`` only to plugins declaring those capabilities:

* ``{"raw": x, "expected": y[, "steps": [...]]}`` — ``normalize(x)`` gives ``y`` (with exactly
  these steps when given); ``{"stored": x, "expected": y}`` — the same for ``normalize_stored``;
* ``{"raw": x, "rejected": true[, "looks_like": kind]}`` — ``normalize(x)`` is ``Rejected`` (and
  names ``kind`` in ``looks_like``);
* ``{"label": x, "key": k}`` — ``label_key(x) == k``;
* ``options`` / ``universe_sample`` on any case run it on ``plugin.configure(options, sample)``.

A plugin with YAML options declares a representative configuration (``example_options`` and
``example_universe``); I-3 and I-4 use it, so a configured ``local_key`` is "its own kind" that
accepts only its declared pattern. Overlaps are symmetric for I-4: a pair is exempt when either
plugin declares the other (by name or id_type), so a new plugin can declare an overlap with a
builtin without editing the builtin.

The ``check_*`` functions raise ``AssertionError`` with every violation and are usable without
pytest; the ``test_*`` functions wrap them.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

import pytest

from ..base import NORMALIZE_STEPS, Normalized, Rejected
from . import for_plugin, selected_plugins

__all__ = [
    "representative", "kind_names", "exempt", "probes", "check_cases", "check_idempotent", "check_canonical",
    "confusion", "check_confusion", "check_steps", "check_describe", "check_universe_options", "universe_cases",
]


def _registry() -> Any:
    from ..registry import discover
    from ...settings import DataSettings
    return discover(DataSettings.from_env())


_REGISTRY = _registry()
ALL: list[Any] = _REGISTRY.all("identifier")
PLUGINS: list[Any] = selected_plugins("identifier", _REGISTRY)
IDS = [p.name for p in PLUGINS]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def representative(plugin: Any) -> Any:
    """The plugin as the suite tests it: configured with its ``example_options`` when it has them."""
    options = getattr(plugin, "example_options", None)
    if options is None:
        return plugin
    return plugin.configure(dict(options), getattr(plugin, "example_universe", None))


def kind_names(plugin: Any) -> set[str]:
    return {str(plugin.name), str(getattr(plugin, "id_type", plugin.name))}


def exempt(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` legitimately share syntax (either declares the other, or they are one kind)."""
    na, nb = kind_names(a), kind_names(b)
    return bool(na & nb) or bool(set(getattr(a, "overlaps", ())) & nb) or bool(set(getattr(b, "overlaps", ())) & na)


def _configured(plugin: Any, case: Mapping[str, Any]) -> Any:
    if "options" in case or "universe_sample" in case:
        return plugin.configure(dict(case.get("options") or {}), case.get("universe_sample"))
    return plugin


def _cases(plugin: Any) -> list[Mapping[str, Any]]:
    return for_plugin(plugin, list(plugin.conformance_cases() or ()))


def probes(plugin: Any) -> list[str]:
    """Inputs the step and idempotence checks run on: case inputs, examples and simple variants of them."""
    out: list[str] = []
    rep = representative(plugin)
    for case in _cases(plugin):
        for key in ("raw", "stored"):
            if key in case and not case.get("options") and not case.get("universe_sample"):
                out.append(case[key])
    for e in rep.examples:
        out.extend([e, str(e).lower(), str(e).upper(), f" {e} "])
    return out


def _fail(problems: Sequence[str], what: str) -> None:
    if problems:
        raise AssertionError(f"{what}:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# I-1 .. I-7
# ---------------------------------------------------------------------------

def check_cases(plugin: Any) -> None:
    """I-1: every case from ``conformance_cases()`` holds."""
    problems = []
    for case in _cases(plugin):
        p = _configured(plugin, case)
        if "label" in case:
            got = p.label_key(case["label"])
            if got != case["key"]:
                problems.append(f"label_key({case['label']!r}) = {got!r}, expected {case['key']!r}")
            continue
        stored = "stored" in case
        value = case["stored"] if stored else case.get("raw")
        n = p.normalize_stored(value) if stored else p.normalize(value)
        fn = "normalize_stored" if stored else "normalize"
        if case.get("rejected"):
            if not isinstance(n, Rejected):
                problems.append(f"{fn}({value!r}) = {n!r}, expected Rejected")
            elif case.get("looks_like") and case["looks_like"] not in n.looks_like:
                problems.append(f"{fn}({value!r}).looks_like = {n.looks_like!r}, expected {case['looks_like']!r}")
            continue
        if not isinstance(n, Normalized) or n.value != case.get("expected"):
            problems.append(f"{fn}({value!r}) = {n!r}, expected {case.get('expected')!r}")
        elif "steps" in case and list(n.steps) != list(case["steps"]):
            problems.append(f"{fn}({value!r}).steps = {list(n.steps)!r}, expected {list(case['steps'])!r}")
    _fail(problems, f"I-1 {plugin.name}")


def check_idempotent(plugin: Any) -> None:
    """I-2: ``normalize`` and ``normalize_stored`` are idempotent (the normalised value needs no further step)."""
    rep = representative(plugin)
    problems = []
    for raw in probes(plugin):
        for fn in ("normalize", "normalize_stored"):
            n = getattr(rep, fn)(raw)
            if not isinstance(n, Normalized):
                continue
            again = getattr(rep, fn)(n.value)
            if not isinstance(again, Normalized) or again.value != n.value or again.steps:
                problems.append(f"{fn}({raw!r}) = {n.value!r} but {fn}({n.value!r}) = {again!r}")
    _fail(problems, f"I-2 {plugin.name}")


def check_canonical(plugin: Any) -> None:
    """I-3: examples pass unchanged and every normalised output matches ``canonical``."""
    rep = representative(plugin)
    problems = []
    if not rep.examples:
        problems.append("no examples")
    for e in rep.examples:
        for fn in ("normalize", "normalize_stored"):
            n = getattr(rep, fn)(e)
            if not isinstance(n, Normalized) or n.value != e or n.steps:
                problems.append(f"example {e!r}: {fn} gives {n!r}")
    for case in _cases(plugin):
        p = _configured(plugin, case)
        for key, fn in (("raw", "normalize"), ("stored", "normalize_stored")):
            if key not in case:
                continue
            n = getattr(p, fn)(case[key])
            if isinstance(n, Normalized) and not re.fullmatch(p.canonical, n.value):
                problems.append(f"{fn}({case[key]!r}) = {n.value!r} does not match {p.canonical}")
    for raw in probes(plugin):
        n = rep.normalize(raw)
        if isinstance(n, Normalized) and not re.fullmatch(rep.canonical, n.value):
            problems.append(f"normalize({raw!r}) = {n.value!r} does not match {rep.canonical}")
    _fail(problems, f"I-3 {plugin.name}")


def confusion(a: Any, plugins: Iterable[Any]) -> list[str]:
    """I-4 for subject ``a``: every non-exempt pair (a, b) and (b, a) rejects the other's examples."""
    ra = representative(a)
    problems = []
    for b in plugins:
        if exempt(a, b):
            continue
        rb = representative(b)
        for x, y in ((ra, rb), (rb, ra)):
            for e in y.examples:
                n = x.normalize(e)
                if not isinstance(n, Rejected):
                    problems.append(f"{x.name}.normalize({e!r}) (an example of {y.name}) = {n!r}")
    return problems


def check_confusion(plugin: Any, plugins: Iterable[Any] | None = None) -> None:
    _fail(confusion(plugin, ALL if plugins is None else plugins), f"I-4 {plugin.name}")


def check_steps(plugin: Any) -> None:
    """I-5: every recorded step is on the whitelist."""
    problems = []
    rep = representative(plugin)
    subjects = [(rep, raw) for raw in probes(plugin)]
    for case in _cases(plugin):
        p = _configured(plugin, case)
        subjects.extend((p, case[k]) for k in ("raw", "stored") if k in case)
    for p, raw in subjects:
        for fn in ("normalize", "normalize_stored"):
            n = getattr(p, fn)(raw)
            if isinstance(n, Normalized):
                bad = [s for s in n.steps if s not in NORMALIZE_STEPS]
                if bad:
                    problems.append(f"{fn}({raw!r}) records steps {bad} outside the whitelist")
    _fail(problems, f"I-5 {plugin.name}")


def check_describe(plugin: Any) -> None:
    """I-6: ``describe()`` is non-empty, at most 200 characters and contains an example."""
    text = plugin.describe()
    problems = []
    if not text or len(text) > 200:
        problems.append(f"describe() has {len(text or '')} characters: {text!r}")
    if not any(str(e) in (text or "") for e in plugin.examples):
        problems.append(f"describe() names no example: {text!r}")
    _fail(problems, f"I-6 {plugin.name}")


def universe_cases(plugin: Any) -> list[Mapping[str, Any]]:
    """The cases that configure the plugin from a universe sample (``prefixes: from_universe``)."""
    out = []
    for case in _cases(plugin):
        opts = case.get("options") or {}
        if case.get("universe_sample") and any(v == "from_universe" for v in opts.values()):
            out.append(case)
    rep_opts = getattr(plugin, "example_options", None) or {}
    if getattr(plugin, "example_universe", None) and any(v == "from_universe" for v in rep_opts.values()):
        out.append({"options": dict(rep_opts), "universe_sample": list(plugin.example_universe)})
    return out


def check_universe_options(plugin: Any, plugins: Iterable[Any] | None = None) -> None:
    """I-7: configured from a universe sample, the plugin accepts every sampled key unchanged and still
    rejects the other kinds' examples (overlaps exempt)."""
    problems = []
    for case in universe_cases(plugin):
        p = plugin.configure(dict(case["options"]), list(case["universe_sample"]))
        for key in case["universe_sample"]:
            for fn in ("normalize", "normalize_stored"):
                n = getattr(p, fn)(key)
                if not isinstance(n, Normalized) or n.value != key:
                    problems.append(f"configured {fn}({key!r}) = {n!r}")
        for b in ALL if plugins is None else plugins:
            if exempt(plugin, b):
                continue
            for e in representative(b).examples:
                if not isinstance(p.normalize(e), Rejected):
                    problems.append(f"configured from {case['universe_sample']!r}, accepts {e!r} of {b.name}")
    _fail(problems, f"I-7 {plugin.name}")


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i1_cases(plugin: Any) -> None:
    check_cases(plugin)


@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i2_idempotent(plugin: Any) -> None:
    check_idempotent(plugin)


@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i3_canonical_examples(plugin: Any) -> None:
    check_canonical(plugin)


@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i4_confusion_matrix(plugin: Any) -> None:
    check_confusion(plugin)


@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i5_step_whitelist(plugin: Any) -> None:
    check_steps(plugin)


@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i6_describe(plugin: Any) -> None:
    check_describe(plugin)


@pytest.mark.parametrize("plugin", PLUGINS, ids=IDS)
def test_i7_universe_options(plugin: Any) -> None:
    if not universe_cases(plugin):
        pytest.skip(f"{plugin.name} takes no options from a universe sample")
    check_universe_options(plugin)

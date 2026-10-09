"""Derived conformance suite (ASN-5: the derived plugin kind) over the registered derived plugins: the shipped
``essentiality`` and ``tissue_specificity`` plugins pass D-1..D-5 here, as a project's plugin does under
``RegisterPlugin`` (tests/test_utilities.py)."""

from __future__ import annotations

from vbt.datalayer.plugins.conformance.derived import (  # noqa: F401  (collects D-1..D-5)
    test_d1_options_are_validated,
    test_d2_recorded_cases,
    test_d3_serving_is_pure,
    test_d4_reads_only_its_columns_within_the_budget,
    test_d5_refusals_are_typed,
)
from vbt.datalayer.plugins.registry import discover, validate_plugin


def test_the_shipped_derived_plugins_satisfy_the_kind():
    reg = discover(entry_points=False)
    for name in ("essentiality", "tissue_specificity"):
        plugin = reg.get("derived", name)
        validate_plugin(plugin)
        assert plugin.validate_options({}) and plugin.columns({}) == []

"""The identifier conformance suite (§9.3 I-1..I-7) over every registered identifier plugin.

The suite lives in ``vbt.datalayer.plugins.conformance.identifier`` so external plugin packages and
``vbt datasource conformance`` run the same checks; importing its test functions here collects them.
"""

from __future__ import annotations

from vbt.datalayer.plugins.conformance.identifier import (  # noqa: F401 - collected by pytest
    test_i1_cases,
    test_i2_idempotent,
    test_i3_canonical_examples,
    test_i4_confusion_matrix,
    test_i5_step_whitelist,
    test_i6_describe,
    test_i7_universe_options,
)

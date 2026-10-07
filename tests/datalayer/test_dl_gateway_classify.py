"""Classification precedence (§11.4 step 1, §11.9): memory signatures, explicit not-found before
legacy envelopes, existence-dependent meaning of a not-found on reviewed tools, nested errors versus
``partial_when``, and the generic guard's uncitable structural empties."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from test_dl_gateway_flow import raw_of
from vbt.datalayer.api import RawResult
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.errors import ErrorKind
from vbt.datalayer.gateway.classify import classify, structural_empty
from vbt.datalayer.gateway.fields import jp_test

SRC: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "cbio", "title": "cBioPortal", "kind": "remote",
    "release": {"from": "as_of"},
    "id_types": {"cbio_study": {"plugin": "cbio_study", "universe": "study.studyId"}},
    "tables": {
        "study": {"kind": "records", "grain": "one study", "format": "none", "key": {"columns": ["studyId"]},
                  "columns": {"studyId": {"role": "identifier", "id_type": "cbio_study", "self": True}}},
        "sample": {"kind": "records", "grain": "one sample", "format": "none",
                   "key": {"columns": ["studyId", "sampleId"]},
                   "columns": {"studyId": {"role": "identifier", "id_type": "cbio_study"},
                               "sampleId": {"role": "label"}}},
    },
}

OV: dict[str, Any] = {
    "schema": "vbt.overlay/1", "server": "clinicaltrials",
    "tools": {
        "get_study": {"reads": {"cbio.study": {"access": "upstream"}},
                      "args": {"study_id": {"binds": "cbio.study.studyId", "accepts": ["cbio_study"],
                                            "existence": "upstream"}},
                      "result": {"kind": "record", "not_found_when": ["$.error =~ '(?i)status code: 404'"],
                                 "nested_errors": ["$.details.error"],
                                 "total": {"path": "$.total", "partial_when": ["$.warning"]}}},
        "get_study_checked": {"reads": {"cbio.study": {"access": "upstream"}},
                              "args": {"study_id": {"binds": "cbio.study.studyId", "accepts": ["cbio_study"]}},
                              "result": {"kind": "record"}},
        "get_samples": {"reads": {"cbio.sample": {"access": "upstream"}},
                        "args": {"study_id": {"binds": "cbio.sample.studyId", "accepts": ["cbio_study"]}},
                        "result": {"rows": "$.data"}},
    },
}

GENERIC: dict[str, Any] = {"schema": "vbt.overlay/1", "server": "*",
                           "generic": {"not_found_when": ["$.status == 404"], "empty_when": ["$.hits == 0"]}}


@pytest.fixture(scope="module")
def catalog():
    return Catalog({"cbio": SourceDescriptor.model_validate(copy.deepcopy(SRC))},
                   {"clinicaltrials": Overlay.model_validate(copy.deepcopy(OV))},
                   [Overlay.model_validate(copy.deepcopy(GENERIC))])


UNIVERSE = {"study_id": ["cbio.study"]}


def plan(contract, args, existence=None):
    return SimpleNamespace(args_raw=args, existence=dict(existence or {}), bound_table=contract.bound_table)


def test_cbioportal_404_legacy_text_is_not_found(catalog):
    c = catalog.contract("clinicaltrials", "get_study")
    raw = raw_of({"error": "Failed to retrieve study data: status code: 404"})
    assert raw.envelope == "legacy_error"               # the bridge alone would call this an outage
    out = classify(raw, c, plan(c, {"study_id": "x_study"}), universe_tables=UNIVERSE)
    assert out.outcome == "not_found" and out.error.kind == ErrorKind.not_found
    assert out.error.envelope()["argument"] == "study_id"


def test_memory_signature_beats_not_found(catalog):
    c = catalog.contract("clinicaltrials", "get_study")
    raw = RawResult("MemoryError: Unable to allocate 8.00 GiB; status code: 404", None, None, "is_error",
                    "MemoryError: Unable to allocate 8.00 GiB; status code: 404")
    out = classify(raw, c, plan(c, {"study_id": "x"}), universe_tables=UNIVERSE)
    assert out.outcome == "oom" and out.error.kind == ErrorKind.oom


@pytest.mark.parametrize("message", ["Study x_study not found", "No samples found"])
def test_empty_lookup_on_universe_table(catalog, message):
    c = catalog.contract("clinicaltrials", "get_study_checked")
    raw = raw_of({"error": message})
    assert raw.envelope == "empty_lookup"
    # resolution proved the study exists on the universe table: a contradiction
    out = classify(raw, c, plan(c, {"study_id": "x"}, {"study_id": "exists"}), universe_tables=UNIVERSE)
    assert out.outcome == "contradiction"
    # existence upstream: the explicit not-found decides
    c2 = catalog.contract("clinicaltrials", "get_study")
    out = classify(raw, c2, plan(c2, {"study_id": "x"}), universe_tables=UNIVERSE)
    assert out.outcome == "not_found"


def test_empty_lookup_on_another_table_depends_on_the_witness(catalog):
    c = catalog.contract("clinicaltrials", "get_samples")
    raw = raw_of({"error": "No samples found"})
    p = plan(c, {"study_id": "x"}, {"study_id": "exists"})
    assert classify(raw, c, p, universe_tables=UNIVERSE, witness_total=0).outcome == "empty"
    assert classify(raw, c, p, universe_tables=UNIVERSE, witness_total=3).outcome == "contradiction"
    assert classify(raw, c, p, universe_tables=UNIVERSE, witness_total=None).outcome == "empty_unverified"


def test_nested_errors_versus_partial_when(catalog):
    c = catalog.contract("clinicaltrials", "get_study")
    out = classify(raw_of({"studyId": "x", "details": {"error": "drug_info failed"}}), c, plan(c, {}))
    assert out.outcome == "source_error" and out.error.kind == ErrorKind.source_error
    out = classify(raw_of({"studyId": "x", "details": {"error": "page 3 failed"}, "warning": "truncated"}), c,
                   plan(c, {}))
    assert out.outcome == "partial"


def test_remaining_error_envelopes_are_source_errors(catalog):
    c = catalog.contract("clinicaltrials", "get_study")
    out = classify(raw_of({"success": False, "error": "upstream HTTP 503"}), c, plan(c, {"study_id": "x"}),
                   universe_tables=UNIVERSE)
    assert out.outcome == "source_error"
    out = classify(raw_of("Traceback ... KeyError", is_error=True), c, plan(c, {}))
    assert out.outcome == "source_error"


def test_generic_guard(catalog):
    c = catalog.contract("unknown_server", "anything")
    assert c.generic
    assert classify(raw_of({"error": "Gene not found"}), c, plan(c, {})).outcome == "not_found"
    assert classify(raw_of({"status": 404, "message": "no"}), c, plan(c, {})).outcome == "not_found"
    assert classify(raw_of({"count": 0, "results": []}), c, plan(c, {})).outcome == "empty_unverified"
    assert classify(raw_of({"hits": 0, "page": {}}), c, plan(c, {})).outcome == "empty_unverified"
    assert classify(raw_of([]), c, plan(c, {})).outcome == "empty_unverified"
    assert classify(raw_of({"count": 1, "results": [{"x": 1}]}), c, plan(c, {})).outcome == "ok"
    assert classify(raw_of("plain text answer"), c, plan(c, {})).outcome == "ok"
    # a plain-text success that mentions an HTTP status is content, not the call's status (R4)
    for text in ("Abstract: during the outage the API returned HTTP 503 for 2 hours.",
                 "Wikipedia: The 404 Not Found error is an HTTP status code.", "server error 500 explained"):
        assert classify(raw_of(text), c, plan(c, {})).outcome == "ok", text
    # the same text as an error envelope still reads its status
    assert classify(raw_of("upstream failed: HTTP 503", is_error=True), c, plan(c, {})).outcome == "source_error"


def test_structural_empty_and_path_predicates():
    assert structural_empty({"success": True, "num_results": 0, "rows": []})
    assert structural_empty({"success": True, "rows": []})
    assert not structural_empty({"success": True, "rows": [], "record": {"id": 1}})
    assert not structural_empty({"count": 0, "rows": [1]})
    obj = {"error": "Failed: Status Code: 404", "items": [{"id": None}, {"id": 2}]}
    assert jp_test(obj, "$.error =~ '(?i)status code: 404'")
    assert jp_test(obj, "$.items[*].id == 2") and jp_test(obj, "$.items[*].id != null")
    assert not jp_test(obj, "$.missing != null") and jp_test(obj, "$.error")
    assert json.loads(json.dumps(obj)) == obj

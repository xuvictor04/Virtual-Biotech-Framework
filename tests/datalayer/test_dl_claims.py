"""Claims against data-layer results (DATA_LAYER.md §15.3): every row of the status table for
``supports: presence`` and ``supports: absence``, the coverage values (covered, censored with and
without its statement, unknown, not_covered, partial_unknown), leakage risk, evidence nature, calls
without a status, the record_claims schema and the statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

from vbt.audit.claims import (
    EvidenceContext,
    carries_censor_statement,
    censor_phrase,
    check_evidence,
    claim_stats,
    refresh_claims,
    validate_claims,
)
from vbt.tools.provenance import provenance_tools

TOOL = "mcp__target__get_target_info"
CENSOR = 'Missing rows are citable only as "not significant at padj 0.10, or not tested".'


def _call(status=None, *, tool=TOOL, is_error=False, error_kind=None, coverage=None, statement=None,
          returned=None, total=None, order_verified=True, nature=None, caveat=None, leakage_risk=None):
    rec = {"tool": tool, "agent": "genomics-analyst", "started_at": "2026-10-06T09:00:00+00:00",
           "pending": False, "is_error": is_error}
    if error_kind:
        rec["error_kind"] = error_kind
    if status is not None:
        dp = {"prov": "dp_0123456789ab", "status": status, "coverage": coverage, "coverage_statement": statement,
              "source": "open_targets@25.09", "tables": [{"name": "target", "fingerprint": "fp1:sha256:abc"}],
              "returned": returned, "total": total, "truncated": None if total is None else returned != total,
              "row_keys_sha256": "f" * 64, "served_by": "upstream", "evidence_nature": nature,
              "leakage_risk": leakage_risk, "order_verified": order_verified}
        if caveat:
            dp["evidence_caveat"] = caveat
        rec.update(result_status=status, data_provenance=dp, prov=dp["prov"], coverage=coverage)
    return rec


def _ctx(tmp_path: Path, **calls) -> EvidenceContext:
    return EvidenceContext(tmp_path, {}, calls)


def _check(tmp_path, call, *, supports=None, text="PCSK9 is expressed in liver", confidence="moderate",
           n_evidence=1):
    ev = {"kind": "tool_call", "tool_use_id": "t1"}
    if supports is not None:
        ev["supports"] = supports
    return check_evidence(ev, _ctx(tmp_path, t1=call), claim={"text": text, "confidence": confidence,
                                                              "n_evidence": n_evidence})


def _ok(result):
    entry, problems, _warnings = result
    assert problems == [], problems
    return entry


def _problem(result, needle):
    entry, problems, _warnings = result
    assert entry["verified"] is False and entry["evidence_status"] == "unresolved"
    assert any(needle in p for p in problems), problems
    return problems


def _warned(result, needle):
    _entry, _problems, warnings = result
    assert any(needle in w for w in warnings), warnings


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize("supports", [None, "presence", "absence"])
@pytest.mark.parametrize("kind", ["not_found", "not_ready", "tool_defect", None])
def test_errors_of_any_kind_are_problems(tmp_path, supports, kind):
    probs = _problem(_check(tmp_path, _call(is_error=True, error_kind=kind), supports=supports),
                     "a failed query cannot support a finding")
    if kind:
        assert f"({kind})" in probs[0]


# --------------------------------------------------------------------------- ok and partial


def test_ok_presence_is_verified_and_copies_provenance(tmp_path):
    entry = _ok(_check(tmp_path, _call("ok", returned=2, total=2)))
    assert entry["verified"] is True and entry["evidence_status"] == "verified"
    assert entry["prov"] == "dp_0123456789ab" and entry["result_status"] == "ok"
    assert entry["source"] == "open_targets" and entry["release"] == "25.09"
    assert entry["fingerprints"] == {"target": "fp1:sha256:abc"}
    assert "supports" not in entry                  # default presence is not written


def test_ok_absence_is_verified_with_a_warning(tmp_path):
    res = _check(tmp_path, _call("ok", returned=2, total=2), supports="absence")
    entry = _ok(res)
    assert entry["evidence_status"] == "verified" and entry["supports"] == "absence"
    _warned(res, "absence claim cites rows; the note must say which rows show it")


def test_partial_presence_warns_n_of_m(tmp_path):
    res = _check(tmp_path, _call("partial", returned=20, total=61))
    assert _ok(res)["evidence_status"] == "verified"
    _warned(res, "20 of 61 rows")
    res = _check(tmp_path, _call("partial", returned=20, total=None))
    _warned(res, "20 of an unknown number of rows")


def test_partial_top_claim_needs_a_verified_order(tmp_path):
    claim = "PCSK9 is among the top targets by score"
    _ok(_check(tmp_path, _call("partial", returned=20, total=61, order_verified=True), text=claim))
    _problem(_check(tmp_path, _call("partial", returned=20, total=61, order_verified=None), text=claim),
             "the claim says 'top' but the result's order is not verified")
    # "top" only as a word
    _ok(_check(tmp_path, _call("partial", returned=20, total=61, order_verified=None), text="topology changes"))


def test_partial_absence_is_verified_with_a_warning(tmp_path):
    res = _check(tmp_path, _call("partial", returned=20, total=61), supports="absence")
    assert _ok(res)["evidence_status"] == "verified"
    _warned(res, "partial result")


# --------------------------------------------------------------------------- empty


def test_empty_presence_is_a_problem(tmp_path):
    _problem(_check(tmp_path, _call("empty", coverage="covered", returned=0, total=0)),
             "an empty result cannot support a positive finding")


def test_empty_absence_with_covered_coverage_is_an_absence(tmp_path):
    res = _check(tmp_path, _call("empty", coverage="covered", returned=0, total=0), supports="absence",
                 text="No ChEMBL drug targets PCSK9 in Open Targets")
    entry = _ok(res)
    assert entry["verified"] is True and entry["evidence_status"] == "absence"
    assert entry["coverage"] == "covered" and entry["supports"] == "absence"


@pytest.mark.parametrize("coverage", ["unknown", "not_covered", "partial_unknown", None])
def test_empty_absence_with_other_coverage_is_a_problem_naming_it(tmp_path, coverage):
    probs = _problem(_check(tmp_path, _call("empty", coverage=coverage, returned=0, total=0), supports="absence"),
                     "as an absence; only 'covered' coverage")
    assert repr(coverage or "unknown") in probs[0]


def test_censored_absence_needs_the_censor_statement(tmp_path):
    call = _call("empty", coverage="censored", statement=CENSOR, returned=0, total=0)
    entry = _ok(_check(tmp_path, call, supports="absence",
                       text="BRD4 is not differentially expressed under JQ1 (not significant at padj 0.10, "
                            "or not tested)."))
    assert entry["evidence_status"] == "absence" and entry["coverage_statement"] == CENSOR
    _problem(_check(tmp_path, call, supports="absence", text="BRD4 is not differentially expressed under JQ1."),
             "must carry the censor statement 'not significant at padj 0.10, or not tested'")
    _problem(_check(tmp_path, _call("empty", coverage="censored", returned=0, total=0), supports="absence"),
             "censor statement was not recorded")
    # presence on a censored empty result is still a positive-finding problem
    _problem(_check(tmp_path, call, text="not significant at padj 0.10, or not tested"),
             "cannot support a positive finding")


def test_censor_phrase_helpers():
    assert censor_phrase(CENSOR) == "not significant at padj 0.10, or not tested"
    assert censor_phrase("A missing row means the gene's effect was not tested.") == \
        "A missing row means the gene's effect was not tested"
    assert carries_censor_statement("X (Not significant at padj 0.10,\n or not tested).", CENSOR)
    assert not carries_censor_statement("X", "")


@pytest.mark.parametrize("supports", [None, "absence"])
def test_empty_unverified_is_never_citable(tmp_path, supports):
    _problem(_check(tmp_path, _call("empty_unverified", coverage="covered", returned=0, total=None),
                    supports=supports), "empty_unverified")


# --------------------------------------------------------------------------- leakage and evidence nature


@pytest.mark.parametrize("supports", [None, "absence"])
@pytest.mark.parametrize("status", ["ok", "empty"])
def test_leakage_risk_is_a_problem_both_ways(tmp_path, supports, status):
    _problem(_check(tmp_path, _call(status, coverage="covered", returned=1, total=1, leakage_risk=True),
                    supports=supports), "may postdate the evidence ceiling")


def test_no_leakage_risk_is_fine(tmp_path):
    _ok(_check(tmp_path, _call("ok", returned=1, total=1, leakage_risk=False)))


def test_evidence_nature_warns_and_blocks_a_lone_strong_claim(tmp_path):
    caveat = "literature co-occurrence, not function"
    call = _call("ok", returned=3, total=3, nature="literature_cooccurrence", caveat=caveat)
    res = _check(tmp_path, call, confidence="moderate")
    assert _ok(res)["evidence_status"] == "verified"
    _warned(res, caveat)
    _problem(_check(tmp_path, call, confidence="strong", n_evidence=1), "only evidence of a 'strong' claim")
    res = _check(tmp_path, call, confidence="strong", n_evidence=2)
    _ok(res)
    _warned(res, caveat)
    # the caveat alone (no kind) is enough
    res = _check(tmp_path, _call("ok", returned=1, total=1, caveat=caveat))
    _warned(res, caveat)


def test_evidence_nature_on_an_absence_follows_its_status(tmp_path):
    call = _call("empty", coverage="covered", returned=0, total=0, nature="literature_cooccurrence",
                 caveat="literature co-occurrence, not function")
    res = _check(tmp_path, call, supports="absence", confidence="strong")
    assert _ok(res)["evidence_status"] == "absence"
    assert not any("caveat" in w for w in res[2])


# --------------------------------------------------------------------------- no status


@pytest.mark.parametrize("supports", [None, "absence"])
def test_mcp_call_without_status_warns(tmp_path, supports):
    res = _check(tmp_path, _call(None), supports=supports)
    assert _ok(res)["evidence_status"] == "verified"
    _warned(res, "has no data-layer status")
    # old traces indexed by audit.provenance read result_status 'unknown'
    res = _check(tmp_path, {**_call(None), "result_status": "unknown"}, supports=supports)
    _warned(res, "has no data-layer status")


def test_non_mcp_and_provenance_calls_get_no_status_warning(tmp_path):
    entry, problems, warnings = _check(tmp_path, _call(None, tool="Bash"))
    assert not problems and not warnings and entry["evidence_status"] == "verified"
    _entry, problems, warnings = _check(tmp_path, _call(None, tool="mcp__provenance__list_artifacts"))
    assert not problems and warnings == ["tool call 't1' is a provenance bookkeeping call, not evidence"]


def test_supports_must_be_presence_or_absence(tmp_path):
    _problem(_check(tmp_path, _call("ok", returned=1, total=1), supports="maybe"), "'supports' must be")


def test_row_key_is_kept(tmp_path):
    ev = {"kind": "tool_call", "tool_use_id": "t1", "row_key": ["CHEMBL25", "ENSG00000169174", None]}
    entry, problems, _w = check_evidence(ev, _ctx(tmp_path, t1=_call("ok", returned=1, total=1)))
    assert not problems and entry["row_key"] == ["CHEMBL25", "ENSG00000169174", None]


# --------------------------------------------------------------------------- batches, refresh, stats


def test_validate_claims_uses_the_claim_text_confidence_and_evidence_count(tmp_path):
    calls = {"e": _call("empty", coverage="censored", statement=CENSOR, returned=0, total=0),
             "n": _call("ok", returned=3, total=3, nature="literature_cooccurrence", caveat="co-occurrence"),
             "x": _call("ok", returned=1, total=1)}
    ctx = EvidenceContext(tmp_path, {}, calls)
    good = [{"id": "C1", "text": "Not significant at padj 0.10, or not tested.", "confidence": "strong",
             "evidence": [{"kind": "tool_call", "tool_use_id": "e", "supports": "absence"}]},
            {"id": "C2", "text": "Co-mentioned", "confidence": "strong",
             "evidence": [{"kind": "tool_call", "tool_use_id": "n"}, {"kind": "tool_call", "tool_use_id": "x"}]}]
    res = validate_claims(good, ctx, strict=True)
    assert res.ok, res.errors
    assert res.claims[0]["evidence"][0]["evidence_status"] == "absence" and res.claims[0]["n_verified"] == 1
    bad = [{"id": "C3", "text": "Co-mentioned", "confidence": "strong",
            "evidence": {"kind": "tool_call", "tool_use_id": "n"}}]
    res = validate_claims(bad, ctx, strict=True)
    assert not res.ok and "only evidence of a 'strong' claim" in res.errors[0]


def test_refresh_and_stats_count_absences(tmp_path):
    calls = {"e": _call("empty", coverage="covered", returned=0, total=0), "x": _call("ok", returned=1, total=1)}
    ctx = EvidenceContext(tmp_path, {}, calls)
    res = validate_claims([{"id": "C1", "text": "No drug targets X", "evidence": [
        {"kind": "tool_call", "tool_use_id": "e", "supports": "absence"},
        {"kind": "tool_call", "tool_use_id": "x"}]}], ctx)
    assert res.ok, res.errors
    refreshed = refresh_claims(res.claims, ctx)
    assert [e["evidence_status"] for e in refreshed[0]["evidence"]] == ["absence", "verified"]
    stats = claim_stats(refreshed)
    assert stats["n_absence_evidence"] == 1 and stats["n_verified_evidence"] == 1
    assert stats["claims_without_verified_evidence"] == []
    # a stored absence claim whose call later reads differently becomes unresolved on refresh
    calls["e"] = _call("empty", coverage="unknown", returned=0, total=0)
    again = refresh_claims(res.claims, EvidenceContext(tmp_path, {}, calls))
    assert again[0]["evidence"][0]["evidence_status"] == "unresolved" and "coverage 'unknown'" in \
        again[0]["evidence"][0]["problem"]


# --------------------------------------------------------------------------- schema


def test_record_claims_schema_gains_supports_and_row_key_with_kinds_unchanged():
    tool = next(t for t in provenance_tools() if t.name == "mcp__provenance__record_claims")
    ev = tool.input_schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]["properties"]
    assert ev["kind"]["enum"] == ["artifact", "figure", "table", "code", "tool_call", "citation"]
    assert ev["supports"]["enum"] == ["presence", "absence"]
    assert "covered" in ev["supports"]["description"] and "empty" in ev["supports"]["description"]
    assert ev["row_key"]["type"] == "array"
    assert "absence" in tool.description
    required = tool.input_schema["properties"]["claims"]["items"].get("required")
    assert required == ["id", "text", "evidence"]

"""The six correctness tests (docs/DATA_LAYER.md §19), written before the gateway exists.

Each call is its own case, run per mode:

* ``off`` (no gateway): the correct-behaviour assertion is a strict ``xfail`` today, and the
  ``test_ctN_today`` pins assert today's exact wrong answer, so an upstream fix shows up as a
  failing pin instead of a silent pass;
* ``enforce``: the phase-1 exit gate (skipped until ``vbt.datalayer.gateway`` and the shipped
  overlays exist).

The unmodified upstream servers run through ``MCPBridge`` (stdio, ``python -B``) on the
OT-25.09-shaped and Tahoe-shaped fixtures of ``dl_fixtures``; expected answers come from the
pyarrow oracles there. Positive controls must pass in both modes. Claim checks go through
``Runtime`` with the mock provider (``dl_upstream.run_with_runtime``). No network: PubMed runs
against the local E-utilities stub and cBioPortal against the ``pybioportal`` stub.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import dl_fixtures as F
from dl_upstream import (CallResult, DataEnv, cases, gateway_missing, mode_params, needs_arrow, needs_fastmcp,
                         needs_pubmed_hook, pubmed_hook_missing, readiness_problems, run_with_runtime)

pytestmark = [needs_arrow, needs_fastmcp, pytest.mark.correctness]

Check = Callable[[CallResult, Path], None]


# ---------------------------------------------------------------------------- assertion helpers


def dump(obj: Any) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


def mentions(obj: Any, text: str) -> bool:
    return text in dump(obj)


def dig(obj: Any, *path: Any) -> Any:
    """``obj[p0][p1]...`` or None (never raises, so off-mode failures are assertion failures)."""
    for p in path:
        if isinstance(obj, Mapping) and p in obj:
            obj = obj[p]
        elif isinstance(obj, list) and isinstance(p, int) and -len(obj) <= p < len(obj):
            obj = obj[p]
        else:
            return None
    return obj


def show(r: CallResult) -> str:
    return r.text[:600]


def ok(r: CallResult) -> None:
    assert not r.is_error, f"expected a result, got an error: {show(r)}"


def err(r: CallResult, kind: str) -> dict[str, Any]:
    assert r.is_error, f"expected {kind}, got a success: {show(r)}"
    assert r.kind == kind, f"expected kind {kind}, got {r.kind}: {show(r)}"
    return r.payload


def not_found(r: CallResult) -> dict[str, Any]:
    payload = err(r, "not_found")
    assert payload.get("citable") is False, payload
    assert payload.get("tried"), payload
    return payload


def status(r: CallResult) -> str | None:
    return r.status or r.header.get("status")


def empty(r: CallResult, coverage: str | None = None) -> dict[str, Any]:
    ok(r)
    assert status(r) == "empty", f"expected status empty: {show(r)}"
    if coverage is not None:
        assert r.header.get("coverage") == coverage, r.header
    return r.header


def served_full(r: CallResult) -> None:
    """Derived or repaired service, with the row keys in the provenance record."""
    assert r.header.get("served_by") in ("derived", "repaired"), r.header
    prov = r.provenance
    summary = prov.to_dict() if hasattr(prov, "to_dict") else prov
    if summary is not None:
        assert dig(summary, "result", "row_keys") or dig(summary, "result", "row_keys_sha256"), summary


def grain(header: Mapping[str, Any], name: str) -> dict[str, Any]:
    g = dig(header, "grains", name)
    return g if isinstance(g, dict) else {"total": g}


def excluded_unknown(header: Mapping[str, Any], column: str) -> Any:
    return dig(header, "excluded_unknown", column)


def keys(rows: list[Any], cols: tuple[str, ...]) -> list[tuple]:
    return [tuple(dig(row, c) for c in cols) for row in rows]


def resolution_text(r: CallResult) -> str:
    return dump({k: r.header.get(k) for k in ("resolved", "resolution_summary", "notes")})


# ---------------------------------------------------------------------------- shared calls

U, T, T2 = F.UNKNOWN_GENE, F.T, F.T2
CALLS: dict[str, tuple[str, str, dict[str, Any]]] = {
    # CT-1
    "symbol": ("target", "get_target_info", {"target_id": "PCSK9"}),
    "versioned": ("target", "get_target_info", {"target_id": "ENSG00000169174.12"}),
    "lowercase": ("target", "get_target_info", {"target_id": "ensg00000169174"}),
    "alias": ("target", "get_target_info", {"target_id": "NARC1"}),
    "unknown": ("target", "get_target_info", {"target_id": U}),
    "disease_curie": ("disease", "get_disease_info", {"disease_id": "EFO:0000685"}),
    "drug_case": ("drug", "get_drug_info", {"drug_id": "chembl25"}),
    "search_tp53": ("target", "search_targets_by_name", {"query": "TP53", "limit": 1}),
    "pmcid": ("pubmed", "fetch_abstracts", {"pmids": ["PMC1234"]}),
    # CT-2
    "safety_unknown": ("target", "get_target_safety_profile", {"target_id": U}),
    "prio_unknown": ("target", "get_target_prioritisation_scores", {"target_id": U}),
    "probes_unknown": ("target", "get_chemical_probes", {"target_id": U}),
    "profile_unknown": ("target", "get_comprehensive_target_profile", {"target_id": U}),
    "safety_tp53": ("target", "get_target_safety_profile", {"target_id": F.TP53}),
    "probes_tp53": ("target", "get_chemical_probes", {"target_id": F.TP53}),
    "probes_null": ("target", "get_chemical_probes", {"target_id": F.TP53BP1}),
    "hallmarks_tp53": ("target", "get_target_hallmarks", {"target_id": F.TP53}),
    "tep_tp53": ("target", "get_target_tep", {"target_id": F.TP53}),
    # items of a parent record that carry their own `id`: the parent's target_id never filters them
    "probes_pcsk9": ("target", "get_chemical_probes", {"target_id": F.PCSK9}),
    "tract_pcsk9": ("target", "get_target_tractability", {"target_id": F.PCSK9}),
    "tract_drug_symbol": ("drug", "get_target_tractability", {"target_id": "pcsk9"}),
    "cbio_phantom": ("clinicaltrials", "get_clinical_data", {"study_id": "study_x", "sample_ids": ["S-01", "NOPE-01"]}),
    "cbio_empty_list": ("clinicaltrials", "get_clinical_data", {"study_id": "study_x", "sample_ids": []}),
    "cbio_unknown_study": ("clinicaltrials", "get_clinical_data", {"study_id": "nope_study"}),
    "cbio_patients": ("clinicaltrials", "get_clinical_data", {"study_id": "study_x"}),
    # CT-3
    "pgx_drug": ("drug", "get_pharmacogenomics", {"drug_id": F.CHEMBL3}),
    "pgx_drug_on_target": ("target", "get_pharmacogenomics", {"drug_id": F.CHEMBL3}),
    "pgx_two_args": ("drug", "get_pharmacogenomics", {"target_id": T, "drug_id": F.CHEMBL3}),
    "pgx_chembl99": ("drug", "get_pharmacogenomics", {"drug_id": F.CHEMBL99}),
    "search_drugs": ("drug", "search_drugs", {"target_id": F.PCSK9}),
    "mouse": ("target", "get_mouse_phenotype", {"target_id": F.PCSK9}),
    "go": ("pathway", "get_gene_ontology", {"target_id": F.PCSK9}),
    "niddm": ("disease", "search_diseases_by_name", {"query": "NIDDM"}),
    "retired": ("disease", "get_disease_info", {"disease_id": F.T2D_RETIRED}),
    "epmc": ("association", "get_evidence_by_publication", {"pmid": F.PMID_EPMC}),
    "go_search": ("pathway", "search_go_terms", {"query": "GO:1"}),
    "biosample": ("expression", "search_biosample_ontology", {"query": F.BIOSAMPLE_SYNONYM}),
    "neg_go": ("pathway", "get_gene_ontology", {"target_id": F.TP53}),
    "neg_mouse": ("target", "get_mouse_phenotype", {"target_id": F.TP53}),
    "neg_search_drugs": ("drug", "search_drugs", {"target_id": F.TP53}),
    "neg_pgx": ("drug", "get_pharmacogenomics", {"target_id": T2, "drug_id": F.CHEMBL25}),
    # CT-4
    "known_drug_top5": ("drug", "search_known_drugs", {"target_id": T, "limit": 5}),
    "adverse_top1": ("drug", "get_drug_adverse_events", {"drug_id": F.CHEMBL559288, "limit": 1}),
    "l2g_top5": ("genetics", "query_l2g_predictions", {"gene_id": F.G_L2G, "min_score": 0.05, "limit": 5}),
    "interactions_top5": ("interaction", "get_interactions", {"target_id": T, "limit": 5}),
    # CT-5
    "datasource_unknown": ("association", "filter_by_datasource",
                           {"datasource": "gwas_catalog", "target_id": T, "output_path": "x"}),
    "datasource_indirect": ("association", "filter_by_datasource",
                            {"datasource": "eva", "target_id": T, "include_indirect": True, "output_path": "x2"}),
    "coloc_typo": ("genetics", "get_colocalisation_by_chromosome", {"chromosome": "19", "method": "colc"}),
    "sort_unknown": ("target", "prioritize_targets", {"sort_by": "nonexistent"}),
    "literal_name": ("disease", "search_diseases_by_name", {"query": "mellitus (T2D)"}),
    "pgx_drug_server": ("drug", "get_pharmacogenomics", {"target_id": T, "drug_id": F.CHEMBL3}),
    "pgx_target_server": ("target", "get_pharmacogenomics", {"target_id": T, "drug_id": F.CHEMBL3}),
    "limit_zero": ("drug", "search_known_drugs", {"target_id": T, "limit": 0}),
    "limit_negative": ("drug", "search_known_drugs", {"target_id": T, "limit": -1}),
    "tahoe_pooled": ("functional_genomics", "query_drug_perturbation",
                     {"drug_name": F.BORTEZOMIB, "cell_line_id": F.A549}),
    "tahoe_dose": ("functional_genomics", "query_drug_perturbation",
                   {"drug_name": F.BORTEZOMIB, "cell_line_id": F.A549, "concentration": 0.5}),
    "tahoe_float32": ("functional_genomics", "query_drug_perturbation",
                      {"drug_name": F.BORTEZOMIB, "cell_line_id": F.A549, "concentration": 0.05}),
    "tahoe_two_plates": ("functional_genomics", "query_drug_perturbation",
                         {"drug_name": F.BORTEZOMIB, "cell_line_id": F.A549, "concentration": F.TWO_PLATE_DOSE}),
    "direct_indirect": ("association", "compare_direct_indirect", {"target_id": T, "limit": 10, "output_path": "y"}),
    # CT-6
    "no_safety": ("target", "prioritize_targets", {"no_safety_events": True}),
    "year_min_2024": ("association", "query_evidence", {"target_id": T, "min_year": 2024}),
    "year_max_2020": ("association", "query_evidence", {"target_id": T, "max_year": 2020}),
    "year_min_2030": ("association", "query_evidence", {"target_id": T, "min_year": 2030}),
    "phenotype_pcs": ("disease", "find_diseases_by_phenotype", {"phenotype_id": F.SEIZURE, "evidence_type": "PCS"}),
    "phenotype_all": ("disease", "find_diseases_by_phenotype", {"phenotype_id": F.SEIZURE}),
    "phenotype_curie": ("disease", "find_diseases_by_phenotype", {"phenotype_id": "HP:0001250"}),
    "phenotype_unknown": ("disease", "find_diseases_by_phenotype", {"phenotype_id": F.UNKNOWN_HP}),
    "pheno_x": ("disease", "get_disease_phenotypes", {"disease_id": F.PHENO["X"]}),
    "pheno_w": ("disease", "get_disease_phenotypes", {"disease_id": F.PHENO["W"]}),
    "pheno_z": ("disease", "get_disease_phenotypes", {"disease_id": F.PHENO["Z"]}),
    "study_min_size": ("genetics", "get_study_metadata", {"min_sample_size": 100000, "limit": 20}),
    # positive controls
    "control_target": ("target", "get_target_info", {"target_id": F.PCSK9}),
    "control_disease": ("disease", "get_disease_info", {"disease_id": F.T2D}),
    "control_drug": ("drug", "get_drug_info", {"drug_id": F.CHEMBL25}),
    "control_known_drug": ("drug", "search_known_drugs", {"target_id": T}),
}


def run_case(live: Callable[[str], Any], fixture_ready: Callable[[Any], None], mode: str, case: str,
             *, cached: bool = True) -> CallResult:
    bridge = live(mode)
    fixture_ready(bridge)
    server, tool, args = CALLS[case]
    if server == "pubmed" and pubmed_hook_missing():
        pytest.skip(pubmed_hook_missing())
    return bridge.call(server, tool, args, cached=cached)


# ---------------------------------------------------------------------------- positive controls


@pytest.mark.parametrize("mode", mode_params())
def test_positive_controls(mode: str, live, fixture_ready) -> None:
    """Known rows come back in both modes; before any deletion disease and drug are not degraded
    and search_known_drugs is ready."""
    r = run_case(live, fixture_ready, mode, "control_target")
    ok(r)
    assert r.obj.get("id") == F.PCSK9 and r.obj.get("approvedSymbol") == "PCSK9"
    r = run_case(live, fixture_ready, mode, "control_disease")
    ok(r)
    assert r.obj.get("id") == F.T2D
    r = run_case(live, fixture_ready, mode, "control_drug")
    ok(r)
    assert r.obj.get("id") == F.CHEMBL25
    r = run_case(live, fixture_ready, mode, "control_known_drug")
    ok(r)
    assert r.rows("drugs"), show(r)
    if mode == "enforce":
        problems = readiness_problems(live(mode).readiness_snapshot(), ("disease", "drug_molecule", "known_drug"))
        assert not problems, problems


# ============================================================================ CT-1


def _resolved_to_pcsk9(rule: str, *, note: bool) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        assert r.obj.get("id") == F.PCSK9, show(r)
        assert rule in resolution_text(r), r.header
        assert mentions(r.header, "ensembl_gene"), r.header
        if note:
            assert r.header.get("notes"), r.header
    return check


def _not_found(r: CallResult, root: Path) -> None:
    not_found(r)


def _ct1_disease(r: CallResult, root: Path) -> None:
    ok(r)
    assert r.obj.get("id") == F.RA, show(r)
    assert "normalized:curie_colon_to_underscore" in resolution_text(r), r.header


def _ct1_drug(r: CallResult, root: Path) -> None:
    ok(r)
    assert r.obj.get("id") == F.CHEMBL25, show(r)


def _ct1_search(r: CallResult, root: Path) -> None:
    ok(r)
    first = dig(r.obj, "results", 0) or {}
    assert first.get("id") == F.TP53, show(r)
    assert first.get("match") == "exact", first


CT1: dict[str, Check] = {
    "symbol": _resolved_to_pcsk9("label_exact:approvedSymbol", note=False),
    "versioned": _resolved_to_pcsk9("normalized:strip_version", note=True),
    "lowercase": _resolved_to_pcsk9("normalized:upper", note=True),
    "alias": _resolved_to_pcsk9("synonym:alias", note=True),
    "unknown": _not_found,
    "disease_curie": _ct1_disease,
    "drug_case": _ct1_drug,
    "search_tp53": _ct1_search,
}


@pytest.mark.parametrize("case,mode", cases(CT1))
def test_ct1_identifiers_resolved_or_rejected(case: str, mode: str, live, fixture_ready, ot_root) -> None:
    CT1[case](run_case(live, fixture_ready, mode, case), ot_root)


@needs_pubmed_hook
@pytest.mark.parametrize("case,mode", cases(["pmcid"]))
def test_ct1_pmcid_is_rejected_before_the_service(case: str, mode: str, live, fixture_ready, eutils) -> None:
    eutils.clear()
    r = run_case(live, fixture_ready, mode, case, cached=False)
    payload = err(r, "invalid_argument")
    assert "pmcid" in (payload.get("looks_like") or []), payload
    assert eutils.requests() == [], eutils.requests()


CT1_PINS: dict[str, Callable[[CallResult], None]] = {
    **{c: (lambda r, v=CALLS[c][2]["target_id"]: _pin_lookup_miss(r, f"Target {v} not found"))
       for c in ("symbol", "versioned", "lowercase", "alias", "unknown")},
    "disease_curie": lambda r: _pin_lookup_miss(r, "Disease EFO:0000685 not found"),
    "drug_case": lambda r: _pin_lookup_miss(r, "Drug chembl25 not found"),
    "search_tp53": lambda r: _pin_first_result(r, F.TP53BP1),
}


def _pin_lookup_miss(r: CallResult, message: str) -> None:
    """An explicit-not-found envelope that the bridge passes as a non-error, citable result."""
    assert r.is_error is False, show(r)
    assert r.obj.get("error") == message, show(r)


def _pin_first_result(r: CallResult, target: str) -> None:
    assert not r.is_error and dig(r.obj, "results", 0, "id") == target, show(r)


@pytest.mark.parametrize("case", sorted(CT1_PINS))
def test_ct1_today(case: str, live) -> None:
    CT1_PINS[case](live("off").call(*CALLS[case]))


@needs_pubmed_hook
def test_ct1_today_pmcid(live, eutils) -> None:
    eutils.clear()
    r = live("off").call(*CALLS["pmcid"], cached=False)
    assert not r.is_error and dig(r.obj, "articles", 0, "pmid") == "1234", show(r)
    assert [q["endpoint"] for q in eutils.requests()] == ["efetch.fcgi"]


# ---------------------------------------------------------------------------- runtime (claims)

_RUNTIME: dict[tuple[str, str], Any] = {}


async def runtime_outcome(key: str, mode: str, tmp_path_factory, data_env: DataEnv,
                          calls: list[tuple[str, str, dict[str, Any]]], claims: list[list[dict[str, Any]]],
                          **kwargs: Any) -> Any:
    """One scripted runtime run per (key, mode), shared by the claim cases."""
    if (key, mode) not in _RUNTIME:
        if mode == "enforce" and gateway_missing():
            pytest.skip(gateway_missing())
        _RUNTIME[(key, mode)] = await run_with_runtime(
            calls, claims, env=data_env, gateway=mode == "enforce",
            tmp_path=tmp_path_factory.mktemp(f"rt-{key}-{mode}"), **kwargs)
    return _RUNTIME[(key, mode)]


def claim(i: int, supports: str | None = None, text: str = "finding") -> dict[str, Any]:
    ev: dict[str, Any] = {"kind": "tool_call", "call": i}
    if supports:
        ev["supports"] = supports
    return {"id": f"C{i}-{supports or 'any'}", "text": text, "evidence": [ev]}


def rejected(result: Mapping[str, Any], phrase: str) -> None:
    assert result.get("ok") is False, result
    assert any(phrase in str(e) for e in result.get("errors") or []), result


def stored_evidence(outcome: Any, claim_id: str) -> dict[str, Any]:
    for c in outcome.stored_claims:
        if c.get("id") == claim_id:
            return (c.get("evidence") or [{}])[0]
    return {}


@pytest.mark.parametrize("case,mode", cases(["claim_unknown"]))
async def test_ct1_claim_citing_unknown_is_rejected(case, mode, tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct1", mode, tmp_path_factory, data_env, [CALLS["unknown"]], [[claim(0)]])
    assert out.end(0).get("is_error") is True, out.end(0)
    rejected(out.claim(0), "cites failed tool call")


async def test_ct1_today_claim(tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct1", "off", tmp_path_factory, data_env, [CALLS["unknown"]], [[claim(0)]])
    assert out.end(0).get("is_error") is False and out.claim(0).get("ok") is True, (out.end(0), out.claim(0))


# ============================================================================ CT-2


def _ct2_safety_tp53(r: CallResult, root: Path) -> None:
    header = empty(r, coverage="unknown")
    assert "not evidence of safety" in str(header.get("coverage_statement")), header
    assert "message" not in r.obj, show(r)


def _ct2_probes_tp53(r: CallResult, root: Path) -> None:
    empty(r, coverage="covered")


def _ct2_coverage_unknown(r: CallResult, root: Path) -> None:
    empty(r, coverage="unknown")


def _ct2_tep(r: CallResult, root: Path) -> None:
    """A null nested tep is not a verified absence: the gene table's existence coverage is not its own."""
    header = empty(r, coverage="unknown")
    assert "citable only as an absence" not in str(header.get("cite") or ""), header
    assert "message" not in r.obj and "has_tep" not in r.obj, show(r)


def _ct2_items(path: str, ids: list[Any]) -> Check:
    """A target that has items gets every one back (status ok, nothing excluded by its own target_id)."""
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        assert status(r) == "ok", r.header
        assert not r.header.get("excluded"), r.header
        assert [dig(x, "id") for x in r.rows(path)] == ids, show(r)
        assert r.header.get("returned") == len(ids) == r.header.get("total"), r.header
    return check


def _ct2_phantom(r: CallResult, root: Path) -> None:
    payload = err(r, "not_found")
    assert mentions(payload, "NOPE-01"), payload


def _ct2_empty_list(r: CallResult, root: Path) -> None:
    payload = err(r, "invalid_argument")
    assert mentions(payload, "min_items"), payload


def _ct2_unknown_study(r: CallResult, root: Path) -> None:
    err(r, "not_found")


def _ct2_patients(r: CallResult, root: Path) -> None:
    ok(r)
    patients = r.rows("patients")
    assert len(patients) == 2, show(r)
    assert grain(r.header, "patient").get("total") == 2, r.header


CT2: dict[str, Check] = {
    "safety_unknown": lambda r, root: _not_found(r, root),
    "prio_unknown": lambda r, root: _not_found(r, root),
    "probes_unknown": lambda r, root: _not_found(r, root),
    "profile_unknown": lambda r, root: _not_found(r, root),        # phase 2: the target_profile view
    "safety_tp53": _ct2_safety_tp53,
    "probes_tp53": _ct2_probes_tp53,
    "probes_null": _ct2_coverage_unknown,
    "hallmarks_tp53": _ct2_coverage_unknown,
    "tep_tp53": _ct2_tep,
    "probes_pcsk9": _ct2_items("chemical_probes", ["PROBE-1"]),
    "tract_pcsk9": _ct2_items("tractability", ["Approved Drug", "Approved Drug"]),
    "tract_drug_symbol": _ct2_items("tractability", ["Approved Drug", "Approved Drug"]),
    "cbio_phantom": _ct2_phantom,
    "cbio_empty_list": _ct2_empty_list,
    "cbio_unknown_study": _ct2_unknown_study,
    "cbio_patients": _ct2_patients,
}


@pytest.mark.parametrize("case,mode", cases(CT2))
def test_ct2_unknown_empty_and_outage(case: str, mode: str, live, fixture_ready, ot_root) -> None:
    CT2[case](run_case(live, fixture_ready, mode, case), ot_root)


CT2_CLAIM_CALLS = [CALLS["safety_tp53"], CALLS["probes_tp53"], CALLS["probes_pcsk9"]]
CT2_CLAIMS = [[claim(0, "presence")], [claim(0, "absence")], [claim(1, "absence")], [claim(1, "presence")],
              [claim(2, "absence", "PCSK9 has no chemical probes")]]


def _claims_safety_presence(out: Any) -> None:
    rejected(out.claim(0), "cannot support a positive finding")


def _claims_safety_absence(out: Any) -> None:
    rejected(out.claim(1), "coverage unknown")


def _claims_probes_absence(out: Any) -> None:
    assert out.claim(2).get("ok") is True, out.claim(2)
    ev = stored_evidence(out, "C1-absence")
    assert ev.get("evidence_status") == "absence", ev
    assert ev.get("coverage") == "covered", ev


def _claims_probes_presence(out: Any) -> None:
    rejected(out.claim(3), "cannot support a positive finding")


def _claims_probes_pcsk9_absence(out: Any) -> None:
    """PCSK9 has a probe: citing that call never records a verified absence."""
    assert out.end(2).get("is_error") is False, out.end(2)
    ev = stored_evidence(out, "C2-absence")
    assert ev.get("evidence_status") != "absence", ev
    assert any("absence claim cites rows" in str(w) for w in out.claim(4).get("warnings") or []), out.claim(4)


CT2_CLAIM_CASES = {"safety_presence": _claims_safety_presence, "safety_absence": _claims_safety_absence,
                   "probes_absence": _claims_probes_absence, "probes_presence": _claims_probes_presence,
                   "probes_pcsk9_absence": _claims_probes_pcsk9_absence}


@pytest.mark.parametrize("case,mode", cases(CT2_CLAIM_CASES))
async def test_ct2_claims(case, mode, tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct2", mode, tmp_path_factory, data_env, CT2_CLAIM_CALLS, CT2_CLAIMS)
    CT2_CLAIM_CASES[case](out)


@pytest.mark.parametrize("case,mode", cases(["tep_absence"]))
async def test_ct2_tep_absence_claim_is_rejected(case, mode, tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct2tep", mode, tmp_path_factory, data_env, [CALLS["tep_tp53"]],
                                [[claim(0, "absence", "TP53 has no Target Enabling Package")]])
    rejected(out.claim(0), "coverage unknown")


async def test_ct2_today_claims(tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct2", "off", tmp_path_factory, data_env, CT2_CLAIM_CALLS, CT2_CLAIMS)
    assert [c.get("ok") for c in out.claim_results] == [True, True, True, True, True], out.claim_results


CT2_PINS: dict[str, Callable[[CallResult], None]] = {
    "safety_unknown": lambda r: _pin_zero_success(r, "adverse_events"),
    "prio_unknown": lambda r: _pin_zero_success(r, "prioritisations"),
    "probes_unknown": lambda r: _pin_lookup_miss(r, f"Target {U} not found"),
    "profile_unknown": lambda r: _pin_profile_zero(r, num_drugs=0),
    "safety_tp53": lambda r: _pin_message(r, "No adverse event data for this target"),
    "probes_tp53": lambda r: _pin_message(r, "No chemical probes available for this target"),
    "probes_null": lambda r: _pin_message(r, "No chemical probes available for this target"),
    "cbio_phantom": lambda r: _pin_phantom(r),
    "cbio_empty_list": lambda r: _pin_lookup_miss(r, "No samples found"),
    "cbio_unknown_study": lambda r: _pin_legacy_error(r, "Failed to retrieve clinical data"),
    "cbio_patients": lambda r: _pin_patient_copies(r),
}


def _pin_zero_success(r: CallResult, key: str) -> None:
    assert not r.is_error and r.obj.get("success") is True and r.obj.get("count") == 0, show(r)
    assert r.rows(key) == [], show(r)


def _pin_message(r: CallResult, message: str) -> None:
    assert not r.is_error and r.obj.get("message") == message, show(r)


def _pin_profile_zero(r: CallResult, *, num_drugs: int) -> None:
    assert not r.is_error and r.obj.get("success") is True, show(r)
    assert dig(r.obj, "summary_stats", "num_drugs") == num_drugs, show(r)


def _pin_legacy_error(r: CallResult, text: str) -> None:
    assert r.is_error and r.kind is None and text in r.text, show(r)


def _pin_phantom(r: CallResult) -> None:
    assert not r.is_error and r.obj.get("sample_count") == 2, show(r)
    assert {"sampleId": "NOPE-01", "patientId": None} in r.rows("data"), show(r)


def _pin_patient_copies(r: CallResult) -> None:
    """Patient-level OS values copied onto every sample: P-01's survival appears twice."""
    rows = r.rows("data")
    assert not r.is_error and len(rows) == 3 and "patients" not in r.obj, show(r)
    assert sum(1 for x in rows if x.get("patientId") == "P-01" and x.get("OS_MONTHS") == "24.5") == 2, rows


@pytest.mark.parametrize("case", sorted(CT2_PINS))
def test_ct2_today(case: str, live) -> None:
    CT2_PINS[case](live("off").call(*CALLS[case]))


# ---- CT-2: a deleted table is not_ready, not an empty answer, and degrades only its readers

KNOWN_DRUG_READERS = {"drug.search_known_drugs", "target.search_known_drugs",
                      "target.get_comprehensive_target_profile"}
_DELETED: dict[str, Path] = {}


def deleted_known_drug(variant: str, ot_root: Path, tmp_path_factory) -> Path:
    """A fixture copy without ``known_drug/``; ``manifest`` keeps the (now stale) manifest."""
    if variant not in _DELETED:
        root = F.copy_fixture(ot_root, tmp_path_factory.mktemp(f"ot-no-known-drug-{variant}") / "25.09")
        F.delete_table(root, "known_drug")
        if variant == "no_manifest":
            (root / F.MANIFEST).unlink()
        _DELETED[variant] = root
    return _DELETED[variant]


def tool_name(name: str) -> str:
    """``mcp__drug__search_known_drugs`` and ``drug.search_known_drugs`` as ``drug.search_known_drugs``."""
    if name.startswith("mcp__"):
        server, _, tool = name[len("mcp__"):].partition("__")
        return f"{server}.{tool}"
    return name


DELETION_CALLS = [("drug", "search_known_drugs", {"target_id": T}), ("target", "get_target_info", {"target_id": T}),
                  ("target", "get_comprehensive_target_profile", {"target_id": T})]


@pytest.mark.parametrize("variant", ["manifest", "no_manifest"])
@pytest.mark.parametrize("mode", mode_params(xfail_off=True))
async def test_ct2_deleted_table_is_not_ready(variant: str, mode: str, ot_root, data_env, live_variant,
                                              tmp_path_factory) -> None:
    root = deleted_known_drug(variant, ot_root, tmp_path_factory)
    env = DataEnv(ot_root=root, output_dir=data_env.output_dir)
    bridge = live_variant(mode, f"no-known-drug-{variant}", ("target", "drug"), env)
    kd, info, profile = (bridge.call(*c) for c in DELETION_CALLS)
    payload = err(kd, "not_ready")
    assert mentions(payload.get("tables"), "open_targets.known_drug"), payload
    ok(info)
    assert info.obj.get("id") == T
    # phase 2: the target_profile view answers with the section marked unavailable, never num_drugs: 0
    ok(profile)
    assert status(profile) == "partial", profile.header
    assert dig(profile.obj, "known_drugs", "_vbt_unavailable"), show(profile)
    assert "num_drugs" not in profile.obj and dig(profile.obj, "summary_stats", "num_drugs") is None, show(profile)
    if mode == "enforce":
        snap = bridge.readiness_snapshot()
        assert any("known_drug" in p for p in readiness_problems(snap, ("known_drug",))), snap
    out = await runtime_outcome(f"ct2-deleted-{variant}", mode, tmp_path_factory, env, DELETION_CALLS, [],
                                preflight=True)
    degraded = out.manifest.get("degraded") or {}
    assert not degraded.get("servers"), degraded
    tools = {tool_name(t) for t in degraded.get("tools") or []}
    assert tools and tools <= KNOWN_DRUG_READERS, degraded
    for end in out.ends:
        assert end.get("is_error") or '"num_drugs": 0' not in str(end.get("output")), end


@pytest.mark.parametrize("variant", ["manifest", "no_manifest"])
async def test_ct2_today_deleted_table(variant: str, ot_root, data_env, live_variant, tmp_path_factory) -> None:
    root = deleted_known_drug(variant, ot_root, tmp_path_factory)
    env = DataEnv(ot_root=root, output_dir=data_env.output_dir)
    bridge = live_variant("off", f"no-known-drug-{variant}", ("target", "drug"), env)
    kd, info, profile = (bridge.call(*c) for c in DELETION_CALLS)
    _pin_legacy_error(kd, "Dataset 'known_drug' not found")
    assert not info.is_error and info.obj.get("id") == T
    _pin_profile_zero(profile, num_drugs=0)
    out = await runtime_outcome(f"ct2-deleted-{variant}", "off", tmp_path_factory, env, DELETION_CALLS, [],
                                preflight=True)
    # Today the doctor's aggregate check fails and every Open Targets server is marked degraded.
    assert set((out.manifest.get("degraded") or {}).get("servers") or {}) == {"target", "drug"}, out.manifest


# ============================================================================ CT-3


def _pgx_keys(r: CallResult) -> list[tuple]:
    return keys(r.rows("pgx_relationships"), F.PGX_KEY)


def _ct3_pgx(target_id: str | None, drug_id: str | None, *, note: bool = False) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        assert sorted(_pgx_keys(r)) == sorted(F.oracle_pgx(root, target_id=target_id, drug_id=drug_id)), show(r)
        if note:
            assert r.header.get("notes"), r.header
        served_full(r)
    return check


def _ct3_search_drugs(r: CallResult, root: Path) -> None:
    ok(r)
    assert [d.get("id") for d in r.rows("drugs")] == F.oracle_drugs_for_target(root, F.PCSK9), show(r)
    served_full(r)


def _ct3_mouse(r: CallResult, root: Path) -> None:
    ok(r)
    assert len(r.rows("phenotypes")) == len(F.oracle_mouse(root, F.PCSK9)) == 2, show(r)
    served_full(r)


def _ct3_go(r: CallResult, root: Path) -> None:
    ok(r)
    expected = F.oracle_go_items(root, F.PCSK9)
    assert sorted(g.get("id") for g in r.rows("go_terms")) == sorted(k[1] for k in expected), show(r)
    served_full(r)
    summary = r.provenance.to_dict() if hasattr(r.provenance, "to_dict") else r.provenance
    row_keys = dump(dig(summary, "result", "row_keys") or [])
    assert all(k[1] in row_keys and k[0] in row_keys for k in expected), summary


def _ct3_niddm(r: CallResult, root: Path) -> None:
    ok(r)
    assert [d.get("id") for d in r.rows("results")] == F.oracle_disease_search(root, "NIDDM") == [F.T2D], show(r)
    assert "synonym:exact" in resolution_text(r) or mentions(r.header, "synonym:exact"), r.header


def _ct3_retired(r: CallResult, root: Path) -> None:
    ok(r)
    assert r.obj.get("id") == F.T2D == F.oracle_retired(root, F.T2D_RETIRED)[0], show(r)
    assert "retired:obsoleteTerms" in resolution_text(r), r.header


def _ct3_epmc(r: CallResult, root: Path) -> None:
    ok(r)
    expected = F.oracle_evidence_by_publication(root, F.PMID_EPMC)
    assert len(expected) >= 1
    assert sorted(e.get("id") for e in r.rows("evidence")) == sorted(expected), show(r)
    served_full(r)


def _ct3_go_search(r: CallResult, root: Path) -> None:
    ok(r)
    got = {t.get("goId"): t.get("gene_count") for t in r.rows("go_terms")}
    assert got == F.oracle_go_search(root, "GO:1"), show(r)


def _ct3_biosample(r: CallResult, root: Path) -> None:
    ok(r)
    got = [b.get("biosampleId") for b in r.rows("biosamples")]
    assert got == F.oracle_biosample_search(root, F.BIOSAMPLE_SYNONYM), show(r)


def _ct3_negative(rows_key: str) -> Check:
    def check(r: CallResult, root: Path) -> None:
        assert r.kind != "tool_defect", show(r)
        header = empty(r)
        assert header.get("coverage"), header
        assert r.rows(rows_key) == [], show(r)
    return check


CT3: dict[str, Check] = {
    "pgx_drug": _ct3_pgx(None, F.CHEMBL3),
    "pgx_drug_on_target": _ct3_pgx(None, F.CHEMBL3),
    "pgx_two_args": _ct3_pgx(T, F.CHEMBL3),
    "pgx_chembl99": _ct3_pgx(None, F.CHEMBL99, note=True),
    "search_drugs": _ct3_search_drugs,
    "mouse": _ct3_mouse,
    "go": _ct3_go,
    "niddm": _ct3_niddm,
    "retired": _ct3_retired,
    "epmc": _ct3_epmc,
    "go_search": _ct3_go_search,
    "biosample": _ct3_biosample,
    "neg_go": _ct3_negative("go_terms"),
    "neg_mouse": _ct3_negative("phenotypes"),
    "neg_search_drugs": _ct3_negative("drugs"),
    "neg_pgx": _ct3_negative("pgx_relationships"),
}


@pytest.mark.parametrize("case,mode", cases(CT3))
def test_ct3_no_false_empty(case: str, mode: str, live, fixture_ready, ot_root) -> None:
    CT3[case](run_case(live, fixture_ready, mode, case), ot_root)


def _pin_zero_rows(key: str) -> Callable[[CallResult], None]:
    def pin(r: CallResult) -> None:
        assert not r.is_error and r.obj.get("count") == 0 and r.rows(key) == [], show(r)
    return pin


def _pin_pgx_violation(r: CallResult) -> None:
    """The two-argument call ignores ``drug_id`` (if/elif): a [CHEMBL25]-only row comes back."""
    drugs = [F.pgx_drug_ids(row) for row in r.rows("pgx_relationships")]
    assert not r.is_error and len(drugs) == 3 and [F.CHEMBL25] in drugs, drugs


CT3_PINS: dict[str, Callable[[CallResult], None]] = {
    "pgx_drug": _pin_zero_rows("pgx_relationships"),
    "pgx_drug_on_target": _pin_zero_rows("pgx_relationships"),
    "pgx_chembl99": _pin_zero_rows("pgx_relationships"),
    "pgx_two_args": _pin_pgx_violation,
    "search_drugs": _pin_zero_rows("drugs"),
    "mouse": _pin_zero_rows("phenotypes"),
    "go": _pin_zero_rows("go_terms"),
    "niddm": _pin_zero_rows("results"),
    "retired": lambda r: _pin_lookup_miss(r, f"Disease {F.T2D_RETIRED} not found"),
    "epmc": _pin_zero_rows("evidence"),
    "go_search": _pin_zero_rows("go_terms"),
    "biosample": _pin_zero_rows("biosamples"),
    "neg_pgx": lambda r: _pin_violating_rows(r, F.CHEMBL25),
}


def _pin_violating_rows(r: CallResult, drug_id: str) -> None:
    rows = r.rows("pgx_relationships")
    assert not r.is_error and rows and all(drug_id not in F.pgx_drug_ids(x) for x in rows), show(r)


@pytest.mark.parametrize("case", sorted(CT3_PINS))
def test_ct3_today(case: str, live) -> None:
    CT3_PINS[case](live("off").call(*CALLS[case]))


# ---- CT-3b: overlays forced to serve: pass / on_contradiction: tool_defect, and profile: fidelity

BOUND_ID_CASES = {"pgx_drug": lambda root: len(F.oracle_pgx(root, drug_id=F.CHEMBL3)),
                  "search_drugs": lambda root: len(F.oracle_drugs_for_target(root, F.PCSK9)),
                  "mouse": lambda root: len(F.oracle_mouse(root, F.PCSK9)),
                  "go": lambda root: len(F.oracle_go_items(root, F.PCSK9)),
                  "epmc": lambda root: len(F.oracle_evidence_by_publication(root, F.PMID_EPMC))}


def forced_overlays(dst: Path) -> Path:
    """The shipped overlays with every binding forced to ``serve: pass, on_contradiction: tool_defect``."""
    import yaml

    from dl_upstream import REPO

    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted((REPO / "configs" / "data" / "overlays").glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for binding in (data.get("tools") or {}).values():
            if isinstance(binding, dict):
                binding["serve"] = "pass"
                binding["on_contradiction"] = "tool_defect"
                binding.pop("derived", None)
                binding.pop("block", None)
        (dst / path.name).write_text(yaml.safe_dump(data, sort_keys=False))
    return dst


def ct3b_overrides(variant: str, tmp: Path) -> dict[str, Any]:
    if variant == "pass":
        return {"data": {"overlays_dir": str(forced_overlays(tmp / "overlays"))}}
    return {"data": {"gateway": {"profile": "fidelity"}}}


@pytest.mark.parametrize("variant", ["pass", "fidelity"])
@pytest.mark.parametrize("case,mode", cases([*BOUND_ID_CASES, "pgx_two_args", "niddm"], modes=("enforce",)))
def test_ct3b_pass_and_fidelity(case: str, mode: str, variant: str, ot_root, data_env, live_variant,
                                fixture_ready, tmp_path_factory) -> None:
    servers = ("drug", "target", "pathway", "association", "disease")
    bridge = live_variant(mode, f"ct3b-{variant}", servers, data_env,
                          ct3b_overrides(variant, tmp_path_factory.mktemp(f"ct3b-{variant}")))
    fixture_ready(bridge)
    r = bridge.call(*CALLS[case])
    if case in BOUND_ID_CASES:
        payload = err(r, "tool_defect")
        assert dig(payload, "witness", "total") == BOUND_ID_CASES[case](ot_root), payload
    elif case == "pgx_two_args":
        assert r.is_error, show(r)
    else:
        ok(r)
        assert status(r) == "empty_unverified", r.header


@pytest.mark.parametrize("variant", ["pass", "fidelity"])
@pytest.mark.parametrize("mode", [p for p in mode_params() if p.values[0] == "enforce"])
async def test_ct3b_unverified_empty_is_not_citable(variant: str, mode: str, data_env, tmp_path_factory) -> None:
    tmp = tmp_path_factory.mktemp(f"ct3b-claims-{variant}")
    out = await runtime_outcome(f"ct3b-{variant}", mode, tmp_path_factory, data_env, [CALLS["niddm"]],
                                [[claim(0, "presence")], [claim(0, "absence")]],
                                overrides=ct3b_overrides(variant, tmp))
    assert out.claim(0).get("ok") is False and out.claim(1).get("ok") is False, out.claim_results


# ============================================================================ CT-4


def _known_drug_keys(r: CallResult) -> list[tuple]:
    return keys(r.rows("drugs"), F.KNOWN_DRUG_KEY)


def _ct4_known_drug(mode: str) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        assert _known_drug_keys(r) == F.oracle_known_drug_topk(root, T, 5), show(r)
        if mode == "enforce":
            h, oracle = r.header, F.oracle_known_drug(root, T)
            assert h.get("total") == oracle["total"] == 35, h
            assert excluded_unknown(h, "phase") == oracle["unknown_phase"] == 2, h
            assert h.get("truncated") is True, h
            top = F.oracle_known_drug_topk(root, T, 5)
            assert grain(h, "drug") == {"returned": len({k[0] for k in top}), "total": oracle["drugs_total"]}, h
            assert r.obj.get("limit") == 5, show(r)
    return check


def _ct4_adverse(mode: str) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        assert keys(r.rows("adverse_events"), ("chembl_id", "meddraCode")) == F.oracle_adverse_topk(root), show(r)
        if mode == "enforce":
            assert r.header.get("truncated") is True, r.header
    return check


def _ct4_l2g(mode: str) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        assert keys(r.rows("results"), ("studyLocusId", "geneId")) == F.oracle_l2g_topk(root), show(r)
        if mode == "enforce":
            assert r.header.get("total") == len(F.oracle_l2g(root)) == 150, r.header
            assert r.header.get("truncated") is True, r.header
    return check


def _ct4_interactions(mode: str) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        (top,) = F.oracle_interactions_topk(root, T, 5).values()
        assert keys(r.rows("interactions"), F.INTERACTION_KEY) == top, show(r)
    return check


CT4 = {"known_drug_top5": _ct4_known_drug, "adverse_top1": _ct4_adverse, "l2g_top5": _ct4_l2g,
       "interactions_top5": _ct4_interactions}


@pytest.mark.parametrize("case,mode", cases(CT4))
def test_ct4_global_topk(case: str, mode: str, live, fixture_ready, ot_root) -> None:
    CT4[case](mode)(run_case(live, fixture_ready, mode, case), ot_root)


@pytest.fixture(scope="session")
def ot_two_sources(tmp_path_factory) -> Path:
    return F.build_ot_fixture(tmp_path_factory.mktemp("ot-two-sources") / "25.09",
                              interaction_sources=("intact", "string"))


@pytest.mark.parametrize("mode", mode_params(xfail_off=True))
def test_ct4_interactions_topk_per_source(mode: str, ot_two_sources, data_env, live_variant, fixture_ready) -> None:
    bridge = live_variant(mode, "two-sources", ("interaction",), DataEnv(ot_root=ot_two_sources))
    fixture_ready(bridge)
    r = bridge.call(*CALLS["interactions_top5"])
    ok(r)
    per_source = F.oracle_interactions_topk(ot_two_sources, T, 5)
    got = keys(r.rows("interactions"), F.INTERACTION_KEY)
    for src, top in per_source.items():
        assert [k for k in got if k[0] == src] == top, (src, got)
    assert "within sourceDatabase" in str(r.header.get("order")), r.header


@pytest.mark.parametrize("topk", [False, True], ids=["topk_off", "topk_on"])
@pytest.mark.parametrize("mode", [p for p in mode_params() if p.values[0] == "enforce"])
def test_ct4_inflation_limit(mode: str, topk: bool, data_env, live_variant, ot_root, fixture_ready) -> None:
    overrides = {"data": {"witness": {"max_inflate_rows": 10, "topk": topk}}}
    bridge = live_variant(mode, f"inflate-{topk}", ("drug",), data_env, overrides)
    fixture_ready(bridge)
    before = bridge.bridge.status().get("drug", {}).get("calls")
    r = bridge.call(*CALLS["known_drug_top5"], cached=False)
    if not topk:
        payload = err(r, "too_large")
        assert payload.get("subkind") == "unranked_truncation", payload
        assert bridge.bridge.status().get("drug", {}).get("calls") == before, "the upstream tool was called"
    else:
        payload = err(r, "tool_defect")
        assert dig(payload, "witness", "total") == F.oracle_known_drug(ot_root, T)["total"] == 35, payload


def _pin_known_drug(r: CallResult) -> None:
    phases = [d.get("phase") for d in r.rows("drugs")]
    assert not r.is_error and r.obj.get("count") == 5 and max(phases) == 3.0, phases


def _pin_adverse(r: CallResult) -> None:
    llrs = [e.get("llr") for e in r.rows("adverse_events")]
    assert not r.is_error and llrs == [10.0], llrs


def _pin_l2g(r: CallResult) -> None:
    scores = [x.get("score") for x in r.rows("results")]
    assert not r.is_error and max(scores) == 0.51, scores


def _pin_interactions(r: CallResult) -> None:
    scoring = [x.get("scoring") for x in r.rows("interactions")]
    assert not r.is_error and scoring == sorted(scoring) and len(scoring) == 5, scoring


CT4_PINS = {"known_drug_top5": _pin_known_drug, "adverse_top1": _pin_adverse, "l2g_top5": _pin_l2g,
            "interactions_top5": _pin_interactions}


@pytest.mark.parametrize("case", sorted(CT4_PINS))
def test_ct4_today(case: str, live) -> None:
    CT4_PINS[case](live("off").call(*CALLS[case]))


# ============================================================================ CT-5


def _ct5_invalid(valid: list[str] | None = None) -> Check:
    def check(r: CallResult, root: Path) -> None:
        payload = err(r, "invalid_argument")
        if valid is not None:
            assert sorted(payload.get("valid_values") or []) == valid, payload
    return check


def _ct5_datasource_unknown(r: CallResult, root: Path) -> None:
    payload = err(r, "invalid_argument")
    assert payload.get("valid_values") == F.oracle_datasources(root) == ["eva", "gwas_credible_sets"], payload


def _ct5_datasource_indirect(r: CallResult, root: Path) -> None:
    ok(r)
    got = sorted((x.get("targetId"), x.get("diseaseId"), x.get("datasourceId")) for x in r.rows("top_associations"))
    assert got == sorted(F.oracle_datasource_rows(root, "eva", T, indirect=True)), show(r)
    assert mentions(r.header.get("tables"), "association_by_datasource_indirect"), r.header


def _ct5_literal(r: CallResult, root: Path) -> None:
    ok(r)
    ids = [d.get("id") for d in r.rows("results")]
    assert ids == F.oracle_disease_search(root, "mellitus (T2D)") and F.T2D in ids, show(r)


def _ct5_pgx(r: CallResult, root: Path) -> None:
    ok(r)
    assert sorted(_pgx_keys(r)) == sorted(F.oracle_pgx(root, target_id=T, drug_id=F.CHEMBL3)), show(r)
    assert all(F.CHEMBL3 in F.pgx_drug_ids(x) for x in r.rows("pgx_relationships")), show(r)


def _tahoe_rows(r: CallResult) -> list[dict[str, Any]]:
    return r.rows("top_upregulated") + r.rows("top_downregulated")


def _ct5_tahoe_pooled(r: CallResult, root: Path) -> None:
    payload = err(r, "incomplete_key")
    assert payload.get("dimension") == "concentration", payload
    assert payload.get("values") == F.oracle_tahoe_concentrations(root, F.BORTEZOMIB, F.A549) == [0.05, 0.5, 5.0]
    assert payload.get("unit") == "uM", payload


def _ct5_tahoe_dose(r: CallResult, root: Path) -> None:
    ok(r)
    rows = _tahoe_rows(r)
    oracle = F.oracle_tahoe(root, F.BORTEZOMIB, F.A549, 0.5)
    assert rows and all(all(c in row for c in ("concentration", "plate", "gene_name")) for row in rows), rows
    row_keys = [(row.get("concentration"), row.get("plate"), row.get("gene_name")) for row in rows]
    assert len(row_keys) == len(set(row_keys)), row_keys
    genes = {x["gene_name"] for x in oracle}
    assert grain(r.header, "gene") == {"returned": len({x.get("gene_name") for x in rows}), "total": len(genes)}
    assert r.obj.get("num_total_significant") == len(genes), show(r)


def _ct5_tahoe_float32(r: CallResult, root: Path) -> None:
    ok(r)
    assert len(_tahoe_rows(r)) == len(F.oracle_tahoe(root, F.BORTEZOMIB, F.A549, 0.05)) > 0, show(r)
    assert dig(r.header, "scope", "concentration") == 0.05, r.header


def _ct5_tahoe_two_plates(r: CallResult, root: Path) -> None:
    ok(r)
    rows = _tahoe_rows(r)
    expected = {x["plate"] for x in F.oracle_tahoe(root, F.BORTEZOMIB, F.A549, F.TWO_PLATE_DOSE)}
    assert expected == {"1", "2"}
    assert all("plate" in row for row in rows) and {row.get("plate") for row in rows} == expected, rows


def _ct5_direct_indirect(r: CallResult, root: Path) -> None:
    """Phase 2: a derived set comparison on the complete key sets (direct is a subset of indirect)."""
    ok(r)
    assert r.header.get("served_by") == "derived", r.header
    assert dig(r.obj, "counts", "unique_to_direct_count") == 0, show(r)
    assert dig(r.obj, "direct_only") == [], show(r)


CT5: dict[str, Check] = {
    "datasource_unknown": _ct5_datasource_unknown,
    "datasource_indirect": _ct5_datasource_indirect,
    "coloc_typo": _ct5_invalid(["coloc", "ecaviar"]),
    "sort_unknown": _ct5_invalid(),
    "literal_name": _ct5_literal,
    "pgx_drug_server": _ct5_pgx,
    "pgx_target_server": _ct5_pgx,
    "limit_zero": _ct5_invalid(),
    "limit_negative": _ct5_invalid(),
    "tahoe_pooled": _ct5_tahoe_pooled,
    "tahoe_dose": _ct5_tahoe_dose,
    "tahoe_float32": _ct5_tahoe_float32,
    "tahoe_two_plates": _ct5_tahoe_two_plates,
    "direct_indirect": _ct5_direct_indirect,
}


@pytest.mark.parametrize("case,mode", cases(CT5))
def test_ct5_arguments_honoured_keys_complete(case: str, mode: str, live, fixture_ready, ot_root, tahoe_root) -> None:
    root = tahoe_root if case.startswith("tahoe_") else ot_root
    CT5[case](run_case(live, fixture_ready, mode, case), root)


def _pin_tahoe_pooled(r: CallResult, tahoe_root: Path) -> None:
    """Doses pooled: SELECTIVE1 repeated, no concentration field, rows counted as genes."""
    rows = _tahoe_rows(r)
    assert not r.is_error and sum(x.get("gene_name") == F.SELECTIVE for x in rows) > 1, rows
    assert all("concentration" not in x for x in rows), rows
    n_rows = len(F.oracle_tahoe(tahoe_root, F.BORTEZOMIB, F.A549))
    assert r.obj.get("num_total_significant") == n_rows > len({x.get("gene_name") for x in rows}), show(r)


def _today(pin: Callable[[CallResult], None]) -> Callable[[CallResult, Path], None]:
    return lambda r, root: pin(r)


CT5_PINS: dict[str, Callable[[CallResult, Path], None]] = {
    "datasource_unknown": _today(lambda r: _pin_message(r, "No associations found for this data source")),
    "coloc_typo": _today(lambda r: _pin_coloc(r)),
    "sort_unknown": _today(lambda r: _pin_sort(r)),
    "literal_name": _today(_pin_zero_rows("results")),
    "pgx_drug_server": _today(_pin_pgx_violation),
    "pgx_target_server": _today(_pin_pgx_violation),
    "limit_zero": _today(lambda r: _pin_message(r, "No known drugs found matching criteria")),
    "limit_negative": _today(lambda r: _pin_count(r, 34)),
    "tahoe_pooled": _pin_tahoe_pooled,
    "tahoe_dose": _today(lambda r: _pin_unexpected_keyword(r)),
    "direct_indirect": _today(lambda r: _pin_direct_indirect(r)),
}


def _pin_coloc(r: CallResult) -> None:
    ids = [x.get("leftStudyLocusId") for x in r.rows("colocalisations")]
    assert not r.is_error and r.obj.get("method") == "colc" and ids and all(i.startswith("ecaviar") for i in ids), ids


def _pin_sort(r: CallResult) -> None:
    ids = [x.get("targetId") for x in r.rows("targets")]
    assert not r.is_error and r.obj.get("filters_applied") == {} and ids == list(F.PRIO.values()), show(r)


def _pin_count(r: CallResult, n: int) -> None:
    assert not r.is_error and r.obj.get("count") == n, show(r)


def _pin_unexpected_keyword(r: CallResult) -> None:
    assert r.is_error and "unexpected_keyword_argument" in r.text.lower().replace(" ", "_"), show(r)


def _pin_direct_indirect(r: CallResult) -> None:
    assert not r.is_error and r.obj.get("unique_to_direct_count") == 7, show(r)


@pytest.mark.parametrize("case", sorted(CT5_PINS))
def test_ct5_today(case: str, live, tahoe_root, ot_root) -> None:
    assert F.oracle_direct_indirect(ot_root)["unique_to_direct"] == 0     # direct ⊆ indirect in the fixture
    CT5_PINS[case](live("off").call(*CALLS[case]), tahoe_root)


# ============================================================================ CT-6


def _ct6_no_safety(r: CallResult, root: Path) -> None:
    ok(r)
    oracle = F.oracle_no_safety_events(root)
    assert [t.get("targetId") for t in r.rows("targets")] == oracle["ids"] == [F.PRIO["C"]], show(r)
    assert excluded_unknown(r.header, "hasSafetyEvent") == oracle["unknown"] == 2, r.header


def _ct6_years(min_year: int | None, max_year: int | None) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        oracle = F.oracle_evidence_years(root, T, min_year=min_year, max_year=max_year)
        assert [e.get("id") for e in r.rows("evidence")] == oracle["ids"], show(r)
        assert excluded_unknown(r.header, "publicationYear") == oracle["unknown"] == 1, r.header
    return check


def _ct6_year_2030(r: CallResult, root: Path) -> None:
    empty(r, coverage="partial_unknown")
    assert r.rows("evidence") == [], show(r)


def _ct6_phenotype(evidence_type: str | None) -> Check:
    def check(r: CallResult, root: Path) -> None:
        ok(r)
        oracle = F.oracle_phenotype(root, F.SEIZURE, evidence_type)
        got = {d.get("disease_id"): d.get("evidence_count") for d in r.rows("diseases")}
        assert got == oracle["diseases"], show(r)
        for d in r.rows("diseases"):
            assert all(e.get("qualifierNot") is not True for e in d.get("evidence") or []), d
            if evidence_type:
                assert all(e.get("evidenceType") == evidence_type for e in d.get("evidence") or []), d
        negated = r.header.get("excluded_negated")
        assert negated and (negated == len(oracle["excluded_negated"]) or mentions(negated, F.PHENO["X"])), r.header
        n = len(oracle["diseases"])
        assert grain(r.header, "disease") == {"returned": n, "total": n}, r.header   # grains read the renamed rows
    return check


def _ct6_pair_negated(r: CallResult, root: Path) -> None:
    """A disease-phenotype pair whose every evidence item is negated has no support (X: PCS NOT; W: IEA NOT)."""
    header = empty(r)
    assert header.get("coverage") != "covered", header
    assert r.rows("phenotypes") == [], show(r)
    assert header.get("excluded_negated") == 1, header


def _ct6_pair_partly_negated(r: CallResult, root: Path) -> None:
    ok(r)
    (pair,) = r.rows("phenotypes")
    ev = pair.get("evidence") or []
    assert len(ev) == 2 and all(e.get("qualifierNot") is not True for e in ev), show(r)


def _ct6_phenotype_curie(r: CallResult, root: Path) -> None:
    ok(r)
    got = {d.get("disease_id"): d.get("evidence_count") for d in r.rows("diseases")}
    assert got == F.oracle_phenotype(root, F.SEIZURE)["diseases"], show(r)
    assert r.obj.get("phenotype_id") == F.SEIZURE or F.SEIZURE in resolution_text(r), show(r)


def _ct6_study(r: CallResult, root: Path) -> None:
    ok(r)
    oracle = F.oracle_studies(root, 100000)
    assert [s.get("studyId") for s in r.rows("studies")] == oracle["ids"] == [F.STUDY_BIG], show(r)
    assert excluded_unknown(r.header, "nSamples") == oracle["unknown"] == 25, r.header


CT6: dict[str, Check] = {
    "no_safety": _ct6_no_safety,
    "year_min_2024": _ct6_years(2024, None),
    "year_max_2020": _ct6_years(None, 2020),
    "year_min_2030": _ct6_year_2030,
    "phenotype_pcs": _ct6_phenotype("PCS"),
    "phenotype_all": _ct6_phenotype(None),
    "phenotype_curie": _ct6_phenotype_curie,
    "phenotype_unknown": lambda r, root: _not_found(r, root),
    "pheno_x": _ct6_pair_negated,
    "pheno_w": _ct6_pair_negated,
    "pheno_z": _ct6_pair_partly_negated,
    "study_min_size": _ct6_study,
}


@pytest.mark.parametrize("case,mode", cases(CT6))
def test_ct6_unknown_or_negated_never_favourable(case: str, mode: str, live, fixture_ready, ot_root) -> None:
    CT6[case](run_case(live, fixture_ready, mode, case), ot_root)


@pytest.fixture(scope="session")
def ot_safety_variants(tmp_path_factory) -> dict[str, Path]:
    return {v: F.build_ot_fixture(tmp_path_factory.mktemp(f"ot-safety-{v}") / "25.09", safety_variant=v)
            for v in ("no_zero", "refuted")}


@pytest.mark.parametrize("variant", ["no_zero", "refuted"])
@pytest.mark.parametrize("mode", mode_params(xfail_off=True))
def test_ct6_unconfirmable_encoding(mode: str, variant: str, ot_safety_variants, live_variant, fixture_ready) -> None:
    root = ot_safety_variants[variant]
    observed = F.oracle_no_safety_events(root)["observed"]
    assert observed != [-1.0, 0.0]                      # the declared codes are not both observed
    bridge = live_variant(mode, f"safety-{variant}", ("target",), DataEnv(ot_root=root))
    r = bridge.call(*CALLS["no_safety"])
    payload = err(r, "unsupported_filter")
    assert payload.get("column") == "hasSafetyEvent" or mentions(payload, "hasSafetyEvent"), payload
    scores = bridge.call("target", "get_target_prioritisation_scores", {"target_id": F.PCSK9})
    ok(scores)


@pytest.mark.parametrize("case,mode", cases(["absence_claim_2030"]))
async def test_ct6_partial_unknown_empty_is_not_absence(case, mode, tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct6", mode, tmp_path_factory, data_env, [CALLS["year_min_2030"]],
                                [[claim(0, "absence")]])
    assert out.end(0).get("is_error") is False, out.end(0)
    rejected(out.claim(0), "coverage partial_unknown")


@pytest.mark.parametrize("case,mode", cases(["negated_pair_presence"]))
async def test_ct6_negated_pair_is_not_presence(case, mode, tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct6pair", mode, tmp_path_factory, data_env, [CALLS["pheno_x"]],
                                [[claim(0, "presence", "Disease X presents with seizure")]])
    assert out.claim(0).get("ok") is False, out.claim(0)
    assert stored_evidence(out, "C0-presence").get("evidence_status") != "verified"


async def test_ct6_today_claim(tmp_path_factory, data_env) -> None:
    out = await runtime_outcome("ct6", "off", tmp_path_factory, data_env, [CALLS["year_min_2030"]],
                                [[claim(0, "absence")]])
    assert out.claim(0).get("ok") is True, out.claim(0)


def _pin_prioritize(r: CallResult) -> None:
    ids = [t.get("targetId") for t in r.rows("targets")]
    assert not r.is_error and {F.PRIO["A"], F.PRIO["B"], F.PRIO["C"]} <= set(ids), ids


def _pin_year(ids: list[str]) -> Callable[[CallResult], None]:
    def pin(r: CallResult) -> None:
        assert not r.is_error and [e.get("id") for e in r.rows("evidence")] == ids, show(r)
    return pin


def _pin_phenotype_pcs(r: CallResult) -> None:
    got = {d.get("disease_id"): d.get("evidence_count") for d in r.rows("diseases")}
    p = F.PHENO
    assert not r.is_error and got == {p["X"]: 1, p["Y"]: 0, p["Z"]: 2, p["W"]: 0}, got


def _pin_study(r: CallResult) -> None:
    rows = r.rows("studies")
    assert not r.is_error and len(rows) == 20 and all(s.get("nSamples") is None for s in rows), show(r)
    assert F.STUDY_BIG not in [s.get("studyId") for s in rows]


CT6_PINS: dict[str, Callable[[CallResult], None]] = {
    "no_safety": _pin_prioritize,
    "year_min_2024": _pin_year(["epmc-2024", "epmc-null"]),
    "year_max_2020": _pin_year(["epmc-2019", "epmc-null"]),
    "year_min_2030": _pin_year(["epmc-null"]),
    "phenotype_pcs": _pin_phenotype_pcs,
    "phenotype_curie": lambda r: _pin_message(r, "No diseases found for phenotype HP:0001250"),
    "phenotype_unknown": lambda r: _pin_message(r, f"No diseases found for phenotype {F.UNKNOWN_HP}"),
    "study_min_size": _pin_study,
}


@pytest.mark.parametrize("case", sorted(CT6_PINS))
def test_ct6_today(case: str, live) -> None:
    CT6_PINS[case](live("off").call(*CALLS[case]))

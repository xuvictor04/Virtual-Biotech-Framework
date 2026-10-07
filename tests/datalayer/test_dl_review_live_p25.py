"""Enforce-mode regressions of the phases 2-5 review findings, on the unmodified upstream servers.

Identifier and argument forms (CT1-V1, SW-6, SW-7, SW-8), service defaults and payloads of the derived
essentiality tools (SW-1, SW-2), per-section status of derived views (SW-4) and explicit nulls on
defaulted filters (GW-NULL). The calls run through the shared live bridges of ``conftest``.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

import dl_fixtures as F
from dl_upstream import CallResult, gateway_missing, needs_arrow, needs_fastmcp

pytestmark = [needs_arrow, needs_fastmcp, pytest.mark.correctness]

RS, VARIANT = "rs11591147", "1_55039974_G_T"


def enforce(live: Callable[[str], Any], fixture_ready: Callable[[Any], None]) -> Any:
    if gateway_missing():
        pytest.skip(gateway_missing())
    bridge = live("enforce")
    fixture_ready(bridge)
    return bridge


def ok(r: CallResult, status: str = "ok") -> CallResult:
    assert not r.is_error, r.text[:800]
    assert (r.status or r.header.get("status")) == status, r.header
    return r


def err(r: CallResult, kind: str) -> dict[str, Any]:
    assert r.is_error and r.kind == kind, r.text[:800]
    return r.payload


# ---------------------------------------------------------------------------- identifier and argument forms


def test_chembl_curie_form_resolves(live, fixture_ready) -> None:
    """CT1-V1: CHEMBL:25 is the CURIE form of CHEMBL25, never 'not found'."""
    r = ok(enforce(live, fixture_ready).call("drug", "get_drug_info", {"drug_id": "CHEMBL:25"}))
    assert "CHEMBL25" in str(r.header.get("resolved"))


@pytest.mark.parametrize("tool,args,rows", [
    ("get_colocalisation_by_chromosome", {"chromosome": "chr19", "method": "ecaviar"}, "colocalisations"),
    ("get_colocalisation_by_chromosome", {"chromosome": "chr19", "method": "coloc", "min_score": 0.0},
     "colocalisations"),
    ("query_gwas_associations", {"chromosome": "chr1", "output_path": "chr1.csv"}, "top_associations"),
])
def test_chr_prefixed_chromosomes_are_normalized(live, fixture_ready, tool: str, args: dict, rows: str) -> None:
    """SW-7 / CT1-V1: 'chr19' is the bare '19' (position role), never a typed not_found."""
    bridge = enforce(live, fixture_ready)
    got = ok(bridge.call("genetics", tool, args))
    bare = dict(args, chromosome=args["chromosome"][3:])
    if "output_path" in bare:
        bare["output_path"] = "bare-" + bare["output_path"]
    want = ok(bridge.call("genetics", tool, bare))
    assert got.header.get("total") == want.header.get("total") and got.header.get("total"), got.header
    assert any("normalized" in n for n in got.header.get("notes") or []), got.header
    payload = err(bridge.call("genetics", tool, dict(args, chromosome="chrM")), "invalid_argument")
    assert "MT" in payload.get("valid_values", [])


@pytest.mark.parametrize("region", ["chr1:55000000-55100000", "1:55000000-55100000"])
def test_gwas_region_queries_are_served(live, fixture_ready, region: str) -> None:
    """SW-8: a region (with or without chr) is a chromosome and a position range, not 'must be a number'."""
    r = ok(enforce(live, fixture_ready).call("genetics", "query_gwas_associations",
                                             {"region": region, "output_path": f"r{region[:4]}.csv"}))
    assert r.header.get("total") == 1, r.header


def test_reversed_gwas_region_is_invalid(live, fixture_ready) -> None:
    payload = err(enforce(live, fixture_ready).call("genetics", "query_gwas_associations",
                                                    {"region": "1:55100000-55000000", "output_path": "rev.csv"}),
                  "invalid_argument")
    assert payload.get("argument") == "region"


@pytest.mark.parametrize("tool,args", [
    ("get_variant_annotation", {"variant_id": RS}),
    ("query_gwas_associations", {"variant_id": RS, "output_path": "rs.csv"}),
])
def test_rsids_resolve_to_the_variant_they_name(live, fixture_ready, tool: str, args: dict) -> None:
    """SW-6: an rsID is translated to its variant ID (maps_to via variant), never sent on verbatim."""
    r = ok(enforce(live, fixture_ready).call("genetics", tool, args))
    assert VARIANT in str(r.header.get("resolved")), r.header


# ---------------------------------------------------------------------------- essentiality


def test_min_cell_lines_default_is_applied(live, fixture_ready) -> None:
    """SW-1: the disclosed default (min_cell_lines=3) is what the data child applies: TP53 in one Lung
    Cancer screen is not returned as an essential gene."""
    bridge = enforce(live, fixture_ready)
    r = bridge.call("functional_genomics", "find_essential_genes", {"disease": "Lung Cancer"})
    explicit = bridge.call("functional_genomics", "find_essential_genes", {"disease": "Lung Cancer", "min_cell_lines": 3})
    assert not r.is_error and not explicit.is_error, r.text[:800]
    assert r.header.get("total") == explicit.header.get("total"), (r.header, explicit.header)
    assert dict(r.header.get("scope") or {}).get("min_cell_lines") in (None, 3)
    assert all((g.get("num_cell_lines") or 0) >= 3 for g in r.rows("essential_genes")), r.text[:800]


def test_unscreened_gene_is_never_found(live, fixture_ready) -> None:
    """SW-2: a resolved gene without screens is empty, never a found:true record."""
    r = enforce(live, fixture_ready).call("functional_genomics", "query_gene_essentiality",
                                          {"gene_id": F.TP53BP1})
    assert not r.is_error, r.text[:800]
    assert (r.status or r.header.get("status")) == "empty", r.header
    assert r.obj.get("found") is not True, r.text[:800]
    assert "cite rows by key" not in str(r.header.get("cite"))


# ---------------------------------------------------------------------------- derived sections, nulls


def test_view_sections_carry_their_own_status_and_coverage(live, fixture_ready) -> None:
    """SW-4: empty sections of the target profile report their own coverage and are not citable absences."""
    r = ok(enforce(live, fixture_ready).call("target", "get_comprehensive_target_profile",
                                             {"target_id": F.TP53BP1}))
    sections = r.header.get("sections") or {}
    assert sections["known_drugs"]["status"] == "empty" and sections["known_drugs"]["coverage"] != "covered"
    assert sections["known_drugs"]["table"] == "open_targets.known_drug"
    assert "known_drug" in (r.header.get("tables") or [])
    assert "known_drugs" in str(r.header.get("cite")) and "not evidence of absence" in str(r.header.get("cite"))
    full = ok(enforce(live, fixture_ready).call("target", "get_comprehensive_target_profile",
                                                {"target_id": F.PCSK9}))
    assert full.header["sections"]["known_drugs"]["status"] == "ok"
    assert full.header["sections"]["known_drugs"]["total"] == len(full.obj.get("known_drugs") or [])


def test_null_on_a_defaulted_filter_is_refused_before_the_call(live, fixture_ready) -> None:
    """GW-NULL: upstream's signature takes no None, so null is invalid_argument naming the loosest value,
    never a pydantic source_error; the listed description no longer advises null."""
    payload = err(enforce(live, fixture_ready).call("drug", "search_known_drugs",
                                                    {"target_id": F.PCSK9, "min_phase": None, "limit": 5}),
                  "invalid_argument")
    assert payload.get("argument") == "min_phase" and "pass 0" in payload.get("message", "")

"""Regressions for the third review on real data (Open Targets 25.09, the live APIs); each test is offline and names
its finding.

* LIVE3-01: under the evidence ceiling a count that selects on the record as it is today (a status, a later date,
  posted results) reports the records unchanged since the ceiling as its total (partial), or risk when it cannot.
* LIVE3-05: a remote source that declares no leakage (cBioPortal, the Census) read under the ceiling records
  ``leakage {risk: true, reason: source not dated}``.
* LIVE3-08: PubMed book records (PubmedBookArticle: GeneReviews) are parsed, never reported as missing PMIDs.
* LIVE3-10: a native lookup of one PMID compiles to ``term=<pmid>[uid]`` (never too_large); title and abstract
  searches are field-tagged ([ti], [tiab]).
* LIVE3-09: rows the PubMed server withheld under its own ceiling are ``withheld.leakage`` with a leakage record.
* OT-RV3-02: a disclosed default of an ``order_direction`` argument (``ascending=False``) orders the rows and the
  header, not the order_by value's own direction.
* OT-RV3-03: a text that casefolds to one gene's approved symbol but is, case for case, an alias of another gene is
  ambiguous (GluD2: GLUD2 by casefold, GRID2 by alias), never silently the casefold gene.
* OT-RV3-04: an ID the bound list column holds but the entity table lacks (withdrawn CHEMBL2107902 in
  drug_warning.chemblIds) is served under ``existence: bound``, never ``not_found``.
* OT-RV3-05: rows a list column stores under the requested ID *and* another family member are in the answer, not
  counted again as left behind under that member.
* OT-RV3-07: a recognised foreign form (a UniProt accession, an Ensembl transcript or protein ID, an ICD10 CURIE)
  that no name matches is ``invalid_argument`` (``unsupported_form``), not a definitive ``not_found``.
* LIVE3-02: a record whose echo accepts a redirect (a merged NCT ID answered by the surviving record) is kept: the
  echo decides, the bound argument is not re-applied to the row.
* LIVE3-03: a native lookup/find compares requested keys in their normal form and finds a key the returned record
  lists as an alias (``key.aliases``): never not_found for a record it just fetched.
* LIVE3-04: a Census gene upstream lists under ``genes_not_found`` is a not_found item (partial), all of them
  not_found; never 'resolved' and a successful empty answer.
* LIVE3-06: ``universe_via`` id_types (cBioPortal cancer types and studies) are decided against the source's own
  listing: an unknown code is not_found before the call, a known one carries no 'could not be decided' note.
* LIVE3-07: a misspelled nested ``where`` path is invalid_argument before any request.
* LIVE3-12: a native find's conditions search is the registry's condition search (as query.cond), and its
  ceiling note no longer attributes every difference from an upstream count to post-ceiling changes.
* LIVE3-11: cBioPortal study rows served upstream without importDate get their record versions from the study
  listing; an empty molecular_data find is dated by its profile's study, not the fetch time.
* RR-3: the offline suite's in-process live transport refuses any non-loopback URL, and a failed release request is
  not repeated for a minute (a hung source costs one timeout, not one per call).
* RR-6: ``VBT_DL_NETWORK`` means the same in every file: 1/true/yes/on/full enable, 0/false/no/empty do not.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from test_dl_gateway_flow import DRUG_OVERLAY, KEYS, OT, TABLES, call, hdr, index_rows, make_gateway, world
from test_dl_resolver import GENE_ARGS
from test_dl_resolver import OT as RESOLVER_OT
from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.plugins.registry import discover
from vbt.datalayer.resolve import Entry, IndexStore, Resolver, error_for

# --------------------------------------------------------------------------- OT-RV3-02


def _ranked_overlay() -> dict[str, Any]:
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["ranked"] = {
        "reads": {"open_targets.known_drug": {"access": "full_table"}},
        "args": {"sort_by": {"role": "order_by", "values": {"phase": {"column": "phase", "direction": "asc"}}},
                 "ascending": {"role": "order_direction", "values": {True: "asc", False: "desc"}},
                 "limit": {"role": "limit", "min": 1, "max": 50}},
        "result": {"rows": "$.rows", "order_from_arg": "sort_by", "order": [{"column": "drugId", "direction": "asc"}]},
        "serve": "derived", "derived": {"verb": "find", "table": "open_targets.known_drug", "envelope": {"ok": True}}}
    return ov


SCHEMA = {"type": "object", "properties": {"sort_by": {"type": "string", "default": None},
                                           "ascending": {"type": "boolean", "default": False},
                                           "limit": {"type": "integer", "default": 3}}}


async def _ranked(gw: Any, args: dict[str, Any]) -> Any:
    gw.rewrite_listing("drug", "ranked", "ranked drugs", SCHEMA)     # the upstream schema, as the listing gives it
    plan = await gw.prepare("drug", "ranked", args, None)
    return await gw.finish(plan, None)


async def test_a_defaulted_direction_orders_the_rows_and_the_header(tmp_path):
    """OT-RV3-02: prioritize_targets(sort_by='geneticConstraint') sorted ascending (the value's own direction) while
    upstream's default ascending=False sorts descending, and the header said `asc` next to `scope: {ascending:
    false}`."""
    gw = world(tmp_path, overlay=_ranked_overlay())
    res = await _ranked(gw, {"sort_by": "phase", "limit": 3})
    h = hdr(res)
    assert [r["phase"] for r in res.obj["rows"]] == [4, 4, 3]
    assert h["order"].startswith("phase desc") and h["scope"] == {"ascending": False}
    assert any("ascending defaults to False" in n for n in h["notes"])
    explicit = await _ranked(gw, {"sort_by": "phase", "ascending": True, "limit": 3})
    up = [r["phase"] for r in explicit.obj["rows"]]
    assert up == sorted(up) and hdr(explicit)["order"].startswith("phase asc")
    same = await _ranked(gw, {"sort_by": "phase", "ascending": False, "limit": 3})
    assert [r["drugId"] for r in same.obj["rows"]] == [r["drugId"] for r in res.obj["rows"]]


# --------------------------------------------------------------------------- OT-RV3-03

GLUD2, GRID2, CGA, CHGA, TP53 = "ENSG00000182890", "ENSG00000152208", "ENSG00000135346", "ENSG00000100604", \
    "ENSG00000141510"


def _alias_resolver(tmp_path: Any) -> Resolver:
    registry = discover(entry_points=False)
    g = registry.get("identifier", "ensembl_gene").label_key
    rows = []
    for gid, sym in ((GLUD2, "GLUD2"), (GRID2, "GRID2"), (CGA, "CGA"), (CHGA, "CHGA"), (TP53, "TP53")):
        rows += [Entry(g(gid), gid, "exact", sym), Entry(g(sym), gid, "label_exact:approvedSymbol", sym)]
    # 25.09: GRID2 lists 'GluD2' (HGNC, UniProt, NCBI), CHGA lists 'CgA' (UniProt); GLUD2 and CGA only themselves
    rows += [Entry(g("GluD2"), GRID2, "synonym:alias", "GluD2"), Entry(g("CgA"), CHGA, "synonym:alias", "CgA"),
             Entry(g("GLUD2"), GLUD2, "synonym:alias", "GLUD2")]
    store = IndexStore(tmp_path / "cache")
    store.write_sidecar("open_targets", "fp-ot", "ensembl_gene", rows)
    catalog = Catalog({"open_targets": SourceDescriptor.model_validate(RESOLVER_OT)})
    return Resolver(registry, catalog, store.provider({"open_targets": "fp-ot"}))


@pytest.mark.parametrize("text,casefold,alias", [("GluD2", GLUD2, GRID2), ("CgA", CGA, CHGA)])
def test_a_case_exact_alias_of_another_gene_makes_a_casefold_match_ambiguous(tmp_path, text, casefold, alias):
    """OT-RV3-03: label_casefold:approvedSymbol ran before synonym:alias, so 'GluD2' (an alias of GRID2) resolved to
    GLUD2 with status ok; both readings are offered now, each with the rule it matched."""
    r = _alias_resolver(tmp_path).resolve(text, **GENE_ARGS)
    assert r.status == "ambiguous" and r.canonical is None
    assert {(c.id, c.via) for c in r.candidates} == {(casefold, "label_casefold:approvedSymbol"),
                                                     (alias, "synonym:alias")}
    assert error_for(r, "target_id", tool="mcp__drug__search_known_drugs").kind == ErrorKind.ambiguous


@pytest.mark.parametrize("text,want,rule", [("Tp53", TP53, "label_casefold:approvedSymbol"),
                                            ("GLUD2", GLUD2, "label_exact:approvedSymbol"),
                                            ("glud2", GLUD2, "label_casefold:approvedSymbol")])
def test_a_casefold_match_no_other_gene_spells_that_way_still_resolves(tmp_path, text, want, rule):
    """Mouse-style case (Tp53) is what the casefold rule is for; an exact symbol and a spelling no gene uses as an
    alias stay resolved."""
    r = _alias_resolver(tmp_path).resolve(text, **GENE_ARGS)
    assert (r.status, r.canonical, r.rule) == ("resolved", want, rule)


# --------------------------------------------------------------------------- OT-RV3-04


def _warning_world(tmp_path: Any, existence: str | None) -> Any:
    desc = copy.deepcopy(OT)
    desc["tables"]["drug_warning"] = {
        "kind": "fact", "grain": "one warning", "key": {"columns": ["id"]},
        "columns": {"id": {"role": "identifier"},
                    "chemblIds": {"role": "identifier", "id_type": "chembl_molecule", "path": "[]",
                                  "cardinality": "many"},
                    "warningType": {"role": "category"}}}
    ov = copy.deepcopy(DRUG_OVERLAY)
    arg: dict[str, Any] = {"binds": "open_targets.drug_warning.chemblIds", "op": "contains",
                           "accepts": ["chembl_molecule"]}
    if existence:
        arg["existence"] = existence
    ov["tools"]["get_drug_warnings"] = {"reads": {"open_targets.drug_warning": {"access": "full_table"}},
                                        "args": {"drug_id": arg}, "result": {"rows": "$.warnings"}}
    tables = {**TABLES, "open_targets.drug_warning": [
        {"id": 3603, "chemblIds": ["CHEMBL2107902"], "warningType": "Withdrawn"},
        {"id": 3604, "chemblIds": ["CHEMBL2107902", "CHEMBL3"], "warningType": "Withdrawn"}]}
    return make_gateway(tmp_path, [desc], [ov], tables, index_rows=index_rows(),
                        keys={**KEYS, "open_targets.drug_warning": ["id"]})


async def test_an_id_only_the_bound_list_column_holds_is_served(tmp_path):
    """OT-RV3-04: get_drug_warnings(drug_id='CHEMBL2107902') answered not_found (no record with this identifier),
    while upstream returns its four withdrawal warnings: the ID is in drug_warning.chemblIds, not drug_molecule."""
    with pytest.raises(GatewayError) as e:
        await _warning_world(tmp_path / "universe", None).prepare("drug", "get_drug_warnings",
                                                                 {"drug_id": "CHEMBL2107902"}, None)
    assert e.value.kind == ErrorKind.not_found                       # the universe alone decides: the defect
    gw = _warning_world(tmp_path / "bound", "bound")
    plan = await gw.prepare("drug", "get_drug_warnings", {"drug_id": "CHEMBL2107902"}, None)
    assert plan.route == "upstream" and plan.args_sent["drug_id"] == "CHEMBL2107902"
    assert plan.existence["drug_id"] == "exists"
    with pytest.raises(GatewayError) as e:                            # in neither: still not found
        await gw.prepare("drug", "get_drug_warnings", {"drug_id": "CHEMBL2107903"}, None)
    assert e.value.kind == ErrorKind.not_found and "drug_warning.chemblIds" in e.value.message


def test_the_shipped_bindings_decide_these_ids_by_the_bound_column():
    from vbt.datalayer.catalog import load_catalog

    cat = load_catalog()
    assert cat.contract("drug", "get_drug_warnings").args["drug_id"].existence == "bound"
    for server in ("drug", "target"):
        assert cat.contract(server, "get_pharmacogenomics").args["target_id"].existence == "bound"


# --------------------------------------------------------------------------- OT-RV3-05


def _family_world(tmp_path: Any, rows: list[dict[str, Any]]) -> Any:
    from test_dl_gateway_flow import REGISTRY

    c = REGISTRY.get("identifier", "chembl_molecule").label_key
    idx = index_rows()
    idx["open_targets:chembl_molecule"] = [e for e in idx["open_targets:chembl_molecule"] if e.canonical != "CHEMBL2"] \
        + [Entry(c("CHEMBL50"), "CHEMBL50", "exact", family="CHEMBL50"),
           Entry(c("CHEMBL2"), "CHEMBL2", "exact", family="CHEMBL50")]
    desc = copy.deepcopy(OT)
    desc["id_types"]["chembl_molecule"]["canonicalize"] = {"parent": "drug_molecule.parentId"}
    desc["tables"]["drug_molecule"]["columns"]["parentId"] = {"role": "identifier", "id_type": "chembl_molecule"}
    desc["tables"]["drug_warning"] = {
        "kind": "fact", "grain": "one warning", "key": {"columns": ["id"]},
        "columns": {"id": {"role": "identifier"},
                    "chemblIds": {"role": "identifier", "id_type": "chembl_molecule", "path": "[]",
                                  "cardinality": "many"}}}
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["warnings"] = {
        "reads": {"open_targets.drug_warning": {"access": "full_table"}},
        "args": {"drug_id": {"binds": "open_targets.drug_warning.chemblIds", "op": "contains",
                             "accepts": ["chembl_molecule"]},
                 "limit": {"role": "limit", "min": 1, "max": 50}},
        "result": {"rows": "$.warnings", "order": [{"column": "id", "direction": "asc"}]},
        "serve": "derived", "derived": {"verb": "find", "table": "open_targets.drug_warning", "envelope": {"ok": True}}}
    tables = {**TABLES, "open_targets.drug_molecule": TABLES["open_targets.drug_molecule"] + [{"id": "CHEMBL50"}],
              "open_targets.drug_warning": rows}
    return make_gateway(tmp_path, [desc], [ov], tables, index_rows=idx,
                        keys={**KEYS, "open_targets.drug_warning": ["id"]})


async def test_rows_naming_the_parent_and_its_salt_are_not_left_behind(tmp_path):
    """OT-RV3-05: get_drug_mechanisms(drug_id='CHEMBL941') returned its 4 rows (each lists CHEMBL1642 and CHEMBL941)
    as partial with family_rows {CHEMBL1642: 4} and told the agent to call again for the rows it already had."""
    both = [{"id": 1, "chemblIds": ["CHEMBL2", "CHEMBL50"]}, {"id": 2, "chemblIds": ["CHEMBL50", "CHEMBL2"]}]
    plan, res = await call(_family_world(tmp_path / "both", both), "drug", "warnings", {"drug_id": "CHEMBL50"})
    h = hdr(res)
    assert [r["id"] for r in res.obj["warnings"]] == [1, 2]
    assert h["status"] == "ok" and h.get("family_rows") is None and "top 2 of 2" not in str(h["cite"])
    # a row stored under the salt alone is still left behind, and counted once
    mixed = both + [{"id": 3, "chemblIds": ["CHEMBL2"]}]
    plan, res = await call(_family_world(tmp_path / "mixed", mixed), "drug", "warnings", {"drug_id": "CHEMBL50"})
    assert hdr(res)["family_rows"] == {"CHEMBL2": 1} and hdr(res)["status"] == "partial"


# --------------------------------------------------------------------------- OT-RV3-07


@pytest.mark.parametrize("text,form", [("P04637", "uniprot_accession"),
                                       ("ENST00000269305", "an Ensembl transcript/protein/exon ID"),
                                       ("ENSP00000269305.4", "an Ensembl transcript/protein/exon ID")])
def test_a_recognised_foreign_gene_form_is_an_unsupported_form(tmp_path, text, form):
    """OT-RV3-07: get_target_info(P04637), (ENST00000269305) answered not_found, which says TP53 does not exist."""
    r = _alias_resolver(tmp_path).resolve(text, **GENE_ARGS)
    assert (r.status, r.subkind) == ("rejected", "unsupported_form") and form in r.notes[0]
    err = error_for(r, "target_id", tool="mcp__target__get_target_info")
    assert err.kind == ErrorKind.invalid_argument and err.envelope()["subkind"] == "unsupported_form"
    assert "a form this argument does not take" in err.message and "ensembl_gene" in err.message
    plain = _alias_resolver(tmp_path).resolve("XQZ12345", **GENE_ARGS)   # nothing recognised: still not found
    assert plain.status == "not_found"


def test_an_unmapped_disease_xref_namespace_is_an_unsupported_form():
    """ICD10:E11 is a disease.dbXRefs namespace the resolver does not map (in 25.09 it maps to two Orphanet terms,
    not type 2 diabetes): refused as a form, never as an absent disease."""
    from vbt.datalayer.plugins.registry import discover

    rej = discover(entry_points=False).get("identifier", "ot_disease").normalize("ICD10:E11")
    assert rej.form == "a cross-reference in the ICD10 namespace"


# --------------------------------------------------------------------------- LIVE3-02


async def test_a_redirected_record_is_kept_not_dropped_by_its_own_argument(tmp_path):
    """LIVE3-02: get_clinical_trial_details(NCT00062153), an alias the registry redirects to NCT00060528: the echo
    accepted the redirect, then T4 re-applied nctId == NCT00062153 to the row and answered empty with body {}."""
    from test_dl_gateway_flow import PCSK9
    from test_dl_gateway_flow import TP53 as TP53_ID

    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["get_target"]["result"]["echo"] = {"target_id": {"path": "$.id", "accept": ["canonical", "redirect"]}}
    gw = world(tmp_path, overlay=ov)
    plan, res = await call(gw, "drug", "get_target", {"target_id": PCSK9},
                           lambda a: {"id": TP53_ID, "approvedSymbol": "TP53"})
    h = hdr(res)
    assert res.obj["id"] == TP53_ID and h["status"] == "ok" and h["returned"] == 1
    assert "alias_redirect" in h["resolved"]["target_id"] and not h.get("excluded")
    assert res.provenance.result.row_keys == [[TP53_ID]]


# --------------------------------------------------------------------------- LIVE3-03, LIVE3-07


class _Registry:
    """ClinicalTrials.gov v2 as a transport, as the registry matches filter.ids: case-insensitively and on aliases."""

    def __init__(self, studies: list[dict[str, Any]]) -> None:
        self.studies = studies
        self.sent: list[dict[str, str]] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None) -> Any:
        import json

        sent = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
        self.sent.append(sent)
        date = {"date": "Fri, 09 Oct 2026 09:00:00 GMT"}
        if url.endswith("/version"):
            return 200, date, json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-10-09T09:00:05"}).encode()
        if sent.get("countTotal") == "true":
            return 200, date, json.dumps({"totalCount": len(self.studies), "studies": []}).encode()
        ids = {i.upper() for i in (sent.get("filter.ids") or "").split(",") if i}

        def named(s: dict[str, Any]) -> bool:
            m = s["protocolSection"]["identificationModule"]
            return not ids or bool({m["nctId"], *m.get("nctIdAliases", [])} & ids)
        return 200, date, json.dumps({"studies": [s for s in self.studies if named(s)]}).encode()


def _trial(nct: str, aliases: list[str] | None = None) -> dict[str, Any]:
    ident: dict[str, Any] = {"nctId": nct, "briefTitle": "A trial"}
    if aliases:
        ident["nctIdAliases"] = aliases
    return {"protocolSection": {"identificationModule": ident,
                                "statusModule": {"overallStatus": "TERMINATED",
                                                 "studyFirstPostDateStruct": {"date": "2003-05-07"},
                                                 "lastUpdatePostDateStruct": {"date": "2017-10-26"}}},
            "hasResults": False}


@pytest.fixture
def registry_api(monkeypatch: pytest.MonkeyPatch) -> _Registry:
    from vbt.datalayer.plugins.layouts import live_api as live

    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})
    api = _Registry([_trial("NCT00060528", ["NCT00062153"]), _trial("NCT00761280"), _trial("NCT04368728")])
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(api))
    return api


CT_TABLE, NCT_COL = "clinicaltrials_gov.studies", "protocolSection.identificationModule.nctId"


@pytest.mark.parametrize("key,want", [("nct00761280", "NCT00761280"), ("NCT00062153", "NCT00060528")])
def test_a_lookup_finds_a_lower_case_or_alias_key_it_just_fetched(tmp_path, registry_api, key, want):
    """LIVE3-03: lookup {nctId: NCT00062153} and {nctId: nct00761280} answered not_found for the records the registry
    had just returned."""
    from test_dl_real_live import _ctx, _run

    out = _run(_ctx(tmp_path), "lookup", {"table": CT_TABLE, "key": {"nctId": key}})
    h = out["_vbt"]
    assert [r["protocolSection"]["identificationModule"]["nctId"] for r in out["rows"]] == [want]
    assert h["status"] == "ok" and not h.get("not_found_items")
    assert (want == "NCT00060528") == any("is an alias of NCT00060528" in n for n in h.get("notes") or [])


def test_a_find_never_reports_a_returned_alias_as_missing(tmp_path, registry_api):
    """LIVE3-03: find nctId in [NCT00062153, NCT04368728] returned both records and listed NCT00062153 as not found."""
    from test_dl_real_live import _ctx, _run

    out = _run(_ctx(tmp_path), "find", {"table": CT_TABLE, "where": {NCT_COL: ["NCT00062153", "NCT04368728",
                                                                             "NCT09999999"]}})
    h = out["_vbt"]
    assert len(out["rows"]) == 2 and h["not_found_items"] == ["NCT09999999"]


def test_a_misspelled_nested_where_column_is_refused_before_any_request(tmp_path, registry_api):
    """LIVE3-07: where {protocolSection.statusModule.overalStatus: ...} read 10 pages and answered empty_unverified."""
    from test_dl_real_live import _ctx, _run

    out = _run(_ctx(tmp_path), "find", {"table": CT_TABLE, "limit": 2,
                                        "where": {"protocolSection.statusModule.overalStatus": "RECRUITING"}})
    assert out["kind"] == "invalid_argument" and "overalStatus" in out["message"]     # the verb's refusal envelope
    assert registry_api.sent == []
    ok = _run(_ctx(tmp_path), "find", {"table": CT_TABLE, "limit": 2,
                                       "where": {"protocolSection.statusModule.overallStatus": "TERMINATED"}})
    assert ok["_vbt"]["status"] in ("ok", "partial")


# --------------------------------------------------------------------------- LIVE3-04

KNOWN_GENE, UNKNOWN_GENE = "ENSG00000103855", "ENSG00000999999"


def _genes_server(tool: str, args: dict[str, Any]) -> Any:
    """Upstream search_genes / get_gene_statistics over a one-gene var table (single_cell_mcp/tools.py:238-282)."""
    found, missing = [], []
    for g in args.get("ensembl_ids") or []:
        (found if g == KNOWN_GENE else missing).append(g)
    rows = [{"feature_id": g, "feature_name": "CD276", "feature_length": 1, "n_measured_obs": 10, "nnz": 5}
            for g in found]
    if tool == "search_genes":
        return {"genes_found": rows, "genes_not_found": missing, "total_found": len(rows),
                "total_not_found": len(missing)}
    return {"gene_statistics": rows, "genes_not_found": missing, "total_found": len(rows)}


@pytest.mark.parametrize("tool", ["search_genes", "get_gene_statistics"])
async def test_a_gene_upstream_lists_as_not_found_is_not_resolved(tmp_path, monkeypatch, tool):
    """LIVE3-04: search_genes([ENSG00000103855, ENSG00000999999]) answered ok with resolution_summary {resolved 2}
    although the body listed ENSG00000999999 under genes_not_found; alone it answered empty, not not_found."""
    from test_dl_live_review_fixes import _call, _ctx_with, _gateway, census

    census(monkeypatch)
    ctx = _ctx_with(tmp_path, ["single_cell"])
    gw = _gateway(ctx, tmp_path, {"single_cell": _genes_server})
    res = await _call(gw, "single_cell", tool, {"ensembl_ids": [KNOWN_GENE, UNKNOWN_GENE]})
    h = res.header
    assert h["status"] == "partial" and h["not_found_items"] == [UNKNOWN_GENE]
    assert h["resolution_summary"]["ensembl_ids"]["resolved"] == 1
    assert h["resolution_summary"]["ensembl_ids"]["unresolved"] == [UNKNOWN_GENE]
    with pytest.raises(GatewayError) as e:
        await _call(gw, "single_cell", tool, {"ensembl_ids": [UNKNOWN_GENE]})
    assert e.value.kind == ErrorKind.not_found and UNKNOWN_GENE in e.value.message
    ok = await _call(gw, "single_cell", tool, {"ensembl_ids": [KNOWN_GENE]})
    assert ok.header["status"] == "ok" and not ok.header.get("not_found_items")


# --------------------------------------------------------------------------- LIVE3-01, LIVE3-05, LIVE3-09


class _Counts:
    """CT.gov count requests: 8 records first posted by the ceiling, 3 of them also unchanged since (the rest were
    terminated, or posted results, after it); every request is recorded."""

    def __init__(self, unchanged: int | None = 3) -> None:
        self.unchanged = unchanged
        self.sent: list[dict[str, str]] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None) -> Any:
        import json

        sent = {str(k): str(v) for k, v in (params or {}).items() if v is not None}
        self.sent.append(sent)
        date = {"date": "Fri, 09 Oct 2026 09:00:00 GMT"}
        if url.endswith("/version"):
            return 200, date, json.dumps({"apiVersion": "2.0.5", "dataTimestamp": "2026-10-09T09:00:05"}).encode()
        if "LastUpdatePostDate" in sent.get("filter.advanced", ""):
            if self.unchanged is None:
                return 500, date, b"{}"
            return 200, date, json.dumps({"totalCount": self.unchanged, "studies": []}).encode()
        return 200, date, json.dumps({"totalCount": 8, "studies": []}).encode()


def _ct_gateway(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, api: Any) -> Any:
    from test_dl_round3_live import OVERLAYS, _fresh_caches, gateway
    from vbt.datalayer.descriptor.load import load_yaml
    from vbt.datalayer.plugins.layouts import live_api as live

    _fresh_caches(monkeypatch)
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(api))
    return gateway(tmp_path, [load_yaml(OVERLAYS / "clinicaltrials.yaml")],
                   data={"leakage": {"ceiling": "2017-12-31"}})


@pytest.mark.parametrize("args", [{"condition": "glioblastoma", "status": ["TERMINATED"], "phase": ["PHASE3"]},
                                  {"condition": "glioblastoma",
                                   "advanced_filter": "AREA[ResultsFirstPostDate]RANGE[2018-01-01,MAX]"}])
async def test_a_count_selecting_on_the_current_record_is_bounded_by_the_unchanged_records(tmp_path, monkeypatch,
                                                                                           args):
    """LIVE3-01: count_clinical_trials(status=[TERMINATED]) under a 2017-12-31 ceiling answered ok, total 8, risk
    false: 3 of the 8 were terminated after the ceiling; a count of results posted after it answered 194."""
    from test_dl_round3_live import call

    gw = _ct_gateway(tmp_path, monkeypatch, _Counts())
    res = await call(gw, "clinicaltrials", "count_clinical_trials", {**args, "country": None},
                     lambda plan, a: {"total_count": plan.witness["total"], "query_params": a})
    h = res.header
    assert h["status"] == "partial" and h["total"] == 3 and h["total_method"] == "ceiling_unchanged"
    assert res.obj["total_count"] == 3                                  # the body no longer says 8
    assert "the call selects on" in h["notes"][0]
    assert h["ceiling_totals"]["available"] == 8


async def test_without_the_unchanged_count_the_answer_carries_the_risk(tmp_path, monkeypatch):
    from test_dl_round3_live import call

    gw = _ct_gateway(tmp_path, monkeypatch, _Counts(unchanged=None))
    res = await call(gw, "clinicaltrials", "count_clinical_trials",
                     {"condition": "glioblastoma", "status": ["TERMINATED"], "country": None},
                     lambda plan, a: {"total_count": plan.witness["total"], "query_params": a})
    assert res.header["status"] == "partial" and res.header["total"] == 8
    assert res.provenance.leakage == {"ceiling": "2017-12-31", "withheld": 0, "risk": True,
                                      "reason": "selects on the current record"}


async def test_a_count_on_the_first_posting_alone_stays_as_it_was(tmp_path, monkeypatch):
    from test_dl_round3_live import call

    gw = _ct_gateway(tmp_path, monkeypatch, _Counts())
    res = await call(gw, "clinicaltrials", "count_clinical_trials",
                     {"condition": "glioblastoma", "phase": ["PHASE3"], "country": None},
                     lambda plan, a: {"total_count": plan.witness["total"], "query_params": a})
    assert res.header["status"] == "ok" and res.header["total"] == 8 and res.obj["total_count"] == 8
    assert res.provenance.leakage == {"ceiling": "2017-12-31", "withheld": 0, "risk": False}


def test_undated_remote_sources_carry_the_risk_under_the_ceiling():
    """LIVE3-05: get_study_details, search_studies (cBioPortal) and count_cells (the Census) under a 2017-12-31 ceiling
    recorded leakage null, exactly like a run without a ceiling."""
    from datetime import date

    from vbt.datalayer.catalog import load_catalog
    from vbt.datalayer.gateway.leakage import leakage_record, prepare_leakage

    cat = load_catalog()
    for server, tool in (("clinicaltrials", "get_study_details"), ("clinicaltrials", "search_studies"),
                         ("single_cell", "count_cells")):
        plan = prepare_leakage(cat.contract(server, tool), {}, date(2017, 12, 31))
        assert plan.recorded and plan.risk and plan.reason == "source not dated", (server, tool)
        assert any("carry no date" in n for n in plan.notes)
        assert leakage_record(plan.ceiling, 0, plan.risk, plan.reason)["reason"] == "source not dated"
        assert not prepare_leakage(cat.contract(server, tool), {}, None).recorded     # no ceiling: no record
    ct = prepare_leakage(cat.contract("clinicaltrials", "get_clinical_trial_details"), {}, date(2017, 12, 31))
    assert ct.recorded and not ct.risk and not ct.undated


async def test_rows_the_pubmed_server_withheld_are_counted_and_recorded(tmp_path, monkeypatch):
    """LIVE3-09: fetch_abstracts(['30403574']) under VBT_LITERATURE_MAXDATE=2017/12/31 answered with no header
    `withheld` and leakage null; the only trace was the body's withheld list."""
    from test_dl_live_review_fixes import _call, _ctx_with, _gateway

    monkeypatch.setenv("VBT_LITERATURE_MAXDATE", "2017/12/31")
    ctx = _ctx_with(tmp_path, ["pubmed"])
    late = {"pmid": "30403574", "reason": "published 2018-11 or later, after the 2017/12/31 literature ceiling"}

    def pubmed(tool: str, args: dict[str, Any]) -> Any:
        arts = [{"pmid": p, "title": "t", "pubdate": "2017 May"} for p in args["pmids"] if p != "30403574"]
        return {"articles": arts, "withheld": [late] if "30403574" in args["pmids"] else [],
                "literature_max_date": "2017/12/31"}

    gw = _gateway(ctx, tmp_path, {"pubmed": pubmed})
    res = await _call(gw, "pubmed", "fetch_abstracts", {"pmids": ["30403574", "28304224"]})
    assert res.header["withheld"] == {"leakage": 1} and res.header["returned"] == 1
    assert res.provenance.leakage == {"ceiling": "2017-12-31", "withheld": 1, "risk": False}
    only = await _call(gw, "pubmed", "fetch_abstracts", {"pmids": ["30403574"]})
    assert only.header["withheld"] == {"leakage": 1} and only.provenance.leakage["withheld"] == 1


# --------------------------------------------------------------------------- LIVE3-06


ACC, PROFILE = "acc_tcga_pan_can_atlas_2018", "acc_tcga_pan_can_atlas_2018_rna_seq_v2_mrna"
STUDIES = {"difg_glass_2019": "2026-01-07 03:51:35", "difg_glass": "2025-12-01 10:00:00",
           "difg_msk_2023": "2026-02-02 02:02:02", "difg_x1": "2025-01-01 00:00:00", "difg_x2": "2025-01-02 00:00:00",
           "difg_x3": "2025-01-03 00:00:00", ACC: "2026-06-05 15:19:54"}


class _Portal:
    """cBioPortal REST as a transport: /cancer-types and /studies listings, /info; every request is recorded."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def __call__(self, url: str, params: Any = None, *, timeout: float = 30.0, headers: Any = None) -> Any:
        import json

        self.sent.append(url)
        hdrs = {"date": "Fri, 09 Oct 2026 09:00:00 GMT"}
        if url.endswith("/info"):
            return 200, hdrs, json.dumps({"portalVersion": "v7.1.2", "dbVersion": "3.0.0"}).encode()
        if url.endswith("/cancer-types"):
            return 200, hdrs, json.dumps([{"cancerTypeId": c, "name": c.upper()} for c in ("difg", "luad", "gbm")]).encode()
        if url.endswith("/studies"):
            return 200, hdrs, json.dumps([{"studyId": s, "cancerTypeId": "difg", "importDate": d}
                                          for s, d in STUDIES.items()]).encode()
        if url.endswith("/molecular-profiles/" + PROFILE):
            return 200, hdrs, json.dumps({"molecularProfileId": PROFILE, "studyId": ACC}).encode()
        if url.endswith("/studies/" + ACC):
            return 200, hdrs, json.dumps({"studyId": ACC, "importDate": STUDIES[ACC]}).encode()
        if url.endswith(f"/molecular-profiles/{PROFILE}/molecular-data"):
            return 200, hdrs, b"[]"
        return 404, hdrs, b"{}"


def test_universe_via_ids_are_decided_against_the_sources_listing(tmp_path, monkeypatch):
    """LIVE3-06: _resolve_remote raised 'the id_type declares no universe' for every universe_via id_type, so
    search_studies(cancer_type='gbmx') answered empty_unverified and every known code carried 'could not be
    decided; an empty result is not citable'."""
    from test_dl_real_live import _ctx, _run
    from vbt.datalayer.plugins.layouts import live_api as live
    from vbt.datalayer.service.verbs import resolve_remote

    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})
    monkeypatch.setattr(resolve_remote, "_LISTED", {})
    portal = _Portal()
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(portal))
    ctx = _ctx(tmp_path)
    out = _run(ctx, "_resolve_remote", {"source": "cbioportal", "id_type": "cbio_cancer_type",
                                        "values": ["difg", "gbmx"]})
    by = {r["value"]: r["existence"] for r in out["resolutions"]}
    assert by == {"difg": "exists", "gbmx": "absent"} and out["existence"] == "unknown"
    gone = _run(ctx, "_resolve_remote", {"source": "cbioportal", "id_type": "cbio_cancer_type", "values": ["gbmx"]})
    assert gone["existence"] == "absent"
    assert sum(u.endswith("/cancer-types") for u in portal.sent) == 1          # the listing is kept for ttl_s
    study = _run(ctx, "_resolve_remote", {"source": "cbioportal", "id_type": "cbio_study",
                                          "values": ["difg_glass_2019", "zz_not_a_study_2099"]})
    assert {r["value"]: r["existence"] for r in study["resolutions"]} == {"difg_glass_2019": "exists",
                                                                          "zz_not_a_study_2099": "absent"}


# --------------------------------------------------------------------------- LIVE3-08

BOOK_XML = """<PubmedArticleSet>
  <PubmedArticle><MedlineCitation><PMID>28304224</PMID><Article><Journal><JournalIssue>
    <PubDate><Year>2017</Year><Month>May</Month></PubDate></JournalIssue><Title>N Engl J Med</Title></Journal>
    <ArticleTitle>Evolocumab and Clinical Outcomes</ArticleTitle></Article></MedlineCitation></PubmedArticle>
  <PubmedBookArticle><BookDocument><PMID Version="1">20301295</PMID>
    <ArticleIdList><ArticleId IdType="bookaccession">NBK1116</ArticleId></ArticleIdList>
    <Book><BookTitle book="gene">GeneReviews<sup>R</sup></BookTitle><PubDate><Year>1993</Year></PubDate>
      <BeginningDate><Year>1993</Year></BeginningDate><EndingDate><Year>2026</Year></EndingDate></Book>
    <PublicationType UI="D016454">Review</PublicationType>
    <Abstract><AbstractText>GeneReviews is an international point-of-care resource.</AbstractText></Abstract>
  </BookDocument></PubmedBookArticle>
</PubmedArticleSet>"""


def test_a_pubmed_book_record_is_a_record_not_a_missing_pmid():
    """LIVE3-08: efetch returns PMID 20301295 (GeneReviews, NBK1116) as a PubmedBookArticle; parse_articles read
    PubmedArticle only, so fetch_abstracts reported the PMID as 'returned no record'."""
    from vbt.mcp_servers import pubmed_server as pm

    arts, withheld = pm.parse_articles(BOOK_XML, None)
    by = {a["pmid"]: a for a in arts}
    assert set(by) == {"28304224", "20301295"} and withheld == []
    book = by["20301295"]
    assert book["record_type"] == "book" and book["title"].startswith("GeneReviews") and book["year"] == "1993"
    assert "point-of-care" in book["abstract"]
    assert pm.reconcile(["20301295", "28304224"], list(by), [])["not_returned"] == []
    # a living book is read as it is today: under a 2017 ceiling its 2026 revision withholds it
    arts, withheld = pm.parse_articles(BOOK_XML, (2017, 12, 31))
    assert [a["pmid"] for a in arts] == ["28304224"] and [w["pmid"] for w in withheld] == ["20301295"]


# --------------------------------------------------------------------------- LIVE3-10


def test_a_native_pubmed_lookup_of_one_pmid_is_answered(tmp_path, monkeypatch):
    """LIVE3-10: lookup {pmid: '28304224'} and {pmid: '99999999'} both failed as too_large ('narrow where'): pmid
    equality did not compile into esearch, and title/abstract searches were refused."""
    import json

    from test_dl_real_live import _ctx, _run
    from vbt.datalayer.plugins.layouts import live_api as live

    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    sent: list[dict[str, str]] = []

    def eutils(url: str, params: Any = None, **kw: Any) -> Any:
        sent.append(dict(params or {}))
        term = (params or {}).get("term", "")
        ids = [x[: -len("[uid]")] for x in term.split(" OR ") if x.endswith("[uid]") and not x.startswith("9999")]
        if not term.endswith("[uid]"):
            ids = ["28304224"]                                     # a title search's one hit
        return 200, {}, json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids}}).encode()

    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(eutils))
    ctx = _ctx(tmp_path)
    out = _run(ctx, "lookup", {"table": "pubmed.records", "key": {"pmid": "28304224"}})
    assert out["_vbt"]["status"] == "ok" and out["rows"] == [{"pmid": "28304224"}]
    assert sent[-1]["term"] == "28304224[uid]"
    gone = _run(ctx, "lookup", {"table": "pubmed.records", "key": {"pmid": "99999999"}})
    assert gone["kind"] == "not_found"
    both = _run(ctx, "find", {"table": "pubmed.records", "where": {"pmid": ["28304224", "29224444"]}})
    assert sent[-1]["term"] == "28304224[uid] OR 29224444[uid]" and both["_vbt"]["returned"] == 2
    hit = _run(ctx, "find", {"table": "pubmed.records", "where": {"title": {"search": "PCSK9 evolocumab"}}})
    assert sent[-1]["term"] == "PCSK9[ti] evolocumab[ti]" and hit["_vbt"]["status"] == "ok"


@pytest.mark.parametrize("text,tagged", [
    ("PCSK9 AND evolocumab", "PCSK9[ti] AND evolocumab[ti]"),
    ('"heart failure" OR (HF AND NOT acute)', '"heart failure"[ti] OR (HF[ti] AND NOT acute[ti])'),
    ("B7-H3 lung[ad]", "B7-H3[ti] lung[ad]"),
])
def test_each_search_term_carries_the_field_tag(text, tagged):
    """E-utilities ignores a tag after a parenthesised group: "(PCSK9 AND evolocumab)[ti]" counted 304 records to
    2017 in every field, "PCSK9[ti] AND evolocumab[ti]" 32 (verified live 2026-10-09)."""
    from vbt.datalayer.plugins.formats.rest_json import tag_terms

    assert tag_terms(text, "[ti]") == tagged


# --------------------------------------------------------------------------- LIVE3-11


def _portal(monkeypatch: pytest.MonkeyPatch) -> _Portal:
    from vbt.datalayer.plugins.layouts import live_api as live
    from vbt.datalayer.service.verbs import resolve_remote, witness

    monkeypatch.setattr(live, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live, "_RELEASES", {})
    monkeypatch.setattr(resolve_remote, "_LISTED", {})
    monkeypatch.setattr(witness, "_STUDY_RELEASES", {})
    monkeypatch.setattr(witness, "_LISTED_RELEASES", {})
    portal = _Portal()
    monkeypatch.setattr(live.LiveApiLayout, "transport", staticmethod(portal))
    return portal


async def test_study_rows_without_import_dates_get_their_record_versions(tmp_path, monkeypatch):
    """LIVE3-11: search_studies(difg) returned 20 studies with source 'cbioportal' and record_versions null: the rows
    carry no importDate and the predicate (cancerTypeId) names no study."""
    from test_dl_live_review_fixes import _call, _ctx_with, _gateway

    _portal(monkeypatch)
    ctx = _ctx_with(tmp_path, ["clinicaltrials"])
    listed = [s for s in STUDIES if s.startswith("difg")]

    def server(tool: str, args: dict[str, Any]) -> Any:
        assert tool == "search_studies"
        return {"count": len(listed), "studies": [{"studyId": s, "cancerTypeId": "difg", "name": s} for s in listed]}

    res = await _call(_gateway(ctx, tmp_path, {"clinicaltrials": server}), "clinicaltrials", "search_studies",
                      {"cancer_type": "difg"})
    versions = res.provenance.result.record_versions or {}
    assert versions.get("cbioportal.study") == {s: STUDIES[s] for s in listed}
    assert not any("could not be decided" in n for n in res.header.get("notes") or [])     # LIVE3-06 too


def test_an_empty_molecular_data_find_is_dated_by_its_profiles_study(tmp_path, monkeypatch):
    """LIVE3-11: find cbioportal.molecular_data with an unknown Entrez ID answered empty_unverified with source
    'cbioportal@<the HTTP fetch time>'; the profile names its study, whose importDate is the release."""
    from test_dl_real_live import _ctx, _run

    _portal(monkeypatch)
    out = _run(_ctx(tmp_path), "find", {"table": "cbioportal.molecular_data",
                                        "where": {"molecularProfileId": PROFILE, "sampleListId": f"{ACC}_all",
                                                  "entrezGeneId": 999999999}})
    h = out["_vbt"]
    assert out["rows"] == [] and h["source"] == f"cbioportal@{STUDIES[ACC]}"
    assert out["record_versions"] == {"cbioportal.study": {ACC: STUDIES[ACC]}}


# --------------------------------------------------------------------------- LIVE3-12


def test_a_conditions_search_is_the_registrys_condition_search(tmp_path, registry_api):
    """LIVE3-12: conditions {search: glioblastoma} compiled to AREA[Condition], which misses trials naming the
    condition only as a keyword (NCT01132547, NCT02443194) while upstream's query.cond counts them; the note blamed
    the whole gap on post-ceiling changes."""
    from test_dl_real_live import _ctx, _run

    out = _run(_ctx(tmp_path, ceiling="2017-12-31"), "find", {
        "table": CT_TABLE, "limit": 2,
        "where": {"protocolSection.conditionsModule.conditions": {"search": "glioblastoma"},
                  "protocolSection.statusModule.overallStatus": "TERMINATED"}})
    pages = [s for s in registry_api.sent if "filter.advanced" in s and s.get("countTotal") != "true"]
    assert pages and "AREA[ConditionSearch]glioblastoma" in pages[0]["filter.advanced"]
    assert "so its total is larger" not in " ".join(out["_vbt"]["notes"])


# --------------------------------------------------------------------------- RR-3


def test_the_offline_suite_refuses_live_requests_at_once(monkeypatch):
    from netgate import network_enabled
    from vbt.datalayer.plugins.layouts import live_api

    if network_enabled():
        pytest.skip("VBT_DL_NETWORK enables the live requests")
    with pytest.raises(live_api.RemoteError, match="offline suite"):
        live_api.LiveApiLayout.transport("https://www.cbioportal.org/api/info")


def test_a_failed_release_request_is_not_repeated_within_a_minute(monkeypatch):
    from vbt.datalayer.plugins.layouts import live_api

    sent: list[str] = []

    def down(url, params=None, **_kw):
        sent.append(url)
        raise live_api.RemoteError(f"request to {url} failed: timed out", url=url)

    monkeypatch.setattr(live_api, "_wait_turn", lambda base, rpm: None)
    monkeypatch.setattr(live_api, "_RELEASES", {})
    monkeypatch.setattr(live_api, "_RELEASE_FAILURES", {})
    monkeypatch.setattr(live_api.LiveApiLayout, "transport", staticmethod(down))
    layout = live_api.LiveApiLayout()
    spec = live_api.LayoutSpec(table="cbioportal.study", path="studies",
                               options={"base_url": "https://example.invalid/api",
                                        "release": {"endpoint": "info", "path": "$.dbVersion"}})
    for _ in range(3):
        with pytest.raises(live_api.RemoteError):
            layout.release_info(spec, None, max_age_s=live_api.RELEASE_TTL_S)
    assert len(sent) == 1                                   # one request, two answers from the failure cache
    # past the failure window it is asked again
    url = next(iter(live_api._RELEASE_FAILURES))
    stamp, why = live_api._RELEASE_FAILURES[url]
    live_api._RELEASE_FAILURES[url] = (stamp - live_api.RELEASE_FAILURE_TTL_S - 1, why)
    with pytest.raises(live_api.RemoteError):
        layout.release_info(spec, None, max_age_s=live_api.RELEASE_TTL_S)
    assert len(sent) == 2


# --------------------------------------------------------------------------- RR-6


@pytest.mark.parametrize("value,on", [("1", True), ("full", True), ("TRUE", True), (" yes ", True), ("on", True),
                                      ("0", False), ("false", False), ("no", False), ("", False), ("off", False)])
def test_the_network_gate_means_the_same_everywhere(monkeypatch, value, on):
    import netgate

    monkeypatch.setenv("VBT_DL_NETWORK", value)
    assert netgate.network_enabled() is on
    assert netgate.network_mode() == (value.strip().lower() if on else "")

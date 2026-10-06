"""The resolver (§11.5): rule grammar and per-id_type order, synonym-kind policy, retired IDs, xrefs,
crosswalk chains, stored forms, parent families, union kinds, ambiguity with disambiguation,
all-rejected -> invalid_argument, existence modes incl. unknown, list arguments, and sidecar
loading without pyarrow. Sidecars are written in tmp_path the way the data child writes them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vbt.datalayer.catalog import Catalog
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.errors import ErrorKind
from vbt.datalayer.plugins.registry import discover
from vbt.datalayer.resolve import (
    HEADER,
    Entry,
    IndexFormatError,
    IndexMissing,
    IndexStore,
    Resolver,
    ResolverConfigError,
    ResolverIndex,
    RuleError,
    default_rules,
    error_for,
    format_rule,
    list_error,
    parse_entry_rule,
    parse_rule,
    read_sidecar,
    rules_for,
    write_sidecar,
)
from vbt.datalayer.resolve.index import within_one_edit
from vbt.datalayer.settings import DataSettings, ResolutionSettings

SRC = Path(__file__).resolve().parents[2] / "src"

# --------------------------------------------------------------------------- descriptors


def _table(key: str, columns: dict[str, Any], kind: str = "entity") -> dict[str, Any]:
    return {"kind": kind, "grain": f"one {key}", "key": {"columns": [key]}, "columns": columns}


OT: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "open_targets", "title": "Open Targets",
    "release": {"expect": "25.09", "from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
    "id_types": {
        "ensembl_gene": {"plugin": "ensembl_gene", "universe": "target.id",
                         "resolve_via": ["target.approvedSymbol", "target.symbolSynonyms", "target.obsoleteSymbols"],
                         "rules": ["exact", "normalized", "label_exact:approvedSymbol", "label_casefold:approvedSymbol",
                                   "synonym:previous", "synonym:alias"],
                         "disambiguate_with": ["biotype"]},
        "hgnc_symbol": {"plugin": "hgnc_symbol", "label_of": "ensembl_gene"},
        "ot_disease": {"plugin": "ot_disease", "options": {"prefixes": "from_universe"}, "universe": "disease.id",
                       "resolve_via": ["disease.name", "disease.synonyms"],
                       "retired": {"listed_in": ["disease.obsoleteTerms"]},
                       "xref_via": [{"column": "disease.dbXRefs", "namespaces": {"DOID": "doid", "MESH": "mesh"}}],
                       "disambiguate_with": ["ontology.isTherapeuticArea"]},
        "disease_name": {"plugin": "disease_name", "label_of": "ot_disease"},
        "chembl_molecule": {"plugin": "chembl_molecule", "universe": "drug_molecule.id",
                            "resolve_via": ["drug_molecule.name", "drug_molecule.tradeNames"],
                            "canonicalize": {"parent": "drug_molecule.parentId"}},
        "drug_name": {"plugin": "drug_name", "label_of": "chembl_molecule"},
        "literature_word": {"plugin": "ot_entity_any", "union": ["ensembl_gene", "ot_disease", "chembl_molecule"],
                            "universe": "literature_vector.word"},
        "pmid": {"plugin": "pmid", "resolvable": False},
        "rsid": {"plugin": "rsid", "universe": "variant.rsIds[]", "index": "remote"},
        "ncbi_taxon": {"plugin": "ncbi_taxon", "resolvable": False},
    },
    "tables": {
        "target": _table("id", {"id": {"role": "identifier", "id_type": "ensembl_gene", "self": True},
                                "approvedSymbol": {"role": "label", "of": "id"},
                                "biotype": {"role": "category"}}),
        "disease": _table("id", {"id": {"role": "identifier", "id_type": "ot_disease", "self": True},
                                 "name": {"role": "label", "of": "id"}}, "ontology"),
        "drug_molecule": _table("id", {"id": {"role": "identifier", "id_type": "chembl_molecule", "self": True},
                                       "parentId": {"role": "identifier", "id_type": "chembl_molecule"}}),
        "literature_vector": _table("word", {"word": {"role": "identifier", "id_type": "literature_word",
                                                      "self": True}}, "vectors"),
        "variant": _table("variantId", {"variantId": {"role": "identifier"}}),
    },
}

TAHOE: dict[str, Any] = {
    "schema": "vbt.datasource/1", "source": "tahoe_100m", "title": "Tahoe-100M DE", "release": {"from": "literal"},
    "defaults": {"format": "parquet", "layout": "single_file"},
    "id_types": {
        "tahoe_drug": {"plugin": "tahoe_drug", "universe": "drug_metadata.drug", "resolve_via": ["drug_metadata.drug"],
                       "stored_forms": {"de_permissive.drug": "as_stored"}},
        "tahoe_gene": {"plugin": "tahoe_gene_name", "universe": "gene_metadata.gene_symbol",
                       "crosswalks": [{"name": "tahoe_ensembl", "table": "gene_metadata", "from": "ensembl_id",
                                       "to": "gene_symbol", "cardinality": "one"}],
                       "maps_to": [{"id_type": "open_targets:ensembl_gene", "via": "tahoe_ensembl"}]},
    },
    "tables": {
        "de_permissive": {"kind": "fact", "grain": "one contrast", "key": {"columns": ["drug", "gene_name"]},
                          "columns": {"drug": {"role": "identifier", "id_type": "tahoe_drug"},
                                      "gene_name": {"role": "identifier", "id_type": "tahoe_gene"}}},
        "drug_metadata": _table("drug", {"drug": {"role": "identifier", "id_type": "tahoe_drug", "self": True}}),
        "gene_metadata": _table("gene_symbol", {"gene_symbol": {"role": "identifier", "id_type": "tahoe_gene",
                                                                "self": True},
                                                "ensembl_id": {"role": "identifier"}}),
    },
}

GENES = {  # id: (approvedSymbol, biotype)
    "ENSG00000169174": ("PCSK9", "protein_coding"), "ENSG00000141510": ("TP53", "protein_coding"),
    "ENSG00000067369": ("TP53BP1", "protein_coding"), "ENSG00000184640": ("SEPTIN9", "protein_coding"),
    "ENSG00000084774": ("CAD", "protein_coding"), "ENSG00000012048": ("BRCA1", "protein_coding"),
    "ENSG00000146648": ("EGFR", "protein_coding"), "ENSG00000133703": ("KRAS", "protein_coding"),
    "ENSG00000000001": ("GENEA", "protein_coding"), "ENSG00000000002": ("GENEB", "lncRNA"),
}
DISEASES = {  # id: (name, isTherapeuticArea)
    "EFO_0000685": ("rheumatoid arthritis", "false"), "HP_0001370": ("Rheumatoid arthritis", "false"),
    "MONDO_0005148": ("type 2 diabetes mellitus", "false"), "EFO_0001645": ("coronary artery disease", "false"),
    "MONDO_0004992": ("cancer", "false"), "OBA_VT0000047": ("blood pressure trait", "false"),
    "Orphanet_558": ("Marfan syndrome", "false"), "EFO_0000319": ("cardiovascular disease", "true"),
}


@pytest.fixture(scope="module")
def registry():
    return discover(entry_points=False)


def _key(registry, plugin: str, text: str) -> str:
    return registry.get("identifier", plugin).label_key(text)


def _ot_rows(registry) -> dict[str, list[Entry]]:
    g = lambda t: _key(registry, "ensembl_gene", t)  # noqa: E731
    d = lambda t: _key(registry, "ot_disease", t)  # noqa: E731
    c = lambda t: _key(registry, "chembl_molecule", t)  # noqa: E731
    gene = []
    for gid, (sym, biotype) in GENES.items():
        gene += [Entry(g(gid), gid, "exact", sym), Entry(g(sym), gid, "label_exact:approvedSymbol", sym),
                 Entry("", gid, "attr:biotype", biotype)]
    gene += [Entry(g("NARC1"), "ENSG00000169174", "synonym:alias", "NARC1"),
             Entry(g("LFS1"), "ENSG00000141510", "synonym:alias", "LFS1"),
             Entry(g("SEPT9"), "ENSG00000184640", "synonym:previous", "SEPT9"),
             Entry(g("SEPT9"), "ENSG00000184640", "label_exact:gene_symbol", "SEPT9", stored_table="zenodo.GSE12251"),
             Entry(g("SHARED1"), "ENSG00000000001", "synonym:alias", "SHARED1"),
             Entry(g("SHARED1"), "ENSG00000000002", "synonym:alias", "SHARED1")]
    disease = []
    for did, (name, ta) in DISEASES.items():
        disease += [Entry(d(did), did, "exact", name), Entry(d(name), did, "label_exact:name", name),
                    Entry("", did, "attr:ontology.isTherapeuticArea", ta)]
    disease += [Entry(d("NIDDM"), "MONDO_0005148", "synonym:exact", "NIDDM"),
                Entry(d("CAD"), "EFO_0001645", "synonym:exact", "CAD"),
                Entry(d("T2D"), "MONDO_0005148", "synonym:related", "T2D"),
                Entry(d("diabetes"), "MONDO_0005148", "synonym:broad", "diabetes"),
                Entry(d("EFO_0001360"), "MONDO_0005148", "retired:obsoleteTerms", "EFO_0001360"),
                Entry(d("MONDO_0000001"), "EFO_0000685", "retired:obsoleteTerms", "MONDO_0000001"),
                Entry(d("MONDO_0000001"), "HP_0001370", "retired:obsoleteTerms", "MONDO_0000001"),
                Entry(d("EFO_0009999"), "", "retired:obsoleteTerms", "EFO_0009999"),
                Entry(d("DOID:9352"), "MONDO_0005148", "xref:DOID", "DOID:9352"),
                Entry(d("MESH:D003924"), "MONDO_0005148", "xref:MESH", "MESH:D003924"),
                Entry(d("DOID:7148"), "EFO_0000685", "xref:DOID", "DOID:7148"),
                Entry(d("DOID:7148"), "HP_0001370", "xref:DOID", "DOID:7148"),
                Entry(d("UMLS:C0011860"), "MONDO_0005148", "xref:UMLS", "UMLS:C0011860")]
    drug = [Entry(c("CHEMBL553"), "CHEMBL553", "exact", "ERLOTINIB"),
            Entry(c("CHEMBL1079742"), "CHEMBL1079742", "exact", "ERLOTINIB HYDROCHLORIDE", family="CHEMBL553"),
            Entry(c("CHEMBL25"), "CHEMBL25", "exact", "ASPIRIN"),
            Entry(c("ERLOTINIB"), "CHEMBL553", "label_exact:name", "ERLOTINIB"),
            Entry(c("ERLOTINIB HYDROCHLORIDE"), "CHEMBL1079742", "label_exact:name", "ERLOTINIB HYDROCHLORIDE"),
            Entry(c("Tarceva"), "CHEMBL553", "synonym:alias", "Tarceva"),
            Entry(c("Tarceva"), "CHEMBL1079742", "synonym:alias", "Tarceva")]
    words = [Entry(w.casefold(), w, "exact") for w in ("ENSG00000084774", "EFO_0001645", "CHEMBL25",
                                                       "ENSG00000169174")]
    return {"ensembl_gene": gene, "ot_disease": disease, "chembl_molecule": drug, "literature_word": words}


def _tahoe_rows(registry) -> dict[str, list[Entry]]:
    td = lambda t: _key(registry, "tahoe_drug", t)  # noqa: E731
    tg = lambda t: _key(registry, "tahoe_gene_name", t)  # noqa: E731
    drugs = [Entry(td("Erdafitinib"), "Erdafitinib", "exact"), Entry(td("Abemaciclib"), "Abemaciclib", "exact"),
             Entry(td("Erdafitinib"), "Erdafitinib", "label_exact:drug", "Erdafitinib"),
             Entry(td("Abemaciclib"), "Abemaciclib", "label_exact:drug", "Abemaciclib"),
             Entry(td("Erdafitinib "), "Erdafitinib", "stored_form", stored_table="tahoe_100m.de_permissive",
                   stored_value="Erdafitinib ")]
    genes = [Entry(tg("PCSK9"), "PCSK9", "exact"), Entry(tg("SEPT9"), "SEPT9", "exact"),
             Entry(tg("TP53"), "TP53", "exact"), Entry(tg("TP53-AS"), "TP53-AS", "exact"),
             Entry(tg("ENSG00000141510"), "TP53", "crosswalk:tahoe_ensembl", "ENSG00000141510"),
             Entry(tg("ENSG00000141510"), "TP53-AS", "crosswalk:tahoe_ensembl", "ENSG00000141510"),
             Entry(tg("ENSG00000184640"), "SEPT9", "crosswalk:tahoe_ensembl", "ENSG00000184640"),
             Entry(tg("ENSG00000169174"), "PCSK9", "crosswalk:tahoe_ensembl", "ENSG00000169174")]
    return {"tahoe_drug": drugs, "tahoe_gene": genes}


@pytest.fixture()
def world(tmp_path, registry):
    store = IndexStore(tmp_path / "cache")
    for id_type, rows in _ot_rows(registry).items():
        store.write_sidecar("open_targets", "fp-ot", id_type, rows)
    for id_type, rows in _tahoe_rows(registry).items():
        store.write_sidecar("tahoe_100m", "fp-tahoe", id_type, rows)
    catalog = Catalog({"open_targets": SourceDescriptor.model_validate(OT),
                       "tahoe_100m": SourceDescriptor.model_validate(TAHOE)})
    provider = store.provider({"open_targets": "fp-ot", "tahoe_100m": "fp-tahoe"})
    return {"store": store, "catalog": catalog, "provider": provider, "registry": registry}


def _resolver(world, remote=None, **resolution) -> Resolver:
    settings = DataSettings.from_config({"data": {"resolution": resolution}}) if resolution else None
    return Resolver(world["registry"], world["catalog"], world["provider"], remote=remote, settings=settings)


GENE_ARGS = {"accepts": ["ensembl_gene", "hgnc_symbol"], "bound_id_type": "open_targets:ensembl_gene"}


# --------------------------------------------------------------------------- CT-1 forms


def test_symbol_resolves_by_label_to_the_bound_kind(world):
    r = _resolver(world).resolve("PCSK9", **GENE_ARGS)
    assert r.status == "resolved" and r.canonical == "ENSG00000169174"
    assert r.rule == "label_exact:approvedSymbol"
    assert r.id_type == "open_targets:ensembl_gene" and r.matched_id_type == "open_targets:hgnc_symbol"
    assert r.existence == "exists" and r.label == "PCSK9" and r.source == "open_targets@25.09" and r.table == "target"
    assert any("label_exact:approvedSymbol" in n for n in r.notes)
    rec = r.record("target_id")
    assert rec.canonical == "ENSG00000169174" and rec.canonical_id_type == "open_targets:ensembl_gene"
    assert rec.rule == "label_exact:approvedSymbol" and rec.index_fingerprint == "fp-ot"
    assert r.summary() == "PCSK9 -> ENSG00000169174 (label_exact:approvedSymbol)"


@pytest.mark.parametrize("raw, rule", [("ENSG00000169174", "exact"), ("ENSG00000169174.12", "normalized:strip_version"),
                                       ("ensg00000169174", "normalized:upper")])
def test_versioned_and_lower_case_ids_record_the_folding_rule(world, raw, rule):
    r = _resolver(world).resolve(raw, **GENE_ARGS)
    assert (r.status, r.canonical, r.rule) == ("resolved", "ENSG00000169174", rule)
    assert r.matched_id_type == "open_targets:ensembl_gene"
    assert bool(r.notes) == (rule != "exact")          # folding is a rule and is reported (I1)


def test_lower_case_symbol_resolves_by_casefold_and_alias_by_synonym(world):
    res = _resolver(world)
    r = res.resolve("pcsk9", **GENE_ARGS)
    assert (r.canonical, r.rule) == ("ENSG00000169174", "label_casefold:approvedSymbol")
    r = res.resolve("NARC1", **GENE_ARGS)
    assert (r.status, r.canonical, r.rule) == ("resolved", "ENSG00000169174", "synonym:alias")
    r = res.resolve("SEPT9", **GENE_ARGS)                # a previous symbol
    assert (r.canonical, r.rule) == ("ENSG00000184640", "synonym:previous")


def test_shared_alias_is_ambiguous_with_disambiguation_values(world):
    r = _resolver(world).resolve("SHARED1", **GENE_ARGS)
    assert r.status == "ambiguous" and r.canonical is None
    assert {c.id for c in r.candidates} == {"ENSG00000000001", "ENSG00000000002"}
    by_id = {c.id: c for c in r.candidates}
    assert by_id["ENSG00000000001"].extra == {"biotype": "protein_coding"}
    assert by_id["ENSG00000000002"].extra == {"biotype": "lncRNA"} and by_id["ENSG00000000002"].label == "GENEB"
    assert all(c.via == "synonym:alias" for c in r.candidates)
    err = error_for(r, "target_id", tool="mcp__target__get_target_info")
    assert err.kind == ErrorKind.ambiguous
    assert {c["biotype"] for c in err.envelope()["candidates"]} == {"protein_coding", "lncRNA"}


def test_near_miss_symbol_is_not_found_with_suggestion(world):
    r = _resolver(world).resolve("PCSK99", **GENE_ARGS)
    assert r.status == "not_found" and r.canonical is None
    assert r.suggestions and r.suggestions[0].id == "ENSG00000169174" and r.suggestions[0].label == "PCSK9"
    assert r.suggestions[0].via == "edit distance 1"
    assert any("hgnc_symbol" in t for t in r.tried) and any("ensembl_gene" in t for t in r.tried)
    env = error_for(r, "target_id").envelope()
    assert env["kind"] == "not_found" and env["citable"] is False
    assert env["suggestions"][0] == {"id": "ENSG00000169174", "label": "PCSK9", "why": "edit distance 1"}
    assert env["source"] == "open_targets@25.09" and env["table"] == "target" and env["tried"]


def test_absent_ensembl_id_is_not_found(world):
    r = _resolver(world).resolve("ENSG00000999999", **GENE_ARGS)
    assert r.status == "not_found" and r.existence == "absent"
    assert any("ENSG00000999999 is not in the universe" in t for t in r.tried)
    assert error_for(r, "target_id").kind == ErrorKind.not_found


def test_disease_curie_and_drug_case_fold(world):
    res = _resolver(world)
    r = res.resolve("EFO:0000685", ["ot_disease", "disease_name"], bound_id_type="open_targets:ot_disease")
    assert (r.canonical, r.rule) == ("EFO_0000685", "normalized:curie_colon_to_underscore")
    r = res.resolve("orphanet_558", ["ot_disease"], bound_id_type="open_targets:ot_disease")
    assert (r.canonical, r.rule) == ("Orphanet_558", "normalized:canonical_prefix_case")
    r = res.resolve("OBA_VT0000047", ["ot_disease"], bound_id_type="open_targets:ot_disease")   # learned prefix
    assert r.status == "resolved" and r.rule == "exact"
    r = res.resolve("chembl25", ["chembl_molecule", "drug_name"], bound_id_type="open_targets:chembl_molecule")
    assert (r.canonical, r.rule) == ("CHEMBL25", "normalized:upper")


# --------------------------------------------------------------------------- retired, xref, synonyms


DISEASE_ARGS = {"accepts": ["ot_disease", "disease_name"], "bound_id_type": "open_targets:ot_disease"}


def test_retired_id_resolves_to_its_replacement(world):
    r = _resolver(world).resolve("EFO_0001360", **DISEASE_ARGS)
    assert (r.status, r.canonical, r.rule) == ("resolved", "MONDO_0005148", "retired:obsoleteTerms")
    assert any("retired" in n for n in r.notes)


def test_retired_with_several_or_no_replacement(world):
    res = _resolver(world)
    r = res.resolve("MONDO_0000001", **DISEASE_ARGS)
    assert r.status == "ambiguous" and {c.id for c in r.candidates} == {"EFO_0000685", "HP_0001370"}
    r = res.resolve("EFO_0009999", **DISEASE_ARGS)
    assert r.status == "not_found" and r.subkind == "obsolete" and r.replacement == ()
    env = error_for(r, "disease_id").envelope()
    assert env["subkind"] == "obsolete" and env["replacement"] == []


def test_xref_resolves_and_shared_xref_is_ambiguous(world):
    res = _resolver(world)
    r = res.resolve("DOID:9352", **DISEASE_ARGS)
    assert (r.status, r.canonical, r.rule) == ("resolved", "MONDO_0005148", "xref:DOID")
    assert any("doid cross-reference" in n for n in r.notes)
    assert res.resolve("doid_9352", **DISEASE_ARGS).canonical == "MONDO_0005148"      # CURIE variant
    r = res.resolve("MESH:D003924", **DISEASE_ARGS)        # MESH is not a disease prefix, but a declared xref namespace
    assert (r.canonical, r.rule) == ("MONDO_0005148", "xref:MESH")
    r = res.resolve("DOID:7148", **DISEASE_ARGS)
    assert r.status == "ambiguous" and {c.id for c in r.candidates} == {"EFO_0000685", "HP_0001370"}
    assert {c.extra["ontology.isTherapeuticArea"] for c in r.candidates} == {"false"}
    # a namespace the id_type does not declare is not used
    assert res.resolve("UMLS:C0011860", **DISEASE_ARGS).status != "resolved"


def test_synonym_kind_policy(world):
    res = _resolver(world)
    r = res.resolve("NIDDM", **DISEASE_ARGS)
    assert (r.canonical, r.rule) == ("MONDO_0005148", "synonym:exact")
    r = res.resolve("T2D", **DISEASE_ARGS)                 # related: resolves, with a warning
    assert r.rule == "synonym:related" and any(n.startswith("warning") for n in r.notes)
    r = res.resolve("diabetes", **DISEASE_ARGS)            # a broad synonym never resolves (I1)
    assert r.status == "not_found"
    assert res.resolve("T2D", **DISEASE_ARGS).status == "resolved"
    narrow = _resolver(world, allow=["normalized", "label", "exact_synonym"])
    assert narrow.resolve("T2D", **DISEASE_ARGS).status == "not_found"                  # data.resolution.allow


def test_casefold_tie_is_ambiguous_unless_prefer_is_declared(world, tmp_path):
    r = _resolver(world).resolve("RHEUMATOID ARTHRITIS", **DISEASE_ARGS)
    assert r.status == "ambiguous" and {c.id for c in r.candidates} == {"EFO_0000685", "HP_0001370"}
    assert r.rule == "label_casefold:name"
    assert _resolver(world).resolve("rheumatoid arthritis", **DISEASE_ARGS).canonical == "EFO_0000685"   # exact label
    ot = {**OT, "id_types": {**OT["id_types"], "ot_disease": {**OT["id_types"]["ot_disease"], "prefer": ["EFO"]}}}
    world = {**world, "catalog": Catalog({"open_targets": SourceDescriptor.model_validate(ot),
                                          "tahoe_100m": SourceDescriptor.model_validate(TAHOE)})}
    r = _resolver(world).resolve("RHEUMATOID ARTHRITIS", **DISEASE_ARGS)
    assert (r.status, r.canonical) == ("resolved", "EFO_0000685")
    assert any("prefer" in n for n in r.notes)


def test_universe_where_restricts_to_listed_values(world):
    allowed = {"EFO_0000319", "MONDO_0005148"}
    r = _resolver(world).resolve("cancer", universe_where=allowed, **DISEASE_ARGS)
    assert r.status == "rejected" and r.subkind == "outside_universe"
    assert r.valid_values == ("EFO_0000319", "MONDO_0005148")
    env = error_for(r, "therapeutic_area").envelope()
    assert env["kind"] == "invalid_argument" and env["valid_values"] == ["EFO_0000319", "MONDO_0005148"]
    r = _resolver(world).resolve("NIDDM", universe_where=lambda c: c in allowed, **DISEASE_ARGS)
    assert r.canonical == "MONDO_0005148"


def test_universe_where_predicate_on_index_attributes(world):
    """ArgBinding.universe_where {isTherapeuticArea = true} over the index's attr rows: an exact term that is
    not a therapeutic area is invalid_argument listing the valid values."""
    ta = {"eq": ["ontology.isTherapeuticArea", True]}
    res = _resolver(world)
    r = res.resolve("cancer", universe_where=ta, **DISEASE_ARGS)
    assert r.status == "rejected" and r.subkind == "outside_universe" and r.valid_values == ("EFO_0000319",)
    assert res.resolve("cardiovascular disease", universe_where=ta, **DISEASE_ARGS).canonical == "EFO_0000319"
    by_param = {"eq": ["ontology.isTherapeuticArea", {"param": "ta"}]}
    r = res.resolve("EFO_0000319", universe_where=by_param, where_params={"ta": True}, **DISEASE_ARGS)
    assert r.status == "resolved"
    with pytest.raises(ResolverConfigError):
        res.resolve("cancer", universe_where={"no_such_op": []}, **DISEASE_ARGS)


# --------------------------------------------------------------------------- families, crosswalks, unions


def test_trade_name_of_salt_forms_collapses_to_the_parent(world):
    r = _resolver(world).resolve("Tarceva", ["chembl_molecule", "drug_name"],
                                 bound_id_type="open_targets:chembl_molecule")
    assert (r.status, r.canonical, r.rule) == ("resolved", "CHEMBL553", "parent_family")
    assert r.family == ("CHEMBL553", "CHEMBL1079742")
    assert r.matched_id_type == "open_targets:drug_name"
    assert any("one parent" in n for n in r.notes)
    assert r.record("drug_id").family == ["CHEMBL553", "CHEMBL1079742"]
    salt = _resolver(world).resolve("CHEMBL1079742", ["chembl_molecule"], bound_id_type="open_targets:chembl_molecule")
    assert salt.canonical == "CHEMBL1079742" and salt.family == ("CHEMBL553", "CHEMBL1079742")
    no_family = _resolver(world, allow=["normalized", "label", "alias"])
    assert no_family.resolve("Tarceva", ["drug_name"], bound_id_type="open_targets:chembl_molecule").status == \
        "ambiguous"


def test_two_hop_crosswalk_to_tahoe_and_stored_forms(world):
    res = _resolver(world)
    r = res.resolve("SEPTIN9", ["tahoe_gene", "open_targets:hgnc_symbol"], bound_id_type="tahoe_100m:tahoe_gene")
    assert (r.status, r.canonical, r.rule) == ("resolved", "SEPT9", "crosswalk:tahoe_ensembl")
    assert r.id_type == "tahoe_100m:tahoe_gene" and r.matched_id_type == "open_targets:hgnc_symbol"
    assert len(r.hops) == 2 and "label_of" in r.hops[0] and "crosswalk:tahoe_ensembl" in r.hops[1]
    assert any("ENSG00000184640 (label_exact:approvedSymbol)" in n for n in r.notes)
    assert res.resolve("PCSK9", ["tahoe_gene"], bound_id_type="tahoe_100m:tahoe_gene").rule == "exact"
    r = res.resolve("ENSG00000169174.2", ["tahoe_gene"], bound_id_type="tahoe_100m:tahoe_gene")   # in-index crosswalk
    assert (r.canonical, r.rule) == ("PCSK9", "crosswalk:tahoe_ensembl")
    # the chain is longer than max_hops allows: a configuration error, never not_found
    with pytest.raises(ResolverConfigError, match="max_hops"):
        _resolver(world, max_hops=1).resolve("SEPTIN9", ["open_targets:hgnc_symbol"],
                                             bound_id_type="tahoe_100m:tahoe_gene")
    # stored forms: Tahoe DE stores 'Erdafitinib ' where the metadata stores 'Erdafitinib'
    d = res.resolve(" erdafitinib", ["tahoe_drug"], bound_id_type="tahoe_100m:tahoe_drug")
    assert (d.canonical, d.rule) == ("Erdafitinib", "label_casefold:drug")
    d = res.resolve("Erdafitinib (hydrochloride)", ["tahoe_drug"], bound_id_type="tahoe_100m:tahoe_drug")
    assert (d.canonical, d.rule) == ("Erdafitinib", "label_casefold:drug")          # salt suffix -> the name
    d = res.resolve("Erdafitinib", ["tahoe_drug"], bound_id_type="tahoe_100m:tahoe_drug")
    assert d.canonical == "Erdafitinib"
    assert res.send_value(d, "stored", "tahoe_100m.de_permissive") == "Erdafitinib "
    assert res.send_value(d, "stored", "de_permissive") == "Erdafitinib "
    assert res.send_value(d, "stored", "tahoe_100m.drug_metadata") == "Erdafitinib"
    assert res.send_value(d, "canonical") == "Erdafitinib" and res.send_value(d, "raw") == "Erdafitinib"
    assert res.resolve("Erdafitinib ", ["tahoe_drug"], bound_id_type="tahoe_100m:tahoe_drug").rule == \
        "normalized:strip"


def test_one_to_many_crosswalk_hop_is_ambiguous_and_native_labels(world):
    res = _resolver(world)
    r = res.resolve("TP53", ["open_targets:hgnc_symbol"], bound_id_type="tahoe_100m:tahoe_gene")
    assert r.status == "ambiguous" and {c.id for c in r.candidates} == {"TP53", "TP53-AS"}
    assert all(c.via == "crosswalk:tahoe_ensembl" for c in r.candidates)
    s = res.resolve("SEPTIN9", **GENE_ARGS)
    assert res.send_value(s, "native_label", "zenodo.GSE12251") == "SEPT9"         # the cohort's own symbol
    assert res.send_value(s, "native_label", "zenodo.GSE99999") == "SEPTIN9"       # no own label: the current one
    assert res.send_value(s, "label") == "SEPTIN9"
    assert res.send_value(s, "stored", "open_targets.target") == "ENSG00000184640"


def test_no_declared_edge_is_a_configuration_error(world):
    with pytest.raises(ResolverConfigError, match="no declared edge"):
        _resolver(world).resolve("CHEMBL25", ["chembl_molecule"], bound_id_type="open_targets:ensembl_gene")


def test_union_kind_word_is_ambiguous_across_members(world):
    res = _resolver(world)
    r = res.resolve("CAD", ["literature_word"], bound_id_type="open_targets:literature_word")
    assert r.status == "ambiguous"
    assert {c.id for c in r.candidates} == {"ENSG00000084774", "EFO_0001645"}
    vias = {c.id: c.via for c in r.candidates}
    assert "ensembl_gene" in vias["ENSG00000084774"] and "ot_disease" in vias["EFO_0001645"]
    r = res.resolve("ENSG00000084774", ["literature_word"], bound_id_type="open_targets:literature_word")
    assert (r.status, r.canonical, r.rule) == ("resolved", "ENSG00000084774", "exact")
    r = res.resolve("PCSK9", ["literature_word"], bound_id_type="open_targets:literature_word")
    assert (r.status, r.canonical) == ("resolved", "ENSG00000169174")
    assert r.matched_id_type == "open_targets:ensembl_gene" and r.existence == "exists"
    # a gene that has no embedding word: resolved by its member, absent from the union's universe
    r = res.resolve("TP53", ["literature_word"], bound_id_type="open_targets:literature_word")
    assert r.status == "not_found"


# --------------------------------------------------------------------------- wrong kind, existence


def test_pmcid_given_as_pmid_is_rejected_with_looks_like(world):
    r = _resolver(world).resolve("PMC1234", ["pmid"], bound_id_type="open_targets:pmid")
    assert r.status == "rejected" and "pmcid" in r.looks_like and r.reasons
    err = error_for(r, "pmids", tool="mcp__pubmed__fetch_abstracts")
    env = err.envelope()
    assert err.kind == ErrorKind.invalid_argument and "pmcid" in env["looks_like"] and env["citable"] is False
    ok = _resolver(world).resolve("PMID:30595370", ["pmid"], bound_id_type="open_targets:pmid")
    assert (ok.status, ok.canonical, ok.existence) == ("resolved", "30595370", None)     # syntax only, no universe


def test_all_kinds_rejected_but_label_rule_applies_is_not_rejected(world):
    # 'MESH:D003924' is rejected by ot_disease syntax and by disease_name, but the xref rule resolves it
    r = _resolver(world).resolve("MESH:D003924", **DISEASE_ARGS)
    assert r.status == "resolved"
    r = _resolver(world).resolve("ENSG00000169174", **DISEASE_ARGS)
    assert r.status == "rejected" and "ensembl_gene" in r.looks_like


def test_existence_modes(world):
    res = _resolver(world)
    args = {"accepts": ["chembl_molecule"], "bound_id_type": "open_targets:chembl_molecule"}
    assert res.resolve("CHEMBL99", existence="universe", **args).status == "not_found"
    r = res.resolve("CHEMBL99", existence="bound", **args)
    assert (r.status, r.canonical, r.existence) == ("resolved_unverified", "CHEMBL99", "absent") and r.notes
    assert error_for(r, "drug_id") is None
    r = res.resolve("chembl99", existence="off", **args)
    assert (r.status, r.canonical, r.existence) == ("resolved", "CHEMBL99", None)
    r = res.resolve("CHEMBL99", existence="upstream", **args)
    assert r.status == "resolved" and any("upstream" in n for n in r.notes)
    assert res.resolve("CHEMBL25", existence="off", **args).existence == "exists"
    with pytest.raises(ValueError):
        res.resolve("CHEMBL25", existence="maybe", **args)


def test_missing_index_gives_unknown_never_not_found(world):
    res = Resolver(world["registry"], world["catalog"], lambda s, t: None)
    r = res.resolve("ENSG00000999999", **GENE_ARGS)
    assert r.status == "unknown" and r.existence == "unknown" and r.canonical == "ENSG00000999999"
    assert error_for(r, "target_id") is None


class BudgetExceeded(Exception):
    pass


def test_remote_universe_budget_error_gives_unknown(world):
    def remote(source, id_type, values):
        raise BudgetExceeded("_resolve_remote over budget (50 requests/min)")

    r = _resolver(world, remote=remote).resolve("RS7412", ["rsid"], bound_id_type="open_targets:rsid")
    assert (r.status, r.existence, r.canonical) == ("unknown", "unknown", "rs7412")
    assert any("over budget" in n for n in r.notes)
    calls = []

    def remote_ok(source, id_type, values):
        calls.append((source, id_type, tuple(values)))
        return {"existence": "exists" if values == ["rs7412"] else "absent", "resolutions": []}

    res = _resolver(world, remote=remote_ok)
    assert res.resolve("rs7412", ["rsid"], bound_id_type="open_targets:rsid").status == "resolved"
    assert res.resolve("rs999", ["rsid"], bound_id_type="open_targets:rsid").status == "not_found"
    assert calls[0] == ("open_targets", "rsid", ("rs7412",))

    def remote_many(source, id_type, values):
        return {"existence": "exists", "resolutions": [{"value": "rs7412", "canonical": "19_44908822_C_T"},
                                                       {"value": "rs7412", "canonical": "19_44908822_C_A"}]}

    r = _resolver(world, remote=remote_many).resolve("rs7412", ["rsid"], bound_id_type="open_targets:rsid")
    assert r.status == "ambiguous" and len(r.candidates) == 2                  # cardinality many: never iloc[0]


async def test_async_remote_through_aresolve(world):
    async def remote(source, id_type, values):
        return {"existence": "exists", "resolutions": [{"value": values[0], "canonical": values[0]}]}

    res = _resolver(world, remote=remote)
    r = await res.aresolve("RS7412", ["rsid"], bound_id_type="open_targets:rsid")
    assert (r.status, r.canonical) == ("resolved", "rs7412")
    sync = res.resolve("RS7412", ["rsid"], bound_id_type="open_targets:rsid")   # an async callable needs aresolve
    assert sync.status == "unknown"
    results, summary = await res.aresolve_list(["rs1", "rs2"], ["rsid"], bound_id_type="open_targets:rsid")
    assert summary["status"] == "ok" and summary["canonicals"] == ["rs1", "rs2"]


# --------------------------------------------------------------------------- lists


def test_gene_list_below_min_resolved_fraction_is_insufficient(world):
    genes = ["PCSK9", "TP53", "TP53BP1", "SEPTIN9", "CAD", "BRCA1", "EGFR", "KRAS", "NOTAGENE1", "NOTAGENE2"]
    results, summary = _resolver(world).resolve_list(genes, min_resolved_fraction=0.95, **GENE_ARGS)
    assert len(results) == 10
    assert summary["status"] == "insufficient"
    assert (summary["requested"], summary["resolved"]) == (10, 8)
    assert summary["unresolved"] == ["NOTAGENE1", "NOTAGENE2"] and summary["ambiguous"] == []
    assert summary["outside_universe"] == [] and summary["duplicates"] == []
    err = list_error(summary, "gene_list")
    assert err.kind == ErrorKind.insufficient_resolution
    env = err.envelope()
    assert env["requested"] == 10 and env["resolved"] == 8 and env["min_resolved_fraction"] == 0.95
    _, ok = _resolver(world).resolve_list(genes, min_resolved_fraction=0.8, on_missing="drop_disclosed", **GENE_ARGS)
    assert ok["status"] == "partial" and len(ok["canonicals"]) == 8
    assert [i["index"] for i in ok["items"]] == [8, 9]
    _, failed = _resolver(world).resolve_list(genes, on_missing="error", **GENE_ARGS)
    assert failed["status"] == "failed"
    env = list_error(failed, "gene_list").envelope()
    assert env["kind"] == "invalid_argument" and [i["value"] for i in env["items"]] == ["NOTAGENE1", "NOTAGENE2"]


def test_list_dedupes_a_symbol_and_its_ensembl_id(world):
    vals = ["PCSK9", "ENSG00000169174", "PCSK9", "SHARED1", "TP53"]
    results, summary = _resolver(world).resolve_list(vals, on_missing="partial", **GENE_ARGS)
    assert summary["canonicals"] == ["ENSG00000169174", "ENSG00000141510"]
    assert summary["duplicates"] == ["ENSG00000169174", "PCSK9"] and summary["requested"] == 3
    assert summary["ambiguous"] == ["SHARED1"] and summary["status"] == "partial"
    assert list_error(summary, "genes") is None


# --------------------------------------------------------------------------- grammar


def test_rule_grammar():
    for text in ("exact", "raw_member", "normalized:strip_version", "normalized:strip+upper",
                 "label_exact:approvedSymbol", "label_casefold:name", "synonym:alias", "synonym:broad",
                 "retired:obsoleteTerms", "xref:DOID", "crosswalk:tahoe_ensembl", "crosswalk:a>b",
                 "parent_family", "stored_form", "label_exact:genomicLocation.chromosome"):
        assert format_rule(parse_rule(text)) == text
    for bad in ("substring", "normalized:extract_digits", "exact:x", "synonym:fuzzy", "label_exact:", "prefix"):
        with pytest.raises(RuleError):
            parse_rule(bad)
    assert parse_rule("normalized:strip+upper").steps == ("strip", "upper")
    assert parse_rule("crosswalk:a>b").chain == ("a", "b")
    assert parse_rule("synonym:narrow").search_only and not parse_rule("synonym:alias").search_only
    assert format_rule("normalized", ["strip", "upper"]) == "normalized:strip+upper"
    assert format_rule("crosswalk", ["a", "b"]) == "crosswalk:a>b"
    assert parse_entry_rule("attr:biotype").arg == "biotype"
    with pytest.raises(RuleError):
        parse_rule("attr:biotype")
    assert [r.text for r in default_rules()] == [
        "raw_member", "exact", "normalized", "label_exact", "label_casefold", "synonym:previous", "synonym:alias",
        "synonym:exact", "synonym:related", "retired", "xref", "crosswalk"]


def test_rules_for_spec_allow_and_resolvable():
    spec = SourceDescriptor.model_validate(OT).id_types
    assert [r.text for r in rules_for(spec["ensembl_gene"])][:3] == ["exact", "normalized", "label_exact:approvedSymbol"]
    narrow = rules_for(spec["ot_disease"], allow=["normalized", "label"])
    assert [r.text for r in narrow] == ["exact", "normalized", "label_exact", "label_casefold"]
    assert [r.text for r in rules_for(spec["pmid"])] == ["raw_member", "exact", "normalized"]   # resolvable: false

    class Listed:
        rules = ["exact", "synonym:broad", "synonym:narrow", "synonym:alias"]

    assert [r.text for r in rules_for(Listed())] == ["exact", "synonym:alias"]      # broad/narrow never resolve


# --------------------------------------------------------------------------- sidecars


def test_sidecar_roundtrip_header_and_atomic_write(tmp_path):
    rows = [Entry("pcsk9", "ENSG00000169174", "label_exact:approvedSymbol", "PCSK9"),
            {"label_key": "x\ty", "canonical": "K1", "rule": "synonym:alias", "label": 'tab\tand "quote"\nline'},
            ("k2", "K2", "exact", "", "", "", "K1")]
    path = tmp_path / "s" / "fp" / "index" / "t.tsv.gz"
    assert write_sidecar(path, rows) == 3
    assert [p.name for p in path.parent.iterdir()] == ["t.tsv.gz"]                # no temporary left behind
    back = list(read_sidecar(path))
    assert {e.label for e in back} == {"PCSK9", 'tab\tand "quote"\nline', ""}
    assert back == sorted(back, key=Entry.as_row)
    first = path.read_bytes()
    write_sidecar(path, list(reversed(rows)))
    assert path.read_bytes() == first                                             # deterministic bytes
    import gzip
    with gzip.open(path, "rt") as fh:
        assert fh.readline().rstrip("\n").split("\t") == list(HEADER)
    with pytest.raises(IndexFormatError):
        write_sidecar(tmp_path / "bad.tsv.gz", [Entry("a", "B", "substring")])
    bad = tmp_path / "hdr.tsv.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("key\tcanonical\n")
    with pytest.raises(IndexFormatError, match="header"):
        list(read_sidecar(bad))
    with pytest.raises(IndexMissing):
        list(read_sidecar(tmp_path / "missing.tsv.gz"))


def test_index_lookups_families_stored_forms_and_suggestions():
    idx = ResolverIndex([
        Entry("chembl553", "CHEMBL553", "exact", "ERLOTINIB"),
        Entry("chembl1079742", "CHEMBL1079742", "exact", "", family="CHEMBL553"),
        Entry("chembl2", "CHEMBL2", "exact"),
        Entry("erlotinib hydrochloride", "CHEMBL1079742", "label_exact:name", "ERLOTINIB HYDROCHLORIDE"),
        Entry("sept9", "ENSG00000184640", "label_exact:symbol", "SEPT9", stored_table="zenodo.GSE12251"),
        Entry("septin9", "ENSG00000184640", "label_exact:symbol", "SEPTIN9"),
        Entry("erdafitinib ", "Erdafitinib", "stored_form", stored_table="tahoe.de", stored_value="Erdafitinib "),
        Entry("", "CHEMBL553", "attr:drugType", "Small molecule"),
        Entry("efo_0001360", "", "retired:obsoleteTerms", "EFO_0001360"),
    ], source="s", id_type="t", fingerprint="fp")
    assert idx.contains("CHEMBL553") and not idx.contains("EFO_0001360") and idx.universe_size == 3
    assert idx.canonicals() == ("CHEMBL1079742", "CHEMBL2", "CHEMBL553")
    assert idx.family("CHEMBL1079742") == ("CHEMBL553", "CHEMBL1079742") == idx.family("CHEMBL553")
    assert idx.family("CHEMBL2") == ("CHEMBL2",) and idx.parent("CHEMBL1079742") == "CHEMBL553"
    assert idx.label("CHEMBL553") == "ERLOTINIB" and idx.label("CHEMBL1079742") == "ERLOTINIB HYDROCHLORIDE"
    assert idx.native_label("ENSG00000184640", "zenodo.GSE12251") == "SEPT9"
    assert idx.native_label("ENSG00000184640", "GSE12251") == "SEPT9"
    assert idx.native_label("ENSG00000184640", "other.GSE12251") is None
    assert idx.stored_value("Erdafitinib", "tahoe.de") == "Erdafitinib " and idx.stored_value("Erdafitinib", "x") is None
    assert idx.stored_values("Erdafitinib") == {"tahoe.de": "Erdafitinib "}
    assert idx.attributes("CHEMBL553") == {"drugType": "Small molecule"}
    assert [e.canonical for e in idx.labels_for("ENSG00000184640")] == ["ENSG00000184640"] * 2
    assert [e.canonical for e in idx.suggest("chembl55")] == ["CHEMBL553"]
    assert [e.canonical for e in idx.suggest("septin8")] == ["ENSG00000184640"]
    assert idx.suggest("chembl553") == [] or all(e.label_key != "chembl553" for e in idx.suggest("chembl553"))
    assert idx.suggest("efo_0001361") == []                                    # retired rows are never suggested
    assert len(idx) == 9 and "s:t" in repr(idx)
    assert within_one_edit("abc", "abd") and within_one_edit("abc", "ab") and within_one_edit("ab", "xab")
    assert not within_one_edit("abc", "acb") and not within_one_edit("abc", "a")


def test_index_store_paths_cache_and_provider(tmp_path):
    store = IndexStore(tmp_path)
    p = store.path("open_targets", "sha256:abc", "open_targets:ensembl_gene")
    assert p == tmp_path / "open_targets" / "sha256_abc" / "index" / "ensembl_gene.tsv.gz"
    with pytest.raises(ValueError):
        store.path("../x", "fp", "t")
    with pytest.raises(IndexMissing, match="vbt ds index build"):
        store.load("open_targets", "fp1", "ensembl_gene")
    store.write_sidecar("open_targets", "fp1", "ensembl_gene", [Entry("a", "A", "exact")])
    one = store.load("open_targets", "fp1", "ensembl_gene")
    assert store.load("open_targets", "fp1", "ensembl_gene") is one                   # cached
    assert (one.source, one.id_type, one.fingerprint) == ("open_targets", "ensembl_gene", "fp1")
    store.write_sidecar("open_targets", "fp1", "ensembl_gene", [Entry("b", "B", "exact")])
    two = store.load("open_targets", "fp1", "ensembl_gene")
    assert two is not one and two.contains("B")                                     # reloaded after a rebuild
    store.write_sidecar("open_targets", "fp2", "ensembl_gene", [Entry("c", "C", "exact")])
    assert set(store.fingerprints("open_targets", "ensembl_gene")) == {"fp1", "fp2"}
    get = store.provider({"open_targets": "fp2"})
    assert get("open_targets", "ensembl_gene").contains("C") and get("open_targets", "other") is None
    assert get("tahoe", "x") is None
    by_type = store.provider(lambda s, t: "fp1" if t == "ensembl_gene" else None)
    assert by_type("open_targets", "ensembl_gene").contains("B")


def test_sidecars_load_and_resolve_with_pyarrow_and_pandas_blocked(tmp_path):
    store = IndexStore(tmp_path)
    store.write_sidecar("open_targets", "fp", "ensembl_gene",
                        [Entry("pcsk9", "ENSG00000169174", "label_exact:approvedSymbol", "PCSK9"),
                         Entry("ensg00000169174", "ENSG00000169174", "exact", "PCSK9")])
    code = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"
        "sys.modules['pandas'] = None\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from vbt.datalayer.resolve import IndexStore, Resolver\n"
        "from vbt.datalayer.catalog import Catalog\n"
        "from vbt.datalayer.descriptor.models import SourceDescriptor\n"
        "from vbt.datalayer.plugins.registry import discover\n"
        f"store = IndexStore({str(tmp_path)!r})\n"
        "idx = store.load('open_targets', 'fp', 'ensembl_gene')\n"
        "assert idx.contains('ENSG00000169174')\n"
        "desc = SourceDescriptor.model_validate({'schema': 'vbt.datasource/1', 'source': 'open_targets', 'title': 't',\n"
        "    'release': {'from': 'literal'}, 'id_types': {'ensembl_gene': {'plugin': 'ensembl_gene',\n"
        "    'universe': 'target.id'}, 'hgnc_symbol': {'plugin': 'hgnc_symbol', 'label_of': 'ensembl_gene'}},\n"
        "    'tables': {'target': {'kind': 'entity', 'grain': 'g', 'key': {'columns': ['id']},\n"
        "    'columns': {'id': {'role': 'identifier', 'id_type': 'ensembl_gene', 'self': True}}}}})\n"
        "res = Resolver(discover(entry_points=False), Catalog({'open_targets': desc}),\n"
        "               store.provider({'open_targets': 'fp'}))\n"
        "r = res.resolve('PCSK9', ['ensembl_gene', 'hgnc_symbol'], bound_id_type='open_targets:ensembl_gene')\n"
        "assert r.canonical == 'ENSG00000169174', r\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('pyarrow', 'pandas') and sys.modules[m] is not None]\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_resolver_accepts_mapping_provider_and_settings_object(world):
    idx = world["provider"]("open_targets", "chembl_molecule")
    res = Resolver(world["registry"], world["catalog"], {"open_targets:chembl_molecule": idx},
                   settings=ResolutionSettings(max_candidates=1))
    r = res.resolve("Tarceva", ["drug_name"], bound_id_type="open_targets:chembl_molecule")
    assert r.canonical == "CHEMBL553"
    assert res.resolve("CHEMBL25", ["chembl_molecule"], bound_id_type="open_targets:chembl_molecule").existence == \
        "exists"

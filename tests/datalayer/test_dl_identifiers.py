"""Phase-1 identifier plugins (§9.4): the normalisation forms CT-1 and the stress test rely on, plugin
options (prefixes from the universe, local_key patterns), stored-value normalisation, wrong-kind hints,
and the suite's own checks catching a broken plugin (§9.3)."""

from __future__ import annotations

import re

import pytest

from vbt.datalayer.plugins.base import NORMALIZE_STEPS, IdentifierBase, Normalized, PluginError, Rejected
from vbt.datalayer.plugins.conformance import identifier as suite
from vbt.datalayer.plugins.identifiers import STRUCTURED, TEXT_KINDS, Trace, structured_hits
from vbt.datalayer.plugins.registry import discover, validate_plugin

PHASE1 = {
    "ensembl_gene", "ensembl_gene_any", "ensembl_gene_mouse", "hgnc_symbol", "ot_disease", "disease_name",
    "chembl_molecule", "drug_name", "inchikey", "go", "reactome", "so", "hpo", "pmid", "pmcid", "doi", "europepmc_ppr",
    "nct_id", "ot_variant", "chromosome", "rsid", "study_locus_id", "gwas_study", "depmap_cell_line", "tahoe_drug",
    "tahoe_gene_name", "cbio_cancer_type", "cbio_study", "cbio_sample", "cbio_patient", "uniprot_accession",
    "ncbi_taxon", "cellosaurus", "local_key", "ot_entity_any",
}


@pytest.fixture(scope="module")
def reg():
    return discover(entry_points=False)


def norm(reg, plugin: str, raw, *, stored: bool = False, options=None, sample=None):
    p = reg.get("identifier", plugin)
    if options is not None or sample is not None:
        p = p.configure(options or {}, sample)
    return p.normalize_stored(raw) if stored else p.normalize(raw)


def value(n) -> str:
    assert isinstance(n, Normalized), n
    return n.value


def test_every_phase1_plugin_is_registered_and_valid(reg):
    assert set(reg.names("identifier")) == PHASE1
    for p in reg.all("identifier"):
        validate_plugin(p)
        assert p.name == p.id_type and p.version == "1.0"
        assert re.fullmatch(p.canonical, p.examples[0]) or p.name == "local_key"


@pytest.mark.parametrize("plugin, raw, expected, steps", [
    ("ensembl_gene", "ENSG00000169174.12", "ENSG00000169174", ("strip_version",)),
    ("ensembl_gene", "ensg00000169174", "ENSG00000169174", ("upper",)),
    ("ot_disease", "EFO:0000685", "EFO_0000685", ("curie_colon_to_underscore",)),
    ("ot_disease", "orphanet_558", "Orphanet_558", ("canonical_prefix_case",)),
    ("go", "GO_0005737", "GO:0005737", ("curie_underscore_to_colon",)),
    ("reactome", "R-HSA-109582.3", "R-HSA-109582", ("strip_version",)),
    ("ot_variant", "chr19:44908822:C:T", "19_44908822_C_T", ("strip_chr", "separator_to_underscore")),
    ("pmcid", "PMC1234", "PMC1234", ()),
    ("nct_id", "nct05653258", "NCT05653258", ("upper",)),
    ("rsid", "RS7412", "rs7412", ("lower",)),
    ("hpo", "HP:0001250", "HP_0001250", ("curie_colon_to_underscore",)),
    ("chembl_molecule", "chembl25", "CHEMBL25", ("upper",)),
    ("ncbi_taxon", 9606, "9606", ()),
    ("chromosome", "chr19", "19", ("strip_chr",)),
    ("depmap_cell_line", "ach-000001", "ACH-000001", ("upper",)),
    ("cellosaurus", "CVCL:0023", "CVCL_0023", ("curie_colon_to_underscore",)),
    ("inchikey", "bsynrymutxbxsq-uhfffaoysa-n", "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", ("upper",)),
    ("doi", "https://doi.org/10.1056/NEJMoa1615664", "10.1056/NEJMoa1615664", ("strip_prefix",)),
    ("study_locus_id", "0A1B2C3D4E5F60718293A4B5C6D7E8F9", "0a1b2c3d4e5f60718293a4b5c6d7e8f9", ("lower",)),
    ("tahoe_gene_name", "ENSG00000169174.3", "ENSG00000169174", ("strip_version",)),
])
def test_normalisation_forms(reg, plugin, raw, expected, steps):
    n = norm(reg, plugin, raw)
    assert isinstance(n, Normalized) and n.value == expected and n.steps == steps
    assert set(n.steps) <= NORMALIZE_STEPS


def test_pmid_rejects_pmcid_doi_and_junk(reg):
    n = norm(reg, "pmid", "PMC1234")
    assert isinstance(n, Rejected) and n.looks_like == ("pmcid",) and "different paper" in n.reason
    assert value(norm(reg, "pmcid", "PMC1234")) == "PMC1234"
    n = norm(reg, "pmid", "10.1056/NEJMoa1615664")
    assert isinstance(n, Rejected) and "doi" in n.looks_like
    assert isinstance(norm(reg, "pmid", "n.v"), Rejected)
    assert value(norm(reg, "pmid", "PMID: 30595370")) == "30595370"
    # digits-only next to other digit kinds: agent input must carry the prefix, stored values need not
    assert isinstance(norm(reg, "pmid", "1234", options={"input_requires_prefix": True}), Rejected)
    assert value(norm(reg, "pmid", "PMID:1234", options={"input_requires_prefix": True})) == "1234"
    assert value(norm(reg, "pmid", "1234", stored=True, options={"input_requires_prefix": True})) == "1234"


def test_nct_wrong_length_is_rejected_never_padded(reg):
    n = norm(reg, "nct_id", "NCT0565325")
    assert isinstance(n, Rejected) and "8 digits" in n.reason
    assert isinstance(norm(reg, "nct_id", "NCT056532589"), Rejected)


def test_ensembl_species_hints(reg):
    n = norm(reg, "ensembl_gene", "ENSMUSG00000044254")
    assert isinstance(n, Rejected) and n.looks_like[0] == "ensembl_gene_mouse"
    assert value(norm(reg, "ensembl_gene_any", "ENSMUSG00000044254")) == "ENSMUSG00000044254"
    assert isinstance(norm(reg, "ensembl_gene", "ENST00000302118"), Rejected)
    assert value(norm(reg, "ensembl_gene", "ENSG00000002586.18_PAR_Y", stored=True)) == "ENSG00000002586"
    kept = norm(reg, "ensembl_gene", "ENSG00000002586_PAR_Y", options={"keep_suffix": True})
    assert value(kept) == "ENSG00000002586_PAR_Y"


def test_ot_disease_prefixes_from_the_universe(reg):
    sample = ["EFO_0000685", "MONDO_0005148", "Orphanet_558", "OBA_VT0000047", "NCIT_C4872"]
    p = reg.get("identifier", "ot_disease").configure({"prefixes": "from_universe"}, sample)
    assert p.prefixes == ("EFO", "MONDO", "Orphanet", "OBA", "NCIT")
    assert value(p.normalize("OBA_VT0000047")) == "OBA_VT0000047"
    assert value(p.normalize("oba:vt0000047")) == "OBA_VT0000047"
    assert value(p.normalize("ORPHANET:558")) == "Orphanet_558"
    n = p.normalize("UBERON_0002107")                     # not in this universe's prefixes
    assert isinstance(n, Rejected) and "UBERON" in n.reason
    assert re.fullmatch(p.canonical, "NCIT_C4872") and not re.fullmatch(p.canonical, "GO_0005737")
    # an empty sample keeps the default list; an explicit list wins
    assert reg.get("identifier", "ot_disease").configure({"prefixes": "from_universe"}, []).prefixes[0] == "EFO"
    only = reg.get("identifier", "ot_disease").configure({"prefixes": ["MONDO"]}, None)
    assert isinstance(only.normalize("EFO_0000685"), Rejected) and value(only.normalize("mondo_0005148")) == \
        "MONDO_0005148"
    # configure returns a copy: the registered plugin keeps the defaults
    assert value(reg.get("identifier", "ot_disease").normalize("UBERON_0002107")) == "UBERON_0002107"


def test_local_key_is_configured_in_yaml(reg):
    base = reg.get("identifier", "local_key")
    assert isinstance(base.normalize("HALLMARK_APOPTOSIS"), Rejected)            # unconfigured: accepts nothing
    p = base.configure({"canonical": "^HALLMARK_[A-Z0-9_]+$", "normalize": ["strip", "upper"]}, None)
    n = p.normalize(" hallmark_apoptosis ")
    assert isinstance(n, Normalized) and n.value == "HALLMARK_APOPTOSIS" and n.steps == ("strip", "upper")
    assert isinstance(p.normalize("KEGG_APOPTOSIS"), Rejected)
    assert p.examples == ("HALLMARK_APOPTOSIS", "HALLMARK_HYPOXIA") and "HALLMARK_APOPTOSIS" in p.describe()
    probe = base.configure({"canonical": "^[A-Za-z0-9][A-Za-z0-9_.\\- ]*$", "examples": ["SGC-CBP30"]}, None)
    assert value(probe.normalize("SGC-CBP30 ")) == "SGC-CBP30" and probe.examples == ("SGC-CBP30",)
    with pytest.raises(PluginError, match="canonical"):
        base.configure({}, None)
    with pytest.raises(PluginError, match="extract_digits"):
        base.configure({"canonical": "^x$", "normalize": ["extract_digits"]}, None)
    with pytest.raises(PluginError, match="regex"):
        base.configure({"canonical": "^(unclosed$"}, None)


def test_tahoe_stored_forms_and_salt_label_keys(reg):
    drug = reg.get("identifier", "tahoe_drug")
    n = drug.normalize_stored("Erdafitinib ")
    assert isinstance(n, Normalized) and n.value == "Erdafitinib" and n.steps == ("strip",)
    assert drug.label_key(" Erdafitinib ") == drug.label_key("erdafitinib") == "erdafitinib"
    assert drug.label_key("Erlotinib (hydrochloride)") == "erlotinib"
    gene = reg.get("identifier", "tahoe_gene_name")
    assert value(gene.normalize("SEPT9")) == "SEPT9"
    assert isinstance(gene.normalize("ENSMUSG00000044254"), Rejected)


def test_labels_are_case_preserving_with_casefold_keys(reg):
    sym = reg.get("identifier", "hgnc_symbol")
    assert value(sym.normalize("pcsk9")) == "pcsk9" and sym.label_key("PCSK9") == "pcsk9"
    assert value(sym.normalize("RS1")) == "RS1"                  # a gene symbol, not an rsID
    assert value(sym.normalize("P2RY12")) == "P2RY12"            # symbol with UniProt syntax: a declared overlap
    dep = reg.get("identifier", "depmap_cell_line")
    assert dep.label_key("NCI-H460") == dep.label_key("nci h460") == "ncih460"
    assert dep.label_key("PC-3") != dep.label_key("BxPC-3")


@pytest.mark.parametrize("raw, kind", [("ENSG00000169174", "ensembl_gene"), ("PMC1234", "pmcid"),
                                       ("rs7412", "rsid"), ("CHEMBL25", "chembl_molecule"), ("12345", "pmid"),
                                       ("ACH-000001", "depmap_cell_line"), ("EFO:0000685", "ot_disease"),
                                       ("NCT05653258", "nct_id"), ("10.1056/NEJMoa1615664", "doi")])
def test_text_kinds_reject_structured_ids_with_hints(reg, raw, kind):
    for name in ("hgnc_symbol", "disease_name", "drug_name", "tahoe_drug", "cbio_study", "cbio_sample"):
        n = norm(reg, name, raw)
        assert isinstance(n, Rejected), (name, raw)
        assert kind in n.looks_like, (name, raw, n.looks_like)
    # stored values are whatever the source holds: they are not judged by other kinds' syntax
    assert value(norm(reg, "drug_name", raw, stored=True)) == raw


def test_cbio_barcodes(reg):
    assert value(norm(reg, "cbio_sample", "tcga-a1-a0sb-01")) == "TCGA-A1-A0SB-01"
    n = norm(reg, "cbio_sample", "TCGA-A1-A0SB")
    assert isinstance(n, Rejected) and n.looks_like == ("cbio_patient",)
    n = norm(reg, "cbio_patient", "TCGA-A1-A0SB-01")
    assert isinstance(n, Rejected) and n.looks_like == ("cbio_sample",)
    assert value(norm(reg, "cbio_patient", "P-0000004")) == "P-0000004"
    assert value(norm(reg, "cbio_study", "BRCA_TCGA_PAN_CAN_ATLAS_2018")) == "brca_tcga_pan_can_atlas_2018"
    assert value(norm(reg, "cbio_cancer_type", "LUAD")) == "luad"


def test_union_syntax_and_members(reg):
    p = reg.get("identifier", "ot_entity_any")
    assert value(p.normalize("ENSG00000169174.3")) == "ENSG00000169174"
    assert value(p.normalize("efo:0000685")) == "EFO_0000685" and value(p.normalize("chembl25")) == "CHEMBL25"
    assert p.member_of("CHEMBL25") == "chembl_molecule" and p.member_of("PCSK9") is None
    n = p.normalize("PCSK9")
    assert isinstance(n, Rejected) and "ensembl_gene" in n.reason
    q = p.configure({"members": ["reactome", "go"]}, None)
    assert value(q.normalize("GO_0005737")) == "GO:0005737" and isinstance(q.normalize("CHEMBL25"), Rejected)
    with pytest.raises(PluginError):
        p.configure({"members": ["no_such_kind"]}, None)
    # the union's prefixes follow the universe like ot_disease's
    learned = p.configure({"prefixes": "from_universe"}, ["ENSG00000169174", "MONDO_0005148"])
    assert isinstance(learned.normalize("EFO_0000685"), Rejected) and value(learned.normalize("MONDO_0005148"))


def test_looks_like_scores(reg):
    assert reg.get("identifier", "pmcid").looks_like("PMC1234") == 1.0
    assert reg.get("identifier", "pmid").looks_like("PMC1234") == 0.0
    assert reg.get("identifier", "hgnc_symbol").looks_like("PCSK9") < suite_threshold()
    assert structured_hits("PMC1234") == ("pmcid",)
    assert "rsid" not in structured_hits("RS1") and "rsid" in structured_hits("RS7412")
    assert {k for k, _ in STRUCTURED} & TEXT_KINDS == set()


def suite_threshold() -> float:
    from vbt.datalayer.resolve.resolver import HINT_THRESHOLD
    return HINT_THRESHOLD


def test_describe_is_short_and_names_an_example(reg):
    for p in reg.all("identifier"):
        text = p.describe()
        assert 0 < len(text) <= 200 and any(e in text for e in p.examples), (p.name, text)


def test_trace_records_only_changing_steps():
    t = Trace("ENSG00000169174").strip().upper()
    assert t.done() == Normalized("ENSG00000169174", ())
    t = Trace(" x ").strip().strip().upper()
    assert t.done() == Normalized("X", ("strip", "upper"))


# --------------------------------------------------------------------------- the suite catches broken plugins


class _DigitsGrabber(IdentifierBase):
    """Reads 'PMC1234' as PMID 1234 (the upstream bug) with an off-whitelist step."""

    name = id_type = "dl_bad_pmid"
    canonical = r"^\d+$"
    examples = ("1234",)

    def normalize(self, raw):
        m = re.search(r"\d+", str(raw))
        return Normalized(m.group(0), ("extract_digits",)) if m else Rejected("no digits")


def test_suite_checks_catch_a_broken_plugin(reg):
    bad = _DigitsGrabber()
    with pytest.raises(AssertionError, match="extract_digits"):
        suite.check_steps(bad)
    problems = suite.confusion(bad, reg.all("identifier"))
    assert any("normalize('PMC1234') (an example of pmcid)" in p for p in problems)
    with pytest.raises(AssertionError, match="I-6"):
        suite.check_describe(type("Mute", (_DigitsGrabber,), {"describe": lambda self: "digits"})())


def test_overlaps_are_symmetric_and_representatives_are_configured(reg):
    ot, go = reg.get("identifier", "ot_disease"), reg.get("identifier", "go")
    assert suite.exempt(ot, go) and suite.exempt(go, ot)
    assert not suite.exempt(reg.get("identifier", "pmid"), reg.get("identifier", "pmcid"))
    rep = suite.representative(reg.get("identifier", "local_key"))
    assert rep.canonical == "^HALLMARK_[A-Z0-9_]+$" and value(rep.normalize("HALLMARK_HYPOXIA"))
    assert suite.universe_cases(ot) and not suite.universe_cases(reg.get("identifier", "rsid"))
    suite.check_universe_options(ot, reg.all("identifier"))

"""The data child's hidden verbs (service/verbs/*.py; §11.6, §11.8, §11.5 index content).

Every verb is called in-process with JSON payloads and its response is validated with the ipc
response model, as the gateway's service client does.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import dl_fixtures as F  # noqa: E402
import yaml  # noqa: E402

from vbt.datalayer.ipc import PHASE1_VERBS, parse_response  # noqa: E402
from vbt.datalayer.resolve.index import read_sidecar  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.verbs import load_verbs  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

V = load_verbs()
TAHOE_GROUP = ["drug", "concentration", "concentration_unit", "Cell_ID_DepMap", "plate"]


def call(ctx: ServiceContext, name: str, /, **payload: Any) -> Any:
    out = V[name](ctx, payload)
    return parse_response(name, out)


def _write(root: Path, name: str, rows: list[dict[str, Any]]) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    pq.write_table(F.table(name, rows), root / name / "part-00000.parquet", row_group_size=4)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> ServiceContext:
    tmp = tmp_path_factory.mktemp("verbs")
    ot = tmp / "ot"
    data = F.ot_rows()
    disease = copy.deepcopy(data["disease"])
    for r in disease:
        if r["id"] == F.T2D:
            r["dbXRefs"] = ["DOID:9352", "MESH:D003924", "UMLS:C0011860", "ICD10:E11"]
        if r["id"] == F.RA:
            r["dbXRefs"] = ["DOID:7148"]
    drugs = copy.deepcopy(data["drug_molecule"])
    for r in drugs:
        if r["id"] == "CHEMBL4001":
            r["parentId"] = "CHEMBL4000"                # a salt of CHEMBL4000
    for name, rows in (("target", data["target"]), ("known_drug", data["known_drug"]), ("disease", disease),
                       ("disease_phenotype", data["disease_phenotype"]), ("drug_molecule", drugs)):
        _write(ot, name, rows)
    tahoe = F.build_tahoe_fixture(tmp / "tahoe")
    syn = {"role": "synonym", "of": "id", "path": "[]"}
    ot_desc = {
        "schema": "vbt.datasource/1", "source": "ot", "title": "ot", "root": str(ot), "release": {"from": "literal"},
        "defaults": {"format": "parquet", "layout": "sharded_dir"},
        "id_types": {
            "ensembl_gene": {"plugin": "ensembl_gene", "universe": "target.id",
                             "resolve_via": ["target.approvedSymbol", "target.symbolSynonyms", "target.obsoleteSymbols"],
                             "disambiguate_with": ["biotype"]},
            "ot_disease": {"plugin": "ot_disease", "options": {"prefixes": "from_universe"}, "universe": "disease.id",
                           "resolve_via": ["disease.name", "disease.synonyms"],
                           "retired": {"listed_in": ["disease.obsoleteTerms"]},
                           "xref_via": [{"column": "disease.dbXRefs", "namespaces": {"DOID": "doid", "MESH": "mesh",
                                                                                      "UMLS": "umls"}}],
                           "disambiguate_with": ["ontology.isTherapeuticArea"]},
            "remote_gene": {"plugin": "ensembl_gene", "universe": "target.id", "index": "remote"},
            "chembl_molecule": {"plugin": "chembl_molecule", "universe": "drug_molecule.id",
                                "resolve_via": ["drug_molecule.name", "drug_molecule.tradeNames"],
                                "canonicalize": {"parent": "drug_molecule.parentId"}},
        },
        "tables": {
            "target": {"kind": "entity", "path": "target", "grain": "gene", "key": {"columns": ["id"]},
                       "rank": [{"column": "id", "direction": "asc"}],
                       "columns": {"id": {"role": "identifier", "id_type": "ensembl_gene", "self": True},
                                   "approvedSymbol": {"role": "label", "of": "id"},
                                   "approvedName": {"role": "label", "of": "id"},
                                   "biotype": {"role": "category"},
                                   "symbolSynonyms": {"role": "synonym", "of": "id", "synonym_kind": "alias",
                                                      "path": "[].label"},
                                   "obsoleteSymbols": {"role": "synonym", "of": "id", "synonym_kind": "previous",
                                                       "path": "[].label"},
                                   "go": {"role": "nested", "item_key": ["id", "aspect", "evidence", "source",
                                                                         "geneProduct"],
                                          "fields": {"id": {"role": "identifier"}, "aspect": {"role": "category"},
                                                     "evidence": {"role": "category"}, "source": {"role": "category"},
                                                     "geneProduct": {"role": "identifier"}}}}},
            "target_go": {"kind": "sets", "items_of": {"table": "target", "path": "go[]"}, "grain": "annotation",
                          "key": {"columns": []}},
            "known_drug": {"kind": "fact", "path": "known_drug", "grain": "record",
                           "key": {"columns": list(F.KNOWN_DRUG_KEY), "nullable": ["phase", "status"]},
                           "grains": {"drug": ["drugId"],
                                      "parent_drug": {"columns": ["drugId"], "canonicalize": "chembl_molecule.parent"},
                                      "pair": {"unordered": ["drugId", "diseaseId"]}},
                           "rank": [{"column": "phase", "direction": "desc"}],
                           "columns": {"drugId": {"role": "identifier"}, "targetId": {"role": "identifier"},
                                       "diseaseId": {"role": "identifier"},
                                       "phase": {"role": "measure", "statistic": "clinical_phase", "scale": [0, 4]},
                                       "status": {"role": "category"}}},
            "disease": {"kind": "ontology", "path": "disease", "grain": "term", "key": {"columns": ["id"]},
                        "columns": {"id": {"role": "identifier", "id_type": "ot_disease", "self": True},
                                    "name": {"role": "label", "of": "id"},
                                    "synonyms": {"role": "nested", "fields": {
                                        "hasExactSynonym": {**syn, "synonym_kind": "exact"},
                                        "hasRelatedSynonym": {**syn, "synonym_kind": "related"},
                                        "hasBroadSynonym": {**syn, "synonym_kind": "broad"},
                                        "hasNarrowSynonym": {**syn, "synonym_kind": "narrow"}}},
                                    "obsoleteTerms": {"role": "payload"}, "dbXRefs": {"role": "payload"},
                                    "ontology": {"role": "nested", "fields": {"isTherapeuticArea": {"role": "flag"},
                                                                              "leaf": {"role": "flag"}}}}},
            "drug_molecule": {"kind": "entity", "path": "drug_molecule", "grain": "molecule", "key": {"columns": ["id"]},
                              "columns": {"id": {"role": "identifier", "id_type": "chembl_molecule", "self": True},
                                          "name": {"role": "label", "of": "id"}, "parentId": {"role": "identifier"},
                                          "tradeNames": {"role": "synonym", "of": "id", "synonym_kind": "alias",
                                                         "path": "[]"}}},
            "disease_phenotype": {"kind": "fact", "path": "disease_phenotype", "grain": "pair",
                                  "key": {"columns": ["disease", "phenotype"]},
                                  "columns": {"disease": {"role": "identifier"}, "phenotype": {"role": "identifier"},
                                              "evidence": {"role": "nested", "item_key": {"identity": "position"},
                                                           "fields": {"evidenceType": {"role": "category"},
                                                                      "qualifierNot": {"role": "qualifier",
                                                                                       "effect": "negate"}}}}},
            "disease_phenotype_evidence": {"kind": "fact", "grain": "evidence item", "key": {"columns": []},
                                           "items_of": {"table": "disease_phenotype", "path": "evidence[]"}},
        },
    }
    tahoe_desc = {
        "schema": "vbt.datasource/1", "source": "tahoe_100m", "title": "tahoe", "root": str(tahoe),
        "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "single_file"},
        "id_types": {
            "tahoe_drug": {"plugin": "tahoe_drug", "universe": "drug_metadata.drug",
                           "stored_forms": {"de_permissive.drug": "as_stored"}},
            "tahoe_gene": {"plugin": "tahoe_gene_name", "universe": "gene_metadata.gene_symbol",
                           "crosswalks": [{"name": "tahoe_ensembl", "table": "gene_metadata", "from": "ensembl_id",
                                           "to": "gene_symbol"}]},
        },
        "tables": {
            "de_permissive": {"kind": "fact", "path": F.TAHOE_DE, "grain": "contrast",
                              "key": {"columns": list(F.TAHOE_KEY)},
                              "rank": [{"column": "padj", "direction": "asc", "within": TAHOE_GROUP}],
                              "access_paths": [{"columns": ["gene_name"], "via": "sidecar_index"}],
                              "columns": {"drug": {"role": "identifier", "id_type": "tahoe_drug",
                                                   "ref": "drug_metadata.drug"},
                                          "concentration": {"role": "scope", "statistic": "numeric"},
                                          "concentration_unit": {"role": "scope"},
                                          "Cell_ID_DepMap": {"role": "identifier"},
                                          "plate": {"role": "scope", "scope": {"kind": "replicate"}},
                                          "gene_name": {"role": "identifier"},
                                          "padj": {"role": "measure", "scale": [0, 1]},
                                          "log2FoldChange": {"role": "measure"}}},
            "drug_metadata": {"kind": "entity", "path": "metadata/drug_metadata.parquet", "grain": "drug",
                              "key": {"columns": ["drug"]}, "columns": {"drug": {"role": "identifier", "self": True}}},
            "gene_metadata": {"kind": "crosswalk", "path": "metadata/gene_metadata.parquet", "grain": "gene",
                              "key": {"columns": ["gene_symbol"]},
                              "columns": {"gene_symbol": {"role": "identifier"}, "ensembl_id": {"role": "identifier"}}},
            "cell_line_metadata": {"kind": "entity_detail", "path": "metadata/cell_line_metadata.parquet",
                                   "grain": "driver alteration of a cell line",
                                   "key": {"columns": ["cell_name", "Driver_Gene_Symbol"]},
                                   "columns": {"cell_name": {"role": "label"},
                                               "Cell_ID_DepMap": {"role": "identifier"},
                                               "Driver_Gene_Symbol": {"role": "identifier"}}},
        },
    }
    (tmp / "sources").mkdir()
    (tmp / "overlays").mkdir()
    for d in (ot_desc, tahoe_desc):
        (tmp / "sources" / f"{d['source']}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp / "sources"), "overlays_dir": str(tmp / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=tmp)
    return ServiceContext(settings)


def test_every_phase1_verb_is_registered_and_schema_valid(ctx):
    assert set(PHASE1_VERBS) <= set(V)
    stats = call(ctx, "_stats", tables=["ot.known_drug", "ot.target", "ot.target_go"])
    kd = stats.tables["ot.known_drug"]
    assert kd.rows == len(F.ot_rows()["known_drug"]) and kd.fragments == 1 and kd.fingerprint.startswith("fp1:")
    assert kd.columns["phase"].null_count == 1 and kd.columns["urls[].url"].max_rep_level == 1
    assert stats.tables["ot.target"].row_bytes_p99["row"] > 0 and "target_go" in stats.tables["ot.target"].row_bytes_p99
    assert stats.tables["ot.target_go"].rows == 2
    assert call(ctx, "_check", tables=["ot.target"], depth="shallow").tables["ot.target"].status == "ready"
    w = call(ctx, "_witness", table="ot.known_drug", predicate={"eq": ["targetId", F.T]})
    assert w.total == 37 and w.total_method == "scan"
    s = call(ctx, "_serve", table="ot.target", verb="lookup", predicate={"eq": ["id", F.PCSK9]})
    assert s.rows[0]["approvedSymbol"] == "PCSK9" and s.served_by == "derived" and s.row_keys == [[F.PCSK9]]
    v = call(ctx, "_vocab", table="tahoe_100m.de_permissive", column="concentration")
    assert v.values == [0.05, 0.5, 5.0] and v.rendered == ["0.05", "0.5", "5.0"] and v.storage_type == "float"
    from collections import Counter
    want = Counter(str(r["concentration"]) for r in F.tahoe_rows())
    assert v.counts == dict(want)
    b = call(ctx, "_build_index", source="ot", id_type="ensembl_gene")
    assert b.rows > 0 and Path(b.path).exists()
    r = call(ctx, "_resolve_remote", source="ot", id_type="remote_gene", values=[F.PCSK9])
    assert r.existence == "exists" and r.resolutions[0]["canonical"] == F.PCSK9


def test_witness_topk_per_group_and_group_totals(ctx):
    w = call(ctx, "_witness", table="tahoe_100m.de_permissive", predicate={"eq": ["drug", F.BORTEZOMIB]},
             order=[{"column": "padj", "direction": "asc", "within": TAHOE_GROUP}], k=1,
             key=list(F.TAHOE_KEY), grains={"genes": ["gene_name"], "doses": ["concentration"]},
             distinct=["plate"], unknown_columns=["padj"])
    rows = F.oracle_tahoe(Path(ctx.catalog.source("tahoe_100m").root), F.BORTEZOMIB)
    assert w.total == len(rows) and sum(w.group_totals.values()) == w.total
    assert isinstance(w.topk, dict) and len(w.topk) == len(w.group_totals)
    for group, keys in w.topk.items():
        assert len(keys) == 1
        members = [r for r in rows if [r[c] for c in TAHOE_GROUP] == [*_json(group)[:1], *_json(group)[1:]]]
        best = min(members, key=lambda r: r["padj"])
        assert keys[0][-1] == best["gene_name"]
    assert w.distinct_counts == {"genes": len({r["gene_name"] for r in rows}), "doses": 3}
    assert w.distinct == {"plate": ["1", "2"]} and w.key_set is not None and len(w.key_set) == w.total
    assert w.excluded_unknown == {} and w.unknown_total == 0


def _json(text: str) -> list[Any]:
    import json
    return json.loads(text)


def test_witness_one_to_many_and_unbound_parameter(ctx):
    w = call(ctx, "_witness", table="ot.known_drug", predicate={"eq": ["targetId", F.T]}, key=["drugId"])
    assert w.one_to_many and all(n > 1 for n in w.one_to_many.values())
    u = call(ctx, "_witness", table="ot.known_drug", predicate={"eq": ["targetId", {"param": "target_id"}]})
    assert u.total_method == "unknown" and "target_id" in u.reason
    p = call(ctx, "_witness", table="ot.known_drug", predicate={"eq": ["targetId", {"param": "target_id"}]},
             params={"target_id": F.T})
    assert p.total == 37


def test_serve_nest_after_negation(ctx):
    s = call(ctx, "_serve", table="ot.disease_phenotype_evidence", predicate={"eq": ["/phenotype", F.SEIZURE]},
             nest={"group_by": ["disease"], "items": "evidence", "count_as": "n_evidence", "having": {"min": 1}})
    by = {g["disease"]: g["n_evidence"] for g in s.rows}
    assert by == {F.PHENO["Y"]: 1, F.PHENO["Z"]: 2}, "X and W hold only negated items"
    assert s.total == 2 and s.sections["_excluded"] == {"negated": 3}
    with_neg = call(ctx, "_serve", table="ot.disease_phenotype_evidence",
                    predicate={"eq": ["/phenotype", F.SEIZURE]},
                    nest={"group_by": ["disease"], "count_as": "n", "include_negated": True})
    assert with_neg.total == 4


def test_serve_split_limit_grain_and_sections(ctx):
    s = call(ctx, "_serve", table="ot.known_drug", predicate={"eq": ["targetId", F.T]}, columns=["drugId", "status"],
             split={"by": "status", "limit": {"Completed": 2, "None": 1}})
    assert isinstance(s.rows, dict) and len(s.rows["Completed"]) == 2 and len(s.rows["None"]) == 1
    g = call(ctx, "_serve", table="ot.known_drug", predicate={"eq": ["targetId", F.T]}, columns=["drugId", "phase"],
             limit=3, limit_grain="drug")
    assert len(g.rows) == 3 and len({r["drugId"] for r in g.rows}) == 3 and g.truncated
    assert g.grains["drug"]["total"] == len({r["drugId"] for r in F.ot_rows()["known_drug"] if r["targetId"] == F.T})
    lk = call(ctx, "_serve", table="ot.target", verb="lookup", predicate={"eq": ["id", F.PCSK9]},
              columns=["id", "approvedSymbol"],
              sections={"go": {"table": "ot.target_go", "key": {"id": F.PCSK9}},
                        "drivers": {"table": "tahoe_100m.cell_line_metadata", "key": {"Cell_ID_DepMap": F.A549},
                                    "single": True}})
    assert [r["id"] for r in lk.sections["go"]] == ["GO:0006629", "GO:1000001"]
    assert isinstance(lk.sections["drivers"], list) and len(lk.sections["drivers"]) == 3, \
        "an entity_detail section is a list even when single"
    with pytest.raises(Exception):
        call(ctx, "_serve", table="ot.target", verb="similar")


def test_search_ranking_and_match_labels(ctx):
    s = call(ctx, "_serve", table="ot.target", verb="search", search_text="TP53")
    assert [r["approvedSymbol"] for r in s.rows][:3] == ["TP53", "TP53BP1", "TP53I3"]
    assert [r["_match"]["class"] for r in s.rows][:3] == ["exact", "prefix", "prefix"]
    alias = call(ctx, "_serve", table="ot.target", verb="search", search_text="narc1")
    assert alias.rows[0]["id"] == F.PCSK9 and alias.rows[0]["_match"]["class"] == "alias"
    prev = call(ctx, "_serve", table="ot.target", verb="search", search_text="HCHOLA3")
    assert prev.rows[0]["_match"]["class"] == "previous_synonym"
    word = call(ctx, "_serve", table="ot.target", verb="search", search_text="tumor")
    assert {r["_match"]["class"] for r in word.rows} == {"prefix"} and len(word.rows) == 3
    sub = call(ctx, "_serve", table="ot.target", verb="search", search_text="p53")
    assert [r["_match"]["class"] for r in sub.rows] == ["word", "word", "word"]
    d = call(ctx, "_serve", table="ot.disease", verb="search", search_text="T2DM")
    assert d.rows[0]["id"] == F.T2D and d.rows[0]["_match"]["class"] == "related_synonym"


def test_build_index_rows_retired_xref_attr_stored_forms_and_crosswalks(ctx):
    b = call(ctx, "_build_index", source="ot", id_type="ot_disease")
    entries = list(read_sidecar(b.path))
    rules = {(e.rule, e.label, e.canonical) for e in entries}
    assert ("exact", F.T2D_NAME, F.T2D) in rules
    assert ("retired:obsoleteTerms", F.T2D_RETIRED, F.T2D) in rules
    assert ("xref:DOID", "DOID:9352", F.T2D) in rules and ("xref:MESH", "MESH:D003924", F.T2D) in rules
    assert not any(e.rule == "xref:ICD10" for e in entries), "unmapped namespaces are not indexed"
    assert ("synonym:exact", "NIDDM", F.T2D) in rules and ("synonym:related", "T2DM", F.T2D) in rules
    assert ("label_exact:name", F.T2D_NAME, F.T2D) in rules
    assert ("attr:ontology.isTherapeuticArea", "true", "EFO_0000651") in rules
    assert ctx.index_store.load("ot", b.fingerprint, "ot_disease").contains(F.T2D)
    assert b.fingerprint == ctx.reader("ot.disease").fingerprint()
    t = call(ctx, "_build_index", source="tahoe_100m", id_type="tahoe_drug")
    stored = [e for e in read_sidecar(t.path) if e.rule == "stored_form"]
    assert ("Erdafitinib", "tahoe_100m.de_permissive", "Erdafitinib ") in \
        [(e.canonical, e.stored_table, e.stored_value) for e in stored]
    g = call(ctx, "_build_index", source="tahoe_100m", id_type="tahoe_gene")
    cw = {(e.label, e.canonical) for e in read_sidecar(g.path) if e.rule == "crosswalk:tahoe_ensembl"}
    assert ("ENSG00000999001", F.SELECTIVE) in cw
    gene = list(read_sidecar(call(ctx, "_build_index", source="ot", id_type="ensembl_gene").path))
    assert ("synonym:previous", "HCHOLA3", F.PCSK9) in {(e.rule, e.label, e.canonical) for e in gene}
    assert ("attr:biotype", "protein_coding") in {(e.rule, e.label) for e in gene if e.canonical == F.PCSK9}


def test_resolve_remote_existence_states(ctx, tmp_path):
    mixed = call(ctx, "_resolve_remote", source="ot", id_type="remote_gene", values=[F.PCSK9, F.UNKNOWN_GENE])
    assert mixed.existence == "unknown"
    assert [r["existence"] for r in mixed.resolutions] == ["exists", "absent"]
    absent = call(ctx, "_resolve_remote", source="ot", id_type="remote_gene", values=[F.UNKNOWN_GENE])
    assert absent.existence == "absent"
    norm = call(ctx, "_resolve_remote", source="ot", id_type="remote_gene", values=[F.PCSK9 + ".12"])
    assert norm.resolutions[0]["rule"] == "normalized:strip_version"
    tight = ServiceContext(DataSettings.from_dict({**{k: str(v) for k, v in (
        ("descriptors_dir", ctx.settings.descriptors_dir), ("overlays_dir", ctx.settings.overlays_dir),
        ("cache_dir", tmp_path))}, "witness": {"max_scan_bytes": 10}}, project_root=tmp_path))
    inside = "ENSG00000100000"                         # inside every row group's id range: nothing prunes it
    assert call(ctx, "_resolve_remote", source="ot", id_type="remote_gene", values=[inside]).existence == "absent"
    over = call(tight, "_resolve_remote", source="ot", id_type="remote_gene", values=[inside])
    assert over.existence == "unknown", "over budget is never absent"
    pruned = call(tight, "_resolve_remote", source="ot", id_type="remote_gene", values=[F.UNKNOWN_GENE])
    assert pruned.existence == "absent", "row-group statistics prove absence without reading"


def test_build_access_path_index(ctx):
    b = call(ctx, "_build_index", table="tahoe_100m.de_permissive", access_path=["gene_name"])
    assert b.rows == len({r["gene_name"] for r in F.tahoe_rows()}) and Path(b.path).name == "de_permissive.gene_name.idx"
    assert b.fingerprint == ctx.reader("tahoe_100m.de_permissive").fingerprint()
    with pytest.raises(Exception, match="no sidecar_index access path"):
        call(ctx, "_build_index", table="ot.known_drug", access_path=["drugId"])


def test_parent_families_in_index_and_canonicalized_grains(ctx):
    b = call(ctx, "_build_index", source="ot", id_type="chembl_molecule")
    entries = {(e.rule, e.canonical): e for e in read_sidecar(b.path)}
    assert entries[("exact", "CHEMBL4001")].family == "CHEMBL4000" and entries[("exact", "CHEMBL4000")].family == ""
    assert ("synonym:alias", "CHEMBL25") in entries and entries[("synonym:alias", "CHEMBL25")].label == "Aspirin"
    w = call(ctx, "_witness", table="ot.known_drug", predicate={"eq": ["targetId", F.T]},
             grains={"drug": ["drugId"],
                     "parent_drug": {"columns": ["drugId"], "canonicalize": "chembl_molecule.parent"},
                     "pair": {"unordered": ["drugId", "diseaseId"]}})
    mine = [r for r in F.ot_rows()["known_drug"] if r["targetId"] == F.T]
    drugs = {r["drugId"] for r in mine}
    assert w.distinct_counts["drug"] == len(drugs)
    assert w.distinct_counts["parent_drug"] == len(drugs) - 1, "a salt counts with its parent"
    assert w.distinct_counts["pair"] == len({(r["drugId"], r["diseaseId"]) for r in mine})


def test_serve_ranks_within_groups_and_keeps_distinct_rows(ctx):
    s = call(ctx, "_serve", table="tahoe_100m.de_permissive", predicate={"eq": ["drug", F.BORTEZOMIB]},
             columns=["gene_name", "padj", *TAHOE_GROUP], limit=1)
    rows = F.oracle_tahoe(Path(ctx.catalog.source("tahoe_100m").root), F.BORTEZOMIB)
    groups = {tuple(r[c] for c in TAHOE_GROUP) for r in rows}
    assert len(s.rows) == len(groups), "the table rank is within groups: limit applies per group"
    for r in s.rows:
        members = [x for x in rows if all(x[c] == r[c] or (c == "concentration" and abs(x[c] - r[c]) < 1e-6)
                                          for c in TAHOE_GROUP)]
        assert r["padj"] == pytest.approx(min(m["padj"] for m in members))
    d = call(ctx, "_serve", table="tahoe_100m.de_permissive", predicate={"eq": ["drug", F.BORTEZOMIB]},
             columns=["plate"], distinct=["plate"], order=[{"column": "plate", "direction": "asc"}])
    assert [r["plate"] for r in d.rows] == ["1", "2"]

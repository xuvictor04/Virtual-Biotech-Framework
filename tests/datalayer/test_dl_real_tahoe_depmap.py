"""Tahoe-100M, DepMap 24Q4, GO, CL, MSigDB Hallmark and the Zenodo archive against the shipped descriptors.

``tests/datalayer/real/`` holds small records of the real files, retrieved on 2026-10-08 (each JSON names its
URL and how it was read):

* ``tahoe/de_shards.json``: the Parquet footers of all 1,026 pseudobulk DE shards of ``tahoebio/Tahoe-100M``
  at revision ``2dc57900`` (schema, row counts and per-row-group statistics, summarised).
* ``tahoe/metadata.json``: the four metadata tables (schemas, sha256, facts and row excerpts).
* ``tahoe/prepared_sample.json``: 13 contrasts read with range requests, the manifest of
  ``tools/prepare_tahoe.py`` run unchanged on six of them, and 30 of the prepared rows.
* ``depmap/``: the figshare listing of DepMap Public 24Q4, matrix facts and CSV excerpts (whole rows and
  columns of the real files).
* ``ontology/``: whole stanzas of go-basic.obo and cl-basic.obo, three Hallmark lines, and their counts.

Offline, the excerpts are laid out as the descriptors expect and checked with the data child's readiness
code in process. With ``VBT_DL_REAL_DATA=<dir>`` (the shared ``data/real`` layout: ``tahoe/<rev>/{metadata,
sample_prepared,footer_stats}``, ``depmap/24Q4``, ``gene_ontology/current``, ``cell_ontology/current``,
``msigdb/2024.1.Hs``) the full files are checked, and the Zenodo facts are recomputed on the archive
(``VBT_ZENODO_DIR``, ``<dir>/../zenodo`` or ``data/zenodo``). With ``VBT_DL_NETWORK=1`` one DE footer, the figshare listing and
the Hallmark file are read live.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from vbt.datalayer.catalog import build_catalog  # noqa: E402
from vbt.datalayer.descriptor.load import load_yaml  # noqa: E402
from vbt.datalayer.plugins.base import Normalized  # noqa: E402
from vbt.datalayer.plugins.registry import discover  # noqa: E402
from vbt.datalayer.rowkey import render_float  # noqa: E402
from vbt.datalayer.service import ServiceContext  # noqa: E402
from vbt.datalayer.service.checks import check_table  # noqa: E402
from vbt.datalayer.settings import DataSettings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "configs" / "data" / "sources"
OVERLAYS = REPO / "configs" / "data" / "overlays"
FIX = Path(__file__).resolve().parent / "real"
REGISTRY = discover(entry_points=False)
REV = "2dc57900b7981cfcf5e211527169a0b006546a95"
REAL = Path(os.environ["VBT_DL_REAL_DATA"]) if os.environ.get("VBT_DL_REAL_DATA") else None
NETWORK = os.environ.get("VBT_DL_NETWORK") == "1"
needs_real = pytest.mark.skipif(REAL is None, reason="set VBT_DL_REAL_DATA=<dir> to check the real files")
needs_network = pytest.mark.skipif(not NETWORK, reason="set VBT_DL_NETWORK=1 to read the live sources")
ARROW = {"string": pa.string(), "float": pa.float32(), "double": pa.float64(), "int64": pa.int64()}
CONTRAST = ["drug", "concentration", "concentration_unit", "Cell_ID_DepMap", "plate"]
# tools/prepare_tahoe.py REQUIRED_COLUMNS (the upstream preparation refuses shards without them)
PREPARE_REQUIRED = {"gene_name", "drug", "Cell_ID_DepMap", "padj", "log2FoldChange", "baseMean"}


def fixture(*parts: str) -> Any:
    return json.loads(FIX.joinpath(*parts).read_text())


def descriptor(name: str) -> dict[str, Any]:
    return load_yaml(SOURCES / f"{name}.yaml", {"project_root": str(REPO)})


def make_ctx(tmp: Path, env: dict[str, str]) -> ServiceContext:
    """The data child's context on the shipped descriptors and overlays, with ``${NAME}`` roots from ``env``."""
    settings = DataSettings.from_dict({"descriptors_dir": str(SOURCES), "overlays_dir": str(OVERLAYS),
                                       "cache_dir": str(tmp / "cache")}, project_root=REPO)
    variables = {"project_root": str(REPO), **{f"env.{k}": v for k, v in env.items()}}
    return ServiceContext(settings, catalog=build_catalog(settings, REGISTRY, variables=variables), registry=REGISTRY)


def failures(model: Any) -> list[str]:
    return [f"{c.name} {c.column or ''}: {c.detail}" for c in model.checks if not c.ok and c.level == "error"]


def checked(ctx: ServiceContext, refs: list[str], depth: str = "deep") -> dict[str, Any]:
    out = {ref: check_table(ctx, ref, depth) for ref in refs}
    bad = {ref: (m.status, failures(m)) for ref, m in out.items() if m.status != "ready"}
    assert not bad, bad
    return out


def declared(table: dict[str, Any]) -> set[str]:
    return set(table["columns"])


def physical_names(schema: list[list[str]]) -> set[str]:
    return {name for name, _ in schema}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- Tahoe: recorded facts


def test_tahoe_de_schema_is_the_declared_one():
    """Every DE shard has one schema (all 1,026 footers); each DE table declares exactly its columns."""
    de = fixture("tahoe", "de_shards.json")
    desc = descriptor("tahoe")
    assert de["shards"] == 1026 and de["schemas_distinct"] == 1
    assert de["rows"] == de["contrasts"] * de["rows_per_contrast"]          # every contrast stores every gene
    schema = dict(de["schema"])
    assert PREPARE_REQUIRED <= set(schema)
    assert schema["concentration"] == "float" and schema["padj"] == "float"   # float32 doses and statistics
    for name in ("de_permissive", "de_significant", "de_high_quality"):
        assert declared(desc["tables"][name]) == set(schema), name


def test_tahoe_metadata_schemas_are_the_declared_ones():
    meta = fixture("tahoe", "metadata.json")
    desc = descriptor("tahoe")
    for name in ("gene", "drug", "cell_line", "sample"):
        assert declared(desc["tables"][f"{name}_metadata"]) == physical_names(meta[name]["schema"]), name
    assert meta["gene"]["rows"] == 62710 and meta["gene"]["facts"]["gene_symbol_unique"]
    assert not any(k.get("verified") is False for k in desc["tables"]["gene_metadata"]["alternate_keys"])
    assert meta["cell_line"]["facts"]["null_depmap_rows"] == 5 and meta["cell_line"]["facts"]["null_effect_rows"] == 25
    targets = meta["drug"]["facts"]
    assert targets["targets_with_pipe"] == 0 and targets["targets_with_comma_space"] > 0
    assert desc["tables"]["drug_metadata"]["columns"]["targets"]["list_delimiter"] == ", "


def test_tahoe_doses_and_plates_render_as_stored():
    """float32 doses render 0.05, 0.5 and 5.0; DE plates '1'..'14' are sample plates 'plate1'..'plate14'."""
    de = fixture("tahoe", "de_shards.json")
    meta = fixture("tahoe", "metadata.json")
    assert sorted(render_float(v, "float") for v in de["concentration_values"]) == ["0.05", "0.5", "5.0"]
    assert de["concentration_values"][0] == 0.05000000074505806 and de["concentration_unit_values"] == ["uM"]
    assert de["concentrations_per_drug"] == {"3": 379}
    alias = descriptor("tahoe")["tables"]["de_permissive"]["columns"]["plate"]["aliases_from"]
    assert {alias["strip_prefix"] + p for p in de["plate_values"]} == set(meta["sample"]["facts"]["plates"])
    several, total = de["drug_concentration_line_on_several_plates"]
    assert 0 < several < total                                   # plate is a replicate: some contrasts repeat


def test_tahoe_drug_spellings_resolve_through_stored_forms():
    """Two DE drugs carry a trailing space; every DE spelling normalises to a metadata drug, and every DE table
    lists its drug column as a stored form."""
    de = fixture("tahoe", "de_shards.json")
    meta = fixture("tahoe", "metadata.json")
    it = descriptor("tahoe")["id_types"]["tahoe_drug"]
    plugin = REGISTRY.get("identifier", "tahoe_drug")
    spaced = sorted(d for d in de["drugs"] if d != d.strip())
    assert spaced == ["Erdafitinib ", "Selinexor "]
    assert {plugin.normalize_stored(d).value for d in de["drugs"]} == set(meta["drug"]["drugs"])
    assert {f"{t}.drug" for t in ("de_permissive", "de_significant", "de_high_quality")} <= set(it["stored_forms"])
    assert set(meta["sample"]["facts"]["drugs_not_in_drug_metadata"]) == {"DMSO_TF", *spaced}
    assert descriptor("tahoe")["tables"]["sample_metadata"]["columns"]["drug"]["integrity"] == "partial"


def test_tahoe_unknown_cell_line_is_a_missing_value():
    """hTERT-HPNE is stored as Cell_ID_DepMap 'NA' in the DE shards: a missing value, never a DepMap ID."""
    de = fixture("tahoe", "de_shards.json")
    desc = descriptor("tahoe")
    assert "NA" in de["cell_lines"] and de["na_cell_line_name"] == ["hTERT-HPNE"]
    assert not isinstance(REGISTRY.get("identifier", "depmap_cell_line").normalize("NA"), Normalized)
    assert desc["tables"]["de_permissive"]["columns"]["Cell_ID_DepMap"]["missing_values"] == ["NA"]
    assert desc["id_types"]["depmap_cell_line"]["universe"] == "cell_line_metadata.Cell_ID_DepMap"
    assert "hTERT-HPNE" in fixture("tahoe", "metadata.json")["cell_line"]["facts"]["names_without_depmap_id"]


def test_tahoe_padj_family_is_the_contrast():
    """padj is BH over each contrast's rows with a padj (13 real contrasts, float32 precision)."""
    sample = fixture("tahoe", "prepared_sample.json")
    padj = descriptor("tahoe")["tables"]["de_permissive"]["columns"]["padj"]
    assert padj["family"] == CONTRAST and "verified" not in padj
    assert len(sample["contrasts"]) == 13
    for c in sample["contrasts"]:
        assert c["rows"] == 62710 and c["dup_genes"] == 0 and c["genes_not_in_metadata"] == 0, c["key"]
        assert c["bh_within_contrast_max_abs_err"] < 1e-6, c["key"]


def test_tahoe_coverage_statement_matches_the_footers():
    de = fixture("tahoe", "de_shards.json")
    share = de["nulls"]["padj"] / de["rows"]
    statement = descriptor("tahoe")["tables"]["de_permissive"]["coverage"]["statement"]
    assert f"{100 * share:.1f}%" in statement and f"{de['nulls']['padj']:,}" in statement


def test_tahoe_prepared_row_groups_are_source_batches():
    """The preparation writes one row group per 65,536-row source batch: 62,586 at most, each spanning about
    two contrasts, so drug pruning reads ~2.5% of them (not the one-drug source row groups)."""
    de = fixture("tahoe", "de_shards.json")
    prep = de["prepared_row_groups"]
    low, high = de["rows_per_shard"]
    per_shard = -(-high // prep["batch_rows"])
    assert per_shard == -(-low // prep["batch_rows"]) and prep["batches"] == de["shards"] * per_shard == 62586
    assert prep["batches_one_drug"] < prep["batches"] // 10
    assert prep["one_drug_scan_row_groups"]["mean_fraction"] < 0.05
    first = de["first_shard"]
    assert (first["row_groups_one_drug"], first["row_groups"]) == (3924, 3987)


def write_prepared(root: Path) -> dict[str, int]:
    """The layout tools/prepare_tahoe.py writes, from the recorded rows and metadata excerpts."""
    sample = fixture("tahoe", "prepared_sample.json")
    meta = fixture("tahoe", "metadata.json")
    de_schema = pa.schema([(n, ARROW[t]) for n, t in sample["schema"]])
    rows = sample["rows"]
    sig = [r for r in rows if r["padj"] < 0.05]
    hq = [r for r in sig if abs(r["log2FoldChange"]) > 0.5]
    (root / "metadata").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=de_schema), root / "tahoe_permissive_padj010.parquet")
    for d, part in (("pseudobulk_de_significant", sig), ("pseudobulk_de_high_quality", hq)):
        (root / d).mkdir()
        pq.write_table(pa.Table.from_pylist(part, schema=de_schema), root / d / "part-00000.parquet")
    for name in ("gene", "drug", "cell_line", "sample"):
        schema = pa.schema([(n, ARROW[t]) for n, t in meta[name]["schema"]])
        pq.write_table(pa.Table.from_pylist(meta[name]["excerpt"], schema=schema),
                       root / "metadata" / f"{name}_metadata.parquet")
    manifest = dict(sample["preparation_manifest"])
    counts = {"source": manifest["rows"]["source"], "permissive": len(rows), "significant": len(sig),
              "high_quality": len(hq)}
    manifest["rows"] = counts
    (root / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return counts


def test_tahoe_prepared_excerpt_is_ready(tmp_path):
    """Real prepared rows (six contrasts: 'Erdafitinib ', 'Selinexor ', Bortezomib and the 'NA' line) in the
    preparation's layout pass every readiness check of the seven Tahoe tables."""
    counts = write_prepared(tmp_path / "tahoe")
    assert counts["high_quality"] > 0
    ctx = make_ctx(tmp_path, {"TAHOE_DATA_PATH": str(tmp_path / "tahoe")})
    tables = ["de_permissive", "de_significant", "de_high_quality", "drug_metadata", "cell_line_metadata",
              "gene_metadata", "sample_metadata"]
    out = checked(ctx, [f"tahoe_100m.{t}" for t in tables])
    perm = out["tahoe_100m.de_permissive"]
    assert any(c.name == "R8" and c.ok and "Bortezomib" in c.detail for c in perm.checks)
    assert any(c.name == "R2:checks.rows" and c.ok for c in perm.checks)
    assert any(c.name == "R9:stored_form" and c.ok for c in out["tahoe_100m.de_high_quality"].checks)
    dangling = [c for c in out["tahoe_100m.sample_metadata"].checks if c.name == "R9" and not c.ok]
    assert dangling and dangling[0].level == "warning" and "DMSO_TF" in dangling[0].detail


# --------------------------------------------------------------------------- DepMap


def test_depmap_files_are_the_declared_release():
    src = fixture("depmap", "source.json")
    desc = descriptor("depmap")
    assert desc["release"]["expect"] == src["release"] == "24Q4"
    paths = {t["path"] for t in desc["tables"].values()}
    assert paths <= {f["name"] for f in src["files"]}
    facts = fixture("depmap", "facts.json")
    model_cols = declared(desc["tables"]["model"])
    assert set(facts["model"]["columns"]) <= model_cols                   # strict: every 24Q4 column declared
    serum = desc["tables"]["model"]["columns"]["SerumFreeMedia"]
    assert set(facts["model"]["SerumFreeMedia"]) == {*serum["encoding"], ""} and "verified" not in serum


def test_depmap_matrix_headers_parse():
    """All 17,916 gene headers of both matrices parse as 'SYMBOL (ENTREZ)'; the row header cell is blank."""
    facts = fixture("depmap", "facts.json")
    gene_effect = descriptor("depmap")["tables"]["gene_effect"]["matrix"]["axes"]
    pattern = re.compile(gene_effect["col"]["parse"]["pattern"])
    for name in ("CRISPRGeneEffect.csv", "CRISPRGeneDependency.csv"):
        f = facts[name]
        assert (f["models"], f["genes"], f["header_unparsed_n"], f["entrez_dupes"]) == (1178, 17916, 0, 0), name
        assert f["first_header_cell"] in gene_effect["row"]["aliases"] and f["model_ids_not_in_Model_csv_n"] == 0
        with open(FIX / "depmap" / name.replace(".csv", ".excerpt.csv"), newline="") as fh:
            header = next(csv.reader(fh))
        assert header[0] == "" and all(pattern.match(h) for h in header[1:]), name
    assert facts["effect_vs_dependency_genes"]["same"]
    assert facts["CRISPRGeneDependency.csv"]["outside_0_1"] == 0


def test_depmap_sentinel_value_is_recorded():
    """The present sentinel (ACH-000001, RPL3 6122) holds in the real 24Q4 value; an empty cell is not measured."""
    sentinel = descriptor("depmap")["tables"]["gene_effect"]["sentinels"]["present"][0]
    with open(FIX / "depmap" / "CRISPRGeneEffect.excerpt.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    col = rows[0].index("RPL3 (6122)")
    value = float(next(r for r in rows if r[0] == sentinel["key"]["ModelID"])[col])
    assert value < sentinel["expect"]["gene_effect"]["lt"]
    assert value == fixture("depmap", "facts.json")["CRISPRGeneEffect.csv"]["sentinel_ACH-000001_RPL3"]["value"]
    assert any(cell == "" for r in rows[1:] for cell in r[1:])           # the excerpt keeps one unmeasured cell


def test_depmap_excerpts_are_ready(tmp_path):
    root = tmp_path / "depmap"
    root.mkdir()
    for name in ("Model", "CRISPRGeneEffect", "CRISPRGeneDependency", "CRISPRInferredCommonEssentials"):
        shutil.copy(FIX / "depmap" / f"{name}.excerpt.csv", root / f"{name}.csv")
    ctx = make_ctx(tmp_path, {"DEPMAP_DATA_PATH": str(root)})
    out = checked(ctx, ["depmap.model", "depmap.gene_effect", "depmap.gene_dependency", "depmap.common_essentials"])
    assert any(c.name == "R8" and c.ok for c in out["depmap.model"].checks)
    assert not [c for c in out["depmap.model"].checks if c.name == "R4:undeclared"]


# --------------------------------------------------------------------------- ontologies


def without_cp_terms(text: str) -> str:
    """cl-basic.obo without its obsolete CP:NNNNNNN stanzas (IDs moved into CL)."""
    return "\n\n".join(b for b in text.split("\n\n") if "\nid: CP:" not in f"\n{b}")


def ontology_ctx(tmp: Path, cl_text: str) -> ServiceContext:
    (tmp / "go").mkdir()
    (tmp / "msigdb").mkdir()
    shutil.copy(FIX / "ontology" / "go-basic.excerpt.obo", tmp / "go" / "go-basic.obo")
    shutil.copy(FIX / "ontology" / "h.all.v2024.1.Hs.symbols.excerpt.gmt", tmp / "msigdb" / "h.all.v2024.1.Hs.symbols.gmt")
    (tmp / "cl-basic.obo").write_text(cl_text)
    return make_ctx(tmp, {"GO_DATA_PATH": str(tmp / "go"), "MSIGDB_DATA_PATH": str(tmp / "msigdb"),
                          "VBT_CL_OBO": str(tmp / "cl-basic.obo")})


def test_ontology_excerpts_are_ready(tmp_path):
    """Whole real stanzas: GO with regulates edges and an obsolete term, CL with CURIE predicates and a
    repeated synonym, and three Hallmark sets."""
    cl = without_cp_terms((FIX / "ontology" / "cl-basic.excerpt.obo").read_text())
    out = checked(ontology_ctx(tmp_path, cl), ["gene_ontology.term", "cell_ontology.term", "msigdb.hallmark"])
    for ref in out:
        assert any(c.name == "R8" and c.ok for c in out[ref].checks), ref


@pytest.mark.xfail(strict=True, reason="contract request: the cell_ontology plugin rejects the 9 obsolete "
                   "CP:NNNNNNN terms of cl-basic.obo (moved into CL), so R4b reports encoding_drift whenever its "
                   "universe sample reaches them (the full file passes only because they come last)")
def test_obsolete_cp_terms_of_cl_are_canonical(tmp_path):
    ctx = ontology_ctx(tmp_path, (FIX / "ontology" / "cl-basic.excerpt.obo").read_text())
    checked(ctx, ["cell_ontology.term"])


def test_ontology_recorded_facts_match_the_descriptors():
    src = fixture("ontology", "source.json")
    cl = descriptor("cell_ontology")["tables"]["term"]["columns"]
    go = descriptor("gene_ontology")["tables"]["term"]["columns"]
    assert set(src["cl"]["relationship_types"]) <= set(cl["relationship"]["fields"]["target"]["predicate"])
    assert set(src["cl"]["relationship_types"]) == set(cl["relationship"]["fields"]["type"]["values"])
    assert cl["synonyms"]["item_key"] == {"identity": "position"}                 # CL:4023064 repeats a synonym
    assert set(src["go"]["relationship_types"]) <= set(go["relationship"]["fields"]["target"]["predicate"])
    hallmark = src["msigdb"]
    members = descriptor("msigdb")["tables"]["hallmark"]["columns"]["members"]["membership"]
    assert hallmark["hallmark_2024"]["sets"] == 50 and not hallmark["login"]
    assert hallmark["hallmark_min_set_resolved_fraction"] >= members["min_resolved_fraction"]


# --------------------------------------------------------------------------- Zenodo


def test_zenodo_unverified_facts_are_resolved():
    """Only the refuted ontology.leaf fact stays unverified (readiness keeps dropping the field)."""
    desc = descriptor("zenodo")
    left = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("verified") is False:
                left.append(path)
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(desc["tables"], "tables")
    assert sorted(left) == ["tables.disease.columns.ontology.fields.leaf", "tables.disease.constraints[0]"]
    assert desc["tables"]["disease"]["constraints"][0]["on_refute"] == "drop_field"


# --------------------------------------------------------------------------- real files (VBT_DL_REAL_DATA)


def real_path(*parts: str) -> Path:
    assert REAL is not None
    p = REAL.joinpath(*parts)
    if not p.exists():
        pytest.skip(f"{p} is not there")
    return p


@needs_real
def test_real_tahoe_metadata_matches_the_record():
    meta = fixture("tahoe", "metadata.json")
    d = real_path("tahoe", REV, "metadata")
    for name in ("gene", "drug", "cell_line", "sample"):
        path = d / f"{name}_metadata.parquet"
        t = pq.read_table(path)
        assert (sha256(path), t.num_rows) == (meta[name]["sha256"], meta[name]["rows"]), name
        assert [[f.name, str(f.type)] for f in t.schema] == meta[name]["schema"], name


@needs_real
def test_real_tahoe_footer_stats_match_the_record():
    de = fixture("tahoe", "de_shards.json")
    lines = real_path("tahoe", REV, "footer_stats", "shards.jsonl").read_text().splitlines()
    shards = [json.loads(line) for line in lines if line.strip()]
    assert len({s["shard"] for s in shards}) == de["shards"]
    assert sum(s["rows"] for s in shards) == de["rows"] and sum(s["row_groups"] for s in shards) == de["row_groups"]
    assert len({s["schema_sha"] for s in shards}) == 1


@needs_real
def test_real_tahoe_prepared_sample_is_ready(tmp_path):
    root = real_path("tahoe", REV, "sample_prepared")
    ctx = make_ctx(tmp_path, {"TAHOE_DATA_PATH": str(root)})
    checked(ctx, [f"tahoe_100m.{t}" for t in ("de_permissive", "de_significant", "de_high_quality", "drug_metadata",
                                               "cell_line_metadata", "gene_metadata", "sample_metadata")])


@needs_real
def test_real_depmap_is_ready(tmp_path):
    """The 24Q4 files (the two 400 MB matrices take about three minutes each)."""
    root = real_path("depmap", "24Q4")
    facts = fixture("depmap", "facts.json")
    with open(root / "CRISPRGeneEffect.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert len(header) - 1 == facts["CRISPRGeneEffect.csv"]["genes"]
    ctx = make_ctx(tmp_path, {"DEPMAP_DATA_PATH": str(root)})
    checked(ctx, ["depmap.model", "depmap.common_essentials", "depmap.gene_effect", "depmap.gene_dependency"])


@needs_real
def test_real_ontologies_are_ready(tmp_path):
    src = fixture("ontology", "source.json")
    go = real_path("gene_ontology", "current")
    cl = real_path("cell_ontology", "current", "cl-basic.obo")
    msig = real_path("msigdb", "2024.1.Hs")
    ctx = make_ctx(tmp_path, {"GO_DATA_PATH": str(go), "MSIGDB_DATA_PATH": str(msig), "VBT_CL_OBO": str(cl)})
    out = checked(ctx, ["gene_ontology.term", "cell_ontology.term", "msigdb.hallmark"])
    if sha256(cl) == src["cl"]["sha256"]:
        assert out["cell_ontology.term"].key_check.detail.startswith(f"{src['cl']['terms']} keys")
    if sha256(go / "go-basic.obo") == src["go"]["sha256"]:
        assert out["gene_ontology.term"].key_check.detail.startswith(f"{src['go']['terms']} keys")


@needs_real
def test_real_zenodo_resolved_facts_hold():
    """ontology.leaf is false for every term; ancestors/descendants are the closures; phase codes; X ranges."""
    ds = pytest.importorskip("pyarrow.dataset")
    assert REAL is not None
    roots = [os.environ.get("VBT_ZENODO_DIR"), REAL.parent / "zenodo", REPO / "data" / "zenodo"]
    found = [Path(r) / "virtualbiotech_submission" for r in roots if r]
    zen = next((z for z in found if (z / "clinical_trials" / "data").is_dir()), None)
    if zen is None:
        pytest.skip(f"no extracted archive under {', '.join(str(z) for z in found)}")
    data = zen / "clinical_trials" / "data"
    rows = ds.dataset(data / "disease", format="parquet").to_table(
        columns=["id", "parents", "children", "ancestors", "ontology"]).to_pylist()
    assert len(rows) == 39530 and not any(r["ontology"]["leaf"] for r in rows)
    assert sum(1 for r in rows if not r["children"]) == 31635
    parents = {r["id"]: r["parents"] or [] for r in rows}
    for r in rows[::97]:
        seen, stack = set(), list(parents[r["id"]])
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(parents.get(x, []))
        assert set(r["ancestors"] or []) == seen and r["id"] not in seen
    phases = ds.dataset(data / "drug_molecule", format="parquet").to_table(
        columns=["maximumClinicalTrialPhase", "isApproved"]).to_pylist()
    values = {p["maximumClinicalTrialPhase"] for p in phases} - {None}
    assert values == {-1.0, 0.5, 1.0, 2.0, 3.0, 4.0}
    assert all((p["maximumClinicalTrialPhase"] == 4) == bool(p["isApproved"]) for p in phases
               if p["maximumClinicalTrialPhase"] is not None and p["isApproved"] is not None)
    anndata = pytest.importorskip("anndata")
    np = pytest.importorskip("numpy")
    for path in sorted((zen / "osmr" / "code" / "data").glob("GSE*.h5ad")):
        a = anndata.read_h5ad(path, backed="r")
        try:
            x = np.asarray(a.X[:], dtype=np.float64)
            assert not np.isnan(x).any() and x.min() >= 0 and x.max() < 20, path.name   # log2 intensities
            assert "X_pca" not in a.obsm
        finally:
            a.file.close()


# --------------------------------------------------------------------------- live (VBT_DL_NETWORK=1)


@needs_network
def test_live_tahoe_footer_matches_the_record():
    from vbt.datalayer.plugins.layouts.http_range import footer_stats

    de = fixture("tahoe", "de_shards.json")
    first = de["first_shard"]
    url = de["_source"]["url"].format(index=0)
    stats = footer_stats(url, size=first["size"])
    assert (stats.rows, stats.row_groups) == (first["rows"], first["row_groups"])
    assert set(stats.columns) == {n for n, _ in de["schema"]}


@needs_network
def test_live_figshare_listing_matches_the_record():
    import httpx

    src = fixture("depmap", "source.json")
    article = httpx.get(f"https://api.figshare.com/v2/articles/{src['figshare_article']}", timeout=60,
                        follow_redirects=True).json()
    files = {f["name"]: (f["size"], f["supplied_md5"]) for f in article["files"]}
    for f in src["files"]:
        assert files[f["name"]] == (f["bytes"], f["md5"]), f["name"]


@needs_network
def test_live_hallmark_needs_no_login():
    import httpx

    src = fixture("ontology", "source.json")["msigdb"]
    r = httpx.get(src["url"], timeout=60, follow_redirects=True)
    assert r.status_code == 200 and hashlib.sha256(r.content).hexdigest() == src["sha256"]

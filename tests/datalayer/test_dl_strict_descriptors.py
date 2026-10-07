"""Strict phase-2 descriptors (docs/DATA_LAYER.md §6, §18 F11).

* ``vbt ds lint --strict`` is clean for every shipped descriptor, and the phase-2 descriptors
  (Open Targets, Tahoe, Zenodo, DepMap, GO, MSigDB, Cell Ontology) declare ``strict: true``.
* The Open Targets tables are exactly the 38 directories of upstream ``src/config/datasets.py``
  ``OPEN_TARGETS_DATASETS`` (parsed with ``ast``, never imported).
* Every physical column and nested field of the OT- and Tahoe-shaped fixtures has exactly one role,
  and every declared non-optional column exists (the fixture schemas are the 25.09 inventory here).
* ``vbt ds check --depth deep`` on the fixtures: every OT, Tahoe and Cell Ontology table is ready,
  with full key checks and no undeclared-column drift; an undeclared column is reported as drift.
* The facts later phases implement are declared and validate now: enrichment contracts with
  per-aspect universes and full families, propagation through id_type hierarchies (Reactome,
  GO through ``gene_ontology`` ``extends``), the literature ``keywordId`` sidecar, the
  interaction-evidence identity and composite reference, DepMap's prefixed ``ncbi_gene``, the MSigDB
  member mapping and the GEO cohort matrix.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import HAVE_ARROW, REPO, upstream_root

pytestmark = pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow and pandas are needed for the data-layer fixtures")

SOURCES_DIR = REPO / "configs" / "data" / "sources"
PHASE2_STRICT = ("open_targets", "tahoe_100m", "zenodo_vbt", "depmap", "gene_ontology", "msigdb", "cell_ontology")


@pytest.fixture(scope="module")
def registry():
    from vbt.datalayer.plugins.registry import discover

    return discover(entry_points=False)


@pytest.fixture(scope="module")
def catalog(registry):
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.load import load_descriptors, load_overlays

    variables = {"project_root": str(REPO)}
    overlays, generic = load_overlays(REPO / "configs" / "data" / "overlays", variables)
    return Catalog(load_descriptors(SOURCES_DIR, variables), overlays, generic, registry=registry)


def _walk(columns: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    from vbt.datalayer.descriptor.columns import is_container

    out: dict[str, Any] = {}
    for name, col in columns.items():
        path = f"{prefix}.{name}" if prefix else name
        out[path] = col
        if is_container(col):
            out.update(_walk(col.fields, path))
    return out


def _physical(schema: Any) -> list[str]:
    import pyarrow as pa

    out: list[str] = []

    def rec(t: Any, prefix: str) -> None:
        while pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            t = t.value_type
        if pa.types.is_struct(t):
            for f in t:
                out.append(f"{prefix}.{f.name}")
                rec(f.type, f"{prefix}.{f.name}")

    for f in schema:
        out.append(f.name)
        rec(f.type, f.name)
    return out


def unroled(spec: Any, schema: Any) -> tuple[list[str], list[str]]:
    """``(physical paths without a role, declared non-optional paths absent from the data)``; a payload or
    ignore column covers everything under it."""
    from vbt.datalayer.descriptor.columns import is_container

    declared = _walk(spec.columns)
    for p in spec.partitions:
        declared[p] = spec.partitions[p].column

    def covered(path: str) -> bool:
        parts = path.split(".")
        for i in range(1, len(parts)):
            col = declared.get(".".join(parts[:i]))
            if col is not None and not is_container(col):
                return True
        return False

    physical = _physical(schema)
    missing = [p for p in physical if p not in declared and not covered(p)]
    absent = [p for p, col in declared.items() if p not in physical and p not in spec.partitions
              and not getattr(col, "optional", False) and not covered(p)
              and not any(getattr(declared.get(".".join(p.split(".")[:i])), "optional", False)
                          for i in range(1, len(p.split("."))))]
    return missing, absent


# ---------------------------------------------------------------------------- lint and inventory


def test_lint_strict_is_clean_and_phase2_descriptors_are_strict(catalog, registry) -> None:
    from vbt.datalayer.descriptor.lint import errors

    findings = catalog.lint(registry, strict=True)
    assert not errors(findings), "\n".join(map(str, errors(findings)))
    for name in PHASE2_STRICT:
        assert catalog.source(name).strict is True, name


def _upstream_datasets() -> tuple[str, ...] | None:
    path = upstream_root() / "src" / "config" / "datasets.py"
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "OPEN_TARGETS_DATASETS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("OPEN_TARGETS_DATASETS not found in upstream src/config/datasets.py")


def test_open_targets_tables_equal_the_upstream_inventory(catalog) -> None:
    import dl_fixtures

    desc = catalog.source("open_targets")
    physical = {spec.path for spec in desc.tables.values() if spec.items_of is None}
    assert physical == set(dl_fixtures.OPEN_TARGETS_DATASETS)
    upstream = _upstream_datasets()
    if upstream is None:
        pytest.skip("the upstream checkout is missing (git submodule update --init)")
    assert len(upstream) == 38 and physical == set(upstream)


def test_every_fixture_column_and_nested_field_is_roled(catalog, tahoe_root) -> None:
    import pyarrow.dataset as ds

    import dl_fixtures

    problems: list[str] = []
    ot = catalog.source("open_targets")
    for name, spec in ot.tables.items():
        if spec.items_of is not None:
            continue
        missing, absent = unroled(spec, dl_fixtures.schema(spec.path))
        problems += [f"open_targets.{name}.{p}: no role" for p in missing]
        problems += [f"open_targets.{name}.{p}: declared, not in the data" for p in absent]
    tahoe = catalog.source("tahoe_100m")
    for name, spec in tahoe.tables.items():
        schema = ds.dataset(str(tahoe_root / spec.path), format="parquet").schema
        missing, absent = unroled(spec, schema)
        problems += [f"tahoe_100m.{name}.{p}: no role" for p in missing]
        problems += [f"tahoe_100m.{name}.{p}: declared, not in the data" for p in absent]
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------- deep readiness on fixtures


@pytest.fixture(scope="module")
def deep_check(ot_root: Path, tahoe_root: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from vbt import cli

    cache = tmp_path_factory.mktemp("strict-deep")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot_root))
        mp.setenv("TAHOE_DATA_PATH", str(tahoe_root))
        mp.setenv("VBT_DATA_DIR", str(cache))
        mp.delenv("VBT_CL_OBO", raising=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert cli.main(["--profile", "mock", "ds", "check", "--depth", "deep", "--json"]) == 0
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def test_deep_check_confirms_keys_and_finds_no_drift(catalog, deep_check) -> None:
    from vbt.datalayer.gateway.readiness import table_status
    from vbt.datalayer.ipc import TableCheckModel

    refs = [f"open_targets.{t}" for t in catalog.source("open_targets").tables] + \
        [f"tahoe_100m.{t}" for t in catalog.source("tahoe_100m").tables] + ["cell_ontology.term"]
    assert not deep_check["table_errors"]
    for ref in refs:
        entry = deep_check["tables"].get(ref)
        assert entry is not None, f"{ref} was not checked"
        status = table_status(TableCheckModel.model_validate(entry))
        errors = [f"{c['name']} {c.get('column') or ''}: {c['detail']}" for c in entry["checks"]
                  if not c["ok"] and c["level"] == "error"]
        assert status == "ready", f"{ref} is {status}: {errors[:5]}"
        names = {c["name"] for c in entry["checks"]}
        assert "R4:undeclared" not in {c["name"] for c in entry["checks"] if not c["ok"]}, ref
        if catalog.table(ref).spec.items_of is None and catalog.table(ref).spec.key.check == "full":
            assert any(n.startswith("R5b") for n in names), f"{ref}: no full key check ran"


def test_an_undeclared_column_is_reported_as_drift(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.service.checks import check_table
    from vbt.datalayer.settings import DataSettings

    (tmp_path / "sources").mkdir()
    (tmp_path / "overlays").mkdir()
    (tmp_path / "data" / "t").mkdir(parents=True)
    pq.write_table(pa.table({"id": ["a", "b"], "v": [1.0, 2.0], "surprise": [1, 2]}), tmp_path / "data/t/part-0.parquet")
    desc = {"schema": "vbt.datasource/1", "source": "s", "title": "s", "root": str(tmp_path / "data"), "strict": True,
            "release": {"from": "literal"}, "defaults": {"format": "parquet", "layout": "sharded_dir"},
            "tables": {"t": {"kind": "fact", "path": "t", "grain": "row", "key": {"columns": ["id"], "check": "full"},
                             "columns": {"id": {"role": "identifier"}, "v": {"role": "measure"}}}}}
    (tmp_path / "sources" / "s.yaml").write_text(yaml.safe_dump(desc))
    settings = DataSettings.from_dict({"descriptors_dir": str(tmp_path / "sources"),
                                       "overlays_dir": str(tmp_path / "overlays"),
                                       "cache_dir": str(tmp_path / "cache")}, project_root=tmp_path)
    model = check_table(ServiceContext(settings), "s.t")
    drift = [c for c in model.checks if c.name == "R4:undeclared" and not c.ok]
    assert drift and "surprise" in drift[0].detail and model.status == "schema_drift"


# ---------------------------------------------------------------------------- later-phase facts


def test_enrichment_contracts_validate(catalog) -> None:
    from vbt.datalayer.predicate import from_json

    target = catalog.source("open_targets").tables["target"]
    for container, universe_table in (("go", "target_go"), ("pathways", "target_pathways")):
        m = target.columns[container].membership
        e = m.enrichment
        assert e is not None and e.test == "hypergeom_enrichment" and e.correction == "fdr_bh", container
        assert e.family.include_zero_overlap and e.family.size_bounds["counted_within"] == "universe"
        assert e.universe.table == universe_table and catalog.table(f"open_targets.{universe_table}").is_item_table
        assert m.propagate_via["relation"] == "ancestor"
    go = target.columns["go"].membership.enrichment.universe
    assert go.per_scope == ["go[].aspect"]                         # an aspect-specific background
    pred = from_json(go.where)
    assert "go[].aspect" in str(pred) and "aspect" in str(pred)
    hallmark = catalog.source("msigdb").tables["hallmark"].columns["members"].membership
    assert hallmark.member.maps_to == "open_targets:ensembl_gene" and hallmark.min_resolved_fraction == 0.95
    assert hallmark.enrichment.universe.table == "open_targets.target"
    from_json(hallmark.enrichment.universe.where)
    msig = catalog.source("msigdb").id_types["msigdb_set"]
    assert msig.plugin == "local_key" and msig.options["canonical"].startswith("^HALLMARK_")


def test_hierarchies_and_propagation(catalog) -> None:
    ot = catalog.source("open_targets")
    reactome = ot.id_types["reactome_pathway"].hierarchy
    assert reactome.table == "reactome" and reactome.columns == ["parents"]
    assert ot.tables["target"].columns["pathways"].membership.propagate_via["id_type"] == "reactome_pathway"
    assert ot.tables["target"].columns["go"].membership.propagate_via["id_type"] == "go_term"
    go = catalog.source("gene_ontology").id_types["go_term"]
    assert go.extends == "open_targets:go_term"
    assert set(go.hierarchy.predicates) == {"is_a", "part_of"} and "regulates" not in go.hierarchy.predicates
    term = catalog.source("gene_ontology").tables["term"]
    assert term.columns["synonyms"].fields["text"].synonym_kind_from == "scope"
    assert go.retired.replaced_by == "term.replaced_by" and go.retired.consider == "term.consider"
    cl = catalog.source("cell_ontology")
    assert cl.id_types["cell_ontology"].hierarchy.predicates == ["is_a"]
    assert cl.tables["term"].path.endswith("tests/fixtures/mini_cl.obo")
    # indirect associations are already propagated over the disease DAG (§11.5); direct ones are not
    for name, spec in ot.tables.items():
        if name.startswith("association_by_"):
            expected = "disease.descendants" if name.endswith("_indirect") else None
            assert spec.columns["diseaseId"].propagated_over == expected, name


def test_literature_sidecar_and_interaction_evidence_identity(catalog) -> None:
    ot = catalog.source("open_targets")
    paths = {(tuple(a.columns), a.via, a.build) for a in ot.tables["literature"].access_paths}
    assert (("keywordId",), "sidecar_index", "on_demand") in paths
    ev = ot.tables["interaction_evidence"]
    assert ev.key.row_identity == "content_hash" and ev.edge is not None
    ref = ev.columns["intA"].ref
    interaction_key = set(ot.tables["interaction"].key.columns)
    assert ref.table == "interaction" and set(ref.on) == interaction_key
    assert ev.columns["intA"].integrity == "partial"


def test_depmap_zenodo_and_item_tables(catalog) -> None:
    depmap = catalog.source("depmap")
    ncbi = depmap.id_types["ncbi_gene"]
    assert ncbi.plugin == "ncbi_gene" and ncbi.options == {"input_requires_prefix": True}
    effect = depmap.tables["gene_effect"]
    assert effect.matrix.axes["col"].from_ == "header" and effect.matrix.axes["col"].parse is not None
    assert effect.matrix.values["gene_effect"].cutoff.value == -0.5
    assert {"gene_effect", "gene_dependency", "model", "common_essentials"} <= set(depmap.tables)
    zen = catalog.source("zenodo_vbt")
    cohorts = zen.tables["ibd_cohorts"]
    assert cohorts.fragment_key.name == "cohort" and set(cohorts.fragment_overrides) == {"GSE73661", "GSE23597"}
    assert zen.id_types["geo_gsm"].plugin == "geo_gsm"
    for name in ("disease", "drug_molecule", "association_by_datatype_direct"):
        assert zen.tables[name].implements.startswith(f"open_targets:{name}@") and zen.tables[name].lineage
    assert zen.tables["clinical_trial_labels"].format == "csv"
    tahoe = catalog.source("tahoe_100m")
    assert {"de_permissive", "de_significant", "de_high_quality", "drug_metadata", "cell_line_metadata",
            "gene_metadata", "sample_metadata"} == set(tahoe.tables)
    for item in ("target_safety_liabilities", "credible_set_locus", "l2g_features", "pharmacogenomics_drugs",
                 "mouse_phenotype_classes", "evidence_mutated_samples"):
        t = catalog.table(f"open_targets.{item}")
        assert t.is_item_table and len(t.key) >= 2, item

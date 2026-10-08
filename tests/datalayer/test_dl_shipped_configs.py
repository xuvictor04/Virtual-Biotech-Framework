"""The shipped phase-1 descriptors and overlays (configs/data; docs/DATA_LAYER.md §6, §8, Appendix A).

* Every descriptor and overlay loads and lints without errors against the plugin registry, and
  the catalog builds.
* Every tool the upstream servers register (``register_tool`` calls in
  ``src/mcp_servers/*/server.py``, provenance excluded) and both PubMed tools have a reviewed
  binding; the serve modes are Appendix A's phase-1 split moved by phase 2 (F14): the comprehensive
  target profile is a derived view and ``compare_direct_indirect`` a derived set comparison, so
  64 pass, 29 derived and 10 block.
* Every identifier-like parameter is bound with ``accepts`` (or is an anchor).
* Every defect names an existing ``file:line`` and, when it names a detector, an existing test.
* The fixture schemas of ``dl_fixtures`` conform to the descriptors (declared columns present with
  compatible types) and the descriptor sentinels hold on the fixtures.
* Spot checks of the phase-1 facts the plan requires (keys, partitions, scope, leakage, bindings).
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import HAVE_ARROW, PUBMED_SERVER, REPO, upstream_root

pytestmark = pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow and pandas are needed for the data-layer fixtures")

SOURCES_DIR = REPO / "configs" / "data" / "sources"
OVERLAYS_DIR = REPO / "configs" / "data" / "overlays"
SERVERS_DIR = upstream_root() / "src" / "mcp_servers"
DETECTORS = Path(__file__).resolve().parent / "test_dl_defect_detectors.py"

SOURCES = {"open_targets", "tahoe_100m", "zenodo_vbt", "cellxgene_census", "clinicaltrials_gov", "cbioportal", "pubmed",
           "depmap", "gene_ontology", "msigdb", "cell_ontology"}
#: Phase 2 makes these descriptors strict (§6): undeclared columns are drift, not passthrough.
STRICT = {"open_targets", "tahoe_100m", "zenodo_vbt", "depmap", "gene_ontology", "msigdb", "cell_ontology"}
SERVERS = {"target", "disease", "drug", "association", "genetics", "expression", "interaction", "functional_genomics",
           "pathway", "single_cell", "clinicaltrials", "pubmed", "data"}
HARNESS_SERVERS = {"pubmed", "data"}             # overlays for harness servers, not upstream code
# Appendix A, per server: (pass, derived, block); same_as tools count on the server that registers them.
# Phase 2 (F14) moved one target and one association tool from block to derived.
APPENDIX_A = {
    "target": (11, 5, 0), "disease": (2, 4, 0), "drug": (5, 4, 0), "association": (7, 4, 0), "genetics": (8, 2, 0),
    "expression": (3, 3, 0), "interaction": (1, 4, 0), "functional_genomics": (0, 9, 0), "pathway": (3, 7, 0),
    "single_cell": (11, 0, 0), "clinicaltrials": (7, 0, 1), "pubmed": (2, 0, 0),
}
IDENTIFIER_PARAM = re.compile(r"(_ids?$|^gene|^pmid|^nct|^rs_id$|^variant_id$|^drug_name$|^entity_)")
ITEM_TABLES = ("target_go", "target_pathways", "target_tractability", "target_homologues", "target_chemical_probes",
               "expression_tissues", "target_essentiality_screens", "drug_indications", "disease_phenotype_evidence")


# ---------------------------------------------------------------------------- loading


@pytest.fixture(scope="module")
def registry():
    from vbt.datalayer.plugins.registry import discover

    return discover()


@pytest.fixture(scope="module")
def catalog(registry):
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.load import load_descriptors, load_overlays

    variables = {"project_root": str(REPO)}
    descriptors = load_descriptors(SOURCES_DIR, variables)
    overlays, generic = load_overlays(OVERLAYS_DIR, variables)
    return Catalog(descriptors, overlays, generic, registry=registry)


def _registered_tools() -> dict[tuple[str, str], tuple[str, ...]]:
    """``{(server, tool): parameter names}`` of every bridged tool, parsed (never imported)."""
    out: dict[tuple[str, str], tuple[str, ...]] = {}
    for server_dir in sorted(p for p in SERVERS_DIR.iterdir() if (p / "server.py").is_file()):
        if server_dir.name == "provenance_mcp":
            continue
        tree = ast.parse((server_dir / "server.py").read_text(encoding="utf-8"))
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.mcp_servers"):
                for alias in node.names:
                    imports[alias.asname or alias.name] = (node.module, alias.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_tool":
                module, name = imports[node.args[1].id]    # type: ignore[attr-defined]
                path = upstream_root() / (module.replace(".", "/") + ".py")
                fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
                out[(server_dir.name.removesuffix("_mcp"), name)] = tuple(a.arg for a in fn.args.args)
    tree = ast.parse(PUBMED_SERVER.read_text(encoding="utf-8"))
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "tool" for d in fn.decorator_list):
            out[("pubmed", fn.name)] = tuple(a.arg for a in fn.args.args)
    return out


TOOLS = _registered_tools() if SERVERS_DIR.is_dir() else {}
needs_upstream = pytest.mark.skipif(not TOOLS, reason="the upstream checkout is missing (git submodule update --init)")


def test_files_load_and_lint_without_errors(catalog, registry) -> None:
    from vbt.datalayer.descriptor.lint import errors

    assert set(catalog.sources) == SOURCES
    assert set(catalog.overlays) == SERVERS
    assert [g.server for g in catalog.generic] == ["*"]
    problems = errors(catalog.lint(registry))
    assert not problems, "\n".join(map(str, problems))


def test_load_catalog_from_the_default_config() -> None:
    """The runtime path: ``load_catalog`` on the shipped config finds the shipped files."""
    from vbt.config import load_config
    from vbt.datalayer.catalog import load_catalog

    cat = load_catalog(load_config())
    assert set(cat.sources) == SOURCES and set(cat.overlays) == SERVERS


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_overlays_name_the_recorded_upstream_commit(catalog) -> None:
    """Upstream overlays were reviewed against the commit the superproject records for the submodule;
    a submodule bump must come with a re-review (and an updated default here)."""
    out = subprocess.run(["git", "ls-tree", "HEAD", "third_party/TheVirtualBiotech"], cwd=REPO, capture_output=True,
                         text=True).stdout.split()
    if len(out) < 3:
        pytest.skip("the superproject records no submodule commit")
    for server, ov in catalog.overlays.items():
        if server not in HARNESS_SERVERS:              # the harness servers are not upstream code
            assert ov.upstream_commit == out[2], server


def test_descriptor_strictness_and_statistics(catalog) -> None:
    for name, desc in catalog.sources.items():
        assert desc.strict is (name in STRICT), name
        for table, spec in desc.tables.items():
            for col in _all_columns(spec.columns):
                if getattr(col, "role", None) == "measure":
                    assert catalog.registry.has("statistic", col.statistic) or col.fallback == "numeric", \
                        f"{name}.{table}: statistic {col.statistic!r} without fallback: numeric"


def test_every_table_resolves_in_the_catalog(catalog) -> None:
    for ref in catalog.table_refs():
        t = catalog.table(ref)
        assert t.key, f"{ref} has no complete key"
    for name in ITEM_TABLES:
        t = catalog.table(f"open_targets.{name}")
        assert t.is_item_table and len(t.key) >= 2, name


# ---------------------------------------------------------------------------- bindings


@needs_upstream
def test_every_bridged_tool_has_a_reviewed_binding(catalog) -> None:
    assert len(TOOLS) == 103
    missing = [f"{s}.{t}" for (s, t) in TOOLS
               if (c := catalog.contract(s, t)).binding is None or c.generic or c.binding.status != "reviewed"]
    assert not missing, f"tools without a reviewed binding: {missing}"
    upstream = {s: ov for s, ov in catalog.overlays.items() if s != "data"}   # the data child is not bridged
    bound = {(s, t) for s, ov in upstream.items() for t in ov.tools}
    bound |= {tuple(a.split(".", 1)) for ov in upstream.values() for b in ov.tools.values() for a in b.same_as}
    stale = sorted(f"{s}.{t}" for s, t in bound - set(TOOLS))
    assert not stale, f"bindings for tools upstream does not register: {stale}"


@needs_upstream
def test_serve_modes_match_appendix_a(catalog) -> None:
    total: Counter[str] = Counter()
    by_server: dict[str, Counter[str]] = {}
    for server, tool in TOOLS:
        mode = catalog.contract(server, tool).binding.serve
        total[mode] += 1
        by_server.setdefault(server, Counter())[mode] += 1
    assert dict(total) == {"pass": 60, "derived": 42, "block": 1}
    for server, (p, d, b) in APPENDIX_A.items():
        got = by_server[server]
        assert (got["pass"], got["derived"], got["block"]) == (p, d, b), server


@needs_upstream
def test_blocked_tools_name_alternatives(catalog) -> None:
    for server, tool in TOOLS:
        b = catalog.contract(server, tool).binding
        if b.serve != "block":
            continue
        hidden = b.hidden or b.block.hidden
        if (server, tool) == ("clinicaltrials", "clear_trial_cache"):
            assert hidden
            continue
        assert not hidden and b.block.alternatives and b.block.until_phase in (2, 3, 4), f"{server}.{tool}"
        for alt in b.block.alternatives:
            s, _, t = alt.partition(".")
            assert (s, t) in TOOLS, f"{server}.{tool}: alternative {alt} is not a bridged tool"
            assert catalog.contract(s, t).binding.serve != "block", f"{server}.{tool}: alternative {alt} is blocked"


@needs_upstream
def test_identifier_parameters_are_resolved(catalog) -> None:
    unbound = []
    for (server, tool), params in TOOLS.items():
        args = catalog.contract(server, tool).binding.args
        for p in params:
            if not IDENTIFIER_PARAM.search(p):
                continue
            a = args.get(p)
            # role unbound is refused at the gateway whenever it is set, so it is never sent unresolved
            if a is None or not (a.accepts or a.role in ("anchor", "unbound")):
                unbound.append(f"{server}.{tool}({p})")
    assert not unbound, f"identifier-like parameters without accepts, anchor or unbound: {unbound}"


@needs_upstream
def test_bound_arguments_exist_upstream(catalog) -> None:
    """A binding never invents an argument: every bound name is an upstream parameter, unless it is a
    gateway-only (x-gateway) argument."""
    extra = []
    for (server, tool), params in TOOLS.items():
        for name, a in catalog.contract(server, tool).binding.args.items():
            if name not in params and not a.gateway_only:
                extra.append(f"{server}.{tool}({name})")
    assert not extra, extra


def _defects(catalog) -> list[tuple[str, str, Any]]:
    return [(server, tool, d) for server, ov in catalog.overlays.items() for tool, b in ov.tools.items()
            for d in b.defects]


def _where_path(where: str) -> tuple[Path, list[int]]:
    file, _, lines = where.rpartition(":")
    base = REPO if file.startswith("src/vbt/") else SERVERS_DIR
    numbers = [int(n) for part in lines.split(",") for n in part.split("-")]
    return base / file, numbers


@needs_upstream
def test_defects_point_at_existing_upstream_lines(catalog) -> None:
    ids: Counter[str] = Counter()
    for server, tool, d in _defects(catalog):
        ids[d.id] += 1
        path, numbers = _where_path(d.where)
        assert path.is_file(), f"{server}.{tool} {d.id}: {d.where} does not exist"
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        assert all(1 <= n <= n_lines for n in numbers), f"{d.id}: {d.where} is past the end ({n_lines} lines)"
        for part in d.where.rpartition(":")[2].split(","):
            lo, _, hi = part.partition("-")
            assert not hi or int(lo) <= int(hi), f"{d.id}: bad range {part}"
        assert d.what and d.effect in ("silent_wrong", "false_empty", "error", "oom"), d.id
    assert not [i for i, n in ids.items() if n > 1], "defect ids are unique"


def _test_names(path: Path) -> set[str]:
    return {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


#: Files whose tests may serve as a defect's detector: the detectors, and the correctness pins of today.
DETECTOR_FILES = (DETECTORS, DETECTORS.parent / "test_dl_correctness_six.py")

#: §11.6: ``order_source: upstream_full_sort`` skips the ranking refusal only when a detector proves it.
FULL_SORT_PROOFS = {
    "association.query_associations": "test_full_sort_associations",
    "association.get_associations_for_disease": "test_full_sort_associations",
    "association.get_associations_for_target": "test_full_sort_associations",
    "association.filter_by_datatype": "test_full_sort_filters",
    "association.filter_by_datasource": "test_full_sort_filters",
    "association.find_similar_entities": "test_full_sort_similar",
    "genetics.query_gwas_associations": "test_full_sort_genetics",
    "genetics.get_credible_sets": "test_full_sort_genetics",
}


def test_defect_detectors_exist(catalog) -> None:
    names = {f.name: _test_names(f) for f in DETECTOR_FILES}
    named = 0
    for server, tool, d in _defects(catalog):
        if d.test is None:
            continue
        file, _, func = d.test.partition("::")
        assert func in names.get(file, set()), f"{server}.{tool} {d.id}: detector {d.test} not found"
        named += 1
    assert named >= 14


def test_remote_defects_have_detectors(catalog) -> None:
    """Until phase 4 every remote tool's known miscount carries a detector test (§11.6)."""
    missing = [f"{s}.{t} {d.id}" for s, t, d in _defects(catalog)
               if s in ("clinicaltrials", "pubmed", "single_cell") and d.test is None]
    assert not missing, missing


def test_upstream_full_sort_is_proven(catalog) -> None:
    detectors = _test_names(DETECTORS)
    declared = sorted(f"{s}.{t}" for s in catalog.servers() for t in catalog.tools(s)
                      if _binding(catalog, f"{s}.{t}").result.order_source == "upstream_full_sort")
    assert declared, "no upstream_full_sort binding"
    for name in declared:
        assert FULL_SORT_PROOFS.get(name) in detectors, f"{name}: upstream_full_sort without a detector proving it"


# ---------------------------------------------------------------------------- fixtures vs descriptors


def _all_columns(columns: dict[str, Any]):
    for col in columns.values():
        yield col
        yield from _all_columns(getattr(col, "fields", {}) or {})


def _type_text(typ: Any) -> str:
    """Arrow type as ``arrow_compatible`` reads it (field names and nullability dropped)."""
    import pyarrow as pa

    if pa.types.is_list(typ) or pa.types.is_large_list(typ):
        return f"list<{_type_text(typ.value_type)}>"
    if pa.types.is_struct(typ):
        return "struct<" + ", ".join(f"{f.name}: {_type_text(f.type)}" for f in typ) + ">"
    return str(typ)


def _conform(columns: dict[str, Any], fields: dict[str, Any], where: str) -> list[str]:
    import pyarrow as pa

    from vbt.datalayer.roles import arrow_compatible

    problems = []
    for name, col in columns.items():
        if name not in fields:
            if not getattr(col, "optional", False):
                problems.append(f"{where}.{name}: declared but absent from the fixture schema")
            continue
        typ = fields[name]
        role = col.role
        if role in ("nested", "member") and col.fields:
            inner = typ
            while pa.types.is_list(inner) or pa.types.is_large_list(inner):
                inner = inner.value_type
            if not pa.types.is_struct(inner):
                problems.append(f"{where}.{name}: {role} with fields on {typ}")
                continue
            problems.extend(_conform(col.fields, {f.name: f.type for f in inner}, f"{where}.{name}"))
            continue
        ok = arrow_compatible(role, _type_text(typ), parse=getattr(col, "parse", None),
                              stored_as=getattr(col, "stored_as", None), encoding=getattr(col, "encoding", None),
                              list_delimiter=getattr(col, "list_delimiter", None))
        if not ok:
            problems.append(f"{where}.{name}: role {role} cannot be stored as {typ}")
    return problems


def test_open_targets_fixture_schemas_conform(catalog) -> None:
    import dl_fixtures as F

    ot = catalog.source("open_targets")
    checked = []
    problems: list[str] = []
    for name in F.SCHEMAS:
        spec = ot.tables.get(name)
        assert spec is not None, f"fixture table {name} has no descriptor table"
        fields = {f.name: f.type for f in F.schema(name)}
        problems.extend(_conform(spec.columns, fields, name))
        for part in spec.partitions:
            assert part not in fields, f"{name}: partition column {part} is not stored in the files"
        checked.append(name)
    assert not problems, "\n".join(problems)
    assert len(checked) >= 30


def test_tahoe_fixture_schemas_conform(catalog, tahoe_root) -> None:
    import pyarrow.dataset as ds

    tahoe = catalog.source("tahoe_100m")
    problems: list[str] = []
    for name, spec in tahoe.tables.items():
        schema = ds.dataset(tahoe_root / spec.path).schema      # a file, or a sharded_dir directory
        problems.extend(_conform(spec.columns, {f.name: f.type for f in schema}, name))
    assert not problems, "\n".join(problems)


def _matches(row: dict[str, Any], key: dict[str, Any]) -> bool:
    return all(row.get(c) == v for c, v in key.items())


def _expect_holds(row: dict[str, Any], expect: dict[str, Any]) -> bool:
    for k, v in expect.items():
        if k == "nonempty":
            if not all(row.get(c) for c in v):
                return False
        elif k not in ("min_rows", "contains", "items", "is_null", "is_empty") and not isinstance(v, dict):
            if row.get(k) != v:
                return False
    return True


def test_sentinels_hold_on_the_fixtures(catalog, ot_root, tahoe_root) -> None:
    import dl_fixtures as F
    import pyarrow.parquet as pq

    assert F.check_descriptor_sentinels(ot_root) == []
    checked = 0
    for source, rows_of in (("open_targets", lambda t: F.read_rows(ot_root, t)),
                            ("tahoe_100m", lambda t: pq.read_table(tahoe_root / catalog.source("tahoe_100m").tables[t].path
                                                                   ).to_pylist())):
        desc = catalog.source(source)
        for name, spec in desc.tables.items():
            if spec.sentinels is None or spec.items_of is not None:
                continue
            rows = rows_of(name)
            columns = set(rows[0]) if rows else set()
            for s in spec.sentinels.present:
                assert set(s.key) <= columns | set(spec.partitions), f"{source}.{name}: sentinel key {s.key}"
                hits = [r for r in rows if _matches(r, s.key)]
                assert hits, f"{source}.{name}: present sentinel {s.key} not in the fixture"
                assert len(hits) >= s.expect.get("min_rows", 1)
                assert any(_expect_holds(r, s.expect) for r in hits), f"{source}.{name}: {s.expect}"
                checked += 1
            for s in spec.sentinels.absent:
                assert not [r for r in rows if _matches(r, s.key)], f"{source}.{name}: absent sentinel {s.key} found"
    assert checked >= 6


# ---------------------------------------------------------------------------- phase-1 facts


def _table(catalog, ref: str):
    src, _, tab = ref.partition(".")
    return catalog.source(src).tables[tab]


def _binding(catalog, name: str):
    server, _, tool = name.partition(".")
    return catalog.contract(server, tool).binding


def test_keys_partitions_and_sizes(catalog) -> None:
    kd = _table(catalog, "open_targets.known_drug")
    assert kd.key.columns == ["drugId", "targetId", "diseaseId", "phase", "status"] and kd.key.nullable == ["phase", "status"]
    assert _table(catalog, "open_targets.interaction").key.nullable == ["targetB"]
    pgx = _table(catalog, "open_targets.pharmacogenomics").key
    assert {"genotypeId", "haplotypeId"} <= set(pgx.nullable) and pgx.row_identity == "content_hash"
    ev = _table(catalog, "open_targets.evidence")
    part = ev.partitions["sourceId"]
    assert part.expect == "declared" and part.mirrored_by == ["datasourceId"] and len(part.column.vocab) == 23
    assert {a.via for a in ev.access_paths} == {"partition", "sidecar_index"}
    for name in ("evidence", "literature", "literature_vector", "variant", "interaction_evidence"):
        assert _table(catalog, f"open_targets.{name}").size_class == "huge", name
    assert _table(catalog, "open_targets.literature_vector").evidence_nature.kind == "literature_cooccurrence"
    leaf = [c for c in _table(catalog, "open_targets.disease").constraints if c.column == "ontology.leaf"]
    assert leaf and leaf[0].on_refute == "drop_field" and leaf[0].verified is False
    hse = _table(catalog, "open_targets.target_prioritisation").columns["hasSafetyEvent"]
    # 25.09 stores -1 or null, never 0 (R1): the encoding is the observed code
    assert hse.encoding == {-1: "known_unfavourable"} and hse.scale == [-1, 0] and hse.verified


def test_tahoe_and_census_facts(catalog) -> None:
    de = _table(catalog, "tahoe_100m.de_permissive")
    assert de.columns["concentration"].scope.pooling == "forbid"
    assert de.columns["plate"].scope.pooling == "list" and de.columns["plate"].aliases_from.table == "sample_metadata"
    assert de.coverage.absence_means == "censored" and de.coverage.censor.column == "padj"
    assert {(a.columns[0], a.via) for a in de.access_paths} == {("drug", "row_group_stats"), ("gene_name", "sidecar_index")}
    tahoe = catalog.source("tahoe_100m")
    assert tahoe.manifests[0].required and tahoe.release.from_ == ["manifest.source_revision", "manifest.filters"]
    assert "de_permissive.drug" in tahoe.id_types["tahoe_drug"].stored_forms
    anndata = _table(catalog, "cellxgene_census.anndata_outputs")
    assert anndata.materialized_by.tool == "single_cell.get_anndata" and anndata.matrix is not None
    assert catalog.source("cellxgene_census").table_layout("obs") == "soma"


def test_zenodo_cbioportal_and_clinicaltrials_facts(catalog) -> None:
    zen = catalog.source("zenodo_vbt")
    for name in ("disease", "drug_molecule", "association_by_datatype_direct"):
        assert zen.tables[name].implements == f"open_targets:{name}@25.09", name
    labels = zen.tables["clinical_trial_labels"]
    assert labels.expose.withhold_from == ["trial-annotator", "clinical-trialist"]
    assert labels.columns["pubmed_ids"].list_delimiter == "|"
    ct = catalog.source("clinicaltrials_gov")
    assert ct.leakage.available_at.endswith("studyFirstPostDateStruct.date") and ct.leakage.counts == "inject_filter"
    assert "version" in ct.tables
    cbio = catalog.source("cbioportal")
    assert cbio.id_types["cbio_study"].universe_via.tool == "clinicaltrials.search_studies"
    assert cbio.id_types["cbio_sample"].universe.keys == ["studyId", "sampleId"]
    for name in ("patient_clinical", "sample_clinical"):
        assert cbio.tables[name].pivot is not None and cbio.tables[name].roles_from is not None


def test_binding_facts(catalog) -> None:
    b = _binding(catalog, "pubmed.fetch_abstracts")
    assert b.args["pmids"].max_items == 50 and b.result.echo_set.mode == "subset"
    b = _binding(catalog, "clinicaltrials.get_clinical_data")
    assert b.result.exists_when and b.args["sample_ids"].min_items == 1 and b.result.levels["patient"]
    b = _binding(catalog, "target.prioritize_targets")
    # "no safety event" cannot be told from "not assessed" in 25.09: the flag is refused
    assert b.args["no_safety_events"].role == "unbound" and b.args["no_safety_events"].when_true is None
    assert b.args["min_clinical_phase"].max == 1                     # maxClinicalTrialPhase is 0-1 (0.25 per phase)
    assert b.args["sort_by"].role == "order_by" and b.result.order_from_arg == "sort_by"
    b = _binding(catalog, "association.find_similar_entities")
    assert b.args["entity_id"].role == "anchor" and b.result.order_source == "upstream_full_sort"
    b = _binding(catalog, "clinicaltrials.search_clinical_trials")
    assert b.result.order_source == "source_server_side" and b.leakage_filter.arg == "advanced_filter"
    assert b.result.total.partial_when == ["$.warning"]
    b = _binding(catalog, "functional_genomics.query_drug_perturbation")
    assert b.args["concentration"].gateway_only and b.args["plate"].gateway_only and b.args["drug_name"].send_as == "stored"
    assert _binding(catalog, "functional_genomics.find_drugs_affecting_gene").args["top_n"].limit_grain == "drug"
    b = _binding(catalog, "interaction.get_interactions")
    assert len(b.args["target_id"].binds_any) == 2 and len(b.args["species"].binds_any) == 2
    b = _binding(catalog, "disease.find_diseases_by_phenotype")
    assert b.args["evidence_type"].item_filter and b.args["evidence_type"].drop_empty_parents
    assert _binding(catalog, "pathway.get_gene_ontology").args["aspect"].send_map
    assert _binding(catalog, "association.filter_by_datatype").args["include_indirect"].role == "selector"
    assert _binding(catalog, "genetics.query_colocalisation").args["method"].role == "selector"
    assert _binding(catalog, "association.query_evidence").args["require_pubmed"].op == "nonempty"
    assert _binding(catalog, "disease.get_disease_hierarchy").result.fields["parents[].name"].on_placeholder == \
        "dangling_ref"
    assert _binding(catalog, "drug.get_pharmacogenomics").same_as == ["target.get_pharmacogenomics"]
    assert _binding(catalog, "target.get_target_tractability").same_as == ["drug.get_target_tractability"]
    assert catalog.contract("target", "get_pharmacogenomics").alias_of == "drug.get_pharmacogenomics"

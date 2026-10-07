"""Defect detectors (docs/DATA_LAYER.md §8.3, §21): one test per overlay ``defects`` entry with a ``test``.

Each detector imports the unmodified upstream function in a child interpreter (``python -B`` with
``PYTHONDONTWRITEBYTECODE=1``, as ``preflight.upstream_doctor`` does, so nothing is written under
``third_party/``), runs it on the ``dl_fixtures`` data and asserts that the documented bug still
reproduces. A detector that starts failing means upstream changed: revisit the overlay entry and the
``serve`` mode it justifies, then update or drop the defect.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import HAVE_ARROW, upstream_missing, upstream_root
from stubs import STUBS_DIR

pytestmark = [
    pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow and pandas are needed for the fixtures"),
    pytest.mark.skipif(upstream_missing() is not None, reason=str(upstream_missing())),
]

MARK = "<<<VBT-DETECTOR>>>"
CHILD = """
import json, sys
paths, module, func, kwargs = json.loads(sys.argv[1])
for p in reversed(paths):
    sys.path.insert(0, p)
import importlib
out = getattr(importlib.import_module(module), func)(**kwargs)
print({mark!r} + json.dumps(out, default=str))
""".format(mark=MARK)


def call_upstream(module: str, func: str, tmp: Path, *, ot_root: Path | None = None, tahoe_root: Path | None = None,
                  extra_paths: tuple[str, ...] = (), **kwargs: Any) -> dict[str, Any]:
    """Run ``module.func(**kwargs)`` of the unmodified upstream checkout and return its JSON result."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", PRELOAD_MCP_DATA="0",
               MCP_OUTPUT_DIR=str(tmp / "mcp_output"))
    if ot_root is not None:
        env["OPEN_TARGETS_DATA_PATH"] = str(ot_root)
    if tahoe_root is not None:
        env["TAHOE_DATA_PATH"] = str(tahoe_root)
    payload = json.dumps([[*extra_paths, str(upstream_root())], module, func, kwargs])
    proc = subprocess.run([sys.executable, "-B", "-c", CHILD, payload], env=env, cwd=str(tmp), capture_output=True,
                          text=True, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(MARK)]
    assert proc.returncode == 0 and lines, f"{module}.{func} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    return json.loads(lines[-1][len(MARK):])


def ot_call(ot_root: Path, tmp: Path, server: str, func: str, **kwargs: Any) -> dict[str, Any]:
    return call_upstream(f"src.mcp_servers.{server}_mcp.tools", func, tmp, ot_root=ot_root, **kwargs)


@pytest.fixture
def ot_copy(ot_root: Path, tmp_path: Path):
    """A private copy of the OT fixture a detector may add rows to."""
    import dl_fixtures as F

    def make(**tables: list[dict[str, Any]]) -> Path:
        root = F.copy_fixture(ot_root, tmp_path / "ot" / "25.09")
        for name, rows in tables.items():
            F.delete_table(root, name)
            F.write_table(root, name, F.table(name, rows))
        return root

    return make


def keys(rows: list[dict[str, Any]], cols: tuple[str, ...]) -> list[tuple]:
    return [tuple(r.get(c) for c in cols) for r in rows]


# ---------------------------------------------------------------------------- target


def test_ot_target_001(ot_root: Path, tmp_path: Path) -> None:
    """search_targets_by_name: regex substring in file order answers 'TP53' with TP53BP1."""
    import dl_fixtures as F

    out = ot_call(ot_root, tmp_path, "target", "search_targets_by_name", query="TP53", limit=1)
    assert [r["id"] for r in out["results"]] == [F.TP53BP1]


def test_ot_target_005(ot_root: Path, tmp_path: Path) -> None:
    """prioritize_targets(no_safety_events=True) keeps -1 (event recorded) and null/NaN (not assessed)."""
    import dl_fixtures as F

    out = ot_call(ot_root, tmp_path, "target", "prioritize_targets", no_safety_events=True, limit=100)
    returned = {r["targetId"] for r in out["targets"]}
    oracle = F.oracle_no_safety_events(ot_root)
    assert set(oracle["ids"]) == {F.PRIO["C"]}
    assert returned == set(F.PRIO.values())                      # A (-1), B (null), D (NaN) included
    assert returned - set(oracle["ids"]) == {F.PRIO["A"], F.PRIO["B"], F.PRIO["D"]}


def test_ot_target_007(ot_root: Path, tmp_path: Path) -> None:
    """get_mouse_phenotype matches the human ID against the mouse-orthologue column: always empty."""
    import dl_fixtures as F

    assert len(F.oracle_mouse(ot_root, F.PCSK9)) == 2
    out = ot_call(ot_root, tmp_path, "target", "get_mouse_phenotype", target_id=F.PCSK9)
    assert out["count"] == 0 and out["phenotypes"] == []


# ---------------------------------------------------------------------------- drug


def test_ot_drug_001(ot_root: Path, tmp_path: Path) -> None:
    """search_drugs(target_id=...) is always empty: linkedTargets is a struct, not a list."""
    import dl_fixtures as F

    assert F.oracle_drugs_for_target(ot_root, F.PCSK9) == [F.CHEMBL1000]
    out = ot_call(ot_root, tmp_path, "drug", "search_drugs", target_id=F.PCSK9)
    assert out["count"] == 0


def test_ot_drug_003(ot_root: Path, tmp_path: Path) -> None:
    """search_known_drugs cuts head(limit) in file order before sorting by phase."""
    import dl_fixtures as F

    out = ot_call(ot_root, tmp_path, "drug", "search_known_drugs", target_id=F.T, limit=5)
    got = keys(out["drugs"], F.KNOWN_DRUG_KEY)
    oracle = F.oracle_known_drug_topk(ot_root, F.T, 5)
    assert got != oracle
    assert all(r["phase"] < 4 for r in out["drugs"]) and all(k[3] == 4.0 for k in oracle)


def test_ot_drug_004(ot_root: Path, tmp_path: Path) -> None:
    """Rows with an unknown phase are dropped even at min_phase=0 (and not counted anywhere)."""
    import dl_fixtures as F

    oracle = F.oracle_known_drug(ot_root, F.T)
    assert oracle["unknown_phase"] == 2
    out = ot_call(ot_root, tmp_path, "drug", "search_known_drugs", target_id=F.T, min_phase=0, limit=1000)
    assert out["count"] == oracle["total"]
    assert not {"CHEMBL6000", "CHEMBL6001"} & {r["drugId"] for r in out["drugs"]}


def test_ot_drug_006(ot_root: Path, tmp_path: Path) -> None:
    """get_drug_adverse_events cuts head(limit) in file order before sorting by llr."""
    import dl_fixtures as F

    out = ot_call(ot_root, tmp_path, "drug", "get_drug_adverse_events", drug_id=F.CHEMBL559288, limit=1)
    got = keys(out["adverse_events"], ("chembl_id", "meddraCode"))
    assert got != F.oracle_adverse_topk(ot_root, F.CHEMBL559288, 1)
    assert out["adverse_events"][0]["llr"] < 99.5


@pytest.mark.parametrize("server", ["drug", "target"])
def test_ot_drug_009(ot_root: Path, tmp_path: Path, server: str) -> None:
    """get_pharmacogenomics(drug_id=...) is always empty (both servers' copies)."""
    import dl_fixtures as F

    assert len(F.oracle_pgx(ot_root, drug_id=F.CHEMBL3)) == 3
    out = ot_call(ot_root, tmp_path, server, "get_pharmacogenomics", drug_id=F.CHEMBL3)
    assert out["count"] == 0


@pytest.mark.parametrize("server", ["drug", "target"])
def test_ot_drug_010(ot_root: Path, tmp_path: Path, server: str) -> None:
    """With target_id and drug_id, drug_id is ignored: rows that do not involve the drug come back."""
    import dl_fixtures as F

    oracle = F.oracle_pgx(ot_root, target_id=F.T, drug_id=F.CHEMBL25)
    out = ot_call(ot_root, tmp_path, server, "get_pharmacogenomics", target_id=F.T, drug_id=F.CHEMBL25)
    got = keys(out["pgx_relationships"], F.PGX_KEY)
    violating = [r for r in out["pgx_relationships"] if F.CHEMBL25 not in F.pgx_drug_ids(r)]
    assert len(oracle) == 2 and len(got) == 3 and violating


# ---------------------------------------------------------------------------- disease


def test_ot_dis_002(ot_root: Path, tmp_path: Path) -> None:
    """Synonym search is dead: 'NIDDM' (an exact synonym of T2D only) finds nothing."""
    import dl_fixtures as F

    assert F.oracle_disease_search(ot_root, "NIDDM") == [F.T2D]
    out = ot_call(ot_root, tmp_path, "disease", "search_diseases_by_name", query="NIDDM")
    assert out["count"] == 0


def test_ot_dis_006(ot_root: Path, tmp_path: Path) -> None:
    """find_diseases_by_phenotype keeps diseases with no matching item and counts negated items."""
    import dl_fixtures as F

    oracle = F.oracle_phenotype(ot_root, F.SEIZURE, evidence_type="IEA")
    out = ot_call(ot_root, tmp_path, "disease", "find_diseases_by_phenotype", phenotype_id=F.SEIZURE,
                  evidence_type="IEA")
    counts = {d["disease_id"]: d["evidence_count"] for d in out["diseases"]}
    assert counts.get(F.PHENO["X"]) == 0                         # no IEA item left, still listed
    assert counts.get(F.PHENO["W"]) == 1                         # its only IEA item is negated
    assert F.PHENO["X"] not in oracle["diseases"] and F.PHENO["W"] not in oracle["diseases"]
    plain = ot_call(ot_root, tmp_path, "disease", "find_diseases_by_phenotype", phenotype_id=F.SEIZURE)
    assert {d["disease_id"] for d in plain["diseases"]} == set(F.PHENO.values())   # negated-only X listed as support


# ---------------------------------------------------------------------------- association


def test_ot_assoc_005(ot_copy, tmp_path: Path) -> None:
    """filter_by_datatype writes exactly the upstream limit's rows to output_path, ranked across data
    types when datatype is omitted: an inflated limit would grow the agent's file."""
    import dl_fixtures as F
    import pyarrow.parquet as pq

    rows = [{"diseaseId": f"EFO_00{60000 + i:05d}", "targetId": F.T, "datatypeId": dt, "score": s, "evidenceCount": 1}
            for i, (dt, s) in enumerate([("genetic_association", 0.9), ("literature", 0.8),
                                         ("literature", 0.7), ("genetic_association", 0.3)])]
    root = ot_copy(association_by_datatype_direct=rows)
    for limit in (2, 3):
        path = tmp_path / f"by_datatype_{limit}.parquet"
        out = ot_call(root, tmp_path, "association", "filter_by_datatype", output_path=str(path), target_id=F.T,
                      limit=limit)
        written = pq.read_table(path).to_pylist()
        assert out["count"] == limit and len(written) == limit
        assert len({r["datatypeId"] for r in written}) == 2       # scores of two data types ranked together


def test_ot_assoc_009(ot_root: Path, tmp_path: Path) -> None:
    """get_evidence_by_publication is always empty: literature is an ndarray, not a list."""
    import dl_fixtures as F

    assert len(F.oracle_evidence_by_publication(ot_root, F.PMID_EPMC)) == 2
    out = ot_call(ot_root, tmp_path, "association", "get_evidence_by_publication", pmid=F.PMID_EPMC)
    assert out["count"] == 0


def _unit(sim: float, dim: int = 100) -> list[float]:
    return [sim, math.sqrt(1 - sim * sim)] + [0.0] * (dim - 2)


def test_ot_assoc_010(ot_copy, tmp_path: Path) -> None:
    """find_similar_entities: a norm-0 candidate gives a NaN cosine and the sort leaves the list unordered."""
    import dl_fixtures as F

    anchor = {"category": "target", "word": F.PCSK9, "norm": 1.0, "vector": _unit(1.0)}
    cands = [("ENSG00000200001", 0.2), ("ENSG00000200002", None), ("ENSG00000200003", 0.9), ("ENSG00000200004", 0.5)]
    rows = [anchor] + [{"category": "target", "word": w, "norm": 0.0 if s is None else 1.0,
                        "vector": [0.0] * 100 if s is None else _unit(s)} for w, s in cands]
    root = ot_copy(literature_vector=rows)
    out = ot_call(root, tmp_path, "association", "find_similar_entities", entity_id=F.PCSK9, top_k=1)
    assert [e["entity_id"] for e in out["similar_entities"]] != ["ENSG00000200003"]   # the true nearest
    full = ot_call(root, tmp_path, "association", "find_similar_entities", entity_id=F.PCSK9, top_k=10)
    sims = [e["similarity"] for e in full["similar_entities"]]
    assert any(isinstance(s, float) and math.isnan(s) for s in sims) or None in sims or "NaN" in map(str, sims)
    finite = [s for s in sims if isinstance(s, float) and not math.isnan(s)]
    assert finite != sorted(finite, reverse=True)


# ---------------------------------------------------------------------------- genetics


def test_ot_gen_002(ot_root: Path, tmp_path: Path) -> None:
    """query_l2g_predictions cuts head(limit) in file order before sorting by score."""
    import dl_fixtures as F

    out = ot_call(ot_root, tmp_path, "genetics", "query_l2g_predictions", gene_id=F.G_L2G, min_score=0.05, limit=5)
    got = keys(out["results"], ("studyLocusId", "geneId"))
    assert got != F.oracle_l2g_topk(ot_root, F.G_L2G, 0.05, 5)
    assert max(r["score"] for r in out["results"]) < 0.8725      # the best locus is never seen


def test_ot_gen_007(ot_root: Path, tmp_path: Path) -> None:
    """get_study_metadata(min_sample_size=...) keeps null sizes, which fill the limit first."""
    import dl_fixtures as F

    oracle = F.oracle_studies(ot_root, 1000)
    out = ot_call(ot_root, tmp_path, "genetics", "get_study_metadata", min_sample_size=1000, limit=5)
    assert F.STUDY_BIG in oracle["ids"] and oracle["unknown"] == 25
    assert out["count"] == 5 and all(s["nSamples"] is None for s in out["studies"])


# ---------------------------------------------------------------------------- pathway


def test_ot_path_003(ot_root: Path, tmp_path: Path) -> None:
    """get_gene_ontology is always empty: go is an ndarray, not a list."""
    import dl_fixtures as F

    assert len(F.oracle_go_items(ot_root, F.PCSK9)) == 2
    out = ot_call(ot_root, tmp_path, "pathway", "get_gene_ontology", target_id=F.PCSK9)
    assert out["count"] == 0


# ---------------------------------------------------------------------------- cBioPortal (stubbed pybioportal)


def cbio_call(tmp: Path, func: str, **kwargs: Any) -> dict[str, Any]:
    return call_upstream("src.mcp_servers.clinicaltrials_mcp.tools", func, tmp, extra_paths=(str(STUBS_DIR),), **kwargs)


def test_ct_cbio_001(tmp_path: Path) -> None:
    """Unknown sample IDs come back as phantom rows (patientId null) and count in sample_count."""
    out = cbio_call(tmp_path, "get_clinical_data", study_id="study_x", sample_ids=["S-01", "NOPE-01"])
    phantom = [r for r in out["data"] if r["sampleId"] == "NOPE-01"]
    assert out["success"] and out["sample_count"] == 2
    assert phantom and phantom[0]["patientId"] is None


def test_ct_cbio_002(tmp_path: Path) -> None:
    """Patient attributes are copied onto every sample: P-01's OS_MONTHS appears once per sample."""
    out = cbio_call(tmp_path, "get_clinical_data", study_id="study_x")
    os_months = [r.get("OS_MONTHS") for r in out["data"] if r.get("patientId") == "P-01"]
    assert os_months == ["24.5", "24.5"]                         # two samples, one patient
    assert len({r["patientId"] for r in out["data"]}) < out["sample_count"]


# ---------------------------------------------------------------------------- order_source: upstream_full_sort
# The ranking refusal is skipped only for a full upstream sort proven here (§11.6): the strongest row is
# put last in file order and must come back first with limit 1.

def _best_last(root: Path, table: str, best: dict[str, Any]) -> list[dict[str, Any]]:
    import dl_fixtures as F

    return [*F.read_rows(root, table), best]


@pytest.mark.parametrize("func,kwargs", [
    ("query_associations", {"target_id": "ENSG00000169174"}),
    ("get_associations_for_target", {"target_id": "ENSG00000169174"}),
    ("get_associations_for_disease", {"disease_id": "EFO_0099999"}),
])
def test_full_sort_associations(func: str, kwargs: dict[str, Any], ot_root: Path, ot_copy, tmp_path: Path) -> None:
    rows = _best_last(ot_root, "association_overall_direct",
                      {"diseaseId": "EFO_0099999", "targetId": "ENSG00000169174", "score": 0.999, "evidenceCount": 1})
    root = ot_copy(association_overall_direct=rows)
    out = ot_call(root, tmp_path, "association", func, output_path=str(tmp_path / f"{func}.parquet"), limit=1,
                  **kwargs)
    assert [(r["diseaseId"], r["score"]) for r in out["top_associations"]] == [("EFO_0099999", 0.999)], out


@pytest.mark.parametrize("func,table,column", [("filter_by_datatype", "association_by_datatype_direct", "datatypeId"),
                                               ("filter_by_datasource", "association_by_datasource_direct",
                                                "datasourceId")])
def test_full_sort_filters(func: str, table: str, column: str, ot_copy, tmp_path: Path) -> None:
    value = "literature" if column == "datatypeId" else "europepmc"
    rows = [{"diseaseId": f"EFO_00{60000 + i:05d}", "targetId": "ENSG00000169174", column: value, "score": sc,
             "evidenceCount": 1} for i, sc in enumerate([0.2, 0.5, 0.3])]
    first = rows[0]
    root = ot_copy(**{table: [*rows, {**first, "diseaseId": "EFO_0099999", "score": 0.999}]})
    arg = "datatype" if column == "datatypeId" else "datasource"
    out = ot_call(root, tmp_path, "association", func, output_path=str(tmp_path / f"{func}.parquet"), limit=1,
                  **{arg: first[column]})
    assert [r["diseaseId"] for r in out["top_associations"]] == ["EFO_0099999"], out


def test_full_sort_similar(ot_copy, tmp_path: Path) -> None:
    """find_similar_entities scores every candidate before top_k (finite cosines: the true nearest wins)."""
    import dl_fixtures as F

    anchor = {"category": "target", "word": F.PCSK9, "norm": 1.0, "vector": _unit(1.0)}
    cands = [("ENSG00000200001", 0.2), ("ENSG00000200002", 0.5), ("ENSG00000200003", 0.9)]   # best last
    rows = [anchor] + [{"category": "target", "word": w, "norm": 1.0, "vector": _unit(s)} for w, s in cands]
    root = ot_copy(literature_vector=rows)
    out = ot_call(root, tmp_path, "association", "find_similar_entities", entity_id=F.PCSK9, top_k=1)
    assert [e["entity_id"] for e in out["similar_entities"]] == ["ENSG00000200003"], out


@pytest.mark.parametrize("func,top,best", [
    ("query_gwas_associations", "top_associations", {"pValueMantissa": 1.0, "pValueExponent": -300}),
    ("get_credible_sets", "top_credible_sets", {"credibleSetlog10BF": 1.0e6}),
])
def test_full_sort_genetics(func: str, top: str, best: dict[str, Any], ot_root: Path, ot_copy,
                            tmp_path: Path) -> None:
    import dl_fixtures as F

    first = F.read_rows(ot_root, "credible_set")[0]
    root = ot_copy(credible_set=_best_last(ot_root, "credible_set",
                                           {**first, "studyLocusId": "f" * 32, **best}))
    out = ot_call(root, tmp_path, "genetics", func, output_path=str(tmp_path / f"{func}.parquet"), limit=1,
                  study_id=first["studyId"])
    assert [r["studyLocusId"] for r in out[top]] == ["f" * 32], out


# ---------------------------------------------------------------------------- ClinicalTrials.gov (stubbed requests)

CTGOV_STUB = STUBS_DIR / "ctgov"


def ctgov_call(tmp: Path, monkeypatch: pytest.MonkeyPatch, pages: list[dict[str, Any]], func: str,
               **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run an upstream clinicaltrials tool against canned registry replies; returns (result, requests)."""
    pf, log = tmp / "ctgov_pages.json", tmp / "ctgov_requests.jsonl"
    pf.write_text(json.dumps(pages))
    monkeypatch.setenv("VBT_CTGOV_PAGES", str(pf))
    monkeypatch.setenv("VBT_CTGOV_LOG", str(log))
    out = call_upstream("src.mcp_servers.clinicaltrials_mcp.tools", func, tmp,
                        extra_paths=(str(CTGOV_STUB), str(STUBS_DIR)), **kwargs)
    reqs = [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []
    return out, reqs


def _studies(n: int, start: int = 0) -> list[dict[str, Any]]:
    return [{"protocolSection": {"identificationModule": {"nctId": f"NCT{start + i:08d}"}}} for i in range(n)]


def test_ct_gov_001(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """country defaults to 'United States': a count with no country is narrowed to US sites."""
    out, reqs = ctgov_call(tmp_path, monkeypatch, [{"json": {"totalCount": 89}}], "count_clinical_trials",
                           condition="melanoma")
    assert out["total_count"] == 89
    assert 'AREA[LocationCountry]"United States"' in reqs[0]["params"]["filter.advanced"]


def test_ct_gov_002(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """eligibility_text is pasted inside double quotes unescaped: a quote in it ends the phrase."""
    _out, reqs = ctgov_call(tmp_path, monkeypatch, [{"json": {"totalCount": 1}}], "count_clinical_trials",
                            eligibility_text=['ECOG "0-1'], country=None)
    assert 'AREA[EligibilityCriteria]"ECOG "0-1"' in reqs[0]["params"]["filter.advanced"]


def test_ct_gov_003(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pagination stops at 1,000 trials and the result is not marked truncated (total 5,000)."""
    pages = [{"json": {"totalCount": 5000, "studies": _studies(500, 500 * p), "nextPageToken": f"t{p}"}}
             for p in range(3)]
    out, reqs = ctgov_call(tmp_path, monkeypatch, pages, "search_clinical_trials", condition="cancer", country=None)
    assert out["trials_returned"] == 1000 and out["total_count"] == 5000 and len(reqs) == 2
    assert "truncated" not in out and "warning" not in out


def test_ct_gov_004(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply without totalCount is reported as total_count 0, a definite count of none."""
    out, _reqs = ctgov_call(tmp_path, monkeypatch, [{"json": {"studies": []}}], "count_clinical_trials",
                            condition="melanoma", country=None)
    assert out["success"] is True and out["total_count"] == 0


def test_ct_gov_005(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """search: without totalCount the rows fetched become total_count while more pages exist."""
    pages = [{"json": {"studies": _studies(600, 600 * p), "nextPageToken": f"t{p}"}} for p in range(3)]
    out, _reqs = ctgov_call(tmp_path, monkeypatch, pages, "search_clinical_trials", condition="cancer", country=None)
    assert out["total_count"] == out["trials_returned"] == 1200 and "truncated" not in out


# ---------------------------------------------------------------------------- CELLxGENE Census (stubbed)

CENSUS_STUB = STUBS_DIR / "census"


def census_call(tmp: Path, monkeypatch: pytest.MonkeyPatch, data: dict[str, Any], func: str,
                **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pytest.importorskip("anndata")
    df, log = tmp / "census.json", tmp / "census_log.jsonl"
    df.write_text(json.dumps(data))
    monkeypatch.setenv("VBT_CENSUS_DATA", str(df))
    monkeypatch.setenv("VBT_CENSUS_LOG", str(log))
    out = call_upstream("src.mcp_servers.single_cell_mcp.tools", func, tmp, extra_paths=(str(CENSUS_STUB),), **kwargs)
    return out, [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def _cells(spec: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
    """``(dataset_id, donor_id, n)`` blocks of T cells in lung, numbered by soma_joinid in that order."""
    out = []
    for dataset, donor, n in spec:
        for _ in range(n):
            out.append({"soma_joinid": len(out), "dataset_id": dataset, "donor_id": donor, "cell_type": "T cell",
                        "tissue": "lung", "disease": "normal", "assay": "10x", "sex": "female",
                        "development_stage": "adult", "suspension_type": "cell", "is_primary_data": True})
    return out


GENES = [{"feature_id": "ENSG00000000001", "feature_name": "G1", "feature_length": 1000, "n_measured_obs": 10,
          "nnz": 4},
         {"feature_id": "ENSG00000000002", "feature_name": "G0", "feature_length": 900, "n_measured_obs": 0, "nnz": 0}]


def test_sc_001(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Donors are grouped by donor_id alone: D1 of dataset d1 and D1 of dataset d2 count as one donor."""
    data = {"obs": _cells([("d1", "D1", 2), ("d2", "D1", 2)]), "var": GENES}
    out, _log = census_call(tmp_path, monkeypatch, data, "get_anndata_donor_balanced",
                            output_path=str(tmp_path / "b.h5ad"), value_filter="cell_type == 'T cell'")
    assert out["n_cells_sampled"] == 4 and out["n_donors"] == 1      # two donors in truth
    assert out["donor_summary"] == {"D1": 4}


def test_sc_002(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """query_cell_metadata keeps the first cells by join id: with limit 3 only the first dataset remains."""
    import pandas as pd

    data = {"obs": _cells([("d1", "A", 3), ("d2", "B", 3)]), "var": GENES}
    out, _log = census_call(tmp_path, monkeypatch, data, "query_cell_metadata",
                            output_path=str(tmp_path / "m.parquet"), value_filter="tissue == 'lung'", limit=3)
    assert out["n_cells_total"] == 6 and out["limited"] is True
    assert set(pd.read_parquet(out["output_path"])["dataset_id"]) == {"d1"}


def test_sc_003(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A gene measured in no cell gets sparsity 1.0 (all zeros) instead of undefined."""
    out, _log = census_call(tmp_path, monkeypatch, {"obs": _cells([("d1", "A", 1)]), "var": GENES},
                            "get_gene_statistics", gene_symbols=["G0"])
    assert out["gene_statistics"][0]["n_measured_obs"] == 0 and out["gene_statistics"][0]["sparsity"] == 1.0


def test_sc_004(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without value_filter get_expression_for_genes reads 'soma_joinid < max_cells': the first cells only."""
    data = {"obs": _cells([("d1", "A", 3), ("d2", "B", 3)]), "var": GENES}
    out, log = census_call(tmp_path, monkeypatch, data, "get_expression_for_genes", gene_symbols=["G1"],
                           output_path=str(tmp_path / "e.h5ad"), max_cells=3)
    assert out["n_cells"] == 3
    assert any(e.get("obs_value_filter") == "soma_joinid < 3" for e in log if e["kind"] == "get_anndata"), log

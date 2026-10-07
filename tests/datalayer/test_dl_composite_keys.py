"""Composite-key completion on the CT-5 Tahoe fixture and ChEMBL families (§6.6, §8.5; phase 3, F17).

The CT-5 fixture profiles Bortezomib in ACH-000681 (A549) and ACH-000001 at 0.05, 0.5 and 5.0 uM on
plate '1', and 5.0 uM in A549 again on plate '2'; SELECTIVE1 is significant only in A549; GENE2 has the
same effect in both lines; 'Erdafitinib ' (stored with a trailing space) was profiled only in A549 at
0.5 uM.

* ``find_cell_line_selective_effects``: SELECTIVE1 is selective (exclusive: significant in no tested
  comparison line, kept rather than dropped as upstream's NaN merge did), at each dose and plate
  separately; GENE2 is not; the comparison covers every line tested at that dose;
* ``compare_drug_effects``: rows pair only within one cell line and concentration: GENE2 pairs once (an
  upstream ``merge(on='gene_name')`` paired it six times, across doses and lines);
* a drug x cell line that was never profiled is ``not_found`` with ``subkind: combination_not_profiled``
  listing the profiled values;
* ``get_drug_indications`` with ``family: include``: a salt form's call returns the parent's and the
  salt's indications, each row keeping its stored ID.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow

pytestmark = needs_arrow


def _ctx(tmp: Path, env: dict[str, str]) -> Any:
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.settings import DataSettings

    settings = DataSettings.from_dict({"descriptors_dir": str(REPO / "configs" / "data" / "sources"),
                                       "overlays_dir": str(REPO / "configs" / "data" / "overlays"),
                                       "cache_dir": str(tmp / "cache")}, project_root=REPO)
    return ServiceContext(settings)


@pytest.fixture(scope="module")
def tahoe(tahoe_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TAHOE_DATA_PATH", str(tahoe_root))
        yield _ctx(tmp_path_factory.mktemp("pairs"), {})


def call(ctx: Any, name: str, /, **payload: Any) -> dict[str, Any]:
    from vbt.datalayer.service.verbs import load_verbs

    out = load_verbs()[name](ctx, payload)
    json.dumps(out)
    return out


def ok(out: dict[str, Any]) -> dict[str, Any]:
    assert out.get("status") != "tool_error", json.dumps(out)[:1500]
    return out


# ---------------------------------------------------------------------------- selectivity


def test_selective_gene_is_kept_and_compared_with_every_tested_line(tahoe) -> None:
    import dl_fixtures as F

    out = ok(call(tahoe, "_pairs", mode="selectivity", drug="Bortezomib", cell_line_of_interest=F.A549))
    rows = out["rows"]
    sel = [r for r in rows if r["gene_name"] == F.SELECTIVE]
    assert len(sel) == 4, "one row per dose and plate (0.05, 0.5, 5.0 on plate 1, 5.0 on plate 2)"
    assert all(r["exclusive"] and r["selectivity_ratio"] is None and r["n_comparison_cell_lines"] == 1 and
               r["n_comparison_significant"] == 0 for r in sel)
    assert {(r["concentration"], r["plate"]) for r in sel} == {(0.05, "1"), (0.5, "1"), (5.0, "1"), (5.0, "2")}
    assert not [r for r in rows if r["gene_name"] == "GENE2"], "the same effect in both lines is not selective"
    stats = out["_vbt"]["selectivity"]
    assert stats["comparison_lines"] == {"0.05 uM": [F.OTHER_LINE], "0.5 uM": [F.OTHER_LINE], "5.0 uM": [F.OTHER_LINE]}
    # exclusive genes rank first, then by selectivity
    assert rows[0]["exclusive"] is True
    one_dose = ok(call(tahoe, "_pairs", mode="selectivity", drug="Bortezomib", cell_line_of_interest=F.A549,
                       concentration=0.5))
    assert {r["gene_name"] for r in one_dose["rows"]} == {F.SELECTIVE}
    # GENE3 (0.7 in both lines at 0.5) becomes selective only against a threshold below 1
    low = ok(call(tahoe, "_pairs", mode="selectivity", drug="Bortezomib", cell_line_of_interest=F.A549,
                  concentration=0.5, selectivity_threshold=0.5))
    gene3 = next(r for r in low["rows"] if r["gene_name"] == "GENE3")
    assert gene3["selectivity_ratio"] == pytest.approx(1.0) and gene3["n_comparison_significant"] == 1


def test_never_profiled_combination_lists_the_profiled_values(tahoe) -> None:
    import dl_fixtures as F

    bad = call(tahoe, "_pairs", mode="selectivity", drug="Erdafitinib", cell_line_of_interest=F.OTHER_LINE)
    assert bad["status"] == "tool_error" and bad["kind"] == "not_found", bad
    assert bad["subkind"] == "combination_not_profiled" and bad["profiled"]["Cell_ID_DepMap"] == [F.A549]
    bad = call(tahoe, "_pairs", mode="compare", drug_a="Bortezomib", drug_b="Erdafitinib", cell_line=F.OTHER_LINE)
    assert bad["kind"] == "not_found" and bad["subkind"] == "combination_not_profiled"
    bad = call(tahoe, "_pairs", mode="selectivity", drug="Bortezomib", cell_line_of_interest=F.A549,
               comparison_cell_lines=[F.A549], concentration=0.05)
    assert bad["kind"] == "not_found" and bad["subkind"] == "combination_not_profiled"


# ---------------------------------------------------------------------------- paired comparison


def test_compare_pairs_within_one_context_without_cartesian_products(tahoe) -> None:
    import dl_fixtures as F

    out = ok(call(tahoe, "_pairs", mode="compare", drug_a="Bortezomib", drug_b="Erdafitinib", min_abs_log2fc=0))
    # "Erdafitinib" resolves and is read under its stored spelling "Erdafitinib "
    ctxs = out["contexts"]
    assert [(c["Cell_ID_DepMap"], c["concentration"]) for c in ctxs] == [(F.A549, 0.5)]
    pairs = out["shared_targets_same_direction"] + out["opposite_effects"]
    assert [(p["gene_name"], p["plate_a"], p["plate_b"]) for p in pairs] == [("GENE2", "1", "1")]
    assert out["opposite_effects"][0]["log2FC_drug_a"] == -2.0 and out["opposite_effects"][0]["log2FC_drug_b"] == 1.1
    assert {u["gene_name"] for u in out["unique_to_a"]} == {F.SELECTIVE, "GENE3"} and out["unique_to_b"] == []
    assert out["signature_correlation"] is None, "one pair: a correlation is not computable (not 0)"
    # upstream's merge on gene_name alone would have paired GENE2 once per Bortezomib dose and line
    upstream_pairs = sum(1 for r in F.tahoe_rows() if r["drug"] == "Bortezomib" and r["gene_name"] == "GENE2")
    assert upstream_pairs == 6 and out["stats"]["num_pairs"] == 1


def test_pearson_is_null_when_not_computable() -> None:
    from vbt.datalayer.service.verbs.pairs import pearson

    assert pearson([1.0, 2.0], [2.0, 4.0]) is None
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.5]) == pytest.approx(statistics.correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.5]))


async def test_tahoe_tools_are_derived_through_the_gateway(tahoe, tmp_path) -> None:
    import dl_fixtures as F
    from test_dl_native_tools import _derived, _gateway

    gw = _gateway(tahoe, tmp_path)
    res = await _derived(gw, "functional_genomics", "find_cell_line_selective_effects",
                         {"drug_name": "Bortezomib", "cell_line_of_interest": "A549"})
    genes = res.obj["selective_genes"]
    assert {g["gene_name"] for g in genes} == {F.SELECTIVE, "GENE4"} and res.obj["num_selective"] == 5
    assert res.obj["drug_info"]["drug"] == "Bortezomib"
    cmp = await _derived(gw, "functional_genomics", "compare_drug_effects",
                         {"drug_a": "Bortezomib", "drug_b": "Erdafitinib", "min_abs_log2fc": 0})
    obj = cmp.obj
    assert [p["gene_name"] for p in obj["opposite_effects"]] == ["GENE2"] and obj["shared_targets_same_direction"] == []
    assert obj["drug_b_info"]["drug"] == "Erdafitinib" and "similarity_score" not in obj


# ---------------------------------------------------------------------------- ChEMBL families


async def test_salt_family_fans_out_and_keeps_stored_ids(tmp_path) -> None:
    import dl_fixtures as F
    from test_dl_native_tools import _derived, _gateway

    ot = tmp_path / "ot" / "25.09"
    F.write_table(ot, "drug_molecule", F.table("drug_molecule", [
        {"id": "CHEMBL25", "name": "ASPIRIN", "drugType": "Small molecule"},
        {"id": "CHEMBL553", "name": "ERLOTINIB", "drugType": "Small molecule", "childChemblIds": ["CHEMBL1079742"]},
        {"id": "CHEMBL1079742", "name": "ERLOTINIB HYDROCHLORIDE", "drugType": "Small molecule",
         "parentId": "CHEMBL553", "tradeNames": ["Tarceva"]}]))
    F.write_table(ot, "drug_indication", F.table("drug_indication", [
        {"id": "CHEMBL553", "indications": [{"disease": "EFO_0000001", "efoName": "a", "maxPhaseForIndication": 4.0}],
         "indicationCount": 1},
        {"id": "CHEMBL1079742", "indications": [{"disease": "EFO_0000002", "efoName": "b",
                                                 "maxPhaseForIndication": 2.0}], "indicationCount": 1}]))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
        ctx = _ctx(tmp_path, {})
        gw = _gateway(ctx, tmp_path)
        res = await _derived(gw, "drug", "get_drug_indications", {"drug_id": "Tarceva"})
    rows = res.obj["indications"]
    assert {(r["id"], r["disease"]) for r in rows} == {("CHEMBL553", "EFO_0000001"), ("CHEMBL1079742", "EFO_0000002")}
    assert any("family members" in n for n in res.header.get("notes", [])), res.header

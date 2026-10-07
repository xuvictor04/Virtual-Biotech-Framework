"""Matrices through the shipped descriptors (§6.7, F12; S4 DepMap, S5 AnnData cohorts).

* A DepMap-shaped ``CRISPRGeneEffect.csv`` read with the ``MatrixSpec`` of ``configs/data/sources/
  depmap.yaml``: the ``SYMBOL (ENTREZ)`` header parses into the column axis, the row axis is found
  under each release's header spelling (``ModelID``, ``DepMap_ID``, blank), the sentinel cell holds,
  a duplicate header is reported and read by position, and a file cut at a row boundary (which
  parses) is caught by the inline manifest's byte count.
* A tiny AnnData whose ``var_names`` are positional digits: flagged, and the declared ``var`` key
  column used instead; keyed by the index, the same file is an error.
* Two tiny GEO cohorts read with ``zenodo_vbt.ibd_cohorts``: fragment keys from the file names, the
  per-fragment overrides (``GSE73661`` has no ``response_clinical`` but ``response_mucosal_healing``),
  per-fragment universes (a gene on one platform only), and the long view against a dense oracle.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402

from vbt.datalayer.descriptor.columns import validate_column  # noqa: E402
from vbt.datalayer.descriptor.load import load_descriptors  # noqa: E402
from vbt.datalayer.plugins.base import Fragment, LayoutSpec  # noqa: E402
from vbt.datalayer.plugins.formats.csv import matrix_layout  # noqa: E402
from vbt.datalayer.plugins.layouts import PROBE_STATUS  # noqa: E402
from vbt.datalayer.plugins.registry import discover  # noqa: E402
from vbt.datalayer.predicate import Eq, In  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NAN = float("nan")


@pytest.fixture(scope="module")
def reg():
    return discover(entry_points=False)


@pytest.fixture(scope="module")
def sources():
    return load_descriptors(REPO / "configs" / "data" / "sources", {"project_root": str(REPO)})


def frag(path: Path | str, **kw: Any) -> Fragment:
    st = os.stat(path)
    return Fragment(uri=str(path), size=st.st_size if os.path.isfile(path) else None, mtime_ns=st.st_mtime_ns, **kw)


def cells(plugin: Any, f: Fragment, value: str, *, row_key: str, col_key: str, **kw: Any) -> list[tuple]:
    out = []
    for batch in plugin.slice(f, value, row_predicate=kw.pop("row_predicate", None),
                              col_keys=kw.pop("col_keys", None), budget_bytes=kw.pop("budget_bytes", None), **kw):
        for r in plugin.to_native(pa.Table.from_batches([batch])):
            out.append((r[row_key], r[col_key], r[value]))
    return sorted(out, key=lambda c: (c[0], c[1]))


# ---------------------------------------------------------------------------- DepMap (S4)

GENES = (("RPL3", "6122"), ("TP53", "7157"), ("A1BG", "1"), ("SEPTIN9", "10801"))
MODELS = ("ACH-000001", "ACH-000002", "ACH-000003")
EFFECT = ((-1.8, 0.1, None, -0.2), (-1.2, NAN, 0.05, -0.4), (-2.1, -0.9, 0.0, None))


def write_depmap(path: Path, *, row_header: str = "ModelID", genes=GENES, rows=EFFECT) -> None:
    lines = [",".join([row_header, *(f"{s} ({e})" for s, e in genes)])]
    for model, vals in zip(MODELS, rows):
        lines.append(",".join([model, *("" if v is None else ("nan" if v != v else repr(v)) for v in vals)]))
    path.write_text("\n".join(lines) + "\n")


def oracle(rows=EFFECT, genes=GENES) -> list[tuple]:
    out = []
    for model, vals in zip(MODELS, rows):
        for (_s, e), v in zip(genes, vals):
            out.append((model, e, None if v is None or (isinstance(v, float) and math.isnan(v)) else v))
    return sorted(out, key=lambda c: (c[0], c[1]))


@pytest.mark.parametrize("row_header", ["ModelID", "DepMap_ID", ""])
def test_depmap_gene_effect_header_parse_and_long_view(reg, sources, tmp_path, row_header) -> None:
    spec = sources["depmap"].tables["gene_effect"]
    path = tmp_path / spec.path
    write_depmap(path, row_header=row_header)
    csv = reg.get("format", "csv").configure({}, spec.matrix)
    f = frag(path)
    layout = matrix_layout(spec.matrix)
    assert (layout.row.key, layout.col.key, layout.values) == (("ModelID",), ("entrez_id",), ("gene_effect",))
    cols = csv.to_native(csv.axis_values(f, "col"))
    assert [(c["symbol"], c["entrez_id"]) for c in cols] == list(GENES)
    assert csv.axis_values(f, "row").column("ModelID").to_pylist() == list(MODELS)
    assert cells(csv, f, "gene_effect", row_key="ModelID", col_key="entrez_id") == oracle()
    # the descriptor's present sentinel: RPL3 in ACH-000001 is a dependency (gene_effect < -0.5)
    (hit,) = cells(csv, f, "gene_effect", row_key="ModelID", col_key="entrez_id",
                   row_predicate=Eq("ModelID", "ACH-000001"), col_keys=["6122"])
    assert hit[2] < -0.5
    assert [i for i in csv.matrix_checks(f) if not i.ok] == []
    st = csv.stats(f)
    assert st.shape == (3, 4) and st.method == "scan"
    # an empty cell is not measured: null in the long view, never 0
    assert ("ACH-000001", "1", None) in cells(csv, f, "gene_effect", row_key="ModelID", col_key="entrez_id")


def test_depmap_duplicate_and_unparseable_headers(reg, sources, tmp_path) -> None:
    spec = sources["depmap"].tables["gene_effect"]
    path = tmp_path / "CRISPRGeneEffect.csv"
    genes = (("RPL3", "6122"), ("TP53", "7157"), ("TP53", "7157"), ("A1BG", "1"))
    write_depmap(path, genes=genes)
    csv = reg.get("format", "csv").configure({}, spec.matrix)
    failed = {i.name: i for i in csv.matrix_checks(frag(path)) if not i.ok}
    assert set(failed) == {"duplicate_header", "duplicate_key"}
    assert "TP53 (7157)" in failed["duplicate_header"].detail
    # values come by position: both TP53 columns keep their own values
    got = cells(csv, frag(path), "gene_effect", row_key="ModelID", col_key="entrez_id", col_keys=["7157"])
    assert got == [c for c in oracle(genes=genes) if c[1] == "7157"] and len(got) == 6
    assert [v for _m, _e, v in got][:2] == [EFFECT[0][1], EFFECT[0][2]]      # 0.1 and the empty cell
    bad = tmp_path / "bad" / "CRISPRGeneEffect.csv"
    bad.parent.mkdir()
    bad.write_text("ModelID,RPL3 (6122),not a gene\nACH-000001,-1.0,0.5\n")
    items = {i.name: i for i in csv.matrix_checks(frag(bad)) if not i.ok}
    assert "header_parse" in items and items["header_parse"].level == "error"
    from vbt.datalayer.plugins.base import FormatError

    with pytest.raises(FormatError, match="do not match"):
        csv.axis_values(frag(bad), "col")


def test_depmap_truncated_file_is_caught_by_the_inline_manifest(reg, sources, tmp_path) -> None:
    from vbt.datalayer.descriptor.models import ManifestSpec
    from vbt.datalayer.service import load_manifest

    spec = sources["depmap"].tables["gene_effect"]
    path = tmp_path / spec.path
    write_depmap(path)
    full = path.stat().st_size
    manifest = load_manifest(str(tmp_path), ManifestSpec.model_validate(
        {"inline": {spec.path: {"bytes": full}}, "required": True}))
    layout = reg.get("layout", "single_file")
    lspec = LayoutSpec(table="depmap.gene_effect", path=spec.path, format="csv")
    assert all(i.ok for i in layout.probe(str(tmp_path), lspec, manifest) if i.level == "error")
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:-1]))                       # cut exactly at a row boundary: still parses
    csv = reg.get("format", "csv").configure({}, spec.matrix)
    assert len(csv.axis_values(frag(path), "row")) == 2       # a smaller valid matrix ...
    failed = [i for i in layout.probe(str(tmp_path), lspec, manifest) if not i.ok]
    assert [i.name for i in failed] == ["manifest_bytes"]      # ... that readiness refuses (I14)
    assert PROBE_STATUS["manifest_bytes"] == "partial"


def test_depmap_model_and_common_essentials(reg, sources, tmp_path) -> None:
    desc = sources["depmap"]
    model = desc.tables["model"]
    assert model.columns["DepMap_ID"].equals == "ModelID" and model.columns["DepMap_ID"].optional
    ess = desc.tables["common_essentials"]
    path = tmp_path / ess.path
    path.write_text("Essentials\nRPL3 (6122)\nPOLR2A (5430)\n")
    csv = reg.get("format", "csv")
    lspec = LayoutSpec(table="depmap.common_essentials", path=ess.path, format="csv",
                       fragment_key=ess.fragment_key.model_dump(by_alias=True))
    (f,) = reg.get("layout", "single_file").fragments(str(tmp_path), lspec)
    assert f.fragment_key == "InferredCommonEssentials"          # the set name: a constant per file
    import re

    pattern = ess.columns["Essentials"].parse.pattern
    rows = [r["Essentials"] for r in csv.to_native(pa.Table.from_batches(list(
        csv.scan([f], columns=None, predicate=None, partitions={}))))]
    assert [re.fullmatch(pattern, r).group("entrez_id") for r in rows] == ["6122", "5430"]


# ---------------------------------------------------------------------------- AnnData (S5)

def _write_h5ad(path: Path, *, obs: dict[str, list[Any]], samples: list[str], var_index: list[str],
                var_cols: dict[str, list[Any]], x: Any, categorical: tuple[str, ...] = ()) -> None:
    import anndata
    import numpy as np
    import pandas as pd

    o = pd.DataFrame(index=pd.Index(samples, name="sample_id"))
    for k, v in obs.items():
        o[k] = pd.Categorical(v) if k in categorical else v
    var = pd.DataFrame(var_cols, index=pd.Index(var_index, name="gene_symbol" if "feature_id" not in var_cols else None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anndata.AnnData(X=np.asarray(x, dtype="float64"), obs=o, var=var).write_h5ad(path)


def test_positional_var_names_are_flagged_and_the_declared_key_used(reg, tmp_path) -> None:
    pytest.importorskip("anndata")
    path = tmp_path / "census.h5ad"
    _write_h5ad(path, obs={}, samples=["c0", "c1"], var_index=["0", "1", "2"],
                var_cols={"feature_id": ["ENSG00000141510", "ENSG00000169174", "ENSG00000000003"],
                          "feature_name": ["TP53", "PCSK9", "TSPAN6"]},
                x=[[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    keyed = {"axes": {"obs": {"name": "cell", "from": "index", "index_name": "soma_joinid",
                              "key": {"columns": ["soma_joinid"]}},
                      "var": {"name": "gene", "from": "column", "column": "feature_id",
                              "key": {"columns": ["feature_id"]}}},
             "values": {"X": {"role": "measure", "missing": "unknown"}}}
    h5ad = reg.get("format", "h5ad").configure({"backed": "r"}, keyed)
    f = frag(path)
    (item,) = [i for i in h5ad.matrix_checks(f) if not i.ok]
    assert item.name == "positional_index" and item.level == "warning" and "feature_id" in item.detail
    var = h5ad.to_native(h5ad.axis_values(f, "var"))
    assert [v["feature_id"] for v in var] == ["ENSG00000141510", "ENSG00000169174", "ENSG00000000003"]
    got = cells(h5ad, f, "X", row_key="soma_joinid", col_key="feature_id", col_keys=["ENSG00000169174"])
    assert got == [("c0", "ENSG00000169174", 0.0), ("c1", "ENSG00000169174", 3.0)]
    by_index = {**keyed, "axes": {**keyed["axes"], "var": {"name": "gene", "from": "index", "index_name": "var_name",
                                                            "key": {"columns": ["var_name"]}}}}
    bad = reg.get("format", "h5ad").configure({}, by_index)
    (err,) = [i for i in bad.matrix_checks(f) if not i.ok]
    assert err.name == "positional_index" and err.level == "error"


COHORTS = {
    "GSE12251": {"samples": ["GSM309000", "GSM309001", "GSM309002"],
                 "genes": ["OSMR", "IL6ST", "SEPT9"],
                 "obs": {"patient_id": ["P1", "P2", "P3"], "timepoint": ["W0", "W0", "nan"],
                         "response_clinical": ["R", "NR", "R"], "disease": ["UC"] * 3, "drug": ["Infliximab"] * 3},
                 "x": [[7.5, 8.1, NAN], [9.2, 8.0, 6.6], [7.1, 7.9, 6.0]]},
    "GSE73661": {"samples": ["GSM1888000", "GSM1888001"],
                 "genes": ["OSMR", "IL6ST"],                     # another platform: no SEPT9 probe
                 "obs": {"patient_id": ["Q1", "Q1"], "timepoint": ["W0", "W8"],
                         "response_mucosal_healing": ["NR", "R"], "disease": ["UC"] * 2, "drug": ["Infliximab"] * 2},
                 "x": [[10.0, 6.5], [9.1, 6.4]]},
}


@pytest.fixture()
def cohorts(tmp_path) -> Path:
    pytest.importorskip("anndata")
    root = tmp_path / "virtualbiotech_submission"
    data = root / "osmr" / "code" / "data"
    data.mkdir(parents=True)
    for gse, c in COHORTS.items():
        _write_h5ad(data / f"{gse}.h5ad", obs=c["obs"], samples=c["samples"], var_index=c["genes"],
                    var_cols={}, x=c["x"], categorical=("patient_id", "timepoint"))
    return root


def _effective_columns(spec: Any, key: str) -> dict[str, Any]:
    """The obs columns one fragment must hold: the declared ones, minus overrides with ``absent``, with
    overridden facets applied; ``present_in`` columns only in their fragments."""
    axis = spec.matrix.axes["row"]
    out: dict[str, Any] = {}
    for name, col in axis.columns.items():
        present_in = getattr(col, "present_in", None)
        if present_in is not None and key not in present_in:
            continue
        out[name] = col
    for name, facets in spec.fragment_overrides.get(key, {}).items():
        if facets.get("absent"):
            out.pop(name, None)
        else:
            out[name] = validate_column(facets)
    return out


def test_ibd_cohorts_fragment_keys_overrides_and_per_fragment_universes(reg, sources, cohorts) -> None:
    spec = sources["zenodo_vbt"].tables["ibd_cohorts"]
    layout = reg.get("layout", "single_file")
    lspec = LayoutSpec(table="zenodo_vbt.ibd_cohorts", path=spec.path, format="h5ad",
                       fragment_key=spec.fragment_key.model_dump(by_alias=True))
    frags = layout.fragments(str(cohorts), lspec)
    assert [f.fragment_key for f in frags] == ["GSE12251", "GSE73661"]
    h5ad = reg.get("format", "h5ad").configure(spec.format.options, spec.matrix)
    universes = {}
    for f in frags:
        obs = h5ad.to_native(h5ad.axis_values(f, "obs"))
        effective = _effective_columns(spec, f.fragment_key)
        for name, col in effective.items():
            if not getattr(col, "optional", False) or name in spec.fragment_overrides.get(f.fragment_key, {}):
                assert name in obs[0], f"{f.fragment_key}: {name} declared for this fragment but absent"
            vocab = getattr(col, "vocab", None)
            if isinstance(vocab, list) and name in obs[0]:
                assert {r[name] for r in obs} <= set(vocab), f"{f.fragment_key}.{name}"
        absent = [n for n, o in spec.fragment_overrides.get(f.fragment_key, {}).items() if o.get("absent")]
        assert all(n not in obs[0] for n in absent)
        assert [r["sample_id"] for r in obs] == COHORTS[f.fragment_key]["samples"]
        # the 'nan' timepoint is a placeholder string, not a missing value of the file
        assert all(r["timepoint"] in ("W0", "W8", "nan") for r in obs)
        universes[f.fragment_key] = set(h5ad.axis_values(f, "var").column("gene_symbol").to_pylist())
        sample_ids = [r["sample_id"] for r in obs]
        geo = reg.get("identifier", "geo_gsm")
        assert all(geo.normalize(s).value == s for s in sample_ids)
    assert "SEPT9" in universes["GSE12251"] and "SEPT9" not in universes["GSE73661"]   # not_measured_in GSE73661
    assert sources["zenodo_vbt"].id_types["cohort_symbol"].authority == "release_snapshot"


def test_ibd_cohorts_long_view_matches_a_dense_oracle(reg, sources, cohorts) -> None:
    spec = sources["zenodo_vbt"].tables["ibd_cohorts"]
    h5ad = reg.get("format", "h5ad").configure(spec.format.options, spec.matrix)
    data = cohorts / "osmr" / "code" / "data"
    for gse, c in COHORTS.items():
        f = frag(data / f"{gse}.h5ad", fragment_key=gse)
        want = sorted(((s, g, None if math.isnan(v) else v) for s, row in zip(c["samples"], c["x"])
                       for g, v in zip(c["genes"], row)), key=lambda t: (t[0], t[1]))
        assert cells(h5ad, f, "X", row_key="sample_id", col_key="gene_symbol") == want
        some = cells(h5ad, f, "X", row_key="sample_id", col_key="gene_symbol",
                     row_predicate=In("timepoint", ("W0",)), col_keys=["OSMR"])
        assert [s for s, _g, _v in some] == [s for s, t in zip(c["samples"], c["obs"]["timepoint"]) if t == "W0"]
        st = h5ad.stats(f)
        assert st.shape == (len(c["samples"]), len(c["genes"])) and st.rows == st.shape[0] * st.shape[1]

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from vbt.analysis import singlecell as scm
from vbt.analysis import spatial


def _toy_obs():
    rows = []
    for donor, cond in [("d1", "ctrl"), ("d2", "ctrl"), ("d3", "ctrl"), ("d4", "case"),
                        ("d5", "case"), ("d6", "case")]:
        rows += [(donor, cond, "Fib")] * 25
        rows += [(donor, cond, "T")] * (25 if donor in {"d1", "d2", "d4", "d5", "d6"} else 5)
        rows += [(donor, cond, "B")] * 10
    return pd.DataFrame(rows, columns=["donor", "condition", "celltype"])


def test_eligible_celltypes():
    obs = _toy_obs()
    elig, table = scm.eligible_celltypes(obs, "celltype", "donor", "condition", return_table=True)
    # Fib: 3 donors per arm; T: only 2 ctrl donors with >=20 cells; B: all < 20 cells
    assert elig == ["Fib"]
    assert table.loc["T", "ctrl"] == 2 and table.loc["T", "case"] == 3
    assert scm.eligible_celltypes(obs, "celltype", "donor", "condition", min_donors=2) == ["Fib", "T"]


@pytest.mark.parametrize("as_sparse", [False, True])
def test_pseudobulk_counts_sums(as_sparse):
    rng = np.random.default_rng(0)
    obs = _toy_obs()
    X = rng.poisson(2, size=(len(obs), 5))
    Xin = sparse.csr_matrix(X) if as_sparse else X
    counts, meta = scm.pseudobulk_counts(Xin, obs, groupby=["donor", "celltype"],
                                         var_names=list("ABCDE"), min_cells=10,
                                         carry_cols=["condition"])
    # groups with 5 cells (d3|T) dropped
    assert "d3|T" not in counts.index and "d1|B" in counts.index
    m = ((obs.donor == "d2") & (obs.celltype == "Fib")).to_numpy()
    np.testing.assert_array_equal(counts.loc["d2|Fib"].to_numpy(), X[m].sum(axis=0))
    assert meta.loc["d2|Fib", "n_cells"] == 25
    assert meta.loc["d4|Fib", "condition"] == "case"
    assert counts.dtypes.iloc[0].kind == "i"


def test_ols_de_finds_planted_gene():
    rng = np.random.default_rng(1)
    n_per, g = 6, 200
    cond = np.array(["ctrl"] * n_per + ["case"] * n_per)
    mu = np.full((2 * n_per, g), 50.0)
    mu[cond == "case", 0] *= 8  # planted up-regulated gene
    mu[cond == "case", 1] /= 8  # planted down-regulated gene
    counts = pd.DataFrame(rng.poisson(mu), index=[f"s{i}" for i in range(2 * n_per)],
                          columns=[f"g{i}" for i in range(g)])
    meta = pd.DataFrame({"condition": cond}, index=counts.index)
    res = scm.pseudobulk_de(counts, meta, "condition", ref_level="ctrl", engine="ols_log_cpm")
    de = res[res.de].set_index("gene")
    assert set(de.index) == {"g0", "g1"}
    assert de.loc["g0", "direction"] == "up" and de.loc["g1", "direction"] == "down"
    assert abs(de.loc["g0", "log2FC"] - 3) < 0.5
    assert res.contrast.iloc[0] == "case_vs_ctrl"


def test_quartile_groups():
    v = pd.Series(np.arange(1, 101, dtype=float), index=[f"c{i}" for i in range(100)])
    v.iloc[5] = np.nan
    lab = scm.quartile_groups(v)
    assert (lab == "high").sum() == 25
    assert lab.loc["c99"] == "high" and lab.loc["c0"] == "low" and lab.loc["c50"] is None
    assert lab.loc["c5"] is None


def test_downsample_preserves_proportions():
    obs = pd.DataFrame({"donor": ["a"] * 1000 + ["b"] * 50,
                        "ct": ["x"] * 700 + ["y"] * 300 + ["x"] * 50})
    idx = scm.downsample_indices(obs, "donor", "ct", max_cells=100, seed=0)
    kept = obs.iloc[idx]
    assert (kept.donor == "a").sum() == 100 and (kept.donor == "b").sum() == 50
    assert ((kept.donor == "a") & (kept.ct == "x")).sum() == 70


def _lr_df():
    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "source": ["Fib"] * n, "target": [f"T{i % 5}" for i in range(n)],
        "ligand_complex": [f"L{i}" for i in range(n)], "receptor_complex": [f"R{i}" for i in range(n)],
        "magnitude_rank": np.linspace(0.001, 1, n),
        "cellphone_pvals": np.where(np.arange(n) % 2 == 0, 0.001, 0.5),
        "cellchat_pvals": 0.01, "lr_logfc": rng.normal(size=n), "lrscore": 0.9,
    })
    df.loc[0, "lrscore"] = 0.1
    df.loc[0, "cellchat_pvals"] = 0.5
    return df


def test_filter_lr_results():
    df = _lr_df()
    out = scm.filter_lr_results(df)
    # top 10% by magnitude_rank => rows 0..9; cpdb p<0.01 => even rows; row 0 lacks support
    assert set(out.ligand_complex) <= {f"L{i}" for i in (2, 4, 6, 8)}
    assert "L0" not in set(out.ligand_complex)
    assert (out.n_methods >= 3).all()
    # expr_prod fallback (higher is better)
    df2 = df.drop(columns="magnitude_rank").assign(expr_prod=np.linspace(1, 0, len(df)), n_methods=5)
    out2 = scm.filter_lr_results(df2)
    assert set(out2.ligand_complex) == {f"L{i}" for i in (0, 2, 4, 6, 8)}


def test_group_specific_interactions():
    a = pd.DataFrame({"source": ["F", "F"], "target": ["T", "M"], "ligand_complex": ["L1", "L2"],
                      "receptor_complex": ["R1", "R2"]})
    b = pd.DataFrame({"source": ["F", "F"], "target": ["T", "B"], "ligand_complex": ["L1", "L3"],
                      "receptor_complex": ["R1", "R3"]})
    res = scm.group_specific_interactions(a, b)
    assert list(res["a_only"].ligand_complex) == ["L2"]
    assert list(res["b_only"].ligand_complex) == ["L3"]
    assert list(res["shared"].ligand_complex) == ["L1"]


def test_knn_rings_sizes_and_sample_separation():
    xs, ys = np.meshgrid(np.arange(8), np.arange(8))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    coords = np.vstack([coords, coords + 1000])
    sids = np.repeat(["s1", "s2"], 64)
    rings = spatial.knn_rings(coords, sample_ids=sids)
    assert rings["1-6"].shape == (128, 6)
    assert rings["7-15"].shape == (128, 9)
    assert rings["16-30"].shape == (128, 15)
    allr = np.hstack(list(rings.values()))
    assert (allr >= 0).all()
    assert (allr[:64] < 64).all() and (allr[64:] >= 64).all()  # neighbours stay within sample
    assert not (allr == np.arange(128)[:, None]).any()  # self excluded
    d1 = np.linalg.norm(coords[rings["1-6"]] - coords[:, None], axis=2).max(axis=1)
    d3 = np.linalg.norm(coords[rings["16-30"]] - coords[:, None], axis=2).min(axis=1)
    assert (d1 <= d3 + 1e-9).all()


def test_neighbor_mean_abundance():
    coords = np.array([[0, 0], [1, 0], [2.5, 0], [10, 0]], float)
    ab = pd.DataFrame({"T": [1.0, 2.0, 3.0, 100.0]})
    m = spatial.neighbor_mean_abundance(ab, coords, np.zeros(4), (1, 1))
    assert m["T"].tolist() == [2.0, 1.0, 2.0, 3.0]
    assert spatial.neighbor_mean_abundance(ab, coords, np.zeros(4), (0, 0)).equals(ab)
    m2 = spatial.neighbor_mean_abundance(ab, coords, np.zeros(4), (1, 2))
    assert m2["T"][0] == pytest.approx(2.5)  # neighbours 1 and 2


def test_immune_neighborhood_recovers_negative_effect():
    rng = np.random.default_rng(3)
    frames = []
    for p in range(5):
        for s in range(2):
            xs, ys = np.meshgrid(np.arange(15), np.arange(15))
            x, y = xs.ravel().astype(float), ys.ravel().astype(float)
            phase = rng.uniform(0, 6)
            field = np.sin(x / 2.5 + phase) + np.cos(y / 3.0 + phase)
            n = x.size
            frames.append(pd.DataFrame({
                "x": x, "y": y, "patient": f"P{p}", "sample": f"P{p}_S{s}",
                "CD276": np.clip(field + 1.5 + rng.normal(0, 0.3, n), 0, None),
                "CD8T": np.clip(3 - 1.0 * field + rng.normal(0, 0.3, n) + 0.3 * p, 0.01, None),
                "Bcell": np.clip(2 + rng.normal(0, 0.3, n), 0.01, None),
                "total_umi": rng.normal(5000, 500, n), "fibroblast": rng.uniform(0, 5, n),
                "epithelial": rng.uniform(0, 5, n), "endothelial": rng.uniform(0, 2, n),
            }))
    spots = pd.concat(frames, ignore_index=True)
    res = spatial.immune_neighborhood_analysis(spots, "CD276", ["CD8T", "Bcell"],
                                               rings=[(1, 6), (7, 15)], n_boot=200)
    assert len(res) == 4
    r = res.set_index(["immune_type", "ring"])
    assert r.loc[("CD8T", "1-6"), "beta_high"] < 0
    assert r.loc[("CD8T", "1-6"), "p"] < 1e-3
    assert r.loc[("CD8T", "1-6"), "pct_change"] < 0
    assert r.loc[("CD8T", "1-6"), "pct_ci_high"] < 0
    assert r.loc[("Bcell", "1-6"), "p"] > 1e-3
    assert r.loc[("CD8T", "1-6"), "n_samples"] == 10 and r.loc[("CD8T", "1-6"), "n_patients"] == 5

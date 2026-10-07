"""The in-process read client ``vbt.datalayer.client`` (§10.6, F14) and the readers moved onto it.

* Every client call goes through the data child's verbs (resolution, readiness, limits) and writes a
  ``vbt.dataprov/1`` record under the run (``run_dir`` / ``VBT_RUN_DIR``) or a standalone log dir; the
  record id is returned and names the resolutions, tables and counts. Typed errors are raised.
* ``read_frame`` is a backed read: the frame equals ``pandas.read_parquet`` of the same files.
* The Case 1 readers (``pipeline.covariates_from_open_targets``, ``pipeline.genetic_pairs_from_open_targets``,
  ``replicate.load_inputs``) and ``biomarker.load_processed_cohort`` give identical outputs on fixtures
  through the client, with provenance recorded; a cohort file no descriptor declares is opened
  unguarded and its record says so.
* ``Run.register_artifact(derived_from=...)`` records the inputs (found or not in this run), and the
  ``mcp__provenance__register_artifact`` tool passes them through.
* Bash commands run under the reaper with ``RLIMIT_DATA = data.memory.workspace_mb`` (Linux).
* The ``child`` backend starts the real data child through an ``MCPBridge``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, needs_arrow

pytestmark = needs_arrow


def _config(tmp: Path) -> dict[str, Any]:
    return {"vars": {"project_root": str(REPO)}, "data": {"cache_dir": str(tmp / "dl-cache")}}


@pytest.fixture
def ot_env(ot_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot_root))
    monkeypatch.delenv("VBT_AGENT", raising=False)
    monkeypatch.delenv("VBT_RUN_DIR", raising=False)
    return ot_root


@pytest.fixture
def client(ot_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from vbt.datalayer import client as C

    monkeypatch.setenv(C.ENV_BACKEND, "inprocess")
    c = C.configure(_config(tmp_path), run_dir=tmp_path / "run")
    yield c
    C.close()


def _record(c: Any, prov: str) -> dict[str, Any]:
    path = c.log_dir / f"{prov}.json"
    assert path.is_file(), f"no provenance record {path}"
    return json.loads(path.read_text())


def test_client_calls_write_provenance_under_the_run(client, tmp_path) -> None:
    import dl_fixtures as F

    res = client.find("open_targets.known_drug", where={"targetId": "PCSK9"}, limit=3)
    assert len(res.rows) == 3 and res.header["total"] == 37 and res.prov.startswith("dp_")
    assert client.log_dir == tmp_path / "run" / "logs" / "data_provenance" / "client"
    rec = _record(client, res.prov)
    assert rec["schema"] == "vbt.dataprov/1" and rec["tool"] == "client.find" and rec["id"] == res.prov
    assert rec["source"]["name"] == "open_targets" and rec["source"]["release"] == "25.09"
    assert rec["tables"][0]["name"] == "open_targets.known_drug"
    assert rec["request"]["resolutions"][0] == {
        "arg": "where.targetId", "raw": "PCSK9", "canonical": F.PCSK9, "matched_id_type": None,
        "canonical_id_type": None, "rule": "label_exact:approvedSymbol", "family": None, "hops": [],
        "index_fingerprint": None, "existence": None}
    assert rec["result"]["total"] == 37 and rec["result"]["returned"] == 3 and rec["result"]["row_keys_sha256"]
    assert len(rec["result"]["row_keys"]) == 3
    agg = client.aggregate("open_targets.known_drug", ["phase"], where={"targetId": F.PCSK9})
    assert sum(r["count"] for r in agg.rows) == 37 and agg.prov in client.records


def test_client_raises_typed_errors(client) -> None:
    import dl_fixtures as F
    from vbt.datalayer.errors import ErrorKind, GatewayError

    with pytest.raises(GatewayError) as exc:
        client.find("open_targets.known_drug", where={"targetId": F.UNKNOWN_GENE})
    assert exc.value.kind is ErrorKind.not_found and exc.value.payload.get("tried")
    with pytest.raises(GatewayError) as exc:
        client.find("open_targets.known_drug", limit=0)
    assert exc.value.kind is ErrorKind.invalid_argument


def test_agents_get_the_native_exposure_rules(ot_env, tmp_path, monkeypatch) -> None:
    from vbt.datalayer import client as C
    from vbt.datalayer.errors import GatewayError

    zen = tmp_path / "zen" / "virtualbiotech_submission" / "clinical_trials" / "data"
    zen.mkdir(parents=True)
    (zen / "clinical_trial_labels_reconciled.csv").write_text("nct_id\nNCT00000001\n")
    monkeypatch.setenv("VBT_AGENT", "trial-annotator")
    c = C.DataClient(_config(tmp_path), backend="inprocess", log_dir=tmp_path / "log")
    with pytest.raises(GatewayError) as exc:
        c.read_frame("zenodo_vbt.clinical_trial_labels", root=zen.parents[1])
    assert exc.value.subkind in ("withheld", "not_exposed")


def test_read_frame_is_a_backed_read_equal_to_pandas(client, ot_env) -> None:
    import pandas as pd

    for table, cols in (("drug_molecule", ["id", "drugType"]), ("disease", ["id", "name", "therapeuticAreas"])):
        got = client.read_frame(f"open_targets.{table}", cols, root=ot_env)
        want = pd.read_parquet(ot_env / table, columns=cols)
        pd.testing.assert_frame_equal(got, want)
        rec = _record(client, got.attrs["vbt_prov"])
        assert rec["tool"] == "client.read_frame" and rec["result"]["output_sha256"]
        assert rec["tables"][0]["fingerprint"].startswith("fp1:")


# ---------------------------------------------------------------------------- the Case 1 readers


def _mapping(ot_root: Path) -> Any:
    import pandas as pd

    import dl_fixtures as F

    kd = F.read_rows(ot_root, "known_drug")
    rows = [{"nct_id": f"NCT{i:08d}", "drugId": r["drugId"], "diseaseId": r["diseaseId"], "targetId": r["targetId"]}
            for i, r in enumerate(kd[:12])]
    return pd.DataFrame(rows)


def test_pipeline_readers_are_identical_through_the_client(client, ot_env) -> None:
    import pandas as pd
    import pyarrow.dataset as ds

    from vbt.case_studies.trial_outcomes import pipeline as P
    from vbt.case_studies.trial_outcomes import stats as st

    mapping = _mapping(ot_env)
    # the readers' former direct reads, inline: the oracle
    drugs = pd.read_parquet(ot_env / "drug_molecule", columns=["id", "drugType"]).set_index("id")
    dis = pd.read_parquet(ot_env / "disease", columns=["id", "name", "therapeuticAreas"])
    want = (st.drugtype_combo(mapping, drugs["drugType"]).rename(columns={"drugType_combo": "modality"})
            .merge(st.ta_combo(mapping, dis).rename(columns={"ta_combo": "therapeutic_area"}), on="nct_id"))
    before = len(client.records)
    pd.testing.assert_frame_equal(P.covariates_from_open_targets(ot_env, mapping), want)
    assert len(client.records) == before + 2, "one provenance record per table read"
    methods = P.covariates_from_open_targets(ot_env, mapping, definitions="methods")
    assert list(methods.columns) == ["nct_id", "modality", "therapeutic_area"] and len(methods) == mapping.nct_id.nunique()
    d = ds.dataset(ot_env / "association_by_datatype_direct", format="parquet")
    t = d.to_table(columns=["targetId", "diseaseId"], filter=ds.field("datatypeId") == "genetic_association")
    assert P.genetic_pairs_from_open_targets(ot_env) == set(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))


def _archive(ot_root: Path, tmp: Path) -> Path:
    """A Zenodo-shaped archive with the OT subsets copied from the OT fixture."""
    import pandas as pd

    root = tmp / "zenodo" / "virtualbiotech_submission"
    data = root / "clinical_trials" / "data"
    data.mkdir(parents=True)
    for table in ("association_by_datatype_direct", "drug_molecule", "disease"):
        shutil.copytree(ot_root / table, data / table)
    mapping = _mapping(ot_root)
    mapping.to_parquet(data / "chembl_clinical_nct_data.parquet")
    pd.DataFrame({"nct_id": mapping.nct_id}).to_csv(data / "clinical_trial_labels_reconciled.csv", index=False)
    pd.DataFrame({"targetId": mapping.targetId.unique(), "tau_cell_type": 0.5}).to_parquet(
        data / "comprehensive_features_aggregated_v2_optimized.parquet")
    return root


def test_replicate_load_inputs_is_identical_through_the_client(client, ot_env, tmp_path) -> None:
    import pandas as pd

    from vbt.case_studies.trial_outcomes import replicate as R

    root = _archive(ot_env, tmp_path)
    d = root / "clinical_trials" / "data"
    before = len(client.records)
    got = R.load_inputs(root)
    assert len(client.records) == before + 3
    g = pd.read_parquet(d / "association_by_datatype_direct", columns=["targetId", "diseaseId", "datatypeId", "score"])
    pd.testing.assert_frame_equal(got["genetic"], g[g["datatypeId"] == "genetic_association"][
        ["targetId", "diseaseId", "score"]])
    pd.testing.assert_series_equal(got["drug_types"], pd.read_parquet(d / "drug_molecule", columns=[
        "id", "drugType"]).set_index("id")["drugType"])
    pd.testing.assert_frame_equal(got["disease"], pd.read_parquet(d / "disease", columns=[
        "id", "name", "therapeuticAreas"]))
    rec = _record(client, got["disease"].attrs["vbt_prov"])
    assert rec["tables"][0]["name"] == "zenodo_vbt.disease"
    assert R.load_inputs(root / "clinical_trials")["disease"].equals(got["disease"])


def _cohort(path: Path) -> None:
    import anndata as ad
    import numpy as np
    import pandas as pd

    obs = pd.DataFrame({"disease": ["UC"] * 4 + ["CD"], "drug": ["Infliximab"] * 5,
                        "response_clinical": ["R", "NR", "NR", "R", "R"], "is_baseline": [True] * 5,
                        "patient_id": [f"P{i}" for i in range(5)], "timepoint": ["W0"] * 5},
                       index=pd.Index([f"GSM{i}" for i in range(5)], name="sample_id"))
    var = pd.DataFrame(index=pd.Index(["OSMR", "IL6ST", "STAT1"], name="gene_symbol"))
    X = np.arange(15, dtype=float).reshape(5, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=X, obs=obs, var=var).write_h5ad(path)


def test_load_processed_cohort_through_the_client(client, tmp_path) -> None:
    pytest.importorskip("anndata")
    import pandas as pd

    from vbt.analysis import biomarker

    archive = tmp_path / "zenodo" / "virtualbiotech_submission"
    path = archive / "osmr" / "code" / "data" / "GSE12251.h5ad"
    _cohort(path)
    expr, labels = biomarker.load_processed_cohort(path)
    assert list(expr.index) == ["GSM0", "GSM1", "GSM2", "GSM3"] and list(labels) == [0, 1, 1, 0]
    assert expr.loc["GSM1", "IL6ST"] == 4.0
    rec = _record(client, expr.attrs["vbt_prov"])
    assert rec["tool"] == "client.open_matrix" and rec["served_by"] == "derived"
    assert rec["tables"][0]["name"] == "zenodo_vbt.ibd_cohorts"
    # the same file outside any declared layout: opened unguarded, and the record says so
    loose = tmp_path / "loose" / "cohort.h5ad"
    _cohort(loose)
    expr2, labels2 = biomarker.load_processed_cohort(loose)
    pd.testing.assert_frame_equal(expr2, expr)
    pd.testing.assert_series_equal(labels2, labels)
    rec2 = _record(client, expr2.attrs["vbt_prov"])
    assert rec2["served_by"] == "unguarded" and rec2["tables"][0]["fingerprint"].startswith("sha256:")


def test_off_backend_reads_directly_and_records_it(ot_env, tmp_path) -> None:
    import pandas as pd

    from vbt.datalayer import client as C
    from vbt.datalayer.errors import GatewayError

    c = C.DataClient(_config(tmp_path), backend="off", log_dir=tmp_path / "log")
    got = c.read_frame("open_targets.drug_molecule", ["id", "drugType"], root=ot_env)
    pd.testing.assert_frame_equal(got, pd.read_parquet(ot_env / "drug_molecule", columns=["id", "drugType"]))
    assert json.loads((tmp_path / "log" / f"{got.attrs['vbt_prov']}.json").read_text())["served_by"] == "unguarded"
    with pytest.raises(GatewayError):
        c.find("open_targets.target")


# ---------------------------------------------------------------------------- artifacts and Bash


def test_register_artifact_records_derived_from(client, tmp_path) -> None:
    from vbt.session import Run
    from vbt.tools.base import Tool  # noqa: F401 - the provenance tools build on it
    from vbt.tools.provenance import provenance_tools

    run = Run(tmp_path / "runs", run_id="DF")
    c = client.__class__(_config(tmp_path), backend="inprocess", run_dir=run.dir)
    res = c.find("open_targets.target", where={"id": "PCSK9"})
    rel = "work/target-biologist/results/pcsk9.csv"
    (run.dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (run.dir / rel).write_text("id\nENSG00000169174\n")
    r = run.register_artifact(rel, "PCSK9 record", "target-biologist", derived_from=[res.prov, "toolu_123"])
    assert r["ok"], r
    assert r["artifact"]["derived_from"] == [{"id": res.prov, "kind": "data_provenance", "found": True},
                                             {"id": "toolu_123", "kind": "tool_use", "found": False}]
    assert "toolu_123" in r["warnings"][0]
    entry = run.manifest["artifacts"][rel]
    assert entry["derived_from"][0]["id"] == res.prov
    assert run.register_artifact(rel, "x", "cso", derived_from=["../etc"])["ok"] is False
    assert run.register_artifact(rel, "x", "cso", derived_from=7)["ok"] is False
    tool = next(t for t in provenance_tools() if t.name == "mcp__provenance__register_artifact")
    assert "derived_from" in tool.input_schema["properties"]
    run.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_DATA under the reaper is Linux-only")
def test_bash_runs_under_the_workspace_memory_limit(tmp_path) -> None:
    from vbt.tools.builtin import WORKSPACE_MB, _strip_exit_marker, _workspace_limit

    probe = [sys.executable, "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_DATA)[0])"]
    limit, argv, status = _workspace_limit({"data": {"memory": {"workspace_mb": 3000}}}, probe, tmp_path, "tu_1")
    assert limit == 3000 and argv is not None and status is not None
    out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == 3000 * 1024 * 1024
    text, reason = _strip_exit_marker(out.stdout + out.stderr)
    assert "VBT_CHILD_EXIT" not in text and reason == "exit_code"
    assert json.loads(status.read_text())["limit_mb"] == 3000
    assert _workspace_limit({}, probe, tmp_path, "tu_2")[0] == WORKSPACE_MB == 8000
    assert _workspace_limit({"data": {"memory": {"workspace_mb": 0}}}, probe, tmp_path, "tu_3") == (None, None, None)
    assert _workspace_limit({"data": {"enabled": False}}, probe, tmp_path, "tu_4") == (None, None, None)


def test_workspace_limit_follows_limit_kind(tmp_path) -> None:
    """``data.memory.limit_kind`` applies to the workspace as to the MCP servers: none lifts the limit,
    cgroup and watchdog reach the reaper as ``--containment`` (INV-6)."""
    from vbt.tools.builtin import _workspace_limit

    if not sys.platform.startswith("linux"):
        pytest.skip("the reaper is Linux-only")
    probe = ["true"]
    cfg = lambda kind: {"data": {"memory": {"workspace_mb": 3000, "limit_kind": kind}}}  # noqa: E731
    assert _workspace_limit(cfg("none"), probe, tmp_path, "k0") == (None, None, None)
    for i, kind in enumerate(("cgroup", "watchdog")):
        argv = _workspace_limit(cfg(kind), probe, tmp_path, f"k{i + 1}")[1]
        assert argv is not None and argv[argv.index("--containment") + 1] == kind
        assert argv.index("--containment") < argv.index("--")
    argv = _workspace_limit(cfg("rlimit_data"), probe, tmp_path, "k3")[1]
    assert argv is not None and "--containment" not in argv


# ---------------------------------------------------------------------------- the child backend


def test_child_backend_starts_the_data_child_through_a_bridge(ot_env, tmp_path) -> None:
    pytest.importorskip("fastmcp")
    import dl_fixtures as F
    from vbt.datalayer import client as C

    from vbt.config import load_config

    config = load_config()                             # the tool env passes OPEN_TARGETS_DATA_PATH to the child
    config.setdefault("data", {})["cache_dir"] = str(tmp_path / "dl-cache")
    c = C.DataClient(config, backend="child", log_dir=tmp_path / "log")
    try:
        res = c.find("open_targets.known_drug", where={"targetId": "PCSK9"}, limit=2)
        assert res.header["total"] == 37 and len(res.rows) == 2
        assert res.rows[0]["targetId"] == F.PCSK9 and (tmp_path / "log" / f"{res.prov}.json").is_file()
        frame = c.read_frame("open_targets.drug_molecule", ["id", "drugType"])
        assert len(frame) == len(F.read_rows(ot_env, "drug_molecule"))
    finally:
        c.close()

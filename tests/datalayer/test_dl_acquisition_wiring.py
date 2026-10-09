"""How acquisition is wired into the rest of the layer (Wave B integration of `vbt data acquire` and `vbt setup`):

* a ``not_ready`` refusal carries the structured ``acquire`` entry, and the gateway's readiness cache knows the
  ``data.acquisition`` policy (typed in ``DataSettings.acquisition``; ``configs/default.yaml`` mirrors it);
* the download manifests ``vbt data acquire`` writes are declared in the descriptors, so R2 checks them (sizes,
  completeness, release), including a manifest kept above the descriptor's root (the Zenodo archive);
* the preflight catalog cache follows the variables the descriptors expand;
* ``vbt ds conformance``, ``vbt ds estimate --json``, ``vbt ds index build --table`` and an empty
  ``vbt data acquire --plan --json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vbt import cli
from vbt.datalayer import cli as ds
from vbt.datalayer.errors import ErrorKind, GatewayError, not_ready_payload
from vbt.datalayer.settings import DATA_DEFAULTS, AcquisitionSettings, DataSettings

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------- refusal payload and policy


def test_not_ready_payload_keeps_the_structured_acquire_entry():
    acquire = {"command": "vbt data acquire open_targets.go", "source": "open_targets", "table": "open_targets.go",
               "release": "25.09", "bytes": 3_161_205, "files": 2, "prepare": [], "mode": "download",
               "licence": "CC0 1.0", "policy": "off", "decision": "an operator runs it"}
    payload = not_ready_payload([
        {"name": "open_targets.go", "check": "missing", "detail": "no files", "hint": "acquire them", "acquire": acquire},
        {"name": "open_targets.target", "column": "id", "check": "R4", "detail": "drift", "hint": "x"},
    ])
    assert payload["tables"][0]["acquire"] == acquire
    assert "acquire" not in payload["tables"][1] and payload["tables"][1]["column"] == "id"
    env = GatewayError(ErrorKind.not_ready, "open_targets.go is not ready", payload=payload).envelope()
    assert json.loads(json.dumps(env, default=str))["tables"][0]["acquire"]["bytes"] == 3_161_205


def test_acquisition_settings_are_typed_and_mirrored_in_default_yaml():
    s = DataSettings.from_dict({})
    assert s.acquisition.auto == "off" and s.acquisition.workers == "auto" and s.acquisition.root.endswith("sources")
    assert AcquisitionSettings().root == "${VBT_DATA_DIR:-data}/sources"      # expanded by DataSettings
    assert s.acquisition.policy() == {"auto": "off", "budget_bytes": 0}
    # YAML 1.1 reads an unquoted `off` as false
    assert DataSettings.from_dict({"acquisition": {"auto": False}}).acquisition.auto == "off"
    s = DataSettings.from_dict({"acquisition": {"auto": "under_budget", "budget_bytes": "5 GB"}})
    assert s.acquisition.policy() == {"auto": "under_budget", "budget_bytes": "5 GB"}
    assert DataSettings.from_json(s.to_json()).acquisition == s.acquisition       # reaches the data child
    with pytest.raises(ValueError, match="off, ask or under_budget"):
        DataSettings.from_dict({"acquisition": {"auto": True}})
    shipped = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())["data"]["acquisition"]
    assert shipped == DATA_DEFAULTS["acquisition"]


def test_the_acquisition_log_lives_in_the_acquisition_root(tmp_path, monkeypatch):
    """``data.provenance.dir`` is relative to a run; the host-wide log goes next to the homes it describes."""
    from vbt.data.acquire import AcquisitionSettings as Resolved

    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path / "d"))
    st = Resolved.from_config({"vars": {"project_root": str(tmp_path)}})
    assert st.provenance_dir == st.root == tmp_path / "d" / "sources"


def test_the_gateway_and_preflight_readiness_caches_carry_the_policy(monkeypatch, tmp_path):
    from vbt.datalayer.gateway.gateway import DataGateway
    from vbt.preflight import data_catalog

    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    config = {"data": {"acquisition": {"auto": "ask"}}}
    settings, catalog, registry = data_catalog(config)
    gw = DataGateway(settings, catalog, registry)
    assert gw.readiness.acquisition == {"auto": "ask", "budget_bytes": 0}
    from vbt.datalayer.gateway.readiness import acquisition_hint

    hint = acquisition_hint(catalog, "open_targets.go", gw.readiness.acquisition)
    assert hint["policy"] == "ask" and hint["command"] == "vbt data acquire open_targets.go"


# ---------------------------------------------------------------------------- download manifests (R2)


def test_a_manifest_above_the_root_is_rebased_on_the_root(tmp_path):
    from vbt.datalayer.descriptor.models import ManifestSpec
    from vbt.datalayer.service import load_manifest

    home = tmp_path / "home"
    (home / "archive" / "a").mkdir(parents=True)
    (home / ".download-manifest.json").write_text(json.dumps({"complete": True, "release": "1", "files": {
        "archive/a/x.parquet": {"bytes": 3}, "archive/b.csv": {"bytes": 4}, "other/y.txt": {"bytes": 5}}}))
    m = load_manifest(str(home / "archive"), ManifestSpec(path="../.download-manifest.json"))
    assert sorted(m.entries) == ["a/x.parquet", "b.csv"] and m.complete is True
    same = load_manifest(str(home), ManifestSpec(path=".download-manifest.json"))
    assert sorted(same.entries) == ["archive/a/x.parquet", "archive/b.csv", "other/y.txt"]


def test_shipped_descriptors_declare_the_acquisition_manifests():
    from vbt.datalayer.descriptor.load import load_descriptors

    descs = load_descriptors(REPO / "configs" / "data" / "sources", {"project_root": str(REPO)})
    for source in ("open_targets", "depmap", "gene_ontology", "msigdb", "zenodo_vbt"):
        acq = descs[source].acquisition
        paths = [m.path for m in descs[source].manifests]
        assert any(p and p.endswith(acq.manifest) for p in paths), source
        assert all(m.require == {"complete": True} and not m.required for m in descs[source].manifests)
    assert "manifest.release" in descs["depmap"].release.from_


@pytest.mark.skipif(not __import__("importlib").util.find_spec("pyarrow"), reason="pyarrow not installed")
def test_r2_reads_the_msigdb_download_manifest(tmp_path, monkeypatch):
    from vbt.datalayer.service import ServiceContext
    from vbt.datalayer.service.checks import check_table

    home = tmp_path / "msigdb"
    home.mkdir()
    gmt = home / "h.all.v2024.1.Hs.symbols.gmt"
    gmt.write_text("HALLMARK_APOPTOSIS\thttp://x\tPCSK9\tTP53\nHALLMARK_HYPOXIA\thttp://y\tBRCA1\n")
    monkeypatch.setenv("MSIGDB_DATA_PATH", str(home))
    settings = DataSettings.from_dict({"cache_dir": str(tmp_path / "cache")}, project_root=REPO)
    ctx = ServiceContext(settings)

    def r2(**manifest):
        path = home / ".download-manifest.json"
        if manifest:
            path.write_text(json.dumps({"release": "2024.1.Hs", "complete": True, **manifest}))
        elif path.exists():
            path.unlink()
        model = check_table(ctx, "msigdb.hallmark", "shallow")
        return {c.name: (c.ok, c.level) for c in model.checks if c.name.startswith("R2")}

    found = r2()
    assert found["R2:manifest_absent"] == (False, "warning")
    assert r2(files={gmt.name: {"bytes": gmt.stat().st_size}})["R2:manifest"][0] is True
    bad = r2(files={gmt.name: {"bytes": 7}})
    assert bad["R2:manifest_bytes"][0] is False
    stale = r2(release="2023.2.Hs", files={gmt.name: {"bytes": gmt.stat().st_size}})
    assert stale["R2:release"][0] is False
    partial = r2(complete=False, files={gmt.name: {"bytes": gmt.stat().st_size}})
    assert partial["R2:manifest_require"][0] is False


# ---------------------------------------------------------------------------- catalog cache


def test_the_preflight_catalog_follows_the_variables_descriptors_expand(monkeypatch, tmp_path):
    from vbt.preflight import data_catalog

    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VBT_CL_OBO", str(tmp_path / "a.obo"))
    first = data_catalog({})[1]
    assert data_catalog({})[1] is first                         # unchanged: the same catalog
    monkeypatch.setenv("VBT_CL_OBO", str(tmp_path / "b.obo"))
    second = data_catalog({})[1]
    assert second is not first
    assert str(second.table("cell_ontology.term").spec.path).endswith("b.obo")


# ---------------------------------------------------------------------------- commands


def test_conformance_command_covers_both_registries(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    args = cli.build_parser().parse_args(["ds", "conformance", "--list", "--json"])
    assert args.handler is ds.cmd_conformance
    assert cli.main(["--profile", "mock", "ds", "conformance", "--list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)["plugins"]
    kinds = {r["kind"] for r in rows}
    assert {"format", "layout", "identifier", "statistic", "envelope", "acquisition"} <= kinds
    assert {"acquisition/http", "acquisition/huggingface", "layout/zip_member"} <= {r["plugin"] for r in rows}
    assert all(r["stamp"] == "none" for r in rows)
    assert cli.main(["--profile", "mock", "ds", "conformance", "--plugin", "acquisition:json_index", "--json"]) == 0
    out = capsys.readouterr().out
    doc = json.loads(out[out.rindex("\n{") + 1:] if "\n{" in out else out)
    assert doc["kinds"] == {"acquisition": 0} and doc["stamps"][0].endswith("acquisition.json_index.json")
    assert cli.main(["--profile", "mock", "ds", "conformance", "--list", "--plugin", "json_index", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["plugins"][0]["stamp"] == "valid"
    assert cli.main(["--profile", "mock", "ds", "conformance", "--kind", "nope"]) == 2
    assert cli.main(["--profile", "mock", "ds", "conformance", "--plugin", "no_such_plugin"]) == 2


def test_index_build_with_tables_builds_only_their_id_types(monkeypatch, tmp_path):
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    seen: list[tuple[str, ...]] = []

    def fake_child(config, *args, timeout=3600.0):
        seen.append(args)
        return 0, json.dumps({"built": [], "errors": {}}), ""

    monkeypatch.setattr(ds, "_run_child", fake_child)
    assert cli.main(["--profile", "mock", "ds", "index", "build", "--table", "cell_ontology.term"]) == 0
    ids = [seen[-1][i + 1] for i, a in enumerate(seen[-1]) if a == "--id-type"]
    assert ids == ["cell_ontology:cell_ontology"]
    assert cli.main(["--profile", "mock", "ds", "index", "build"]) == 0
    every = [seen[-1][i + 1] for i, a in enumerate(seen[-1]) if a == "--id-type"]
    assert len(every) > 5 and "open_targets:ensembl_gene" in every and "cell_ontology:cell_ontology" in every
    assert cli.main(["--profile", "mock", "ds", "index", "build", "--table", "open_targets.nosuch"]) == 2


def test_estimate_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    stats = {"tables": {"open_targets.go": {"fingerprint": "fp1:x", "rows": 48_000, "fragments": 1,
                                            "bytes_on_disk": 4_000_000}},
             "table_errors": {"open_targets.target": "no files"}}
    monkeypatch.setattr(ds, "_table_stats", lambda config, tables: stats)
    rc = cli.main(["--profile", "mock", "ds", "estimate", "--json", "--table", "open_targets.go", "--table",
                   "open_targets.target"])
    doc = json.loads(capsys.readouterr().out)
    assert rc == 1 and doc["errors"] == {"open_targets.target": "no files"}
    go = doc["tables"]["open_targets.go"]
    assert go["rows"] == 48_000 and go["upstream_mb"] >= 0 and go["full_load"] is True
    from vbt.setup.steps import parse_estimate

    sizes, errors = parse_estimate(json.dumps(doc))
    assert sizes == {"open_targets.go": go["upstream_mb"]} and "open_targets.target" in errors


def test_an_empty_acquisition_plan_is_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VBT_DATA_DIR", str(tmp_path))
    assert cli.main(["--profile", "mock", "data", "acquire", "--pending", "--plan", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["bytes_remaining"] == 0 and doc["sources"] == [] and doc["root"] == str(tmp_path / "sources")
    from vbt.setup.steps import _bytes_to_fetch, _plan_pending

    assert _bytes_to_fetch(doc) == 0 and _plan_pending(doc) == []


def test_host_env_is_ignored_in_the_test_suite():
    assert os.environ.get(cli.NO_HOST_ENV)
    assert cli.apply_host_config(SimpleNamespace(cmd="chat", profile=[])) is None

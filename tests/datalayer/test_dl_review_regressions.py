"""Regressions for the review's ``RR`` findings (behaviour that changed since 59900de, or a gap the suite missed);
all offline.

* RR-1: the reaper keeps a caller's merged stdout/stderr in order (the Bash tool merges them).
* RR-2: a tool bound through ``same_as`` in a quarantined overlay is quarantined, not generically guarded.
* RR-3: a ``limit_kind`` typo never launches a server without containment; doctor and ``ds lint`` name it.
* RR-5: ``vbt ds check --tool`` and ``ds explain`` report a quarantined tool as quarantined.
* RR-6: the data child of an offline test run reads the live sources at a dead loopback port.
* RR-7: an OOM kill elsewhere on the host is not the child's memory exit when the kernel log is unreadable.
* RR-8: ``VBT_DL_REAL_DATA`` names the shared root or the OT 25.09 directory alike for every real-data module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import REPO, harness_config, offline_bases, real_data_root, real_ot_dir
from test_dl_hardening import _build, _catalog_copy, linux_only, reaper
from vbt.datalayer.launch import REAPER
from vbt.datalayer.memory import crash

OVERLAYS = REPO / "configs" / "data" / "overlays"


# ---------------------------------------------------------------------------- RR-1


def _merged(tmp_path: Path, code: str, limit_kind: str | None = None) -> str:
    """The reaper's output as the Bash tool reads it: one pipe for stdout and stderr (``stderr=STDOUT``)."""
    from vbt.tools.builtin import _workspace_limit

    memory: dict[str, Any] = {"workspace_mb": 2000, **({"limit_kind": limit_kind} if limit_kind else {})}
    _limit, argv, _status = _workspace_limit({"data": {"memory": memory}},
                                             [sys.executable, "-c", code], tmp_path, "tu_rr1")
    assert argv is not None and str(REAPER) in argv
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    return proc.stdout.decode()


@linux_only
def test_merged_output_keeps_its_order_under_the_reaper(tmp_path: Path) -> None:
    """RR-1: stderr went through a tee thread while stdout went straight to the pipe: every stderr line came after
    every stdout line (``step 1..step 5`` and then the traceback of step 3)."""
    code = ("import sys\nfor i in range(10):\n    print(f'out {i}', flush=True)\n"
            "    print(f'err {i}', file=sys.stderr, flush=True)\n")
    lines = [ln for ln in _merged(tmp_path, code).splitlines() if not ln.startswith("VBT_CHILD_EXIT")]
    assert lines == [f"{k} {i}" for i in range(10) for k in ("out", "err")]


@linux_only
def test_merged_output_never_splits_a_line(tmp_path: Path) -> None:
    """RR-1: 64 KiB tee writes are not atomic on a full pipe; stdout lines landed inside them."""
    code = ("import sys\nfor i in range(4000):\n    sys.stdout.write('O' * 99 + '\\n')\n"
            "    sys.stderr.write('E' * 99 + '\\n')\n")
    lines = [ln for ln in _merged(tmp_path, code).splitlines() if not ln.startswith("VBT_CHILD_EXIT")]
    assert len(lines) == 8000 and all(ln in ("O" * 99, "E" * 99) for ln in lines)


@linux_only
def test_merged_output_still_names_a_memory_exit(tmp_path: Path) -> None:
    # under RLIMIT_DATA the allocation raises MemoryError (the shipped containment is rss since RR-1 of the third
    # review, where an untouched calloc is not resident: the address-space limit is the one this output names)
    out = _merged(tmp_path, "print('step 1', flush=True)\nb = bytearray(4000 * 2**20)\n", "rlimit_data")
    marker = crash.parse_exit_marker(out)
    assert out.index("step 1") < out.index("MemoryError") < out.index("VBT_CHILD_EXIT")
    assert marker["reason"] == "memory_limit" and marker["cause"] == "memory_error"


def test_merged_output_is_detected_on_one_pipe_only(tmp_path: Path) -> None:
    import os

    r, w = os.pipe()
    r2, w2 = os.pipe()
    f = (tmp_path / "log").open("w")
    try:
        assert reaper.merged_output(w, w) and not reaper.merged_output(w, w2)
        assert not reaper.merged_output(w, f.fileno()) and reaper.merged_output(f.fileno(), f.fileno())
    finally:
        f.close()
        for fd in (r, w, r2, w2):
            os.close(fd)


# ---------------------------------------------------------------------------- RR-2


def _broken_drug() -> str:
    return (OVERLAYS / "drug.yaml").read_text() + "\n  oops: [unclosed\n"


def test_a_same_as_binding_in_a_quarantined_overlay_quarantines_the_tool(tmp_path: Path) -> None:
    """RR-2: target.get_pharmacogenomics is bound in drug.yaml (same_as); with drug.yaml quarantined it fell back to
    the generic guard and was forwarded upstream."""
    src, ov = _catalog_copy(tmp_path, overlays={"drug.yaml": _broken_drug()})
    cat = _build(src, ov)
    (q,) = cat.quarantined
    assert q.kind == "overlay" and q.name == "drug" and "target.get_pharmacogenomics" in (q.same_as or ())
    c = cat.contract("target", "get_pharmacogenomics")
    assert c.binding is None and not c.generic and c.quarantined == (q,)
    assert "target.get_pharmacogenomics" in cat.quarantined_tools()
    assert cat.contract("target", "get_target_info").quarantined == ()        # its own binding is untouched
    # the reverse: drug.get_target_tractability is bound in target.yaml
    src, ov = _catalog_copy(tmp_path / "t", overlays={
        "target.yaml": (OVERLAYS / "target.yaml").read_text() + "\n  oops: [unclosed\n"})
    cat = _build(src, ov)
    assert cat.contract("drug", "get_target_tractability").quarantined
    assert cat.contract("drug", "search_known_drugs").quarantined == ()


def test_unreadable_same_as_lines_may_bind_any_unbound_tool_of_a_reviewed_server(tmp_path: Path) -> None:
    broken = "server: drug\ntools:\n  get_pharmacogenomics:\n    same_as: [target.get_pharmacogenomics\n"
    src, ov = _catalog_copy(tmp_path, overlays={"drug.yaml": broken})
    cat = _build(src, ov)
    (q,) = cat.quarantined
    assert q.same_as is None
    assert cat.contract("target", "get_pharmacogenomics").quarantined == (q,)
    assert cat.contract("nosuchserver", "tool").quarantined == ()       # no reviewed overlay: the generic guard


async def test_the_gateway_refuses_a_same_as_tool_of_a_quarantined_overlay(tmp_path: Path) -> None:
    from vbt.datalayer.errors import ErrorKind, GatewayError
    from vbt.datalayer.gateway import DataGateway

    from vbt.datalayer.catalog import build_catalog

    src, ov = _catalog_copy(tmp_path, overlays={"drug.yaml": _broken_drug()})
    settings = _settings(src, ov, tmp_path)
    cat = build_catalog(settings)
    gw = DataGateway(settings, cat, cat.registry)
    with pytest.raises(GatewayError) as e:
        await gw.prepare("target", "get_pharmacogenomics", {"target_id": "CYP2D6"}, None)
    assert e.value.kind == ErrorKind.quarantined and "drug.yaml" in e.value.message


def _settings(src: Path, ov: Path, tmp_path: Path) -> Any:
    from vbt.datalayer.settings import DataSettings

    return DataSettings.from_dict({"descriptors_dir": str(src), "overlays_dir": str(ov),
                                   "cache_dir": str(tmp_path / "cache")}, project_root=REPO)


# ---------------------------------------------------------------------------- RR-3


def test_a_limit_kind_typo_launches_under_the_default_containment(tmp_path: Path, caplog) -> None:
    """RR-3: ``limit_kind: rsss`` made build_launch_spec raise and the gateway launch the server with no reaper."""
    from vbt.datalayer.gateway import build_gateway
    from vbt.tools.mcp_bridge import MCPServerConfig

    if not sys.platform.startswith("linux"):
        pytest.skip("the reaper is Linux-only")
    gw = build_gateway({"data": {"cache_dir": str(tmp_path / "c"), "memory": {"limit_kind": "watchdog"}}})
    cfg = MCPServerConfig("target", command=sys.executable, args=["-c", "pass"], limit_kind="rsss")
    spec = gw.launch_spec(cfg)
    assert spec is not None and str(REAPER) in spec.args
    assert spec.args[spec.args.index("--containment") + 1] == "watchdog"
    assert any("rsss" in r.getMessage() for r in caplog.records)


def test_doctor_and_lint_name_a_limit_kind_typo() -> None:
    from vbt.config import load_config
    from vbt.datalayer.cli import _server_findings
    from vbt.preflight import check_mcp_commands

    config = load_config(["mock"], overrides={"mcp_servers_file": "configs/mcp_servers.yaml"})
    for s in config["mcp_servers"]["servers"]:
        if s["name"] == "target":
            s["limit_kind"] = "rsss"
    bad = [r for r in check_mcp_commands(config) if r.label == "MCP server target: limit_kind"]
    assert len(bad) == 1 and not bad[0].ok and bad[0].required and "rsss" in bad[0].detail
    (f,) = _server_findings(config)
    assert f.level == "error" and "target" in f.where and "rsss" in f.message


# ---------------------------------------------------------------------------- RR-5


def _profile(tmp_path: Path, src: Path, ov: Path) -> str:
    p = tmp_path / "profile.yaml"
    p.write_text(json.dumps({"data": {"descriptors_dir": str(src), "overlays_dir": str(ov),
                                      "cache_dir": str(tmp_path / "cache")}}))
    return str(p)


def test_check_tool_reports_a_quarantined_tool_as_not_ready(tmp_path: Path, capsys) -> None:
    """RR-5: with open_targets.yaml quarantined ``ds check --tool target.get_target_info`` said ready (exit 0)."""
    from vbt import cli

    broken = (REPO / "configs" / "data" / "sources" / "open_targets.yaml").read_text() + "\nbogus_key: 1\n"
    src, ov = _catalog_copy(tmp_path, descriptors={"open_targets.yaml": broken})
    prof = _profile(tmp_path, src, ov)
    rc = cli.main(["--profile", "mock", "--profile", prof, "ds", "check", "--tool", "target.get_target_info",
                   "--json"])
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 1 and body["tool"]["ready"] is False
    assert body["tool"]["reasons"][0]["check"] == "quarantined" and "open_targets.yaml" in body["tool"]["reasons"][0][
        "name"]
    rc = cli.main(["--profile", "mock", "--profile", prof, "ds", "explain", "target.get_target_info"])
    out = capsys.readouterr().out
    assert rc == 0 and "QUARANTINED" in out and "open_targets.yaml" in out


def test_graduate_names_a_quarantined_overlay(tmp_path: Path) -> None:
    from vbt.datalayer.cli import graduation_checklist
    from vbt.datalayer.plugins.registry import discover

    src, ov = _catalog_copy(tmp_path, overlays={"target.yaml": (OVERLAYS / "target.yaml").read_text()[:200]})
    cat = _build(src, ov)
    out = graduation_checklist({}, ["target"], catalog=cat, registry=discover(entry_points=False))
    assert not out["target"]["graduated"] and "quarantined" in out["target"]["items"]["overlay"]["detail"]


# ---------------------------------------------------------------------------- RR-6


def test_the_offline_data_child_reads_live_sources_at_a_dead_port(tmp_path: Path, monkeypatch) -> None:
    """RR-6: the data child read cBioPortal at www.cbioportal.org (the descriptor default): two CONNECTs per default
    run, 120 s each behind a proxy that never answers."""
    from vbt.config import base_tool_env

    monkeypatch.delenv("VBT_DL_NETWORK", raising=False)
    cfg = harness_config(gateway=True, tmp_path=tmp_path)
    env = base_tool_env(cfg)
    for key, base in offline_bases().items():
        assert env[key] == base and base.startswith("http://127.0.0.1:9/")
    monkeypatch.setenv("VBT_DL_NETWORK", "1")
    assert offline_bases() == {} and "VBT_CBIOPORTAL_BASE" not in base_tool_env(
        harness_config(gateway=True, tmp_path=tmp_path / "n"))


# ---------------------------------------------------------------------------- RR-7


def test_a_host_oom_count_is_never_the_childs_cause_without_the_kernel_log() -> None:
    """RR-7: with /dev/kmsg unreadable the reaper read the host's vmstat oom_kill moving (another agent's process
    killed in its own cgroup) as the child's kernel OOM kill."""
    assert reaper.kill_attribution(True, 3, None) == (True, False)
    assert reaper.kill_attribution(False, 3, None) == (False, False)
    assert reaper.kill_attribution(None, 3, 4) == (False, True)        # possible, never the cause
    assert reaper.kill_attribution(None, 3, 3) == (False, False)
    assert reaper.kill_attribution(None, None, 4) == (False, False)
    cause, _ = reaper.exit_cause(abnormal=True, signum=9, limit_mb=1000, peak_mb=13, kernel_oom=False)
    assert cause is None
    d = crash.crash_decision("closed", 'VBT_CHILD_EXIT {"code": null, "signal": 9, "reason": "signal", '
                             '"possible_kernel_oom": true}\n', server="s", tool="t")
    assert d.error.payload["exit"]["possible_kernel_oom"] is True and "OOM killer" not in d.error.message


# ---------------------------------------------------------------------------- RR-8


@pytest.mark.parametrize("value,root,ot", [("data/real", "data/real", "data/real/open_targets/25.09"),
                                           ("data/real/open_targets/25.09", "data/real",
                                            "data/real/open_targets/25.09")])
def test_one_real_data_value_serves_every_module(tmp_path: Path, monkeypatch, value: str, root: str, ot: str) -> None:
    """RR-8: the schema module took the value as the 25.09 directory and the Tahoe module as the shared root, so no
    one value ran both (a skip that read as missing data)."""
    (tmp_path / ot).mkdir(parents=True)
    monkeypatch.setenv("VBT_DL_REAL_DATA", str(tmp_path / value))
    assert real_data_root() == tmp_path / root and real_ot_dir() == tmp_path / ot
    monkeypatch.delenv("VBT_DL_REAL_DATA")
    assert real_data_root() is None and real_ot_dir() is None


@pytest.mark.parametrize("value", ["data", "data/sources"])
def test_real_data_is_found_where_vbt_data_acquire_puts_it(tmp_path: Path, monkeypatch, value: str) -> None:
    """DEP-4: on a host `vbt setup` filled, the releases are at $VBT_HOME/data/sources/<acquisition.dir>; the tests
    found only this project's hand-made data/real layout and skipped everything else (green, checking nothing)."""
    from dl_upstream import acquisition_home, real_source_home

    sources = tmp_path / "data" / "sources"
    for source in ("open_targets", "gene_ontology", "tahoe_100m"):
        (sources / acquisition_home(source)[0]).mkdir(parents=True)
    monkeypatch.setenv("VBT_DL_REAL_DATA", str(tmp_path / value))
    assert real_data_root() == sources
    assert real_ot_dir() == sources / "open_targets" / "25.09"
    assert real_source_home("gene_ontology") == sources / "gene_ontology" / acquisition_home("gene_ontology")[1]
    assert real_source_home("tahoe_100m") == sources / "tahoe" / acquisition_home("tahoe_100m")[1]
    assert real_source_home("depmap") is None                       # not acquired here
    # the older hand-made <source>/current layout is still found
    (sources / "cell_ontology" / "current").mkdir(parents=True)
    assert real_source_home("cell_ontology") == sources / "cell_ontology" / "current"


def test_a_strict_real_data_run_that_finds_no_data_fails(tmp_path: Path) -> None:
    """DEP-4: with VBT_DL_REAL_DATA_STRICT=1 (the real-data workflow) a data directory the tests cannot use is a failed
    run, not a run of skips; without it the same run passes as before."""
    test = "tests/datalayer/test_dl_real_tahoe_depmap.py::test_real_ontologies_are_ready"
    env = {**os.environ, "VBT_DL_REAL_DATA": str(tmp_path)}
    env.pop("VBT_DL_REAL_DATA_STRICT", None)
    loose = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=300)
    assert loose.returncode == 0, loose.stdout[-2000:]
    env["VBT_DL_REAL_DATA_STRICT"] = "1"
    strict = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test], cwd=REPO,
                            env=env, capture_output=True, text=True, timeout=300)
    assert strict.returncode == 1 and "none passed" in strict.stdout, strict.stdout[-2000:]

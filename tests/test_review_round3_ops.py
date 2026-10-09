"""Regressions for the third review's deployment, memory and accounting findings (offline).

* RR-5: memory settings that do not parse are reported (lint, doctor, validate) and sizes with units are read;
  an unknown ``limit_kind`` runs under rss, never the reaper's RLIMIT_DATA.
* DEP-14: the ``auto`` rules share one budget: at full load they add up to the plan less the reserve, and
  ``vbt validate``'s host step prints the sum.
* DEP-6: ``vbt setup``'s size step plans from ``$VBT_HOST_MEMORY_MB`` (or ``host_mb``) as the runtime does.
* ACC-1: ``vbt ds estimate`` reports MiB and decides admissibility with admission's rule (x safety + baseline).
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from typing import Any

import pytest

from vbt.config import load_config

# --------------------------------------------------------------------------- RR-5


@pytest.mark.parametrize("value,mb", [("6 GB", 6144.0), ("6GB", 6144.0), ("6144", 6144.0), ("6144 MB", 6144.0),
                                      ("1.5T", 1.5 * 1024 * 1024), (8000, 8000.0), ("six", None), ("6 parsecs", None)])
def test_memory_sizes_are_read_with_their_unit(value: Any, mb: float | None) -> None:
    from vbt.datalayer.memory import sizing

    assert sizing._number(value) == mb


def test_a_host_mb_with_a_unit_is_the_plan(monkeypatch) -> None:
    from vbt.datalayer.memory import sizing

    monkeypatch.delenv("VBT_HOST_MEMORY_MB", raising=False)
    assert sizing.plan_mb({"host_mb": "6 GB"}) == 6144.0
    assert sizing.plan_source({"host_mb": "6 GB"}) == "data.memory.host_mb"


def test_memory_settings_that_would_be_ignored_are_reported() -> None:
    from vbt.datalayer.launch import LIMIT_KINDS
    from vbt.datalayer.memory.sizing import memory_problems

    bad = {"host_mb": "lots", "host_budget_mb": "4 bananas", "limit_kind": "rsss", "workspace_mb": -1}
    problems = memory_problems(bad, LIMIT_KINDS)
    assert len(problems) == 4 and any("rsss" in p for p in problems)
    good = {"host_mb": "6 GB", "host_budget_mb": "off", "limit_kind": "rss", "workspace_mb": "auto",
            "default_server_mb": 8192}
    assert memory_problems(good, LIMIT_KINDS) == []


def test_an_unknown_limit_kind_runs_under_rss(caplog) -> None:
    from types import SimpleNamespace

    from vbt.datalayer.launch import DEFAULT_LIMIT_KIND, containment_args, global_limit_kind, limit_kind

    assert global_limit_kind("rsss") == DEFAULT_LIMIT_KIND == "rss"
    assert "rsss" in caplog.text
    settings = SimpleNamespace(memory=SimpleNamespace(limit_kind="rsss"))
    assert containment_args(limit_kind("target", settings)) == ["--containment", "rss"]
    assert containment_args("rlimit_data") == []


def test_lint_reports_bad_memory_settings() -> None:
    from vbt.datalayer.cli import _server_findings

    cfg = load_config(["mock"], overrides={"data": {"memory": {"limit_kind": "rsss", "host_mb": "6 bananas"}}})
    found = [str(f) for f in _server_findings(cfg)]
    assert any("rsss" in f for f in found) and any("6 bananas" in f for f in found)


def test_a_host_budget_with_a_unit_is_not_switched_off() -> None:
    from types import SimpleNamespace

    from vbt.datalayer.memory.host import host_budget_mb

    assert host_budget_mb(SimpleNamespace(memory=SimpleNamespace(host_budget_mb="4 GB"))) == 4096.0
    assert host_budget_mb(SimpleNamespace(memory=SimpleNamespace(host_budget_mb="off"))) is None


# --------------------------------------------------------------------------- DEP-14


@pytest.mark.parametrize("plan", [512 * 1024, 1024 * 1024])
def test_the_auto_limits_share_one_budget(plan: int) -> None:
    """Above the floors the rules add up to the plan less the harness reserve (on 64 GB with 8 agents the 8,000 MB
    workspace floor alone is the plan: validate reports that sum as over the plan)."""
    from vbt.datalayer.memory import sizing

    budget = sizing.host_budget_for(plan)
    child = sizing.data_child_for(plan)
    ws = sizing.workspace_for(plan, 8)
    load = sizing.full_load(plan, host_budget=budget, data_child=child, workspace=ws, parallel=8,
                            reserve=sizing._reserve(plan, None))
    assert load["sum_mb"] <= plan - load["reserve_mb"] + 8        # rounding of eight workspaces
    assert load["left_mb"] >= load["reserve_mb"] - 8


def test_validate_prints_the_full_load(monkeypatch, tmp_path) -> None:
    from vbt.datalayer.memory import sizing

    monkeypatch.setenv("VBT_HOST_MEMORY_MB", str(1024 * 1024))
    desc = sizing.describe({"memory": {}}, parallel=8)
    load = desc["full_load"]
    assert load["plan_mb"] == 1024 * 1024 and load["workspace_mb"] == 28672 and load["left_mb"] > 0
    # pinned limits from a bigger plan (DEP-6: host.yaml, then VBT_HOST_MEMORY_MB=6000) are over the plan
    monkeypatch.setenv("VBT_HOST_MEMORY_MB", "6000")
    pinned = sizing.describe({"memory": {"host_budget_mb": 8212, "workspace_mb": 6840},
                              "service": {"mem_limit_mb": 3000}}, parallel=8)
    assert pinned["full_load"]["left_mb"] < 0


# --------------------------------------------------------------------------- DEP-6


def test_setup_sizes_from_the_host_memory_override(monkeypatch) -> None:
    from vbt.setup.hostconfig import size_host

    facts = {"memory": {"effective_mb": 13680, "total_mb": 13680}, "cpu": {"effective": 4}}
    monkeypatch.setenv("VBT_HOST_MEMORY_MB", "6000")
    out = size_host(facts, {"data": {"memory": {}}})
    assert out["ram_mb"] == 6000 and out["plan_from"] == "VBT_HOST_MEMORY_MB"
    assert out["host_budget_mb"] < 6000 and any("VBT_HOST_MEMORY_MB" in n for n in out["notes"])
    monkeypatch.delenv("VBT_HOST_MEMORY_MB")
    measured = size_host(facts, {"data": {"memory": {}}})
    assert measured["ram_mb"] == 13680 and measured["plan_from"] == "measured"


def test_setup_sizes_the_largest_server_with_admissions_rule(monkeypatch) -> None:
    """ACC-1: a 9,000 MB plan, target's whole-table loads 3,022.8 MiB: admission needs 3,022.8 x 1.3 + 300 MB."""
    from vbt.setup.hostconfig import size_host

    monkeypatch.setenv("VBT_HOST_MEMORY_MB", "9000")
    out = size_host({"memory": {"effective_mb": 9000}}, {"data": {"memory": {}}}, server_need_mb={"target": 3022.8})
    assert out["largest_server"]["with_safety_mb"] == 4230
    assert out["default_server_mb"] >= 4230 or any("target" in n for n in out["notes"])


# --------------------------------------------------------------------------- ACC-1


def test_ds_estimate_decides_with_admissions_rule(monkeypatch) -> None:
    from vbt.datalayer import cli as ds_cli
    from vbt.datalayer.memory import MemoryEstimator

    stats = {"tables": {"open_targets.target": {"fingerprint": "f", "rows": 62000, "fragments": 200,
                                                "bytes_on_disk": 400_000_000}}}
    monkeypatch.setattr(ds_cli, "_table_stats", lambda config, tables: stats)
    monkeypatch.setattr(MemoryEstimator, "peak_upstream", lambda self, model, *a, **k: 3_169_600_000)
    cfg = load_config([], overrides={"data": {"memory": {"default_server_mb": 3500}}})   # with the server list
    contract_tables = ds_cli._catalog(cfg)[1].contract("target", "get_chemical_probes").tables
    assert any(str(r).startswith("open_targets.target") for r in contract_tables)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ds_cli.cmd_estimate(argparse.Namespace(tool="target.get_chemical_probes", table=None, json=True), cfg)
    doc = json.loads(buf.getvalue().strip().splitlines()[-1])
    tool = doc["tool"]
    assert rc == 0 and doc["unit"] == "MiB"
    assert tool["upstream_mb"] == pytest.approx(3022.8, abs=0.1)              # MiB, not 3,169.6 decimal MB
    assert tool["need_mb"] == pytest.approx(3022.8 * 1.3 + 300, abs=0.2)
    assert tool["limit_mb"] == 3500 and tool["admissible"] is False           # admission refuses it too

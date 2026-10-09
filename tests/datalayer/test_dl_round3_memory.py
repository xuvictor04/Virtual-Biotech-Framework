"""Host-scaled memory defaults and containment that works for every server (round 3, package D3).

* ``memory/sizing.py``: what ``auto`` means on a host (the smaller of ``MemTotal`` and the memory cgroup limit, or
  ``data.memory.host_mb``): the host budget, one server's limit, the data child's limit and the witness and readiness
  budgets, each with a floor; numbers stay as configured.
* The shipped configuration plans from the host: ``default_server_mb: auto``, ``limit_kind: rss`` (a memory cgroup
  when one can be created, else the RSS watchdog; no ``RLIMIT_DATA``, which TileDB's Census reads fail under).
* On a simulated 512 GB host every Open Targets server's whole-table loads are admitted (as upstream loads them);
  on a 16 GB host the largest are refused with ``too_large`` naming the auto limit and the host that would admit
  them.
* The session check runs each large local table in a data child of its own (``preflight.check_groups``); a
  group whose child fails marks its tables, the other groups' results stand.
* A wide CSV matrix's statistics decode many rows per block (DepMap 24Q4 CRISPRGeneEffect.csv: 182 s to 7.7 s).
* R6/R7/R9/R10 on a matrix's axes (contract request: the ``service/checks.py`` patch in the D3 report), skipped
  until ``checks.matrix_axis_checks`` exists.
* Opt-in (``VBT_DL_NETWORK=1``, ``cellxgene_census`` installed, the upstream checkout): the unmodified
  ``single_cell`` server under the default containment answers small Census pulls through ``MCPBridge`` with the
  gateway enforcing, and under ``RLIMIT_DATA`` the same pull fails with ``std::bad_alloc``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from netgate import network_enabled
from vbt.datalayer.errors import GatewayError
from vbt.datalayer.launch import build_launch_spec, server_limit_mb
from vbt.datalayer.memory import AdmissionController, ResidencyLedger, TableRead, sizing
from vbt.datalayer.memory.estimate import MB
from vbt.datalayer.memory.host import HostBudget, host_budget_mb
from vbt.datalayer.settings import DataSettings
from vbt.tools.mcp_bridge import MCPServerConfig

REPO = Path(__file__).resolve().parents[2]
NETWORK = network_enabled()
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the reaper is Linux-only")

GB = 1024


@pytest.fixture
def host(monkeypatch):
    """``host(mb)``: plan every ``auto`` from a host of ``mb`` MB (``$VBT_HOST_MEMORY_MB``)."""
    def set_mb(mb: float) -> float:
        monkeypatch.setenv(sizing.HOST_MB_ENV, str(mb))
        return mb
    return set_mb


# --------------------------------------------------------------------------- the rules


@pytest.mark.parametrize("plan, budget, server, child", [
    (8 * GB, 4096, 3276, 3000),                 # small: the floors and the 2,048 MB reserve
    (16094, 10022, 8018, 3000),                 # this sandbox's MemTotal
    (13680, 8212, 6569, 3000),                  # ... and its memory cgroup
    (64 * GB, 45875, 36700, 3276),
    (512 * GB, 367002, 293601, 26214),          # the full configuration of docs/DEPLOYMENT.md
    (1024 * GB, 734003, 587202, 32768),         # the data child stops at 32 GB
])
def test_auto_rules_scale_with_the_host(plan, budget, server, child):
    assert round(sizing.host_budget_for(plan)) == budget
    assert sizing.server_limit_for(plan) == server
    assert sizing.data_child_for(plan) == child
    assert sizing.server_limit_for(plan) <= max(sizing.SERVER_FLOOR_MB, sizing.host_budget_for(plan))
    assert sizing.witness_scale(child) == max(1.0, child / 3000)


def test_floors_hold_on_a_tiny_host():
    assert sizing.host_budget_for(2048) == sizing.BUDGET_FLOOR_MB
    assert sizing.server_limit_for(2048) == sizing.SERVER_FLOOR_MB
    assert sizing.data_child_for(2048) == sizing.CHILD_FLOOR_MB
    assert sizing.witness_scale(1500) == 1.0


def test_the_reserve_is_the_larger_of_the_configured_one_and_five_percent():
    assert sizing.host_budget_for(16000, {"harness_reserve_mb": 1024}) == 0.75 * 16000 - 1024
    assert sizing.host_budget_for(512 * GB, {"harness_reserve_mb": 1024}) == 0.75 * 512 * GB - 0.05 * 512 * GB


def test_plan_for_server_inverts_the_server_rule():
    for need in (3000, 7031, 15034, 103000):
        plan = sizing.plan_for_server(need)
        assert sizing.server_limit_for(plan) >= need > sizing.server_limit_for(plan - 2)


def test_resolve_auto_scales_the_budgets_and_keeps_numbers(host):
    data = {"service": {"mem_limit_mb": "auto", "max_resident_mb": "auto", "max_concurrency": 4},
            "witness": {"max_scan_bytes": "auto", "max_inflate_bytes": "auto", "max_key_set": "auto",
                        "repair_max_bytes": 123, "max_inflate_rows": 5000},
            "readiness": {"vocab_budget_bytes": "auto"},
            "memory": {"default_server_mb": "auto", "host_budget_mb": "auto"}}
    host(16 * GB)
    small = sizing.resolve_auto(data)
    assert small["service"] == {"mem_limit_mb": 3000, "max_resident_mb": 2000, "max_concurrency": 4}
    assert small["witness"] == {"max_scan_bytes": 2_000_000_000, "max_inflate_bytes": 20_000_000, "max_key_set": 20000,
                                "repair_max_bytes": 123, "max_inflate_rows": 5000}
    assert small["readiness"]["vocab_budget_bytes"] == 500_000_000
    assert small["memory"] == data["memory"], "the launcher and the host budget resolve these where they are used"
    host(512 * GB)
    big = sizing.resolve_auto(data)
    scale = 26214 / 3000
    assert big["service"]["mem_limit_mb"] == 26214 and big["service"]["max_resident_mb"] == int(26214 * 2 / 3)
    assert big["witness"]["max_scan_bytes"] == int(2_000_000_000 * scale)
    assert big["witness"]["max_key_set"] == int(20000 * scale) and big["witness"]["repair_max_bytes"] == 123
    assert big["readiness"]["vocab_budget_bytes"] == int(500_000_000 * scale)
    assert data["service"]["mem_limit_mb"] == "auto", "the input is not changed"
    fixed = sizing.resolve_auto({"service": {"mem_limit_mb": 6000}, "witness": {"max_scan_bytes": "auto"}})
    assert fixed["witness"]["max_scan_bytes"] == 4_000_000_000, "the budgets follow a configured data child limit"


def test_host_mb_setting_wins_over_the_probe_and_the_environment(host):
    host(512 * GB)
    assert sizing.plan_mb({"host_mb": 32000}) == 32000
    assert sizing.plan_mb({"host_mb": "auto"}) == 512 * GB
    assert sizing.plan_mb({}, measured=1234.0) == 512 * GB, "the environment declares the host"


def _cgroup_tree(tmp: Path, *, v1: dict[str, int] | None = None, v2: dict[str, str] | None = None,
                 self_cgroup: str) -> tuple[str, str]:
    proc, cg = tmp / "proc", tmp / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text(self_cgroup)
    (proc / "meminfo").write_text("MemTotal:       16480952 kB\nMemFree:  1 kB\n")
    for rel, value in (v1 or {}).items():
        d = cg / "memory" / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.limit_in_bytes").write_text(f"{value}\n")
    for rel, value in (v2 or {}).items():
        d = cg / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.max").write_text(f"{value}\n")
    return str(proc), str(cg)


def test_the_cgroup_limit_is_the_tightest_over_the_ancestors(tmp_path):
    # this sandbox's shape: cgroup v1, the limit on the leaf, "unlimited" (2^63 rounded) above it
    proc, cg = _cgroup_tree(tmp_path / "v1", self_cgroup="4:memory:/process_api/x/bash\n0::/\n",
                            v1={"": 9223372036854771712, "process_api": 9223372036854771712,
                                "process_api/x": 9223372036854771712, "process_api/x/bash": 14345019392})
    assert sizing.cgroup_limit_mb(proc, cg) == pytest.approx(13680.48, abs=0.01)
    assert sizing.effective_memory_mb(proc, cg) == pytest.approx(13680.48, abs=0.01)
    # a container (v2): the parent's memory.max binds, the child says max
    proc, cg = _cgroup_tree(tmp_path / "v2", self_cgroup="0::/system.slice/vbt.service/sub\n",
                            v2={"": "max", "system.slice": "max", "system.slice/vbt.service": str(8 << 30),
                                "system.slice/vbt.service/sub": "max"})
    assert sizing.cgroup_limit_mb(proc, cg) == 8192
    assert sizing.effective_memory_mb(proc, cg) == 8192
    # no limit anywhere: MemTotal
    proc, cg = _cgroup_tree(tmp_path / "none", self_cgroup="0::/\n", v2={"": "max"})
    assert sizing.cgroup_limit_mb(proc, cg) is None
    assert sizing.effective_memory_mb(proc, cg) == pytest.approx(16094.68, abs=0.01)


# --------------------------------------------------------------------------- the launcher and the host budget


def test_the_launcher_resolves_auto_limits(host):
    host(512 * GB)
    auto = DataSettings.from_dict({"memory": {"default_server_mb": "auto"}, "service": {"mem_limit_mb": "auto"}})
    target = MCPServerConfig("target", command="python")
    assert server_limit_mb(target, auto) == 293601
    assert server_limit_mb(MCPServerConfig("data", command="python"), auto) == 26214
    assert server_limit_mb(MCPServerConfig("x", command="python", mem_limit_mb="auto"),
                           DataSettings.from_dict({"memory": {"default_server_mb": 5000}})) == 293601
    # numbers stay as configured, the server's own first
    fixed = DataSettings.from_dict({"memory": {"default_server_mb": 5000}, "service": {"mem_limit_mb": 2500}})
    assert server_limit_mb(target, fixed) == 5000
    assert server_limit_mb(MCPServerConfig("data", command="python"), fixed) == 2500
    assert server_limit_mb(MCPServerConfig("t", command="python", mem_limit_mb=777), auto) == 777
    host(16 * GB)
    assert server_limit_mb(target, auto) == 8192 and server_limit_mb("data", auto) == 3000


def test_the_host_budget_plans_from_the_host(host):
    host(512 * GB)
    assert host_budget_mb(DataSettings.from_dict({})) == pytest.approx(367001.6)
    assert host_budget_mb(DataSettings.from_dict({"memory": {"host_mb": 64 * GB}})) == pytest.approx(45875.2)
    assert host_budget_mb(DataSettings.from_dict({"memory": {"host_budget_mb": 4096}})) == 4096
    assert host_budget_mb(DataSettings.from_dict({"memory": {"host_budget_mb": "off"}})) is None


@linux_only
def test_the_shipped_configuration_scales_and_contains_by_resident_memory(host, tmp_path):
    from vbt.config import load_config

    config = load_config(["mock"])
    memory = config["data"]["memory"]
    assert memory["default_server_mb"] == "auto" and memory["host_mb"] == "auto"
    assert memory["host_budget_mb"] == "auto" and memory["limit_kind"] == "rss"
    host(512 * GB)
    settings = DataSettings.from_config(config)
    raw = yaml.safe_load((REPO / "configs" / "mcp_servers.yaml").read_text())
    servers = {s["name"]: s for s in raw["servers"]}
    for name in ("target", "genetics", "single_cell"):
        cfg = MCPServerConfig(**{k: v for k, v in servers[name].items() if k in MCPServerConfig.__dataclass_fields__})
        spec = build_launch_spec(cfg, settings, tmp_path)
        assert spec.args[spec.args.index("--containment") + 1] == "rss", name
        assert int(spec.args[spec.args.index("--limit-mb") + 1]) == 293601, name
    pinned = {s["name"]: s.get("limit_kind") for s in raw["servers"] if s.get("limit_kind")}
    assert pinned == {"single_cell": "rss"}, "single_cell keeps rss even when a host sets rlimit_data"


#: Whole-table loads per server of the full Open Targets 25.09 release (docs/DEPLOYMENT.md §2.2: measured where
#: R5 could load the table on 16 GB, else estimated), MB.
OT_WHOLE_TABLE_MB = {"genetics": 79_500, "interaction": 10_300, "association": 4_900, "expression": 5_800,
                     "target": 3_400, "drug": 800, "disease": 500, "pathway": 20}


def _stats(mb: float) -> dict:
    """A flat table whose upstream peak is ``mb`` MB with the in-code factors (flat 1.5 x fragmentation 1.15)."""
    return {"fingerprint": f"f{mb}", "rows": 1000,
            "columns": {"x": {"uncompressed_bytes": int(mb * MB / (1.5 * 1.15)), "num_values": 1000}}}


async def test_a_512_gb_host_admits_every_whole_table_load_as_upstream_does(host):
    host(512 * GB)
    settings = DataSettings.from_dict({"memory": {"default_server_mb": "auto"}})
    ac = AdmissionController(settings, None, ResidencyLedger())
    ac.enable_host_budget()
    assert ac.host is not None and ac.host.budget_mb > sum(OT_WHOLE_TABLE_MB.values()) * 1.3
    for server, mb in sorted(OT_WHOLE_TABLE_MB.items(), key=lambda kv: -kv[1]):
        adm = await ac.admit(server, [TableRead(f"open_targets.{server}_tables", "full_table", _stats(mb))],
                             "upstream", tool=f"{server}.tool")
        assert adm.admitted and adm.cold_tables, server
        ac.commit(adm)
    assert ac.host.evictions == [], "every server keeps its tables: nothing is recycled"


async def test_a_16_gb_host_refuses_the_genetics_load_naming_the_auto_limit(host):
    host(16 * GB)
    settings = DataSettings.from_dict({"memory": {"default_server_mb": "auto"}})
    ac = AdmissionController(settings, None, ResidencyLedger())
    with pytest.raises(GatewayError) as exc:
        await ac.admit("genetics", [TableRead("open_targets.l2g_prediction", "full_table", _stats(11_564))],
                       "upstream", tool="genetics.query_l2g_predictions")
    err = exc.value
    assert err.kind.value == "too_large" and err.subkind == "over_limit"
    text = str(err)
    # ACC-4: the host size it names carries the MB figure the payload holds and what it was sized for
    need = err.payload["host_mb_needed"]
    assert "default_server_mb is auto" in text and "16,384 MB host" in text and f"({need:,} MB, sized for" in text \
        and ") admits it" in text, text
    assert err.payload["limit_source"] == "auto" and err.payload["host_mb"] == 16 * GB
    assert sizing.server_limit_for(err.payload["host_mb_needed"]) >= 11_564 * 1.3
    # the same refusal under a configured limit says so
    fixed = AdmissionController(DataSettings.from_dict({"memory": {"default_server_mb": 5000}}), None,
                                ResidencyLedger())
    with pytest.raises(GatewayError) as exc2:
        await fixed.admit("genetics", [TableRead("open_targets.l2g_prediction", "full_table", _stats(11_564))],
                          "upstream")
    assert exc2.value.payload["limit_source"] == "configured" and "the limit is configured" in str(exc2.value)
    # the target table fits on the same host
    ok = await ac.admit("target", [TableRead("open_targets.target", "full_table", _stats(3_031))], "upstream")
    assert ok.admitted


async def test_host_busy_names_how_the_budget_was_set(host):
    host(16 * GB)
    settings = DataSettings.from_dict({})
    ledger = ResidencyLedger()
    budget = HostBudget.from_settings(settings, ledger)
    assert budget.origin and "host_budget_mb is auto" in budget.origin and "16,384 MB" in budget.origin
    budget.begin("busy")
    ledger.add("busy", {"t": 9000.0})
    with pytest.raises(GatewayError) as exc:
        await budget.reserve("target", 4000.0)
    assert "host_budget_mb is auto" in str(exc.value) and exc.value.subkind == "host_busy"


def test_describe_lists_every_auto_setting(host):
    host(512 * GB)
    data = {"memory": {"default_server_mb": "auto", "host_budget_mb": "auto", "limit_kind": "rss"},
            "service": {"mem_limit_mb": "auto", "max_resident_mb": 2000},
            "witness": {"max_scan_bytes": "auto"}, "readiness": {"vocab_budget_bytes": 500_000_000}}
    out = sizing.describe(data, servers={"genetics": "auto", "pathway": 2048})
    s = out["settings"]
    assert out["host"]["plan_mb"] == 512 * GB and out["host"]["plan_from"] == sizing.HOST_MB_ENV
    assert s["memory.default_server_mb"]["effective"] == 293601
    assert s["memory.host_budget_mb"]["effective"] == 367002
    assert s["service.mem_limit_mb"]["effective"] == 26214 and s["service.max_resident_mb"]["rule"] == "as configured"
    assert s["witness.max_scan_bytes"]["effective"] == int(2e9 * 26214 / 3000)
    assert s["mcp_servers.genetics.mem_limit_mb"]["effective"] == 293601
    assert s["mcp_servers.pathway.mem_limit_mb"]["effective"] == 2048
    assert "no RLIMIT_DATA" in s["memory.limit_kind"]["rule"]
    json.dumps(out)


def test_the_data_child_scales_by_its_own_rule_and_rss_is_the_fallback_containment(host):
    from vbt.datalayer.launch import DEFAULT_LIMIT_KIND, limit_kind, limit_source

    host(512 * GB)
    fixed = DataSettings.from_dict({"service": {"mem_limit_mb": 2500}, "memory": {"default_server_mb": 5000}})
    child = MCPServerConfig("data", command="python", mem_limit_mb="auto")
    assert server_limit_mb(child, fixed) == 26214, "a data child set to auto scales as the data child, not a server"
    assert limit_source(child, fixed) == "auto" and limit_source(MCPServerConfig("data", command="p"), fixed) == \
        "configured"
    assert limit_source("target", fixed) == "configured" and limit_source("target", DataSettings.from_dict(
        {"memory": {"default_server_mb": "auto"}})) == "auto"
    assert server_limit_mb("target", None) == 293601 and server_limit_mb("data", None) == 26214
    assert DEFAULT_LIMIT_KIND == "rss" and limit_kind("target", None) == "rss"


# --------------------------------------------------------------------------- the session check, grouped


@pytest.fixture(scope="module")
def ot_small(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dl_fixtures

    return dl_fixtures.build_ot_fixture(tmp_path_factory.mktemp("ot") / "25.09")


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def test_large_tables_are_checked_in_a_child_of_their_own(monkeypatch, ot_small):
    import vbt.preflight as pf
    from vbt.config import load_config

    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot_small))
    config = load_config(["mock"])
    sizes = {p.name: _dir_bytes(p) for p in ot_small.iterdir() if p.is_dir()}
    monkeypatch.setattr(pf, "CHECK_ISOLATE_BYTES", sizes["target"])
    tables = ["open_targets.target", "open_targets.target_go", "open_targets.go", "open_targets.known_drug",
              "open_targets.so"]
    groups = pf.check_groups(config, tables)
    flat = [t for g in groups for t in g]
    assert sorted(flat) == sorted(tables), "every table once"
    where = {t: i for i, g in enumerate(groups) for t in g}
    assert where["open_targets.target_go"] == where["open_targets.target"], "an item table goes with its parent"
    assert groups[where["open_targets.target"]] == ["open_targets.target", "open_targets.target_go"]
    big = {f"open_targets.{n}" for n in ("go", "known_drug", "so") if sizes[n] >= sizes["target"]}
    assert all(groups[where[t]] == [t] for t in big)
    assert groups[-1] == sorted({"open_targets.go", "open_targets.known_drug", "open_targets.so"} - big,
                                key=tables.index)
    monkeypatch.setattr(pf, "CHECK_ISOLATE_BYTES", 1 << 62)
    assert pf.check_groups(config, ["open_targets.target", "open_targets.so"]) == [["open_targets.target",
                                                                                   "open_targets.so"]]


def test_a_failing_group_marks_its_tables_and_keeps_the_others(monkeypatch):
    import vbt.preflight as pf

    monkeypatch.setattr(pf, "check_groups", lambda config, tables=(): [["s.big"], ["s.a", "s.b"]])

    def once(config, tables, depth, timeout):
        if tables == ["s.big"]:
            raise pf.DataCheckUnavailable("python exited -9: killed at the data child's limit")
        return {"tables": {t: {"status": "ready", "checks": []} for t in tables}, "depth": depth,
                "quarantined": [{"file": "x.yaml"}]}

    monkeypatch.setattr(pf, "_run_check_once", once)
    out = pf.run_data_check({}, depth="standard")
    assert set(out["tables"]) == {"s.a", "s.b"} and out["depth"] == "standard"
    assert "exited -9" in out["table_errors"]["s.big"] and out["quarantined"] == [{"file": "x.yaml"}]
    monkeypatch.setattr(pf, "_run_check_once", lambda *a, **k: (_ for _ in ()).throw(pf.DataCheckUnavailable("x")))
    with pytest.raises(pf.DataCheckUnavailable):
        pf.run_data_check({}, depth="standard")


# --------------------------------------------------------------------------- wide CSV statistics


def test_wide_csv_statistics_decode_many_rows_per_block(tmp_path, monkeypatch):
    """A matrix-shaped CSV (wide lines) is scanned for its statistics in blocks of many rows: 1 MiB blocks decoded
    a few lines each, and computing per-column statistics per tiny batch took minutes on DepMap's 17,917 columns."""
    from vbt.datalayer.plugins.base import Fragment
    from vbt.datalayer.plugins.formats import csv as csvmod

    cols, rows = 3000, 300
    path = tmp_path / "wide.csv"
    with path.open("w") as f:
        f.write(",".join(["id", *(f"G{i} ({i})" for i in range(cols))]) + "\n")
        for r in range(rows):
            f.write(",".join([f"ACH-{r:06d}", *(f"{(r * 7 + i) % 97 / 10:.4f}" for i in range(cols))]) + "\n")
    assert path.stat().st_size > 4 * csvmod.INFER_BYTES
    batches: list[int] = []
    real = csvmod.stats_from_batches

    def counting(it, **kw):
        def gen():
            for b in it:
                batches.append(b.num_rows)
                yield b
        return real(gen(), **kw)

    monkeypatch.setattr(csvmod, "stats_from_batches", counting)
    fmt = csvmod.CsvFormat().configure({}, None)
    st = os.stat(path)
    stats = fmt._scan_stats(Fragment(uri=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns))
    assert stats.rows == rows and len(stats.columns) == cols + 1
    assert len(batches) == 1 and batches == [rows], batches
    assert stats.columns["G5 (5)"].max == pytest.approx(9.6) and stats.columns["G5 (5)"].null_count == 0


# --------------------------------------------------------------------------- matrix axes (contract request)


def _depmap_ctx(tmp_path: Path, monkeypatch) -> object:
    import shutil

    from vbt.datalayer.service import ServiceContext

    for d in ("sources", "overlays", "depmap"):
        (tmp_path / d).mkdir()
    shutil.copy(REPO / "configs" / "data" / "sources" / "depmap.yaml", tmp_path / "sources" / "depmap.yaml")
    monkeypatch.setenv("DEPMAP_DATA_PATH", str(tmp_path / "depmap"))
    return ServiceContext(DataSettings.from_dict({"descriptors_dir": str(tmp_path / "sources"),
                                                  "overlays_dir": str(tmp_path / "overlays"),
                                                  "cache_dir": str(tmp_path / "cache")}, project_root=tmp_path))


def test_matrix_axis_columns_get_r9_references(tmp_path, monkeypatch):
    """R9 on the row axis of a DepMap matrix: every ModelID must name a row of the model table. Runs once
    ``service/checks.py`` has ``matrix_axis_checks`` (D3 contract request)."""
    from vbt.datalayer.service import checks

    if not hasattr(checks, "matrix_axis_checks"):
        pytest.skip("service/checks.py has no matrix_axis_checks yet (D3 contract request)")
    ctx = _depmap_ctx(tmp_path, monkeypatch)
    genes = [("RPL3", "6122"), ("TP53", "7157")]
    (tmp_path / "depmap" / "CRISPRGeneEffect.csv").write_text(
        "ModelID," + ",".join(f"{s} ({e})" for s, e in genes) + "\nACH-000001,-1.5,0.1\nACH-000002,-1.2,0.2\n")
    model_header = "ModelID,CellLineName,StrippedCellLineName,OncotreeLineage,OncotreePrimaryDisease\n"
    (tmp_path / "depmap" / "Model.csv").write_text(model_header + "ACH-000001,A,A,Lung,X\nACH-000002,B,B,Skin,Y\n")
    model = checks.check_table(ctx, "depmap.gene_effect")
    r9 = [c for c in model.checks if c.name.startswith("R9") and (c.column or "").startswith("@row")]
    assert r9 and all(c.ok for c in r9), [c.detail for c in r9]
    (tmp_path / "depmap" / "Model.csv").write_text(model_header + "ACH-000001,A,A,Lung,X\n")
    model = checks.check_table(ctx, "depmap.gene_effect")
    bad = [c for c in model.checks if c.name.startswith("R9") and not c.ok]
    assert bad and "@row.ModelID" in (bad[0].column or ""), [c.detail for c in model.checks if not c.ok]


# --------------------------------------------------------------------------- live: the Census through MCPBridge

#: One spleen dataset of the 2025-11-08 Census (S1: 293 cells drawn by upstream's own donor-balanced sample).
CENSUS_DATASET = "f7c1c579-2dc0-47e2-ba19-8165c5a0e353"
CENSUS_GENES = ["ENSG00000198851", "ENSG00000156738"]          # CD3E, MS4A1
needs_census = pytest.mark.skipif(not NETWORK, reason="set VBT_DL_NETWORK=1 to read the live Census")


async def _census_bridge(tmp_path: Path, *, server: dict | None = None):
    """``single_cell`` and the data child through MCPBridge, gateway enforcing, with the shipped memory settings
    (``server`` updates the single_cell entry: its ``env``, ``limit_kind``, ``mem_limit_mb``)."""
    from dl_upstream import (FAST_START, DataEnv, _bridge_configs, _require_gateway, harness_config, server_specs,
                             upstream_missing)

    from vbt.tools.mcp_bridge import MCPBridge

    if upstream_missing():
        pytest.skip(upstream_missing())
    pytest.importorskip("cellxgene_census")
    env = DataEnv(output_dir=tmp_path / "out").env()
    config = harness_config(gateway=True, tmp_path=tmp_path, env=env)
    specs = server_specs(config, ["single_cell"], env)
    for spec in specs:
        spec.update({k: v for k, v in (server or {}).items() if k != "env"})
        spec["env"] = {**spec["env"], **((server or {}).get("env") or {})}
        spec["timeout_s"] = 900
    gw = _require_gateway()(config, None)
    specs += list(gw.extra_servers() or [])
    bridge = MCPBridge(_bridge_configs(specs), log_dir=tmp_path / "mcp-logs",
                       options={**FAST_START, "default_timeout_s": 900}, gateway=gw)
    await bridge.start()
    assert not bridge.failures, bridge.failures
    return bridge


def _containment(bridge) -> dict:
    status = bridge.status()["single_cell"]
    data = json.loads(Path(status["status_file"]).read_text())
    limits = Path(f"/proc/{data['pid']}/limits").read_text() if data.get("pid") else ""
    line = next((ln for ln in limits.splitlines() if ln.startswith("Max data size")), "")
    return {**data, "data_limit_line": line}


async def _small_pulls(bridge, tmp_path: Path) -> dict:
    from dl_upstream import call

    out: dict = {}
    count = await call(bridge, "single_cell", "count_cells", {"value_filter": f"dataset_id == '{CENSUS_DATASET}'"})
    assert not count.is_error, count.text[:600]
    out["count_cells"] = count.obj
    pull = await call(bridge, "single_cell", "get_anndata_donor_balanced",
                      {"value_filter": f"dataset_id == '{CENSUS_DATASET}'", "max_cells": 300,
                       "ensembl_ids": CENSUS_GENES,
                       "obs_columns": ["soma_joinid", "cell_type", "donor_id", "dataset_id"],
                       "output_path": "small.h5ad"})
    assert not pull.is_error, pull.text[:600]
    out["pull"] = {k: v for k, v in (pull.obj or {}).items() if not isinstance(v, (dict, list))}
    import h5py

    path = Path(out["pull"]["output_path"])
    path = path if path.is_absolute() else tmp_path / "out" / path
    with h5py.File(str(path), "r") as f:
        out["file_cells"] = int(f["obs"]["soma_joinid"].shape[0])
    out["header"] = {k: pull.header.get(k) for k in ("status", "total", "served_by", "source")}
    return out


@needs_census
@linux_only
async def test_live_census_pulls_run_under_the_default_containment(tmp_path):
    """The unmodified single_cell server under the shipped ``limit_kind: rss``: a memory cgroup where one can be
    created (cgroup v1 on this sandbox), no RLIMIT_DATA, and the count and a 300-cell pull are answered."""
    bridge = await _census_bridge(tmp_path)
    try:
        got = await _small_pulls(bridge, tmp_path)
        cont = _containment(bridge)
    finally:
        await bridge.aclose()
    print("D3 census default:", json.dumps({"calls": got, "containment": cont}, default=str))
    assert cont["containment"] in ("cgroup_v1", "cgroup_v2", "systemd_scope", "watchdog")
    assert cont["rlimit_data"] is False and "unlimited" in cont["data_limit_line"]
    assert cont["limit_mb"] == sizing.server_limit_for(sizing.plan_mb())
    assert 0 < got["file_cells"] <= 300


@needs_census
@linux_only
async def test_live_census_pulls_fall_back_to_the_rss_watchdog(tmp_path):
    """Without a writable memory cgroup the same server runs under the RSS watchdog (still no RLIMIT_DATA)."""
    nowhere = str(tmp_path / "no-cgroup")
    bridge = await _census_bridge(tmp_path, server={"env": {"VBT_REAPER_CGROUP_ROOT": nowhere,
                                                            "VBT_REAPER_CGROUP_V1_ROOT": nowhere}})
    try:
        got = await _small_pulls(bridge, tmp_path)
        cont = _containment(bridge)
    finally:
        await bridge.aclose()
    print("D3 census watchdog:", json.dumps({"calls": got, "containment": cont}, default=str))
    assert cont["containment"] == "watchdog" and cont["rlimit_data"] is False
    assert cont["watchdog_kill_mb"] == pytest.approx(cont["limit_mb"] - max(512, 0.05 * cont["limit_mb"]), abs=0.2)
    assert "unlimited" in cont["data_limit_line"]
    assert 0 < got["file_cells"] <= 300


@needs_census
@linux_only
async def test_live_census_reads_under_rlimit_data_are_answered_or_a_typed_oom(tmp_path):
    """Why ``rss`` is the default: under ``RLIMIT_DATA`` the same server's TileDB reads can fail with
    ``std::bad_alloc`` while resident memory stays far below the limit. It does not fail every time (at 6,569 MB the
    count failed in one run of 2026-10-09 and was answered in a later one), so this records the outcome and asserts
    only that a failure is the typed ``oom``, never an untyped crash, and happens far below the limit."""
    from dl_upstream import call

    bridge = await _census_bridge(tmp_path, server={"limit_kind": "rlimit_data"})
    outcomes = {}
    try:
        outcomes["count_cells"] = await call(bridge, "single_cell", "count_cells",
                                             {"value_filter": f"dataset_id == '{CENSUS_DATASET}'"})
        outcomes["get_anndata_donor_balanced"] = await call(
            bridge, "single_cell", "get_anndata_donor_balanced",
            {"value_filter": f"dataset_id == '{CENSUS_DATASET}'", "max_cells": 300, "ensembl_ids": CENSUS_GENES,
             "obs_columns": ["soma_joinid", "cell_type", "donor_id", "dataset_id"], "output_path": "rlimit.h5ad"})
        cont = _containment(bridge)
    finally:
        await bridge.aclose()
    print("D3 census rlimit_data:", json.dumps({"calls": {k: {"is_error": v.is_error, "kind": v.kind,
                                                               "text": v.text[:300]} for k, v in outcomes.items()},
                                                "containment": cont}, default=str))
    assert cont["containment"] == "rlimit_data" and "unlimited" not in cont["data_limit_line"]
    for name, out in outcomes.items():
        if out.is_error:
            assert out.kind == "oom" and "bad_alloc" in out.text, (name, out.text[:600])
    assert float(cont["peak_rss_mb"]) < 0.6 * float(cont["limit_mb"])

"""`vbt setup` and the deployment files (offline): host probes on fixture /proc and cgroup trees, sizing, serving
pick, host files, the roster's needs, resume, the step commands (run through a fake runner), the entrypoint
script, and the consistency of deploy/full, the CI workflows and the production profile."""

import json
import os
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vbt.config import PROJECT_ROOT, load_config
from vbt.local.profiles import load_local_profiles, resolve_serve
from vbt.setup import hostconfig, probe
from vbt.setup.cli import build_context, cmd_setup, make_plan, run_steps, select_steps
from vbt.setup.layout import resolve_layout
from vbt.setup.needs import compute_needs, root_variables
from vbt.setup.state import SetupState, fingerprint
from vbt.setup.steps import (
    STEP_NAMES,
    _bytes_to_fetch,
    _plan_pending,
    _plan_roots,
    _unavailable,
    data_roots,
    parse_estimate,
)

FULL = PROJECT_ROOT / "deploy" / "full"
WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
GiB_MB = 1024

H100_SMI = "0, NVIDIA H100 80GB HBM3, 81559, 580.65.06\n"
H200x4_SMI = "".join(f"{i}, NVIDIA H200, 143771, 580.65.06\n" for i in range(4))


# ------------------------------------------------------------------------------------------------ fixtures

def _proc(tmp_path, *, total_kb=16480256, avail_kb=12000000, cgroup="0::/user.slice/vbt\n"):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemTotal:       {total_kb} kB\nMemFree:  1 kB\nMemAvailable:   {avail_kb} kB\n"
                                  "SwapTotal:      0 kB\n")
    (proc / "self" / "cgroup").write_text(cgroup)
    return proc


def _facts(ram_mb, cpus, *, cgroup=False, bwrap=False, smi=None):
    gpu = probe.gpu_facts(smi, source="test") if smi else {"count": 0, "gpus": [], "raw": None}
    return {"memory": {"total_mb": ram_mb, "effective_mb": ram_mb}, "cpu": {"effective": cpus},
            "containment": {"limit_kind": "cgroup" if cgroup else "rlimit_data"},
            "sandbox": {"bwrap": {"works": bwrap}}, "gpu": gpu}


@pytest.fixture(autouse=True)
def _host_memory_from_the_facts(monkeypatch):
    """These tests size from the probe's facts they pass in: the suite's fixed VBT_HOST_MEMORY_MB (tests/conftest.py)
    would be the plan instead, as it is for `vbt setup` on a host that sets it (DEP-6)."""
    monkeypatch.delenv("VBT_HOST_MEMORY_MB", raising=False)


@pytest.fixture
def base_config(tmp_path):
    return load_config([], overrides={"paths": {"runs_dir": str(tmp_path / "runs")}})


def _args(tmp_path, **kw):
    ns = dict(home=str(tmp_path / "home"), state_dir=None, data_dir=None, projects_dir=None, models_dir=None,
              runs_dir=None, serving_profile=None, variant=[], harness_profile=None, llm_url=None,
              nvidia_smi_file=None, deploy="host", no_network=True, check_url=[], no_analysis=True, json=False,
              plan=False, probe=False, status=False, only=None, from_step=None, skip=None, force=False,
              keep_going=False, verbose=False, yes=True)
    ns.update(kw)
    return SimpleNamespace(**ns)


class FakeRunner:
    """Records the `vbt` argv of every step and answers from ``replies`` (first matching key in the argv)."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies or {}

    def __call__(self, cmd, *, env, timeout):
        argv = cmd[cmd.index("vbt.cli") + 1:] if "vbt.cli" in cmd else list(cmd)
        self.calls.append({"argv": argv, "env": env})
        joined = " ".join(argv)
        for key, (code, out, err) in self.replies.items():
            if key in joined:
                return subprocess.CompletedProcess(cmd, code, out, err)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def ran(self, word):
        return [c for c in self.calls if word in " ".join(c["argv"])]


# ------------------------------------------------------------------------------------------------ layout

def test_layout_home_env_and_defaults(tmp_path, base_config):
    lay = resolve_layout(base_config, home=str(tmp_path / "h"), environ={})
    assert lay.home == (tmp_path / "h").resolve()
    assert (lay.state, lay.data, lay.runs, lay.projects, lay.models) == tuple(
        (tmp_path / "h" / n).resolve() for n in ("state", "data", "runs", "projects", "models"))
    env = {"VBT_HOME": str(tmp_path / "h"), "VBT_DATA_DIR": str(tmp_path / "bigdisk"), "VBT_STATE_DIR": ""}
    lay = resolve_layout(base_config, environ=env)
    assert lay.data == (tmp_path / "bigdisk").resolve() and lay.state == (tmp_path / "h" / "state").resolve()
    lay = resolve_layout(base_config, environ={})                 # a clone used in place
    assert lay.home is None and lay.data == PROJECT_ROOT / "data" and lay.state == PROJECT_ROOT / "data" / ".vbt-setup"
    assert lay.runs == (tmp_path / "runs").resolve()


# ------------------------------------------------------------------------------------------------ probes

def test_memory_facts_see_a_cgroup_v2_limit_below_memtotal(tmp_path):
    proc = _proc(tmp_path, total_kb=512 * 1024 * 1024)
    cg = tmp_path / "cg"
    (cg / "user.slice" / "vbt").mkdir(parents=True)
    (cg / "cgroup.controllers").write_text("cpu memory pids\n")
    (cg / "user.slice" / "vbt" / "memory.max").write_text(str(64 * 1024 ** 3) + "\n")
    (cg / "user.slice" / "vbt" / "cpu.max").write_text("800000 100000\n")
    mem = probe.memory_facts(proc_root=str(proc), cgroup_root=str(cg))
    assert mem["total_mb"] == 512 * GiB_MB and mem["cgroup_limit_mb"] == 64 * GiB_MB
    assert mem["effective_mb"] == 64 * GiB_MB and mem["cgroup_version"] == 2
    cpu = probe.cpu_facts(proc_root=str(proc), cgroup_root=str(cg))
    assert cpu["cgroup_quota"] == 8.0 and cpu["effective"] <= 8
    (cg / "user.slice" / "vbt" / "memory.max").write_text("max\n")
    assert probe.memory_facts(proc_root=str(proc), cgroup_root=str(cg))["effective_mb"] == 512 * GiB_MB


def test_memory_facts_cgroup_v1_unlimited_is_no_limit(tmp_path):
    proc = _proc(tmp_path, total_kb=16 * 1024 * 1024, cgroup="4:memory:/docker/abc\n1:cpu,cpuacct:/docker/abc\n")
    cg = tmp_path / "cg"
    (cg / "memory" / "docker" / "abc").mkdir(parents=True)
    (cg / "memory" / "docker" / "abc" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    mem = probe.memory_facts(proc_root=str(proc), cgroup_root=str(cg))
    assert mem["cgroup_version"] == 1 and mem["cgroup_limit_mb"] is None and mem["effective_mb"] == 16 * GiB_MB
    (cg / "memory" / "docker" / "abc" / "memory.limit_in_bytes").write_text(str(6 * 1024 ** 3))
    assert probe.memory_facts(proc_root=str(proc), cgroup_root=str(cg))["effective_mb"] == 6 * GiB_MB


def test_gpu_facts_from_saved_nvidia_smi_output():
    g = probe.gpu_facts(H200x4_SMI, source="state/nvidia-smi.csv")
    assert g["count"] == 4 and g["driver_version"] == "580.65.06" and g["gpus"][0]["vram_gib"] > 139
    assert probe.gpu_facts("", source="file")["count"] == 0


def test_disk_facts_use_the_nearest_existing_directory(tmp_path):
    d = probe.disk_facts({"data": tmp_path / "not" / "yet", "state": tmp_path})
    assert d["data"]["mount_of"] == str(tmp_path) and d["data"]["free_bytes"] > 0
    assert d["state"].get("same_filesystem_as") == "data"


class _Ok(BaseHTTPRequestHandler):
    def do_HEAD(self):  # noqa: N802
        self.send_response(405)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        self.send_response(206)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_network_facts_reachable_and_unreachable():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        closed = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
        port = closed.server_address[1]
        closed.server_close()
        res = probe.network_facts({"up": f"http://127.0.0.1:{srv.server_address[1]}/x",
                                   "down": f"http://127.0.0.1:{port}/"}, timeout_s=3)
    finally:
        srv.shutdown()
    assert res["up"]["reachable"] is True and res["up"]["status"] == 206     # HEAD refused, one-byte GET answered
    assert res["down"]["reachable"] is False and "error" in res["down"]


def test_endpoint_urls_one_per_host_and_port(base_config):
    urls = probe.endpoint_urls(base_config, ["https://example.org/a", "https://example.org/b"])
    assert urls["model_server"].endswith(":8000/health") and urls["searxng"] == "http://localhost:8888"
    hosts = [u.split("/")[2] for u in urls.values()]
    assert len(hosts) == len(set(hosts))
    assert any("clinicaltrials.gov" in u for u in urls.values())      # from the descriptors
    assert sum("example.org" in u for u in urls.values()) == 1


def test_probe_host_is_json_ready(tmp_path, base_config):
    lay = resolve_layout(base_config, home=str(tmp_path / "h"), environ={})
    facts = probe.probe_host(lay, base_config, nvidia_smi_text=H100_SMI, nvidia_smi_source="test", network=False)
    json.dumps(facts)
    assert facts["gpu"]["count"] == 1 and facts["containment"]["limit_kind"] in ("cgroup", "rlimit_data", "none")
    assert all(v.get("skipped") for v in facts["network"].values())


# ------------------------------------------------------------------------------------------------ sizing / serving

def test_sizing_keeps_the_shipped_defaults_on_a_small_host(base_config):
    z = hostconfig.size_host(_facts(16094, 4), base_config)
    assert z["service_mem_limit_mb"] == 3000 and z["workspace_mb"] == 8000 and z["service_max_concurrency"] == 4
    assert z["host_budget_mb"] == int(0.75 * 16094 - 2048)
    assert z["default_server_mb"] == int(0.8 * (0.75 * 16094 - 2048)) < 12000   # never above what the host gives
    assert "limit_kind" not in z and z["bwrap"] is False


def test_sizing_is_the_rule_auto_applies_at_run_time(base_config):
    """``vbt setup`` writes the numbers the harness's ``auto`` gives on the same host (one rule:
    vbt.datalayer.memory.sizing), not a rule of its own (it gave 128,000 MB per server on 512 GB, the rule 293,601)."""
    from vbt.datalayer.memory import sizing

    for ram in (16 * GiB_MB, 64 * GiB_MB, 512 * GiB_MB):
        z = hostconfig.size_host(_facts(ram, 32), base_config)
        memory = base_config["data"]["memory"]
        assert z["host_budget_mb"] == int(sizing.host_budget_for(ram, memory))
        assert z["default_server_mb"] == sizing.server_limit_for(ram, memory)
        assert z["service_mem_limit_mb"] == sizing.data_child_for(ram)
    assert hostconfig.size_host(_facts(512 * GiB_MB, 32), base_config)["default_server_mb"] == 293601
    shared = {**base_config, "data": {**base_config["data"],
                                      "memory": {**base_config["data"]["memory"], "host_mb": 64 * GiB_MB}}}
    z = hostconfig.size_host(_facts(512 * GiB_MB, 32), shared)
    assert z["default_server_mb"] == 36700 and z["ram_mb"] == 64 * GiB_MB and "host_mb" in z["notes"][0]


def test_sizing_scales_to_a_large_host_and_to_the_measured_loads(base_config):
    ram = 1024 * GiB_MB
    z = hostconfig.size_host(_facts(ram, 96, cgroup=True, bwrap=True), base_config)
    assert z["host_budget_mb"] == int(0.75 * ram - 0.05 * ram)
    assert z["default_server_mb"] == int(0.8 * (0.75 * ram - 0.05 * ram))
    assert z["service_mem_limit_mb"] == 32768 and z["service_max_concurrency"] == 16
    parallel = int(base_config["limits"]["max_parallel_agents"])
    # the rules share one budget (DEP-14): what the host budget, the data child and the reserve leave, per agent
    assert z["workspace_mb"] == int((ram - z["host_budget_mb"] - 32768 - 0.05 * ram) / parallel) == 28672
    assert 0 <= z["full_load"]["left_mb"] and z["full_load"]["sum_mb"] <= ram
    assert z["limit_kind"] == "cgroup" and z["bwrap"] is True
    need = {"genetics": 80434.0, "target": 3474.0}
    z = hostconfig.size_host(_facts(ram, 96), base_config, server_need_mb=need)
    assert z["default_server_mb"] == int(0.8 * (0.75 * ram - 0.05 * ram)) and z["largest_server"]["server"] == \
        "genetics"                                          # the rule's limit already admits the largest load
    z = hostconfig.size_host(_facts(160 * GiB_MB, 32), base_config, server_need_mb={"genetics": 80434.0})
    # above 0.8 x budget: raised to admission's need, x1.3 safety plus an idle server's 300 MB baseline (ACC-1)
    assert z["default_server_mb"] == int(80434 * 1.3 + 300 + 1) < z["host_budget_mb"]
    z = hostconfig.size_host(_facts(96 * GiB_MB, 32), base_config, server_need_mb=need)
    assert z["default_server_mb"] == z["host_budget_mb"]
    assert any("genetics" in n and "too_large" in n for n in z["notes"])


def test_pick_serving_from_gpus_and_overrides():
    profiles = load_local_profiles()
    s = hostconfig.pick_serving(_facts(256 * GiB_MB, 32, smi=H100_SMI), profiles=profiles)
    assert s["serving_profile"] == "h100" and s["harness_profile"] == "local-h100"
    s = hostconfig.pick_serving(_facts(1024 * GiB_MB, 64, smi=H200x4_SMI), profiles=profiles)
    assert s["serving_profile"] == "dp" and s["data_parallel_size"] == 4 and s["harness_profile"] == "local-dp"
    s = hostconfig.pick_serving(_facts(64 * GiB_MB, 16))
    assert s["serving_profile"] is None and s["warnings"]
    s = hostconfig.pick_serving(_facts(64 * GiB_MB, 16), harness_profile="claude")
    assert s["harness_profile"] == "claude" and not s["warnings"]
    s = hostconfig.pick_serving(_facts(64 * GiB_MB, 16), serving_profile="h200", llm_url="http://gpu:8000/v1")
    assert s["serving_profile"] == "h200" and s["harness_profile"] == "local-h200"
    with pytest.raises(ValueError, match="unknown serving profile"):
        hostconfig.pick_serving(_facts(1, 1), serving_profile="nope")


def test_compose_vllm_service_matches_vbt_local_serve():
    profiles = load_local_profiles()
    doc = hostconfig.render_compose_vllm({"serving_profile": "h100", "driver_version": "580.65.06"},
                                         profiles=profiles)
    svc = doc["services"]["vllm"]
    spec = resolve_serve(profiles, "h100")
    assert svc["command"] == [*spec.server_argv, "--host", "0.0.0.0", "--port", str(spec.container_port)]
    assert svc["image"] == "${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
    assert svc["volumes"] == ["${VBT_HOME:?set VBT_HOME}/models:/root/.cache/huggingface"]
    assert svc["ports"] == ["${VLLM_BIND:-127.0.0.1}:${VLLM_PORT:-8000}:8000"]
    assert svc["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["${VLLM_GPU:-0}"]
    dp = hostconfig.render_compose_vllm({"serving_profile": "dp", "data_parallel_size": 4,
                                         "driver_version": "576.1"}, profiles=profiles)["services"]["vllm"]
    assert dp["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 4
    assert "--data-parallel-size" in dp["command"] and dp["image"].endswith("v0.31.0-cu129}")
    assert hostconfig.render_compose_vllm({"serving_profile": None}) is None


def test_host_files_roundtrip(tmp_path, base_config):
    lay = resolve_layout(base_config, home=str(tmp_path / "h"), environ={})
    z = hostconfig.size_host(_facts(256 * GiB_MB, 32, cgroup=True), base_config)
    prof = hostconfig.host_profile(z, lay, tool_env={"OPEN_TARGETS_DATA_PATH": "/srv/vbt/data/open_targets/25.09"})
    env = hostconfig.host_env(lay, {"serving_profile": "h100", "data_parallel_size": 1},
                              data_roots={"OPEN_TARGETS_DATA_PATH": "/srv/vbt/data/open targets/25.09"},
                              profiles=["local-h100", "production", str(lay.state / "host.yaml")], deploy="compose")
    files = hostconfig.write_host_files(lay.state, prof, env, {"services": {"vllm": {"image": "x"}}}, header="h1\nh2")
    assert [f.name for f in files] == ["host.yaml", "host.env", "compose.vllm.yaml"]
    assert oct(os.stat(lay.state / "host.env").st_mode & 0o777) == "0o640"
    back = hostconfig.read_env_file(lay.state / "host.env")
    assert back["OPEN_TARGETS_DATA_PATH"] == "/srv/vbt/data/open targets/25.09"
    assert back["VBT_LLM_BASE_URL"] == "http://vllm:8000/v1" and back["SEARXNG_URL"] == "http://searxng:8080"
    assert back["VBT_PROFILES"].split()[-1].endswith("host.yaml")
    host = yaml.safe_load((lay.state / "host.yaml").read_text())
    assert host["data"]["memory"]["limit_kind"] == "cgroup" and host["paths"]["runs_dir"] == str(lay.runs)
    cfg = load_config([str(lay.state / "host.yaml")])          # host.yaml is a valid harness profile
    assert cfg["data"]["memory"]["host_budget_mb"] == z["host_budget_mb"]
    assert cfg["tool_env"]["OPEN_TARGETS_DATA_PATH"] == "/srv/vbt/data/open_targets/25.09"
    hostconfig.write_host_files(lay.state, prof, env, None, header="h")
    assert not (lay.state / "compose.vllm.yaml").exists()


# ------------------------------------------------------------------------------------------------ needs

def test_needs_of_the_shipped_roster(base_config):
    needs = compute_needs(base_config)
    assert not needs.errors and len(needs.tools) > 50
    assert "data" not in needs.servers                      # the native tools are not a reason to fetch data
    assert needs.sources["open_targets"]["root_var"] == "OPEN_TARGETS_DATA_PATH"
    assert needs.sources["open_targets"]["kind"] == "local" and needs.sources["open_targets"]["release"] == "25.09"
    assert needs.sources["clinicaltrials_gov"]["kind"] == "remote"
    full = needs.server_full_loads()
    assert "open_targets.target" in full["target"] and set(needs.local_tables()) <= set(needs.tables)
    for t in needs.tables:
        assert t.split(".", 1)[0] in needs.sources


def test_needs_follow_the_enabled_servers(base_config):
    cfg = json.loads(json.dumps(base_config, default=str))
    for s in cfg["mcp_servers"]["servers"]:
        if s["name"] != "drug":
            s["enabled"] = False
    needs = compute_needs(cfg)
    assert set(needs.servers) == {"drug"} and set(needs.sources) == {"open_targets"}


def test_root_variables():
    assert root_variables("${OPEN_TARGETS_DATA_PATH}") == "OPEN_TARGETS_DATA_PATH"
    assert root_variables("${DEPMAP_DATA_PATH:-${vars.project_root}/data/depmap}") == "DEPMAP_DATA_PATH"
    assert root_variables("${VBT_ZENODO_DIR:-x}/virtualbiotech_submission") is None
    assert root_variables(None) is None


# ------------------------------------------------------------------------------------------------ state / helpers

def test_state_resume_and_rates(tmp_path):
    st = SetupState(tmp_path)
    fp = fingerprint("check", [1, 2])
    st.record("check", status="done", fingerprint=fp)
    st.observe_rate("check", 100.0)
    st.observe_rate("check", 200.0)
    again = SetupState(tmp_path)
    assert again.is_current("check", fp) and not again.is_current("check", fingerprint("check", [3]))
    assert again.rate("check") == 150.0
    (tmp_path / "setup-state.json").write_text("{not json")
    assert SetupState(tmp_path).steps == {}


def test_select_steps():
    assert select_steps() == STEP_NAMES
    assert select_steps(only="check,probe") == ["probe", "check"]
    assert select_steps(from_step="index", skip="smoke") == ["index", "check", "calibrate"]
    with pytest.raises(ValueError, match="unknown step"):
        select_steps(only="dowload")


def test_parse_estimate_and_acquisition_plans():
    out = ("open_targets.target: 78726 rows, 10 fragments, 75.6 MB on disk; upstream full load ~3170 MB, projected "
           "scan ~234 MB\nopen_targets.variant: error: no files\nhost memory: 16094 MB\n")
    sizes, errors = parse_estimate(out)
    assert sizes == {"open_targets.target": 3170.0} and "open_targets.variant" in errors
    # the shape of `vbt data acquire --plan --json` (AcquisitionPlan.to_dict)
    ot = {"source": "open_targets", "mode": "download", "release": "25.09", "home": "/d/sources/open_targets/25.09",
          "bytes_remaining": 31_000_000_000, "states": {"verified": 0, "present": 0, "partial": 0, "missing": 3508},
          "prepare": {}, "env": {"OPEN_TARGETS_DATA_PATH": "/d/sources/open_targets/25.09"}}
    tahoe = {"source": "tahoe_100m", "mode": "download", "bytes_remaining": 0, "prepare": {"de": "pending"},
             "states": {"verified": 3, "missing": 0, "partial": 0}, "env": {"TAHOE_DATA_PATH": "/d/t"}}
    census = {"source": "cellxgene_census", "mode": "remote", "bytes_remaining": None, "env": {}}
    plan = {"bytes_remaining": None, "sources": [ot, tahoe, census]}
    assert _bytes_to_fetch(plan) == 31_000_000_000 and _bytes_to_fetch({"bytes_remaining": 7}) == 7
    assert _bytes_to_fetch({"bytes_remaining": 5, "sources": [ot]}) == 5
    assert _plan_roots(plan) == {"OPEN_TARGETS_DATA_PATH": "/d/sources/open_targets/25.09", "TAHOE_DATA_PATH": "/d/t"}
    assert _plan_pending(plan) == ["open_targets: files to fetch", "tahoe_100m: prepare de"]
    assert _plan_pending({"sources": [dict(ot, bytes_remaining=0, states={"verified": 3508})]}) == []
    assert _unavailable(subprocess.CompletedProcess([], 2, "", "vbt data: error: argument data_source: invalid "
                                                               "choice: 'acquire' (choose from 'ot', 'zenodo')"))
    assert not _unavailable(subprocess.CompletedProcess([], 2, "", "error: disk full"))


# ------------------------------------------------------------------------------------------------ steps (fake runner)

def _ctx(tmp_path, config, runner, **opts):
    args = _args(tmp_path, **opts)
    ctx = build_context(args, config, out=open(os.devnull, "w"))
    ctx.runner = runner
    ctx.facts = {**_facts(64 * GiB_MB, 16), "network": {}, "disk": {}, "hostname": "test"}
    return ctx


ESTIMATE_OUT = "\n".join(f"open_targets.{t}: 1 rows, 1 fragments, 1.0 MB on disk; upstream full load ~{mb} MB, "
                         "projected scan ~1 MB" for t, mb in (("target", 3170), ("l2g_prediction", 12126),
                                                              ("study", 3921), ("known_drug", 494)))


INDEX_OUT = ("built open_targets:ensembl_gene: 465286 rows -> /d/.vbt-datalayer/open_targets/x\n"
             "failed tahoe_100m:tahoe_drug: TableUnavailable: tahoe_100m.drug_metadata: no data files under "
             "'metadata/drug_metadata.parquet' (missing; an empty location is not an empty table)\n")


def test_index_step_tells_absent_data_from_errors(tmp_path, base_config):
    broken = INDEX_OUT + "failed open_targets:ot_disease: ArrowInvalid: corrupt footer\n"
    ctx = _ctx(tmp_path, base_config, FakeRunner({"index build": (1, broken, "")}))
    assert run_steps(ctx, ["probe", "configure", "index"]) == 1
    rec = SetupState(ctx.layout.state).steps["index"]
    assert rec["status"] == "failed" and "corrupt footer" in rec["detail"]


def test_a_roster_without_upstream_tools_fetches_and_checks_nothing(tmp_path, base_config):
    from vbt.setup.needs import Needs

    runner = FakeRunner()
    ctx = _ctx(tmp_path, base_config, runner)
    ctx.needs = Needs()                                       # e.g. the mock profile: no MCP servers
    assert run_steps(ctx, ["probe", "configure", "acquire", "size", "check", "calibrate"]) == 0
    st = SetupState(ctx.layout.state).steps
    assert st["acquire"]["detail"] == "nothing to fetch" and st["check"]["detail"] == "no enabled tool reads a table"
    assert not runner.ran("data acquire") and not runner.ran("ds check") and not runner.ran("ds estimate")


def test_run_steps_records_resumes_and_stops_on_failure(tmp_path, base_config, monkeypatch):
    monkeypatch.delenv("OPEN_TARGETS_DATA_PATH", raising=False)
    check = json.dumps({"tables": {"open_targets.target": {"status": "ready"}, "open_targets.go": {"status":
                        "not_ready"}}, "tools": {"unready": {"mcp__pathway__x": {}}}, "table_errors": [],
                        "quarantined": []})
    runner = FakeRunner({"data acquire": (2, "", "vbt data: error: argument data_source: invalid choice: 'acquire'"),
                         "ds estimate": (0, ESTIMATE_OUT, ""), "ds check": (0, check, ""),
                         "index build": (1, INDEX_OUT, "")})
    ctx = _ctx(tmp_path, base_config, runner)
    # data fetched earlier is found at its acquisition home (DEP-9: a home that is not there names no root)
    (ctx.layout.data / "sources" / "open_targets" / "25.09").mkdir(parents=True)
    rc = run_steps(ctx, select_steps(skip="smoke"))
    assert rc == 0
    st = SetupState(ctx.layout.state)
    assert st.steps["index"]["data"]["built"] == ["open_targets:ensembl_gene"]
    assert st.steps["index"]["data"]["absent_data"] == ["tahoe_100m:tahoe_drug"]
    assert {k: v["status"] for k, v in st.steps.items()} == {
        "probe": "done", "configure": "done", "acquire": "unavailable", "size": "done", "index": "done",
        "check": "done", "calibrate": "done"}
    assert st.steps["size"]["data"]["server_need_mb"]["genetics"] == 12126 + 3921
    host = yaml.safe_load((ctx.layout.state / "host.yaml").read_text())
    assert host["data"]["memory"]["default_server_mb"] >= 12000
    env = hostconfig.read_env_file(ctx.layout.state / "host.env")
    assert env["OPEN_TARGETS_DATA_PATH"] == str(ctx.layout.data / "sources" / "open_targets" / "25.09")
    # every data command ran under the host configuration
    for c in runner.ran("ds "):
        assert c["argv"][-1 - c["argv"][::-1].index("--profile") + 1].endswith("host.yaml") or \
            str(ctx.layout.state / "host.yaml") in c["argv"]
        assert c["env"]["OPEN_TARGETS_DATA_PATH"] == env["OPEN_TARGETS_DATA_PATH"]
    check_argv = runner.ran("ds check")[0]["argv"]
    assert check_argv.count("--table") == len(ctx.needs.tables) and "open_targets.target" in check_argv
    acq = runner.ran("data acquire")[0]["argv"]           # `--for-tools T [T ...]`, then the plan flags
    i = acq.index("--for-tools")                          # (--offline: these tests run with --no-network)
    assert acq[i + 1:i + 1 + len(ctx.needs.tools)] == ctx.needs.tools
    assert acq[-3:] == ["--plan", "--json", "--offline"]
    # resume: nothing changed, the data steps are skipped
    runner2 = FakeRunner(runner.replies)
    ctx2 = _ctx(tmp_path, base_config, runner2)
    assert run_steps(ctx2, select_steps(skip="smoke")) == 0
    assert not runner2.ran("index build") and not runner2.ran("ds check")
    # --force runs them again; a failed step stops the later ones
    runner3 = FakeRunner({**runner.replies, "index build": (1, "", "boom")})
    ctx3 = _ctx(tmp_path, base_config, runner3)
    assert run_steps(ctx3, ["index", "check"], force=True) == 1
    assert runner3.ran("index build") and not runner3.ran("ds check")
    assert SetupState(ctx.layout.state).steps["index"]["status"] == "failed"
    assert SetupState(ctx.layout.state).steps["index"]["fingerprint"] is None
    runner4 = FakeRunner({**runner.replies, "index build": (1, "", "boom")})
    assert run_steps(_ctx(tmp_path, base_config, runner4), ["index", "check"], force=True, keep_going=True) == 1
    assert runner4.ran("ds check")


def test_new_data_reruns_the_data_steps(tmp_path, base_config, monkeypatch):
    ot = tmp_path / "ot"
    (ot / "target").mkdir(parents=True)
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", str(ot))
    replies = {"data acquire": (2, "", "invalid choice"), "ds estimate": (0, ESTIMATE_OUT, ""),
               "ds check": (0, json.dumps({"tables": {"a.b": {"status": "ready"}}}), "")}
    ctx = _ctx(tmp_path, base_config, FakeRunner(replies))
    assert run_steps(ctx, ["probe", "configure", "check"]) == 0
    assert hostconfig.read_env_file(ctx.layout.state / "host.env")["OPEN_TARGETS_DATA_PATH"] == str(ot)
    runner = FakeRunner(replies)
    assert run_steps(_ctx(tmp_path, base_config, runner), ["check"]) == 0 and not runner.ran("ds check")
    (ot / "target" / "part-0.parquet").write_bytes(b"PAR1")
    runner = FakeRunner(replies)
    assert run_steps(_ctx(tmp_path, base_config, runner), ["check"]) == 0 and runner.ran("ds check")


def test_acquisition_plan_sizes_and_disk_guard(tmp_path, base_config):
    plan = json.dumps({"bytes_remaining": 10 ** 18, "sources": [
        {"source": "open_targets", "mode": "download", "bytes_remaining": 10 ** 18,
         "env": {"OPEN_TARGETS_DATA_PATH": str(tmp_path / "x")}}]})
    runner = FakeRunner({"--plan": (0, plan, ""), "data acquire": (0, "", "")})
    ctx = _ctx(tmp_path, base_config, runner)
    out = make_plan(ctx, STEP_NAMES)
    row = next(r for r in out["steps"] if r["step"] == "acquire")
    assert row["action"] == "run" and row["bytes"] == 10 ** 18 and row["seconds"] > 0
    assert any("to fetch but" in n for n in out["notes"]) or out["free_bytes"] is None
    assert not (ctx.layout.state / "host.yaml").exists()            # --plan writes nothing
    res = run_steps(ctx, ["probe", "configure", "acquire"])
    assert res == 1 and "free under" in SetupState(ctx.layout.state).steps["acquire"]["detail"]
    acquire_calls = [c for c in runner.ran("data acquire") if "--plan" not in c["argv"]]
    assert not acquire_calls                                        # refused before fetching anything


def test_smoke_defers_until_the_model_server_answers(tmp_path, base_config):
    runner = FakeRunner({"run -q": (0, "hello", ""), "doctor": (0, "PASS", "")})
    ctx = _ctx(tmp_path, base_config, runner, llm_url="http://127.0.0.1:9/v1")
    assert run_steps(ctx, ["probe", "configure", "smoke"]) == 0
    rec = SetupState(ctx.layout.state).steps["smoke"]
    assert rec["status"] == "deferred" and "does not answer" in rec["detail"]
    mock = runner.ran("run -q")[0]["argv"]
    assert mock[:2] == ["--profile", "mock"] and "--smoke" not in " ".join(runner.ran("doctor")[0]["argv"])
    runner = FakeRunner({"run -q": (1, "", "boom"), "doctor": (0, "PASS", "")})
    ctx = _ctx(tmp_path, base_config, runner, llm_url="http://127.0.0.1:9/v1")
    assert run_steps(ctx, ["smoke"]) == 1


def test_smoke_fails_on_an_incomplete_analysis_stack(tmp_path, base_config, monkeypatch):
    doctor = ("[ok] analysis: python import scanpy: 1.11.5\n[!!] data: open_targets.variant: missing\n"
              "[!!] analysis: python import rpy2: ModuleNotFoundError: No module named 'rpy2'\nFAIL\n")
    rbin = tmp_path / "rbin"                                             # a host with R: rpy2 missing is a failure
    rbin.mkdir()
    (rbin / "Rscript").write_text("#!/bin/sh\nexit 0\n")
    (rbin / "Rscript").chmod(0o755)
    path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{rbin}{os.pathsep}{path}")
    runner = FakeRunner({"run -q": (0, "hello", ""), "doctor": (1, doctor, "")})
    ctx = _ctx(tmp_path, base_config, runner, llm_url="http://127.0.0.1:9/v1", no_analysis=False)
    assert run_steps(ctx, ["probe", "configure", "smoke"]) == 1
    rec = SetupState(ctx.layout.state).steps["smoke"]
    assert "rpy2" in rec["detail"] and "--no-analysis" in rec["detail"] and rec["data"]["analysis"] == {
        "ok": 1, "failed": ["python import rpy2: ModuleNotFoundError: No module named 'rpy2'"]}
    # DEP-11: a pip-only install has no R by design: the same report is a warning, the step is not failed
    monkeypatch.setenv("PATH", os.pathsep.join(d for d in path.split(os.pathsep)
                                               if d and not (Path(d) / "Rscript").exists()))
    runner = FakeRunner({"run -q": (0, "hello", ""), "doctor": (1, doctor, "")})
    ctx = _ctx(tmp_path, base_config, runner, llm_url="http://127.0.0.1:9/v1", no_analysis=False)
    assert run_steps(ctx, ["smoke"]) == 0
    rec = SetupState(ctx.layout.state).steps["smoke"]
    assert rec["status"] == "deferred" and "pip-only" in rec["data"]["analysis"]["warning"]


def test_smoke_confirms_a_reported_missing_r_package_before_failing(tmp_path, base_config, monkeypatch):
    rbin = tmp_path / "rbin"
    rbin.mkdir()
    (rbin / "Rscript").write_text("#!/bin/sh\nexit 0\n")
    (rbin / "Rscript").chmod(0o755)
    monkeypatch.setenv("PATH", f"{rbin}{os.pathsep}{os.environ.get('PATH', '')}")
    doctor = ("[ok] analysis: R package lme4: /opt/conda/bin/Rscript\n[!!] analysis: R package lmerTest: not installed\n"
              "[!!] analysis: R package MuMIn: not installed\nFAIL\n")
    replies = {"run -q": (0, "hello", ""), "doctor": (1, doctor, ""),
               'requireNamespace("lmerTest"': (0, "TRUE", ""), 'requireNamespace("MuMIn"': (0, "FALSE", "")}
    runner = FakeRunner(replies)
    ctx = _ctx(tmp_path, base_config, runner, llm_url="http://127.0.0.1:9/v1", no_analysis=False)
    assert run_steps(ctx, ["probe", "configure", "smoke"]) == 1
    rec = SetupState(ctx.layout.state).steps["smoke"]
    assert rec["data"]["analysis"]["failed"] == ["R package MuMIn: not installed"]
    assert rec["data"]["analysis"]["r_loaded_on_recheck"] == ["lmerTest"]
    assert [c["argv"][0] for c in runner.ran("requireNamespace")] == [str(rbin / "Rscript")] * 2
    replies['requireNamespace("MuMIn"'] = (0, "TRUE", "")                # R loads both: the step is not failed
    assert run_steps(_ctx(tmp_path, base_config, FakeRunner(replies), llm_url="http://127.0.0.1:9/v1",
                          no_analysis=False), ["smoke"]) == 0
    assert SetupState(ctx.layout.state).steps["smoke"]["status"] == "deferred"


def test_acquire_writes_the_reported_roots_and_resumes(tmp_path, base_config, monkeypatch):
    monkeypatch.delenv("OPEN_TARGETS_DATA_PATH", raising=False)
    home = tmp_path / "acq" / "open_targets" / "25.09"
    plan = json.dumps({"bytes_remaining": 229_577, "rate_mbps": 80.0, "rate_measured": True, "sources": [
        {"source": "open_targets", "mode": "download", "bytes_remaining": 229_577, "prepare": {},
         "states": {"missing": 2}, "env": {"OPEN_TARGETS_DATA_PATH": str(home)}}]})
    runner = FakeRunner({"--plan": (0, plan, ""), "data acquire": (0, "2 file(s) downloaded", "")})
    ctx = _ctx(tmp_path, base_config, runner)
    row = next(r for r in make_plan(ctx, STEP_NAMES)["steps"] if r["step"] == "acquire")
    assert "229.58 KB to fetch at the measured 80 MB/s" in row["detail"]
    assert run_steps(ctx, ["probe", "configure", "acquire"]) == 0
    rec = SetupState(ctx.layout.state)
    assert rec.steps["acquire"]["data"]["roots"] == {"OPEN_TARGETS_DATA_PATH": str(home)}
    assert "download_bps" not in (rec.data.get("rates") or {})        # a 230 KB transfer measures start-up
    env = hostconfig.read_env_file(ctx.layout.state / "host.env")
    assert env["OPEN_TARGETS_DATA_PATH"] == str(home)                  # host.env follows the acquisition
    fetches = [c for c in runner.ran("data acquire") if "--plan" not in c["argv"]]
    assert len(fetches) == 1
    runner2 = FakeRunner(runner.replies)                                # nothing changed: up to date
    assert run_steps(_ctx(tmp_path, base_config, runner2), ["probe", "configure", "acquire"]) == 0
    assert not runner2.ran("data acquire")
    home.mkdir(parents=True)                                            # files changed: the step runs again
    (home / "partial.parquet").write_bytes(b"x")
    runner3 = FakeRunner(runner.replies)
    assert run_steps(_ctx(tmp_path, base_config, runner3), ["probe", "configure", "acquire"]) == 0
    assert runner3.ran("data acquire")


def test_data_roots_keep_a_set_variable_else_use_the_acquisition_home(tmp_path, base_config, monkeypatch):
    """DEP-9: the fallback is the descriptor's acquisition home and env template (Tahoe: tahoe/<rev>/prepared, where
    `vbt data acquire tahoe_100m` puts it), and only for a home on this host: a source the host lacks gets no
    variable (doctor reports it absent instead of failing a path setup invented)."""
    monkeypatch.delenv("OPEN_TARGETS_DATA_PATH", raising=False)
    monkeypatch.delenv("TAHOE_DATA_PATH", raising=False)
    ctx = _ctx(tmp_path, base_config, FakeRunner())
    ctx.needs = compute_needs(base_config)
    assert "TAHOE_DATA_PATH" not in data_roots(ctx) and "OPEN_TARGETS_DATA_PATH" not in data_roots(ctx)
    tahoe_rev = yaml.safe_load((PROJECT_ROOT / "configs/data/sources/tahoe.yaml").read_text())["acquisition"]["release"]
    ot = ctx.layout.data / "sources" / "open_targets" / "25.09"
    prepared = ctx.layout.data / "sources" / "tahoe" / tahoe_rev / "prepared"
    ot.mkdir(parents=True)
    prepared.mkdir(parents=True)
    roots = data_roots(ctx)
    assert roots["OPEN_TARGETS_DATA_PATH"] == str(ot)
    assert roots["TAHOE_DATA_PATH"] == str(prepared)
    monkeypatch.setenv("OPEN_TARGETS_DATA_PATH", "/mnt/ot")
    assert data_roots(ctx)["OPEN_TARGETS_DATA_PATH"] == "/mnt/ot"


def test_a_fresh_plan_runs_the_real_acquisition_plan_child(tmp_path, capsys, monkeypatch):
    """DEP-1/RR-4: on a fresh state directory `vbt setup --plan` handed the planning child VBT_PROFILES ending in
    <state>/host.yaml, which configure had not written: `the acquisition plan failed (exit 1): config/profile not
    found`, exit 0. The real child runs here (no FakeRunner), with the host configuration applied as in production."""
    monkeypatch.delenv("VBT_NO_HOST_ENV", raising=False)            # the child applies host.env, as vbt does
    monkeypatch.setenv("VBT_HOME", str(tmp_path / "home"))
    config = load_config(["production"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")}})
    rc = cmd_setup(_args(tmp_path, plan=True, json=True, only="probe,configure,acquire"), config)
    plan = json.loads(capsys.readouterr().out)
    row = next(r for r in plan["steps"] if r["step"] == "acquire")
    assert row["action"] not in ("failed", "unknown"), row["detail"]
    assert row["action"] == "run" and row["bytes"] > 0 and rc == 0
    state = tmp_path / "home" / "state"
    assert not (state / "host.yaml").exists() and not (state / "logs").exists()   # --plan changes nothing


def test_a_failed_step_plan_makes_the_plan_fail(tmp_path, base_config, capsys, monkeypatch):
    import vbt.setup.steps as steps_mod

    monkeypatch.setattr(steps_mod, "_run", FakeRunner({"--plan": (1, "", "Traceback: boom")}))
    rc = cmd_setup(_args(tmp_path, plan=True, json=True, only="probe,configure,acquire"), base_config)
    row = next(r for r in json.loads(capsys.readouterr().out)["steps"] if r["step"] == "acquire")
    assert rc == 1 and row["action"] == "failed" and "boom" in row["detail"]


def test_a_recorded_profile_file_that_is_gone_is_skipped_not_a_traceback(tmp_path, monkeypatch, capsys):
    """RR-4: host.env naming a host.yaml that was moved made every vbt command (doctor included) exit 1 with a raw
    FileNotFoundError traceback."""
    from vbt.cli import apply_host_config, main

    state = tmp_path / "S"
    state.mkdir()
    (state / "host.env").write_text(f"VBT_PROFILES='mock {state / 'host.yaml'}'\n")    # quoted as setup writes it
    env = {"VBT_STATE_DIR": str(state)}
    args = SimpleNamespace(cmd="tools", profile=[])
    assert apply_host_config(args, env) == state / "host.env"
    assert args.profile == ["mock"] and "does not exist; skipped" in capsys.readouterr().err
    monkeypatch.delenv("VBT_NO_HOST_ENV", raising=False)
    assert main(["--profile", str(tmp_path / "nope.yaml"), "tools"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cmd_setup_plan_json_and_status(tmp_path, base_config, capsys, monkeypatch):
    import vbt.setup.steps as steps_mod

    monkeypatch.setattr(steps_mod, "_run", FakeRunner({"data acquire": (2, "", "invalid choice")}))
    rc = cmd_setup(_args(tmp_path, plan=True, json=True, nvidia_smi_file=None), base_config)
    plan = json.loads(capsys.readouterr().out)
    assert rc == 0 and [r["step"] for r in plan["steps"]] == STEP_NAMES
    assert plan["env"]["VBT_DATA_DIR"] == str((tmp_path / "home" / "data").resolve())
    assert next(r for r in plan["steps"] if r["step"] == "acquire")["action"] == "unavailable"
    assert cmd_setup(_args(tmp_path, status=True), base_config) == 0
    assert "state:" in capsys.readouterr().out
    assert cmd_setup(_args(tmp_path, only="nope"), base_config) == 2


def test_vbt_setup_is_registered():
    from vbt.cli import build_parser

    args = build_parser().parse_args(["--profile", "production", "setup", "--plan", "--serving-profile", "h200"])
    assert args.cmd == "setup" and args.plan and args.serving_profile == "h200" and args.profile == ["production"]


# ------------------------------------------------------------------------------------------------ entrypoint script

@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available")
def test_vbt_host_loads_host_env_and_profiles(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "host.env").write_text("# written by vbt setup\nVBT_PROFILES='local-h100 production /s/host.yaml'\n"
                                    "OPEN_TARGETS_DATA_PATH='/srv/vbt/data/open targets/25.09'\nKEEP=from_file\n"
                                    "EMPTY=from_file\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "vbt").write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$OUT\"\n"
                              "printf '%s|%s|%s\\n' \"$OPEN_TARGETS_DATA_PATH\" \"$KEEP\" \"$EMPTY\" >> \"$OUT\"\n")
    (fake / "vbt").chmod(0o755)
    out = tmp_path / "out.txt"
    env = {"PATH": f"{fake}:/usr/bin:/bin", "VBT_STATE_DIR": str(state), "OUT": str(out), "KEEP": "from_env",
           "EMPTY": "", "VBT_BASE_PROFILES": "production"}
    res = subprocess.run(["bash", str(FULL / "vbt-host"), "chat", "--resume", "latest"], env=env,
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    lines = out.read_text().splitlines()
    assert lines[:-1] == ["--profile", "local-h100", "--profile", "production", "--profile", "/s/host.yaml", "chat",
                          "--resume", "latest"]
    assert lines[-1] == "/srv/vbt/data/open targets/25.09|from_env|from_file"   # set wins, empty does not
    res = subprocess.run(["bash", str(FULL / "vbt-host"), "setup", "--plan"], env=env, capture_output=True,
                         text=True, timeout=30)
    assert res.returncode == 0 and out.read_text().splitlines()[:4] == ["--profile", "production", "setup", "--plan"]


# ------------------------------------------------------------------------------------------------ deployment files

def _conda_names(spec_lines):
    out = set()
    for item in spec_lines:
        if isinstance(item, str):
            out.add(re.split(r"[=<>! ]", item.strip(), maxsplit=1)[0])
    return out


def test_image_environment_covers_the_root_environment():
    root = yaml.safe_load((PROJECT_ROOT / "environment.yml").read_text())["dependencies"]
    image = yaml.safe_load((FULL / "environment.yml").read_text())["dependencies"]
    pip_in = (FULL / "requirements.in").read_text().lower()
    root_pip = next((d["pip"] for d in root if isinstance(d, dict)), [])
    # packages the image provides another way (documented in deploy/full/environment.yml / requirements.in)
    other_way = {"matplotlib": "matplotlib-base", "tiledbsoma": "cellxgene-census", "pydeseq2": "pip",
                 "decoupler": "pip", "liana": "pip", "lifelines": "pip", "httpx": "pip", "python-dotenv": "pip",
                 "jsonschema": None, "cell2location": "optional"}
    have = _conda_names(image)
    for name in sorted(_conda_names(root) - {"pip"}):
        assert name in have or name in other_way or name in pip_in, f"{name} of environment.yml is not in the image"
    for item in root_pip:
        name = re.split(r"[=<>! ]", item, maxsplit=1)[0].lower()
        assert name in pip_in or name in other_way, f"pip {name} of environment.yml is not in requirements.in"


def test_image_pins_follow_the_upstream_environment(base_config):
    up = Path(base_config["vars"]["upstream"]) / "environment.yml"
    if not up.is_file():
        pytest.skip("upstream submodule not checked out")
    upstream = yaml.safe_load(up.read_text())["dependencies"]
    image = "\n".join(str(d) for d in yaml.safe_load((FULL / "environment.yml").read_text())["dependencies"])
    pip_in = (FULL / "requirements.in").read_text()
    for d in upstream:
        if isinstance(d, str) and "=" in d:
            name, ver = d.split("=", 1)
            if name in ("python", "matplotlib"):
                continue
            assert f"{name}={ver}" in image, f"upstream pins {d}"
    agent_only = {"anthropic", "claude-agent-sdk", "gradio"}         # the harness replaces the upstream agents and app
    for d in next(x["pip"] for x in upstream if isinstance(x, dict)):
        name = d.split("==")[0]
        if name not in agent_only:
            assert d in pip_in, f"upstream pins {d}"


def test_locks_are_complete_and_pip_never_replaces_conda():
    conda = [ln for ln in (FULL / "conda-linux-64.lock").read_text().splitlines() if ln and not ln.startswith("#")]
    assert conda[0] == "@EXPLICIT"
    pkgs = conda[1:]
    assert len(pkgs) > 300 and all(re.match(r"https://conda\.anaconda\.org/\S+#[0-9a-f]{32}$", p) for p in pkgs)
    names = {re.sub(r"-[^-]+-[^-]+\.(conda|tar\.bz2)$", "", p.split("#")[0].rsplit("/", 1)[1]) for p in pkgs}
    for must in ("python", "numpy", "pyarrow", "r-base", "cellxgene-census", "tini", "bubblewrap"):
        assert must in names, must
    assert not ({"mkl", "jupyterlab", "qt6-main"} & names)          # trimmed out of a server image
    pip = [ln for ln in (FULL / "requirements.lock").read_text().splitlines() if ln and not ln.startswith("#")]
    assert pip and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[^\s=]+", ln) for ln in pip)
    norm = {re.sub(r"[-_.]+", "-", ln.split("==")[0]).lower() for ln in pip}
    assert {"fastmcp", "mcp", "pybioportal", "anthropic", "starlette", "uvicorn"} <= norm
    assert not (norm & {"numpy", "scipy", "pandas", "pyarrow", "scanpy", "anndata"})   # conda owns them


def test_dockerfile_and_apptainer_definition_share_the_install_scripts():
    docker = (FULL / "Dockerfile").read_text()
    apptainer = (FULL / "vbt-harness.def").read_text()
    for script in ("install-env.sh", "install-harness.sh"):
        assert script in docker and script in apptainer and (FULL / script).is_file()
    for f in re.findall(r"deploy/full/[\w.\-]+", docker):
        assert (PROJECT_ROOT / f).exists(), f
    assert "FROM env AS harness" in docker and "AS lock" in docker
    assert 'ENTRYPOINT ["tini", "-g", "--", "/opt/vbt/deploy/full/vbt-host"]' in docker
    assert "Stage: micromamba" in apptainer and "From: ubuntu:22.04" in apptainer
    assert "third_party /opt/vbt/third_party" in apptainer
    for text in (docker, apptainer):           # relative runs/ and data/ land on the volumes before setup
        assert "ln -s /srv/vbt/runs /opt/vbt/runs" in text and "ln -s /srv/vbt/data /opt/vbt/data" in text
    ignore = (PROJECT_ROOT / ".dockerignore").read_text().split()
    for must in (".git", "data/", "runs/", ".env", "**/.env"):
        assert must in ignore


def test_compose_file_shape():
    doc = yaml.safe_load((FULL / "compose.yaml").read_text())
    s = doc["services"]
    assert set(s) == {"vbt", "setup", "searxng"} and s["setup"]["profiles"] == ["setup"]
    assert s["vbt"]["ports"] == ["${VBT_WEB_BIND:-127.0.0.1}:${VBT_WEB_PORT:-7860}:7860"]
    assert "ports" not in s["searxng"]
    vols = s["vbt"]["volumes"]
    assert all(v.startswith("${VBT_HOME:?set VBT_HOME}/") and v.split(":")[-1].startswith("/srv/vbt/") for v in vols)
    assert s["setup"]["command"] == ["setup", "--deploy", "compose"]
    assert s["vbt"]["environment"]["SEARXNG_URL"] == "http://searxng:8080"
    # SearxNG on IPv4 (its image binds [::] and exits where IPv6 is off) with a health check `deploy.sh setup` waits on
    assert s["searxng"]["environment"]["GRANIAN_HOST"] == "0.0.0.0"
    assert "/healthz" in " ".join(s["searxng"]["healthcheck"]["test"])
    assert "ulimits" not in s["vbt"]          # a fixed limit above the daemon's own stops the container from starting


@pytest.mark.skipif(not shutil.which("docker"), reason="docker not installed")
def test_compose_file_is_valid_for_docker_compose(tmp_path):
    ok = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, timeout=30)
    if ok.returncode != 0:
        pytest.skip("docker compose not available")
    vllm = hostconfig.render_compose_vllm({"serving_profile": "h100"})
    (tmp_path / "compose.vllm.yaml").write_text(yaml.safe_dump(vllm))
    env = {**os.environ, "VBT_HOME": str(tmp_path), "SEARXNG_SECRET": "test-secret"}
    res = subprocess.run(["docker", "compose", "--project-directory", str(FULL), "-f", str(FULL / "compose.yaml"),
                          "-f", str(tmp_path / "compose.vllm.yaml"), "--profile", "setup", "config", "--services"],
                         capture_output=True, text=True, timeout=60, env=env)
    assert res.returncode == 0, res.stderr
    assert set(res.stdout.split()) == {"vbt", "setup", "searxng", "vllm"}


SANDBOX_MARKERS = ("/home/user", "/tmp/claude", "44615", ".ccr", "agentproxy", "mirror.gcr.io")


def test_no_sandbox_specific_values_in_the_deployment():
    files = [*FULL.rglob("*"), *WORKFLOWS.glob("*.yml"), PROJECT_ROOT / ".dockerignore",
             PROJECT_ROOT / "docs" / "DEPLOYMENT.md", *(PROJECT_ROOT / "src" / "vbt" / "setup").glob("*.py"),
             *(PROJECT_ROOT / "configs" / "profiles").glob("production*.yaml")]
    for f in files:
        if f.is_file() and f.suffix not in (".lock",):
            text = f.read_text()
            for marker in SANDBOX_MARKERS:
                assert marker not in text, f"{f.relative_to(PROJECT_ROOT)} contains {marker!r}"


def test_workflows():
    if not WORKFLOWS.parent.is_dir():
        pytest.skip("no .github in this tree (the harness image leaves it out)")
    ci = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
    on = ci.get("on", ci.get(True))
    assert "push" in on and "pull_request" in on
    steps = " ".join(str(s.get("run", "")) for j in ci["jobs"].values() for s in j["steps"])
    assert "ruff check" in steps and "pytest" in steps and "tests/datalayer" in steps
    assert any(s.get("with", {}).get("cache") == "pip" for s in ci["jobs"]["test"]["steps"])
    real = yaml.safe_load((WORKFLOWS / "real-data.yml").read_text())
    assert set(real.get("on", real.get(True))) == {"workflow_dispatch"}
    text = (WORKFLOWS / "real-data.yml").read_text()
    assert "VBT_DL_NETWORK" in text and "VBT_DL_REAL_DATA" in text


def test_production_profile_loads_and_keeps_the_guards():
    cfg = load_config(["local-h200", "production"])
    assert cfg["data"]["gateway"]["mode"] == "enforce" and cfg["preflight"]["allow_missing_data"] is False
    assert cfg["mcp"]["inherit_env"] is False and cfg["provider"]["name"] == "vllm"


def test_deploy_script_help_and_syntax():
    for script in ("deploy.sh", "install-env.sh", "install-harness.sh", "vbt-host"):
        res = subprocess.run(["bash", "-n", str(FULL / script)], capture_output=True, text=True, timeout=30)
        assert res.returncode == 0, (script, res.stderr)
    res = subprocess.run(["bash", str(FULL / "deploy.sh"), "--help"], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0 and "deploy.sh up" in res.stdout
    res = subprocess.run(["bash", str(FULL / "deploy.sh"), "setup"], capture_output=True, text=True, timeout=30,
                         env={"PATH": os.environ["PATH"]})
    assert res.returncode == 1 and "set VBT_HOME" in res.stderr


def test_scripts_the_docs_run_directly_are_executable_in_the_index():
    """DEP-2: deploy/full/vbt-host was committed 100644, so DEPLOYMENT §4's `deploy/full/vbt-host setup` failed with
    Permission denied on a fresh clone (the images chmod it, the bare-metal path did not)."""
    root = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("not a git checkout")
    scripts = ["deploy/full/deploy.sh", "deploy/full/install-env.sh", "deploy/full/install-harness.sh",
               "deploy/full/vbt-host", "deploy/local/serve_vllm.sh", "scripts/dev/cpu_server.sh"]
    out = subprocess.run(["git", "ls-files", "-s", *scripts], cwd=root, capture_output=True, text=True, timeout=30)
    modes = {line.split()[3]: line.split()[0] for line in out.stdout.splitlines()}
    assert modes == {s: "100755" for s in scripts}, modes
    docs = (root / "docs" / "DEPLOYMENT.md").read_text()
    assert "deploy/full/vbt-host setup" in docs                  # the docs still run it directly


def test_secrets_env_is_read_by_vbt_host_and_by_vbt_itself(tmp_path):
    """DEP-7: DEPLOYMENT says secrets live only in $VBT_HOME/secrets.env, but only compose read it: a bare-metal
    `vbt-host web` answered 'VBT_WEB_PASSWORD is not set'. Values are read as text (never evaluated); a variable already
    set wins; a file other users can read is refused."""
    from vbt.cli import apply_host_config
    from vbt.config import ProfileError

    home = tmp_path / "home"
    home.mkdir()
    secrets = home / "secrets.env"
    secrets.write_text("# operator secrets\nVBT_WEB_PASSWORD='p w$(id)'\nexport NCBI_API_KEY=\"k1\"\nHF_TOKEN=hf\n")
    secrets.chmod(0o600)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "vbt").write_text('#!/bin/sh\nprintf "%s|%s|%s\\n" "$VBT_WEB_PASSWORD" "$NCBI_API_KEY" "$HF_TOKEN"\n')
    (fake / "vbt").chmod(0o755)
    env = {"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "VBT_HOME": str(home), "HF_TOKEN": "set"}
    res = subprocess.run(["bash", str(FULL / "vbt-host"), "web"], capture_output=True, text=True, timeout=30, env=env)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "p w$(id)|k1|set"
    py_env = {"VBT_HOME": str(home), "HF_TOKEN": "set"}
    apply_host_config(SimpleNamespace(cmd="web", profile=[]), py_env)
    assert (py_env["VBT_WEB_PASSWORD"], py_env["NCBI_API_KEY"], py_env["HF_TOKEN"]) == ("p w$(id)", "k1", "set")
    secrets.chmod(0o644)
    res = subprocess.run(["bash", str(FULL / "vbt-host"), "web"], capture_output=True, text=True, timeout=30, env=env)
    assert res.returncode == 1 and "chmod 600" in res.stderr
    with pytest.raises(ProfileError, match="chmod 600"):
        apply_host_config(SimpleNamespace(cmd="web", profile=[]), {"VBT_HOME": str(home)})


def test_deploy_setup_plan_starts_no_service_and_changes_no_ownership(tmp_path):
    """DEP-12: `deploy.sh setup --plan`, offered as 'see what it will do', started searxng (pull and start) and chowned
    the home directories before planning."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "docker").write_text('#!/bin/sh\necho "ARGS $*" >> "$DOCKER_LOG"\n')
    (fake / "docker").chmod(0o755)
    home = tmp_path / "home"
    env = {"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "VBT_HOME": str(home),
           "DOCKER_LOG": str(tmp_path / "docker.log"), "VBT_UID": str(os.getuid() + 1)}
    res = subprocess.run(["bash", str(FULL / "deploy.sh"), "setup", "--plan"], capture_output=True, text=True,
                         timeout=60, env=env)
    assert res.returncode == 0, res.stderr
    calls = (tmp_path / "docker.log").read_text()
    assert "searxng" not in calls.replace("--profile setup run --rm --no-deps setup setup", "") and "--plan" in calls
    assert all((home / d).stat().st_uid == os.getuid() for d in ("data", "runs", "projects", "state"))


def test_deploy_script_hands_only_the_vllm_secrets_to_compose(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "docker").write_text('#!/bin/sh\necho "ARGS $*"\necho "HF=[$HF_TOKEN] KEY=[$VLLM_API_KEY] '
                                 'PW=[$VBT_WEB_PASSWORD]"\n')
    (fake / "docker").chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / "secrets.env").write_text("VBT_WEB_PASSWORD=pw\nHF_TOKEN='hf_x y'\nexport VLLM_API_KEY=\"k1\"\n")
    (home / "state").mkdir()
    (home / "state" / "compose.vllm.yaml").write_text("services: {}\n")
    env = {"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "VBT_HOME": str(home), "VBT_UID": str(os.getuid()),
           "VBT_GID": str(os.getgid())}
    res = subprocess.run(["bash", str(FULL / "deploy.sh"), "config", "--services"], capture_output=True, text=True,
                         timeout=30, env=env)
    assert res.returncode == 0, res.stderr
    assert "HF=[hf_x y] KEY=[k1] PW=[]" in res.stdout
    assert f"-f {FULL / 'compose.yaml'} -f {home / 'state' / 'compose.vllm.yaml'} config --services" in res.stdout
    res = subprocess.run(["bash", str(FULL / "deploy.sh"), "ps"], capture_output=True, text=True, timeout=30,
                         env={**env, "VLLM_API_KEY": "from-shell"})
    assert "KEY=[from-shell]" in res.stdout

"""Hardening (package R3): per-file quarantine of catalog files (R8), the reaper's stderr tee and memory-exit
causes (INV-1), observe mode without side effects, and an explicit null sent to an upstream that takes it.

* R8: a descriptor or overlay file that does not load is quarantined on its own. Only the servers whose
  overlay it is and the tools that read a table or id_type it declares are refused (typed ``quarantined``,
  subkind ``catalog_file``, naming the file and its error); the rest of the catalog keeps working, in the
  catalog, the gateway, the runtime and preflight.
* INV-1: the reaper tees the child's stderr (bounded) and labels a crash ``memory_limit`` from its cause:
  ``memory_error`` (RLIMIT_DATA allocation failure in the last stderr line), ``cgroup_oom_kill``,
  ``watchdog``, ``kernel_oom_kill`` (the kernel log names the child), ``peak_rss`` only as a fallback.
* Observe mode asks the data child, the source and the watched server nothing in the call path and
  renames no file.

Tests that make the kernel OOM-kill a child (a real memory cgroup) run only with ``VBT_DL_HOST_LIMITS=1``;
the whole-table loads of real Open Targets 25.09 tables only with ``VBT_DL_REAL_DATA=<dir>``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from dl_upstream import REPO
from test_dl_gateway_flow import DRUG_OVERLAY, KNOWN_DRUG, OT, PCSK9, TABLES, call, make_gateway, world
from test_dl_review_fixes import shipped, soma_vocab
from vbt import runtime as runtime_mod
from vbt.datalayer.catalog import Catalog, build_catalog
from vbt.datalayer.descriptor.load import DescriptorError, Quarantined, load_descriptors, load_overlays
from vbt.datalayer.errors import ErrorKind, GatewayError
from vbt.datalayer.launch import REAPER
from vbt.datalayer.memory import crash
from vbt.datalayer.result import DataResult
from vbt.datalayer.settings import DataSettings
from vbt.tools.base import Tool, schema

pytest.importorskip("pydantic")

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the reaper is Linux-only")
SOURCES = REPO / "configs" / "data" / "sources"
OVERLAYS = REPO / "configs" / "data" / "overlays"
FIXTURE_SERVER = Path(__file__).resolve().parent / "servers" / "gateway_fixture_server.py"
TAHOE_TOOLS = {"functional_genomics.compare_drug_effects", "functional_genomics.find_cell_line_selective_effects",
               "functional_genomics.find_drugs_affecting_gene", "functional_genomics.query_drug_perturbation"}


def _reaper_module() -> Any:
    spec = importlib.util.spec_from_file_location("vbt_reaper_under_test", REAPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


reaper = _reaper_module()


def _catalog_copy(tmp_path: Path, *, descriptors: dict[str, str] | None = None,
                  overlays: dict[str, str] | None = None) -> tuple[Path, Path]:
    """Copies of the shipped descriptors and overlays, with ``{file name: new text}`` written over them."""
    src, ov = tmp_path / "sources", tmp_path / "overlays"
    shutil.copytree(SOURCES, src)
    shutil.copytree(OVERLAYS, ov)
    for name, text in (descriptors or {}).items():
        (src / name).write_text(text)
    for name, text in (overlays or {}).items():
        (ov / name).write_text(text)
    return src, ov


def _build(src: Path, ov: Path, **kw: Any) -> Catalog:
    settings = DataSettings.from_dict({"descriptors_dir": str(src), "overlays_dir": str(ov)}, project_root=REPO)
    return build_catalog(settings, **kw)


def _broken_tahoe() -> str:
    return (SOURCES / "tahoe.yaml").read_text().replace("tables:", "tables: [oops", 1)


def _broken_target() -> str:
    return (OVERLAYS / "target.yaml").read_text()[:200]


# ---------------------------------------------------------------------------- R8: the catalog


def test_a_broken_descriptor_quarantines_only_the_tools_that_read_it(tmp_path: Path) -> None:
    src, ov = _catalog_copy(tmp_path, descriptors={"tahoe.yaml": _broken_tahoe()})
    cat = _build(src, ov)
    (q,) = cat.quarantined
    assert q.kind == "descriptor" and q.name == "tahoe_100m" and q.file == "tahoe.yaml"
    assert "cannot read YAML" in q.error and "\n" not in q.summary
    assert "tahoe_100m" not in cat.sources and "open_targets" in cat.sources and "depmap" in cat.sources
    assert set(cat.quarantined_tools()) == TAHOE_TOOLS
    blocked = cat.contract("functional_genomics", "query_drug_perturbation")
    assert blocked.quarantined == (q,) and "tahoe.yaml" in blocked.quarantine_reason
    assert any(ref.startswith("tahoe_100m.") for ref in blocked.missing)
    # the rest keeps its bindings: DepMap tools of the same server, and every other server
    for server, tool in (("functional_genomics", "query_gene_essentiality"), ("drug", "search_known_drugs"),
                         ("target", "get_target_info")):
        c = cat.contract(server, tool)
        assert not c.quarantined and c.binding is not None and not c.generic
    assert not cat.quarantined_servers()


def test_a_broken_overlay_quarantines_only_its_server(tmp_path: Path) -> None:
    src, ov = _catalog_copy(tmp_path, overlays={"target.yaml": _broken_target()})
    cat = _build(src, ov)
    (q,) = cat.quarantined
    assert q.kind == "overlay" and q.name == "target" and "server: Field required" in q.error
    assert list(cat.quarantined_servers()) == ["target"] and "target" not in cat.servers()
    c = cat.contract("target", "get_target_info")
    # never served on the generic guard: the server's bindings are unknown, so every tool of it is refused
    assert c.quarantined == (q,) and c.binding is None and not c.generic
    assert not cat.contract("drug", "search_known_drugs").quarantined
    assert not cat.quarantined_tools()                       # the loaded overlays depend on nothing broken


def test_an_unreadable_descriptor_without_a_name_quarantines_unresolved_references(tmp_path: Path) -> None:
    """No ``source:`` can be read: any binding whose references the loaded catalog cannot resolve depends
    on it, and nothing else does."""
    src, ov = _catalog_copy(tmp_path, descriptors={"tahoe.yaml": "{{{ not yaml\n"})
    cat = _build(src, ov)
    (q,) = cat.quarantined
    assert q.name is None and q.tables is None and q.id_types is None
    assert set(cat.quarantined_tools()) == TAHOE_TOOLS


def test_a_name_declared_twice_quarantines_every_file_that_declares_it(tmp_path: Path) -> None:
    src, ov = _catalog_copy(tmp_path)
    shutil.copy(src / "msigdb.yaml", src / "msigdb_copy.yaml")
    shutil.copy(ov / "pathway.yaml", ov / "pathway_copy.yaml")
    cat = _build(src, ov)
    assert sorted((q.file, q.name) for q in cat.quarantined) == [
        ("msigdb.yaml", "msigdb"), ("msigdb_copy.yaml", "msigdb"), ("pathway.yaml", "pathway"),
        ("pathway_copy.yaml", "pathway")]
    assert "msigdb" not in cat.sources and "pathway" not in cat.servers()
    assert all("also declared" in q.error for q in cat.quarantined)
    # the strict loaders (and build_catalog(quarantine=False)) still raise on the first fault
    with pytest.raises(DescriptorError, match="also declared"):
        load_descriptors(src)
    with pytest.raises(DescriptorError, match="also has an overlay"):
        load_overlays(ov)
    with pytest.raises(DescriptorError):
        _build(src, ov, quarantine=False)


def test_a_broken_generic_overlay_quarantines_only_the_generic_guard() -> None:
    from vbt.datalayer.descriptor.models import SourceDescriptor
    from vbt.datalayer.descriptor.overlay import Overlay

    q = Quarantined("/x/_generic.yaml", "generic", "*", "cannot read YAML: bad")
    cat = Catalog({"open_targets": SourceDescriptor.model_validate(copy.deepcopy(OT))},
                  {"drug": Overlay.model_validate(copy.deepcopy(DRUG_OVERLAY))}, quarantined=[q])
    assert cat.contract("thirdparty", "anything").quarantined == (q,)
    assert not cat.contract("drug", "search_known_drugs").quarantined
    assert cat.contract("drug", "unbound_tool").quarantined == (q,)       # it would fall to the generic guard


def test_lint_and_the_digest_name_each_quarantined_file(tmp_path: Path) -> None:
    src, ov = _catalog_copy(tmp_path, descriptors={"tahoe.yaml": _broken_tahoe()},
                            overlays={"target.yaml": _broken_target()})
    cat = _build(src, ov)
    lint = cat.lint()
    findings = [f for f in lint if f.rule == "quarantined"]
    assert sorted(Path(f.where).name for f in findings) == ["tahoe.yaml", "target.yaml"]
    assert all(f.level == "error" and "does not load" in f.message for f in findings)
    # only the files are errors: an overlay's references to what tahoe.yaml declares point at it
    assert [f for f in lint if f.level == "error"] == findings
    assert any(f.message.endswith("(it depends on the quarantined tahoe.yaml)") and f.where.startswith(
        "functional_genomics.") for f in lint if f.level == "warning")
    healthy = _build(*_catalog_copy(tmp_path / "ok"))
    assert healthy.quarantined == [] and healthy.digest() != cat.digest()


# ---------------------------------------------------------------------------- R8: the gateway


def _quarantined_world(tmp_path: Path, **kw: Any) -> Any:
    """The flow world plus a drug tool that reads a table of a quarantined descriptor ``other``."""
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["needs_other"] = {"reads": {"other.records": {"access": "full_table"}},
                                  "args": {"q": {"binds": "other.records.id"}}, "result": {"rows": "$.rows"}}
    gw = world(tmp_path, overlay=ov, **kw)
    gw.catalog.quarantined = [Quarantined(str(tmp_path / "other.yaml"), "descriptor", "other",
                                          "tables.records.key: Field required")]
    return gw


async def test_the_gateway_refuses_only_dependent_tools_with_a_typed_error(tmp_path: Path) -> None:
    gw = _quarantined_world(tmp_path)
    with pytest.raises(GatewayError) as e:
        await gw.prepare("drug", "needs_other", {"q": "x"}, None)
    err = e.value
    assert err.kind == ErrorKind.quarantined and err.subkind == "catalog_file" and err.retryable == "no"
    assert str(tmp_path / "other.yaml") in err.message and "tables.records.key: Field required" in err.message
    env = err.envelope()
    assert env["files"] == [{"file": str(tmp_path / "other.yaml"), "kind": "descriptor", "name": "other",
                             "error": "tables.records.key: Field required"}]
    assert "vbt ds lint" in env["instruction"] and env["citable"] is False
    # the rest of the catalog keeps working
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": PCSK9},
                           lambda a: {"success": True, "count": 8, "drugs": [r for r in KNOWN_DRUG
                                                                               if r["targetId"] == PCSK9]})
    assert isinstance(res, DataResult) and plan.route == "upstream"
    listing = gw.rewrite_listing("drug", "needs_other", "Reads other records.", {"type": "object"})
    assert listing.visible and listing.description.startswith("UNAVAILABLE: quarantined (other.yaml: ")
    assert not gw.rewrite_listing("drug", "search_known_drugs", "x", {"type": "object"}).description.startswith(
        "UNAVAILABLE")
    assert gw.degraded_tools()["mcp__drug__needs_other"].startswith("quarantined: other.yaml")
    assert gw.pinned()["quarantined"][0]["name"] == "other"


async def test_observe_and_lenient_never_refuse_a_quarantined_tool(tmp_path: Path) -> None:
    gw = _quarantined_world(tmp_path / "o", data={"gateway": {"mode": "observe"}})
    plan = await gw.prepare("drug", "needs_other", {"q": "x"}, None)
    assert plan.route == "upstream" and plan.args_sent == {"q": "x"}
    (ev,) = [d for k, d in gw.bridge.events if k == "data_observe"]
    assert ev["decision"] == "would_quarantined" and ev["error"]["subkind"] == "catalog_file"

    gw = _quarantined_world(tmp_path / "l", data={"gateway": {"when_service_down": "lenient"}})
    plan, res = await call(gw, "drug", "needs_other", {"q": "x"}, lambda a: {"rows": [{"id": "x"}]})
    assert isinstance(res, str) and json.loads(res) == {"rows": [{"id": "x"}]}
    assert any(d["decision"] == "quarantined_unguarded" for k, d in gw.bridge.events if k == "data_observe")
    assert "mcp__drug__needs_other" not in gw.degraded_tools()


# ---------------------------------------------------------------------------- R8: runtime and preflight


class _Bridge:
    """An MCPBridge stand-in that starts nothing and lists one tool per started server."""

    made: list[Any] = []

    def __init__(self, specs: Any, *, extra_env: Any = None, log_dir: Any = None, options: Any = None,
                 on_event: Any = None, gateway: Any = None, on_tools_changed: Any = None) -> None:
        self.specs, self.gateway = specs, gateway
        self.sessions: dict[str, Any] = {}
        self.failures: dict[str, str] = {}
        self.started_with: Any = None
        _Bridge.made.append(self)

    async def start(self, only: Any = None) -> list[Tool]:
        self.started_with = set(only) if only is not None else {s.name for s in self.specs}
        return [Tool(f"mcp__{n}__t", "t", schema({}), lambda ctx, a: "ok", source=f"mcp:{n}")
                for n in sorted(self.started_with)]

    async def aclose(self) -> None:
        pass


def _servers_config(config: dict[str, Any], tmp_path: Path, **gateway: Any) -> None:
    _src, ov = _catalog_copy(tmp_path, overlays={"target.yaml": _broken_target()})
    config["data"] = {"enabled": True, "overlays_dir": str(ov), "cache_dir": str(tmp_path / "cache"),
                      **({"gateway": gateway} if gateway else {})}
    config["mcp_servers"] = {"servers": [{"name": "target", "command": "python", "args": []},
                                         {"name": "drug", "command": "python", "args": []}]}


async def test_the_runtime_starts_the_rest_and_refuses_only_the_quarantined_server(
        config: Any, scripted_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_mod, "MCPBridge", _Bridge)
    _servers_config(config, tmp_path)
    session = await scripted_session(config, {"cso": []})
    rt = session.rt
    failures = await rt.start_mcp()
    bridge = _Bridge.made[-1]
    assert rt.gateway is not None and rt.gateway_error is None and bridge.gateway is rt.gateway
    assert set(failures) == {"target"} and "target.yaml" in failures["target"]
    assert "not started: the data catalog could not be loaded for this server" in failures["target"]
    assert bridge.started_with == {"drug", "data"}           # the data child serves the rest
    assert [s.name for s in bridge.specs] == ["drug", "data"]  # the refused server is not even configured
    assert rt.registry.get("mcp__drug__t") is not None and rt.registry.get("mcp__target__t") is None
    events = {e["type"]: e for e in rt.run.events()}
    assert events["data_catalog_quarantined"]["kind"] == "overlay"
    assert events["data_catalog_quarantined"]["file"].endswith("target.yaml")
    assert events["data_gateway_refused"]["servers"] == ["target"] and "target.yaml" in rt.gateway_refused
    assert "data_gateway_unavailable" not in events
    await session.close()


async def test_lenient_starts_every_server(config: Any, scripted_session: Any, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_mod, "MCPBridge", _Bridge)
    _servers_config(config, tmp_path, when_service_down="lenient")
    session = await scripted_session(config, {"cso": []})
    failures = await session.rt.start_mcp()
    assert failures == {} and _Bridge.made[-1].started_with == {"target", "drug", "data"}
    await session.close()


@pytest.mark.skipif(importlib.util.find_spec("fastmcp") is None, reason="fastmcp not installed")
async def test_a_refused_server_is_never_started_lazily(config: Any, scripted_session: Any, tmp_path: Path) -> None:
    """A refused server is not handed to the bridge: a direct call cannot start it lazily. Before, the bridge
    started it on the first call and, the catalog being broken, served it with no gateway at all."""
    from vbt.tools.base import ToolFailure

    _src, ov = _catalog_copy(tmp_path, overlays={"target.yaml": _broken_target()})
    state = tmp_path / "fixture"
    config["data"] = {"enabled": True, "overlays_dir": str(ov), "cache_dir": str(tmp_path / "cache")}
    config["mcp_servers"] = {"servers": [{"name": "target", "command": sys.executable,
                                          "args": ["-E", str(FIXTURE_SERVER), str(state)]}]}
    session = await scripted_session(config, {"cso": []})
    rt = session.rt
    failures = await rt.start_mcp()
    # nothing else to start: neither the gateway nor its data child runs (R8, as before quarantine)
    assert set(failures) == {"target"} and rt.gateway is None and "DescriptorError" in rt.gateway_error
    with pytest.raises(ToolFailure, match="unknown MCP server"):
        await rt.mcp.call("target", "lookup", {"target_id": PCSK9})
    assert not state.exists()                                  # the fixture server never started
    await session.close()


async def test_a_server_the_bridge_refuses_is_never_started(tmp_path: Path) -> None:
    """``MCPBridge.refuse``: callers that build their own bridge (doctor smoke, replay) cannot start a refused
    server either: not at start, not by a retry, not lazily on a call."""
    from vbt.tools.base import ToolFailure
    from vbt.tools.mcp_bridge import MCPBridge, MCPServerConfig

    state = tmp_path / "fixture"
    bridge = MCPBridge([MCPServerConfig("target", command=sys.executable, args=["-E", str(FIXTURE_SERVER), str(state)])],
                       log_dir=tmp_path / "logs")
    bridge.refuse("target", "target.yaml is quarantined")
    try:
        await bridge.start()
        await bridge.retry_failed()
        with pytest.raises(ToolFailure, match="refused: target.yaml is quarantined"):
            await bridge.call("target", "lookup", {"target_id": PCSK9})
        assert bridge.failures == {"target": "target.yaml is quarantined"} and not state.exists()
    finally:
        await bridge.aclose()


def test_preflight_names_the_file_and_marks_only_dependent_tools_unready(
        config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vbt import preflight

    src, ov = _catalog_copy(tmp_path, descriptors={"tahoe.yaml": _broken_tahoe()},
                            overlays={"target.yaml": _broken_target()})
    monkeypatch.setattr(preflight, "run_data_check", lambda cfg, tables=(), depth=None, **kw: {"tables": {
        t: {"status": "ready", "fingerprint": f"fp:{t}", "checks": []} for t in tables}})
    config["data"] = {"enabled": True, "descriptors_dir": str(src), "overlays_dir": str(ov),
                      "cache_dir": str(tmp_path / "cache")}
    config["mcp_servers"] = {"servers": [{"name": n, "command": "python", "args": []}
                                         for n in ("target", "functional_genomics")]}
    results = preflight.check_reference_data(config)
    (fail,) = [r for r in results if r.label == "data catalog"]
    assert not fail.ok and fail.required and fail.scope is not None    # tool-scoped: blocks no session itself
    assert "tahoe.yaml" in fail.detail and "target.yaml" in fail.detail and "servers refused: target" in fail.detail
    assert {q["name"] for q in fail.scope["quarantined"]} == {"tahoe_100m", "target"}
    expected = {f"mcp__{t.replace('.', '__')}" for t in TAHOE_TOOLS}
    assert set(fail.tools) == expected
    assert expected <= set(preflight.degraded_tools(config, results))
    dr, _cache, _catalog = preflight.load_data_readiness(config)
    assert {t for t, r in dr.unready.items() if r["status"] == "quarantined"} == expected
    assert "mcp__functional_genomics__query_gene_essentiality" in dr.bound["functional_genomics"]
    assert "mcp__functional_genomics__query_gene_essentiality" not in dr.unready


# ---------------------------------------------------------------------------- INV-1: the reaper's causes


def _run_reaper(tmp_path: Path, child: list[str], *args: str) -> tuple[subprocess.CompletedProcess, dict[str, Any]]:
    status = tmp_path / "s.status.json"
    proc = subprocess.run([sys.executable, "-E", str(REAPER), "--status", str(status), "--server", "t", *args,
                           "--", *child], capture_output=True, text=True, timeout=120)
    return proc, json.loads(status.read_text())


@linux_only
def test_an_rlimit_memory_error_is_a_memory_exit_named_from_the_teed_stderr(tmp_path: Path) -> None:
    """RLIMIT_DATA fails the allocation: the child exits 1 at a low RSS, which the RSS fraction never
    called a memory exit. The teed stderr names it, before the exit marker."""
    proc, status = _run_reaper(tmp_path, [sys.executable, "-c", "b = bytearray(600 * 2**20)"], "--limit-mb", "300")
    marker = crash.parse_exit_marker(proc.stderr)
    assert marker["code"] == 1 and marker["reason"] == "memory_limit" and marker["cause"] == "memory_error"
    assert marker["memory_error"] == "MemoryError" and marker["maxrss_kb"] < 0.9 * 300 * 1024
    assert proc.stderr.index("MemoryError") < proc.stderr.index("VBT_CHILD_EXIT")
    assert status["exit"]["cause"] == "memory_error"
    d = crash.crash_decision("Connection closed", proc.stderr, server="t", tool="load")
    assert d.oom and not d.retry and "an allocation failed" in d.error.message
    assert d.error.payload["exit"]["cause"] == "memory_error"


@linux_only
def test_the_tee_forwards_everything_in_order(tmp_path: Path) -> None:
    code = ("import sys\nfor i in range(3000):\n    sys.stderr.write(f'line {i:05d} ' + 'x' * 90 + '\\n')\n"
            "sys.stdout.write('out\\n')\n")
    proc, status = _run_reaper(tmp_path, [sys.executable, "-c", code], "--limit-mb", "0")
    lines = proc.stderr.splitlines()
    assert proc.returncode == 0 and proc.stdout == "out\n"
    assert [ln[:10] for ln in lines[:3000]] == [f"line {i:05d}" for i in range(3000)]
    assert lines[-1].startswith("VBT_CHILD_EXIT") and len(lines) == 3001
    assert crash.parse_exit_marker(proc.stderr)["reason"] == "exit_code" and "cause" not in status["exit"]


def test_the_stderr_tee_keeps_a_bounded_tail() -> None:
    r_src, w_src = os.pipe()
    r_dst, w_dst = os.pipe()
    tee = reaper.StderrTee(r_src, w_dst, keep=1024).start()
    payload = b"".join(f"{i:07d}\n".encode() for i in range(8000))      # 64,000 bytes
    got = bytearray()
    os.write(w_src, payload[:32000])
    os.write(w_src, payload[32000:])
    os.close(w_src)
    deadline = time.monotonic() + 10
    while len(got) < len(payload) and time.monotonic() < deadline:
        got += os.read(r_dst, 1 << 16)
    tail = tee.finish(timeout=5)
    assert bytes(got) == payload and tee.total == len(payload) and tee.eof
    assert len(tail) == 1024 and payload.decode().endswith(tail) and len(tee.tail) <= 2 * 1024
    for fd in (r_src, r_dst, w_dst):
        os.close(fd)


@pytest.mark.parametrize("kw,expected", [
    ({"watchdog": True, "signum": 9}, "watchdog"),
    ({"cgroup_oom": True, "signum": 9}, "cgroup_oom_kill"),
    ({"stderr_tail": "Traceback ...\nMemoryError\n"}, "memory_error"),
    ({"stderr_tail": "terminate called after throwing an instance of 'std::bad_alloc'\n  what():  std::bad_alloc\n",
      "signum": 6}, "memory_error"),
    ({"stderr_tail": "memory allocation of 4096 bytes failed\n", "signum": 6}, "memory_error"),
    ({"stderr_tail": "OSError: [Errno 12] Cannot allocate memory\n"}, "memory_error"),
    # seen on real Open Targets loads under RLIMIT_DATA: OpenBLAS at import, Arrow's thread pool (stacks count)
    ({"stderr_tail": "OpenBLAS error: Memory allocation still failed after 10 retries, giving up.\n"}, "memory_error"),
    ({"stderr_tail": "terminate called after throwing an instance of 'std::system_error'\n"
                     "  what():  Resource temporarily unavailable\n", "signum": 6}, "memory_error"),
    ({"stderr_tail": "  what():  Resource temporarily unavailable\n", "signum": 6, "limit_mb": 0}, None),
    ({"signum": 9, "kernel_oom": True}, "kernel_oom_kill"),
    ({"signum": 11, "peak_mb": 950.0}, "peak_rss"),
    # not memory exits: a memory error the server survived, a plain kill, a crash far from the limit, a clean exit
    ({"stderr_tail": "MemoryError\nINFO handled request 7\n"}, None),
    ({"signum": 9}, None),
    ({"signum": 11, "peak_mb": 100.0}, None),
    ({"abnormal": False, "stderr_tail": "MemoryError\n"}, None),
])
def test_exit_causes(kw: dict[str, Any], expected: str | None) -> None:
    args = {"abnormal": True, "signum": None, "limit_mb": 1000, "peak_mb": 10.0, **kw}
    cause, line = reaper.exit_cause(**args)
    assert cause == expected
    assert (line is not None) == (expected == "memory_error")


def test_the_kernel_log_names_the_killed_child(tmp_path: Path) -> None:
    log = tmp_path / "kmsg"
    log.write_bytes(b"3,1830,7112139779,-;Memory cgroup out of memory: Killed process 5281 (python3) "
                    b"total-vm:423560kB, anon-rss:2816kB\n")
    assert reaper.kernel_oom_killed(5281, 7_000_000_000, path=str(log)) is True
    assert reaper.kernel_oom_killed(5281, 7_200_000_000, path=str(log)) is False     # an earlier process
    assert reaper.kernel_oom_killed(528, 0, path=str(log)) is False                  # pid 528, not 5281
    assert reaper.kernel_oom_killed(5281, 0, path=str(tmp_path / "none")) is None


@linux_only
@pytest.mark.skipif(not os.access("/dev/kmsg", os.R_OK), reason="needs a readable kernel log")
def test_a_plain_sigkill_is_not_a_memory_exit(tmp_path: Path) -> None:
    status = tmp_path / "k.status.json"
    proc = subprocess.Popen([sys.executable, "-E", str(REAPER), "--limit-mb", "500", "--status", str(status),
                             "--server", "k", "--", sys.executable, "-c", "import time; time.sleep(60)"],
                            stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 20
    while not status.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    os.kill(json.loads(status.read_text())["pid"], signal.SIGKILL)
    _out, err = proc.communicate(timeout=30)
    marker = crash.parse_exit_marker(err)
    assert marker["signal"] == 9 and marker["reason"] == "signal" and "cause" not in marker


def test_the_reaper_and_the_crash_classifier_share_their_memory_patterns() -> None:
    assert reaper.MEMORY_PATTERNS == crash.MEMORY_PATTERNS
    for text in ("pyarrow.lib.ArrowMemoryError: malloc of size 123 failed", "std::bad_alloc", "ENOMEM",
                 "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 7.45 GiB", "Out of memory"):
        assert crash.classify_error_text(text) == "oom" and reaper.memory_signature(text)
    assert crash.classify_error_text("memorandum") is None


def test_the_crash_decision_names_the_cause() -> None:
    for cause, words in (("watchdog", "memory watchdog"), ("cgroup_oom_kill", "cgroup's memory limit"),
                         ("kernel_oom_kill", "kernel's OOM killer"), ("peak_rss", "ended at its memory limit")):
        tail = "VBT_CHILD_EXIT " + json.dumps({"pid": 5, "code": None, "signal": 9, "maxrss_kb": 1,
                                               "reason": "memory_limit", "cause": cause}) + "\n"
        d = crash.crash_decision("Connection closed", tail, server="s", tool="t")
        assert d.oom and not d.retry and words in d.error.message and d.error.payload["exit"]["cause"] == cause


def _real_ot_table(name: str) -> Path | None:
    """``<VBT_DL_REAL_DATA>/open_targets/25.09/<name>`` (or ``<VBT_DL_REAL_DATA>/<name>``) when present."""
    root = os.environ.get("VBT_DL_REAL_DATA")
    if not root:
        return None
    for cand in (Path(root) / "open_targets" / "25.09" / name, Path(root) / name):
        if cand.is_dir() and any(cand.glob("*.parquet")):
            return cand
    return None


READ_TABLE = ("import sys, pyarrow.parquet as pq\ndf = pq.read_table(sys.argv[1]).to_pandas()\n"
              "print(f'rows={len(df)}', flush=True)\n")


@linux_only
@pytest.mark.skipif(_real_ot_table("target") is None or _real_ot_table("known_drug") is None,
                    reason="VBT_DL_REAL_DATA=<dir> with Open Targets 25.09 target and known_drug needed")
def test_real_whole_table_loads_are_labelled_by_their_cause(tmp_path: Path) -> None:
    """Real Open Targets 25.09 tables loaded whole into pandas (as upstream servers do) under the reaper: the
    data limit fails Arrow's allocation or its thread creation, which the base reaper called ``exit_code`` or
    ``signal`` (and retried); the control load under a wide limit is a clean exit."""
    pytest.importorskip("pyarrow")
    target, known_drug = _real_ot_table("target"), _real_ot_table("known_drug")
    for kind, limit in (("rlimit_data", "400"), ("cgroup", "600")):
        proc, status = _run_reaper(tmp_path, [sys.executable, "-c", READ_TABLE, str(target)], "--limit-mb", limit,
                                   "--containment", kind)
        marker = crash.parse_exit_marker(proc.stderr)
        assert marker["reason"] == "memory_limit" and marker["cause"] == "memory_error", proc.stderr[-2000:]
        assert crash.crash_decision("Connection closed", proc.stderr, server="t", tool="load").oom
    proc, status = _run_reaper(tmp_path, [sys.executable, "-c", READ_TABLE, str(known_drug)], "--limit-mb", "3000")
    assert proc.returncode == 0 and proc.stdout.strip() == "rows=253442" and "cause" not in status["exit"]


def _cgroup_v1_parent() -> Path | None:
    for line in Path("/proc/self/cgroup").read_text().splitlines() if Path("/proc/self/cgroup").exists() else []:
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            path = Path("/sys/fs/cgroup/memory") / parts[2].lstrip("/")
            return path if (path / "memory.limit_in_bytes").exists() and os.access(path, os.W_OK) else None
    return None


host_limits = pytest.mark.skipif(os.environ.get("VBT_DL_HOST_LIMITS") != "1" or _cgroup_v1_parent() is None,
                                 reason="VBT_DL_HOST_LIMITS=1 and a writable cgroup v1 memory hierarchy needed")
SHM_EATER = ("import mmap, sys, time\nn = int(sys.argv[1]) * 2**20\nm = mmap.mmap(-1, n)\n"
             "for i in range(0, n, 4096):\n    m[i] = 1\ntime.sleep(1)\n")


@host_limits
def test_a_real_cgroup_kill_is_a_cgroup_oom_kill(tmp_path: Path) -> None:
    """Shared anonymous memory is not counted by RLIMIT_DATA but is charged to the cgroup: the kernel kills."""
    proc, status = _run_reaper(tmp_path, [sys.executable, "-c", SHM_EATER, "400"], "--limit-mb", "200",
                               "--containment", "cgroup")
    marker = crash.parse_exit_marker(proc.stderr)
    assert status["containment"] in ("cgroup_v1", "cgroup_v2")
    assert marker["signal"] == 9 and marker["reason"] == "memory_limit" and marker["cause"] == "cgroup_oom_kill"


@host_limits
def test_a_kill_by_an_enclosing_cgroup_is_a_kernel_oom_kill(tmp_path: Path) -> None:
    """The child's own limit is far away (RLIMIT_DATA 1000 MB, no cgroup of its own); the cgroup the reaper
    runs in kills it. Before INV-1 this was ``reason: signal`` (peak 300 MB < 90% of 1000 MB)."""
    parent = _cgroup_v1_parent()
    cg = parent / f"vbt-r3-test-{os.getpid()}"
    cg.mkdir()
    try:
        (cg / "memory.limit_in_bytes").write_text(str(300 * 2**20))
        status = tmp_path / "e.status.json"

        def join() -> None:
            (cg / "cgroup.procs").write_text(str(os.getpid()))

        proc = subprocess.run([sys.executable, "-E", str(REAPER), "--limit-mb", "1000", "--status", str(status),
                               "--server", "e", "--", sys.executable, "-c", SHM_EATER, "400"],
                              capture_output=True, text=True, timeout=120, preexec_fn=join)
        marker = crash.parse_exit_marker(proc.stderr)
        assert marker["signal"] == 9 and marker["reason"] == "memory_limit" and marker["cause"] == "kernel_oom_kill"
    finally:
        for _ in range(50):
            try:
                cg.rmdir()
                break
            except OSError:
                time.sleep(0.1)


# ---------------------------------------------------------------------------- observe mode: no side effects


def _with_output_arg() -> dict[str, Any]:
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs"]["args"]["out"] = {"role": "output_path"}
    return ov


async def test_observe_mode_has_no_side_effects_end_to_end(tmp_path: Path) -> None:
    """prepare and finish of an observed call: the existing output is neither renamed nor copied, no file is
    written, the data child and the server are asked nothing, nothing is reserved or registered, and the
    upstream answer comes back unchanged (no ``_vbt`` header, no provenance record)."""
    gw = world(tmp_path, overlay=_with_output_arg(), data={"gateway": {"mode": "observe"}})
    assert await gw._ensure_index("open_targets:ensembl_gene")    # built off the call path in observe mode
    gw.service.log.clear()
    out = tmp_path / "out"
    (out / "cited.csv").write_text("a,b\n1,2\n")
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    answer = {"success": True, "count": 1, "drugs": [KNOWN_DRUG[0]]}
    plan, res = await call(gw, "drug", "search_known_drugs", {"target_id": "PCSK9", "out": "cited.csv"},
                           lambda a: answer)
    assert plan.args_sent == {"target_id": "PCSK9", "out": "cited.csv"}
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()} == before
    assert gw.service.log == [] and gw.bridge.calls == []
    assert not gw.admission.ledger.pending("drug") and gw.materialized.entries == []
    assert isinstance(res, str) and json.loads(res) == answer
    assert [d["decision"] for k, d in gw.bridge.events if k == "data_observe"] == ["would_route"]


async def test_observe_mode_asks_the_watched_server_for_no_soma_vocabulary(tmp_path: Path) -> None:
    args = {"value_filter": "cell_type == 'T cell'"}
    gw = shipped(tmp_path / "o", gateway={"mode": "observe"})
    soma_vocab(gw, {"cell_type": ["T cell"], "is_primary_data": ["True", "False"]})
    plan = await gw.prepare("single_cell", "count_cells", args, None)
    assert plan.args_sent == args and gw.bridge.calls == []
    gw = shipped(tmp_path / "e")
    soma_vocab(gw, {"cell_type": ["T cell"], "is_primary_data": ["True", "False"]})
    await gw.prepare("single_cell", "count_cells", args, None)
    assert [(t, a) for _s, t, a in gw.bridge.calls] == [("list_metadata_values", {"column_name": "cell_type",
                                                                                  "limit": 100000})]


async def test_observe_mode_counts_nothing_and_asks_no_remote_question(tmp_path: Path) -> None:
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["by_drug"] = {"reads": {"open_targets.known_drug": {"access": "full_table"}},
                              "args": {"drug_id": {"binds": "open_targets.known_drug.drugId",
                                                   "accepts": ["chembl_molecule"], "existence": "bound"}},
                              "result": {"rows": "$.drugs"}}
    gw = world(tmp_path / "bound", overlay=ov, data={"gateway": {"mode": "observe"}})
    await gw._ensure_index("open_targets:chembl_molecule")
    assert await gw.refresh_readiness(["open_targets.known_drug", "open_targets.drug_molecule"])  # checked: not lenient
    gw.service.log.clear()
    plan = await gw.prepare("drug", "by_drug", {"drug_id": "CHEMBL777"}, None)
    assert gw.service.log == [] and plan.args_sent == {"drug_id": "CHEMBL777"}   # no existence count in the path
    remote = copy.deepcopy(OT)
    remote["id_types"]["chembl_molecule"]["index"] = "remote"
    for mode, asked in (("observe", False), ("enforce", True)):
        gw = make_gateway(tmp_path / mode, [remote], [ov], TABLES, data={"gateway": {"mode": mode}})
        await gw.prepare("drug", "by_drug", {"drug_id": "CHEMBL555"}, None) if mode == "observe" else \
            await _prepare_quietly(gw, "drug", "by_drug", {"drug_id": "CHEMBL555"})
        assert bool(gw.service.verbs("_resolve_remote")) is asked


async def _prepare_quietly(gw: Any, server: str, tool: str, args: dict[str, Any]) -> None:
    try:
        await gw.prepare(server, tool, args, None)
    except GatewayError:
        pass                                     # the fake child cannot answer a remote question: unknown is fine


# ---------------------------------------------------------------------------- an explicit null sent upstream


async def test_an_explicit_null_reaches_an_upstream_that_takes_none(tmp_path: Path) -> None:
    """A pass binding whose upstream signature takes None gets the null itself; dropping it let upstream apply
    the default the note says is lifted (CT.gov ``country=null`` counted US trials only, 300 of 705)."""
    ov = copy.deepcopy(DRUG_OVERLAY)
    ov["tools"]["search_known_drugs_derived"]["args"]["min_phase"] = {"binds": "open_targets.known_drug.phase",
                                                                      "op": "ge", "min": 0, "max": 4}
    gw = world(tmp_path, overlay=ov)
    nullable = {"type": "object", "properties": {
        "target_id": {"type": "string"},
        "min_phase": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 2}}}
    gw.rewrite_listing("drug", "search_known_drugs", "x", nullable)
    plan = await gw.prepare("drug", "search_known_drugs", {"target_id": PCSK9, "min_phase": None}, None)
    assert "min_phase" in plan.args_sent and plan.args_sent["min_phase"] is None
    gw.rewrite_listing("drug", "search_known_drugs_derived", "x", nullable)
    plan = await gw.prepare("drug", "search_known_drugs_derived", {"target_id": PCSK9, "min_phase": None}, None)
    assert plan.route == "derived" and "min_phase" not in plan.args_sent      # served here: nothing is sent


def test_yaml_of_the_shipped_catalog_has_no_quarantine() -> None:
    """The shipped descriptors and overlays all load (a quarantine there would hide a broken shipped file)."""
    cat = _build(SOURCES, OVERLAYS)
    assert cat.quarantined == []
    assert yaml.safe_load((OVERLAYS / "target.yaml").read_text())["server"] == "target"

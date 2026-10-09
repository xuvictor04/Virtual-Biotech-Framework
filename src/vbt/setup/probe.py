"""What this host offers: CPUs, memory (and the container's memory limit), GPUs, disk, memory containment,
the OS sandbox, container runtime and network reachability (``vbt setup --probe``).

Every probe reads the host and never changes it, except two throwaway checks that create and remove a test
cgroup (the reaper's own functions) and run ``true`` under bubblewrap. Each function takes its roots as
arguments so tests can point them at fixture trees. The result is a plain JSON-ready mapping, stored in the
setup state and used by :mod:`vbt.setup.hostconfig` to size the host configuration.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

__all__ = ["probe_host", "cpu_facts", "memory_facts", "gpu_facts", "disk_facts", "containment_facts",
           "sandbox_facts", "container_facts", "network_facts", "endpoint_urls", "effective_memory_mb"]

#: cgroup memory limits at or above this are "no limit" (v1 reports 2^63 rounded to the page size).
_UNLIMITED = 1 << 60


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text()
    except (OSError, UnicodeDecodeError):
        return None


def _own_cgroup_paths(proc_root: str) -> dict[str, str]:
    """``{"v2": path}`` and ``{"<controller>": path}`` (v1) from ``/proc/self/cgroup``."""
    out: dict[str, str] = {}
    for line in (_read(f"{proc_root}/self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            out["v2"] = parts[2] or "/"
        for ctl in parts[1].split(","):
            if ctl:
                out[ctl] = parts[2] or "/"
    return out


def _cgroup_file(cgroup_root: str, own: Mapping[str, str], v1_controller: str, v2_name: str,
                 v1_name: str) -> tuple[int, str | None]:
    """``(version, content)`` of a cgroup interface file of this process's cgroup (v2 first)."""
    if "v2" in own and Path(cgroup_root, "cgroup.controllers").exists():
        text = _read(Path(cgroup_root, own["v2"].lstrip("/"), v2_name))
        if text is None:
            text = _read(Path(cgroup_root, v2_name))
        return 2, text
    rel = own.get(v1_controller, "/").lstrip("/")
    for base in (Path(cgroup_root, v1_controller, rel), Path(cgroup_root, v1_controller)):
        text = _read(base / v1_name)
        if text is not None:
            return 1, text
    return 0, None


def cpu_facts(*, proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> dict[str, Any]:
    """Logical CPUs, the affinity set and the cgroup CPU quota; ``effective`` is the smallest."""
    logical = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = logical
    own = _own_cgroup_paths(proc_root)
    quota = None
    version, text = _cgroup_file(cgroup_root, own, "cpu", "cpu.max", "cpu.cfs_quota_us")
    if version == 2 and text:
        parts = text.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota = float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                quota = None
    elif version == 1 and text:
        _v, period = _cgroup_file(cgroup_root, own, "cpu", "cpu.max", "cpu.cfs_period_us")
        try:
            q, p = int(text.strip()), int((period or "100000").strip())
            quota = q / p if q > 0 and p > 0 else None
        except ValueError:
            quota = None
    effective = min([logical, affinity] + ([max(1, int(quota))] if quota else []))
    return {"logical": logical, "affinity": affinity, "cgroup_quota": round(quota, 2) if quota else None,
            "effective": effective}


def _meminfo(proc_root: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (_read(f"{proc_root}/meminfo") or "").splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) // 1024
    return out


def memory_facts(*, proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> dict[str, Any]:
    """MemTotal, MemAvailable, swap and the cgroup memory limit, in MB. ``effective_mb`` is the memory this
    process tree may use: the smaller of MemTotal and the cgroup limit (a container's ``--memory``), which
    ``/proc/meminfo`` does not show inside a container."""
    info = _meminfo(proc_root)
    total = info.get("MemTotal")
    own = _own_cgroup_paths(proc_root)
    version, text = _cgroup_file(cgroup_root, own, "memory", "memory.max", "memory.limit_in_bytes")
    limit = None
    if text:
        value = text.strip()
        if value != "max":
            try:
                raw = int(value)
                limit = raw // (1024 * 1024) if raw < _UNLIMITED else None
            except ValueError:
                limit = None
    effective = min([v for v in (total, limit) if v]) if (total or limit) else None
    return {"total_mb": total, "available_mb": info.get("MemAvailable"), "swap_mb": info.get("SwapTotal"),
            "cgroup_version": version or None, "cgroup_limit_mb": limit, "effective_mb": effective}


def effective_memory_mb(facts: Mapping[str, Any]) -> int | None:
    mem = facts.get("memory") if isinstance(facts.get("memory"), Mapping) else facts
    value = mem.get("effective_mb") if isinstance(mem, Mapping) else None
    return int(value) if value else None


def gpu_facts(nvidia_smi_text: str | None = None, *, source: str | None = None) -> dict[str, Any]:
    """GPUs from ``nvidia-smi`` (run here, or the given output: the harness container sees no GPU, so
    ``deploy.sh`` saves the host's output in the state directory)."""
    from ..local.profiles import detect_nvidia_smi, parse_nvidia_smi

    text = nvidia_smi_text
    origin = source or ("file" if text is not None else "nvidia-smi")
    if text is None:
        text = detect_nvidia_smi()
    if not text:
        return {"source": origin, "count": 0, "gpus": [], "driver_version": None, "cuda_version": None,
                "nvidia_smi": shutil.which("nvidia-smi") is not None}
    info = parse_nvidia_smi(text)
    gpus = [{"index": g.index, "name": g.name, "memory_mib": g.memory_mib,
             "vram_gib": round(g.vram_gib, 1) if g.vram_gib else None} for g in info.gpus]
    return {"source": origin, "count": len(gpus), "gpus": gpus, "driver_version": info.driver_version,
            "cuda_version": info.cuda_version, "nvidia_smi": True, "raw": text}


def _existing_ancestor(path: Path) -> Path:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def disk_facts(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Free and total bytes of the filesystem holding each path (its nearest existing ancestor)."""
    out: dict[str, Any] = {}
    seen: dict[int, dict[str, Any]] = {}
    for name, raw in paths.items():
        p = _existing_ancestor(Path(raw))
        try:
            st = os.stat(p)
            usage = shutil.disk_usage(p)
        except OSError as exc:
            out[name] = {"path": str(raw), "error": str(exc)}
            continue
        rec = {"path": str(raw), "mount_of": str(p), "free_bytes": usage.free, "total_bytes": usage.total,
               "device": st.st_dev}
        if st.st_dev in seen:
            rec["same_filesystem_as"] = seen[st.st_dev]["name"]
        else:
            seen[st.st_dev] = {"name": name}
        out[name] = rec
    return out


def containment_facts() -> dict[str, Any]:
    """Which memory containment the reaper can use here: a v2 or v1 memory cgroup it may create (tried by
    creating and removing a 64 MB test cgroup with the reaper's own functions), else ``rlimit_data``."""
    if not sys.platform.startswith("linux"):
        return {"cgroup": None, "limit_kind": "none", "note": "not Linux: no reaper containment"}
    from ..datalayer.launch import reaper

    for version, make in ((2, reaper.make_cgroup_v2), (1, reaper.make_cgroup_v1)):
        try:
            cg = make("vbt-setup-probe", 64)
        except Exception:  # noqa: BLE001 - a probe never fails setup
            cg = None
        if cg is not None:
            cg.remove()
            return {"cgroup": version, "limit_kind": "cgroup",
                    "note": f"memory cgroup v{version} is writable: servers are limited by resident memory"}
    return {"cgroup": None, "limit_kind": "rlimit_data",
            "note": "no writable memory cgroup: servers are limited with RLIMIT_DATA (address space reservations "
                    "count; single_cell keeps its RSS watchdog)"}


def _runs(argv: list[str], timeout_s: float = 10.0) -> tuple[bool, str]:
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return res.returncode == 0, (res.stderr or res.stdout).strip()[-300:]


def sandbox_facts() -> dict[str, Any]:
    """Whether ``bash.sandbox.os: bwrap`` works here (bubblewrap present and allowed to create namespaces),
    and whether ``unshare -rn`` (network isolation of the no-web profile) works."""
    out: dict[str, Any] = {}
    bwrap = shutil.which("bwrap")
    if bwrap:
        ok, msg = _runs([bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net", "true"])
        out["bwrap"] = {"path": bwrap, "works": ok, "detail": "" if ok else msg}
    else:
        out["bwrap"] = {"path": None, "works": False, "detail": "bubblewrap is not installed"}
    unshare = shutil.which("unshare")
    if unshare:
        ok, msg = _runs([unshare, "-rn", "true"])
        out["unshare_net"] = {"works": ok, "detail": "" if ok else msg}
    else:
        out["unshare_net"] = {"works": False, "detail": "unshare is not installed"}
    return out


def container_facts(*, proc_root: str = "/proc") -> dict[str, Any]:
    """The container runtime this process runs in (if any) and the container tools on PATH."""
    kind = None
    if Path("/.dockerenv").exists():
        kind = "docker"
    elif Path("/run/.containerenv").exists():
        kind = "podman"
    elif os.environ.get("APPTAINER_CONTAINER") or os.environ.get("SINGULARITY_CONTAINER"):
        kind = "apptainer"
    elif os.environ.get("KUBERNETES_SERVICE_HOST"):
        kind = "kubernetes"
    else:
        text = _read(f"{proc_root}/1/cgroup") or ""
        if re.search(r"docker|kubepods|containerd|libpod", text):
            kind = "container"
    tools = {name: shutil.which(name) for name in ("docker", "podman", "apptainer", "singularity", "nvidia-smi")}
    compose = None
    if tools["docker"]:
        ok, msg = _runs([tools["docker"], "compose", "version", "--short"])
        compose = msg.splitlines()[-1] if ok and msg else None
    return {"inside": kind, "tools": tools, "docker_compose": compose}


def endpoint_urls(config: Mapping[str, Any] | None, extra: Iterable[str] = ()) -> dict[str, str]:
    """``{label: url}`` worth probing: every http(s) URL the loaded descriptors name (remote APIs, release
    servers; one per host), the model server, the web search backend and ``extra``."""
    urls: dict[str, str] = {}

    def add(label: str, url: str) -> None:
        url = str(url).strip()
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or "{" in parts.netloc:
            return
        if any(urlsplit(u).netloc == parts.netloc for u in urls.values()):
            return
        urls[label] = f"{parts.scheme}://{parts.netloc}{parts.path.split('{')[0]}" if parts.path else url

    cfg = config or {}
    opts = ((cfg.get("provider") or {}).get("options") or {}) if isinstance(cfg.get("provider"), Mapping) else {}
    name = (cfg.get("provider") or {}).get("name") if isinstance(cfg.get("provider"), Mapping) else None
    if name in ("vllm", "sglang", "openai_compat", "llamacpp"):
        for i, base in enumerate(opts.get("base_urls") or [opts.get("base_url")]):
            if base:
                add(f"model_server{'' if i == 0 else i}", str(base).rstrip("/").removesuffix("/v1") + "/health")
    elif name == "anthropic":
        add("anthropic", str(opts.get("anthropic_base_url") or "https://api.anthropic.com"))
    search = ((cfg.get("web") or {}).get("search") or {}) if isinstance(cfg.get("web"), Mapping) else {}
    if search.get("searxng_url") and search.get("backend") not in ("none", "brave", "provider"):
        add("searxng", str(search["searxng_url"]))
    for label, url in _descriptor_urls(cfg):
        add(label, url)
    for i, url in enumerate(extra):
        add(f"extra{i}", url)
    return urls


def _descriptor_urls(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(source, url)`` for every http(s) URL in the loaded descriptors (expanded with the config)."""
    try:
        from ..datalayer.descriptor.load import expand, variables_from_config
        from ..datalayer.settings import DataSettings
        import yaml

        settings = DataSettings.from_config(dict(config))
        root = Path(settings.descriptors_dir)
        variables = variables_from_config(config)
    except Exception:  # noqa: BLE001 - no data layer: nothing to probe
        return []
    found: list[tuple[str, str]] = []

    def walk(source: str, value: Any) -> None:
        if isinstance(value, str):
            for m in re.finditer(r"https?://[^\s\"'<>()]+", value):
                found.append((source, m.group(0)))
        elif isinstance(value, Mapping):
            for v in value.values():
                walk(source, v)
        elif isinstance(value, list):
            for v in value:
                walk(source, v)

    for path in sorted(root.glob("*.yaml")) if root.is_dir() else []:
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:  # noqa: BLE001 - lint reports broken descriptors
            continue
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("source") or path.stem)
        walk(source, expand({k: v for k, v in raw.items() if k != "description"}, variables))
    return found


def _probe_url(url: str, timeout_s: float) -> dict[str, Any]:
    import httpx

    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False, trust_env=True) as client:
            res = client.head(url)
            if res.status_code in (405, 501):
                res = client.get(url, headers={"Range": "bytes=0-0"})
        return {"url": url, "reachable": True, "status": res.status_code,
                "seconds": round(time.monotonic() - t0, 3)}
    except Exception as exc:  # noqa: BLE001 - every failure is a finding
        return {"url": url, "reachable": False, "error": f"{type(exc).__name__}: {exc}"[:300],
                "seconds": round(time.monotonic() - t0, 3)}


def network_facts(urls: Mapping[str, str], *, timeout_s: float = 10.0, workers: int = 8) -> dict[str, Any]:
    """HEAD each URL (GET of one byte when HEAD is refused). Any HTTP answer means reachable: the probe tests
    the network path (DNS, proxy, TLS), not the service."""
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(urls)))) as pool:
        futures = {label: pool.submit(_probe_url, url, timeout_s) for label, url in urls.items()}
        return {label: f.result() for label, f in futures.items()}


def probe_host(layout: Any, config: Mapping[str, Any] | None = None, *, nvidia_smi_text: str | None = None,
               nvidia_smi_source: str | None = None, network: bool = True, extra_urls: Iterable[str] = (),
               timeout_s: float = 10.0) -> dict[str, Any]:
    """Every probe, as one JSON-ready mapping (``vbt setup --probe --json``)."""
    t0 = time.monotonic()
    paths = {k: str(v) for k, v in layout.dirs().items()} | {"models": str(layout.models)}
    facts: dict[str, Any] = {
        "schema": "vbt.setup.probe/1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "cpu": cpu_facts(),
        "memory": memory_facts(),
        "gpu": gpu_facts(nvidia_smi_text, source=nvidia_smi_source),
        "disk": disk_facts(paths),
        "containment": containment_facts(),
        "sandbox": sandbox_facts(),
        "container": container_facts(),
    }
    urls = endpoint_urls(config, extra_urls)
    facts["network"] = network_facts(urls, timeout_s=timeout_s) if network else {
        label: {"url": url, "reachable": None, "skipped": True} for label, url in urls.items()}
    facts["seconds"] = round(time.monotonic() - t0, 2)
    return facts

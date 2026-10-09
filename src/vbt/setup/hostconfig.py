"""The host configuration ``vbt setup`` writes: memory limits scaled to the host, data and project paths, the
serving profile and how the harness reaches it.

Three files in the state directory:

* ``host.yaml``: a harness profile (``vbt --profile <state>/host.yaml``), layered after the harness profile of
  the serving profile (``local-h100``, ``claude``, ...) and ``production``;
* ``host.env``: ``KEY=VALUE`` lines (shell-quoted; read by ``deploy/full/vbt-host`` and by compose): the data
  roots the descriptors name, the deployment directories, ``VBT_PROFILES`` and the model server's address;
* ``compose.vllm.yaml``: the ``vllm`` service for ``deploy/full/compose.yaml`` (Docker deployments with GPUs),
  rendered from ``configs/local_models.yaml`` by the same code as ``vbt local serve --docker``.

Sizing (MB; ``ram`` = the smaller of MemTotal and the container's memory limit, or ``data.memory.host_mb`` when
it is a number): the rule of :mod:`vbt.datalayer.memory.sizing`, the one the harness applies to ``auto`` at run time,
so ``vbt setup`` writes the numbers ``auto`` would give on this host:

* ``data.memory.host_budget_mb`` = ``0.75 x ram - reserve``, ``reserve = max(harness_reserve_mb (2048), 0.05 x
  ram)``, at least 1,024: the sum of the upstream servers' resident memory (``memory/host.py``);
* ``data.memory.default_server_mb`` = ``0.8 x`` the host budget, at least 2,048 (8,192 on a 16 GB host, 293,601 on
  512 GB). Once ``vbt setup`` has estimated the tables the enabled servers load whole (step ``size``), a largest
  server's estimate x ``estimate_safety`` above that raises it, up to the host budget;
* ``data.service.mem_limit_mb`` = ``clamp(0.05 x ram, 3000, 32768)`` (the data child; ``max_resident_mb`` 2/3
  of it), ``data.service.max_concurrency`` = ``clamp(cpus // 2, 4, 16)``;
* ``data.memory.workspace_mb`` = ``clamp(0.25 x ram / limits.max_parallel_agents, 8000, 65536)``, at most half of
  ram (agent Bash; ``sizing.workspace_for``, what ``auto`` gives at run time);
* ``data.memory.limit_kind`` = ``cgroup`` where a memory cgroup can be created, else left at its default (``rss``);
* ``bash.sandbox.os`` = ``bwrap`` where bubblewrap works.
"""

from __future__ import annotations

import json
import math
import os
import shlex
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..datalayer.memory import sizing as _sizing

__all__ = ["size_host", "pick_serving", "host_profile", "host_env", "render_compose_vllm", "write_host_files",
           "read_env_file", "HOST_PROFILE", "HOST_ENV", "COMPOSE_VLLM", "DEFAULT_SERVER_MB_MIN"]

HOST_PROFILE = "host.yaml"
HOST_ENV = "host.env"
COMPOSE_VLLM = "compose.vllm.yaml"

#: The floor of one server's limit (``vbt.datalayer.memory.sizing.SERVER_FLOOR_MB``).
DEFAULT_SERVER_MB_MIN = _sizing.SERVER_FLOOR_MB
SERVICE_MB_MIN, SERVICE_MB_MAX = _sizing.CHILD_FLOOR_MB, _sizing.CHILD_CEILING_MB
WORKSPACE_MB_MIN, WORKSPACE_MB_MAX = _sizing.WORKSPACE_FLOOR_MB, _sizing.WORKSPACE_CEILING_MB
RESERVE_MB_MIN = int(_sizing.DEFAULT_RESERVE_MB)


def _clamp(value: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, value)))


def size_host(facts: Mapping[str, Any], config: Mapping[str, Any], *,
              server_need_mb: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Memory and concurrency settings for this host (see the module docstring). ``server_need_mb`` is the
    estimated whole-table load of each enabled server (step ``size``); without it the server limit is scaled
    from RAM alone."""
    mem = facts.get("memory") or {}
    ram = mem.get("effective_mb") or mem.get("total_mb")
    cpus = int((facts.get("cpu") or {}).get("effective") or os.cpu_count() or 1)
    data = config.get("data") or {}
    memory_cfg = data.get("memory") or {}
    safety = float(memory_cfg.get("estimate_safety") or 1.3)
    parallel = int(((config.get("limits") or {}).get("max_parallel_agents")) or 8)
    out: dict[str, Any] = {"ram_mb": ram, "cpus": cpus, "notes": []}
    # the plan the runtime uses (sizing.plan_mb): data.memory.host_mb, else $VBT_HOST_MEMORY_MB, else the probed
    # memory. The harness's share of a host it shares (a model server next to it) is honoured here too (DEP-6).
    planned = _sizing.plan_mb(memory_cfg, measured=float(ram) if ram else None)
    source = _sizing.plan_source(memory_cfg, measured=float(ram) if ram else None)
    if planned and source != "measured":
        ram = out["ram_mb"] = planned
        out["notes"].append(f"sized from {source} ({planned:,.0f} MB), not the host's memory")
    out["plan_from"] = source if planned else None
    if not ram:
        out["notes"].append("host memory unknown: the shipped memory settings are kept")
        return out
    ram = float(ram)
    budget = int(_sizing.host_budget_for(ram, memory_cfg))
    out["host_budget_mb"] = budget
    rule = _sizing.server_limit_for(ram, memory_cfg)
    if server_need_mb:
        # admission's rule: the whole-table peaks (MiB, `vbt ds estimate`) x safety, plus an idle server's baseline
        from ..datalayer.memory.ledger import DEFAULT_BASELINE_MB

        largest_server, largest = max(server_need_mb.items(), key=lambda kv: kv[1])
        need = math.ceil(largest * safety + DEFAULT_BASELINE_MB)
        out["largest_server"] = {"server": largest_server, "estimate_mb": round(largest), "with_safety_mb": need}
        server_mb = max(rule, min(budget, need))
        if need > budget:
            out["notes"].append(
                f"server {largest_server} loads ~{largest:,.0f} MB of tables whole (x{safety} safety + "
                f"{DEFAULT_BASELINE_MB:.0f} MB baseline = {need:,} MB), "
                f"more than this host's budget of {budget:,} MB: its largest tools are refused too_large here")
        total = sum(server_need_mb.values()) * safety
        if total > budget:
            out["notes"].append(
                f"all enabled servers together load ~{total:,.0f} MB whole; the host budget ({budget:,} MB) recycles "
                "idle servers (least recently used first) to stay within it")
    else:
        server_mb = rule
    out["default_server_mb"] = int(server_mb)
    out["service_mem_limit_mb"] = _sizing.data_child_for(ram)
    out["service_max_resident_mb"] = int(out["service_mem_limit_mb"] * 2 / 3)
    out["service_max_concurrency"] = _clamp(cpus // 2, 4, 16)
    out["workspace_mb"] = _sizing.workspace_for(ram, parallel, memory_cfg)   # the rule `auto` applies at run time
    out["full_load"] = _sizing.full_load(ram, host_budget=budget, data_child=out["service_mem_limit_mb"],
                                         workspace=out["workspace_mb"], parallel=parallel,
                                         reserve=_sizing._reserve(ram, memory_cfg))
    if out["full_load"]["left_mb"] < 0:
        out["notes"].append(f"at full load (host budget, data child, {parallel} agent commands) the limits add up "
                            f"to {out['full_load']['sum_mb']:,} MB, over the {ram:,.0f} MB plan: the floors of a "
                            "small host; lower limits.max_parallel_agents or data.memory.workspace_mb")
    kind = (facts.get("containment") or {}).get("limit_kind")
    if kind == "cgroup":
        out["limit_kind"] = "cgroup"
    bwrap = ((facts.get("sandbox") or {}).get("bwrap") or {})
    out["bwrap"] = bool(bwrap.get("works"))
    return out


def pick_serving(facts: Mapping[str, Any], *, serving_profile: str | None = None, variants: Iterable[str] = (),
                 harness_profile: str | None = None, llm_url: str | None = None,
                 profiles: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The serving profile (``configs/local_models.yaml``) and the harness profile that goes with it.

    ``serving_profile`` wins; else the GPUs found by the probe pick one (``vbt local profiles --detect``); else,
    without GPUs, the harness talks to a server elsewhere (``llm_url``) or to Claude (``harness_profile``
    ``claude``/``paper``). ``harness_profile`` overrides the profile's own."""
    from ..local.profiles import load_local_profiles, parse_nvidia_smi, pick_profile

    profiles = profiles if profiles is not None else load_local_profiles()
    out: dict[str, Any] = {"serving_profile": None, "variants": list(variants), "data_parallel_size": 1,
                           "harness_profile": harness_profile, "llm_url": llm_url, "docker_tag": None,
                           "driver_version": None, "reason": "", "warnings": []}
    gpu = facts.get("gpu") or {}
    out["driver_version"] = gpu.get("driver_version")
    if serving_profile:
        if serving_profile not in profiles:
            raise ValueError(f"unknown serving profile {serving_profile!r}; available: {', '.join(profiles)}")
        out["serving_profile"] = serving_profile
        out["reason"] = "chosen with --serving-profile"
        hw = profiles[serving_profile].get("hardware") or {}
        out["data_parallel_size"] = int(hw.get("gpu_count", 1)) if serving_profile in ("dp",) else 1
    elif gpu.get("raw"):
        pick = pick_profile(parse_nvidia_smi(gpu["raw"]), profiles)
        out["reason"] = pick.reason
        out["warnings"].extend(pick.warnings)
        if pick.profile:
            out["serving_profile"] = pick.profile
            out["variants"] = list(dict.fromkeys([*pick.variants, *out["variants"]]))
            out["data_parallel_size"] = pick.data_parallel_size
            out["docker_tag"] = pick.docker_tag
    elif llm_url:
        out["reason"] = f"no GPU here; the model server is {llm_url}"
    else:
        out["reason"] = "no NVIDIA GPU found"
        if not harness_profile:
            out["warnings"].append("no GPU and no --llm-url: pass --serving-profile NAME --llm-url URL for a model "
                                   "server on another host, or --harness-profile claude for the Anthropic API")
    if out["serving_profile"] and not harness_profile:
        out["harness_profile"] = ((profiles[out["serving_profile"]].get("harness") or {}).get("profile"))
    return out


def host_profile(sizing: Mapping[str, Any], layout: Any, *, tool_env: Mapping[str, str],
                 generated_by: str = "vbt setup") -> dict[str, Any]:
    """The ``host.yaml`` mapping (a harness profile)."""
    prof: dict[str, Any] = {"paths": {"runs_dir": str(layout.runs)}}
    data: dict[str, Any] = {"cache_dir": str(Path(layout.data) / ".vbt-datalayer")}
    memory: dict[str, Any] = {}
    for key, cfg_key in (("host_budget_mb", "host_budget_mb"), ("default_server_mb", "default_server_mb"),
                         ("workspace_mb", "workspace_mb"), ("limit_kind", "limit_kind")):
        if sizing.get(key) is not None:
            memory[cfg_key] = sizing[key]
    if memory:
        data["memory"] = memory
    service = {k: sizing[s] for k, s in (("mem_limit_mb", "service_mem_limit_mb"),
                                         ("max_resident_mb", "service_max_resident_mb"),
                                         ("max_concurrency", "service_max_concurrency")) if sizing.get(s) is not None}
    if service:
        data["service"] = service
    prof["data"] = data
    if sizing.get("bwrap"):
        prof["bash"] = {"sandbox": {"os": "bwrap"}}
    if tool_env:
        prof["tool_env"] = dict(sorted(tool_env.items()))
    return prof


def host_env(layout: Any, serving: Mapping[str, Any], *, data_roots: Mapping[str, str], profiles: list[str],
             deploy: str, llm_url: str | None = None) -> dict[str, str]:
    """The ``host.env`` variables (never secrets: those stay in the operator's secrets file)."""
    env: dict[str, str] = {
        "VBT_HOME": str(layout.home) if layout.home else "",
        "VBT_STATE_DIR": str(layout.state),
        "VBT_DATA_DIR": str(layout.data),
        "VBT_PROJECTS_DIR": str(layout.projects),
        "VBT_PROFILES": " ".join(profiles),
        "VBT_DEPLOY": deploy,
    }
    if serving.get("serving_profile"):
        env["VBT_SERVING_PROFILE"] = str(serving["serving_profile"])
    dp = int(serving.get("data_parallel_size") or 1)
    if dp > 1:
        env["VBT_LLM_DP_SIZE"] = str(dp)
    if llm_url:
        env["VBT_LLM_BASE_URL"] = llm_url
    elif deploy == "compose" and serving.get("serving_profile"):
        env["VBT_LLM_BASE_URL"] = "http://vllm:8000/v1"
    if deploy == "compose":
        env["SEARXNG_URL"] = "http://searxng:8080"
    env.update({k: v for k, v in data_roots.items() if v})
    return {k: v for k, v in env.items() if v != ""}


def render_compose_vllm(serving: Mapping[str, Any], *, profiles: Mapping[str, Mapping[str, Any]] | None = None
                        ) -> dict[str, Any] | None:
    """The ``vllm`` service of ``deploy/full/compose.yaml`` for the picked serving profile (None without one).
    Volumes and ports are compose variables (``${VBT_HOME}`` on the Docker host), never paths of this process."""
    from ..local.profiles import DOCKER_SHM_SIZE, load_local_profiles, resolve_serve, select_docker_tag

    name = serving.get("serving_profile")
    if not name:
        return None
    profiles = profiles if profiles is not None else load_local_profiles()
    dp = int(serving.get("data_parallel_size") or 1)
    spec = resolve_serve(profiles, name, variants=serving.get("variants") or (),
                         data_parallel=dp if dp > 1 else None)
    tag, _note = select_docker_tag(spec.docker, serving.get("driver_version"), engine=spec.engine,
                                   version=spec.engine_version)
    image = f"{spec.docker.get('image', 'vllm/vllm-openai')}:{tag}"
    gpus = max(1, int(spec.data_parallel_size or 1),
               int((profiles[name].get("hardware") or {}).get("gpu_count", 1) or 1))
    env = {"HF_TOKEN": "${HF_TOKEN:-}", "VLLM_API_KEY": "${VLLM_API_KEY:-}", **spec.env}
    device: dict[str, Any] = {"driver": "nvidia", "capabilities": ["gpu"]}
    if gpus == 1:
        device["device_ids"] = ["${VLLM_GPU:-0}"]
    else:
        device["count"] = gpus
    service = {
        "image": f"${{VLLM_IMAGE:-{image}}}",
        "command": [*spec.server_argv, "--host", "0.0.0.0", "--port", str(spec.container_port)],
        "shm_size": DOCKER_SHM_SIZE,
        "restart": "unless-stopped",
        "environment": env,
        "volumes": ["${VBT_HOME:?set VBT_HOME}/models:/root/.cache/huggingface"],
        "ports": [f"${{VLLM_BIND:-127.0.0.1}}:${{VLLM_PORT:-8000}}:{spec.container_port}"],
        "deploy": {"resources": {"reservations": {"devices": [device]}}},
        "healthcheck": {
            "test": ["CMD", "python3", "-c", "import urllib.request; "
                     f"urllib.request.urlopen('http://localhost:{spec.container_port}/health', timeout=5)"],
            "interval": "30s", "timeout": "10s", "retries": 3, "start_period": "40m"},
        "labels": {"io.vbt.serving-profile": name, "io.vbt.variants": ",".join(spec.variants)},
    }
    if spec.unpinned_remote_code:
        service["labels"]["io.vbt.warning"] = "trust-remote-code without a pinned revision"
    return {"services": {"vllm": service}}


def _quote_env(value: str) -> str:
    return shlex.quote(str(value))


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of a file written by :func:`write_host_files` (shell-quoted values)."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = [raw]
        out[key.strip()] = parts[0] if parts else ""
    return out


SECRETS_ENV = "secrets.env"


def secrets_path(environ: Mapping[str, str]) -> Path | None:
    """``$VBT_SECRETS_FILE``, else ``$VBT_HOME/secrets.env`` (DEPLOYMENT §7.4); None without either."""
    if environ.get("VBT_SECRETS_FILE"):
        return Path(environ["VBT_SECRETS_FILE"]).expanduser()
    if environ.get("VBT_HOME"):
        return Path(environ["VBT_HOME"]).expanduser() / SECRETS_ENV
    return None


def read_secrets_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines of the operator's secrets file (optionally ``export``, optionally quoted), read as text and
    never evaluated (a secret may hold ``$``, quotes or backticks), as deploy.sh and vbt-host read it. A file other
    users can read is refused (ValueError)."""
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError(f"{path} is readable by other users (mode {path.stat().st_mode & 0o777:o}); chmod 600 it")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def write_host_files(state_dir: Path, profile: Mapping[str, Any], env: Mapping[str, str],
                     compose: Mapping[str, Any] | None, *, header: str) -> list[Path]:
    """Write ``host.yaml``, ``host.env`` and ``compose.vllm.yaml`` (removed when there is no vLLM service)."""
    state_dir = Path(state_dir)
    written = []
    prof_path = state_dir / HOST_PROFILE
    _atomic_write(prof_path, "".join(f"# {ln}\n" for ln in header.splitlines())
                  + yaml.safe_dump(dict(profile), sort_keys=False, default_flow_style=False))
    written.append(prof_path)
    env_path = state_dir / HOST_ENV
    lines = [f"# {ln}" for ln in header.splitlines()]
    lines += [f"{k}={_quote_env(v)}" for k, v in env.items()]
    _atomic_write(env_path, "\n".join(lines) + "\n", mode=0o640)
    written.append(env_path)
    compose_path = state_dir / COMPOSE_VLLM
    if compose:
        _atomic_write(compose_path, "".join(f"# {ln}\n" for ln in header.splitlines())
                      + yaml.safe_dump(json.loads(json.dumps(compose)), sort_keys=False, width=120))
        written.append(compose_path)
    elif compose_path.exists():
        compose_path.unlink()
    return written

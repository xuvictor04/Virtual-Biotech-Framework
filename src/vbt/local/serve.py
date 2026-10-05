"""``vbt local profiles`` and ``vbt local serve``: pick and start a local inference server.

* ``vbt local profiles`` lists the serving profiles of ``configs/local_models.yaml``;
  ``--show NAME`` prints one (hardware, capacity, caveats, rendered command);
  ``--detect`` reads ``nvidia-smi`` (or ``--nvidia-smi-file``) and recommends one.
* ``vbt local serve --profile h100 [--bulk] [--variant V] [--docker] [--dry-run]``
  renders the ``vllm serve`` (bare metal) or ``docker run`` command for a profile
  and executes it (``--dry-run`` prints it instead). Without ``--profile`` the
  profile is picked from ``nvidia-smi``. The docker image tag follows the NVIDIA
  driver (``v0.31.0`` for driver >= 580, ``v0.31.0-cu129`` for 575-579).

Also runnable without the harness's dependencies (``deploy/local/serve_vllm.sh``)::

    PYTHONPATH=src python3 -m vbt.local.serve --profile h100 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .profiles import (
    LocalProfileError,
    ProfilePick,
    ServeSpec,
    detect_nvidia_smi,
    docker_run_argv,
    load_local_profiles,
    parse_nvidia_smi,
    pick_profile,
    resolve_serve,
    shell_command,
)


def _exec(argv: Sequence[str], env: Mapping[str, str]) -> int:  # pragma: no cover - replaced in tests
    """Replace this process with ``argv`` (the server keeps the terminal and its signals)."""
    os.execvpe(argv[0], list(argv), dict(env))
    return 0


def _nvidia_smi_text(args: argparse.Namespace) -> str | None:
    path = getattr(args, "nvidia_smi_file", None)
    if path:
        return Path(path).read_text()
    return detect_nvidia_smi()


def _print_pick(pick: ProfilePick, out: Any = None) -> None:
    out = out or sys.stdout
    gpu = f"{pick.gpu_count} x {pick.gpu_name}" if pick.gpu_name else "none"
    vram = f", {pick.vram_gib:g} GiB" if pick.vram_gib else ""
    driver = f", driver {pick.driver_version}" if pick.driver_version else ""
    print(f"GPUs: {gpu}{vram}{driver}", file=out)
    if pick.profile:
        print(f"Recommended: {pick.reason}", file=out)
        print(f"  vbt local serve {' '.join(pick.serve_args())}" + (" --docker" if pick.docker_tag else ""), file=out)
        if pick.docker_tag:
            print(f"  docker image tag: {pick.docker_tag}", file=out)
    else:
        print(f"No serving profile fits: {pick.reason}", file=out)
    for a in pick.alternatives:
        print(f"  alternative: {a}", file=out)
    for w in pick.warnings:
        print(f"  note: {w}", file=out)


# ---------------------------------------------------------------- vbt local profiles

def _profile_rows(profiles: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    rows = [["profile", "GPUs", "checkpoint", "context", "KV", "seqs", "MTP", "harness profile", "verification"]]
    for name, p in profiles.items():
        mtp = p.get("mtp")
        rows.append([
            name, f"{p['hardware'].get('gpu_count', 1)}x", str(p["hf_id"]), f"{int(p['context_tokens']) // 1024}K",
            str(p.get("kv_cache_dtype") or "auto"), str(p.get("max_num_seqs") or "default"),
            f"k={mtp.get('num_speculative_tokens')}" if isinstance(mtp, Mapping) else "off",
            str((p.get("harness") or {}).get("profile", "")), str(p.get("verification", "")),
        ])
    return rows


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _show_profile(p: Mapping[str, Any], spec: ServeSpec) -> str:
    out = [f"{p['name']}: {p.get('title', '')}", f"  digest profile: {p.get('digest_profile', '')}",
           f"  hardware: {p['hardware'].get('gpus', '')}", f"  checkpoint: {p['hf_id']}"]
    if p.get("tokenizer"):
        out.append(f"  tokenizer: {p['tokenizer']}")
    for alt in p.get("alternatives") or []:
        out.append(f"  alternative: {alt.get('hf_id')} [{alt.get('verification')}] {alt.get('note', '').strip()}")
    eng = p.get("engine") or {}
    out.append(f"  engine: {eng.get('name', 'vllm')} >= {eng.get('min_version')} (fallback {eng.get('fallback_version')})")
    out.append(f"  served model: {p['served_model_name']} (family {p['family']}); vision: {bool(p.get('vision'))}")
    out.append(f"  verification: {p.get('verification')} - {str(p.get('verification_detail', '')).strip()}")
    for k, v in (p.get("capacity") or {}).items():
        out.append(f"  capacity.{k}: {v}")
    for vname, var in (p.get("variants") or {}).items():
        out.append(f"  variant {vname}: {str((var or {}).get('description', '')).strip()}")
    h = p.get("harness") or {}
    out.append(f"  harness: vbt --profile {h.get('profile')} ... (client max_concurrency {h.get('max_concurrency')}, "
               f"bulk concurrency {h.get('bulk_concurrency')})")
    for c in p.get("caveats") or []:
        out.append(f"  caveat: {c}")
    out.append("")
    out.append(shell_command(spec.vllm_argv(), spec.env))
    return "\n".join(out)


def cmd_profiles(args: argparse.Namespace, config: Mapping[str, Any] | None = None) -> int:
    try:
        profiles = load_local_profiles(getattr(args, "models_file", None))
    except LocalProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "detect", False) or getattr(args, "nvidia_smi_file", None):
        text = _nvidia_smi_text(args)
        if text is None:
            print("nvidia-smi not found or failed; pass --nvidia-smi-file FILE (output of "
                  "`nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits`)",
                  file=sys.stderr)
            return 2
        pick = pick_profile(text, profiles)
        if getattr(args, "json", False):
            print(json.dumps(pick.as_dict(), indent=1))
        else:
            _print_pick(pick)
        return 0 if pick.profile else 1
    if getattr(args, "show", None):
        name = args.show
        if name not in profiles:
            print(f"error: unknown serving profile {name!r}; available: {', '.join(profiles)}", file=sys.stderr)
            return 2
        spec = resolve_serve(profiles, name)
        if getattr(args, "json", False):
            print(json.dumps({"profile": profiles[name], "serve": spec.as_dict()}, indent=1, default=str))
        else:
            print(_show_profile(profiles[name], spec))
        return 0
    if getattr(args, "json", False):
        print(json.dumps(profiles, indent=1, default=str))
    else:
        print(_table(_profile_rows(profiles)))
        print("\n`vbt local profiles --show NAME` for details; `--detect` picks one from nvidia-smi.")
    return 0


# ---------------------------------------------------------------- vbt local serve

def cmd_serve(args: argparse.Namespace, config: Mapping[str, Any] | None = None) -> int:
    try:
        profiles = load_local_profiles(getattr(args, "models_file", None))
    except LocalProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    name = getattr(args, "serving_profile", None)
    variants = list(getattr(args, "variant", None) or [])
    data_parallel = getattr(args, "data_parallel", None)
    smi = None
    if not name or (getattr(args, "docker", False) and not getattr(args, "driver", None)):
        smi = _nvidia_smi_text(args)
    info = parse_nvidia_smi(smi) if smi else None
    if not name:
        if not smi:
            print("error: no --profile given and nvidia-smi is unavailable; pass --profile NAME "
                  f"({', '.join(profiles)})", file=sys.stderr)
            return 2
        pick = pick_profile(info, profiles)
        if not pick.profile:
            _print_pick(pick, out=sys.stderr)
            print(f"error: pass --profile NAME ({', '.join(profiles)})", file=sys.stderr)
            return 2
        name = pick.profile
        variants = list(dict.fromkeys([*pick.variants, *variants]))
        if data_parallel is None and pick.data_parallel_size > 1:
            data_parallel = pick.data_parallel_size
        print(f"# auto-detected: {pick.reason}")
    try:
        spec = resolve_serve(profiles, name, variants=variants, bulk=bool(getattr(args, "bulk", False)),
                             hf_id=getattr(args, "hf_id", None), engine_version=getattr(args, "engine_version", None),
                             data_parallel=data_parallel, host=getattr(args, "host", None),
                             port=getattr(args, "port", None), extra_args=getattr(args, "vllm_arg", None) or [])
    except LocalProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    notes = list(spec.notes)
    if getattr(args, "docker", False):
        driver = getattr(args, "driver", None) or (info.driver_version if info else None)
        try:
            argv, dnotes = docker_run_argv(spec, driver=driver, hf_cache=getattr(args, "hf_cache", None),
                                           name=getattr(args, "name", None) or "vbt-vllm",
                                           bind=getattr(args, "bind", None) or "127.0.0.1",
                                           detach=bool(getattr(args, "detach", False)))
        except LocalProfileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        notes += dnotes
        env = dict(os.environ)
        command = shell_command(argv, head=3)
    else:
        argv = spec.vllm_argv()
        env = {**os.environ, **spec.env}
        command = shell_command(argv, spec.env)
        if spec.host in ("0.0.0.0", "::"):
            notes.append("listening on all interfaces: vLLM has no authentication unless VLLM_API_KEY (or "
                         "--api-key) is set; set the same key as VBT_LLM_API_KEY for the harness")
    harness = spec.harness.get("profile")
    if getattr(args, "json", False):
        print(json.dumps({**spec.as_dict(), "argv": argv, "notes": notes, "docker": bool(getattr(args, "docker", False))},
                         indent=1))
    else:
        title = f"serving profile {spec.profile}" + (f" + {', '.join(spec.variants)}" if spec.variants else "")
        print(f"# {title}: {spec.hf_id} as '{spec.served_model_name}' (vLLM {spec.engine_version}, "
              f"{spec.context_tokens} tokens, KV {spec.kv_cache_dtype or 'auto'}, "
              f"max-num-seqs {spec.max_num_seqs or 'default'})")
        for n in notes:
            print(f"# note: {n}")
        if harness:
            print(f"# harness: export VBT_LLM_BASE_URL={spec.base_url}; vbt --profile {harness} ...; "
                  "check the server with `vbt local check`")
        print(command)
    if getattr(args, "dry_run", False) or getattr(args, "json", False):
        return 0
    exe = argv[0]
    if not shutil.which(exe):
        hint = ("install Docker with the NVIDIA Container Toolkit" if exe == "docker" else
                f"install vLLM {spec.engine.get('min_version')} in its own environment "
                f"(pip install 'vllm=={spec.engine_version}') or use --docker")
        print(f"error: {exe!r} not found on PATH: {hint}", file=sys.stderr)
        return 127
    sys.stdout.flush()
    return _exec(argv, env)


# ---------------------------------------------------------------- parsers

def add_profiles_parser(lsub: Any) -> argparse.ArgumentParser:
    p = lsub.add_parser("profiles", help="list serving profiles (configs/local_models.yaml); --detect picks one")
    p.add_argument("--show", metavar="NAME", help="details and the rendered command of one profile")
    p.add_argument("--detect", action="store_true", help="recommend a profile from nvidia-smi")
    p.add_argument("--nvidia-smi-file", metavar="FILE", help="use this nvidia-smi output instead of running it")
    p.add_argument("--models-file", metavar="FILE", help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_profiles)
    return p


def add_serve_parser(lsub: Any) -> argparse.ArgumentParser:
    p = lsub.add_parser("serve", help="start (or print with --dry-run) the vLLM server for a serving profile")
    _add_serve_arguments(p)
    p.set_defaults(handler=cmd_serve)
    return p


def _add_serve_arguments(p: argparse.ArgumentParser) -> None:
    # dest differs from the global --profile (harness config profiles)
    p.add_argument("--profile", dest="serving_profile", metavar="NAME",
                   help="serving profile (h100, h200, rtxpro6000, b200, 5090, dp, deepseek-v4); default: from "
                        "nvidia-smi")
    p.add_argument("--variant", action="append", default=[], metavar="NAME",
                   help="profile variant (repeatable), e.g. bulk, mtp2, fp8kv, eager")
    p.add_argument("--bulk", action="store_true", help="bulk-annotation-only server (= --variant bulk)")
    p.add_argument("--docker", action="store_true", help="render/run a `docker run` line instead of bare metal")
    p.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    p.add_argument("--json", action="store_true", help="print the resolved spec as JSON (implies --dry-run)")
    p.add_argument("--hf-id", metavar="REPO", help="serve another checkpoint (e.g. a listed alternative)")
    p.add_argument("--engine-version", metavar="X.Y.Z",
                   help="vLLM version to target (0.30.0 drops --tool-strict-level; picks the docker tag)")
    p.add_argument("--driver", metavar="VERSION", help="NVIDIA driver version for the docker tag (default: nvidia-smi)")
    p.add_argument("--data-parallel", type=int, metavar="N", help="data-parallel replicas (--data-parallel-size)")
    p.add_argument("--host", help="bind address (bare metal; default 127.0.0.1)")
    p.add_argument("--port", type=int, help="port (default 8000)")
    p.add_argument("--bind", help="docker: host address the port is published on (default 127.0.0.1)")
    p.add_argument("--hf-cache", metavar="DIR", help="docker: Hugging Face cache to mount (default $HF_HOME or "
                                                     "~/.cache/huggingface)")
    p.add_argument("--name", help="docker: container name (default vbt-vllm)")
    p.add_argument("--detach", action="store_true", help="docker: run in the background (-d)")
    p.add_argument("--vllm-arg", action="append", default=[], metavar="ARG",
                   help="extra vllm serve argument, appended verbatim (repeatable; use --vllm-arg=--enforce-eager)")
    p.add_argument("--nvidia-smi-file", metavar="FILE", help="use this nvidia-smi output instead of running it")
    p.add_argument("--models-file", metavar="FILE", help=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point (``python -m vbt.local.serve``): the ``serve`` options,
    plus ``--list`` and ``--detect``."""
    p = argparse.ArgumentParser(prog="python -m vbt.local.serve",
                                description="Render or start the vLLM server for a serving profile")
    _add_serve_arguments(p)
    p.add_argument("--list", action="store_true", help="list the serving profiles")
    p.add_argument("--detect", action="store_true", help="recommend a profile from nvidia-smi")
    args = p.parse_args(argv)
    if args.list or args.detect:
        args.show = None
        return cmd_profiles(args)
    return cmd_serve(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

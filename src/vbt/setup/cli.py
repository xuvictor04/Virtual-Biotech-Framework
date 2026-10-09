"""``vbt setup``: probe the host, write its configuration, and run the generic bring-up steps with resume."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, TextIO

from . import hostconfig
from .layout import resolve_layout
from .state import SetupState, fingerprint
from .steps import STEP_NAMES, STEPS, SetupContext, StepResult, ensure_configured

__all__ = ["add_setup_parser", "cmd_setup", "select_steps", "build_context", "run_steps", "make_plan"]

#: Outcomes that let the next steps run.
_OK = ("done", "skipped", "deferred", "unavailable", "up to date")


def add_setup_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "setup", help="bring up this host: probe, host config, data, indexes, readiness, calibration, smoke tests",
        description="Probe the host (GPUs, memory, disk, containment, network), pick the local-model serving profile, "
                    "write the host configuration (<state>/host.yaml, host.env, compose.vllm.yaml) and run the "
                    "generic steps: " + ", ".join(STEP_NAMES) + ". Finished steps are skipped on the next run "
                    "unless their inputs changed. docs/DEPLOYMENT.md.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print the steps, sizes and time estimates; change nothing")
    mode.add_argument("--probe", action="store_true", help="print the host facts only")
    mode.add_argument("--status", action="store_true", help="print the recorded outcome of every step")
    p.add_argument("--json", action="store_true", help="JSON output")
    sel = p.add_argument_group("steps")
    sel.add_argument("--only", metavar="STEP[,STEP]", help=f"run only these steps ({', '.join(STEP_NAMES)})")
    sel.add_argument("--from", dest="from_step", metavar="STEP", help="start at this step")
    sel.add_argument("--skip", metavar="STEP[,STEP]", help="leave these steps out")
    sel.add_argument("--force", action="store_true", help="run the selected steps even when they are up to date")
    sel.add_argument("--keep-going", action="store_true", help="run the later steps after a failed one")
    where = p.add_argument_group("directories (default: $VBT_HOME/<name>, else the checkout's data/ and runs/)")
    where.add_argument("--home", help="deployment root (VBT_HOME): data/, runs/, projects/, state/, models/")
    where.add_argument("--state-dir", help="host configuration and setup state (VBT_STATE_DIR)")
    where.add_argument("--data-dir", help="reference data (VBT_DATA_DIR)")
    where.add_argument("--projects-dir", help="project files (VBT_PROJECTS_DIR)")
    where.add_argument("--models-dir", help="the model server's Hugging Face cache (HF_CACHE)")
    model = p.add_argument_group("model")
    model.add_argument("--serving-profile", metavar="NAME",
                       help="serving profile of configs/local_models.yaml (default: picked from the GPUs)")
    model.add_argument("--variant", action="append", default=[], metavar="NAME", help="serving profile variant")
    model.add_argument("--harness-profile", metavar="NAME",
                       help="harness profile instead of the serving profile's own (e.g. claude, paper)")
    model.add_argument("--llm-url", metavar="URL", help="the model server's OpenAI-compatible URL when it runs elsewhere")
    model.add_argument("--nvidia-smi-file", metavar="FILE",
                       help="nvidia-smi output to use (a container sees no GPU; deploy.sh writes <state>/nvidia-smi.csv)")
    misc = p.add_argument_group("other")
    misc.add_argument("--deploy", choices=("host", "compose"), default=None,
                      help="how the harness runs: on this host, or in deploy/full/compose.yaml (writes the vllm "
                           "service and the in-network URLs); default: compose inside a container, else host")
    misc.add_argument("--no-network", action="store_true", help="do not probe the network endpoints")
    misc.add_argument("--check-url", action="append", default=[], metavar="URL", help="also probe this URL")
    misc.add_argument("--no-analysis", action="store_true", help="smoke: skip the Python/R analysis stack check")
    misc.add_argument("--yes", action="store_true", help="accepted for scripts; setup never prompts")
    p.set_defaults(handler=cmd_setup)
    return p


def select_steps(only: str | None = None, from_step: str | None = None, skip: str | None = None) -> list[str]:
    """The step names to run, in order. Raises ValueError for an unknown name."""
    def names(text: str | None) -> list[str]:
        out = [n.strip() for n in (text or "").split(",") if n.strip()]
        bad = [n for n in out if n not in STEP_NAMES]
        if bad:
            raise ValueError(f"unknown step(s) {', '.join(bad)}; steps: {', '.join(STEP_NAMES)}")
        return out

    chosen = names(only) or list(STEP_NAMES)
    if from_step:
        start = names(from_step)[0]
        chosen = [n for n in chosen if STEP_NAMES.index(n) >= STEP_NAMES.index(start)]
    dropped = set(names(skip))
    return [n for n in STEP_NAMES if n in chosen and n not in dropped]


def _nvidia_text(args: argparse.Namespace, state_dir: Path) -> tuple[str | None, str | None]:
    if getattr(args, "nvidia_smi_file", None):
        return Path(args.nvidia_smi_file).read_text(), "file"
    saved = state_dir / "nvidia-smi.csv"
    if shutil.which("nvidia-smi") is None and saved.is_file():
        return saved.read_text(), "state/nvidia-smi.csv"
    return None, None


def build_context(args: argparse.Namespace, config: dict[str, Any], out: TextIO | None = None) -> SetupContext:
    layout = resolve_layout(config, home=args.home, state=args.state_dir, data=args.data_dir,
                            runs=getattr(args, "runs_dir", None), projects=args.projects_dir, models=args.models_dir)
    state = SetupState(layout.state)
    text, source = _nvidia_text(args, layout.state)
    from .probe import container_facts

    deploy = args.deploy or ("compose" if container_facts().get("inside") in ("docker", "podman", "container")
                             else "host")
    options = {
        "serving_profile": args.serving_profile, "variants": list(args.variant or []),
        "harness_profile": args.harness_profile, "llm_url": args.llm_url, "deploy": deploy,
        "analysis": not args.no_analysis,
        "probe_kwargs": {"nvidia_smi_text": text, "nvidia_smi_source": source, "network": not args.no_network,
                         "extra_urls": list(args.check_url or [])},
    }
    return SetupContext(config=config, layout=layout, state=state, options=options, out=out or sys.stdout,
                        verbose=bool(getattr(args, "verbose", False)))


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "-"
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _fmt_bytes(b: float | None) -> str:
    if not b:
        return "-"
    return f"{b / 1e9:,.2f} GB" if b >= 1e8 else f"{b / 1e6:,.1f} MB"


def make_plan(ctx: SetupContext, names: list[str]) -> dict[str, Any]:
    """Every selected step's plan: action, sizes, time estimate, whether it is up to date. Changes nothing."""
    steps = {s.name: s for s in STEPS}
    rows: list[dict[str, Any]] = []
    steps["probe"].run(ctx)
    ensure_configured(ctx, write=False)
    for name in names:
        step = steps[name]
        try:
            plan = step.plan(ctx)
        except Exception as exc:  # noqa: BLE001 - a plan never aborts the others
            plan = {"action": "unknown", "detail": f"{type(exc).__name__}: {exc}"}
        if name == "acquire":
            ctx.planned_fetch_bytes = int(plan.get("bytes") or 0)
        fp = fingerprint(name, step.inputs(ctx))
        rec = ctx.state.step(name)
        if name not in ("probe", "configure", "smoke") and ctx.state.is_current(name, fp):
            plan = {**plan, "action": "up to date", "seconds": 0}
        rows.append({"step": name, "help": step.help, "last": rec.get("status"), **plan})
    total_s = sum(float(r.get("seconds") or 0) for r in rows if r.get("action") not in ("up to date", "nothing"))
    fetch = sum(int(r.get("bytes") or 0) for r in rows if r.get("action") == "run")
    disk = (ctx.facts or {}).get("disk") or {}
    free = (disk.get("data") or {}).get("free_bytes")
    notes = list((ctx.sizing or {}).get("notes") or []) + list((ctx.serving or {}).get("warnings") or [])
    if fetch and free is not None and fetch > free:
        notes.append(f"{_fmt_bytes(fetch)} to fetch but {_fmt_bytes(free)} free under {ctx.layout.data}")
    unreachable = sorted(k for k, v in ((ctx.facts or {}).get("network") or {}).items()
                         if v.get("reachable") is False)
    if unreachable:
        notes.append("unreachable: " + ", ".join(f"{k} ({(ctx.facts['network'][k] or {}).get('url')})"
                                                 for k in unreachable))
    if (ctx.needs and ctx.needs.errors):
        notes.extend(ctx.needs.errors[:10])
    return {"layout": ctx.layout.as_dict(), "serving": ctx.serving, "sizing": ctx.sizing,
            "needs": {"tools": len(ctx.needs.tools) if ctx.needs else 0,
                      "sources": {k: {kk: v.get(kk) for kk in ("kind", "release", "root_var")}
                                  for k, v in (ctx.needs.sources if ctx.needs else {}).items()}},
            "env": ctx.pending_env, "steps": rows, "total_seconds": total_s, "fetch_bytes": fetch,
            "free_bytes": free, "notes": notes}


def print_plan(plan: Mapping[str, Any], out: TextIO) -> None:
    lay = plan["layout"]
    s = plan.get("serving") or {}
    z = plan.get("sizing") or {}
    out.write(f"state {lay['state']}\ndata  {lay['data']}\nruns  {lay['runs']}\nprojects {lay['projects']}\n")
    out.write(f"model: serving profile {s.get('serving_profile') or 'none'} "
              f"(harness profile {s.get('harness_profile') or 'default'}): {s.get('reason')}\n")
    if z.get("ram_mb"):
        out.write(f"memory: {z['ram_mb']:,} MB RAM, {z.get('cpus')} CPUs -> host budget "
                  f"{z.get('host_budget_mb', 0):,} MB, server limit {z.get('default_server_mb', 0):,} MB, data child "
                  f"{z.get('service_mem_limit_mb', 0):,} MB, workspace {z.get('workspace_mb', 0):,} MB"
                  f"{', cgroup containment' if z.get('limit_kind') == 'cgroup' else ''}"
                  f"{', bwrap sandbox' if z.get('bwrap') else ''}\n")
    out.write(f"needs: {plan['needs']['tools']} tool(s); sources: "
              + ", ".join(f"{k} ({v.get('kind')}{', ' + v['release'] if v.get('release') else ''})"
                          for k, v in sorted(plan['needs']['sources'].items())) + "\n\n")
    out.write(f"{'step':<10} {'action':<12} {'download':>10} {'time':>8}  detail\n")
    for r in plan["steps"]:
        out.write(f"{r['step']:<10} {str(r.get('action')):<12} {_fmt_bytes(r.get('bytes')):>10} "
                  f"{_fmt_seconds(r.get('seconds')):>8}  {r.get('detail') or ''}\n")
    out.write(f"\ntotal: {_fmt_bytes(plan['fetch_bytes'])} to fetch, about {_fmt_seconds(plan['total_seconds'])}"
              f"{'; free under data: ' + _fmt_bytes(plan['free_bytes']) if plan.get('free_bytes') else ''}\n")
    for n in plan.get("notes") or []:
        out.write(f"note: {n}\n")


def run_steps(ctx: SetupContext, names: list[str], *, force: bool = False, keep_going: bool = False) -> int:
    """Run ``names`` in order with resume; 0 when none failed."""
    steps = {s.name: s for s in STEPS}
    failed = 0
    for name in names:
        step = steps[name]
        if name not in ("probe", "configure"):
            if ctx.needs is None or ctx.serving is None:
                ensure_configured(ctx, write=not Path(ctx.layout.state, hostconfig.HOST_PROFILE).exists())
        fp = fingerprint(name, step.inputs(ctx)) if name not in ("probe", "configure", "smoke") else None
        if fp and not force and ctx.state.is_current(name, fp):
            ctx.say(f"[{name}] up to date ({ctx.state.step(name).get('detail', '')})")
            continue
        ctx.say(f"[{name}] {step.help} ...")
        t0 = time.monotonic()
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            result = step.run(ctx)
        except Exception as exc:  # noqa: BLE001 - recorded as the step's failure
            result = StepResult("failed", f"{type(exc).__name__}: {exc}")
        took = round(time.monotonic() - t0, 1)
        if result.status == "done" and (fp is None or getattr(step, "refingerprint", False)):
            ctx.state.record(name, data=result.data)      # the inputs may read what the step just recorded
            fp = fingerprint(name, step.inputs(ctx))
        ctx.state.record(name, status=result.status, detail=result.detail, data=result.data, seconds=took,
                         started=started, finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         fingerprint=fp if result.status == "done" else None)
        ctx.say(f"[{name}] {result.status} in {_fmt_seconds(took)}: {result.detail}")
        if result.status not in _OK:
            failed += 1
            if not keep_going:
                ctx.say(f"stopped after {name}; fix it and rerun `vbt setup` (finished steps are skipped)")
                break
    ctx.state.save()
    return 1 if failed else 0


def _print_probe(facts: Mapping[str, Any], out: TextIO) -> None:
    cpu, mem, gpu = facts.get("cpu") or {}, facts.get("memory") or {}, facts.get("gpu") or {}
    out.write(f"host {facts.get('hostname')}: {cpu.get('effective')} CPUs, {mem.get('effective_mb')} MB RAM "
              f"(MemTotal {mem.get('total_mb')} MB, cgroup limit {mem.get('cgroup_limit_mb') or 'none'})\n")
    out.write(f"GPUs: {gpu.get('count', 0)}" + (f" x {gpu['gpus'][0]['name']} ({gpu['gpus'][0].get('vram_gib')} GiB), "
                                                f"driver {gpu.get('driver_version')}" if gpu.get("gpus") else "") + "\n")
    for name, d in (facts.get("disk") or {}).items():
        if "free_bytes" in d:
            out.write(f"disk {name}: {_fmt_bytes(d['free_bytes'])} free of {_fmt_bytes(d['total_bytes'])} "
                      f"({d['path']})\n")
    cont = facts.get("containment") or {}
    out.write(f"containment: {cont.get('limit_kind')} ({cont.get('note')})\n")
    sb = facts.get("sandbox") or {}
    out.write(f"sandbox: bwrap {'works' if (sb.get('bwrap') or {}).get('works') else 'unavailable'}, "
              f"unshare -rn {'works' if (sb.get('unshare_net') or {}).get('works') else 'unavailable'}\n")
    c = facts.get("container") or {}
    out.write(f"container: {c.get('inside') or 'none'}; docker compose {c.get('docker_compose') or 'absent'}\n")
    for label, r in sorted((facts.get("network") or {}).items()):
        status = "skipped" if r.get("skipped") else (f"HTTP {r.get('status')}" if r.get("reachable")
                                                     else f"UNREACHABLE ({r.get('error')})")
        out.write(f"net {label}: {r.get('url')} {status}\n")


def cmd_setup(args: argparse.Namespace, config: dict[str, Any]) -> int:
    try:
        names = select_steps(args.only, args.from_step, args.skip)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = sys.stdout
    try:
        ctx = build_context(args, config, out=sys.stderr if args.json else out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.status:
        data = {"state": str(ctx.state.path), "steps": ctx.state.summary(), "rates": ctx.state.data.get("rates")}
        if args.json:
            print(json.dumps(data, indent=1, default=str))
        else:
            print(f"state: {data['state']}")
            for name in STEP_NAMES:
                rec = data["steps"].get(name) or {}
                print(f"  {name:<10} {rec.get('status', '-'):<12} {rec.get('finished', ''):<21} {rec.get('detail', '')}")
        return 0
    if args.probe:
        from .probe import probe_host

        facts = probe_host(ctx.layout, config, **ctx.options["probe_kwargs"])
        facts.get("gpu", {}).pop("raw", None)
        if args.json:
            print(json.dumps(facts, indent=1, default=str))
        else:
            _print_probe(facts, out)
        return 0
    if args.plan:
        try:
            plan = make_plan(ctx, names)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        ctx.state.data["plan"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  "steps": plan["steps"], "total_seconds": plan["total_seconds"]}
        if args.json:
            print(json.dumps(plan, indent=1, default=str))
        else:
            print_plan(plan, out)
        return 0
    try:
        return run_steps(ctx, names, force=args.force, keep_going=args.keep_going)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

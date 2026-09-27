"""Command-line interface: `vbt chat | run | tools | doctor | verify | bulk | case1 | scenario`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path


# ---------------------------------------------------------------- helpers

def _config(args) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if getattr(args, "model", None):
        overrides["models"] = {t: {"model": args.model} for t in ("orchestrator", "scientist", "bulk")}
    if getattr(args, "no_web", False):
        overrides["web"] = {"enabled": False}
    if getattr(args, "runs_dir", None):
        overrides["paths"] = {"runs_dir": args.runs_dir}
    return load_config(args.profile, overrides)


def _printer(verbose: bool):
    from rich.console import Console

    console = Console(highlight=False)

    def on_event(kind: str, d: dict[str, Any]) -> None:
        if kind == "text" and d.get("agent") == "cso":
            console.out(d["text"], end="")
        elif kind == "delegation":
            console.print(f"\n[bold cyan]→ delegating to {d['agent']}[/]: {d.get('description', '')}")
        elif kind == "agent_end" and d.get("depth", 0) > 0:
            console.print(f"[cyan]✓ {d['agent']} finished (${d.get('cost_usd', 0):.2f})[/]")
        elif kind == "tool" and (verbose or d["agent"] == "cso") and d["tool"] != "Task":
            console.print(f"[dim]  [{d['agent']}] {d['tool']}[/]")
        elif kind == "warning":
            console.print(f"[yellow]! {d['message']}[/]")
    return console, on_event


async def _session(args, config):
    from .orchestrator import open_session

    console, on_event = _printer(args.verbose)
    session = await open_session(config, on_event=on_event, start_mcp=not args.no_mcp)
    console.print(f"[bold]The Virtual Biotech[/] — run [green]{session.run.run_id}[/] "
                  f"({config['provider']['name']}: {config['models']['orchestrator']['model']})")
    console.print(f"Run directory: {session.run.dir}")
    if session.rt.mcp and session.rt.mcp.failures:
        console.print(f"[yellow]Unavailable MCP servers: {', '.join(sorted(session.rt.mcp.failures))}[/]")
    return console, session


def _summary(console, session) -> None:
    c = session.run.cost.report()
    console.print(f"\n[bold]Session cost:[/] ${c['total_usd']:.2f}")
    for a, d in c["agents"].items():
        console.print(f"  {a:30s} ${d['usd']:.2f}  ({d['model_calls']} calls)")


# ---------------------------------------------------------------- commands

async def cmd_chat(args) -> int:
    config = _config(args)
    console, session = await _session(args, config)
    console.print('Type your question. Commands: /summary, /done. Multi-line: start and end with """.\n')
    try:
        while True:
            try:
                line = input("\nYou: ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.startswith('"""'):
                buf = [line[3:]] if line[3:] else []
                while (nxt := input("... ")).strip() != '"""':
                    buf.append(nxt)
                line = "\n".join(buf).strip()
            if line in ("/done", "quit", "exit"):
                break
            if line == "/summary":
                _summary(console, session)
                continue
            console.print("\n[bold magenta]CSO:[/] ", end="")
            try:
                await session.ask(line)
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                console.print(f"\n[red]Turn failed: {exc}[/]")
            console.print(f"\n[dim](turn cost ${session.run.turns[-1]['cost_usd']:.2f}; "
                          f"total ${session.run.cost.total_usd:.2f})[/]")
    finally:
        _summary(console, session)
        await session.close()
        console.print(f"Records: {session.run.dir}")
    return 0


async def cmd_run(args) -> int:
    config = _config(args)
    turns = list(args.queries)
    if args.file:
        turns += [l.strip() for l in Path(args.file).read_text().splitlines() if l.strip()]
    if not turns:
        print("no queries given", file=sys.stderr)
        return 2
    console, session = await _session(args, config)
    try:
        for q in turns:
            console.print(f"\n[bold]You:[/] {q}\n[bold magenta]CSO:[/] ", end="")
            await session.ask(q)
    finally:
        _summary(console, session)
        await session.close()
        console.print(f"Records: {session.run.dir}")
    return 0


async def cmd_tools(args) -> int:
    from .runtime import Runtime
    from .session import Run
    import tempfile

    config = _config(args)
    rt = Runtime(config, Run(Path(tempfile.mkdtemp())))
    if not args.no_mcp:
        await rt.start_mcp()
    try:
        for name, agent in [("cso", rt.cso), *rt.agents.items()]:
            tools = [t.name for t in rt.tools_for(agent)]
            missing = rt.registry.missing(agent.tools)
            print(f"\n{name} [{agent.tier}: {agent.settings(config).model}] — {len(tools)} tools")
            print("  " + ", ".join(tools))
            if missing:
                print(f"  (unresolved: {', '.join(missing)})")
        if rt.mcp and rt.mcp.failures:
            print("\nMCP failures:", json.dumps(rt.mcp.failures, indent=1))
    finally:
        await rt.aclose()
    return 0


def cmd_doctor(args) -> int:
    config = _config(args)
    ok = True

    def check(label: str, cond: bool, hint: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"[{'ok' if cond else '!!'}] {label}" + ("" if cond else f"  -> {hint}"))

    up = Path(config["vars"]["upstream"])
    check("upstream submodule present", (up / "src" / "agents" / "cso" / "system_prompt.md").exists(),
          "git submodule update --init")
    try:
        from .agents import load_roster
        cso, agents = load_roster(config)
        check(f"agent roster loads ({len(agents)} agents + CSO)", True)
    except Exception as exc:  # noqa: BLE001
        check("agent roster loads", False, str(exc))
    if config["provider"]["name"] == "anthropic":
        check("ANTHROPIC_API_KEY set", bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
              "export ANTHROPIC_API_KEY or add it to .env")
    ot = os.environ.get("OPEN_TARGETS_DATA_PATH")
    check("OPEN_TARGETS_DATA_PATH set and exists", bool(ot) and Path(ot).exists(),
          "python third_party/TheVirtualBiotech/tools/download_open_targets.py <dir>")
    check("upstream clinical-trial labels present",
          (up / "datasets" / "clinical_trials" / "clinical_trial_labels_reconciled.csv").exists(), "submodule")
    if args.smoke:
        async def smoke():
            from .runtime import Runtime
            from .session import Run
            import tempfile
            rt = Runtime(config, Run(Path(tempfile.mkdtemp())))
            failures = await rt.start_mcp()
            n = len([t for t in rt.registry.names() if t.startswith("mcp__") and "provenance" not in t])
            await rt.aclose()
            return failures, n
        failures, n = asyncio.run(smoke())
        check(f"MCP servers start ({n} data tools)", not failures, json.dumps(failures))
    return 0 if ok else 1


def cmd_verify(args) -> int:
    from .verify import verify_run
    report = verify_run(resolve_path(args.run_dir))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "COMPLETE" else 1


# ---------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vbt", description="The Virtual Biotech — provider-agnostic harness")
    p.add_argument("--profile", action="append", default=[], help="config profile(s), e.g. paper, no-web, mock")
    p.add_argument("--model", help="override the model for CSO, scientists and bulk agents")
    p.add_argument("--no-web", action="store_true", help="disable WebSearch/WebFetch (no-leakage setting)")
    p.add_argument("--no-mcp", action="store_true", help="do not start MCP data servers")
    p.add_argument("--runs-dir")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat", help="interactive CSO session")
    r = sub.add_parser("run", help="headless: one argument per conversation turn")
    r.add_argument("queries", nargs="*")
    r.add_argument("-f", "--file", help="file with one turn per line")
    sub.add_parser("tools", help="list agents and their resolved tools")
    d = sub.add_parser("doctor", help="check installation")
    d.add_argument("--smoke", action="store_true", help="also start every MCP server")
    v = sub.add_parser("verify", help="check a run's artifacts and claim evidence")
    v.add_argument("run_dir")

    from .bulk import add_bulk_parser
    from .case_studies import add_case_parsers
    add_bulk_parser(sub)
    add_case_parsers(sub)

    args = p.parse_args(argv)
    if args.cmd in ("chat", "run", "tools"):
        return asyncio.run({"chat": cmd_chat, "run": cmd_run, "tools": cmd_tools}[args.cmd](args))
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    return args.handler(args, _config(args))


if __name__ == "__main__":
    sys.exit(main())

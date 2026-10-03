"""Command-line interface.

Sessions: ``vbt chat | run | replay | tools``; records: ``vbt verify | list |
index | export | audit | show`` (``vbt.audit.cli``); ``vbt doctor``
(``vbt.preflight``); ``vbt web`` (``vbt.web``); ``vbt bulk``; case studies
``vbt case1 | scenario | data`` (``vbt.case_studies``).

* ``vbt chat`` -- interactive CSO session. Ctrl+C during a turn interrupts that
  turn (recorded as ``interrupted``; the session stays usable); Ctrl+C at the
  prompt ends the session and writes the reports; a second Ctrl+C within 2 s
  force-exits. Commands: /help, /summary, /claims, /evidence <ID>, /done.
* ``vbt run`` -- headless: one argument (or ``-f`` file line) per turn. Exits
  non-zero when a turn did not complete or ``verify`` is not COMPLETE
  (``--no-strict`` disables this); ``--events ndjson`` streams the event bus as
  JSON lines on stdout.
* ``vbt replay RUN`` -- re-run a recorded session's turns and write
  ``replay_diff.json`` (a comparison, not a reproduction).
* ``--resume RUN`` (chat, run) continues a recorded session.
* ``--skip-preflight`` / ``--allow-missing-data`` (global) map to
  ``preflight.skip`` / ``preflight.allow_missing_data``: the readiness gate run
  before a session and before every turn (``vbt.preflight.require_ready``).
* ``--model`` accepts a ``model_aliases`` label, a configured model id, or an id
  matching ``provider.model_pattern``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .config import deep_merge, load_config, resolve_path


def _e(value: Any) -> str:
    """Escape text for rich markup."""
    from rich.markup import escape
    return escape(str(value if value is not None else ""))

DEFAULT_MODEL_PATTERN = r"^claude-[a-z0-9.-]+$"
MODEL_TIERS = ("orchestrator", "scientist", "bulk")
FORCE_EXIT_WINDOW_S = 2.0

HELP_TEXT = """Commands:
  /help            this help
  /summary         per-turn table and session cost
  /claims          claims filed in this run
  /evidence <ID>   evidence of one claim (e.g. /evidence C3)
  /done            end the session (also: quit, exit, Ctrl+D)
Multi-line input: start and end with three double quotes (\"\"\").
Ctrl+C during a turn interrupts it (recorded as 'interrupted'); at the prompt it ends the session."""


# ---------------------------------------------------------------- configuration

class ModelResolutionError(ValueError):
    """--model is neither an alias, a configured model, nor a valid model id."""


def _model_pattern(config: Mapping[str, Any]) -> str | None:
    prov = config.get("provider") or {}
    if prov.get("model_pattern"):
        return str(prov["model_pattern"])
    return DEFAULT_MODEL_PATTERN if prov.get("name") == "anthropic" else None


def resolve_model(config: Mapping[str, Any], value: str | None) -> str | None:
    """Resolve ``--model``: a ``model_aliases`` label, a configured model id, or an id
    matching ``provider.model_pattern``. Raises ModelResolutionError otherwise."""
    if not value:
        return None
    value = str(value).strip()
    aliases = config.get("model_aliases") or {}
    if value in aliases:
        target = aliases[value]
        return str(target.get("model") if isinstance(target, Mapping) else target)
    configured = sorted({str(t.get("model")) for t in (config.get("models") or {}).values()
                         if isinstance(t, Mapping) and t.get("model")})
    if value in configured:
        return value
    pat = _model_pattern(config)
    if pat is None or re.fullmatch(pat, value):
        return value
    alias_list = ", ".join(f"{k} -> {v.get('model') if isinstance(v, Mapping) else v}" for k, v in aliases.items())
    raise ModelResolutionError(
        f"unknown model {value!r}: not an alias, not a configured model, and not matching {pat!r}. "
        f"Aliases: {alias_list or '(none; add model_aliases to the config)'}. "
        f"Configured models: {', '.join(configured) or '(none)'}.")


def _profiles(args, extra_profiles: Iterable[str] = ()) -> list[str]:
    return list(dict.fromkeys(list(getattr(args, "profile", None) or []) + list(extra_profiles or [])))


#: Pinned-config sections a resumed session inherits (as replay does).
RESUME_KEYS = ("models", "web", "orchestration", "limits", "agent_overrides", "model_aliases")


def flag_overrides(args) -> dict[str, Any]:
    """Config overrides from the global flags (--no-web, --runs-dir, --no-clarify,
    --skip-preflight, --allow-missing-data); --model and --profile are handled
    by the callers."""
    overrides: dict[str, Any] = {}
    if getattr(args, "no_web", False):
        overrides["web"] = {"enabled": False}
    if getattr(args, "runs_dir", None):
        overrides["paths"] = {"runs_dir": args.runs_dir}
    if getattr(args, "no_clarify", False):
        overrides["orchestration"] = {"strategic_orientation": False}
    pre: dict[str, Any] = {}
    if getattr(args, "skip_preflight", False):
        pre["skip"] = True
    if getattr(args, "allow_missing_data", False):
        pre["allow_missing_data"] = True
    if pre:
        overrides["preflight"] = pre
    return overrides


def build_config(args, extra_profiles: Iterable[str] = (), *, pinned: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """The configuration for a command: profiles (plus ``extra_profiles``) and the
    global flags (--model, --no-web, --runs-dir, --no-clarify, --skip-preflight,
    --allow-missing-data). ``config['profiles']``
    records the profile list for pinning.

    With ``pinned`` (a resumed run's pinned config) and no explicit
    ``--profile``, the run's profiles, provider, models and orchestration /
    web / limit settings are the base, so a resumed session continues on the
    settings it was made with; explicit flags still apply on top."""
    profiles = _profiles(args, extra_profiles)
    base: dict[str, Any] = {}
    if pinned and not profiles:
        profiles = [str(p) for p in (pinned.get("profiles") or [])]
        for key in RESUME_KEYS:
            if isinstance(pinned.get(key), Mapping) and pinned[key]:
                base[key] = dict(pinned[key])
        prov = pinned.get("provider")
        pname = prov.get("name") if isinstance(prov, Mapping) else prov if isinstance(prov, str) else None
        if pname:
            base["provider"] = {"name": pname}  # options were redacted when pinned; profiles restore them
    overrides = flag_overrides(args)
    cfg = load_config(profiles, deep_merge(base, overrides) if base else overrides)
    model = getattr(args, "model", None)
    if model:
        m = resolve_model(cfg, model)
        cfg = deep_merge(cfg, {"models": {t: {"model": m} for t in MODEL_TIERS if t in (cfg.get("models") or {})}})
    cfg["profiles"] = profiles
    return cfg


_config = build_config  # backwards-compatible name (case-study handlers)


class RunResolutionError(Exception):
    """A run argument (--resume, replay) did not resolve to exactly one run."""


def _resolve_run(value: str, config: Mapping[str, Any]) -> Path:
    from .audit.index import AmbiguousRunError, RunNotFound, resolve_run  # lazy (handlers only)
    try:
        return resolve_run(value, resolve_path((config.get("paths") or {}).get("runs_dir", "runs")))
    except (RunNotFound, AmbiguousRunError) as exc:
        raise RunResolutionError(str(exc)) from None


# ---------------------------------------------------------------- printing

def _short(value: Any, n: int = 100) -> str:
    if isinstance(value, Mapping):
        for k in ("command", "file_path", "path", "query", "pattern", "url", "description", "subagent_type"):
            if value.get(k):
                value = value[k]
                break
        else:
            value = json.dumps(value, default=str, ensure_ascii=False)
    s = " ".join(str(value or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


class _Printer:
    """Rich renderer for the runtime event stream (CLI chat/run and scenarios)."""

    def __init__(self, console, *, verbose: bool = False, show_reasoning: bool = False, quiet: bool = False):
        from .audit.render import StreamingAnchorStripper

        self.console = console
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.quiet = quiet
        self._stripper_cls = StreamingAnchorStripper
        self.stripper = StreamingAnchorStripper()
        self.msg_chars = 0
        self.blank = False
        self.thinking_open = False
        self.seen_tools: set[str] = set()
        self.running: dict[str, dict[str, Any]] = {}
        self.status = None
        self.live_status = bool(getattr(console, "is_terminal", False)) and not verbose and not quiet

    # -- helpers
    def _out(self, text: str, **kw) -> None:
        if text:
            self.console.out(text, end="", highlight=False, **kw)

    def _end_thinking(self) -> None:
        if self.thinking_open:
            self.console.out("", highlight=False)
            self.thinking_open = False

    def _flush_text(self) -> None:
        rest = self.stripper.flush()
        if rest:
            self._out(rest)
            self.msg_chars += len(rest)

    def _status_text(self) -> str:
        now = time.time()
        bits = []
        for r in self.running.values():
            tool = f" {r['tool']}" if r.get("tool") else ""
            bits.append(f"{r['agent']}{tool} {int(now - r['t0'])}s")
        return "running: " + ", ".join(bits)

    def _update_status(self) -> None:
        if not self.live_status:
            return
        try:
            if self.running:
                if self.status is None:
                    self.status = self.console.status(self._status_text(), spinner="dots")
                    self.status.start()
                else:
                    self.status.update(self._status_text())
            elif self.status is not None:
                self.status.stop()
                self.status = None
        except Exception:  # noqa: BLE001 - the status line is cosmetic
            self.live_status = False

    def _stop_status(self) -> None:
        if self.status is not None:
            try:
                self.status.stop()
            except Exception:  # noqa: BLE001
                pass
            self.status = None

    def _footnotes(self, reply: str, run_dir: str | None) -> None:
        from .audit.render import describe_evidence, evidence_status, find_refs, number_refs

        if not reply or not find_refs(reply) or not run_dir:
            return
        try:
            from .audit.claims import load_claims
            claims = load_claims(run_dir)
        except Exception:  # noqa: BLE001
            claims = []
        _, notes = number_refs(reply, claims)
        if not notes:
            return
        self.console.print("[bold]References[/]")
        for fn in notes:
            if fn.get("missing"):
                self.console.print(f"  [red]{_e('[' + str(fn['label']) + ']')}[/] {_e(fn['id'])}: "
                                   "no claim with this id was filed")
                continue
            flag = "" if fn.get("verified") else " [yellow](no verified evidence)[/]"
            ev = "; ".join(f"{_short(describe_evidence(e), 80)} ({evidence_status(e)})"
                           for e in fn.get("evidence") or [] if isinstance(e, Mapping))
            self.console.print(f"  {_e('[' + str(fn['label']) + ']')} {_e(fn['id'])}: "
                               f"{_e(_short(fn.get('text'), 160))}{flag}"
                               + (f" [dim]— {_e(ev)}[/]" if ev else ""), highlight=False)

    # -- dispatch
    def __call__(self, kind: str, d: dict[str, Any] | None = None, **kw: Any) -> None:
        d = dict(d or {}, **kw)
        try:
            self._handle(kind, d)
        except Exception:  # noqa: BLE001 - display problems never break a run
            pass

    def _handle(self, kind: str, d: dict[str, Any]) -> None:
        c = self.console
        agent = d.get("agent")
        if kind == "turn_start":
            self.stripper = self._stripper_cls()
            self.msg_chars = 0
            self.seen_tools.clear()
            return
        if kind == "turn_end":
            self._end_thinking()
            self._flush_text()
            self._stop_status()
            self.running.clear()
            if self.msg_chars:
                c.out("", highlight=False)
                self.msg_chars = 0
            if self.quiet:
                c.print(f"[dim]turn {d.get('turn')}: {d.get('status')}[/]")
                for u in d.get("unstreamed") or []:
                    if "warning" in str(u).lower() or "interrupted" in str(u).lower():
                        c.print(f"[yellow]{_e(u)}[/]", highlight=False)
                return
            for u in d.get("unstreamed") or []:
                style = "yellow" if ("warning" in str(u).lower() or "interrupted" in str(u).lower()
                                     or str(u).startswith("[Turn incomplete")) else None
                from .audit.render import strip_refs
                c.print(f"\n{strip_refs(str(u))}", style=style, highlight=False, markup=False)
            self._footnotes(str(d.get("reply") or ""), d.get("run_dir"))
            return
        if self.quiet and kind not in ("warning",):
            return
        if kind == "text":
            if agent != "cso" or int(d.get("depth") or 0) > 0:
                return
            self._end_thinking()
            self._stop_status()
            out = self.stripper.feed(d.get("text") or "")
            self._out(out)
            self.msg_chars += len(out)
            if out:
                self.blank = False
        elif kind == "message_end":
            if agent == "cso":
                self._end_thinking()
                self._flush_text()
                if self.msg_chars:
                    c.out("\n", highlight=False)  # ends the line and leaves a blank one
                    self.blank = True
                    self.msg_chars = 0
        elif kind == "thinking":
            if not self.show_reasoning or (agent != "cso" and not self.verbose):
                return
            if not self.thinking_open:
                c.out(f"\n[{agent} reasoning] ", style="dim italic", end="", highlight=False)
                self.thinking_open = True
            c.out(d.get("text") or "", style="dim italic", end="", highlight=False)
            if not d.get("streamed"):
                self._end_thinking()
        elif kind == "delegation":
            self._end_thinking()
            self._flush_text()
            lead, self.blank = ("" if self.blank else "\n"), False
            c.print(f"{lead}[bold cyan]→ delegating to {_e(agent)}[/]: {_e(d.get('description') or '')}",
                    highlight=False)
            self.running[str(d.get("invocation_id") or agent)] = {"agent": agent, "tool": None, "t0": time.time()}
            self._update_status()
        elif kind == "delegation_end":
            self.running.pop(str(d.get("invocation_id") or agent), None)
            self._update_status()
            st = d.get("status") or "completed"
            cost = float(d.get("cost_usd") or 0)
            dur = d.get("duration_s")
            extra = f"${cost:.2f}" + (f", {dur}s" if dur is not None else "")
            if st == "completed":
                c.print(f"[cyan]✓ {_e(agent)} finished ({extra})[/]", highlight=False)
            else:
                c.print(f"[yellow]! {_e(agent)} stopped: {_e(st)} ({extra})[/]", highlight=False)
        elif kind in ("tool_start", "tool"):
            key = str(d.get("tool_use_id") or f"{kind}:{agent}:{d.get('tool')}:{d.get('ts')}")
            if d.get("tool_use_id"):
                if key in self.seen_tools:
                    return
                self.seen_tools.add(key)
            tool = d.get("tool")
            inv = str(d.get("invocation_id") or "")
            if inv in self.running:
                self.running[inv]["tool"] = tool
                self._update_status()
            if tool == "Task":
                return
            if self.verbose or agent == "cso":
                self._end_thinking()
                self._flush_text()
                preview = d.get("input_preview", d.get("input"))
                c.print(f"[dim]  {_e('[' + str(agent) + ']')} {_e(tool)}"
                        + (f": {_e(_short(preview))}" if self.verbose else "") + "[/]", highlight=False)
        elif kind == "tool_end":
            inv = str(d.get("invocation_id") or "")
            if inv in self.running:
                self.running[inv]["tool"] = None
                self._update_status()
            if self.verbose and d.get("is_error") and d.get("tool") != "Task":
                c.print(f"[red]  ✗ {_e('[' + str(agent) + ']')} {_e(d.get('tool'))} failed: "
                        f"{_e(_short(d.get('output_preview'), 160))}[/]", highlight=False)
        elif kind == "briefing":
            from rich.panel import Panel
            self._stop_status()
            c.print(Panel(str(d.get("text") or ""), title="Chief of Staff briefing", border_style="blue"))
        elif kind == "draft_superseded":
            self._flush_text()
            lead, self.blank = ("" if self.blank else "\n"), False
            why = {"plan": "plan requested", "review": "review requested"}.get(str(d.get("reason")), "")
            c.print(f"{lead}[dim]\\[draft, superseded{': ' + _e(why) if why else ''}][/]", highlight=False)
        elif kind == "review_enforced":
            self._flush_text()
            agents = ", ".join(d.get("agents") or [])
            lead, self.blank = ("" if self.blank else "\n"), False
            c.print(f"{lead}[magenta]\\[harness] review required: dispatching scientific-reviewer[/]"
                    + (f" [dim](unreviewed: {_e(agents)})[/]" if agents else ""), highlight=False)
        elif kind == "retry":
            c.print(f"[dim yellow]\\[harness] provider retry {_e(d.get('attempt'))} in {_e(d.get('delay_s'))}s: "
                    f"{_e(_short(d.get('error'), 120))}[/]", highlight=False)
        elif kind == "compaction":
            c.print(f"[dim]\\[harness] context compacted for {_e(agent)} ({_e(d.get('strategy'))}): "
                    f"{_e(d.get('tokens_before'))} -> {_e(d.get('tokens_after'))} tokens[/]", highlight=False)
        elif kind == "warning":
            c.print(f"[yellow]! {_e(d.get('message'))}[/]", highlight=False)


def _printer(verbose: bool, *, show_reasoning: bool = False, quiet: bool = False, console=None):
    """(console, on_event) for the rich event renderer (also used by the scenario runner)."""
    from rich.console import Console

    console = console or Console(highlight=False, soft_wrap=True)
    return console, _Printer(console, verbose=verbose, show_reasoning=show_reasoning, quiet=quiet)


def _ndjson_printer(stream=None) -> Callable[..., None]:
    from .events import to_json_line

    lock = threading.Lock()

    def on_event(kind: str, d: dict[str, Any] | None = None, **kw: Any) -> None:
        out = stream or sys.stdout
        line = to_json_line(kind, dict(d or {}, **kw))
        with lock:
            out.write(line + "\n")
            out.flush()
    return on_event


# ---------------------------------------------------------------- session helpers

async def _start_session(args, config, on_event, *, interface: str, console=None, banner: bool = True):
    from .orchestrator import open_session

    resume = None
    if getattr(args, "resume", None):
        resume = _resolve_run(args.resume, config)
        if not _profiles(args):
            # Continue on the run's pinned profiles/models/policy, not today's defaults.
            from .replay import load_pinned
            pinned = load_pinned(resume)
            if pinned:
                config = build_config(args, pinned=pinned)
    session = await open_session(config, on_event=on_event, start_mcp=not getattr(args, "no_mcp", False),
                                 interface=interface, profiles=tuple(config.get("profiles") or ()), resume=resume)
    if banner and console is not None:
        verb = "resumed" if resume else "run"
        console.print(f"[bold]The Virtual Biotech[/] — {verb} [green]{session.run.run_id}[/] "
                      f"({config['provider']['name']}: {config['models']['orchestrator']['model']})")
        console.print(f"Run directory: {session.run.dir}")
        if resume:
            console.print(f"[dim]Resumed after turn {session.turn} ({len(session.history)} messages restored)[/]")
        if session.rt.mcp and session.rt.mcp.failures:
            console.print(f"[yellow]Unavailable MCP servers: {', '.join(sorted(session.rt.mcp.failures))}[/]")
    return session


def _turn_record(session, n: int | None = None) -> dict[str, Any] | None:
    n = session.turn if n is None else n
    for t in reversed(getattr(session.run, "turns", None) or []):
        if t.get("turn") == n:
            return t
    return None


def _report_paths(run_dir: Path) -> list[Path]:
    return [p for p in (run_dir / "README.md", run_dir / "audit.html") if p.is_file()]


def _after_turn(console, session) -> None:
    rec = _turn_record(session)
    if rec is None:
        return
    bits = [f"turn {rec.get('turn')}: {rec.get('status')}"]
    try:
        bits.append(f"cost ${float(rec.get('cost_usd') or 0):.2f} (total ${session.run.cost.total_usd:.2f})")
    except (TypeError, ValueError):
        pass
    console.print(f"\n[dim]({'; '.join(bits)})[/]")
    agents = rec.get("agents") or []
    if agents:
        console.print(f"[dim]  agents: {_e(', '.join(agents))}[/]", highlight=False)
    data = rec.get("data_source_failures") or []
    if data:
        names = ", ".join(dict.fromkeys(str(f.get("tool")) for f in data if isinstance(f, Mapping)))
        console.print(f"[yellow]  unresolved data-source failures: {_e(names)}[/]", highlight=False)
    filed = rec.get("claims_filed") or []
    unresolved = rec.get("claims_unresolved") or []
    if filed or unresolved or rec.get("dangling_claim_refs"):
        line = f"  claims filed: {len(filed)}" + (f" ({', '.join(filed[:12])})" if filed else "")
        if unresolved:
            line += f"; with unresolved evidence: {', '.join(unresolved)}"
        if rec.get("dangling_claim_refs"):
            line += f"; dangling refs: {', '.join(rec['dangling_claim_refs'])}"
        console.print(f"[dim]{_e(line)}[/]", highlight=False)
    for p in _report_paths(session.run.dir):
        console.print(f"[dim]  {p.name}: {_e(p)}[/]", highlight=False)


def _summary(console, session) -> None:
    turns = getattr(session.run, "turns", None) or []
    if turns:
        from rich.table import Table
        table = Table(title="Turns", show_lines=False)
        for col in ("turn", "status", "agents", "claims", "cost"):
            table.add_column(col)
        for t in turns:
            try:
                cost = f"${float(t.get('cost_usd') or 0):.2f}"
            except (TypeError, ValueError):
                cost = "?"
            table.add_row(str(t.get("turn")), str(t.get("status")), ", ".join(t.get("agents") or []) or "-",
                          str(len(t.get("claims_filed") or [])), cost)
        console.print(table)
    c = session.run.cost.report()
    console.print(f"\n[bold]Session cost:[/] ${c['total_usd']:.2f}")
    for a, d in c["agents"].items():
        console.print(f"  {a:30s} ${d['usd']:.2f}  ({d['model_calls']} calls)")


def _print_claims(console, run_dir: Path) -> None:
    from .audit.claims import load_claims
    from .audit.render import evidence_status

    claims = load_claims(run_dir)
    if not claims:
        console.print("No claims filed yet.")
        return
    for c in claims:
        ev = [e for e in c.get("evidence") or [] if isinstance(e, Mapping)]
        nv = sum(1 for e in ev if evidence_status(e) == "verified")
        console.print(f"  [bold]{_e(c.get('id'))}[/] ({nv}/{len(ev)} verified, turn {_e(c.get('turn'))}): "
                      f"{_e(_short(c.get('text'), 140))}", highlight=False)


def _print_evidence(console, run_dir: Path, cid: str) -> None:
    from .audit.claims import load_claims
    from .audit.render import EVIDENCE_STATUS_LABELS, describe_evidence, evidence_status

    for c in load_claims(run_dir):
        if str(c.get("id")) == cid:
            console.print(f"[bold]{_e(cid)}[/]: {_e(c.get('text'))}", highlight=False)
            meta = ", ".join(str(x) for x in (c.get("confidence"), c.get("agent")) if x)
            if meta:
                console.print(f"  [dim]{_e(meta)}[/]")
            for e in c.get("evidence") or []:
                if isinstance(e, Mapping):
                    st = evidence_status(e)
                    console.print(f"  - [{EVIDENCE_STATUS_LABELS[st]}] {describe_evidence(e)}", highlight=False,
                                  markup=False)
            return
    console.print(f"No claim with id {cid!r}. Use /claims to list them.")


# ---------------------------------------------------------------- input and Ctrl+C

class _ExitRequested(Exception):
    pass


class _Control:
    """Ctrl+C handling and non-blocking line input for the REPL and headless runs."""

    def __init__(self, session, console, *, loop: asyncio.AbstractEventLoop | None = None):
        self.session = session
        self.console = console
        self.loop = loop
        self.last_sigint = float("-inf")
        self.exit_requested = False
        self.turn_cancelled = False
        self.read_fut: asyncio.Future | None = None
        self.installed = False
        self._original = None
        self._pt_session = None

    # -- signals
    def install(self) -> bool:
        try:
            self._original = signal.getsignal(signal.SIGINT)
            self.loop = self.loop or asyncio.get_running_loop()
            self.loop.add_signal_handler(signal.SIGINT, self.on_sigint)
            self.installed = True
        except (NotImplementedError, RuntimeError, ValueError, AttributeError):
            self.installed = False
        return self.installed

    def restore(self) -> None:
        if not self.installed:
            return
        try:
            self.loop.remove_signal_handler(signal.SIGINT)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._original is not None:
                signal.signal(signal.SIGINT, self._original)
        except Exception:  # noqa: BLE001
            pass
        self.installed = False

    def on_sigint(self) -> None:
        now = time.monotonic()
        if now - self.last_sigint < FORCE_EXIT_WINDOW_S:
            try:
                self.console.print("\n[red]Forced exit.[/]")
            finally:
                os._exit(130)
        self.last_sigint = now
        if getattr(self.session, "busy", False):
            self.turn_cancelled = True
            if self.session.cancel():
                self.console.print("\n[yellow]^C interrupting the turn (press Ctrl+C again within 2 s to "
                                   "force exit)[/]")
            return
        self.exit_requested = True
        fut = self.read_fut
        if fut is not None and not fut.done():
            fut.cancel()

    # -- input
    def _use_prompt_toolkit(self) -> bool:
        try:
            if not sys.stdin.isatty():
                return False
            import prompt_toolkit  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    async def read(self, prompt: str) -> str:
        """One line of input without blocking the event loop.

        Raises EOFError at end of input and _ExitRequested after Ctrl+C at the prompt."""
        if self.exit_requested:
            raise _ExitRequested()
        if self._use_prompt_toolkit():
            from prompt_toolkit import PromptSession
            if self._pt_session is None:
                self._pt_session = PromptSession()
            try:
                return await self._pt_session.prompt_async(prompt)
            except KeyboardInterrupt:
                now = time.monotonic()
                if now - self.last_sigint < FORCE_EXIT_WINDOW_S:
                    os._exit(130)
                self.last_sigint = now
                self.exit_requested = True
                raise _ExitRequested() from None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def settle(value: Any, exc: BaseException | None) -> None:
            if fut.done():
                return
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(value)

        def worker() -> None:
            try:
                value = input(prompt)
            except BaseException as exc:  # noqa: BLE001 - EOFError / KeyboardInterrupt go to the loop
                try:
                    loop.call_soon_threadsafe(settle, None, exc if isinstance(exc, Exception) else EOFError())
                except RuntimeError:
                    pass
                return
            try:
                loop.call_soon_threadsafe(settle, value, None)
            except RuntimeError:
                pass

        threading.Thread(target=worker, name="vbt-input", daemon=True).start()
        self.read_fut = fut
        try:
            return await fut
        except asyncio.CancelledError:
            if self.exit_requested:
                raise _ExitRequested() from None
            raise
        finally:
            self.read_fut = None


async def _read_multiline(ctl: _Control, first: str) -> str | None:
    """Collect lines after an opening triple quote until the closing one.
    EOF submits what was collected; Ctrl+C abandons the block (None)."""
    buf = [first[3:]] if first[3:].strip() else []
    if first.rstrip().endswith('"""') and len(first.strip()) > 3:
        return first.strip()[3:-3].strip()
    while True:
        try:
            nxt = await ctl.read("... ")
        except EOFError:
            break
        except (KeyboardInterrupt, _ExitRequested):
            return None
        if nxt.strip() == '"""':
            break
        if nxt.rstrip().endswith('"""'):
            buf.append(nxt.rstrip()[:-3])
            break
        buf.append(nxt)
    return "\n".join(buf).strip()


async def _ask_turn(console, session, ctl: _Control, text: str) -> str:
    """Run one turn. Returns 'completed' | 'interrupted' | 'failed' | 'busy'."""
    from .orchestrator import SessionBusy

    ctl.turn_cancelled = False
    try:
        await session.ask(text)
        return "completed"
    except asyncio.CancelledError:
        if ctl.turn_cancelled or getattr(session, "cancel_requested", False):
            ctl.last_sigint = float("-inf")  # the press was handled: the next one is a fresh first press
            console.print("\n[yellow]Turn interrupted (recorded as 'interrupted'); evidence coverage is "
                          "incomplete.[/]")
            return "interrupted"
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Turn interrupted.[/]")
        return "interrupted"
    except SessionBusy as exc:
        console.print(f"[yellow]{_e(exc)}[/]")
        return "busy"
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive
        console.print(f"\n[red]Turn failed: {type(exc).__name__}: {_e(exc)}[/]")
        return "failed"


# ---------------------------------------------------------------- commands

def _console_for(args):
    return _printer(args.verbose, show_reasoning=getattr(args, "show_reasoning", False))


async def cmd_chat(args, config: dict[str, Any] | None = None) -> int:
    config = config if config is not None else build_config(args)
    console, on_event = _console_for(args)
    session = await _start_session(args, config, on_event, interface="chat", console=console)
    ctl = _Control(session, console, loop=asyncio.get_running_loop())
    ctl.install()
    console.print('Type your question. /help for commands, /done to finish. Multi-line: start and end with """.\n')
    try:
        while True:
            try:
                line = (await ctl.read("\nYou: ")).strip()
            except (EOFError, _ExitRequested):
                break
            except KeyboardInterrupt:
                break
            if not line:
                continue
            if line.startswith('"""'):
                block = await _read_multiline(ctl, line)
                if block is None:
                    if ctl.exit_requested:
                        break
                    console.print("[dim](input cancelled)[/]")
                    continue
                line = block
                if not line:
                    continue
            if line in ("/done", "quit", "exit", "/exit", "/quit"):
                break
            if line in ("/help", "help", "/?"):
                console.print(HELP_TEXT, markup=False, highlight=False)
                continue
            if line == "/summary":
                _summary(console, session)
                continue
            if line == "/claims":
                _print_claims(console, session.run.dir)
                continue
            if line.startswith("/evidence"):
                parts = line.split()
                if len(parts) < 2:
                    console.print("usage: /evidence <claim id>")
                else:
                    _print_evidence(console, session.run.dir, parts[1])
                continue
            console.print("\n[bold magenta]CSO:[/] ", end="")
            await _ask_turn(console, session, ctl, line)
            _after_turn(console, session)
            if ctl.exit_requested:
                break
    finally:
        ctl.restore()
        try:
            _summary(console, session)
        finally:
            await session.close()
        for p in _report_paths(session.run.dir):
            console.print(f"{p.name}: {p}")
        console.print(f"Records: {session.run.dir}")
    return 0


def _read_turn_file(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _run_summary(session) -> dict[str, Any]:
    run = session.run
    man = getattr(run, "manifest", None) or {}
    kinds: dict[str, int] = {}
    for e in (man.get("artifacts") or {}).values():
        if isinstance(e, Mapping):
            k = str(e.get("kind") or "other")
            kinds[k] = kinds.get(k, 0) + 1
    specialists = [a for a in man.get("agents") or [] if a not in ("cso",) and not str(a).startswith("_")]
    try:
        from .audit.claims import load_claims
        claims = load_claims(run.dir)
    except Exception:  # noqa: BLE001
        claims = []
    return {"run_id": run.run_id, "run_dir": str(run.dir), "status": man.get("status"),
            "turns": [{"turn": t.get("turn"), "status": t.get("status")} for t in run.turns],
            "artifacts_by_kind": dict(sorted(kinds.items())), "specialists": specialists,
            "claims": len(claims), "cost_usd": round(run.cost.total_usd, 6),
            "reports": [str(p) for p in _report_paths(run.dir)]}


async def cmd_run(args, config: dict[str, Any] | None = None) -> int:
    config = config if config is not None else build_config(args)
    turns = list(args.queries)
    if args.file:
        turns += _read_turn_file(args.file)
    if not turns:
        print("no queries given", file=sys.stderr)
        return 2
    ndjson = getattr(args, "events", None) == "ndjson"
    quiet = bool(getattr(args, "quiet", False))
    if ndjson:
        from rich.console import Console
        console = Console(file=sys.stderr, highlight=False, soft_wrap=True)
        on_event: Any = _ndjson_printer()
    else:
        console, on_event = _printer(args.verbose, show_reasoning=getattr(args, "show_reasoning", False),
                                     quiet=quiet)
    session = await _start_session(args, config, on_event, interface="run", console=console,
                                   banner=not (ndjson or quiet))
    ctl = _Control(session, console, loop=asyncio.get_running_loop())
    ctl.install()
    failed: list[int] = []
    interrupted = False
    try:
        for i, q in enumerate(turns, 1):
            if not (ndjson or quiet):
                console.print(f"\n[bold]You:[/] {_e(q)}\n[bold magenta]CSO:[/] ", end="", highlight=False)
            ctl.turn_cancelled = False
            try:
                await session.ask(q)
            except asyncio.CancelledError:
                if ctl.turn_cancelled or getattr(session, "cancel_requested", False):
                    ctl.last_sigint = float("-inf")
                    interrupted = True
                    console.print(f"[yellow]\\[INTERRUPTED] Turn {i} was interrupted; remaining turns skipped.[/]")
                    break
                raise
            except KeyboardInterrupt:
                interrupted = True
                console.print(f"[yellow]\\[INTERRUPTED] Turn {i} was interrupted.[/]")
                break
            except Exception as exc:  # noqa: BLE001 - record and continue with the next turn
                failed.append(i)
                console.print(f"[red]\\[ERROR] Turn {i} did not complete: {type(exc).__name__}: {_e(exc)}[/]",
                              highlight=False)
                console.print(f"Run directory: {session.run.dir}", highlight=False)
                continue
            if not (ndjson or quiet):
                _after_turn(console, session)
    finally:
        ctl.restore()
        await session.close()
    summary = _run_summary(session)
    try:
        from .verify import verify_run
        summary["verify"] = str(verify_run(session.run.dir).get("status"))
    except Exception as exc:  # noqa: BLE001
        summary["verify"] = f"error: {type(exc).__name__}: {exc}"
    summary["failed_turns"] = failed
    summary["interrupted"] = interrupted
    not_completed = [t for t in summary["turns"] if t.get("status") != "completed"]
    strict = not getattr(args, "no_strict", False)
    bad = bool(failed or interrupted or not_completed or summary["verify"] != "COMPLETE")
    summary["exit_code"] = 1 if (strict and bad) else 0
    if ndjson:
        on_event("run_summary", summary)
    else:
        _print_run_summary(console, summary)
    return summary["exit_code"]


def _print_run_summary(console, s: Mapping[str, Any]) -> None:
    console.print(f"\n[bold]Run {s['run_id']}[/]: status {s.get('status')}, verify {s.get('verify')}, "
                  f"cost ${float(s.get('cost_usd') or 0):.2f}", highlight=False)
    for t in s.get("turns") or []:
        style = "" if t.get("status") == "completed" else "yellow"
        console.print(f"  turn {t.get('turn')}: {t.get('status')}", style=style or None, highlight=False)
    arts = s.get("artifacts_by_kind") or {}
    console.print("  artifacts: " + (", ".join(f"{k} {v}" for k, v in arts.items()) or "none"), highlight=False)
    console.print("  specialists: " + (", ".join(s.get("specialists") or []) or "none"), highlight=False)
    console.print(f"  claims filed: {s.get('claims', 0)}", highlight=False)
    for p in s.get("reports") or []:
        console.print(f"  report: {p}", highlight=False)
    console.print(f"Records: {s.get('run_dir')}", highlight=False)


async def cmd_replay(args, config: dict[str, Any]) -> int:
    from .replay import DIFF_FILE, format_diff, replay_run

    run_dir = _resolve_run(args.run, config)
    model = getattr(args, "replay_model", None) or getattr(args, "model", None)
    on_event = None
    if not args.quiet:
        _, on_event = _printer(args.verbose, show_reasoning=getattr(args, "show_reasoning", False))
    # The replay runs on the recorded run's pinned settings; the global flags
    # (--profile, --no-web, --skip-preflight, --allow-missing-data) apply on top.
    overrides = flag_overrides(args)
    overrides.pop("paths", None)  # --runs-dir is passed as runs_dir below
    new_dir, diff = await replay_run(run_dir, model=model, quiet=args.quiet, on_event=on_event,
                                     runs_dir=getattr(args, "runs_dir", None) or None,
                                     start_mcp=not args.no_mcp,
                                     profiles=list(getattr(args, "profile", None) or []) or None,
                                     overrides=overrides or None)
    print(format_diff(diff))
    print(f"\nReplay run: {new_dir}\nDiff: {new_dir / DIFF_FILE}")
    return 1 if diff.get("replay_errors") else 0


async def cmd_tools(args, config: dict[str, Any] | None = None) -> int:
    from .runtime import Runtime
    from .session import Run
    import tempfile

    config = config if config is not None else build_config(args)
    rt = Runtime(config, Run(Path(tempfile.mkdtemp())))
    if not args.no_mcp:
        await rt.start_mcp()
    try:
        for name, agent in [("cso", rt.cso), *rt.agents.items()]:
            tools = [t.name for t in rt.tools_for(agent)]
            missing = rt.registry.missing(agent.tools)
            off = [m for m in missing if m in ("BulkDispatch", "BulkStatus")]
            no_mcp = [m for m in missing if m.startswith("mcp__") and args.no_mcp]
            missing = [m for m in missing if m not in off and m not in no_mcp]
            print(f"\n{name} [{agent.tier}: {agent.settings(config).model}] — {len(tools)} tools")
            print("  " + ", ".join(tools))
            if missing:
                print(f"  (unresolved: {', '.join(missing)})")
            if off:
                print(f"  (off: {', '.join(off)}; set bulk.dispatch_enabled: true)")
            if no_mcp:
                print(f"  ({len(no_mcp)} MCP tool pattern(s) not resolved: --no-mcp)")
        if rt.mcp and rt.mcp.failures:
            print("\nMCP failures:", json.dumps(rt.mcp.failures, indent=1))
    finally:
        await rt.aclose()
    return 0


# ---------------------------------------------------------------- entry point

def _add_session_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--resume", metavar="RUN", help="continue a recorded session (run id, prefix, path or 'latest')")
    p.add_argument("--show-reasoning", action="store_true",
                   help="print the CSO's reasoning (dim); with -v also the specialists'")
    p.add_argument("--no-clarify", action="store_true",
                   help="skip strategic orientation (orchestration.strategic_orientation=false)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vbt", description="The Virtual Biotech — provider-agnostic harness")
    p.add_argument("--profile", action="append", default=[], help="config profile(s), e.g. paper, no-web, mock")
    p.add_argument("--model", help="override the model for CSO, scientists and bulk agents "
                                   "(a model_aliases label or a model id)")
    p.add_argument("--no-web", action="store_true", help="disable WebSearch/WebFetch (no-leakage setting)")
    p.add_argument("--no-mcp", action="store_true", help="do not start MCP data servers")
    p.add_argument("--runs-dir")
    p.add_argument("--skip-preflight", action="store_true",
                   help="do not check credentials/reference data/MCP commands before the session and each turn")
    p.add_argument("--allow-missing-data", action="store_true",
                   help="start even when reference data is missing; the run is marked degraded and the "
                        "agents are told which servers lack data")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat", help="interactive CSO session")
    _add_session_flags(c)
    r = sub.add_parser("run", help="headless: one argument per conversation turn")
    r.add_argument("queries", nargs="*")
    r.add_argument("-f", "--file", help="file with one turn per line ('#' lines are comments)")
    r.add_argument("-q", "--quiet", action="store_true", help="no streaming; print the final summary only")
    r.add_argument("--no-strict", action="store_true",
                   help="exit 0 even when a turn did not complete or verify is not COMPLETE")
    r.add_argument("--events", choices=["ndjson"], help="write the event stream as JSON lines to stdout")
    _add_session_flags(r)
    rp = sub.add_parser("replay", help="re-run a recorded session's turns and compare (writes replay_diff.json)")
    rp.add_argument("run", help="run id, prefix, path or 'latest'")
    rp.add_argument("-q", "--quiet", action="store_true")
    rp.add_argument("--model", dest="replay_model", help="replay with a different model (alias or id)")
    sub.add_parser("tools", help="list agents and their resolved tools")

    from .audit.cli import add_audit_parsers
    from .bulk import add_bulk_parser
    from .case_studies import add_case_parsers
    from .preflight import add_doctor_parser
    from .web import add_web_parser
    add_audit_parsers(sub)      # verify, list, index, export, audit, show
    add_doctor_parser(sub)      # doctor [--smoke] [--analysis]
    add_web_parser(sub)         # web [--host] [--port] [--no-auth]
    add_bulk_parser(sub)
    add_case_parsers(sub)       # case1, scenario, data (P9)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        config = build_config(args)
        if args.cmd == "replay" and getattr(args, "replay_model", None):
            resolve_model(config, args.replay_model)
    except ModelResolutionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.cmd in ("chat", "run", "tools", "replay"):
        fn = {"chat": cmd_chat, "run": cmd_run, "tools": cmd_tools, "replay": cmd_replay}[args.cmd]
        from .preflight import DataReadinessError
        try:
            return asyncio.run(fn(args, config))
        except RunResolutionError as exc:  # --resume / replay run argument
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except DataReadinessError as exc:  # session preflight: nothing was sent to the model
            print(f"error: {exc}\nRun `vbt doctor` for details; --allow-missing-data starts a degraded run, "
                  "--skip-preflight skips the check.", file=sys.stderr)
            return 2
    return args.handler(args, config)


if __name__ == "__main__":
    sys.exit(main())

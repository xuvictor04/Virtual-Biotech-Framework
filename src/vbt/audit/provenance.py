"""Provenance index: reconstruct "who did what" from ``logs/trace.jsonl``.

The index is derived from *observed execution*, not from any agent's
self-report. For every tool call it records the agent and agent run that made
it, its timing, whether it failed, the files its input names (Write/Edit/
NotebookEdit targets, Bash redirections, ``python x.py`` script links) and the
files an MCP tool returned. ``attribute_path`` then explains where a file came
from, trying the evidence in decreasing strength:

1. ``tool_input``  — a tool call named the file in its input (last writer wins);
2. ``mcp_return``  — an MCP tool returned the path (MCP servers write their bulk
   results under ``work/_mcp`` and return the path);
3. ``script``      — a unique writer statement in a unique script with a unique
   creator (``created_by = 'script.py:LINE'``; port of upstream
   ``attribute_script_outputs``);
4. ``window``      — the file's mtime falls inside exactly one specialist's
   agent_start/agent_end window (``ambiguous`` when windows overlap).

Both trace formats are understood: the original harness events (``agent`` on
each event, no run ids) and the extended ones (``agent_run_id``,
``parent_run_id``, ``input_ref`` spills, ``files_returned``, ``delegation_end``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .storage import (
    is_ignored_rel,
    read_json,
    read_jsonl,
    rel_parts,
    to_rel,
    work_owner,
)

#: Agents whose activity alone does not make a turn "research".
SUPPORT_AGENTS = frozenset({"cso", "_cso", "chief-of-staff", "scientific-reviewer", "unknown"})

#: Agents whose tools resolve relative paths against the run root.
ROOT_WORKSPACE_AGENTS = frozenset({"cso", "_cso", "scientific-reviewer"})

#: Relative paths starting with these components are run-relative.
RUN_PREFIXES = frozenset({"work", "inputs", "evidence", "report", "logs", ".claude", "memory"})

#: Tools after which the run directory is re-snapshotted.
CAPTURE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash", "QueryToolOutput"})

_PATH_INPUT_KEYS = ("file_path", "notebook_path", "path", "filename")
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
_SCRIPT_SUFFIXES = (".py", ".r", ".R")
_INTERPRETER_RE = re.compile(r"^(?:.*/)?(python(?:\d+(?:\.\d+)?)?|Rscript)$")

#: A filename literal inside source code, e.g. ``to_csv(f"{ws}/il33_expr.csv")``.
_OUTPUT_LITERAL_RE = re.compile(
    r"['\"][^'\"]*?([\w./{}\-]+\.(?:csv|tsv|parquet|png|pdf|svg|jpg|jpeg|"
    r"h5ad|h5|xlsx|json|jsonl|md|txt|npz|pkl|rds|html))['\"]")
_WRITER_RE = re.compile(
    r"\.(?:to_csv|to_parquet|to_excel|to_json|to_html|savefig|write_text|write_bytes|write_h5ad|"
    r"write_csv|write_parquet|write)\s*\(|\b(?:saveRDS|write\.csv|write\.table|ggsave|fwrite|"
    r"write_tsv|write_csv|png|pdf)\s*\(|\bopen\s*\([^)]*,\s*['\"](?:w|a|x)")
_REDIRECT_TARGET_RE = re.compile(r"^[\w./~{}$\-]+\.[A-Za-z][A-Za-z0-9]{0,7}$")
_BASH_WRITE_FALLBACK_RE = re.compile(
    r"(?:>>?|(?:-o|--out|--output|--outfile|--output-file)\s+)\s*([\w./\-]+\.[A-Za-z][A-Za-z0-9]{0,7})\b")


def is_data_tool(name: str | None) -> bool:
    """mcp__* (except the harness's own provenance tools), WebSearch and WebFetch."""
    name = name or ""
    return (name.startswith("mcp__") and not name.startswith("mcp__provenance__")) or name in ("WebSearch", "WebFetch")


def is_capture_tool(name: str | None) -> bool:
    name = name or ""
    return name in CAPTURE_TOOLS or name.startswith("mcp__")


def data_call_fields(ev: Mapping[str, Any]) -> dict[str, Any]:
    """The data-layer outcome recorded on a ``tool_end``/``tool_error`` event (§15.2):
    ``result_status``, ``error_kind``, the provenance id ``prov``, ``coverage`` and the
    provenance summary ``data_provenance``. Only the fields the event carries are returned.

    The live call index (``session.Run``) and this index both use it, so record_claims and
    ``vbt verify`` decide from the same fields."""
    out: dict[str, Any] = {}
    for k in ("result_status", "error_kind"):
        if ev.get(k):
            out[k] = str(ev[k])
    dp = ev.get("data_provenance")
    if isinstance(dp, Mapping):
        out["data_provenance"] = dict(dp)
        if dp.get("prov"):
            out["prov"] = dp["prov"]
        if dp.get("coverage"):
            out["coverage"] = dp["coverage"]
        if "result_status" not in out and dp.get("status"):
            out["result_status"] = str(dp["status"])
    return out


def read_trace(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Read a trace.jsonl, skipping malformed/partial lines. Returns (events, n_bad)."""
    return read_jsonl(path)


def _num(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _event_t(ev: Mapping[str, Any]) -> float | None:
    t = _num(ev.get("t"))
    if t is not None:
        return t
    ts = ev.get("ts")
    if ts:
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(ts)).timestamp()
        except (TypeError, ValueError):
            return None
    return None


def coerce_input(raw: Any) -> dict[str, Any]:
    """Tool input as a dict. Accepts dicts, JSON strings, repr strings and clipped JSON."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    s = raw.strip()
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except ValueError:
        pass
    try:
        import ast
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
    except (ValueError, SyntaxError):
        pass
    # A clipped JSON string ('{"file_path": "...", "content": "...(clipped)'): recover the short keys.
    out: dict[str, Any] = {}
    for key in (*_PATH_INPUT_KEYS, "command"):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(key), s)
        if m:
            try:
                out[key] = json.loads('"' + m.group(1) + '"')
            except ValueError:
                out[key] = m.group(1)
    return out


# ----------------------------------------------------------------- bash parsing

def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies (``<<EOF ... EOF``) so their text is not parsed as shell."""
    lines = command.splitlines()
    out: list[str] = []
    delim: str | None = None
    for line in lines:
        if delim is not None:
            if line.strip() == delim:
                delim = None
            continue
        out.append(line)
        m = re.search(r"<<-?\s*(['\"]?)([A-Za-z_][\w]*)\1", line)
        if m:
            delim = m.group(2)
    return "\n".join(out)


def _tokens(command: str) -> list[str] | None:
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:
        return None


def parse_bash(command: str) -> dict[str, Any]:
    """Files a shell command writes and scripts it runs.

    Returns ``{"writes": [(cwd_change, target)], "scripts": [(cwd_change, interpreter, script, args)]}``
    where ``cwd_change`` is the directory given to a preceding ``cd`` (or None).
    """
    writes: list[tuple[str | None, str]] = []
    scripts: list[tuple[str | None, str, str, list[str]]] = []
    body = _strip_heredocs(command or "")
    toks = _tokens(body)
    if toks is None:
        writes += [(None, m.group(1)) for m in _BASH_WRITE_FALLBACK_RE.finditer(body)]
        for m in re.finditer(r"(?:^|[;&|]\s*|\s)(python(?:\d+(?:\.\d+)?)?|Rscript)\s+([\w./\-]+\.(?:py|R|r))\b", body):
            scripts.append((None, m.group(1), m.group(2), []))
        return {"writes": writes, "scripts": scripts}
    seps = {"&&", "||", ";", "|", "&", "\n", "(", ")", ";;"}
    cwd: str | None = None
    i, n = 0, len(toks)
    cmd_start = True
    while i < n:
        tok = toks[i]
        if tok in seps:
            cmd_start = True
            i += 1
            continue
        if tok in (">", ">>", ">|", "&>", "&>>") and i + 1 < n:
            target = toks[i + 1]
            if _REDIRECT_TARGET_RE.match(target) and not target.startswith("/dev/"):
                writes.append((cwd, target))
            i += 2
            cmd_start = False
            continue
        if tok in ("-o", "--out", "--output", "--outfile", "--output-file") and i + 1 < n:
            target = toks[i + 1]
            if _REDIRECT_TARGET_RE.match(target):
                writes.append((cwd, target))
            i += 2
            continue
        if cmd_start and (re.match(r"^[A-Za-z_]\w*=", tok) or tok in ("time", "nohup", "env", "exec", "command")):
            i += 1
            continue
        if cmd_start and tok == "timeout" and i + 1 < n:
            i += 2
            continue
        if cmd_start and tok == "cd" and i + 1 < n and toks[i + 1] not in seps:
            cwd = toks[i + 1] if cwd is None or toks[i + 1].startswith("/") else f"{cwd}/{toks[i + 1]}"
            i += 2
            cmd_start = False
            continue
        if cmd_start and _INTERPRETER_RE.match(tok):
            interp = "Rscript" if tok.endswith("Rscript") else "python"
            j = i + 1
            while j < n and toks[j].startswith("-") and toks[j] not in ("-c", "-e", "-m", "-"):
                j += 1
            if j < n and toks[j] not in seps and toks[j] not in ("-c", "-e", "-m", "-") \
                    and toks[j].endswith(_SCRIPT_SUFFIXES):
                args: list[str] = []
                k = j + 1
                while k < n and toks[k] not in seps and toks[k] not in (">", ">>", ">&", "2", "&>", "<"):
                    args.append(toks[k])
                    k += 1
                scripts.append((cwd, interp, toks[j], args))
        cmd_start = False
        i += 1
    return {"writes": writes, "scripts": scripts}


def extract_written_paths(tool_name: str, tool_input: Any) -> list[str]:
    """Raw paths a tool call writes, from its input (as the agent wrote them)."""
    inp = coerce_input(tool_input)
    if not inp:
        return []
    name = (tool_name or "").split("__")[-1]
    paths: list[str] = []
    if name in _WRITE_TOOLS:
        for k in _PATH_INPUT_KEYS:
            v = inp.get(k)
            if isinstance(v, str) and v.strip():
                paths.append(v.strip())
                break
    elif name == "Bash":
        parsed = parse_bash(str(inp.get("command") or ""))
        for cwd, target in parsed["writes"]:
            paths.append(target if cwd is None or target.startswith("/") else f"{cwd}/{target}")
        for cwd, _interp, script, _args in parsed["scripts"]:
            paths.append(script if cwd is None or script.startswith("/") else f"{cwd}/{script}")
    return list(dict.fromkeys(paths))


def workspace_rel(agent: str | None) -> str:
    """Run-relative workspace of an agent ('' for run-root agents)."""
    if not agent or agent in ROOT_WORKSPACE_AGENTS:
        return ""
    return f"work/{agent}"


def normalize_path(raw: str, agent: str | None, run_dir: Path | None, base_rel: str | None = None) -> str:
    """Run-relative key for a path as an agent wrote it (absolute paths outside stay absolute)."""
    raw = os.path.expanduser(str(raw).strip())
    if not raw:
        return raw
    if os.path.isabs(raw):
        if run_dir is not None:
            rel = to_rel(raw, run_dir)
            if rel is not None:
                return rel
        return os.path.normpath(raw)
    first = PurePosixPath(raw).parts[0] if PurePosixPath(raw).parts else ""
    if first in RUN_PREFIXES and base_rel is None:
        joined = raw
    else:
        base = base_rel if base_rel is not None else workspace_rel(agent)
        joined = f"{base}/{raw}" if base else raw
    norm = os.path.normpath(joined).replace(os.sep, "/")
    return norm


def extract_returned_paths(tool_name: str, text: Any, run_dir: Path | None, *, must_exist: bool = False) -> list[str]:
    """Run-relative paths under the run directory named in an MCP tool's output."""
    if not is_data_tool(tool_name) or not str(tool_name).startswith("mcp__") or run_dir is None:
        return []
    s = text if isinstance(text, str) else json.dumps(text, default=str) if text is not None else ""
    if not s:
        return []
    out: list[str] = []
    roots = {str(run_dir)}
    try:
        roots.add(str(Path(run_dir).resolve()))
    except OSError:
        pass
    for root in roots:
        for m in re.finditer(re.escape(root) + r"/[^\s\"'<>|`,;)\]}]+", s):
            p = m.group(0).rstrip(".:")
            rel = to_rel(p, run_dir)
            if rel and not is_ignored_rel(rel) and (not must_exist or (Path(run_dir) / rel).is_file()):
                out.append(rel)
    for m in re.finditer(r"(?<![\w/])(work/_mcp/[^\s\"'<>|`,;)\]}]+)", s):
        rel = m.group(1).rstrip(".:")
        if not must_exist or (Path(run_dir) / rel).is_file():
            out.append(rel)
    return list(dict.fromkeys(out))


def _status_from_stop(stop: Any) -> str:
    if not stop or stop in ("end_turn", "completed"):
        return "completed"
    return str(stop)


# ----------------------------------------------------------------- the index

class Provenance:
    """Per-run provenance index built from trace events."""

    def __init__(self, events: list[dict[str, Any]], run_dir: str | Path | None = None, *,
                 malformed_lines: int = 0):
        self.events = list(events or [])
        self.run_dir = Path(run_dir) if run_dir else None
        self.malformed_lines = malformed_lines
        self.runs: dict[str, dict[str, Any]] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.delegations: dict[str, dict[str, Any]] = {}
        self.turns: dict[int, dict[str, Any]] = {}
        self.bash_events: list[dict[str, Any]] = []
        self._script_outputs: dict[str, dict[str, Any]] = {}
        self._written: dict[str, str] | None = None
        self._written_names: dict[str, list[tuple[str, str]]] | None = None
        self._returned: dict[str, str] | None = None
        self._texts: dict[str, str | None] = {}
        self._keys_ctx: tuple[int, int, dict[str, int], list[str], dict[str, int]] | None = None
        self._build()

    # ------------------------------------------------------------ construction

    def _build(self) -> None:
        open_runs: dict[str, list[str]] = defaultdict(list)
        pending_deleg: dict[str, deque] = defaultdict(deque)
        counter: dict[str, int] = defaultdict(int)
        current_turn: int | None = None
        for ev in self.events:
            if not isinstance(ev, Mapping):
                continue
            t = ev.get("type")
            et = _event_t(ev)
            if t == "turn_start":
                try:
                    current_turn = int(ev.get("turn"))
                except (TypeError, ValueError):
                    current_turn = None
                if current_turn is not None:
                    rec = self.turns.setdefault(current_turn, {"turn": current_turn})
                    rec.setdefault("start_t", et)
                    rec["start"] = ev.get("ts")
                    rec["prompt"] = ev.get("prompt")
                continue
            if t == "turn_end":
                try:
                    tn = int(ev.get("turn"))
                except (TypeError, ValueError):
                    tn = current_turn
                if tn is not None:
                    rec = self.turns.setdefault(tn, {"turn": tn})
                    rec["end_t"], rec["end"], rec["status"] = et, ev.get("ts"), ev.get("status")
                current_turn = None
                continue
            if t == "delegation":
                tuid = ev.get("tool_use_id") or ev.get("invocation_id")
                rec = {"agent": ev.get("agent"), "parent": ev.get("parent"), "tool_use_id": ev.get("tool_use_id"),
                       "invocation_id": ev.get("invocation_id"), "description": ev.get("description", ""),
                       "prompt": ev.get("prompt") or ev.get("prompt_preview"), "t": et, "turn": current_turn,
                       "status": None}
                if tuid:
                    self.delegations[str(tuid)] = rec
                pending_deleg[str(ev.get("agent"))].append(rec)
                continue
            if t == "delegation_end":
                tuid = ev.get("tool_use_id") or ev.get("invocation_id")
                if tuid and str(tuid) in self.delegations:
                    self.delegations[str(tuid)]["status"] = ev.get("status")
                    rid = ev.get("invocation_id") or tuid
                    if str(rid) in self.runs and ev.get("status"):
                        self.runs[str(rid)]["status"] = ev.get("status")
                continue
            if t == "agent_start":
                agent = str(ev.get("agent") or "unknown")
                depth = int(ev.get("depth") or 0)
                rid = ev.get("agent_run_id") or ev.get("invocation_id")
                deleg = None
                if rid and str(rid) in self.delegations:
                    deleg = self.delegations[str(rid)]
                    try:
                        pending_deleg[agent].remove(deleg)
                    except ValueError:
                        pass
                elif depth >= 1 and pending_deleg[agent]:
                    deleg = pending_deleg[agent].popleft()
                if not rid:
                    counter[agent] += 1
                    rid = (deleg or {}).get("tool_use_id") or f"{agent}#{counter[agent]}"
                rid = str(rid)
                while rid in self.runs:
                    rid += "'"
                run = {
                    "agent_run_id": rid, "agent": agent, "depth": depth,
                    "parent_run_id": ev.get("parent_run_id") or ev.get("parent_invocation_id"),
                    "tool_use_id": (deleg or {}).get("tool_use_id") or ev.get("tool_use_id"),
                    "description": (deleg or {}).get("description") or ev.get("description") or "",
                    "delegation_prompt": (deleg or {}).get("prompt") or ev.get("task"),
                    "start": ev.get("ts"), "start_t": et, "end": None, "end_t": None, "duration_s": None,
                    "status": "running", "cost_usd": None, "model_calls": None, "tool_calls": None,
                    "transcript_path": None, "turn": current_turn, "model": ev.get("model"),
                }
                if deleg is not None and deleg.get("status"):
                    run["status_hint"] = deleg["status"]
                self.runs[rid] = run
                open_runs[agent].append(rid)
                continue
            if t == "agent_end":
                agent = str(ev.get("agent") or "unknown")
                rid = ev.get("agent_run_id") or ev.get("invocation_id")
                if not rid or str(rid) not in self.runs:
                    depth = int(ev.get("depth") or 0)
                    cands = [r for r in open_runs[agent] if self.runs[r]["depth"] == depth] or open_runs[agent]
                    rid = cands[0] if cands else None
                if rid is None:
                    continue
                rid = str(rid)
                if rid in open_runs[agent]:
                    open_runs[agent].remove(rid)
                run = self.runs[rid]
                run.update(end=ev.get("ts"), end_t=et, duration_s=ev.get("duration_s"),
                           cost_usd=ev.get("cost_usd"), model_calls=ev.get("model_calls"),
                           tool_calls=ev.get("tool_calls"), transcript_path=ev.get("transcript_path"))
                status = ev.get("status") or _status_from_stop(ev.get("stop") or ev.get("stop_reason"))
                tuid = run.get("tool_use_id")
                if tuid and self.delegations.get(str(tuid), {}).get("status"):
                    status = self.delegations[str(tuid)]["status"]
                run["status"] = status
                continue
            if t in ("tool_start", "tool_end", "tool_error"):
                tuid = ev.get("tool_use_id")
                if not tuid:
                    continue
                tuid = str(tuid)
                agent = ev.get("agent")
                rec = self.calls.get(tuid)
                if rec is None:
                    rec = self.calls[tuid] = {
                        "tool_use_id": tuid, "tool": None, "agent": agent, "agent_run_id": None,
                        "parent_run_id": None, "started_at": None, "ended_at": None, "start_t": None,
                        "end_t": None, "duration_s": None, "is_error": False, "pending": True,
                        "input": None, "input_ref": None, "input_sha256": None, "output_path": None,
                        "files_written": [], "files_returned": [], "turn": current_turn, "_output": None,
                        "result_status": None, "error_kind": None, "prov": None, "coverage": None,
                        "data_provenance": None,
                    }
                rec["tool"] = rec["tool"] or ev.get("tool") or ev.get("tool_name")
                rec["agent"] = rec["agent"] or agent
                rid = ev.get("agent_run_id") or ev.get("invocation_id")
                if rid:
                    rec["agent_run_id"] = str(rid)
                    rec["parent_run_id"] = rec["parent_run_id"] or ev.get("parent_run_id")
                elif rec["agent_run_id"] is None and agent and len(open_runs.get(str(agent), [])) == 1:
                    rec["agent_run_id"] = open_runs[str(agent)][0]
                if t == "tool_start":
                    rec["started_at"], rec["start_t"] = ev.get("ts"), et
                    raw_in = ev.get("input", ev.get("tool_input"))
                    if raw_in is not None:
                        rec["input"] = raw_in
                    rec["input_ref"] = ev.get("input_ref") or rec["input_ref"]
                    rec["input_sha256"] = ev.get("input_sha256") or rec["input_sha256"]
                    if rec["input"] is None and ev.get("input_preview") is not None:
                        rec["input"] = ev.get("input_preview")
                    rec["turn"] = current_turn if current_turn is not None else rec["turn"]
                else:
                    rec["ended_at"], rec["end_t"] = ev.get("ts"), et
                    rec["duration_s"] = ev.get("duration_s", rec["duration_s"])
                    rec["is_error"] = bool(ev.get("is_error")) or t == "tool_error"
                    rec["pending"] = False
                    if ev.get("input") is not None and rec["input"] is None:
                        rec["input"] = ev.get("input")
                    rec["output_path"] = ev.get("output_path") or rec["output_path"]
                    rec["_output"] = ev.get("output") if ev.get("output") is not None else ev.get("tool_response")
                    if ev.get("files_returned"):
                        rec["files_returned"] = list(ev.get("files_returned") or [])
                    rec.update(data_call_fields(ev))
                continue
            if t == "bash":
                self.bash_events.append(dict(ev))
                tuid = ev.get("tool_use_id")
                if tuid and str(tuid) in self.calls:
                    self.calls[str(tuid)]["exit_code"] = ev.get("exit_code")
                continue
        for rec in self.calls.values():
            self._finish_call(rec)

    def _load_input(self, rec: dict[str, Any]) -> dict[str, Any]:
        inp = coerce_input(rec.get("input"))
        if (not inp or not any(k in inp for k in (*_PATH_INPUT_KEYS, "command"))) and rec.get("input_ref") \
                and self.run_dir is not None:
            ref = Path(str(rec["input_ref"]))
            if not ref.is_absolute():
                ref = self.run_dir / ref
            loaded = read_json(ref, None)
            if isinstance(loaded, Mapping):
                inp = dict(loaded.get("input", loaded)) if isinstance(loaded.get("input", loaded), Mapping) else inp
        return inp

    def _finish_call(self, rec: dict[str, Any]) -> None:
        tool = rec.get("tool") or ""
        agent = rec.get("agent")
        inp = self._load_input(rec) if (tool.split("__")[-1] in _WRITE_TOOLS or tool == "Bash") else {}
        raw_written = extract_written_paths(tool, inp)
        rec["files_written"] = list(dict.fromkeys(
            normalize_path(p, agent, self.run_dir) for p in raw_written if p))
        if tool == "Bash":
            rec["command"] = str(inp.get("command") or "") if inp else None
        returned = []
        for p in rec.get("files_returned") or []:
            rel = to_rel(p, self.run_dir) if (self.run_dir is not None and os.path.isabs(str(p))) else str(p)
            if rel:
                returned.append(rel)
        if not returned and not rec.get("is_error") and str(tool).startswith("mcp__"):
            returned = extract_returned_paths(tool, rec.get("_output"), self.run_dir)
            if not returned and rec.get("output_path") and self.run_dir is not None:
                spill = Path(str(rec["output_path"]))
                if not spill.is_absolute():
                    spill = self.run_dir / spill
                try:
                    with open(spill, "r", encoding="utf-8", errors="replace") as f:
                        returned = extract_returned_paths(tool, f.read(5_000_000), self.run_dir)
                except OSError:
                    pass
        rec["files_returned"] = list(dict.fromkeys(returned))
        # A bounded copy of the input (long file contents replaced by their length);
        # the full input stays in the trace or its input_ref spill.
        rec["input"] = _preview(inp if inp else rec.get("input"))
        run = self.runs.get(str(rec.get("agent_run_id"))) if rec.get("agent_run_id") else None
        rec["parent"] = rec.get("parent_run_id") or (run or {}).get("parent_run_id") or (run or {}).get("tool_use_id")
        rec.pop("_output", None)
        # Old traces and tools outside the data layer carry no status (§15.2).
        rec["result_status"] = rec.get("result_status") or "unknown"

    # ------------------------------------------------------------ queries

    def has_tool_call(self, tool_use_id: str) -> bool:
        return str(tool_use_id) in self.calls

    def agent_for(self, tool_use_id: str) -> str | None:
        rec = self.calls.get(str(tool_use_id))
        return rec.get("agent") if rec else None

    def specialist_types(self) -> list[str]:
        return sorted({r["agent"] for r in self.runs.values() if r["depth"] >= 1})

    def is_orientation_run(self, run: Mapping[str, Any]) -> bool:
        """A Chief of Staff run not started by a CSO ``Task`` call (the orientation brief)."""
        if run.get("agent") != "chief-of-staff" or int(run.get("depth") or 0) < 1:
            return False
        tuid = run.get("tool_use_id")
        return not (tuid and (self.calls.get(str(tuid)) or {}).get("tool") == "Task")

    def execution(self) -> list[dict[str, Any]]:
        """Observed dispatches (agent runs at depth >= 1), in start order."""
        rows = []
        for r in self.runs.values():
            if r["depth"] < 1:
                continue
            dur = r.get("duration_s")
            if dur is None and r.get("start_t") is not None and r.get("end_t") is not None:
                dur = round(r["end_t"] - r["start_t"], 2)
            rows.append({"agent": r["agent"], "agent_run_id": r["agent_run_id"], "tool_use_id": r.get("tool_use_id"),
                         "description": r.get("description") or "", "start": r.get("start"), "end": r.get("end"),
                         "start_t": r.get("start_t"), "end_t": r.get("end_t"), "duration_s": dur,
                         "status": r.get("status") if r.get("end") or r.get("status") != "running" else "unfinished",
                         "turn": r.get("turn"), "orientation": self.is_orientation_run(r)})
        rows.sort(key=lambda e: (e["start_t"] is None, e["start_t"] or 0))
        return rows

    def files_written(self) -> dict[str, str]:
        """{rel: tool_use_id} from tool inputs, last writer wins."""
        if self._written is None:
            out: dict[str, str] = {}
            for tuid, rec in sorted(self.calls.items(), key=lambda kv: kv[1].get("start_t") or 0):
                if rec.get("is_error") and rec.get("tool") in _WRITE_TOOLS:
                    continue
                for p in rec["files_written"]:
                    out[p] = tuid
            self._written = out
        return self._written

    def files_returned(self) -> dict[str, str]:
        """{rel: tool_use_id} for files MCP tools returned, last caller wins."""
        if self._returned is None:
            out: dict[str, str] = {}
            for tuid, rec in sorted(self.calls.items(), key=lambda kv: kv[1].get("start_t") or 0):
                for p in rec["files_returned"]:
                    out[p] = tuid
            self._returned = out
        return self._returned

    def written_by_name(self) -> dict[str, list[tuple[str, str]]]:
        """{basename: [(path, tool_use_id)]} over ``files_written``."""
        if self._written_names is None:
            idx: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for path, tuid in self.files_written().items():
                idx[PurePosixPath(path).name].append((path, tuid))
            self._written_names = dict(idx)
        return self._written_names

    def _text(self, rel: str) -> str | None:
        if rel not in self._texts:
            try:
                self._texts[rel] = (self.run_dir / rel).read_text(errors="replace") if self.run_dir else None
            except OSError:
                self._texts[rel] = None
        return self._texts[rel]

    def _keys_context(self, keys: list[str]) -> tuple[dict[str, int], list[str], dict[str, int]]:
        """(basename counts, script keys, script basename counts), cached per key list."""
        if self._keys_ctx is None or self._keys_ctx[0] != id(keys) or self._keys_ctx[1] != len(keys):
            names: dict[str, int] = defaultdict(int)
            for k in keys:
                names[PurePosixPath(k).name] += 1
            scripts = [k for k in keys if k.endswith((".py", ".r", ".R", ".sh", ".ipynb"))]
            snames: dict[str, int] = defaultdict(int)
            for k in scripts:
                snames[PurePosixPath(k).name] += 1
            self._keys_ctx = (id(keys), len(keys), names, scripts, snames)
        return self._keys_ctx[2], self._keys_ctx[3], self._keys_ctx[4]

    def agents_active_at(self, ts: float) -> list[str]:
        active = set()
        for r in self.runs.values():
            if r["depth"] < 1 or r.get("start_t") is None:
                continue
            end = r.get("end_t") if r.get("end_t") is not None else float("inf")
            if r["start_t"] <= ts <= end:
                active.add(r["agent"])
        return sorted(active)

    def attribute_by_mtime(self, mtime: float) -> tuple[str | None, str]:
        active = self.agents_active_at(mtime)
        if len(active) == 1:
            return active[0], "exact"
        if len(active) > 1:
            return None, "ambiguous"
        return None, "none"

    def index_script_outputs(self, script_rels: Iterable[str]) -> None:
        """Index which run script names which output file (first writer wins)."""
        written = self.files_written()
        for rel in script_rels:
            if self.run_dir is None or not str(rel).endswith((".py", ".r", ".R", ".sh", ".ipynb")):
                continue
            text = self._text(rel)
            if text is None:
                continue
            tuid = written.get(rel)
            agent = self.agent_for(tuid) if tuid else work_owner(rel)
            for lineno, line in enumerate(text.splitlines(), start=1):
                for m in _OUTPUT_LITERAL_RE.finditer(line):
                    base = os.path.basename(m.group(1))
                    self._script_outputs.setdefault(base, {
                        "script": rel, "line": lineno, "agent": agent, "statement": line.strip()[:200]})

    def attribute_by_script(self, path: str) -> dict[str, Any] | None:
        return self._script_outputs.get(os.path.basename(str(path)))

    def attribute_script(self, rel: str, artifact_keys: Iterable[str]) -> dict[str, Any] | None:
        """created_by='script.py:LINE' when writer statement, script and creator are all unique."""
        keys = artifact_keys if isinstance(artifact_keys, list) else list(artifact_keys)
        names, scripts, snames = self._keys_context(keys)
        name = PurePosixPath(rel).name
        if names.get(name, 0) != 1:
            return None
        hit = self.attribute_by_script(rel)
        if not hit or not _WRITER_RE.search(hit["statement"]):
            return None
        mentions = []
        pattern = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])")
        for sk in scripts:
            text = self._text(sk)
            if text and name in text and pattern.search(text):
                mentions.append(sk)
        if len(mentions) != 1 or mentions[0] != hit["script"]:
            return None
        sname = PurePosixPath(hit["script"]).name
        if snames.get(sname, 0) != 1:
            return None
        creators = {self.agent_for(tuid) for _p, tuid in self.written_by_name().get(sname, [])}
        creators.discard(None)
        if len(creators) != 1:
            return None
        agent = next(iter(creators))
        if agent in SUPPORT_AGENTS - {"cso"} or agent == "_mcp":
            return None
        return {"agent": agent, "created_by": f"{sname}:{hit['line']}", "script": hit["script"],
                "line": hit["line"], "statement": hit["statement"]}

    def attribute_path(self, path: str, mtime: float | None = None,
                       artifact_keys: Iterable[str] | None = None) -> dict[str, Any]:
        """Where a run file came from: tool input > MCP return > script:line > window."""
        rel = str(path)
        if self.run_dir is not None and os.path.isabs(rel):
            rel = to_rel(rel, self.run_dir) or rel
        written = self.files_written()
        tuid = written.get(rel)
        if tuid is None:
            same = self.written_by_name().get(PurePosixPath(rel).name, [])
            if len(same) == 1:
                tuid = same[0][1]
        if tuid is not None:
            rec = self.calls[tuid]
            return {"agent": rec.get("agent"), "tool_use_id": tuid, "tool": rec.get("tool"),
                    "created_by": rec.get("tool"), "method": "tool_input", "confidence": "exact"}
        returned = self.files_returned()
        if rel in returned:
            rec = self.calls[returned[rel]]
            return {"agent": rec.get("agent"), "tool_use_id": returned[rel], "tool": rec.get("tool"),
                    "created_by": rec.get("tool"), "method": "mcp_return", "confidence": "exact"}
        if artifact_keys is not None:
            hit = self.attribute_script(rel, artifact_keys)
            if hit:
                return {"agent": hit["agent"], "tool_use_id": None, "tool": None,
                        "created_by": hit["created_by"], "method": "script", "confidence": "exact"}
        if mtime is not None:
            agent, conf = self.attribute_by_mtime(mtime)
            return {"agent": agent, "tool_use_id": None, "tool": None, "created_by": None,
                    "method": "window" if conf != "none" else None, "confidence": conf}
        return {"agent": None, "tool_use_id": None, "tool": None, "created_by": None, "method": None,
                "confidence": "none"}

    # ------------------------------------------------------------ per-turn activity

    def turn_activity(self) -> dict[int, dict[str, Any]]:
        """{turn: dispatched agents, specialists, data tools used, plan writes} from turn windows."""
        out: dict[int, dict[str, Any]] = {}

        def rec(n: int) -> dict[str, Any]:
            if n not in out:
                base = self.turns.get(n, {})
                out[n] = {"turn": n, "agents": [], "specialists": [], "data_tools": [], "n_data_tool_calls": 0,
                          "n_delegations": 0, "plan_writes": 0, "orientation_runs": 0,
                          "start_t": base.get("start_t"), "end_t": base.get("end_t")}
            return out[n]

        for n in self.turns:
            rec(n)
        for r in self.runs.values():
            if r.get("turn") is None or r["depth"] < 1:
                continue
            a = rec(int(r["turn"]))
            if self.is_orientation_run(r):
                a["orientation_runs"] += 1
            a["n_delegations"] += 1
            if r["agent"] not in a["agents"]:
                a["agents"].append(r["agent"])
            if r["agent"] not in SUPPORT_AGENTS and r["agent"] not in a["specialists"]:
                a["specialists"].append(r["agent"])
        for d in self.delegations.values():
            if d.get("turn") is None:
                continue
            a = rec(int(d["turn"]))
            ag = str(d.get("agent"))
            if ag not in a["agents"]:
                a["agents"].append(ag)
            if ag not in SUPPORT_AGENTS and ag not in a["specialists"]:
                a["specialists"].append(ag)
        for c in self.calls.values():
            if c.get("turn") is None:
                continue
            a = rec(int(c["turn"]))
            tool = c.get("tool") or ""
            if tool == "mcp__provenance__write_plan" and not c.get("is_error") and not c.get("pending"):
                a["plan_writes"] += 1
            if is_data_tool(tool):
                run = self.runs.get(str(c.get("agent_run_id"))) if c.get("agent_run_id") else None
                if run is not None and self.is_orientation_run(run):
                    continue
                if run is None and c.get("agent") == "chief-of-staff" and not any(
                        r["agent"] == "chief-of-staff" and not self.is_orientation_run(r)
                        and r.get("turn") == c.get("turn") for r in self.runs.values()):
                    continue
                a["n_data_tool_calls"] += 1
                if tool not in a["data_tools"]:
                    a["data_tools"].append(tool)
        return dict(sorted(out.items()))

    # ------------------------------------------------------------ summaries

    def timeline(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self.runs.values():
            if r.get("start"):
                rows.append({"t": r.get("start_t"), "ts": r["start"], "event": "agent_start", "agent": r["agent"],
                             "agent_run_id": r["agent_run_id"], "detail": r.get("description") or ""})
            if r.get("end"):
                rows.append({"t": r.get("end_t"), "ts": r["end"], "event": "agent_end", "agent": r["agent"],
                             "agent_run_id": r["agent_run_id"],
                             "detail": f"{r.get('status')}; {r.get('duration_s')}s"})
        for c in self.calls.values():
            if c.get("started_at"):
                rows.append({"t": c.get("start_t"), "ts": c["started_at"],
                             "event": "tool_error" if c["is_error"] else ("tool_pending" if c["pending"] else "tool"),
                             "agent": c.get("agent"), "agent_run_id": c.get("agent_run_id"),
                             "detail": c.get("tool") or "", "tool_use_id": c["tool_use_id"]})
        rows.sort(key=lambda r: (r["t"] is None, r["t"] or 0, r["ts"] or ""))
        return rows

    def tool_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for c in self.calls.values():
            a = c.get("agent") or "unknown"
            name = c.get("tool") or "?"
            out.setdefault(a, {})
            out[a][name] = out[a].get(name, 0) + 1
        return out

    def summary(self) -> dict[str, Any]:
        n_spec = sum(1 for c in self.calls.values() if c.get("agent") not in (None, "cso", "_cso"))
        return {
            "n_agent_runs": len(self.runs),
            "n_delegated_runs": sum(1 for r in self.runs.values() if r["depth"] >= 1),
            "specialists": self.specialist_types(),
            "n_tool_calls": len(self.calls),
            "n_attributed_to_specialist": n_spec,
            "n_cso_tool_calls": len(self.calls) - n_spec,
            "n_tool_errors": sum(1 for c in self.calls.values() if c["is_error"]),
            "n_pending_tool_calls": sum(1 for c in self.calls.values() if c["pending"]),
            "n_files_written": len(self.files_written()),
            "n_files_returned": len(self.files_returned()),
            "malformed_trace_lines": self.malformed_lines,
        }

    def to_dict(self, attribution: Mapping[str, Any] | None = None) -> dict[str, Any]:
        calls = []
        for c in sorted(self.calls.values(), key=lambda c: (c.get("start_t") is None, c.get("start_t") or 0)):
            row = dict(c)
            if isinstance(row.get("command"), str) and len(row["command"]) > 4000:
                row["command"] = row["command"][:4000] + "...(clipped)"
            calls.append(row)
        return {
            "summary": self.summary(),
            "agent_runs": list(self.runs.values()),
            "execution": self.execution(),
            "tool_calls": calls,
            "files_written": self.files_written(),
            "files_returned": self.files_returned(),
            "tool_counts": self.tool_counts(),
            "turns": [{k: v for k, v in a.items()} for a in self.turn_activity().values()],
            "timeline": self.timeline(),
            "attribution": dict(attribution or {}),
        }


def _preview(inp: Any, n: int = 2000) -> Any:
    if isinstance(inp, Mapping):
        out = {}
        for k, v in inp.items():
            if isinstance(v, str) and len(v) > 500 and k in ("content", "new_source", "new_string", "old_string"):
                out[k] = f"<{len(v):,} chars>"
            else:
                out[k] = v
        s = json.dumps(out, default=str)
        return out if len(s) <= n else s[:n] + "...(clipped)"
    if inp is None:
        return None
    s = str(inp)
    return s if len(s) <= n else s[:n] + "...(clipped)"


def build_index(events: list[dict[str, Any]] | None = None, run_dir: str | Path | None = None, *,
                malformed_lines: int = 0) -> Provenance:
    """Build the provenance index. With ``events=None`` the run's trace is read (tolerantly)."""
    if events is None:
        if run_dir is None:
            return Provenance([], None)
        events, malformed_lines = read_trace(Path(run_dir) / "logs" / "trace.jsonl")
    return Provenance(events, run_dir, malformed_lines=malformed_lines)


def research_turns(prov: Provenance, artifacts: Mapping[str, Mapping[str, Any]] | None = None,
                   turns: Iterable[Mapping[str, Any]] | None = None) -> dict[int, list[str]]:
    """Turns that performed research, with reasons.

    A research turn delegated to a non-support specialist, used an ``mcp__`` data
    tool or WebSearch/WebFetch (outside the Chief of Staff's orientation brief),
    or produced new files under ``work/`` (outside support agents' and harness
    directories).
    """
    out: dict[int, list[str]] = defaultdict(list)
    for n, a in prov.turn_activity().items():
        if a["specialists"]:
            out[n].append("delegated to " + ", ".join(a["specialists"]))
        if a["data_tools"]:
            out[n].append("used " + ", ".join(a["data_tools"][:6]))
    counts: dict[int, int] = defaultdict(int)
    for rel, e in (artifacts or {}).items():
        if not isinstance(e, Mapping):
            continue
        owner = work_owner(rel)
        if owner is None or owner in SUPPORT_AGENTS - {"cso", "_cso"} or owner == "_tool_outputs":
            continue
        for n in {e.get("turn"), e.get("modified_turn")}:
            if isinstance(n, int) and not isinstance(n, bool) and n > 0:
                counts[n] += 1
    for n, c in counts.items():
        out[n].append(f"produced {c} work/ file(s)")
    for t in turns or []:
        if isinstance(t, Mapping) and t.get("research") is True and isinstance(t.get("turn"), int):
            if not out.get(t["turn"]):
                out[t["turn"]].append("recorded as a research turn")
    return {n: v for n, v in sorted(out.items()) if v}


__all__ = [
    "Provenance", "build_index", "read_trace", "research_turns", "extract_written_paths",
    "extract_returned_paths", "parse_bash", "normalize_path", "is_data_tool", "is_capture_tool",
    "SUPPORT_AGENTS", "coerce_input", "rel_parts", "data_call_fields",
]

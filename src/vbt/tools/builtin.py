"""Built-in tools with the same names and arguments as Claude Code's tools.

The upstream system prompts and skills were written against the Claude Code
tool surface (Read, Write, Edit, Glob, Grep, Bash, TodoWrite, Skill, WebFetch,
WebSearch). Re-implementing that surface here, provider-neutrally, lets the
original prompts run unchanged on any model backend.

Path policy: writes are confined to the run directory; reads are allowed in
the run directory plus configured read roots (skills, reference data).
"""

from __future__ import annotations

import asyncio
import fnmatch
import html
import os
import re
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolFailure, schema


# ---------------------------------------------------------------- path policy

def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = ctx.workspace / p
    return p.resolve()


def _check_read(ctx: ToolContext, p: Path) -> None:
    roots = [ctx.run.dir, *ctx.runtime.read_roots]
    if not any(p == r or r in p.parents for r in roots):
        raise ToolFailure(f"read outside permitted roots: {p}. Allowed: {[str(r) for r in roots]}")


def _check_write(ctx: ToolContext, p: Path) -> None:
    if not (p == ctx.run.dir or ctx.run.dir in p.parents):
        raise ToolFailure(f"writes must stay inside the run directory {ctx.run.dir}; got {p}")


# ---------------------------------------------------------------- file tools

def _read(ctx: ToolContext, a: dict[str, Any]) -> str:
    p = _resolve(ctx, a["file_path"])
    _check_read(ctx, p)
    if p.is_dir():
        raise ToolFailure(f"{p} is a directory; use Glob or Bash ls")
    if not p.exists():
        raise ToolFailure(f"file not found: {p}")
    if p.suffix.lower() in (".h5ad", ".h5", ".parquet", ".pkl", ".png", ".pdf", ".gz", ".zip"):
        return f"{p} is a binary file ({p.stat().st_size:,} bytes). Load it from code via Bash."
    lines = p.read_text(errors="replace").splitlines()
    offset = max(int(a.get("offset") or 1), 1)
    limit = int(a.get("limit") or 2000)
    chunk = lines[offset - 1: offset - 1 + limit]
    body = "\n".join(f"{i:6d}\t{line[:2000]}" for i, line in enumerate(chunk, start=offset))
    more = len(lines) - (offset - 1 + len(chunk))
    return body + (f"\n... ({more} more lines; use offset/limit)" if more > 0 else "")


def _write(ctx: ToolContext, a: dict[str, Any]) -> str:
    p = _resolve(ctx, a["file_path"])
    _check_write(ctx, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(a["content"])
    ctx.run.trace("file_write", agent=ctx.agent, path=ctx.run.rel(p), bytes=len(a["content"]))
    return f"Wrote {len(a['content']):,} chars to {p}"


def _edit(ctx: ToolContext, a: dict[str, Any]) -> str:
    p = _resolve(ctx, a["file_path"])
    _check_write(ctx, p)
    if not p.exists():
        raise ToolFailure(f"file not found: {p}")
    text = p.read_text()
    old, new = a["old_string"], a["new_string"]
    n = text.count(old)
    if n == 0:
        raise ToolFailure("old_string not found in file")
    if n > 1 and not a.get("replace_all"):
        raise ToolFailure(f"old_string occurs {n} times; make it unique or set replace_all")
    p.write_text(text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1))
    ctx.run.trace("file_write", agent=ctx.agent, path=ctx.run.rel(p), edit=True)
    return f"Edited {p} ({n if a.get('replace_all') else 1} replacement(s))"


def _glob(ctx: ToolContext, a: dict[str, Any]) -> str:
    base = _resolve(ctx, a.get("path") or str(ctx.run.dir))
    _check_read(ctx, base)
    hits = sorted(str(p) for p in base.glob(a["pattern"]))
    return "\n".join(hits[:500]) + (f"\n... ({len(hits) - 500} more)" if len(hits) > 500 else "") or "(no matches)"


def _grep(ctx: ToolContext, a: dict[str, Any]) -> str:
    base = _resolve(ctx, a.get("path") or str(ctx.run.dir))
    _check_read(ctx, base)
    flags = re.IGNORECASE if a.get("-i") else 0
    rx = re.compile(a["pattern"], flags)
    files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
    if a.get("glob"):
        files = [p for p in files if fnmatch.fnmatch(p.name, a["glob"])]
    mode = a.get("output_mode", "files_with_matches")
    out: list[str] = []
    for f in files:
        if f.stat().st_size > 20_000_000:
            continue
        try:
            lines = f.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        matched = [(i, l) for i, l in enumerate(lines, 1) if rx.search(l)]
        if not matched:
            continue
        if mode == "content":
            out += [f"{f}:{i}:{l[:500]}" for i, l in matched]
        elif mode == "count":
            out.append(f"{f}:{len(matched)}")
        else:
            out.append(str(f))
        if len(out) > int(a.get("head_limit") or 250):
            break
    return "\n".join(out) or "(no matches)"


# ---------------------------------------------------------------- bash

async def _bash(ctx: ToolContext, a: dict[str, Any]) -> str:
    cfg = ctx.runtime.config.get("bash", {})
    if not cfg.get("enabled", True):
        raise ToolFailure("Bash is disabled in this configuration")
    for pat in cfg.get("blocked_patterns", []):
        if re.search(pat, a["command"]):
            raise ToolFailure(f"command blocked by policy ({pat}); do not install packages or leave the workspace")
    timeout = min(float(a.get("timeout") or cfg.get("default_timeout_s", 600)), float(cfg.get("max_timeout_s", 7200)))
    env = {**os.environ, **ctx.runtime.tool_env(ctx), "MPLBACKEND": "Agg"}
    proc = await asyncio.create_subprocess_shell(
        a["command"], cwd=str(ctx.workspace), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        executable=cfg.get("shell", "/bin/bash"),
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise ToolFailure(f"command timed out after {timeout:.0f}s")
    text = out.decode(errors="replace")
    ctx.run.trace("bash", agent=ctx.agent, command=a["command"][:4000], exit_code=proc.returncode)
    if proc.returncode != 0:
        raise ToolFailure(f"exit code {proc.returncode}\n{text}")
    return text or "(no output)"


# ---------------------------------------------------------------- todo

def _todo(ctx: ToolContext, a: dict[str, Any]) -> str:
    ctx.run.todos[ctx.agent] = a.get("todos", [])
    ctx.run.trace("todo", agent=ctx.agent, todos=a.get("todos", []))
    return "Todo list updated."


# ---------------------------------------------------------------- skills

def _skill(ctx: ToolContext, a: dict[str, Any]) -> str:
    name = a["skill"].strip().lstrip("/")
    for root in ctx.runtime.skill_roots:
        d = root / name
        if (d / "SKILL.md").exists():
            files = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.name != "SKILL.md")
            ctx.run.trace("skill", agent=ctx.agent, skill=name)
            listing = "\n".join(f"  - {d / f}" for f in files)
            return (f"# Skill: {name}\n(base directory: {d})\n\n{(d / 'SKILL.md').read_text()}\n\n"
                    f"Supporting files (Read them when the skill tells you to):\n{listing or '  (none)'}")
    available = sorted({p.parent.name for r in ctx.runtime.skill_roots for p in r.glob('*/SKILL.md')})
    raise ToolFailure(f"unknown skill {name!r}; available: {available}")


# ---------------------------------------------------------------- web

_TAG = re.compile(r"<(script|style)[\s\S]*?</\1>|<[^>]+>", re.I)


async def _web_fetch(ctx: ToolContext, a: dict[str, Any]) -> str:
    if not ctx.runtime.config.get("web", {}).get("enabled", True):
        raise ToolFailure("web access is disabled for this run (e.g. to prevent information leakage)")
    import httpx

    async with httpx.AsyncClient(follow_redirects=True, timeout=60,
                                 headers={"User-Agent": "vbt-harness/0.1 (research)"}) as client:
        r = await client.get(a["url"])
    if r.status_code >= 400:
        raise ToolFailure(f"HTTP {r.status_code} fetching {a['url']}")
    text = r.text
    if "html" in r.headers.get("content-type", ""):
        text = html.unescape(_TAG.sub(" ", text))
        text = re.sub(r"\s+", " ", text)
    limit = int(ctx.runtime.config.get("web", {}).get("fetch_max_chars", 60000))
    note = f"\n\n[truncated at {limit} chars]" if len(text) > limit else ""
    ctx.run.trace("web_fetch", agent=ctx.agent, url=a["url"], status=r.status_code)
    prompt = f"(Requested focus: {a['prompt']})\n\n" if a.get("prompt") else ""
    return f"{prompt}Content of {a['url']}:\n{text[:limit]}{note}"


async def _web_search(ctx: ToolContext, a: dict[str, Any]) -> Any:
    if not ctx.runtime.config.get("web", {}).get("enabled", True):
        raise ToolFailure("web search is disabled for this run (e.g. to prevent information leakage)")
    backend = ctx.runtime.search_backend
    if backend is None:
        raise ToolFailure("no web search backend is configured for this provider")
    result = await backend(a["query"], max_results=int(a.get("max_results") or 8))
    ctx.run.cost.extra_usd += float(result.pop("cost_usd", 0.0) or 0.0)
    ctx.run.trace("web_search", agent=ctx.agent, query=a["query"], n=len(result.get("results", [])))
    return result


# ---------------------------------------------------------------- registry

def builtin_tools() -> list[Tool]:
    path = {"type": "string", "description": "Absolute path (relative paths resolve to your workspace)"}
    return [
        Tool("Read", "Read a text file with line numbers. Use offset/limit for large files.",
             schema({"file_path": path, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["file_path"]),
             _read),
        Tool("Write", "Create or overwrite a file inside the run directory.",
             schema({"file_path": path, "content": {"type": "string"}}, ["file_path", "content"]), _write),
        Tool("Edit", "Exact string replacement in a file. old_string must be unique unless replace_all.",
             schema({"file_path": path, "old_string": {"type": "string"}, "new_string": {"type": "string"},
                     "replace_all": {"type": "boolean"}}, ["file_path", "old_string", "new_string"]), _edit),
        Tool("Glob", "Find files by glob pattern, e.g. '**/*.csv'.",
             schema({"pattern": {"type": "string"}, "path": path}, ["pattern"]), _glob),
        Tool("Grep", "Regex search in files. output_mode: files_with_matches | content | count.",
             schema({"pattern": {"type": "string"}, "path": path, "glob": {"type": "string"},
                     "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
                     "-i": {"type": "boolean"}, "head_limit": {"type": "integer"}}, ["pattern"]), _grep),
        Tool("Bash", "Run a shell command in your workspace (Python, R and the analysis stack are "
                     "pre-installed). Write scripts to code/scripts/ and run them; outputs go under your workspace. "
                     "Never install packages.",
             schema({"command": {"type": "string"}, "timeout": {"type": "number", "description": "seconds"},
                     "description": {"type": "string"}}, ["command"]), _bash),
        Tool("TodoWrite", "Record your task list (items: content, status pending|in_progress|completed).",
             schema({"todos": {"type": "array", "items": {"type": "object", "properties": {
                 "content": {"type": "string"}, "status": {"type": "string"}}}}}, ["todos"]), _todo),
        Tool("Skill", "Load a skill (a packaged workflow with procedures and references) by name.",
             schema({"skill": {"type": "string"}}, ["skill"]), _skill),
        Tool("WebFetch", "Fetch a URL and return its text content.",
             schema({"url": {"type": "string"}, "prompt": {"type": "string"}}, ["url"]), _web_fetch),
        Tool("WebSearch", "Search the web; returns sources (title, url) and a cited summary.",
             schema({"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]), _web_search),
    ]

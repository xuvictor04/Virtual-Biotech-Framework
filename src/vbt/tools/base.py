"""Tool abstraction shared by built-in tools, MCP-bridged tools and delegation.

Also home of the harness-side tool-output utilities:

* :func:`shrink_json` -- structure-aware truncation of large JSON tool outputs
  (every top-level key survives; long arrays and strings are shortened with
  markers naming the exact ``QueryToolOutput`` call that reads the rest);
* :func:`parse_json_path` / :func:`eval_json_path` -- the small JSON-path
  dialect (``$.a.b``, ``[n]``, ``[a:b]``, ``["key"]``, ``.keys()``);
* :func:`query_tool_output_tool` -- the ``QueryToolOutput`` harness tool that
  every agent gets automatically, restricted to ``logs/tool_outputs``.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Union

from ..providers.base import ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from ..providers.base import Usage
    from ..runtime import Runtime
    from ..session import Run

#: Agents whose relative paths resolve against the run directory when the
#: runtime cannot be asked (``Runtime.workspace_for`` is authoritative).
RUN_WORKSPACE_AGENTS = frozenset({"cso", "scientific-reviewer"})

QUERY_TOOL = "QueryToolOutput"
TOOL_OUTPUTS_DIR = ("logs", "tool_outputs")


class ToolFailure(Exception):
    """Raise inside a handler to return an error result to the model."""


@dataclass
class ToolContext:
    """Everything a tool may need about the calling agent and the run."""

    agent: str
    run: "Run"
    runtime: "Runtime"
    tool_call_id: str = ""
    depth: int = 0
    invocation_id: str = ""   # the calling agent's run_agent invocation (agent_run_id in the trace)

    @property
    def workspace(self) -> Path:
        """The calling agent's base directory for relative paths (``Runtime.workspace_for``)."""
        fn = getattr(self.runtime, "workspace_for", None)
        if callable(fn):
            return fn(self.agent)
        if self.agent in RUN_WORKSPACE_AGENTS:
            return self.run.dir
        return self.run.agent_dir(self.agent)

    def add_cost(self, usd: float, label: str, usage: "Usage | None" = None) -> None:
        """Charge a tool-side cost (web search fee, extraction model call, ...) to the
        calling agent: its ledger entry, its ``AgentResult.cost_usd`` and every
        enclosing budget scope. Pass ``usage`` when the cost is a model call."""
        fn = getattr(self.runtime, "_charge", None)
        if callable(fn):
            fn(self.agent, usage, usd, label=label)
            return
        cost = getattr(self.run, "cost", None)
        if cost is None:
            return
        if usage is not None:
            cost.add(self.agent, usage, float(usd or 0.0))
        elif hasattr(cost, "add_extra"):
            cost.add_extra(agent=self.agent, usd=usd, label=label)
        else:
            cost.extra_usd += float(usd or 0.0)

    def trace(self, type: str, **data: Any) -> None:
        """Trace an event tagged with this call's agent run ids and tool_use_id."""
        data.setdefault("agent", self.agent)
        if self.tool_call_id:
            data.setdefault("tool_use_id", self.tool_call_id)
        fn = getattr(self.runtime, "_trace", None)
        if callable(fn):
            fn(type, **data)
        else:
            self.run.trace(type, **data)

    def emit(self, kind: str, **data: Any) -> None:
        """Publish a UI event (never raises)."""
        fn = getattr(self.runtime, "emit", None)
        if callable(fn):
            try:
                fn(kind, **data)
            except Exception:  # noqa: BLE001 - UI events never break tools
                pass


Handler = Callable[[ToolContext, dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass
class Tool:
    """A tool as the harness runs it.

    ``terminal``: a successful call ends the agent's loop without another model
    call (e.g. a bulk ``submit_result``). ``blocking``: the handler is
    synchronous and may block (file walks, hashing, parsing); it runs in a
    worker thread so parallel agents keep streaming. ``strict``: ask providers
    that support it to constrain the arguments to ``input_schema``
    (``ToolSpec.strict``; grammar-enforced by vLLM), for schema-heavy tools.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    source: str = "builtin"  # builtin | mcp:<server> | provenance | delegation | harness
    tags: set[str] = field(default_factory=set)
    terminal: bool = False
    blocking: bool = False
    strict: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema, strict=bool(self.strict))

    async def __call__(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        if self.blocking and not inspect.iscoroutinefunction(self.handler):
            r = await asyncio.to_thread(self.handler, ctx, args)
        else:
            r = self.handler(ctx, args)
        if inspect.isawaitable(r):
            r = await r
        return r


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs``/``$ref`` so the schema works with every provider's tool API."""
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and node["$ref"].split("/")[-1] in defs:
                target = walk(defs[node["$ref"].split("/")[-1]])
                return {**target, **{k: walk(v) for k, v in node.items() if k != "$ref"}}
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=1, default=str)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.add(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def select(self, patterns: list[str]) -> list[Tool]:
        """Resolve an allowlist of exact names or glob patterns (``mcp__genetics__*``)."""
        chosen: dict[str, Tool] = {}
        for pat in patterns:
            for name in sorted(fnmatch.filter(self._tools, pat)):
                chosen[name] = self._tools[name]
        return list(chosen.values())

    def missing(self, patterns: list[str]) -> list[str]:
        return [p for p in patterns if not fnmatch.filter(self._tools, p)]


# ---------------------------------------------------------------------------
# JSON paths
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def path_key(base: str, key: Any) -> str:
    """``base`` extended by a dict key, as ``.key`` or ``["key"]``."""
    k = str(key)
    return f"{base}.{k}" if _IDENT.match(k) else f"{base}[{json.dumps(k, ensure_ascii=False)}]"


def parse_json_path(path: str) -> list[tuple[Any, ...]]:
    """Parse ``$.a.b[3]["odd key"][10:20].keys()`` into tokens.

    Tokens: ``("key", str)``, ``("index", int)``, ``("slice", start|None, stop|None)``, ``("keys",)``.
    The leading ``$`` is optional (``a.b`` and ``.a.b`` work too).
    """
    s = (path or "").strip()
    if s.startswith("$"):
        s = s[1:]
    toks: list[tuple[Any, ...]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if s.startswith(".keys()", i) or (i == 0 and s.startswith("keys()", i)):
            toks.append(("keys",))
            i += 7 if s[i] == "." else 6
            continue
        if c == "[":
            j, quote = i + 1, None
            while j < n:
                ch = s[j]
                if quote:
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == quote:
                        quote = None
                elif ch in "\"'":
                    quote = ch
                elif ch == "]":
                    break
                j += 1
            if j >= n:
                raise ValueError(f"unclosed '[' in json_path {path!r}")
            inner = s[i + 1:j].strip()
            i = j + 1
            if inner[:1] in ("'", '"') and inner[-1:] == inner[:1] and len(inner) >= 2:
                if inner[0] == '"':
                    toks.append(("key", json.loads(inner)))
                else:
                    toks.append(("key", inner[1:-1].replace("\\'", "'")))
            elif ":" in inner:
                a, b = inner.split(":", 1)
                toks.append(("slice", int(a) if a.strip() else None, int(b) if b.strip() else None))
            elif re.fullmatch(r"-?\d+", inner):
                toks.append(("index", int(inner)))
            elif inner == "*":
                toks.append(("slice", None, None))
            else:
                raise ValueError(f"bad selector [{inner}] in json_path {path!r}")
            continue
        if c == ".":
            i += 1
            if i < n and s[i] == "[":
                continue
        m = re.match(r"[^.\[]+", s[i:])
        if not m:
            raise ValueError(f"bad json_path {path!r} at position {i}")
        toks.append(("key", m.group(0).strip()))
        i += m.end()
    return toks


def format_json_path(toks: list[tuple[Any, ...]]) -> str:
    """Canonical ``$...`` form of parsed tokens."""
    out = "$"
    for tok in toks:
        if tok[0] == "key":
            out = path_key(out, tok[1])
        elif tok[0] == "index":
            out += f"[{tok[1]}]"
        elif tok[0] == "slice":
            out += f"[{'' if tok[1] is None else tok[1]}:{'' if tok[2] is None else tok[2]}]"
        elif tok[0] == "keys":
            out += ".keys()"
    return out


def eval_json_path(data: Any, path: str) -> Any:
    """Select ``path`` from parsed JSON ``data`` (raises ToolFailure with hints)."""
    try:
        toks = parse_json_path(path)
    except ValueError as exc:
        raise ToolFailure(str(exc)) from None
    cur, where = data, "$"
    for tok in toks:
        kind = tok[0]
        if kind == "keys":
            if isinstance(cur, dict):
                return list(cur.keys())
            if isinstance(cur, list):
                return {"type": "array", "length": len(cur)}
            return {"type": type(cur).__name__}
        if kind == "key":
            key = tok[1]
            if isinstance(cur, dict):
                if key not in cur:
                    keys = list(cur.keys())
                    raise ToolFailure(f"key {key!r} not found at {where}; keys: "
                                      f"{keys[:50]}{' ...' if len(keys) > 50 else ''}")
                cur, where = cur[key], path_key(where, key)
                continue
            if isinstance(cur, list) and re.fullmatch(r"-?\d+", str(key)):
                tok = ("index", int(key))
                kind = "index"
            else:
                raise ToolFailure(f"{where} is a {type(cur).__name__}, not an object; cannot select {key!r}")
        if kind == "index":
            if not isinstance(cur, (list, str)):
                raise ToolFailure(f"{where} is a {type(cur).__name__}, not an array; cannot index [{tok[1]}]")
            try:
                cur, where = cur[tok[1]], f"{where}[{tok[1]}]"
            except IndexError:
                raise ToolFailure(f"index {tok[1]} out of range at {where} (length {len(cur)})") from None
            continue
        if kind == "slice":
            if not isinstance(cur, (list, str)):
                raise ToolFailure(f"{where} is a {type(cur).__name__}, not an array; cannot slice")
            a, b = tok[1], tok[2]
            cur, where = cur[a:b], f"{where}[{'' if a is None else a}:{'' if b is None else b}]"
    return cur


# ---------------------------------------------------------------------------
# Structure-aware truncation
# ---------------------------------------------------------------------------

# (max list items, max string chars, max nesting depth), from gentle to drastic
_SHRINK_LEVELS = [(50, 4000, 10), (30, 2000, 8), (20, 1000, 6), (10, 500, 5), (5, 300, 4), (3, 160, 3),
                  (2, 100, 2), (1, 80, 1)]


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str, ensure_ascii=False)


def _marker(spill: str | None, json_path: str, what: str) -> str:
    where = f"QueryToolOutput path={spill} json_path={json_path}" if spill else f"json_path={json_path}"
    return f"...({what}; {where})"


def _shrink(obj: Any, jp: str, level: tuple[int, int, int], depth: int, spill: str | None) -> Any:
    max_items, max_str, max_depth = level
    if isinstance(obj, str):
        if len(obj) > max_str:
            return obj[:max_str] + _marker(spill, jp, f"{len(obj) - max_str:,} more chars")
        return obj
    if isinstance(obj, dict):
        if depth >= max_depth and obj:
            return _marker(spill, jp, f"object with {len(obj)} keys")
        return {k: _shrink(v, path_key(jp, k), level, depth + 1, spill) for k, v in obj.items()}
    if isinstance(obj, list):
        if depth >= max_depth and obj:
            return [_marker(spill, f"{jp}[0:{min(len(obj), 20)}]", f"array of {len(obj)} items")]
        out = [_shrink(v, f"{jp}[{i}]", level, depth + 1, spill) for i, v in enumerate(obj[:max_items])]
        if len(obj) > max_items:
            k = max_items
            chunk = max(k, 20)
            out.append(_marker(spill, f"{jp}[{k}:{k + chunk}]", f"{len(obj) - k:,} more items"))
        return out
    return obj


def shrink_json(obj: Any, limit: int, *, spill_path: str | None = None, root: str = "$") -> str:
    """Render ``obj`` as JSON text of at most ``limit`` chars where possible.

    Every top-level key is kept; long arrays and strings are progressively
    shortened (and deep containers collapsed) with markers such as
    ``...(180 more items; QueryToolOutput path=<p> json_path=$.key[20:40])``.
    ``root`` is the JSON path of ``obj`` inside the spilled file.
    """
    full = _dump(obj)
    if len(full) <= limit:
        return full
    for level in _SHRINK_LEVELS:
        text = _dump(_shrink(obj, root, level, 0, spill_path))
        if len(text) <= limit:
            return text
    # Too many top-level entries: keep the keys, summarise every value.
    if isinstance(obj, dict):
        summary = {}
        for k, v in obj.items():
            kind = ("object" if isinstance(v, dict) else "array" if isinstance(v, list) else type(v).__name__)
            size = len(v) if isinstance(v, (dict, list, str)) else None
            summary[k] = _marker(spill_path, path_key(root, k), f"{kind}" + (f" of size {size:,}" if size else ""))
        text = json.dumps(summary, default=str, ensure_ascii=False)
        if len(text) <= limit:
            return text
        keys = [str(k) for k in obj.keys()]
        k = len(keys)
        while k > 0:
            text = json.dumps({"__truncated__": _marker(spill_path, f"{root}.keys()",
                                                        f"{len(keys):,} top-level keys, {len(keys) - k:,} not listed; "
                                                        f"read a value with json_path={root}[\"<key>\"]"),
                               "keys": keys[:k]}, ensure_ascii=False)
            if len(text) <= limit:
                return text
            k = k // 2
        return _marker(spill_path, f"{root}.keys()", f"object with {len(keys):,} keys")[:max(limit, 0)]
    text = _dump(_shrink(obj, root, _SHRINK_LEVELS[-1], 0, spill_path))
    return text[:limit] + "\n" + _marker(spill_path, root, "output clipped")


def truncation_note(total_chars: int, spill_path: str, *, structured: bool, has_read: bool) -> str:
    """Note appended to a truncated tool output; names only tools the agent has."""
    how = ("use QueryToolOutput(path, json_path) to read any part (e.g. json_path=$.key[20:40], or "
           "$.key.keys() to list keys)") if structured else (
        "use QueryToolOutput(path, offset, limit) to page through it")
    if has_read:
        how += " or Read with offset/limit"
    kind = "structure kept; long arrays/strings shortened" if structured else "middle omitted"
    return f"[Output truncated: {total_chars:,} chars ({kind}). Full output saved to {spill_path}; {how}.]"


def truncate_text(text: str, limit: int) -> str:
    """Head and tail of ``text`` (errors often sit at the end of command output)."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    return text[:head] + f"\n\n[... {omitted:,} chars omitted ...]\n\n" + (text[-tail:] if tail else "")


def maybe_parse_json(text: str, max_chars: int = 200_000_000) -> Any:
    """Parsed JSON if ``text`` is a JSON object/array, else None."""
    s = text.lstrip()
    if not s or s[0] not in "{[" or len(text) > max_chars:
        return None
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return obj if isinstance(obj, (dict, list)) else None


# ---------------------------------------------------------------------------
# QueryToolOutput
# ---------------------------------------------------------------------------

_JSON_CACHE: "OrderedDict[tuple[str, int, int], Any]" = OrderedDict()
_JSON_CACHE_LOCK = threading.Lock()


def _load_json_cached(p: Path) -> Any:
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _JSON_CACHE_LOCK:
        if key in _JSON_CACHE:
            _JSON_CACHE.move_to_end(key)
            return _JSON_CACHE[key]
    with open(p, encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[key] = data
        while len(_JSON_CACHE) > 4:
            _JSON_CACHE.popitem(last=False)
    return data


def tool_outputs_root(run_dir: Path) -> Path:
    return (Path(run_dir).joinpath(*TOOL_OUTPUTS_DIR)).resolve()


def _resolve_output_path(ctx: ToolContext, raw: str) -> Path:
    root = tool_outputs_root(ctx.run.dir)
    raw = os.path.expanduser(raw.strip().strip("'\""))
    if not raw:
        raise ToolFailure("path is required (the spill file named in a truncated tool output)")
    p = Path(raw)
    cands = [p] if p.is_absolute() else [ctx.run.dir / p, root / p, root / "cleared" / p]
    inside = [c for c in (x.resolve() for x in cands) if c == root or root in c.parents]
    for c in inside:
        if c.is_file():
            return c
    if inside:
        raise ToolFailure(f"no such tool output: {inside[0]}")
    raise ToolFailure(f"QueryToolOutput reads only saved tool outputs under {root}; got {raw}")


def _query(ctx: ToolContext, a: dict[str, Any]) -> Any:
    """QueryToolOutput: :func:`_query_raw` with secret values masked (saved outputs are not trusted to be clean)."""
    from ..envpolicy import redact

    value = _query_raw(ctx, a)
    if isinstance(value, str):
        return redact(value, os.environ)
    try:
        dumped = json.dumps(value, ensure_ascii=False)
        masked = redact(dumped, os.environ)
        return value if masked == dumped else json.loads(masked)
    except (TypeError, ValueError):
        return value


def _query_raw(ctx: ToolContext, a: dict[str, Any]) -> Any:
    p = _resolve_output_path(ctx, str(a.get("path") or ""))
    limit_chars = int(getattr(ctx.runtime, "tool_output_max", 40000) or 40000) - 600
    jp = a.get("json_path")
    offset = a.get("offset")
    limit = a.get("limit")
    if jp:
        try:
            data = _load_json_cached(p)
        except ValueError as exc:
            raise ToolFailure(f"{p.name} is not JSON ({exc}); omit json_path and page it with offset/limit") \
                from None
        value = eval_json_path(data, str(jp))
        toks = parse_json_path(str(jp))
        if toks and toks[-1][0] == "keys":
            return value
        root = format_json_path(toks)
        if isinstance(value, (list, str)) and (offset is not None or limit is not None):
            start = max(int(offset or 0), 0)
            stop = start + int(limit) if limit is not None else None
            value = value[start:stop]
            root = f"{root}[{start}:{'' if stop is None else stop}]"
        if isinstance(value, (dict, list)):
            text = shrink_json(value, limit_chars, spill_path=str(p), root=root)
            if len(text) < len(_dump(value)):
                text += "\n[Selection shortened; narrow json_path (markers above name the exact calls).]"
            return text
        return value
    # line paging (1-based offset, like Read)
    start = max(int(offset or 1), 1)
    n = max(int(limit or 400), 1)
    out: list[str] = []
    used, total = 0, 0
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            total = i
            if i < start or len(out) >= n:
                continue
            line = line.rstrip("\n")
            if len(line) > 4000:
                line = line[:4000] + f"...({len(line) - 4000:,} more chars on this line; use json_path for JSON)"
            if used + len(line) > limit_chars:
                n = len(out)
                continue
            used += len(line) + 8
            out.append(f"{i:6d}\t{line}")
    body = "\n".join(out) or "(no lines in range)"
    last = start + len(out) - 1
    if total > last:
        body += f"\n... ({total - last} more lines; next: offset={last + 1})"
    return body


def query_tool_output_tool() -> Tool:
    return Tool(
        QUERY_TOOL,
        "Read part of a saved tool output (the file named in an '[Output truncated ...]' note or a "
        "'[cleared by harness ...]' stub; only files under logs/tool_outputs). For JSON outputs give json_path: "
        "$.key, $.a.b[3], $[\"odd key\"], $.items[20:40], or $.key.keys() to list keys; offset/limit then slice "
        "a selected array (0-based items) or string. Without json_path, pages the file by lines (offset is the "
        "1-based first line, default limit 400).",
        schema({"path": {"type": "string", "description": "saved output path (absolute, run-relative or file name)"},
                "json_path": {"type": "string", "description": "JSON path into the output, e.g. $.adverseEvents"},
                "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
        _query, source="harness", blocking=True)


__all__ = [
    "Tool", "ToolContext", "ToolFailure", "ToolRegistry", "schema", "inline_refs", "to_text", "QUERY_TOOL",
    "RUN_WORKSPACE_AGENTS", "parse_json_path", "format_json_path", "eval_json_path", "path_key", "shrink_json", "truncation_note",
    "truncate_text", "maybe_parse_json", "query_tool_output_tool", "tool_outputs_root",
]

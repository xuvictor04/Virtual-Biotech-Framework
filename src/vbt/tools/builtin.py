"""Built-in tools with the same names and arguments as Claude Code's tools.

The upstream system prompts and skills were written against the Claude Code
tool surface (Read, Write, Edit, NotebookEdit, Glob, Grep, Bash, TodoWrite,
Skill, WebFetch, WebSearch). Re-implementing that surface here,
provider-neutrally, lets the original prompts run unchanged on any model
backend. ``UpdateMemory`` adds the per-agent project memory.

Every file tool and Bash share one :class:`~vbt.tools.policy.PathPolicy`
(reads: run dir + read roots, a precedence blocklist for credentials; writes:
the agent's own ``work/<agent>/``, never harness records). Bash additionally
runs under the command policy (package installs, destructive and system
commands, optional network block), with an allow-listed environment (no
provider keys), its own process group (killed on timeout or cancellation) and
a byte cap on captured output.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import envpolicy
from ..providers.base import DocumentPart, ImagePart, TextBlock
from .base import Tool, ToolContext, ToolFailure, schema, tool_outputs_root
from .notebook import notebook_edit_tool, render_notebook
from .policy import CommandDenied, CommandPolicy, PathPolicy, check_command
from .skills import SkillIndex, normalize_skill_name
from .web import web_fetch as _web_fetch
from .web import web_search as _web_search

# ---------------------------------------------------------------- shared path handling


def _policy(ctx: ToolContext) -> PathPolicy:
    return PathPolicy.from_ctx(ctx)


def _resolve(ctx: ToolContext, path: str, *, for_read: bool = True) -> Path:
    """Run-relative prefixes (work/, inputs/, evidence/, report/, logs/, .claude/) resolve
    against the run dir, other relative paths against the agent's workspace (falling back
    to the run dir for reads when the workspace path does not exist)."""
    if path is None or not str(path).strip():
        raise ToolFailure("a path is required")
    return _policy(ctx).resolve(str(path), for_read=for_read)


def _check_read(ctx: ToolContext, p: Path) -> None:
    why = _policy(ctx).read_denial(p)
    if why:
        raise ToolFailure(why)


def _check_write(ctx: ToolContext, p: Path) -> None:
    why = _policy(ctx).write_denial(p)
    if why:
        raise ToolFailure(why)


def _cfg(ctx: ToolContext, section: str) -> dict[str, Any]:
    return dict(((getattr(ctx.runtime, "config", None) or {}).get(section) or {}))


def _caps(ctx: ToolContext) -> Any:
    """The calling agent's provider capabilities (None when unknown)."""
    rt = ctx.runtime
    prov = getattr(rt, "provider", None)
    if prov is None or not hasattr(prov, "capabilities"):
        return None
    model = None
    try:
        d = rt._definition(ctx.agent)  # noqa: SLF001
        if d is not None:
            model = d.settings(rt.config).model
    except Exception:  # noqa: BLE001
        model = None
    try:
        return prov.capabilities(model)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- Read

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
               ".webp": "image/webp"}
BINARY_SUFFIXES = frozenset({
    ".h5ad", ".h5", ".hdf5", ".h5mu", ".loom", ".parquet", ".feather", ".arrow", ".pkl", ".pickle", ".joblib",
    ".npy", ".npz", ".rds", ".rda", ".rdata", ".gz", ".bgz", ".zip", ".bz2", ".xz", ".zst", ".tar", ".7z",
    ".bam", ".cram", ".bw", ".bigwig", ".bed.gz", ".sqlite", ".db", ".duckdb", ".xlsx", ".xls", ".docx",
    ".pptx", ".so", ".dylib", ".bin", ".tiff", ".tif", ".bmp", ".ico", ".mp4", ".mov", ".svgz",
})
MAX_IMAGE_PX = 1568
MAX_IMAGE_BYTES = 3_750_000          # raw bytes (~5 MB base64)
MAX_LINE_CHARS = 2000
_READ_LINE_CAP = 100_000             # chars of one physical line kept in memory


def _human(n: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return str(n)  # pragma: no cover


def _image_result(ctx: ToolContext, p: Path) -> Any:
    size = p.stat().st_size
    caps = _caps(ctx)
    if not (caps is not None and getattr(caps, "images", False)):
        return (f"{p} is an image ({_human(size)}). This model cannot view images; describe or analyse it from "
                f"code via Bash (e.g. inspect the data that produced it).")
    media = IMAGE_TYPES[p.suffix.lower()]
    if size > 50_000_000:
        return f"{p} is an image of {_human(size)}, too large to view; downscale it from code first."
    data = p.read_bytes()
    note = ""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        Image = None  # type: ignore[assignment]
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
                note = f"{w}x{h}px"
                if max(w, h) > MAX_IMAGE_PX or len(data) > MAX_IMAGE_BYTES:
                    im = im.copy()
                    if getattr(im, "is_animated", False):
                        im.seek(0)
                    im.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX))
                    buf = io.BytesIO()
                    if im.mode in ("RGBA", "LA", "P") or media == "image/png":
                        if im.mode not in ("RGB", "RGBA", "L", "LA"):
                            im = im.convert("RGBA")
                        im.save(buf, format="PNG", optimize=True)
                        media = "image/png"
                    else:
                        im.convert("RGB").save(buf, format="JPEG", quality=85)
                        media = "image/jpeg"
                    if buf.tell() > MAX_IMAGE_BYTES:
                        buf = io.BytesIO()
                        im.convert("RGB").save(buf, format="JPEG", quality=70)
                        media = "image/jpeg"
                    data = buf.getvalue()
                    note += f", downscaled to {im.size[0]}x{im.size[1]}px"
        except Exception as exc:  # noqa: BLE001 - unreadable image
            return f"{p} could not be decoded as an image: {exc}"
    elif len(data) > MAX_IMAGE_BYTES:
        return (f"{p} is an image of {_human(size)}, above the {_human(MAX_IMAGE_BYTES)} limit; install Pillow "
                f"(pip install 'vbt-harness[tools]') so Read can downscale it, or save a smaller version.")
    header = f"Image {p} ({_human(size)}{', ' + note if note else ''})"
    ctx.trace("file_read", path=ctx.run.rel(p), kind="image", bytes=len(data))
    return [TextBlock(header), ImagePart(media, base64.b64encode(data).decode("ascii"), source=str(p))]


def _parse_pages(spec: Any) -> list[int]:
    """'1-5', '3', '1,3,5-7' -> 1-based page numbers (at most 20)."""
    if spec in (None, ""):
        return []
    pages: list[int] = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ToolFailure(f"bad pages {spec!r}; use e.g. '1-5' or '3'")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if a < 1 or b < a:
            raise ToolFailure(f"bad page range {part!r}")
        pages.extend(range(a, b + 1))
    if len(pages) > 20:
        raise ToolFailure("at most 20 pages per Read; request a smaller range")
    return pages


def _pdf_result(ctx: ToolContext, p: Path, a: dict[str, Any]) -> Any:
    size = p.stat().st_size
    pages = _parse_pages(a.get("pages"))
    caps = _caps(ctx)
    max_bytes = int(_cfg(ctx, "read").get("pdf_max_bytes") or 20_000_000)
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
    except ImportError:
        PdfReader = PdfWriter = None  # type: ignore[assignment]
    if pages and caps is not None and getattr(caps, "documents", False):
        data = None
        if PdfReader is not None:
            try:
                reader = PdfReader(str(p))
                n = len(reader.pages)
                bad = [x for x in pages if x > n]
                if bad:
                    raise ToolFailure(f"{p} has {n} pages; requested {bad}")
                writer = PdfWriter()
                for x in pages:
                    writer.add_page(reader.pages[x - 1])
                buf = io.BytesIO()
                writer.write(buf)
                data = buf.getvalue()
            except ToolFailure:
                raise
            except Exception:  # noqa: BLE001 - fall back to the whole file
                data = None
        if data is None and size <= max_bytes:
            data = p.read_bytes()
        if data is not None and len(data) <= max_bytes:
            ctx.trace("file_read", path=ctx.run.rel(p), kind="pdf", pages=a.get("pages"), bytes=len(data))
            return [TextBlock(f"PDF {p} ({_human(size)}), pages {a.get('pages')}"),
                    DocumentPart(data_b64=base64.b64encode(data).decode("ascii"),
                                 title=f"{p.name} pages {a.get('pages')}")]
    if PdfReader is None:
        return (f"{p} is a PDF ({_human(size)}). Install pypdf (pip install 'vbt-harness[tools]') to read its "
                f"text, or extract it from code via Bash.")
    try:
        reader = PdfReader(str(p))
        n = len(reader.pages)
        want = pages or list(range(1, min(n, 10) + 1))
        out = [f"PDF {p} ({n} pages, {_human(size)}); text of pages {want[0]}-{want[-1]}:"]
        for x in want:
            if x > n:
                break
            out.append(f"--- page {x} ---\n{(reader.pages[x - 1].extract_text() or '').strip()}")
        if not pages and n > 10:
            out.append(f"... ({n - 10} more pages; pass pages='11-20' etc.)")
        return "\n\n".join(out)
    except Exception as exc:  # noqa: BLE001
        raise ToolFailure(f"cannot read PDF {p}: {exc}") from None


def _iter_lines(f: Any) -> Iterable[tuple[str, int]]:
    """(line kept in memory, chars dropped from an overlong physical line)."""
    while True:
        line = f.readline(_READ_LINE_CAP)
        if not line:
            return
        dropped = 0
        if not line.endswith("\n"):
            while True:
                chunk = f.readline(_READ_LINE_CAP)
                if not chunk:
                    break
                dropped += len(chunk)
                if chunk.endswith("\n"):
                    break
        yield line.rstrip("\r\n"), dropped


def _read(ctx: ToolContext, a: dict[str, Any]) -> Any:
    p = _resolve(ctx, a.get("file_path"))
    _check_read(ctx, p)
    if p.is_dir():
        raise ToolFailure(f"{p} is a directory; use Glob or Bash ls")
    if not p.exists():
        raise ToolFailure(f"file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in IMAGE_TYPES:
        return _image_result(ctx, p)
    if suffix == ".pdf":
        return _pdf_result(ctx, p, a)
    if suffix == ".ipynb":
        return render_notebook(p)
    size = p.stat().st_size
    if suffix in BINARY_SUFFIXES or "".join(p.suffixes[-2:]).lower() in BINARY_SUFFIXES:
        return f"{p} is a binary file ({size:,} bytes). Load it from code via Bash."
    with open(p, "rb") as fb:
        if b"\0" in fb.read(8192):
            return f"{p} is a binary file ({size:,} bytes). Load it from code via Bash."
    if size == 0:
        return f"{p} exists but is empty."
    offset = max(int(a.get("offset") or 1), 1)
    limit = max(int(a.get("limit") or 2000), 1)
    max_chars = int(_cfg(ctx, "read").get("max_output_chars") or 250_000)
    out: list[str] = []
    used = 0
    more = False
    capped = False
    last = offset - 1
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        for i, (line, dropped) in enumerate(_iter_lines(f), start=1):
            if i < offset:
                continue
            if len(out) >= limit:
                more = True
                break
            if len(line) > MAX_LINE_CHARS or dropped:
                line = line[:MAX_LINE_CHARS] + f"... [line truncated: {len(line) + dropped - MAX_LINE_CHARS:,} " \
                                               f"more chars]"
            entry = f"{i:6d}\t{line}"
            if used + len(entry) > max_chars and out:
                more = capped = True
                break
            out.append(entry)
            used += len(entry) + 1
            last = i
    if not out:
        return f"{p} has fewer than {offset} lines (offset {offset} is past the end)."
    body = "\n".join(out)
    if more:
        why = f" (output capped at {max_chars:,} chars)" if capped else ""
        body += f"\n... (more lines follow{why}; continue with offset={last + 1})"
    if _is_tool_output(ctx, p):  # saved tool outputs may predate redaction or come from other tools
        body = envpolicy.redact(body, os.environ)
    return body


def _is_tool_output(ctx: ToolContext, p: Path) -> bool:
    root = tool_outputs_root(ctx.run.dir)
    try:
        rp = p.resolve()
    except OSError:
        return False
    return rp == root or root in rp.parents


# ---------------------------------------------------------------- Write / Edit

def _write(ctx: ToolContext, a: dict[str, Any]) -> str:
    p = _resolve(ctx, a.get("file_path"), for_read=False)
    _check_write(ctx, p)
    if p.is_dir():
        raise ToolFailure(f"{p} is a directory")
    content = a.get("content")
    if content is None:
        raise ToolFailure("content is required")
    content = str(content)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    ctx.trace("file_write", path=ctx.run.rel(p), bytes=len(content.encode("utf-8")))
    return f"Wrote {len(content):,} chars to {p}"


def _edit(ctx: ToolContext, a: dict[str, Any]) -> str:
    p = _resolve(ctx, a.get("file_path"), for_read=False)
    _check_write(ctx, p)
    if not p.exists():
        raise ToolFailure(f"file not found: {p}")
    if p.suffix.lower() == ".ipynb":
        raise ToolFailure(f"{p} is a notebook; use NotebookEdit")
    text = p.read_text(encoding="utf-8")
    old, new = a.get("old_string"), a.get("new_string")
    if old is None or new is None:
        raise ToolFailure("old_string and new_string are required")
    if old == new:
        raise ToolFailure("old_string and new_string are identical")
    n = text.count(old) if old else 0
    if n == 0:
        raise ToolFailure("old_string not found in file")
    if n > 1 and not a.get("replace_all"):
        raise ToolFailure(f"old_string occurs {n} times; make it unique or set replace_all")
    p.write_text(text.replace(old, new) if a.get("replace_all") else text.replace(old, new, 1), encoding="utf-8")
    ctx.trace("file_write", path=ctx.run.rel(p), edit=True, bytes=p.stat().st_size)
    return f"Edited {p} ({n if a.get('replace_all') else 1} replacement(s))"


# ---------------------------------------------------------------- Glob

_WILD = re.compile(r"[*?\[{]")
MAX_GLOB_RESULTS = 500


def expand_braces(pattern: str) -> list[str]:
    """``a.{csv,tsv}`` -> ``['a.csv', 'a.tsv']`` (nested braces supported)."""
    start = pattern.find("{")
    while start >= 0:
        depth, alts, cur = 0, [], start + 1
        for i in range(start, len(pattern)):
            c = pattern[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    alts.append(pattern[cur:i])
                    if len(alts) > 1:
                        out: list[str] = []
                        for alt in alts:
                            out.extend(expand_braces(pattern[:start] + alt + pattern[i + 1:]))
                        return list(dict.fromkeys(out))
                    break
            elif c == "," and depth == 1:
                alts.append(pattern[cur:i])
                cur = i + 1
        start = pattern.find("{", start + 1)
    return [pattern]


def _glob(ctx: ToolContext, a: dict[str, Any]) -> str:
    pattern = str(a.get("pattern") or "").strip()
    if not pattern:
        raise ToolFailure("pattern is required")
    if ".." in Path(pattern).parts or ".." in pattern.replace("\\", "/").split("/"):
        raise ToolFailure("Glob patterns cannot traverse parent directories; set path to an allowed directory "
                          "and use a pattern within it")
    pol = _policy(ctx)
    expanded = os.path.expanduser(pattern)
    if os.path.isabs(expanded):
        parts = Path(expanded).parts
        i = next((k for k, s in enumerate(parts) if _WILD.search(s)), len(parts))
        base = Path(os.path.normpath(os.path.join(*parts[:i]))) if i else Path("/")
        rest = "/".join(parts[i:])
    else:
        base = pol.resolve(a.get("path") or str(ctx.workspace), for_read=True) if a.get("path") \
            else Path(os.path.normpath(ctx.workspace))
        rest = pattern
    why = pol.read_denial(base)
    if why:
        raise ToolFailure(why)
    if not base.exists():
        raise ToolFailure(f"directory not found: {base}")
    if not rest:
        return str(base) if base.exists() else "(no matches)"
    hits: dict[str, float] = {}
    t0 = time.time()
    stopped = False
    for pat in expand_braces(rest):
        try:
            gen = base.glob(pat)
            for h in gen:
                if len(hits) >= 20 * MAX_GLOB_RESULTS or time.time() - t0 > 30:
                    stopped = True
                    break
                s = str(h)
                if s in hits or not pol.allows_read(h):
                    continue
                try:
                    hits[s] = h.stat().st_mtime
                except OSError:
                    continue
        except (ValueError, NotImplementedError) as exc:
            raise ToolFailure(f"bad glob pattern {pat!r}: {exc}") from None
    ordered = sorted(hits, key=lambda k: (-hits[k], k))
    out = "\n".join(ordered[:MAX_GLOB_RESULTS])
    if len(ordered) > MAX_GLOB_RESULTS:
        out += f"\n... ({len(ordered) - MAX_GLOB_RESULTS} more; narrow the pattern)"
    if stopped:
        out += "\n[search stopped early (result or time budget); narrow the pattern]"
    return out or "(no matches)"


# ---------------------------------------------------------------- Grep

TYPE_EXTS = {
    "py": (".py", ".pyi"), "python": (".py", ".pyi"), "r": (".r", ".rmd", ".qmd"), "csv": (".csv",),
    "tsv": (".tsv", ".tab"), "md": (".md", ".markdown"), "markdown": (".md", ".markdown"), "json": (".json",),
    "yaml": (".yaml", ".yml"), "yml": (".yaml", ".yml"), "txt": (".txt",), "ipynb": (".ipynb",),
    "sh": (".sh", ".bash"), "html": (".html", ".htm"), "js": (".js", ".mjs"), "toml": (".toml",),
}
GREP_MAX_FILES = 20_000
GREP_MAX_SECONDS = 30.0
_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".ipynb_checkpoints", ".cache", ".tmp", ".home"})


def glob_to_regex(glob: str) -> re.Pattern:
    """Translate a glob with ``**``, ``*``, ``?``, ``[..]`` and ``{a,b}`` to a regex."""
    i, n = 0, len(glob)
    out: list[str] = []
    depth = 0
    while i < n:
        c = glob[i]
        if c == "*":
            if glob.startswith("**", i):
                j = i + 2
                if j < n and glob[j] == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = glob.find("]", i + 1)
            if j < 0:
                out.append(re.escape(c))
            else:
                body = glob[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j + 1
                continue
        elif c == "{":
            depth += 1
            out.append("(?:")
        elif c == "}" and depth:
            depth -= 1
            out.append(")")
        elif c == "," and depth:
            out.append("|")
        else:
            out.append(re.escape(c))
        i += 1
    out.extend(")" * depth)
    return re.compile("".join(out) + r"\Z")


def _grep_files(base: Path, pol: PathPolicy, glob_rx: re.Pattern | None, glob_has_slash: bool,
                exts: tuple[str, ...] | None, budget: dict[str, Any]) -> Iterable[Path]:
    def wanted(p: Path, rel: str) -> bool:
        if glob_rx is not None:
            target = rel if glob_has_slash else p.name
            if not glob_rx.match(target):
                return False
        if exts is not None and not p.name.lower().endswith(exts):
            return False
        return True

    if base.is_file():
        if wanted(base, base.name):
            yield base
        return
    for root, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not (d.startswith(".") and d != ".claude"))
        for name in sorted(files):
            budget["files"] += 1
            if budget["files"] > GREP_MAX_FILES or time.monotonic() - budget["t0"] > GREP_MAX_SECONDS:
                budget["stopped"] = True
                return
            p = Path(root) / name
            rel = p.relative_to(base).as_posix()
            if not wanted(p, rel):
                continue
            if p.is_symlink() and not pol.allows_read(p):
                continue  # symlinked file re-checked against the policy
            yield p


def _is_binary(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def _grep(ctx: ToolContext, a: dict[str, Any]) -> str:
    pattern = a.get("pattern")
    if not pattern:
        raise ToolFailure("pattern is required")
    pol = _policy(ctx)
    base = pol.resolve(a["path"], for_read=True) if a.get("path") else Path(os.path.normpath(ctx.workspace))
    why = pol.read_denial(base)
    if why:
        raise ToolFailure(why)
    if not base.exists():
        raise ToolFailure(f"path not found: {base}")
    multiline = bool(a.get("multiline"))
    flags = (re.IGNORECASE if a.get("-i") else 0) | (re.MULTILINE | re.DOTALL if multiline else 0)
    try:
        rx = re.compile(str(pattern), flags)
    except re.error as exc:
        raise ToolFailure(f"invalid regex {pattern!r}: {exc}") from None
    glob = a.get("glob")
    glob_rx = glob_to_regex(str(glob)) if glob else None
    exts = None
    if a.get("type"):
        t = str(a["type"]).lower().lstrip(".")
        exts = TYPE_EXTS.get(t, ("." + t,))
    mode = a.get("output_mode") or "files_with_matches"
    if mode not in ("files_with_matches", "content", "count"):
        raise ToolFailure("output_mode must be files_with_matches, content or count")
    ctx_c = a.get("-C")
    before = int(a.get("-B") if a.get("-B") is not None else (ctx_c or 0))
    after = int(a.get("-A") if a.get("-A") is not None else (ctx_c or 0))
    show_n = a.get("-n", True) is not False
    head_limit = a.get("head_limit")
    head_limit = int(head_limit) if head_limit not in (None, "") else 250
    offset = max(int(a.get("offset") or 0), 0)
    need = offset + head_limit if head_limit > 0 else None

    budget = {"files": 0, "t0": time.monotonic(), "stopped": False}
    entries: list[str] = []
    matched_files: list[tuple[float, str]] = []
    for f in _grep_files(base, pol, glob_rx, bool(glob) and "/" in str(glob), exts, budget):
        if _is_binary(f):
            continue
        try:
            if multiline:
                if f.stat().st_size > 50_000_000:
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
                lines = text.split("\n")
                starts = [0]
                for ln in lines[:-1]:
                    starts.append(starts[-1] + len(ln) + 1)
                import bisect

                hit_lines: list[int] = []
                for m in rx.finditer(text):
                    s = bisect.bisect_right(starts, m.start()) - 1
                    e = bisect.bisect_right(starts, max(m.end() - 1, m.start())) - 1
                    hit_lines.extend(range(s + 1, e + 2))
                hits = sorted(set(hit_lines))
                count = len(list(rx.finditer(text))) if mode == "count" else len(hits)
                if not hits:
                    continue
                if mode == "content":
                    entries.extend(_context_block(str(f), lines, hits, before, after, show_n))
            else:
                hitset: set[int] = set()
                count = 0
                shown: dict[int, str] = {}
                window: collections.deque = collections.deque(maxlen=max(before, 0) or None)
                pending_after = 0
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, start=1):
                        line = line.rstrip("\r\n")
                        if rx.search(line):
                            count += 1
                            hitset.add(i)
                            if mode == "files_with_matches":
                                break
                            if mode == "content":
                                for j, wl in window:
                                    shown[j] = wl
                                window.clear()
                                shown[i] = line
                                pending_after = after
                        elif mode == "content" and pending_after > 0:
                            shown[i] = line
                            pending_after -= 1
                        elif before > 0:
                            window.append((i, line))
                        if mode == "content" and need is not None and pending_after == 0 \
                                and len(entries) + len(shown) >= need:
                            break
                if not hitset:
                    continue
                if mode == "content":
                    entries.extend(_format_shown(str(f), shown, hitset, show_n))
        except OSError:
            continue
        if mode == "files_with_matches":
            try:
                matched_files.append((f.stat().st_mtime, str(f)))
            except OSError:
                matched_files.append((0.0, str(f)))
        elif mode == "count":
            entries.append(f"{f}:{count}")
        if mode != "files_with_matches" and need is not None and len(entries) >= need:
            budget["more"] = True
            break
    if mode == "files_with_matches":
        entries = [s for _, s in sorted(matched_files, key=lambda x: (-x[0], x[1]))]
    total = len(entries)
    sel = entries[offset:offset + head_limit] if head_limit > 0 else entries[offset:]
    out = "\n".join(sel) or "(no matches)"
    if head_limit > 0 and (total > offset + head_limit or budget.get("more")):
        out += f"\n... (output limited to {head_limit} entries; use offset={offset + head_limit} for more)"
    if budget["stopped"]:
        out += (f"\n[search truncated: scanned {min(budget['files'], GREP_MAX_FILES):,} files / "
                f"{GREP_MAX_SECONDS:.0f} s budget; narrow path, glob or type]")
    return out


def _fmt(path: str, i: int, line: str, show_n: bool, sep: str) -> str:
    line = line if len(line) <= 500 else line[:500] + "..."
    return f"{path}{sep}{i}{sep}{line}" if show_n else f"{path}{sep}{line}"


def _format_shown(path: str, shown: dict[int, str], hits: set[int], show_n: bool) -> list[str]:
    out: list[str] = []
    prev = None
    for i in sorted(shown):
        if prev is not None and i > prev + 1:
            out.append("--")
        out.append(_fmt(path, i, shown[i], show_n, ":" if i in hits else "-"))
        prev = i
    return out


def _context_block(path: str, lines: list[str], hits: list[int], before: int, after: int,
                   show_n: bool) -> list[str]:
    show: dict[int, str] = {}
    for h in hits:
        for i in range(max(1, h - before), min(len(lines), h + after) + 1):
            show[i] = lines[i - 1]
    return _format_shown(path, show, set(hits), show_n)


# ---------------------------------------------------------------- Bash

_UNSHARE_OK: bool | None = None


def _unshare_available() -> bool:
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        try:
            r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
            _UNSHARE_OK = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _UNSHARE_OK = False
    return _UNSHARE_OK



def network_isolation_status(config: dict[str, Any]) -> tuple[bool, str]:
    """Whether Bash commands run without network access at the OS level, and why (not).

    ``bash.network: false`` alone is a pattern block on command text and scanned
    scripts -- a guardrail, not isolation. Isolation needs
    ``bash.network_isolation: unshare`` with either a working ``unshare -rn`` or
    ``bash.sandbox.os: bwrap`` (which adds ``--unshare-net``).
    """
    cfg = (config or {}).get("bash") or {}
    iso = cfg.get("network_isolation")
    os_sandbox = str(((cfg.get("sandbox") or {}).get("os")) or "none").lower()
    if iso != "unshare":
        return False, "bash.network_isolation is not set to 'unshare'; only the command pattern block applies"
    if os_sandbox == "bwrap":
        if shutil.which("bwrap"):
            return True, "bubblewrap --unshare-net"
        return False, "bash.sandbox.os is 'bwrap' but bubblewrap is not installed"
    if _unshare_available():
        return True, "unshare -rn (new network namespace)"
    return False, ("`unshare -rn` is unavailable here (no util-linux unshare or user namespaces disabled); "
                   "only the command pattern block applies")

def bwrap_argv(pol: PathPolicy, argv: list[str], cwd: Path, *, bwrap: str = "bwrap",
               unshare_net: bool = False) -> list[str]:
    """Wrap ``argv`` in bubblewrap (``bash.sandbox.os: bwrap``).

    The whole filesystem is mounted read-only; only the agent's own work
    directory (all of ``work/`` when ``protect_other_workspaces`` is off) and
    the run's ``.tmp``/``.home`` are writable, so harness records and other
    agents' outputs cannot change even from interpreter code. Blocked paths
    (``.env``, ``.git``, credentials) are hidden: directories behind an empty
    tmpfs, files behind ``/dev/null``.
    """
    run = pol.run_dir
    rw = [pol.own_dir if pol.protect_other_workspaces else run / "work", run / ".tmp", run / ".home"]
    out = [bwrap, "--die-with-parent", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    if unshare_net:
        out.append("--unshare-net")
    for d in rw:
        d.mkdir(parents=True, exist_ok=True)
        out += ["--bind", str(d), str(d)]
    for b in pol.blocked:
        if os.path.isdir(b):
            out += ["--tmpfs", b]
        elif os.path.lexists(b):
            out += ["--ro-bind", "/dev/null", b]
    return out + ["--chdir", str(cwd), "--", *argv]


def _timeout_s(raw: Any, cfg: dict[str, Any]) -> tuple[float, list[str]]:
    """Claude Code's Bash timeout is in milliseconds; values under 1000 are read as seconds."""
    notes: list[str] = []
    default_s = float(cfg.get("default_timeout_s") or 1800)
    max_s = float(cfg.get("max_timeout_s") or 14400)
    if raw in (None, "", 0, "0"):
        t = default_s
    else:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ToolFailure(f"timeout must be a number of milliseconds, got {raw!r}") from None
        if v <= 0:
            t = default_s
        elif v < 1000:
            t = v
            notes.append(f"timeout={v:g} read as seconds (the parameter is in milliseconds)")
        else:
            t = v / 1000.0
    if t > max_s:
        notes.append(f"timeout clamped to bash.max_timeout_s={max_s:g}s")
        t = max_s
    return max(t, 1.0), notes


class _OutputSink:
    """Keep the head and tail of a command's output; spill everything once over the cap."""

    def __init__(self, max_bytes: int, spill: Path, secrets: Mapping[str, str] | None = None) -> None:
        self.max = max(int(max_bytes), 1024)
        self.half = self.max // 2
        self.spill = spill
        self.buf = bytearray()
        self.head = b""
        self.tail = bytearray()
        self.total = 0
        self.fh: Any = None
        # The spill file is shareable (Read, export, web UI): secret values are masked as it is written.
        # ``pending`` holds the last len(longest secret)-1 bytes so a secret split across chunks is caught.
        self._secrets = sorted(((v.encode("utf-8"), f"[redacted:{n}]".encode("utf-8"))
                                for v, n in envpolicy.secret_values(secrets or {}).items()),
                               key=lambda kv: -len(kv[0]))
        self._keep = max((len(v) for v, _ in self._secrets), default=1) - 1
        self.pending = bytearray()

    def _redact(self, data: bytes) -> bytes:
        for value, mark in self._secrets:
            data = data.replace(value, mark)
        return data

    def _spill_write(self, data: bytes, *, final: bool = False) -> None:
        self.pending += data
        if not self._secrets:
            self.fh.write(bytes(self.pending))
            self.pending = bytearray()
            return
        red = self._redact(bytes(self.pending))
        cut = len(red) if final else max(len(red) - self._keep, 0)
        self.fh.write(red[:cut])
        self.pending = bytearray(red[cut:])

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.fh is None:
            self.buf += chunk
            if len(self.buf) > self.max:
                self.spill.parent.mkdir(parents=True, exist_ok=True)
                self.fh = open(self.spill, "wb")  # noqa: SIM115
                self._spill_write(bytes(self.buf))
                self.head = bytes(self.buf[: self.half])
                self.tail = bytearray(self.buf[-self.half:])
                self.buf = bytearray()
            return
        self._spill_write(chunk)
        self.tail += chunk
        if len(self.tail) > 2 * self.half:
            del self.tail[: len(self.tail) - self.half]

    def close(self) -> None:
        if self.fh is not None and not self.fh.closed:
            self._spill_write(b"", final=True)
            self.fh.close()

    def text(self, rel_spill: str) -> str:
        if self.fh is None:
            return self.buf.decode("utf-8", errors="replace")
        tail = bytes(self.tail[-self.half:])
        omitted = self.total - len(self.head) - len(tail)
        return (self.head.decode("utf-8", errors="replace")
                + f"\n\n[... output truncated: {self.total:,} bytes in total, {omitted:,} bytes omitted here. "
                  f"Full output saved to {rel_spill}; page it with Read (offset/limit) or QueryToolOutput ...]\n\n"
                + tail.decode("utf-8", errors="replace"))


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


async def _kill_group(proc: asyncio.subprocess.Process, grace: float = 5.0) -> None:
    """SIGTERM the process group, wait ``grace`` s, SIGKILL it, then reap the shell."""
    pgid = proc.pid
    _signal_group(pgid, signal.SIGTERM)
    try:
        if proc.returncode is None:
            await asyncio.wait_for(proc.wait(), grace)
        else:
            deadline = time.monotonic() + min(grace, 1.0)
            while _signal_group(pgid, 0) and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
    except BaseException:  # noqa: BLE001 - timeout or a second cancellation: escalate below
        pass
    finally:
        _signal_group(pgid, signal.SIGKILL)
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except BaseException:  # noqa: BLE001
                pass


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s or "")[:120] or f"t{int(time.time() * 1000)}"


async def _bash(ctx: ToolContext, a: dict[str, Any]) -> str:
    config = getattr(ctx.runtime, "config", None) or {}
    cfg = dict(config.get("bash") or {})
    if not cfg.get("enabled", True):
        raise ToolFailure("Bash is disabled in this configuration")
    command = str(a.get("command") or "")
    if not command.strip():
        raise ToolFailure("command is required")
    run_dir = Path(ctx.run.dir)
    pol = _policy(ctx)
    cwd = Path(ctx.workspace)
    cwd.mkdir(parents=True, exist_ok=True)
    tool_env_fn = getattr(ctx.runtime, "tool_env", None)
    extra = dict(tool_env_fn(ctx) if callable(tool_env_fn) else {})
    extra.setdefault("MPLBACKEND", "Agg")
    extra.setdefault("PWD", str(cwd))
    env = envpolicy.child_env(os.environ, passthrough=cfg.get("env_passthrough") or [], extra=extra,
                              home=run_dir / ".home", tmp=run_dir / ".tmp")
    secrets_env = {k: v for k, v in os.environ.items()}
    try:
        check_command(command, pol, CommandPolicy.from_config(config), env=env, cwd=cwd)
    except CommandDenied as exc:
        ctx.trace("bash_blocked", command=envpolicy.redact(command[:4000], secrets_env), group=exc.group,
                  rule=exc.rule, reason=str(exc)[:500])
        raise ToolFailure(str(exc)) from None

    timeout, notes = _timeout_s(a.get("timeout"), cfg)
    shell = cfg.get("shell") or "/bin/bash"
    argv = [shell, "-c", command]
    iso = cfg.get("network_isolation")
    os_sandbox = str(((cfg.get("sandbox") or {}).get("os")) or "none").lower()
    if os_sandbox == "bwrap":
        exe = shutil.which("bwrap")
        if not exe:
            raise ToolFailure("bash.sandbox.os is 'bwrap' but bubblewrap is not installed; install it or set "
                              "bash.sandbox.os: none (the command policy alone is a guardrail, not a sandbox)")
        argv = bwrap_argv(pol, argv, cwd, bwrap=exe, unshare_net=iso == "unshare")
    elif os_sandbox not in ("none", "", "false"):
        raise ToolFailure(f"unknown bash.sandbox.os {os_sandbox!r} (use bwrap or none)")
    elif iso == "unshare":
        if _unshare_available():
            argv = ["unshare", "-rn", "--", *argv]
        else:
            notes.append("bash.network_isolation=unshare is unavailable here; the pattern block still applies")
    max_bytes = int(cfg.get("max_output_bytes") or 2_000_000)
    spill = run_dir / "logs" / "tool_outputs" / f"bash_{_safe_id(ctx.tool_call_id)}.txt"
    sink = _OutputSink(max_bytes, spill, secrets_env)
    t0 = time.time()
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd), env=env, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)

    async def pump() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                return
            sink.feed(chunk)

    reader = asyncio.ensure_future(pump())
    # proc.wait() also waits for the pipes to close, which a background child can hold
    # open; the shell's own exit is visible as proc.returncode.
    waiter = asyncio.ensure_future(proc.wait())
    timed_out = False
    try:
        deadline = time.monotonic() + timeout
        while proc.returncode is None and not waiter.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                await _kill_group(proc)
                break
            await asyncio.wait({reader, waiter}, timeout=min(0.2, remaining),
                               return_when=asyncio.FIRST_COMPLETED)
        # The shell exited; background children may still hold the pipe open.
        try:
            await asyncio.wait_for(asyncio.shield(reader), 2.0)
        except asyncio.TimeoutError:
            await _kill_group(proc, grace=1.0)
            try:
                await asyncio.wait_for(reader, 5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                reader.cancel()
        if _signal_group(proc.pid, 0):  # leftover group members (e.g. `cmd &`)
            await _kill_group(proc, grace=1.0)
        try:
            await asyncio.wait_for(waiter, 5.0)
        except asyncio.TimeoutError:
            pass
    except BaseException:
        reader.cancel()
        waiter.cancel()
        await _kill_group(proc)
        sink.close()
        ctx.trace("bash", command=envpolicy.redact(command[:4000], secrets_env), exit_code=None,
                  interrupted=True, duration_s=round(time.time() - t0, 2))
        raise
    finally:
        sink.close()
    rel_spill = ctx.run.rel(spill)
    text = envpolicy.redact(sink.text(rel_spill), secrets_env)
    truncated = sink.fh is not None
    ctx.trace("bash", command=envpolicy.redact(command[:4000], secrets_env), exit_code=proc.returncode,
              timed_out=timed_out, timeout_s=timeout, duration_s=round(time.time() - t0, 2),
              output_bytes=sink.total, output_path=rel_spill if truncated else None, notes=notes or None,
              network_isolation=iso if iso and (argv[0] == "unshare" or os_sandbox == "bwrap") else None,
              os_sandbox=os_sandbox if os_sandbox == "bwrap" else None)
    prefix = "".join(f"[note: {n}]\n" for n in notes)
    if timed_out:
        raise ToolFailure(f"{prefix}[timed out after {timeout:g}s; process group killed]\n{text}")
    if proc.returncode != 0:
        raise ToolFailure(f"{prefix}exit code {proc.returncode}\n{text}")
    return prefix + (text or "(no output)")


# ---------------------------------------------------------------- TodoWrite

def _todo(ctx: ToolContext, a: dict[str, Any]) -> str:
    ctx.run.todos[ctx.agent] = a.get("todos", [])
    ctx.trace("todo", todos=a.get("todos", []))
    return "Todo list updated."


# ---------------------------------------------------------------- Skill

def _skill_index(ctx: ToolContext, fallback: list[Path] | None) -> SkillIndex:
    roots = getattr(ctx.runtime, "skill_roots", None) or fallback or []
    return SkillIndex.build(roots)


def _make_skill_handler(fallback_roots: list[Path] | None):
    def _skill(ctx: ToolContext, a: dict[str, Any]) -> str:
        try:
            name = normalize_skill_name(a.get("skill") or a.get("name") or "")
        except ValueError as exc:
            raise ToolFailure(str(exc)) from None
        idx = _skill_index(ctx, fallback_roots)
        sk = idx.get(name)
        if sk is None:
            raise ToolFailure(f"unknown skill {name!r}; available: {idx.names()}")
        run_copy = Path(ctx.run.dir) / ".claude" / "skills" / sk.name
        base = run_copy if run_copy.exists() else sk.path
        files = sorted(p.relative_to(sk.path).as_posix() for p in sk.path.rglob("*")
                       if p.is_file() and p.name != "SKILL.md" and "__pycache__" not in p.parts)
        ctx.trace("skill", skill=sk.name, path=str(sk.path))
        listing = "\n".join(f"  - {base / f}" for f in files)
        args = f"\n\nArguments: {a['args']}" if a.get("args") else ""
        return (f"# Skill: {sk.name}\n(base directory: {base})\n\n{sk.skill_md.read_text(errors='replace')}\n\n"
                f"Supporting files (Read them when the skill tells you to):\n{listing or '  (none)'}{args}")

    return _skill


def _skill_description(skill_roots: Iterable[Any] | None) -> str:
    desc = "Load a skill (a packaged workflow with procedures and references) by name."
    if skill_roots is None:
        return desc
    catalog = SkillIndex.build(skill_roots).catalog_text(6000)
    if not catalog:
        return desc
    return (desc + " Use it when a task matches a skill below; '.claude/skills/<name>' and '/<name>' forms are "
            "accepted.\n\nAvailable skills:\n" + catalog)


# ---------------------------------------------------------------- UpdateMemory

def _update_memory(ctx: ToolContext, a: dict[str, Any]) -> str:
    from ..agents import memory_path  # local import: agents is loaded after the tool modules

    content = a.get("content")
    if content is None or not str(content).strip():
        raise ToolFailure("content is required")
    mode = (a.get("mode") or "append").lower()
    if mode not in ("append", "replace"):
        raise ToolFailure("mode must be append or replace")
    max_chars = int(_cfg(ctx, "memory").get("max_chars") or 50_000)
    p = memory_path(Path(ctx.run.dir), ctx.agent)  # harness-owned: bypasses the protected-write list
    p.parent.mkdir(parents=True, exist_ok=True)
    text = str(content).rstrip() + "\n"
    if mode == "append" and p.exists():
        old = p.read_text(encoding="utf-8", errors="replace")
        text = old + ("" if old.endswith("\n") or not old else "\n") + text
    if len(text) > max_chars:
        raise ToolFailure(f"memory would exceed {max_chars:,} chars; call UpdateMemory with mode='replace' and a "
                          f"condensed version")
    p.write_text(text, encoding="utf-8")
    ctx.trace("memory_write", path=ctx.run.rel(p), mode=mode, chars=len(str(content)))
    return f"Memory {'updated' if mode == 'append' else 'replaced'}: {ctx.run.rel(p)} ({text.count(chr(10))} lines)."


# ---------------------------------------------------------------- registry

def builtin_tools(skill_roots: Iterable[Any] | None = None) -> list[Tool]:
    """The built-in tool surface. ``skill_roots`` puts the skill catalog in the Skill
    tool's description (the runtime's ``skill_roots`` are used at call time)."""
    roots = [Path(r) for r in skill_roots] if skill_roots is not None else None
    path = {"type": "string", "description": "Absolute path, or relative: work/... inputs/... evidence/... "
                                             "report/... logs/... .claude/... resolve to the run directory, other "
                                             "relative paths to your workspace"}
    return [
        Tool("Read", "Read a file with line numbers (offset/limit page large files; lines over 2000 chars are "
                     "cut). Images (png/jpg/gif/webp) are shown to models that can view them; PDFs: pass pages "
                     "(e.g. '1-5'); notebooks render as cells with outputs.",
             schema({"file_path": path, "offset": {"type": "integer", "description": "1-based first line"},
                     "limit": {"type": "integer", "description": "number of lines (default 2000)"},
                     "pages": {"type": "string", "description": "PDF page range, e.g. '1-5'"}}, ["file_path"]),
             _read, blocking=True),
        Tool("Write", "Create or overwrite a file inside your workspace.",
             schema({"file_path": path, "content": {"type": "string"}}, ["file_path", "content"]), _write,
             blocking=True),
        Tool("Edit", "Exact string replacement in a file. old_string must be unique unless replace_all.",
             schema({"file_path": path, "old_string": {"type": "string"}, "new_string": {"type": "string"},
                     "replace_all": {"type": "boolean"}}, ["file_path", "old_string", "new_string"]), _edit,
             blocking=True),
        notebook_edit_tool(),
        Tool("Glob", "Find files by glob pattern (e.g. '**/*.csv', '*.{csv,tsv}', or an absolute pattern); "
                     "results sorted by modification time, newest first.",
             schema({"pattern": {"type": "string"}, "path": path}, ["pattern"]), _glob, blocking=True),
        Tool("Grep", "Regex search in files. output_mode: files_with_matches (default) | content | count. glob "
                     "filters paths ('**/*.csv', '*.{csv,tsv}'); type filters by language (py, r, csv, tsv, md, "
                     "json, yaml, txt, ipynb). -A/-B/-C context and -n line numbers apply to content mode.",
             schema({"pattern": {"type": "string"}, "path": path, "glob": {"type": "string"},
                     "type": {"type": "string"},
                     "output_mode": {"type": "string", "enum": ["files_with_matches", "content", "count"]},
                     "-i": {"type": "boolean"}, "-n": {"type": "boolean"}, "-A": {"type": "integer"},
                     "-B": {"type": "integer"}, "-C": {"type": "integer"}, "multiline": {"type": "boolean"},
                     "head_limit": {"type": "integer"}, "offset": {"type": "integer"}}, ["pattern"]),
             _grep, blocking=True),
        Tool("Bash", "Run a shell command in your workspace (Python, R and the analysis stack are "
                     "pre-installed). Write scripts to code/scripts/ and run them; outputs go under your workspace. "
                     "Never install packages. Reads are limited to the run and reference data, writes to your "
                     "workspace; provider credentials are not in the environment.",
             schema({"command": {"type": "string"},
                     "timeout": {"type": "number", "description": "milliseconds (as in Claude Code); default from "
                                                                  "bash.default_timeout_s"},
                     "description": {"type": "string"}}, ["command"]), _bash),
        Tool("TodoWrite", "Record your task list (items: content, status pending|in_progress|completed).",
             schema({"todos": {"type": "array", "items": {"type": "object", "properties": {
                 "content": {"type": "string"}, "status": {"type": "string"}}}}}, ["todos"]), _todo),
        Tool("Skill", _skill_description(roots),
             schema({"skill": {"type": "string", "description": "skill name"},
                     "args": {"type": "string"}}, ["skill"]), _make_skill_handler(roots), blocking=True),
        Tool("UpdateMemory", "Record notes for later instances of your role in this run (key findings, file paths, "
                             "data quirks, failed approaches). mode: append (default) or replace.",
             schema({"content": {"type": "string"}, "mode": {"type": "string", "enum": ["append", "replace"]}},
                    ["content"]), _update_memory, source="harness", blocking=True),
        Tool("WebFetch", "Fetch a URL (http/https; public hosts only). With a prompt, returns a focused extract of "
                         "the page; the full page is saved under your workspace data/raw/web/.",
             schema({"url": {"type": "string"}, "prompt": {"type": "string"}}, ["url"]), _web_fetch),
        Tool("WebSearch", "Search the web; returns sources (title, url) and a cited summary.",
             schema({"query": {"type": "string"}, "max_results": {"type": "integer"},
                     "allowed_domains": {"type": "array", "items": {"type": "string"}},
                     "blocked_domains": {"type": "array", "items": {"type": "string"}}}, ["query"]), _web_search),
    ]


__all__ = ["builtin_tools", "expand_braces", "glob_to_regex", "IMAGE_TYPES"]

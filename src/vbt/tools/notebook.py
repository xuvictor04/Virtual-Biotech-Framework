"""Jupyter notebooks: Read rendering and the ``NotebookEdit`` tool.

Notebooks are handled as plain JSON (nbformat 4); ``nbformat`` is used for
validation only when it is installed. ``NotebookEdit`` mirrors Claude Code's
tool: ``notebook_path``, ``cell_id`` or ``cell_number`` (0-based),
``new_source``, ``cell_type`` (code|markdown) and ``edit_mode``
(replace|insert|delete). Writes go through the same path policy as Write/Edit
and are traced as ``file_write``.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolFailure, schema

MAX_OUTPUT_CHARS = 4000


def _source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src or "")


def _split_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines


def _output_text(out: dict[str, Any]) -> str:
    kind = out.get("output_type")
    if kind == "stream":
        t = out.get("text", "")
        return "".join(t) if isinstance(t, list) else str(t)
    if kind in ("execute_result", "display_data"):
        data = out.get("data") or {}
        if "text/plain" in data:
            t = data["text/plain"]
            text = "".join(t) if isinstance(t, list) else str(t)
        else:
            text = ""
        others = [k for k in data if k != "text/plain"]
        if any(k.startswith("image/") for k in others):
            text += ("\n" if text else "") + f"[{', '.join(k for k in others if k.startswith('image/'))} output]"
        return text
    if kind == "error":
        tb = out.get("traceback") or []
        import re

        clean = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(tb))
        return f"{out.get('ename', 'Error')}: {out.get('evalue', '')}\n{clean}".strip()
    return ""


def load_notebook(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, ValueError) as exc:
        raise ToolFailure(f"cannot parse notebook {path}: {exc}") from None
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise ToolFailure(f"{path} is not a Jupyter notebook (no 'cells' list)")
    return nb


def render_notebook(path: Path, *, max_chars: int = 200_000) -> str:
    """Numbered cells with their source and text outputs (images noted)."""
    nb = load_notebook(path)
    lang = ((nb.get("metadata") or {}).get("kernelspec") or {}).get("language") or \
        ((nb.get("metadata") or {}).get("language_info") or {}).get("name") or "python"
    parts = [f"Notebook {path} ({len(nb['cells'])} cells, language {lang})"]
    for i, cell in enumerate(nb["cells"]):
        ctype = cell.get("cell_type", "code")
        cid = cell.get("id")
        head = f"--- cell {i}" + (f" (id {cid})" if cid else "") + f" [{ctype}]"
        if ctype == "code" and cell.get("execution_count") is not None:
            head += f" In[{cell['execution_count']}]"
        parts.append(head)
        parts.append(_source(cell))
        outs = [t for t in (_output_text(o) for o in cell.get("outputs") or []) if t]
        if outs:
            text = "\n".join(outs)
            if len(text) > MAX_OUTPUT_CHARS:
                text = text[:MAX_OUTPUT_CHARS] + f"\n... ({len(text) - MAX_OUTPUT_CHARS:,} more output chars)"
            parts.append("[output]\n" + text)
    body = "\n".join(parts)
    if len(body) > max_chars:
        body = body[:max_chars] + f"\n... (notebook rendering truncated at {max_chars:,} chars)"
    return body


def _new_cell(cell_type: str, source: str, nbformat_minor: int) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}, "source": _split_source(source)}
    if nbformat_minor >= 5:
        cell["id"] = uuid.uuid4().hex[:8]
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _find_index(nb: dict[str, Any], a: dict[str, Any], *, insert: bool) -> int:
    cells = nb["cells"]
    cid = a.get("cell_id")
    if cid not in (None, ""):
        for i, c in enumerate(cells):
            if str(c.get("id")) == str(cid):
                return i + 1 if insert else i
        if str(cid).lstrip("-").isdigit():  # Claude Code also accepts a numeric cell_id
            a = {**a, "cell_number": int(cid)}
        else:
            raise ToolFailure(f"no cell with id {cid!r}; ids: {[c.get('id') for c in cells][:50]}")
    num = a.get("cell_number")
    if num in (None, ""):
        if insert:
            return 0
        raise ToolFailure("give cell_id or cell_number (0-based)")
    try:
        n = int(num)
    except (TypeError, ValueError):
        raise ToolFailure(f"cell_number must be an integer, got {num!r}") from None
    limit = len(cells) if insert else len(cells) - 1
    if n < 0 or n > limit:
        raise ToolFailure(f"cell_number {n} out of range (notebook has {len(cells)} cells)")
    return n


def edit_notebook(path: Path, a: dict[str, Any]) -> str:
    """Apply one NotebookEdit to ``path`` in place; returns a summary."""
    mode = (a.get("edit_mode") or "replace").lower()
    if mode not in ("replace", "insert", "delete"):
        raise ToolFailure("edit_mode must be replace, insert or delete")
    if path.exists():
        nb = load_notebook(path)
    elif mode == "insert":
        nb = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    else:
        raise ToolFailure(f"notebook not found: {path}")
    minor = int(nb.get("nbformat_minor") or 4)
    cells = nb["cells"]
    ctype = a.get("cell_type")
    if ctype not in (None, "", "code", "markdown", "raw"):
        raise ToolFailure("cell_type must be code or markdown")
    if mode == "insert":
        if a.get("new_source") is None:
            raise ToolFailure("new_source is required to insert a cell")
        idx = _find_index(nb, a, insert=True)
        cells.insert(idx, _new_cell(ctype or "code", str(a["new_source"]), minor))
        summary = f"Inserted {ctype or 'code'} cell at index {idx}"
    elif mode == "delete":
        idx = _find_index(nb, a, insert=False)
        cells.pop(idx)
        summary = f"Deleted cell {idx}"
    else:
        if a.get("new_source") is None:
            raise ToolFailure("new_source is required to replace a cell")
        idx = _find_index(nb, a, insert=False)
        cell = cells[idx]
        cell["source"] = _split_source(str(a["new_source"]))
        if ctype and ctype != cell.get("cell_type"):
            cell["cell_type"] = ctype
            if ctype == "code":
                cell.setdefault("outputs", [])
                cell.setdefault("execution_count", None)
            else:
                cell.pop("outputs", None)
                cell.pop("execution_count", None)
        elif cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        summary = f"Replaced cell {idx}"
    _validate(nb)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return f"{summary} in {path} (now {len(cells)} cells)"


def _validate(nb: dict[str, Any]) -> None:
    try:
        import nbformat  # type: ignore
    except ImportError:
        return
    try:
        nbformat.validate(nbformat.from_dict(nb))
    except Exception as exc:  # noqa: BLE001
        raise ToolFailure(f"edit would produce an invalid notebook: {exc}") from None


def _notebook_edit(ctx: ToolContext, a: dict[str, Any]) -> str:
    from .builtin import _policy, _resolve  # shared path handling

    raw = a.get("notebook_path") or a.get("file_path")
    if not raw:
        raise ToolFailure("notebook_path is required")
    p = _resolve(ctx, raw, for_read=False)
    why = _policy(ctx).write_denial(p)
    if why:
        raise ToolFailure(why)
    if p.suffix.lower() != ".ipynb":
        raise ToolFailure(f"{p} is not a .ipynb notebook; use Edit for other files")
    out = edit_notebook(p, a)
    ctx.trace("file_write", path=ctx.run.rel(p), bytes=p.stat().st_size, edit=True,
              notebook_edit=(a.get("edit_mode") or "replace"))
    return out


def notebook_edit_tool() -> Tool:
    return Tool(
        "NotebookEdit",
        "Edit a Jupyter notebook (.ipynb) cell: edit_mode replace (default) | insert | delete. Identify the cell "
        "with cell_id or cell_number (0-based; insert places the new cell after cell_id or at cell_number). "
        "Outputs of a replaced code cell are cleared.",
        schema({"notebook_path": {"type": "string", "description": "Path to the .ipynb (relative paths resolve to "
                                                                    "your workspace)"},
                "cell_id": {"type": "string"}, "cell_number": {"type": "integer"},
                "new_source": {"type": "string"},
                "cell_type": {"type": "string", "enum": ["code", "markdown"]},
                "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"]}},
               ["notebook_path"]),
        _notebook_edit, blocking=True)


__all__ = ["render_notebook", "edit_notebook", "load_notebook", "notebook_edit_tool"]

"""Tool-surface fidelity: Read (text/images/PDF/notebooks), Glob, Grep, NotebookEdit,
UpdateMemory, path bases, and the registry surface. Offline."""

from __future__ import annotations

import base64
import json
import os
import struct
import time
import zlib
from pathlib import Path

import pytest

from vbt.agents import memory_path
from vbt.providers.base import DocumentPart, ImagePart, ProviderCapabilities, TextBlock
from vbt.providers.mock import ScriptedProvider
from vbt.runtime import Runtime
from vbt.session import Run
from vbt.tools.base import ToolContext, ToolFailure
from vbt.tools.builtin import builtin_tools, expand_braces, glob_to_regex
from vbt.tools.policy import PathPolicy

AGENT = "genomics-analyst"


def _runtime(config, tmp_path, caps=None):
    run = Run(tmp_path / "runs", config=config)
    rt = Runtime(config, run, provider=ScriptedProvider.from_rules({}, capabilities=caps))
    rt.read_roots = [(tmp_path / "ref").resolve()]
    (tmp_path / "ref").mkdir(exist_ok=True)
    return rt


@pytest.fixture
def rt(config, tmp_path):
    runtime = _runtime(config, tmp_path)
    yield runtime
    try:
        runtime.run.close()
    except Exception:  # noqa: BLE001
        pass


async def _call(rt, tool, agent=AGENT, tid="tu_1", **args):
    return await rt.registry.get(tool)(ToolContext(agent, rt.run, rt, tool_call_id=tid), args)


def _ws(rt, agent=AGENT) -> Path:
    return rt.workspace_for(agent)


def _png(w: int = 4, h: int = 3) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\0" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- registry

def test_builtin_surface_and_blocking_flags():
    tools = {t.name: t for t in builtin_tools()}
    for name in ("Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep", "Bash", "TodoWrite", "Skill",
                 "UpdateMemory", "WebFetch", "WebSearch"):
        assert name in tools, name
    for name in ("Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep", "Skill", "UpdateMemory"):
        assert tools[name].blocking, name
    assert not tools["Bash"].blocking
    assert "milliseconds" in tools["Bash"].input_schema["properties"]["timeout"]["description"]


# --------------------------------------------------------------------------- path bases

async def test_cso_reads_run_relative_and_specialists_read_other_workspaces(rt):
    report = _ws(rt) / "results" / "reports" / "x.md"
    report.write_text("genetics summary\n")
    assert "genetics summary" in await _call(rt, "Read", agent="cso",
                                             file_path=f"work/{AGENT}/results/reports/x.md")
    assert "genetics summary" in await _call(rt, "Read", agent="scientific-reviewer",
                                             file_path=f"work/{AGENT}/results/reports/x.md")
    other = rt.workspace_for("single-cell-analyst") / "x.txt"
    other.write_text("from sc\n")
    assert "from sc" in await _call(rt, "Read", file_path="work/single-cell-analyst/x.txt")
    # workspace-relative still works, and inputs/ (run-relative) is readable
    assert "genetics summary" in await _call(rt, "Read", file_path="results/reports/x.md")
    (rt.run.dir / "inputs" / "q.md").write_text("question\n")
    assert "question" in await _call(rt, "Read", file_path="inputs/q.md")
    assert not (rt.run.dir / "work" / "cso").exists()


def test_policy_resolve_rules(tmp_path):
    run = tmp_path / "run"
    ws = run / "work" / "a"
    (ws / "results").mkdir(parents=True)
    (run / "notes.md").write_text("x")
    pol = PathPolicy(run_dir=run, workspace=ws, agent="a")
    assert pol.resolve("work/b/x") == run / "work" / "b" / "x"
    assert pol.resolve("results/t.csv") == ws / "results" / "t.csv"
    assert pol.resolve("notes.md") == run / "notes.md"              # read fallback to the run dir
    assert pol.resolve("notes.md", for_read=False) == ws / "notes.md"
    assert pol.resolve("~/x") == Path(os.path.expanduser("~/x"))
    assert pol.own_dir == run / "work" / "a"
    assert PathPolicy(run_dir=run, workspace=run, agent="cso").own_dir == run / "work" / "_cso"


# --------------------------------------------------------------------------- Read

async def test_read_streams_with_offset_limit_and_long_lines(rt):
    f = _ws(rt) / "big.txt"
    with open(f, "w") as fh:
        for i in range(1, 5001):
            fh.write(f"line {i}\n")
        fh.write("y" * 300_000 + "\n")
        fh.write("last\n")
    out = await _call(rt, "Read", file_path="big.txt", offset=10, limit=3)
    assert out.splitlines()[0].endswith("line 10") and "continue with offset=13" in out
    tail = await _call(rt, "Read", file_path="big.txt", offset=5001)
    assert "line truncated" in tail and tail.rstrip().endswith("last")
    assert len(tail) < 5000


async def test_read_binary_and_empty_files(rt):
    (_ws(rt) / "x.h5ad").write_bytes(b"\x89HDF\0\0")
    assert "binary" in await _call(rt, "Read", file_path="x.h5ad")
    (_ws(rt) / "y.dat").write_bytes(b"abc\0def")
    assert "binary" in await _call(rt, "Read", file_path="y.dat")
    (_ws(rt) / "e.txt").write_text("")
    assert "empty" in await _call(rt, "Read", file_path="e.txt")
    with pytest.raises(ToolFailure, match="not found"):
        await _call(rt, "Read", file_path="nope.txt")


async def test_read_png_returns_image_part_when_supported(rt):
    (_ws(rt) / "results" / "figures" / "umap.png").write_bytes(_png())
    out = await _call(rt, "Read", file_path="results/figures/umap.png")
    assert isinstance(out, list) and isinstance(out[0], TextBlock) and isinstance(out[1], ImagePart)
    assert out[1].media_type == "image/png" and base64.b64decode(out[1].data_b64)[:4] == b"\x89PNG"


async def test_read_png_is_a_notice_without_image_support(config, tmp_path):
    rt = _runtime(config, tmp_path, caps=ProviderCapabilities(images=False, documents=False))
    (_ws(rt) / "fig.png").write_bytes(_png())
    out = await _call(rt, "Read", file_path="fig.png")
    assert isinstance(out, str) and "cannot view images" in out


def _pdf(path: Path, pages: int = 3) -> None:
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)


async def test_read_pdf_pages_document_part_or_text(config, tmp_path, rt):
    _pdf(_ws(rt) / "label.pdf", pages=3)
    out = await _call(rt, "Read", file_path="label.pdf", pages="1-2")
    assert isinstance(out[-1], DocumentPart)
    import pypdf
    import io
    assert len(pypdf.PdfReader(io.BytesIO(base64.b64decode(out[-1].data_b64))).pages) == 2
    text = await _call(rt, "Read", file_path="label.pdf")
    assert isinstance(text, str) and "--- page 1 ---" in text
    rt2 = _runtime(config, tmp_path / "b", caps=ProviderCapabilities(images=False, documents=False))
    _pdf(_ws(rt2) / "label.pdf", pages=2)
    text2 = await _call(rt2, "Read", file_path="label.pdf", pages="2")
    assert isinstance(text2, str) and "--- page 2 ---" in text2
    with pytest.raises(ToolFailure, match="at most 20"):
        await _call(rt, "Read", file_path="label.pdf", pages="1-25")


def _nb(path: Path) -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"language": "python"}},
          "cells": [{"id": "a1", "cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
                    {"id": "b2", "cell_type": "code", "metadata": {}, "execution_count": 1,
                     "source": ["print(1 + 1)\n"],
                     "outputs": [{"output_type": "stream", "name": "stdout", "text": ["2\n"]},
                                 {"output_type": "display_data", "metadata": {},
                                  "data": {"image/png": "AAA", "text/plain": ["<Figure>"]}}]}]}
    path.write_text(json.dumps(nb))


async def test_read_renders_notebooks(rt):
    _nb(_ws(rt) / "analysis.ipynb")
    out = await _call(rt, "Read", file_path="analysis.ipynb")
    assert "--- cell 0 (id a1) [markdown]" in out and "# Title" in out
    assert "--- cell 1 (id b2) [code] In[1]" in out and "print(1 + 1)" in out
    assert "[output]\n2" in out and "image/png output" in out


async def test_notebook_edit_replace_insert_delete(rt):
    p = _ws(rt) / "analysis.ipynb"
    _nb(p)
    out = await _call(rt, "NotebookEdit", notebook_path="analysis.ipynb", cell_id="b2", new_source="print(3)")
    assert "Replaced cell 1" in out
    nb = json.loads(p.read_text())
    assert "".join(nb["cells"][1]["source"]) == "print(3)" and nb["cells"][1]["outputs"] == []
    await _call(rt, "NotebookEdit", notebook_path="analysis.ipynb", cell_id="a1", new_source="x = 1",
                cell_type="code", edit_mode="insert")
    nb = json.loads(p.read_text())
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code"]
    assert "".join(nb["cells"][1]["source"]) == "x = 1" and nb["cells"][1]["id"]
    await _call(rt, "NotebookEdit", notebook_path="analysis.ipynb", cell_number=0, edit_mode="delete")
    nb = json.loads(p.read_text())
    assert len(nb["cells"]) == 2 and nb["cells"][0]["source"] == ["x = 1"]
    with pytest.raises(ToolFailure, match="out of range"):
        await _call(rt, "NotebookEdit", notebook_path="analysis.ipynb", cell_number=9, new_source="x")
    other = rt.workspace_for("single-cell-analyst") / "o.ipynb"
    _nb(other)
    with pytest.raises(ToolFailure, match="own workspace"):
        await _call(rt, "NotebookEdit", notebook_path=str(other), cell_number=0, new_source="x")
    ev = [e for e in rt.run.events() if e["type"] == "file_write" and e.get("notebook_edit")]
    assert ev and ev[0]["path"] == f"work/{AGENT}/analysis.ipynb"


# --------------------------------------------------------------------------- Glob

async def test_glob_absolute_braces_and_mtime_order(rt):
    t = _ws(rt) / "results" / "tables"
    (t / "a.csv").write_text("a")
    time.sleep(0.02)
    (t / "b.tsv").write_text("b")
    os.utime(t / "a.csv", (time.time() - 100, time.time() - 100))
    out = await _call(rt, "Glob", pattern=str(rt.run.dir / "work" / "*" / "results" / "tables" / "*.csv"))
    assert out.strip() == str(t / "a.csv")
    out = await _call(rt, "Glob", pattern="**/*.{csv,tsv}")
    assert out.splitlines() == [str(t / "b.tsv"), str(t / "a.csv")]
    assert await _call(rt, "Glob", pattern="*.nothing") == "(no matches)"
    assert str(t / "a.csv") in await _call(rt, "Glob", agent="cso", pattern="a.csv",
                                           path=f"work/{AGENT}/results/tables")


# --------------------------------------------------------------------------- Grep

def test_glob_translator():
    assert glob_to_regex("**/*.csv").match("a.csv") and glob_to_regex("**/*.csv").match("x/y/a.csv")
    assert glob_to_regex("*.{csv,tsv}").match("t.tsv") and not glob_to_regex("*.{csv,tsv}").match("t.txt")
    assert glob_to_regex("data/[ab]?.txt").match("data/a1.txt") and not glob_to_regex("*.csv").match("d/x.csv")
    assert expand_braces("a.{x,y{1,2}}") == ["a.x", "a.y1", "a.y2"]


async def test_grep_glob_type_context_and_limits(rt):
    t = _ws(rt) / "results" / "tables"
    (t / "l2g.csv").write_text("gene,score\nEGFR,0.9\nKRAS,0.2\n")
    (t / "de.tsv").write_text("gene\tlogfc\nEGFR\t2.1\n")
    (_ws(rt) / "notes.md").write_text("intro\nctx before\nEGFR is a target\nctx after\nend\n")
    (_ws(rt) / "blob.bin").write_bytes(b"EGFR\0\0binary")
    files = await _call(rt, "Grep", pattern="EGFR", glob="**/*.csv")
    assert files.strip() == str(t / "l2g.csv")
    files = await _call(rt, "Grep", pattern="EGFR", glob="*.{csv,tsv}")
    assert set(files.splitlines()) == {str(t / "l2g.csv"), str(t / "de.tsv")}
    assert "blob.bin" not in await _call(rt, "Grep", pattern="EGFR")
    assert (await _call(rt, "Grep", pattern="EGFR", type="md")).strip() == str(_ws(rt) / "notes.md")
    ctx = await _call(rt, "Grep", pattern="target", type="md", output_mode="content", **{"-C": 1})
    assert ctx.splitlines() == [f"{_ws(rt) / 'notes.md'}-2-ctx before", f"{_ws(rt) / 'notes.md'}:3:EGFR is a target",
                                f"{_ws(rt) / 'notes.md'}-4-ctx after"]
    cnt = await _call(rt, "Grep", pattern="EGFR|KRAS", glob="*.csv", output_mode="count")
    assert cnt.strip() == f"{t / 'l2g.csv'}:2"
    many = await _call(rt, "Grep", pattern="e", output_mode="content", head_limit=2)
    assert "output limited to 2" in many
    ml = await _call(rt, "Grep", pattern=r"before\nEGFR", multiline=True, output_mode="content")
    assert ":2:ctx before" in ml and ":3:EGFR" in ml
    ci = await _call(rt, "Grep", pattern="egfr", glob="*.md", **{"-i": True})
    assert "notes.md" in ci


# --------------------------------------------------------------------------- memory

async def test_update_memory_append_replace_and_trace(rt):
    await _call(rt, "UpdateMemory", content="L2G table at results/tables/l2g.csv")
    await _call(rt, "UpdateMemory", content="OT API rate-limits; use the local dump")
    p = memory_path(rt.run.dir, AGENT)
    assert p.read_text().splitlines() == ["L2G table at results/tables/l2g.csv",
                                         "OT API rate-limits; use the local dump"]
    await _call(rt, "UpdateMemory", content="condensed", mode="replace")
    assert p.read_text() == "condensed\n"
    ev = [e for e in rt.run.events() if e["type"] == "memory_write"]
    assert [e["mode"] for e in ev] == ["append", "append", "replace"]
    assert ev[0]["path"] == f"memory/{AGENT}/MEMORY.md" and ev[0]["agent"] == AGENT


async def test_write_and_edit_trace_file_write(rt):
    await _call(rt, "Write", file_path="code/scripts/a.py", content="x = 1\n")
    await _call(rt, "Edit", file_path="code/scripts/a.py", old_string="x = 1", new_string="x = 2")
    assert (_ws(rt) / "code" / "scripts" / "a.py").read_text() == "x = 2\n"
    ev = [e for e in rt.run.events() if e["type"] == "file_write"]
    assert [e["path"] for e in ev] == [f"work/{AGENT}/code/scripts/a.py"] * 2 and ev[1]["edit"]

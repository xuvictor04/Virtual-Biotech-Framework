"""Skill catalog: frontmatter index, shadowing, materialisation, name validation. Offline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vbt.providers.mock import ScriptedProvider
from vbt.runtime import Runtime
from vbt.session import Run
from vbt.tools.base import ToolContext, ToolFailure
from vbt.tools.builtin import builtin_tools
from vbt.tools.skills import SkillIndex, materialize, normalize_skill_name, parse_frontmatter, tree_sha256


def _skill(root: Path, name: str, desc: str, body: str = "Do the thing.", extra: dict | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\n{body}\n")
    for rel, text in (extra or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return d


@pytest.fixture
def roots(tmp_path):
    local, upstream = tmp_path / "skills", tmp_path / "upstream" / ".claude" / "skills"
    _skill(local, "run-organization", "Harness run layout (local override).")
    _skill(upstream, "run-organization", "Upstream run layout.")
    _skill(upstream, "single-cell-analysis", "Pseudobulk DE and GSEA for scRNA-seq.",
           extra={"references/methods.md": "methods"})
    return [local, upstream]


def test_frontmatter_parser_handles_folded_blocks():
    meta = parse_frontmatter("---\nname: x\ndescription: >\n  first line\n  second line\n---\nbody")
    assert meta["name"] == "x" and meta["description"].split() == ["first", "line", "second", "line"]
    assert parse_frontmatter("no frontmatter") == {}


def test_catalog_lists_names_and_descriptions_and_local_shadows_upstream(roots):
    idx = SkillIndex.build(roots)
    assert idx.names() == ["run-organization", "single-cell-analysis"]
    assert idx.get("run-organization").description == "Harness run layout (local override)."
    assert [s.root for s in idx.shadowed] == [roots[1].resolve()]
    text = idx.catalog_text()
    assert "- run-organization: Harness run layout (local override)." in text
    assert "- single-cell-analysis: Pseudobulk DE and GSEA for scRNA-seq." in text
    short = idx.catalog_text(max_chars=60)
    assert len(short) <= 60 and "more" in short


def test_skill_tool_description_carries_the_catalog(roots):
    desc = {t.name: t for t in builtin_tools(skill_roots=roots)}["Skill"].description
    assert "single-cell-analysis: Pseudobulk" in desc and "Upstream run layout" not in desc
    plain = {t.name: t for t in builtin_tools()}["Skill"].description
    assert "Available skills" not in plain


def test_upstream_catalog_parses(config):
    up = Path(config["vars"]["upstream"]) / ".claude" / "skills"
    if not up.is_dir():
        pytest.skip("upstream skills not available")
    idx = SkillIndex.build([Path("skills").resolve(), up])
    assert {"run-organization", "single-cell-analysis", "evidence-citation"} <= set(idx.names())
    assert all(idx.get(n).description for n in idx.names())


def test_materialize_links_skills_into_the_run(roots, tmp_path):
    run = tmp_path / "run"
    hashes = materialize(run, roots)
    assert set(hashes) == {"run-organization", "single-cell-analysis"}
    d = run / ".claude" / "skills"
    assert "local override" in (d / "run-organization" / "SKILL.md").read_text()
    assert (d / "single-cell-analysis" / "references" / "methods.md").read_text() == "methods"
    assert hashes["single-cell-analysis"] == tree_sha256(roots[1] / "single-cell-analysis")
    assert materialize(run, roots) == hashes          # idempotent
    copied = materialize(tmp_path / "run2", roots, copy=True)
    assert copied == hashes and not os.path.islink(tmp_path / "run2" / ".claude" / "skills" / "run-organization")


@pytest.mark.parametrize("raw,name", [("single-cell-analysis", "single-cell-analysis"),
                                      ("/single-cell-analysis", "single-cell-analysis"),
                                      (".claude/skills/single-cell-analysis", "single-cell-analysis"),
                                      (".claude/skills/single-cell-analysis/SKILL.md", "single-cell-analysis")])
def test_skill_name_forms(raw, name):
    assert normalize_skill_name(raw) == name


@pytest.mark.parametrize("raw", ["../secrets", "a/b", "..", "", "x y", "/etc/passwd"])
def test_invalid_skill_names_are_rejected(raw):
    with pytest.raises(ValueError):
        normalize_skill_name(raw)


async def test_skill_tool_loads_by_name_and_paths_resolve_in_the_run(config, roots, tmp_path):
    run = Run(tmp_path / "runs", config=config)
    rt = Runtime(config, run, provider=ScriptedProvider.from_rules({}))
    rt.skill_roots = roots
    rt.read_roots = [r.resolve() for r in roots]
    materialize(run.dir, roots)
    ctx = ToolContext("single-cell-analyst", run, rt, tool_call_id="s1")
    out = await rt.registry.get("Skill")(ctx, {"skill": "/single-cell-analysis"})
    assert "# Skill: single-cell-analysis" in out and str(run.dir / ".claude" / "skills") in out
    assert "references/methods.md" in out
    read = await rt.registry.get("Read")(ctx, {"file_path": ".claude/skills/single-cell-analysis/references/methods.md"})
    assert "methods" in read
    with pytest.raises(ToolFailure, match="invalid skill name"):
        await rt.registry.get("Skill")(ctx, {"skill": "../../etc"})
    with pytest.raises(ToolFailure, match="unknown skill"):
        await rt.registry.get("Skill")(ctx, {"skill": "nope"})
    assert [e["skill"] for e in run.events() if e["type"] == "skill"] == ["single-cell-analysis"]
    run.close()

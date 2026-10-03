"""Skill catalog: discovery, the advertised index and the run's ``.claude/skills``.

A skill is a directory with a ``SKILL.md`` whose YAML frontmatter carries
``name`` and ``description`` (the upstream ``.claude/skills`` layout). Roots
are searched in order and the first root that defines a name wins, so local
``skills/`` overrides shadow the upstream ones.

* :meth:`SkillIndex.build` parses the frontmatter of every root;
* :meth:`SkillIndex.catalog_text` is the ``name: description`` list put in the
  Skill tool description (capped, default 6000 chars);
* :func:`materialize` links (or copies) each skill into
  ``<run>/.claude/skills/<name>`` so the ``.claude/skills/...`` paths the
  upstream prompts use resolve inside a run, and returns the sha256 of each
  skill tree for the run manifest.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SKILL_FILE = "SKILL.md"
NAME_RE = re.compile(r"^[\w.-]+$")
_FRONT_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


def parse_frontmatter(text: str) -> dict[str, str]:
    """``name``/``description`` (and other scalar keys) from a SKILL.md frontmatter block.

    Uses PyYAML when installed, else a small ``key: value`` parser that also
    handles folded (``>``) and literal (``|``) blocks.
    """
    m = _FRONT_RE.match(text or "")
    if not m:
        return {}
    block = m.group(1)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(block)
        if isinstance(data, dict):
            return {str(k): ("" if v is None else str(v)).strip() for k, v in data.items()
                    if not isinstance(v, (dict, list))}
    except Exception:  # noqa: BLE001 - fall back to the simple parser
        pass
    out: dict[str, str] = {}
    key = None
    buf: list[str] = []
    for line in block.splitlines():
        km = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if km and not line.startswith((" ", "\t")):
            if key is not None:
                out[key] = " ".join(buf).strip()
            key, val = km.group(1), km.group(2).strip()
            buf = [] if val in (">", "|", ">-", "|-") else [val.strip("'\"")]
        elif key is not None:
            buf.append(line.strip())
    if key is not None:
        out[key] = " ".join(buf).strip()
    return out


@dataclass
class Skill:
    name: str
    description: str
    path: Path          # the skill directory (resolved)
    root: Path          # the root it came from

    @property
    def skill_md(self) -> Path:
        return self.path / SKILL_FILE


@dataclass
class SkillIndex:
    skills: dict[str, Skill] = field(default_factory=dict)
    shadowed: list[Skill] = field(default_factory=list)

    @classmethod
    def build(cls, roots: Iterable[str | os.PathLike] | None) -> "SkillIndex":
        idx = cls()
        for root in roots or []:
            if not root:
                continue
            r = Path(root).expanduser()
            try:
                r = r.resolve()
            except OSError:
                continue
            if not r.is_dir():
                continue
            for md in sorted(r.glob(f"*/{SKILL_FILE}")):
                d = md.parent
                try:
                    meta = parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    continue
                name = (meta.get("name") or d.name).strip()
                if not NAME_RE.match(name):
                    name = d.name
                if not NAME_RE.match(name):
                    continue
                sk = Skill(name, " ".join((meta.get("description") or "").split()), d.resolve(), r)
                if name in idx.skills:
                    idx.shadowed.append(sk)
                    continue
                idx.skills[name] = sk
                # also reachable by directory name when it differs from the frontmatter name
                if d.name != name and d.name not in idx.skills and NAME_RE.match(d.name):
                    idx.skills.setdefault(d.name, sk)
        return idx

    def names(self) -> list[str]:
        return sorted({s.name for s in self.skills.values()})

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def catalog_text(self, max_chars: int = 6000) -> str:
        """``- name: description`` lines, capped at ``max_chars`` (a final line names the rest)."""
        names = self.names()
        lines = [f"- {n}: {self.skills[n].description}" if self.skills[n].description else f"- {n}"
                 for n in names]
        text = "\n".join(lines)
        k = len(lines)
        while len(text) > max_chars and k > 0:
            k -= 1
            rest = names[k:]
            trailer = f"- ... {len(rest)} more: {', '.join(rest)}"
            if len(trailer) > max_chars // 2:
                trailer = f"- ... {len(rest)} more skills (load any by name)"
            text = "\n".join(lines[:k] + [trailer])
        return text[:max_chars]


def normalize_skill_name(raw: str) -> str:
    """Accept ``name``, ``/name``, ``.claude/skills/name`` (and a trailing ``/SKILL.md``);
    raise ValueError for anything that is not a plain skill name."""
    s = str(raw or "").strip().strip("'\"")
    s = s.replace("\\", "/")
    if s.endswith("/" + SKILL_FILE):
        s = s[: -len(SKILL_FILE) - 1]
    s = s.rstrip("/")
    for prefix in (".claude/skills/", "./.claude/skills/", "skills/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.lstrip("/")
    if not s or not NAME_RE.match(s) or s in (".", ".."):
        raise ValueError(f"invalid skill name {raw!r}; use a name like 'single-cell-analysis'")
    return s


def tree_sha256(path: Path) -> str:
    """sha256 over the relative paths and contents of every file in a skill tree."""
    h = hashlib.sha256()
    base = Path(path)
    files = sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for p in files:
        rel = p.relative_to(base).as_posix()
        h.update(rel.encode() + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def materialize(run_dir: str | os.PathLike, roots: Iterable[str | os.PathLike] | None, *,
                copy: bool = False) -> dict[str, str]:
    """Expose every skill under ``<run>/.claude/skills/<name>``.

    Symlinks the skill directories (copies when symlinks are unavailable or
    ``copy=True``). Returns ``{name: sha256 of the skill tree}``.
    """
    idx = SkillIndex.build(roots)
    dest_root = Path(run_dir) / ".claude" / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name in idx.names():
        sk = idx.skills[name]
        dest = dest_root / name
        if dest.is_symlink() or dest.exists():
            if dest.is_symlink() and Path(os.path.realpath(dest)) == sk.path:
                out[name] = tree_sha256(sk.path)
                continue
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        linked = False
        if not copy:
            try:
                os.symlink(sk.path, dest, target_is_directory=True)
                linked = True
            except (OSError, NotImplementedError):
                linked = False
        if not linked:
            shutil.copytree(sk.path, dest, ignore=shutil.ignore_patterns("__pycache__"))
        out[name] = tree_sha256(sk.path)
    return out


__all__ = ["Skill", "SkillIndex", "parse_frontmatter", "normalize_skill_name", "materialize", "tree_sha256",
           "NAME_RE", "SKILL_FILE"]

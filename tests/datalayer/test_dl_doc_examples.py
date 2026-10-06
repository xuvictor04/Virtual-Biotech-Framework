"""Every YAML example in docs/DATA_LAYER.md validates against the models and lints clean (§5, §21).

Examples are excerpts, so they are wrapped minimally before linting:

* ``{…}`` placeholders (elided fields) become one ``payload`` field;
* table-level excerpts join the source named in their first comment line
  (``# configs/data/sources/<source>.yaml``), source-level blocks merge by ``source``;
* id_types declared in comments (``# id_types: name {...}``) are parsed and validated too;
* what an excerpt references but does not show (a table, a column, an id_type of the elided
  part of the file) is stubbed from the lint findings, one minimal declaration each.

Overlay examples are linted against the doc's own descriptors (stubbing only what no example
declares). Nothing else is relaxed: a facet typo, a wrong role facet, a reference that the
scoping rules cannot resolve inside a declared table, or a broken binding fails the test.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from vbt.datalayer.descriptor.lint import errors, lint_descriptor, lint_overlay, split_qualified
from vbt.datalayer.descriptor.models import SourceDescriptor
from vbt.datalayer.descriptor.overlay import Overlay
from vbt.datalayer.roles import parse_path
from vbt.datalayer.settings import DATA_DEFAULTS, DataSettings

DOC = Path(__file__).resolve().parents[2] / "docs" / "DATA_LAYER.md"
_FENCE = re.compile(r"^```yaml\n(.*?)^```", re.S | re.M)
STUB_ROUNDS = 12


def _blocks() -> list[tuple[int, str]]:
    text = DOC.read_text(encoding="utf-8")
    return [(text.count("\n", 0, m.start()) + 1, m.group(1)) for m in _FENCE.finditer(text)]


def _fill_elisions(obj: Any, parent: str | None = None) -> Any:
    """``{…}`` -> one payload field (inside ``fields``/``columns``) or a payload column."""
    if isinstance(obj, dict):
        if set(obj) == {"…"} and obj["…"] is None:
            return {"_elided": {"role": "payload"}} if parent in ("fields", "columns") else {"role": "payload"}
        return {k: _fill_elisions(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_elisions(v, parent) for v in obj]
    return obj


def _comment_id_types(text: str) -> dict[str, dict[str, Any]]:
    """id_types declared in comments: ``# id_types: name {flow mapping}`` (continued on ``#`` lines)."""
    out: dict[str, dict[str, Any]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("# id_types:"):
            i += 1
            continue
        chunk = line[len("# id_types:"):]
        i += 1
        while i < len(lines) and lines[i].strip().startswith("#"):
            cont = lines[i].strip()[1:]
            if chunk.count("{") == chunk.count("}") and not re.match(r"^\s*\w+\s*\{", cont):
                break
            chunk += " " + cont.strip()
            i += 1
        pos = 0
        while True:
            m = re.compile(r"\s*(\w+)\s*\{").match(chunk, pos)
            if not m:
                break
            depth, j = 0, m.end() - 1
            for j in range(m.end() - 1, len(chunk)):
                depth += {"{": 1, "}": -1}.get(chunk[j], 0)
                if depth == 0:
                    break
            out[m.group(1)] = yaml.safe_load(chunk[m.end() - 1: j + 1])
            pos = j + 1
    return out


def _source_name(first_line: str) -> str:
    m = re.search(r"(\w+)\.yaml", first_line)
    return m.group(1) if m else "doc_excerpt"


def _new_source(name: str) -> dict[str, Any]:
    return {"schema": "vbt.datasource/1", "source": name, "title": f"{name} (doc excerpt)",
            "release": {"from": "literal"}, "id_types": {}, "tables": {}}


def _collect() -> tuple[dict[str, dict[str, Any]], list[tuple[int, dict[str, Any]]], list[tuple[int, dict]]]:
    """``(descriptor dicts by source, overlay dicts, other blocks)``."""
    sources: dict[str, dict[str, Any]] = {}
    overlays: list[tuple[int, dict[str, Any]]] = []
    other: list[tuple[int, dict[str, Any]]] = []
    for lineno, text in _blocks():
        data = _fill_elisions(yaml.safe_load(text))
        assert isinstance(data, dict), f"line {lineno}: a YAML example is a mapping"
        first = text.splitlines()[0]
        if data.get("schema") == "vbt.datasource/1":
            name = data["source"]
            data.setdefault("title", f"{name} (doc excerpt)")
            if name in sources:
                sources[name]["tables"].update(data.get("tables") or {})
                sources[name]["id_types"].update(data.get("id_types") or {})
            else:
                data.setdefault("id_types", {})
                sources[name] = data
            target = name
        elif data.get("schema") == "vbt.overlay/1":
            overlays.append((lineno, data))
            continue
        elif "data" in data and len(data) == 1:
            other.append((lineno, data))
            continue
        elif all(isinstance(v, dict) and "kind" in v and "grain" in v for v in data.values()):
            target = _source_name(first)
            sources.setdefault(target, _new_source(target))["tables"].update(data)
        elif all(isinstance(v, dict) and ({"args", "result", "reads"} & set(v)) for v in data.values()):
            overlays.append((lineno, {"schema": "vbt.overlay/1", "server": f"doc_excerpt_{lineno}", "tools": data}))
            continue
        else:
            raise AssertionError(f"line {lineno}: unrecognised YAML example (keys {sorted(data)[:5]})")
        for name, spec in _comment_id_types(text).items():
            sources[target]["id_types"].setdefault(name, spec)
    return sources, overlays, other


# ---------------------------------------------------------------------------
# Stubs for what an excerpt references but does not show
# ---------------------------------------------------------------------------

def _stub_column(table: dict[str, Any], tables: dict[str, Any], path: str, leaf: dict[str, Any] | None = None) -> None:
    """Declare ``path`` (nested containers on the way) in ``table`` (an item table's container fields)."""
    fields = _fields_of(table, tables)
    segs = [s for s in parse_path(path).segments if s.name]
    for i, seg in enumerate(segs):
        last = i == len(segs) - 1
        if seg.name not in fields:
            if last:
                fields[seg.name] = dict(leaf or {"role": "identifier"})
            else:
                fields[seg.name] = {"role": "nested", "fields": {}}
                if seg.is_list:
                    fields[seg.name]["item_key"] = {"identity": "position"}
        col = fields[seg.name]
        if not last:
            if col.get("role") not in ("nested", "member"):
                return
            fields = col.setdefault("fields", {})


def _fields_of(table: dict[str, Any], tables: dict[str, Any]) -> dict[str, Any]:
    items = table.get("items_of")
    if not items:
        return table.setdefault("columns", {})
    parent = tables.setdefault(items["table"], {"kind": "entity", "grain": "stub", "key": {"columns": ["id"]},
                                               "columns": {"id": {"role": "identifier", "self": True}}})
    fields = _fields_of(parent, tables)
    for seg in [s for s in parse_path(items["path"]).segments if s.name]:
        col = fields.setdefault(seg.name, {"role": "nested", "fields": {}})
        if seg.is_list and col.get("role") == "nested" and "item_key" not in col:
            col["item_key"] = {"identity": "position"}
        fields = col.setdefault("fields", {})
    return fields


def _stub_table(tables: dict[str, Any], name: str, column: str | None = None) -> dict[str, Any]:
    if name not in tables:
        key = [s.name for s in parse_path(column).segments][0] if column else "id"
        tables[name] = {"kind": "entity", "grain": "stub", "key": {"columns": [key]},
                        "columns": {key: {"role": "identifier"}}}
    return tables[name]


def _stub_ref(sources: dict[str, dict[str, Any]], home: str, target: str) -> bool:
    """Stub ``table``, ``table.col...`` or ``source.table.col...`` relative to source ``home``."""
    segs = [s.name for s in parse_path(target).segments]
    src = home
    if len(segs) >= 3 and segs[0] in sources and segs[0] not in sources[home]["tables"]:
        src, segs = segs[0], segs[1:]
        target = target[len(src) + 1:]
    tables = sources[src]["tables"]
    table = segs[0]
    column = target[len(table) + 1:] if len(segs) > 1 else None
    before = copy.deepcopy(tables.get(table))
    t = _stub_table(tables, table, column)
    if column:
        _stub_column(t, tables, column)
    return tables.get(table) != before


def _apply_descriptor_stub(sources: dict[str, dict[str, Any]], f: Any) -> bool:
    home = f.where.split(".", 1)[0]
    if f.rule == "id_type":
        src, bare = split_qualified(f.target)
        src = src or home
        sources.setdefault(src, _new_source(src))
        if bare in sources[src]["id_types"]:
            return False
        others = [(name, s["id_types"][bare]) for name, s in sorted(sources.items()) if bare in s["id_types"]]
        sources[src]["id_types"][bare] = _mirror(*others[0]) if others else {"plugin": bare}
        return True
    if f.target is None:
        return False
    tables = sources[home]["tables"]
    if f.where.endswith(".items_of"):
        _fields_of(tables[f.table.split(".", 1)[1]], tables)    # the parent table and its container chain
        return True
    if f.table is not None:
        # a name looked up inside a declared table: declare it at the level it was looked up from
        tname = f.table.split(".", 1)[1]
        if tname in tables:
            container = f.scope
            before = copy.deepcopy(tables)
            _stub_column(tables[tname], tables, f"{container}.{f.target}" if container else f.target)
            return tables != before
    return _stub_ref(sources, home, f.target)


def _mirror(source: str, spec: dict[str, Any]) -> dict[str, Any]:
    """A stub of an id_type another example declares under the same bare name: the same identity
    (plugin, universe, label_of, union) with every name qualified, so the two are interchangeable."""
    def q(name: str) -> str:
        return name if ":" in name else f"{source}:{name}"

    out: dict[str, Any] = {"plugin": spec["plugin"]}
    if spec.get("label_of"):
        out["label_of"] = q(spec["label_of"])
    if spec.get("union"):
        out["union"] = [q(u) for u in spec["union"]]
    universe = spec.get("universe")
    if isinstance(universe, str):
        out["universe"] = f"{source}.{universe}"
    elif isinstance(universe, dict):
        out["universe"] = dict(universe, table=universe["table"] if "." in universe["table"]
                               else f"{source}.{universe['table']}")
    return out


def _validated(sources: dict[str, dict[str, Any]]) -> dict[str, SourceDescriptor]:
    return {name: SourceDescriptor.model_validate(d) for name, d in sources.items()}


def _lint_descriptors(sources: dict[str, dict[str, Any]]) -> tuple[dict[str, SourceDescriptor], list[Any]]:
    for _ in range(STUB_ROUNDS):
        models = _validated(sources)
        errs = [f for d in models.values() for f in errors(lint_descriptor(d, None, None, models))]
        if not errs:
            return models, []
        changed = False
        for f in errs:
            changed |= _apply_descriptor_stub(sources, f)
        if not changed:
            return models, errs
    models = _validated(sources)
    return models, [f for d in models.values() for f in errors(lint_descriptor(d, None, None, models))]


def _arg_for(overlay: dict[str, Any], target: str) -> dict[str, Any] | None:
    for tool in overlay.get("tools", {}).values():
        for arg in (tool.get("args") or {}).values():
            binds = arg.get("binds")
            if binds == target or (isinstance(binds, str) and target.startswith(binds)):
                return arg
    return None


def _link_stub_kind(sources: dict[str, dict[str, Any]], overlay: dict[str, Any], f: Any,
                    stubbed: set[tuple[str, str]]) -> bool:
    """An accepted kind that cannot reach a *stubbed* bound kind: the elided descriptor would declare
    the crosswalk, so add a maps_to edge. A real (non-stub) bound kind keeps the error."""
    tool = f.where.split(".tools.", 1)[1].split(".args.", 1)[0]
    argname = f.where.split(".args.", 1)[1].split(".", 1)[0]
    binds = overlay["tools"][tool]["args"][argname]["binds"]
    src, table = binds.split(".")[:2]
    column = binds[len(src) + len(table) + 2:]
    col = sources[src]["tables"][table]["columns"].get(column.split(".")[0], {})
    kind = col.get("id_type")
    if not kind or (src, kind) not in stubbed:
        return False
    raw = f.target
    if ":" not in raw:
        owners = sorted(s for s, d in sources.items() if raw in d["id_types"])
        raw = f"{owners[0]}:{raw}" if owners else raw
    edges = sources[src]["id_types"][kind].setdefault("maps_to", [])
    if any(e["id_type"] == raw for e in edges):
        return False
    edges.append({"id_type": raw})
    return True


def _apply_overlay_stub(sources: dict[str, dict[str, Any]], overlay: dict[str, Any], f: Any,
                        stubbed: set[tuple[str, str]]) -> bool:
    if f.target is None:
        return False
    if f.rule == "accepts" and "does not reach" in f.message:
        return _link_stub_kind(sources, overlay, f, stubbed)
    if f.rule == "accepts":
        src = f.target.split(":")[0] if ":" in f.target else None
        if src is None:
            return False
        sources.setdefault(src, _new_source(src))
        bare = f.target.split(":", 1)[1]
        if bare in sources[src]["id_types"]:
            return False
        sources[src]["id_types"][bare] = {"plugin": bare}
        stubbed.add((src, bare))
        return True
    segs = [s.name for s in parse_path(f.target).segments]
    src = segs[0]
    if src not in sources:
        sources[src] = _new_source(src)
    if len(segs) < 2:
        return False
    tables = sources[src]["tables"]
    table = segs[1]
    column = f.target[len(src) + len(table) + 2:] if len(segs) > 2 else None
    before = copy.deepcopy(tables.get(table))
    t = _stub_table(tables, table, column)
    if column:
        arg = _arg_for(overlay, f.target)
        leaf: dict[str, Any] = {"role": "category", "vocab": "data"}
        if arg and arg.get("accepts"):
            first = arg["accepts"][0]
            kind = first.split(":", 1)[1] if ":" in first else first
            leaf = {"role": "identifier", "id_type": kind}
            ids = sources[src]["id_types"]
            if kind not in ids:
                ids[kind] = {"plugin": kind}
                stubbed.add((src, kind))
            for other in arg["accepts"][1:]:
                if ":" not in other and other not in ids and not any(other in s["id_types"] for s in sources.values()):
                    ids[other] = {"plugin": other, "label_of": kind}
                    stubbed.add((src, other))
        _stub_column(t, tables, column, leaf)
    return tables.get(table) != before


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_doc_has_the_examples():
    blocks = _blocks()
    assert len(blocks) >= 13
    assert any("vbt.overlay/1" in b for _, b in blocks) and any("vbt.datasource/1" in b for _, b in blocks)


def test_comment_id_types_parse():
    text = ("# id_types: cbio_study  {plugin: cbio_study, universe_via: {tool: t.s, path: \"$.x\", ttl_s: 1}}\n"
            "#           cbio_sample {plugin: cbio_sample, universe: {table: sample, keys: [a, b]}}\n"
            "#   No upstream tool lists a study's samples.\n")
    parsed = _comment_id_types(text)
    assert set(parsed) == {"cbio_study", "cbio_sample"}
    assert parsed["cbio_sample"]["universe"]["keys"] == ["a", "b"]


def test_descriptor_examples_validate_and_lint_clean():
    sources, _, _ = _collect()
    assert {"open_targets", "tahoe_100m", "depmap", "cellxgene_census", "zenodo"} <= set(sources)
    models, errs = _lint_descriptors(sources)
    assert not errs, "doc descriptor examples fail lint:\n" + "\n".join(map(str, errs))
    # the stubs only add declarations; the shipped examples' own tables all survive
    assert {"target", "target_go", "known_drug", "disease", "drug_molecule", "evidence", "interaction",
            "literature_vector", "hallmark"} <= set(models["open_targets"].tables)


def test_overlay_examples_validate_and_lint_clean():
    sources, overlays, _ = _collect()
    models, errs = _lint_descriptors(sources)
    assert not errs
    assert len(overlays) >= 4
    stubbed: set[tuple[str, str]] = set()
    for lineno, raw in overlays:
        ov = Overlay.model_validate(raw)
        remaining: list[Any] = []
        for _ in range(STUB_ROUNDS):
            catalog = _validated(sources)
            remaining = errors(lint_overlay(ov, catalog))
            if not remaining:
                break
            changed = False
            for f in remaining:
                changed |= _apply_overlay_stub(sources, raw, f, stubbed)
            if not changed:
                break
        assert not remaining, f"overlay example at line {lineno} fails lint:\n" + "\n".join(map(str, remaining))
    # the stubbed sources still lint clean as descriptors
    models = _validated(sources)
    errs = [f for d in models.values() for f in errors(lint_descriptor(d, None, None, models))]
    assert not errs, "\n".join(map(str, errs))


def test_config_and_pinned_examples():
    _, _, other = _collect()
    config = [d for _, d in other if "gateway" in d["data"]]
    pinned = [d for _, d in other if "catalog_sha256" in d["data"]]
    assert len(config) == 1 and len(pinned) == 1
    block = config[0]["data"]
    assert block == DATA_DEFAULTS, "§17 and settings.DATA_DEFAULTS disagree"
    settings = DataSettings.from_config({"data": block})
    assert settings.gateway.mode == "enforce" and settings.witness.max_scan_bytes == 2_000_000_000
    assert settings.memory.object_overhead_bytes["nested_item"] == 120
    assert {"mode", "profile", "descriptors", "overlays", "plugins", "sources", "determinism"} <= set(pinned[0]["data"])


@pytest.mark.parametrize("snippet", [
    "{role: label, of: id, scale: [0, 1]}",                    # a measure facet on a label
    "{role: identifier, id_type: x, selff: true}",             # a typo
    "{role: measure, statistic: numeric, vocab: data}",        # a category facet on a measure
])
def test_wrapper_does_not_relax_facets(snippet):
    sources = {"s": _new_source("s")}
    sources["s"]["tables"]["t"] = {"kind": "entity", "grain": "g", "key": {"columns": ["id"]},
                                   "columns": {"id": {"role": "identifier", "self": True},
                                               "x": yaml.safe_load(snippet)}}
    with pytest.raises(Exception):
        _validated(sources)

"""Derived argument schemas and agent-facing text for every bridged tool (§10.1, §10.2, F13).

Each of the 103 bridged tools (the upstream servers' ``register_tool`` calls and the two PubMed tools,
parsed, never imported) has a golden snapshot ``tests/datalayer/golden/<server>.<tool>.json``: the
tool's serve mode, the annotated schema and the derived description, built from the upstream
signature (types, defaults, docstring) and the shipped descriptors and overlays. A changed derivation
fails here until the snapshot is regenerated with ``VBT_UPDATE_GOLDEN=1`` and the diff reviewed
(``golden/README.md``). Also:

* every description fits ``data.derive.description_max_chars`` and keeps no upstream ``Returns:`` or
  ``Example:`` section;
* the description equals the ``derived text`` lines of ``vbt ds explain`` for the same docstring and
  schema (explain is the snapshot source);
* generic (unbound) tools are returned unchanged;
* the role sentences of §10.2 appear where the roles call for them: item grains, levels, cutoffs and
  their inclusivity, multiple-testing families, censoring, propagation (with a measured fraction),
  lossy projections, evidence nature and default filters that are not applied;
* the data child's public verbs are listed with native schemas (under ``request`` while the child
  registers one argument per verb).
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import PUBMED_SERVER, REPO, upstream_root

GOLDEN = Path(__file__).resolve().parent / "golden"
SERVERS_DIR = upstream_root() / "src" / "mcp_servers"
UPDATE = os.environ.get("VBT_UPDATE_GOLDEN") == "1"
MAX_CHARS = 1200


# ---------------------------------------------------------------------------- upstream signatures


def _type_schema(node: ast.expr | None) -> dict[str, Any]:
    """A JSON schema of a parameter annotation (the subset FastMCP derives for these signatures)."""
    if node is None:
        return {}
    text = ast.unparse(node).replace("typing.", "")
    if text.startswith("Optional[") and text.endswith("]"):
        return _type_schema(ast.parse(text[len("Optional["):-1], mode="eval").body)
    if " | None" in text:
        return _type_schema(ast.parse(text.replace(" | None", ""), mode="eval").body)
    base = text.split("[")[0]
    simple = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
              "Dict": "object", "Any": None}
    if base in simple:
        return {"type": simple[base]} if simple[base] else {}
    if base in ("list", "List", "Sequence", "tuple", "Tuple"):
        inner = text[len(base) + 1:-1] if "[" in text else ""
        items = _type_schema(ast.parse(inner, mode="eval").body) if inner else {}
        return {"type": "array", "items": items} if items else {"type": "array"}
    return {}


def _function_schema(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[dict[str, Any], str]:
    args = fn.args.args
    defaults = [None] * (len(args) - len(fn.args.defaults)) + list(fn.args.defaults)
    props: dict[str, Any] = {}
    required = []
    for a, d in zip(args, defaults):
        if a.arg in ("self", "ctx"):
            continue
        prop = _type_schema(a.annotation)
        if d is None:
            required.append(a.arg)
        else:
            try:
                prop["default"] = ast.literal_eval(d)
            except ValueError:
                pass
        props[a.arg] = prop
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema, ast.get_docstring(fn) or ""


def registered_tools() -> dict[tuple[str, str], tuple[dict[str, Any], str]]:
    """``{(server, tool): (input schema, docstring)}`` of every bridged tool, parsed (never imported)."""
    out: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    if not SERVERS_DIR.is_dir():
        return out
    for server_dir in sorted(p for p in SERVERS_DIR.iterdir() if (p / "server.py").is_file()):
        if server_dir.name == "provenance_mcp":
            continue
        tree = ast.parse((server_dir / "server.py").read_text(encoding="utf-8"))
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.mcp_servers"):
                for alias in node.names:
                    imports[alias.asname or alias.name] = (node.module, alias.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_tool":
                module, name = imports[node.args[1].id]    # type: ignore[attr-defined]
                path = upstream_root() / (module.replace(".", "/") + ".py")
                fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
                out[(server_dir.name.removesuffix("_mcp"), name)] = _function_schema(fn)
    tree = ast.parse(PUBMED_SERVER.read_text(encoding="utf-8"))
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "tool" for d in fn.decorator_list):
            out[("pubmed", fn.name)] = _function_schema(fn)
    return out


TOOLS = registered_tools()
needs_upstream = pytest.mark.skipif(not TOOLS, reason="the upstream checkout is missing (git submodule update --init)")


# ---------------------------------------------------------------------------- the catalog


@pytest.fixture(scope="module")
def shipped() -> tuple[Any, Any]:
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.load import load_descriptors, load_overlays
    from vbt.datalayer.plugins.registry import discover

    registry = discover(entry_points=False)
    variables = {"project_root": str(REPO), "upstream_commit": ""}
    descriptors = load_descriptors(REPO / "configs" / "data" / "sources", variables)
    overlays, generic = load_overlays(REPO / "configs" / "data" / "overlays", variables)
    return Catalog(descriptors, overlays, generic, registry=registry), registry


def snapshot(catalog: Any, registry: Any, server: str, tool: str, schema: dict[str, Any], doc: str) -> dict[str, Any]:
    from vbt.datalayer.derive import annotate_schema, describe_tool

    contract = catalog.contract(server, tool)
    b = contract.binding
    return {"tool": f"{server}.{tool}", "serve": b.serve if b is not None else "generic",
            "schema": annotate_schema(contract, schema, catalog=catalog, registry=registry),
            "description": describe_tool(contract, doc, catalog=catalog, max_chars=MAX_CHARS)}


def golden_path(server: str, tool: str) -> Path:
    return GOLDEN / f"{server}.{tool}.json"


@needs_upstream
@pytest.mark.parametrize("server,tool", sorted(TOOLS), ids=[f"{s}.{t}" for s, t in sorted(TOOLS)])
def test_golden_snapshot(shipped, server: str, tool: str) -> None:
    catalog, registry = shipped
    schema, doc = TOOLS[(server, tool)]
    got = snapshot(catalog, registry, server, tool, copy.deepcopy(schema), doc)
    path = golden_path(server, tool)
    text = json.dumps(got, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.is_file(), f"no golden snapshot for {server}.{tool}: run with VBT_UPDATE_GOLDEN=1 and review it"
    assert json.loads(path.read_text(encoding="utf-8")) == got, \
        f"{server}.{tool}: the derivation changed; regenerate with VBT_UPDATE_GOLDEN=1 and review the diff"


@needs_upstream
def test_every_bridged_tool_has_one_snapshot_and_none_is_stale() -> None:
    assert len(TOOLS) == 103
    names = {p.name for p in GOLDEN.glob("*.json")}
    want = {golden_path(s, t).name for s, t in TOOLS}
    assert want <= names, sorted(want - names)
    assert not names - want, f"stale snapshots: {sorted(names - want)}"


@needs_upstream
def test_descriptions_fit_the_cap_and_drop_returns_sections(shipped) -> None:
    catalog, registry = shipped
    for (server, tool), (schema, doc) in TOOLS.items():
        text = snapshot(catalog, registry, server, tool, copy.deepcopy(schema), doc)["description"]
        assert len(text) <= MAX_CHARS, f"{server}.{tool}: {len(text)} chars"
        assert "Returns:" not in text and "Example:" not in text, f"{server}.{tool}"
        assert text.strip(), f"{server}.{tool} has no description"


@needs_upstream
def test_explain_is_the_snapshot_source(shipped) -> None:
    from vbt.datalayer.cli import explain_tool

    catalog, registry = shipped
    for key in [("drug", "search_known_drugs"), ("association", "compare_direct_indirect"),
                ("target", "get_comprehensive_target_profile"), ("pathway", "get_gene_ontology")]:
        schema, doc = TOOLS[key]
        lines = explain_tool(catalog, registry, *key, description=doc, schema=copy.deepcopy(schema))
        start = lines.index("  derived text:") + 1
        text = "\n".join(ln[4:] for ln in lines[start:] if ln.startswith("    "))
        assert text == snapshot(catalog, registry, *key, copy.deepcopy(schema), doc)["description"]


def test_generic_tools_untouched(shipped) -> None:
    from vbt.datalayer.derive import annotate_schema, describe_tool

    catalog, registry = shipped
    c = catalog.contract("someone_elses_server", "tool")
    schema = {"type": "object", "properties": {"q": {"type": "string", "pattern": "x"}}}
    assert annotate_schema(c, copy.deepcopy(schema), catalog=catalog, registry=registry) == schema
    assert describe_tool(c, "Upstream text. Returns: x", catalog=catalog) == "Upstream text. Returns: x"


# ---------------------------------------------------------------------------- role sentences


def _text(catalog: Any, server: str, tool: str, **kw: Any) -> str:
    from vbt.datalayer.derive import describe_tool

    return describe_tool(catalog.contract(server, tool), "Upstream first sentence.", catalog=catalog,
                         max_chars=4000, **kw)


def test_role_sentences_on_the_shipped_bindings(shipped) -> None:
    catalog, _ = shipped
    go = _text(catalog, "pathway", "get_gene_ontology")
    assert "one row = one item" in go and "`_vbt.total` counts items" in go
    assert "direct annotations only" in go or "propagat" in go
    kd = _text(catalog, "drug", "search_known_drugs")
    assert "Evidence:" in kd or "Empty vs not found" in kd
    assert "excluded and counted in `_vbt.excluded_unknown`" in kd


def _custom_catalog() -> Any:
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.models import SourceDescriptor
    from vbt.datalayer.descriptor.overlay import Overlay
    from vbt.datalayer.plugins.registry import discover

    desc = {
        "schema": "vbt.datasource/1", "source": "s", "title": "S", "release": {"expect": "1", "from": "literal"},
        "defaults": {"format": "parquet", "layout": "single_file"},
        "id_types": {"gene": {"plugin": "ensembl_gene", "universe": "de.gene"}},
        "tables": {
            "de": {"kind": "fact", "path": "de.parquet", "grain": "one gene in one contrast",
                   "key": {"columns": ["contrast", "gene"]},
                   "evidence_nature": {"kind": "co_occurrence", "caveat": "literature co-occurrence, not function"},
                   "coverage": {"statement": "genes tested", "absence_means": "censored",
                                "censor": {"column": "padj", "op": "<=", "value": 0.1}},
                   "columns": {
                       "contrast": {"role": "category", "vocab": ["a", "b"]},
                       "gene": {"role": "identifier", "id_type": "gene", "self": True},
                       "padj": {"role": "measure", "scale": [0, 1], "family": ["contrast"]},
                       "effect": {"role": "measure", "cutoff": {"value": -0.5, "op": "le", "meaning": "dependent"}},
                       "os_months": {"role": "measure", "level": "patient"},
                       "os_status": {"role": "flag", "event_of": "os_months"},
                       "top_level": {"role": "category", "projection_of": "pathways", "lossy": True},
                       "duplicate": {"role": "qualifier", "effect": "duplicate", "default_filter": True},
                       "terms": {"role": "member", "item_key": {"identity": "value"},
                                 "membership": {"set": {"parent": "gene"}, "member": {"path": "[]"},
                                                "propagation": "mixed"}},
                   }},
        },
    }
    ov = {"schema": "vbt.overlay/1", "server": "x", "sources": ["s"],
          "tools": {"t": {"reads": {"s.de": {"access": "full_table"}},
                          "args": {"gene": {"binds": "s.de.gene", "accepts": ["gene"]},
                                   "max_padj": {"binds": "s.de.padj", "op": "le"}},
                          "result": {"rows": "$.rows", "kind": "file"}}}}
    return Catalog({"s": SourceDescriptor.model_validate(desc)}, {"x": Overlay.model_validate(ov)},
                   registry=discover(entry_points=False))


def test_every_role_sentence_of_section_10_2() -> None:
    from vbt.datalayer.derive import describe_tool

    catalog = _custom_catalog()
    c = catalog.contract("x", "t")
    text = describe_tool(c, "Upstream.", catalog=catalog, max_chars=4000, measured={"s.de.terms": 0.09})
    assert "Cutoff: dependent = effect ≤ -0.5, inclusive." in text
    assert "padj is adjusted within one contrast family." in text
    assert "os_months is right-censored when os_status is false; medians need Kaplan-Meier." in text
    assert "os_months is a patient value repeated on each row; count patients, not rows." in text
    assert "top_level keeps one value of pathways; records under several are missed." in text
    assert "duplicate: NOT applied by this tool; such rows are included." in text
    assert "terms: mixed: 9% of items also list an ancestor." in text
    assert "Evidence: literature co-occurrence, not function." in text
    assert "Missing rows are censored (padj <= 0.1): not significant or not tested." in text
    assert "Rows with unknown padj are excluded and counted" in text
    short = describe_tool(c, "Upstream.", catalog=catalog, max_chars=200)
    assert len(short) <= 200 and short.endswith("…")


# ---------------------------------------------------------------------------- native tools


def test_native_tools_are_listed_with_derived_schemas(shipped) -> None:
    from vbt.datalayer.derive import annotate_schema, describe_tool
    from vbt.datalayer.derive.tools import NATIVE_VERBS, native_tool_names, native_tools

    catalog, registry = shipped
    tools = {t.verb: t for t in native_tools(catalog)}
    assert tuple(tools) == NATIVE_VERBS
    find = tools["find"]
    assert find.name == "mcp__data__find" and "open_targets.known_drug" in find.tables
    where = find.input_schema["properties"]["where"]["x-vbt-where"]["open_targets.known_drug"]
    assert where["targetId"]["x-vbt-id-type"] == "open_targets:ensembl_gene"
    assert where["phase"]["type"] == "number" and "ge" in where["phase"]["x-vbt-ops"]
    assert "open_targets.literature_vector" in tools["similar"].tables
    assert "open_targets.interaction" in tools["neighbors"].tables
    assert all(catalog.table(t).spec.edge is not None for t in tools["neighbors"].tables)
    assert "depmap.gene_effect" in tools["aggregate"].tables
    assert "depmap.gene_effect" not in tools["search"].tables
    # the Case 1 answer key is native: false and withheld from the trial agents
    assert "zenodo_vbt.clinical_trial_labels" not in find.tables
    assert "zenodo_vbt.clinical_trial_labels" not in {t for x in native_tools(catalog, agent="trial-annotator")
                                                     for t in x.tables}
    ready = {"open_targets.target"}
    names = native_tool_names(catalog, ready=ready)
    assert "mcp__data__find" in names and "mcp__data__search" in names
    assert "mcp__data__similar" not in names and "mcp__data__neighbors" not in names, "no ready table supports them"
    # the gateway's listing: the child registers one `request` argument, the native schema goes under it
    c = catalog.contract("data", "find")
    listed = annotate_schema(c, {"type": "object", "properties": {"request": {}}}, catalog=catalog, registry=registry)
    assert listed["properties"]["request"]["properties"]["table"]["enum"] == find.tables
    assert describe_tool(c, "internal verb", catalog=catalog).startswith("Rows of a table matching `where`")


def test_withhold_from_names_the_agent(shipped) -> None:
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.derive.tools import tables_for

    catalog, registry = shipped
    sources = dict(catalog.sources)
    ot = sources["open_targets"].model_copy(deep=True)
    ot.tables["known_drug"].expose.withhold_from = ["clinical-trialist"]
    sources["open_targets"] = ot
    cat = Catalog(sources, catalog.overlays, catalog.generic, registry=registry)
    assert "open_targets.known_drug" not in tables_for(cat, "find", agent="clinical-trialist")
    assert "open_targets.known_drug" in tables_for(cat, "find", agent="target-scientist")

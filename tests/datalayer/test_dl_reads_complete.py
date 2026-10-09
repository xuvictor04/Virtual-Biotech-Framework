"""Every dataset an upstream tool reads is in its binding's ``reads`` (docs/DATA_LAYER.md §8.1, §13).

Readiness gates a tool on the tables it reads and admission charges them, so a read the overlay
does not list is a table whose outage or size the gateway never sees. The upstream source is
parsed (never imported): for each tool registered in ``src/mcp_servers/*/server.py`` the
function body and the same-module helpers it calls (transitively) are scanned for loader calls
with a string literal (``get_dataset``, ``get_arrow_dataset``, ``get_tahoe_dataset``,
``get_tahoe_metadata``). A read that happens only under an ``if`` on one of the tool's
arguments (``include_indirect``, ``method``) must carry ``when`` naming that argument.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dl_upstream import HAVE_ARROW, REPO, upstream_root

pytestmark = pytest.mark.skipif(not HAVE_ARROW, reason="pyarrow and pandas are needed for the data-layer fixtures")

SERVERS_DIR = upstream_root() / "src" / "mcp_servers"
OVERLAYS = REPO / "configs" / "data" / "overlays"
LOADERS = ("get_dataset", "get_arrow_dataset", "get_tahoe_dataset", "get_tahoe_metadata")
EXCLUDED = ("provenance_mcp",)                 # replaced by native harness tools, not bridged
TAHOE = {
    ("get_tahoe_dataset", "tahoe_pseudobulk_permissive"): "tahoe_100m.de_permissive",
    ("get_tahoe_metadata", "drug"): "tahoe_100m.drug_metadata",
    ("get_tahoe_metadata", "cell_line"): "tahoe_100m.cell_line_metadata",
    ("get_tahoe_metadata", "gene"): "tahoe_100m.gene_metadata",
    ("get_tahoe_metadata", "sample"): "tahoe_100m.sample_metadata",
}


@dataclass
class ToolReads:
    server: str
    tool: str
    params: tuple[str, ...]
    # table -> argument names every read of it depends on (empty: read unconditionally at least once)
    reads: dict[str, set[str]] = field(default_factory=dict)
    lines: dict[str, int] = field(default_factory=dict)
    loaders: dict[str, set[str]] = field(default_factory=dict)      # table -> the loader functions that read it


def _table(loader: str, name: str) -> str:
    if loader in ("get_dataset", "get_arrow_dataset"):
        return f"open_targets.{name}"
    return TAHOE.get((loader, name), f"tahoe_100m.{name}")


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class _Scanner:
    """Loader calls of one function and its same-module helpers, with the argument names of the
    enclosing ``if`` tests."""

    def __init__(self, funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef], params: tuple[str, ...]) -> None:
        self.funcs = funcs
        self.params = set(params)
        self.found: list[tuple[str, frozenset[str], int, str]] = []

    def scan(self, name: str, conds: frozenset[str], stack: tuple[str, ...] = ()) -> None:
        if name in stack or name not in self.funcs:
            return
        for stmt in self.funcs[name].body:
            self._visit(stmt, conds, stack + (name,))

    def _visit(self, node: ast.AST, conds: frozenset[str], stack: tuple[str, ...]) -> None:
        if isinstance(node, (ast.If, ast.IfExp)):
            self._visit(node.test, conds, stack)
            inner = conds | (_names(node.test) & self.params)
            for part in ([node.body] if isinstance(node.body, ast.AST) else node.body) + \
                    ([node.orelse] if isinstance(node.orelse, ast.AST) else node.orelse):
                self._visit(part, frozenset(inner), stack)
            return
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in LOADERS and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self.found.append((_table(f.attr, node.args[0].value), conds, node.lineno, f.attr))
            if isinstance(f, ast.Name) and f.id in self.funcs:
                self.scan(f.id, conds, stack)
        for child in ast.iter_child_nodes(node):
            self._visit(child, conds, stack)


def _registered(server_py: Path) -> list[tuple[str, str]]:
    """``[(imported name, module)]`` of the ``register_tool(mcp, fn)`` calls in a server file."""
    tree = ast.parse(server_py.read_text(encoding="utf-8"))
    imports: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src.mcp_servers"):
            for alias in node.names:
                imports[alias.asname or alias.name] = (node.module, alias.name)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_tool" and len(node.args) == 2:
            local = node.args[1].id                    # type: ignore[attr-defined]
            module, name = imports[local]
            out.append((name, module))
    return out


def upstream_reads() -> list[ToolReads]:
    tools = []
    for server_dir in sorted(p for p in SERVERS_DIR.iterdir() if (p / "server.py").is_file()):
        if server_dir.name in EXCLUDED:
            continue
        server = server_dir.name.removesuffix("_mcp")
        for name, module in _registered(server_dir / "server.py"):
            path = upstream_root() / (module.replace(".", "/") + ".py")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            fn = funcs[name]
            params = tuple(a.arg for a in fn.args.args + fn.args.kwonlyargs)
            sc = _Scanner(funcs, params)
            sc.scan(name, frozenset())
            tr = ToolReads(server, name, params)
            for table, conds, line, loader in sc.found:
                prev = tr.reads.get(table)
                tr.reads[table] = set(conds) if prev is None else (prev & set(conds))
                tr.lines.setdefault(table, line)
                tr.loaders.setdefault(table, set()).add(loader)
            tools.append(tr)
    return tools


@pytest.fixture(scope="module")
def catalog():
    from vbt.datalayer.catalog import Catalog
    from vbt.datalayer.descriptor.load import load_descriptors, load_overlays

    variables = {"project_root": str(REPO)}
    descriptors = load_descriptors(REPO / "configs" / "data" / "sources", variables)
    overlays, generic = load_overlays(OVERLAYS, variables)
    return Catalog(descriptors, overlays, generic)


def _tools() -> list[ToolReads]:
    if not SERVERS_DIR.is_dir():
        return []
    return upstream_reads()


TOOLS = _tools()


@pytest.mark.skipif(not TOOLS, reason="the upstream checkout is missing (git submodule update --init)")
def test_scanner_sees_the_known_reads() -> None:
    """The scanner itself: helpers are followed and branches on arguments are recorded."""
    by = {(t.server, t.tool): t for t in TOOLS}
    assert len(TOOLS) == 101
    profile = by[("target", "get_comprehensive_target_profile")]
    assert profile.reads["open_targets.known_drug"] == {"include_drugs"}       # through search_known_drugs()
    assert profile.reads["open_targets.target"] == set()                        # get_target_info() runs always
    assoc = by[("association", "query_associations")]
    assert assoc.reads == {"open_targets.association_by_overall_indirect": {"include_indirect"},
                           "open_targets.association_overall_direct": {"include_indirect"}}
    tahoe = by[("functional_genomics", "query_drug_perturbation")]
    assert tahoe.reads["tahoe_100m.cell_line_metadata"] == {"cell_line_id"}
    assert tahoe.reads["tahoe_100m.drug_metadata"] == set()


@pytest.mark.skipif(not TOOLS, reason="the upstream checkout is missing (git submodule update --init)")
@pytest.mark.parametrize("tr", TOOLS, ids=lambda t: f"{t.server}.{t.tool}")
def test_binding_reads_cover_upstream_reads(tr: ToolReads, catalog) -> None:
    contract = catalog.contract(tr.server, tr.tool)
    assert contract.binding is not None and not contract.generic, f"{tr.server}.{tr.tool} has no reviewed binding"
    reads = contract.binding.reads
    missing = sorted(t for t in tr.reads if t not in reads)
    assert not missing, (f"{tr.server}.{tr.tool} reads {missing} (upstream line "
                         f"{', '.join(str(tr.lines[t]) for t in missing)}) but the binding does not list them")
    for table, conds in sorted(tr.reads.items()):
        if not conds:
            continue
        when = reads[table].when or {}
        assert set(when) & conds, (f"{tr.server}.{tr.tool}: {table} is read only when {sorted(conds)} say so "
                                   f"(upstream line {tr.lines[table]}); its ReadSpec needs `when` on that argument")


@pytest.mark.skipif(not TOOLS, reason="the upstream checkout is missing (git submodule update --init)")
@pytest.mark.parametrize("tr", TOOLS, ids=lambda t: f"{t.server}.{t.tool}")
def test_whole_table_loads_are_not_declared_projections(tr: ToolReads, catalog) -> None:
    """``get_dataset`` loads the whole table into pandas (``to_table().to_pandas()``, every column), so a read the
    tool makes only through it is ``full_table``, whatever columns the tool then uses: admission sizes
    ``full_table`` reads and not ``projection`` ones. The target server's ``get_chemical_probes`` and
    ``get_genetic_constraint`` were declared projections, admitted at a 3,000 MB limit and the server was
    OOM-killed loading the whole 25.09 target table (``vbt validate``, D3)."""
    reads = catalog.contract(tr.server, tr.tool).binding.reads
    wrong = sorted(t for t, loaders in tr.loaders.items()
                   if loaders == {"get_dataset"} and t in reads and reads[t].access == "projection")
    assert not wrong, (f"{tr.server}.{tr.tool} loads {wrong} whole (get_dataset, upstream line "
                       f"{', '.join(str(tr.lines[t]) for t in wrong)}) but declares a projection: use full_table")


@pytest.mark.skipif(not TOOLS, reason="the upstream checkout is missing (git submodule update --init)")
def test_functional_genomics_loads_target_essentiality_whole(catalog) -> None:
    """Every functional_genomics tool on ``target_essentiality`` calls ``get_dataset`` (no Arrow scan): its ``off``
    calls were OOM-killed at a 4,400 MB limit while the overlay declared a bounded scan, which admission sizes as a
    transient slice."""
    by = {(t.server, t.tool): t for t in TOOLS}
    tools = [t for (server, _tool), t in by.items() if server == "functional_genomics"
             and "open_targets.target_essentiality" in t.reads]
    assert len(tools) == 5
    for t in tools:
        assert t.loaders["open_targets.target_essentiality"] == {"get_dataset"}
        rs = catalog.contract(t.server, t.tool).binding.reads["open_targets.target_essentiality"]
        assert rs.access == "full_table", t.tool

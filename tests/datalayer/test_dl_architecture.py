"""Architecture rules of the data layer (§5, §21).

* Harness-side modules import without pyarrow or pandas (both blocked in a child interpreter): the
  gateway, catalog, derivation, resolver, readiness, memory, launch, CLI and retro-audit run in the
  harness process, and only the data child reads data.
* Core modules (``gateway/``, ``derive/``, ``catalog.py``, ``descriptor/``, ``service/reader.py``) name no
  registered plugin and no dataset table in their code: plugins are found through the registry and
  tables through descriptors and overlays. Words that only coincide with a plugin or table name are
  listed in ``ALLOWED`` with the reason; an entry that is no longer needed fails too.
* Nothing under ``third_party/TheVirtualBiotech`` is modified.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
DATALAYER = SRC / "vbt" / "datalayer"
UPSTREAM = REPO / "third_party" / "TheVirtualBiotech"

HARNESS_MODULES = (
    "vbt.datalayer", "vbt.datalayer.api", "vbt.datalayer.errors", "vbt.datalayer.result", "vbt.datalayer.record",
    "vbt.datalayer.rowkey", "vbt.datalayer.settings", "vbt.datalayer.ipc", "vbt.datalayer.roles",
    "vbt.datalayer.predicate", "vbt.datalayer.catalog", "vbt.datalayer.descriptor.load",
    "vbt.datalayer.descriptor.lint", "vbt.datalayer.descriptor.scoping", "vbt.datalayer.plugins.registry",
    "vbt.datalayer.resolve", "vbt.datalayer.gateway", "vbt.datalayer.gateway.readiness", "vbt.datalayer.derive",
    "vbt.datalayer.memory", "vbt.datalayer.launch", "vbt.datalayer.cli", "vbt.datalayer.retro_audit",
    "vbt.preflight", "vbt.runtime", "vbt.orchestrator", "vbt.tools.mcp_bridge", "vbt.cli",
)

_BLOCKED = r"""
import importlib.abc, sys
BLOCK = ("pyarrow", "pandas")
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError(f"{name} is blocked: harness-side modules must not import it")
        return None
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, sys.argv[1])
import importlib
for m in sys.argv[2:]:
    importlib.import_module(m)
# building the catalog, the registry (builtin plugins included) and a gateway stays pyarrow-free
from vbt.config import load_config
from vbt.datalayer import build_gateway, load_catalog
from vbt.datalayer.plugins.registry import discover
config = load_config(["mock"])
registry = discover(entry_points=False)
catalog = load_catalog(config, registry)
gateway = build_gateway(config, None, registry=registry)
assert gateway.catalog.servers()
leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCK)
assert not leaked, leaked
print("OK", len(catalog.sources))
"""

#: Literals in core modules that equal a plugin or table name without naming one: {value: {file: reason}}.
ALLOWED: dict[str, dict[str, str]] = {
    "count": {f: "the `count` column role or result kind (§7, §8.1), not the count statistic"
              for f in ("gateway/classify.py", "gateway/contracts.py", "gateway/gateway.py", "gateway/leakage.py",
                        "gateway/transforms.py", "derive/schema.py", "descriptor/columns.py", "descriptor/models.py",
                        "descriptor/overlay.py")},
    "numeric": {
        "descriptor/columns.py": "the default `statistic` of a measure column, part of the descriptor schema (§7)",
        "gateway/contracts.py": "falls back to the descriptor schema's default statistic (contract request: read "
                                "the MeasureCol default instead of repeating it)",
        "gateway/transforms.py": "same fallback as gateway/contracts.py",
    },
    "obs": {"gateway/files.py": "the AnnData obs axis of an .h5ad header", "descriptor/models.py": "the matrix obs axis"},
    "var": {"gateway/files.py": "the AnnData var axis of an .h5ad header", "descriptor/models.py": "the matrix var axis"},
    "records": {"descriptor/models.py": "the `records` table kind (§6.2)"},
}


def _core_files() -> list[Path]:
    return [*sorted((DATALAYER / "gateway").glob("*.py")), *sorted((DATALAYER / "derive").glob("*.py")),
            DATALAYER / "catalog.py", *sorted((DATALAYER / "descriptor").glob("*.py")),
            DATALAYER / "service" / "reader.py"]


def _code_literals(path: Path) -> list[tuple[int, str]]:
    """String constants in code (docstrings excluded): what the module names."""
    tree = ast.parse(path.read_text())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_harness_modules_import_without_pyarrow_or_pandas() -> None:
    proc = subprocess.run([sys.executable, "-c", _BLOCKED, str(SRC), *HARNESS_MODULES], capture_output=True,
                          text=True, timeout=300, cwd=REPO)
    assert proc.returncode == 0 and proc.stdout.startswith("OK"), proc.stderr[-3000:]


def test_core_modules_name_no_plugin_or_table() -> None:
    from vbt.config import load_config
    from vbt.datalayer.catalog import load_catalog
    from vbt.datalayer.plugins.registry import discover

    registry = discover(entry_points=False)
    plugins = {name for kind in registry.kinds for name in registry.names(kind)}
    catalog = load_catalog(load_config(["mock"]), registry)
    tables = {t for desc in catalog.sources.values() for t in desc.tables} | set(catalog.sources)
    assert {"parquet", "hive", "ensembl_gene", "numeric"} <= plugins and {"target", "known_drug"} <= tables
    found: dict[tuple[str, str], list[int]] = {}
    for path in _core_files():
        rel = path.relative_to(DATALAYER).as_posix()
        for line, value in _code_literals(path):
            if value in plugins or value in tables:
                found.setdefault((value, rel), []).append(line)
    violations = {k: v for k, v in found.items() if rel_allowed(*k) is None}
    assert not violations, "core modules name plugins or tables: " + "; ".join(
        f"{rel}:{lines} {value!r}" for (value, rel), lines in sorted(violations.items()))
    stale = [(value, rel) for value, files in ALLOWED.items() for rel in files if (value, rel) not in found]
    assert not stale, f"ALLOWED entries no longer needed: {stale}"


def rel_allowed(value: str, rel: str) -> str | None:
    return ALLOWED.get(value, {}).get(rel)


def test_upstream_checkout_is_unmodified() -> None:
    if not (UPSTREAM / ".git").exists():
        pytest.skip("third_party/TheVirtualBiotech is not checked out")
    status = subprocess.run(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True, timeout=60)
    assert status.returncode == 0, status.stderr
    assert status.stdout.strip() == "", f"third_party/TheVirtualBiotech was modified:\n{status.stdout[:2000]}"
    recorded = subprocess.run(["git", "-C", str(REPO), "ls-tree", "HEAD", "third_party/TheVirtualBiotech"],
                              capture_output=True, text=True, timeout=60).stdout.split()
    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=60).stdout.strip()
    if len(recorded) >= 3:
        assert head == recorded[2], f"upstream HEAD {head} differs from the recorded commit {recorded[2]}"

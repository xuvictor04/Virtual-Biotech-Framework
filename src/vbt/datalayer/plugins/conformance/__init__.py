"""Conformance suite runner and helpers (§9.3). No pyarrow at import.

Each kind has a parametrized pytest module ``conformance/<kind>.py`` over
:func:`selected_plugins`, so a new plugin is tested without editing any test and external
packages import the same suites. Golden cases are tagged by capability: a plugin runs only
the cases for capabilities it declares (:func:`has_capability`, :func:`applicable`,
:func:`for_plugin`). ``vbt datasource conformance --plugin <name>`` runs the suites through
:func:`run` and writes a stamp (module digest + suite version) with :func:`write_stamp`;
``data.plugins.require_conformance`` checks it with :func:`stamp_valid`.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..base import plugin_key

__all__ = [
    "SUITE_VERSION", "PLUGIN_ENV", "has_capability", "applicable", "for_plugin", "selected_plugins", "suite_path",
    "run", "module_digest", "write_stamp", "read_stamp", "stamp_valid", "stamp_path",
]

SUITE_VERSION = "1"
#: Restricts the parametrized suites to one plugin name (set by :func:`run`).
PLUGIN_ENV = "VBT_CONFORMANCE_PLUGIN"


def has_capability(plugin: Any, cap: str) -> bool:
    """True when ``plugin`` declares capability ``cap``."""
    return cap in (getattr(plugin, "capabilities", ()) or ())


def applicable(plugin: Any, requires: Iterable[str] | str | None) -> bool:
    """True when ``plugin`` declares every capability a golden case is tagged with."""
    if requires is None:
        return True
    caps = [requires] if isinstance(requires, str) else list(requires)
    return all(has_capability(plugin, c) for c in caps)


def for_plugin(plugin: Any, cases: Iterable[Any]) -> list[Any]:
    """The cases ``plugin`` must pass: a case's tags come from its ``requires``/``capabilities``
    attribute or key (a string or a list); untagged cases always apply."""
    out = []
    for case in cases:
        tags = case.get("requires") if isinstance(case, dict) else getattr(case, "requires", None)
        if tags is None:
            tags = case.get("capabilities") if isinstance(case, dict) else getattr(case, "capabilities", None)
        if applicable(plugin, tags):
            out.append(case)
    return out


def selected_plugins(kind: str, registry: Any = None) -> list[Any]:
    """Plugins of ``kind`` the suite parametrizes over: the registry's (discovered with the
    environment's settings when not given), restricted by ``VBT_CONFORMANCE_PLUGIN``."""
    if registry is None:
        from ..registry import discover
        from ...settings import DataSettings
        registry = discover(DataSettings.from_env())
    plugins = registry.all(kind)
    only = os.environ.get(PLUGIN_ENV)
    if only:
        plugins = [p for p in plugins if p.name == only or plugin_key(p) == only]
    return plugins


def suite_path(kind: str) -> Path:
    return Path(__file__).resolve().parent / f"{kind}.py"


def run(kind: str, plugin: str | None = None, *, extra_args: Sequence[str] = ()) -> int:
    """Run the ``kind`` suite (all plugins, or one by name) with pytest; returns its exit code
    (4 when the suite module does not exist)."""
    path = suite_path(kind)
    if not path.is_file():
        return 4
    import pytest  # only the CLI command needs pytest

    previous = os.environ.get(PLUGIN_ENV)
    if plugin:
        os.environ[PLUGIN_ENV] = plugin
    try:
        return int(pytest.main([str(path), "-q", "-p", "no:cacheprovider", *extra_args]))
    finally:
        if plugin:
            if previous is None:
                os.environ.pop(PLUGIN_ENV, None)
            else:
                os.environ[PLUGIN_ENV] = previous


def module_digest(plugin: Any) -> str:
    """sha256 of the source file defining the plugin's class (or of its qualified name)."""
    cls = plugin if isinstance(plugin, type) else type(plugin)
    try:
        data = Path(inspect.getfile(cls)).read_bytes()
    except (TypeError, OSError):
        data = f"{cls.__module__}:{cls.__qualname__}".encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def stamp_path(plugin: Any, directory: str | Path) -> Path:
    return Path(directory) / "conformance" / f"{plugin.kind}.{plugin.name}.json"


def write_stamp(plugin: Any, directory: str | Path | None = None) -> Path:
    """Record that ``plugin`` passed its suite: ``{plugin, version, module_digest, suite_version, at}``
    under ``<directory>/conformance/`` (default: ``data.cache_dir``)."""
    if directory is None:
        from ...settings import DataSettings
        directory = DataSettings.from_env().cache_dir
    path = stamp_path(plugin, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = {"plugin": plugin_key(plugin), "version": str(plugin.version), "module_digest": module_digest(plugin),
             "suite_version": SUITE_VERSION,
             "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True))
    tmp.replace(path)
    return path


def read_stamp(plugin: Any, directory: str | Path) -> dict[str, Any] | None:
    path = stamp_path(plugin, directory)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def stamp_valid(plugin: Any, directory: str | Path) -> bool:
    """True when a stamp exists for the plugin's current module digest and this suite version."""
    stamp = read_stamp(plugin, directory)
    if not stamp:
        return False
    return stamp.get("module_digest") == module_digest(plugin) and stamp.get("suite_version") == SUITE_VERSION \
        and stamp.get("version") == str(plugin.version)

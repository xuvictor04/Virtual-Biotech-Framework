#!/usr/bin/env python3
"""What runs inside the sandbox for project utilities and plugins (docs/PROJECTS.md). Stdlib only.

Executed by path with ``python -I`` (no ``PYTHONPATH``, no user site, the script's directory not on ``sys.path``)
by :func:`vbt.projects.sandbox.run_sandboxed`; the harness reads the last ``@@VBT_RESULT@@ <json>`` line. With
``--nonce-stdin`` (the validation commands ``test`` and ``conformance``) the first stdin line is a per-run nonce,
read before any candidate code is imported, and the result line is ``@@VBT_RESULT@@ <nonce> <json>``: a line the
code under validation prints without the nonce is not its verdict (ASN-2)::

    runner.py test <dir>                     run the test_* functions of <dir>/test_*.py against <dir>/utility.py
    runner.py call <dir> <entry>             call <entry>(**args) of <dir>/utility.py, args as JSON on stdin
    runner.py script <dir>                   run <dir>/utility.py as __main__ (args JSON on stdin), stdout is the result
    runner.py conformance <kind> <file> <name> <settings.json>
                                             the kind's conformance suite, restricted to plugin <name> of <file>
    runner.py inspect <path> [sample_rows]   InspectDataset's profile of a file or directory (pyarrow)

The utility module is loaded as ``utility`` (tests ``import utility`` or ``from utility import ...``) and test
modules by file, so nothing in the directory can shadow a standard module. A test function may take
``tmp_path`` (a fresh directory). ``src/`` of this checkout is appended to ``sys.path`` so utilities can use
``vbt.datalayer.client``; bytecode is never written.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

MARKER = "@@VBT_RESULT@@"
_SRC = str(Path(__file__).resolve().parents[2])
sys.dont_write_bytecode = True
if _SRC not in sys.path:
    sys.path.append(_SRC)


_NONCE: list[str] = []          # set by --nonce-stdin before any candidate code runs


def _emit(obj: object, code: int = 0) -> int:
    sys.stdout.flush()
    tag = f"{_NONCE[0]} " if _NONCE else ""
    print("\n" + MARKER + " " + tag + json.dumps(obj, default=str), flush=True)
    return code


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tb(limit: int = 4000) -> str:
    return traceback.format_exc()[-limit:]


def cmd_test(directory: Path) -> int:
    t0 = time.time()
    try:
        _load("utility", directory / "utility.py")
    except BaseException as exc:  # noqa: BLE001 - reported as the result
        return _emit({"passed": 0, "failed": [{"name": "<import utility>", "error": f"{type(exc).__name__}: {exc}",
                                               "traceback": _tb()}], "tests": []}, 1)
    passed, failed, names = 0, [], []
    for i, path in enumerate(sorted(directory.glob("test_*.py"))):
        try:
            module = _load(f"vbt_utility_tests_{i}", path)
        except BaseException as exc:  # noqa: BLE001
            failed.append({"name": f"<import {path.name}>", "error": f"{type(exc).__name__}: {exc}",
                           "traceback": _tb()})
            continue
        for name, fn in sorted(vars(module).items()):
            if not name.startswith("test_") or not inspect.isfunction(fn) or fn.__module__ != module.__name__:
                continue
            names.append(f"{path.name}::{name}")
            kwargs = {}
            params = inspect.signature(fn).parameters
            unknown = [p for p in params if p != "tmp_path"]
            if unknown:
                failed.append({"name": names[-1], "error": f"unsupported fixture(s) {unknown}: a test takes no "
                                                           "arguments or only tmp_path"})
                continue
            if "tmp_path" in params:
                kwargs["tmp_path"] = Path(tempfile.mkdtemp(prefix=f"{name}-"))
            try:
                fn(**kwargs)
                passed += 1
            except BaseException as exc:  # noqa: BLE001
                failed.append({"name": names[-1], "error": f"{type(exc).__name__}: {exc}", "traceback": _tb()})
    ok = passed > 0 and not failed
    return _emit({"passed": passed, "failed": failed, "tests": names,
                  "duration_s": round(time.time() - t0, 3)}, 0 if ok else 1)


def _args() -> dict:
    text = sys.stdin.read()
    args = json.loads(text) if text.strip() else {}
    if not isinstance(args, dict):
        raise ValueError("the arguments must be a JSON object")
    return args


def cmd_call(directory: Path, entry: str) -> int:
    try:
        args = _args()
        module = _load("utility", directory / "utility.py")
        fn = getattr(module, entry)
        result = fn(**args)
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            result = repr(result)
        return _emit({"ok": True, "result": result})
    except BaseException as exc:  # noqa: BLE001
        return _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": _tb()}, 1)


def cmd_script(directory: Path) -> int:
    path = directory / "utility.py"
    sys.argv = [str(path)]
    try:
        spec = importlib.util.spec_from_file_location("__main__", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        return _emit({"ok": code == 0, "exit_code": code}, code)
    except BaseException as exc:  # noqa: BLE001
        return _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": _tb()}, 1)
    return _emit({"ok": True, "exit_code": 0})


def cmd_conformance(kind: str, plugin_file: str, name: str, settings_file: str) -> int:
    settings_text = Path(settings_file).read_text(encoding="utf-8")
    os.environ["VBT_DATA_SETTINGS"] = settings_text
    os.environ["VBT_CONFORMANCE_PLUGIN"] = name
    try:
        from vbt.datalayer.plugins import HARNESS_KINDS
        from vbt.datalayer.plugins.base import plugin_key
        from vbt.datalayer.plugins.conformance import SUITE_VERSION, module_digest, suite_path
        from vbt.datalayer.plugins.registry import discover, discover_harness
        from vbt.datalayer.settings import DataSettings

        settings = DataSettings.from_json(settings_text)
        reg = discover_harness(settings) if kind in HARNESS_KINDS else discover(settings)
        plugin = reg.find(kind, name)
        if plugin is None:
            return _emit({"exit_code": 2, "error": f"{plugin_file} registers no {kind} plugin named {name!r} "
                                                   f"(registered {kind} plugins: {reg.names(kind)})"}, 1)
        origin = Path(sys.modules[type(plugin).__module__].__file__ or "").resolve()
        if origin != Path(plugin_file).resolve():
            return _emit({"exit_code": 2, "error": f"{kind}/{name} is defined by {origin}, not by {plugin_file}"}, 1)
        path = suite_path(kind)
        if not path.is_file():
            return _emit({"exit_code": 4, "error": f"no conformance suite for kind {kind!r}"}, 1)
        import pytest

        code = int(pytest.main([str(path), "-q", "-p", "no:cacheprovider", "-x", "-rf", "--no-header"]))
        stamp = {"plugin": plugin_key(plugin), "version": str(plugin.version), "module_digest": module_digest(plugin),
                 "suite_version": SUITE_VERSION}
        if code == 5:
            return _emit({"exit_code": code, "error": "the suite collected no case for the plugin"}, 1)
        return _emit({"exit_code": code, "stamp": stamp}, 0 if code == 0 else 1)
    except BaseException as exc:  # noqa: BLE001
        return _emit({"exit_code": 3, "error": f"{type(exc).__name__}: {exc}", "traceback": _tb()}, 1)


def cmd_inspect(path: str, sample_rows: int) -> int:
    try:
        from vbt.tools.utilities import inspect_dataset

        info = inspect_dataset(Path(path), sample_rows=sample_rows)
        return _emit({"ok": True, "info": info})
    except BaseException as exc:  # noqa: BLE001 - an unreadable file is the answer
        return _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 1)


def main(argv: list[str]) -> int:
    if "--nonce-stdin" in argv:
        argv = [a for a in argv if a != "--nonce-stdin"]
        line = sys.stdin.readline().strip()
        if line:
            _NONCE.append(line)
    if len(argv) >= 2 and argv[0] == "test":
        return cmd_test(Path(argv[1]))
    if len(argv) >= 3 and argv[0] == "call":
        return cmd_call(Path(argv[1]), argv[2])
    if len(argv) >= 2 and argv[0] == "script":
        return cmd_script(Path(argv[1]))
    if len(argv) >= 5 and argv[0] == "conformance":
        return cmd_conformance(argv[1], argv[2], argv[3], argv[4])
    if len(argv) >= 2 and argv[0] == "inspect":
        return cmd_inspect(argv[1], int(argv[2]) if len(argv) > 2 else 5)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

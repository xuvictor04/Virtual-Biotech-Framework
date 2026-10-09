"""Fixtures for the data-layer correctness tests (§19).

* Markers ``correctness`` and ``needs_fastmcp``.
* Session fixtures ``ot_root`` and ``tahoe_root`` (built with ``dl_fixtures``; skipped without
  pyarrow/pandas) and ``eutils`` (the E-utilities stub).
* :func:`cases` parametrizes a test over ``case x mode`` with mode ``off`` (no gateway: a strict
  xfail of the correct-behaviour assertion) and ``enforce`` (skipped until
  ``vbt.datalayer.gateway`` and ``configs/data/overlays/target.yaml`` exist).
* ``live`` starts the unmodified upstream servers once on the shared fixtures; ``live_variant`` starts
  extra bridges for variant fixtures and config overrides. Both modes of a label share one set of
  server processes: ``enforce`` calls go through the gateway, ``off`` calls go to the same upstream
  servers without it (``MCPBridge.call_raw``, the call a bridge without a gateway makes). At most
  ``MAX_LIVE_BRIDGES`` labels run at once (least recently used closed first) and a module's bridges
  stop with the module, which keeps the whole directory in one process under about 4 GB (the shared
  bridge runs eleven upstream servers and the data child, about 2 GB resident).
* ``fixture_ready`` (enforce mode): the gateway's readiness must report every table the six
  tests read and the ``ensembl_gene``, ``ot_disease`` and ``chembl_molecule`` indexes ready;
  otherwise the test fails with a ``FIXTURE:`` message (a fixture error, not a CT failure).
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from dl_upstream import HAVE_ARROW, ensure_fixture_ready, gateway_missing, pubmed_hook_missing, upstream_missing


ROOT_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "correctness: the six correctness tests of docs/DATA_LAYER.md section 19")
    config.addinivalue_line("markers", "slow: long-running data-layer conformance variants (VBT_DL_SLOW=1)")
    config.addinivalue_line("markers", "needs_fastmcp: launches MCP servers (fastmcp and mcp installed)")
    # pytest imports this file as module ``conftest`` too, replacing tests/conftest.py in
    # sys.modules; the top-level tests ``from conftest import open_scripted_session``, so hand the
    # name back (the data-layer tests import their helpers from dl_upstream instead).
    for plugin in config.pluginmanager.get_plugins():
        if getattr(plugin, "__file__", None) and Path(plugin.__file__).resolve() == ROOT_CONFTEST:
            sys.modules["conftest"] = plugin
            break


_EMPTY_ZENODO = tempfile.mkdtemp(prefix="vbt-no-zenodo-")
atexit.register(shutil.rmtree, _EMPTY_ZENODO, ignore_errors=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Run every data-layer test, and the servers its fixtures start, with an empty Zenodo directory.

    ``zenodo_vbt`` defaults to ``<project>/data/zenodo``; a partial extract there would make each
    readiness check scan it and leak host data into the fixtures. Tests outside this directory
    (``test_replicate``, ``test_analysis_fidelity``) still see the real extract."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VBT_ZENODO_DIR", _EMPTY_ZENODO)
        yield


# ---------------------------------------------------------------------------- data fixtures


@pytest.fixture(scope="session")
def ot_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_ARROW:
        pytest.skip("pyarrow and pandas are needed for the fixtures")
    import dl_fixtures

    root = dl_fixtures.build_ot_fixture(tmp_path_factory.mktemp("ot") / "25.09")
    try:
        dl_fixtures.assert_file_order_traps(root)
    except AssertionError as exc:
        pytest.fail(str(exc))
    problems = dl_fixtures.check_descriptor_sentinels(root)
    if problems:
        pytest.fail("FIXTURE: descriptor sentinels missing: " + "; ".join(problems))
    return root


@pytest.fixture(scope="session")
def tahoe_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not HAVE_ARROW:
        pytest.skip("pyarrow and pandas are needed for the fixtures")
    import dl_fixtures

    return dl_fixtures.build_tahoe_fixture(tmp_path_factory.mktemp("tahoe"))


@pytest.fixture(scope="session")
def eutils(tmp_path_factory: pytest.TempPathFactory):
    from stubs import eutils_stub

    stub = eutils_stub.start(tmp_path_factory.mktemp("eutils") / "requests.jsonl")
    yield stub
    stub.stop()


@pytest.fixture(scope="session")
def cbioportal(tmp_path_factory: pytest.TempPathFactory):
    """The REST stub of the ``pybioportal`` stub's study: what the data child's live cBioPortal tables read."""
    from stubs import cbioportal_stub

    stub = cbioportal_stub.start(tmp_path_factory.mktemp("cbioportal") / "requests.jsonl")
    yield stub
    stub.stop()


@pytest.fixture(scope="session")
def data_env(ot_root: Path, tahoe_root: Path, eutils: Any, cbioportal: Any,
             tmp_path_factory: pytest.TempPathFactory) -> Any:
    from dl_upstream import DataEnv

    return DataEnv(ot_root=ot_root, tahoe_root=tahoe_root, eutils_base=eutils.base, cbioportal_base=cbioportal.base,
                   output_dir=tmp_path_factory.mktemp("mcp-output"))


# ---------------------------------------------------------------------------- live servers

OT_SERVERS = ("target", "disease", "drug", "association", "genetics", "interaction", "pathway", "expression")
ALL_SERVERS = OT_SERVERS + ("functional_genomics", "clinicaltrials")


def live_servers() -> tuple[str, ...]:
    return ALL_SERVERS + (("pubmed",) if pubmed_hook_missing() is None else ())


#: Labels whose bridge is kept running at once. The main bridge runs eleven upstream servers plus the data
#: child (about 2 GB resident; a variant usually one or two servers), so keeping every variant for the whole
#: session outgrows the test budget; the least recently used bridge is closed and starts again on its next use
#: (results are cached per bridge and mode, the fixtures are read-only).
MAX_LIVE_BRIDGES = 2
_POOLS: list["_Bridges"] = []


class _OffView:
    """``off`` mode on a started enforce bridge: the same upstream server processes, called without the gateway
    (``MCPBridge.call_raw``, which is what a bridge built without a gateway calls), with its own result cache. A
    second set of eleven servers for the off answers doubled the suite's memory (5.0 GB tree RSS)."""

    gateway = None

    def __init__(self, live: Any) -> None:
        self.live = live
        self._cache: dict[str, Any] = {}

    @property
    def bridge(self) -> Any:
        return self.live.bridge

    def run(self, coro: Any, timeout: float = 600) -> Any:
        return self.live.run(coro, timeout)

    def call(self, server: str, tool: str, args: Mapping[str, Any] | None = None, *, cached: bool = True) -> Any:
        import json

        from dl_upstream import error_of, result_of

        from vbt.tools.base import ToolFailure

        key = json.dumps([server, tool, args or {}], sort_keys=True, default=str)
        if cached and key in self._cache:
            return self._cache[key]

        async def raw() -> Any:
            try:
                return result_of(await self.live.bridge.call_raw(server, tool, dict(args or {})))
            except ToolFailure as exc:
                return error_of(exc)

        res = self.live.run(raw())
        self._cache[key] = res
        return res

    def readiness_snapshot(self) -> dict[str, Any]:
        return {}

    def stderr(self, server: str) -> str:
        return self.live.stderr(server)

    def close(self) -> None:
        """The enforce bridge it views is closed by the pool."""


class _Bridges:
    """Started bridges keyed by label: at most :data:`MAX_LIVE_BRIDGES` at once, all closed at the end of each
    test module that started any (the next module starts them again). ``off`` and ``enforce`` of one label share
    the label's enforce bridge (:class:`_OffView`); without the gateway module an off bridge is started alone."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        self.tmp = tmp_path_factory
        self.started: dict[str, Any] = {}
        self.views: dict[str, _OffView] = {}
        self.failed: dict[str, str] = {}
        _POOLS.append(self)

    def get(self, mode: str, label: str, servers: Sequence[str], env: Any,
            overrides: Mapping[str, Any] | None = None) -> Any:
        from dl_upstream import LiveBridge

        shared = gateway_missing() is None
        key = label if shared else f"{mode}:{label}"
        if key in self.failed:
            pytest.fail(self.failed[key])
        if key in self.started:
            self.started[key] = self.started.pop(key)          # most recently used last
        else:
            reason = upstream_missing()
            if reason:
                pytest.skip(reason)
            if mode == "enforce" and not shared:
                pytest.skip(f"enforce mode: {gateway_missing()}")
            self._evict(MAX_LIVE_BRIDGES - 1)
            try:
                self.started[key] = LiveBridge(servers, env=env, gateway=shared or mode == "enforce",
                                               tmp_path=self.tmp.mktemp(f"bridge-{label}"), overrides=overrides)
            except pytest.skip.Exception:
                raise
            except Exception as exc:  # noqa: BLE001 - reported once, as a fixture error
                msg = str(exc) if str(exc).startswith("FIXTURE:") else f"FIXTURE: bridge start failed: {exc}"
                self.failed[key] = msg
                pytest.fail(msg)
        bridge = self.started[key]
        if shared and mode == "off":
            view = self.views.get(key)
            if view is None or view.live is not bridge:
                view = self.views[key] = _OffView(bridge)
            return view
        return bridge

    def _evict(self, keep: int) -> None:
        while len(self.started) > keep:
            key = next(iter(self.started))                         # the least recently used
            self.views.pop(key, None)
            _close(self.started.pop(key))

    def close(self) -> None:
        for bridge in self.started.values():
            _close(bridge)
        self.started.clear()
        self.views.clear()


def _close(bridge: Any) -> None:
    try:
        bridge.close()
    except Exception:  # noqa: BLE001 - shutdown noise
        pass


@pytest.fixture(scope="session")
def bridges(tmp_path_factory: pytest.TempPathFactory):
    b = _Bridges(tmp_path_factory)
    yield b
    b.close()


@pytest.fixture(scope="module", autouse=True)
def _close_bridges_after_module():
    """The bridges a module started stop with it, so they never add up with the next module's servers; the memory
    the module's in-process tests freed goes back to the system (:func:`_release_memory`)."""
    yield
    for pool in _POOLS:
        pool.close()
    _release_memory()


def _release_memory() -> None:
    """Collect cycles, hand Arrow's unused pool memory back and trim the C heap (glibc ``malloc_trim``): freed
    pages otherwise stay in this process between modules (in-process readers, fixtures and wide matrices)."""
    import ctypes
    import gc

    gc.collect()
    pa = sys.modules.get("pyarrow")
    if pa is not None:
        try:
            pa.default_memory_pool().release_unused()
        except Exception:  # noqa: BLE001 - an older pyarrow has no release_unused
            pass
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


@pytest.fixture(scope="session")
def live(bridges: _Bridges, data_env: Any) -> Callable[[str], Any]:
    """``live(mode)``: the shared bridge over every server the six tests call."""
    def get(mode: str) -> Any:
        return bridges.get(mode, "main", live_servers(), data_env)
    return get


@pytest.fixture(scope="session")
def live_variant(bridges: _Bridges) -> Callable[..., Any]:
    """``live_variant(mode, label, servers, env, overrides=None)``: a separately started bridge."""
    return bridges.get


# ---------------------------------------------------------------------------- readiness precondition

@pytest.fixture
def fixture_ready() -> Callable[[Any], None]:
    """``fixture_ready(bridge)``: in enforce mode, fail as a fixture error unless every table the
    six tests read and the resolver indexes are ready (``vbt ds check`` through the gateway)."""
    return ensure_fixture_ready

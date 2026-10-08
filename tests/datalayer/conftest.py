"""Fixtures for the data-layer correctness tests (§19).

* Markers ``correctness`` and ``needs_fastmcp``.
* Session fixtures ``ot_root`` and ``tahoe_root`` (built with ``dl_fixtures``; skipped without
  pyarrow/pandas) and ``eutils`` (the E-utilities stub).
* :func:`cases` parametrizes a test over ``case x mode`` with mode ``off`` (no gateway: a strict
  xfail of the correct-behaviour assertion) and ``enforce`` (skipped until
  ``vbt.datalayer.gateway`` and ``configs/data/overlays/target.yaml`` exist).
* ``live`` starts the unmodified upstream servers once per mode on the shared fixtures;
  ``live_variant`` starts extra bridges for variant fixtures and config overrides. At most
  ``MAX_LIVE_BRIDGES`` run at once (least recently used closed first) and a module's bridges stop
  with the module, which keeps the whole directory in one process within a few GB.
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
def data_env(ot_root: Path, tahoe_root: Path, eutils: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    from dl_upstream import DataEnv

    return DataEnv(ot_root=ot_root, tahoe_root=tahoe_root, eutils_base=eutils.base,
                   output_dir=tmp_path_factory.mktemp("mcp-output"))


# ---------------------------------------------------------------------------- live servers

OT_SERVERS = ("target", "disease", "drug", "association", "genetics", "interaction", "pathway", "expression")
ALL_SERVERS = OT_SERVERS + ("functional_genomics", "clinicaltrials")


def live_servers() -> tuple[str, ...]:
    return ALL_SERVERS + (("pubmed",) if pubmed_hook_missing() is None else ())


#: Bridges kept running at once. A bridge runs up to eleven upstream servers plus the data child
#: (about 2 GB resident), so keeping every variant for the whole session outgrows a 6 GB test budget;
#: the least recently used bridge is closed and starts again on its next use (results are cached per
#: bridge, the fixtures are read-only).
MAX_LIVE_BRIDGES = 2
_POOLS: list["_Bridges"] = []


class _Bridges:
    """Started bridges keyed by (mode, label): at most :data:`MAX_LIVE_BRIDGES` at once, all closed at
    the end of each test module that started any (the next module starts them again)."""

    def __init__(self, tmp_path_factory: pytest.TempPathFactory) -> None:
        self.tmp = tmp_path_factory
        self.started: dict[tuple[str, str], Any] = {}
        self.failed: dict[tuple[str, str], str] = {}
        _POOLS.append(self)

    def get(self, mode: str, label: str, servers: Sequence[str], env: Any,
            overrides: Mapping[str, Any] | None = None) -> Any:
        from dl_upstream import LiveBridge

        key = (mode, label)
        if key in self.failed:
            pytest.fail(self.failed[key])
        if key in self.started:
            self.started[key] = self.started.pop(key)          # most recently used last
        else:
            reason = upstream_missing()
            if reason:
                pytest.skip(reason)
            if mode == "enforce" and gateway_missing():
                pytest.skip(f"enforce mode: {gateway_missing()}")
            self._evict(MAX_LIVE_BRIDGES - 1)
            try:
                self.started[key] = LiveBridge(servers, env=env, gateway=mode == "enforce",
                                               tmp_path=self.tmp.mktemp(f"bridge-{mode}-{label}"),
                                               overrides=overrides)
            except pytest.skip.Exception:
                raise
            except Exception as exc:  # noqa: BLE001 - reported once, as a fixture error
                msg = str(exc) if str(exc).startswith("FIXTURE:") else f"FIXTURE: bridge start failed: {exc}"
                self.failed[key] = msg
                pytest.fail(msg)
        return self.started[key]

    def _evict(self, keep: int) -> None:
        while len(self.started) > keep:
            _close(self.started.pop(next(iter(self.started))))      # the least recently used

    def close(self) -> None:
        for bridge in self.started.values():
            _close(bridge)
        self.started.clear()


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
    """The bridges a module started stop with it, so they never add up with the next module's servers."""
    yield
    for pool in _POOLS:
        pool.close()


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

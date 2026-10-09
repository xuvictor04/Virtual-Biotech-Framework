"""Shared test fixtures and helpers (offline: scripted provider, no MCP, no network).

* ``config`` -- the ``mock`` profile with ``paths.runs_dir`` under ``tmp_path``.
* :func:`scripted_provider` -- a strict :class:`ScriptedProvider` from rules
  (``{agent: [items]}``), a script callable, or an existing provider.
* :func:`open_scripted_session` -- ``open_session`` with a scripted provider,
  no MCP servers, review enforcement off by default and ``limits`` overrides.
* ``scripted_session`` fixture -- the same function, for tests that prefer a
  fixture over an import.
* ``mock_provider`` fixture -- makes the ``mock`` provider factory (used by the
  CLI, replay and the web UI, which build their own provider) return a
  scripted provider: ``mock_provider.use(rules)`` returns the provider that the
  next sessions will get.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# a host configuration `vbt setup` wrote on this machine (host.env, VBT_PROFILES) must not reach the tests
os.environ["VBT_NO_HOST_ENV"] = "1"
# nor the checkout's own .env (README step 2 writes OPEN_TARGETS_DATA_PATH there: DEP-8)
os.environ["VBT_NO_DOTENV"] = "1"
# The offline suite is hermetic: it plans memory from a fixed 16 GB host, not this machine's MemTotal or cgroup
# limit (under ~9.2 GB the auto server limit refused the round-3 Census pulls: RR-2), and the shell's model server
# and data roots (docs/E2E_RUN.md step 2 exports them) never reach it. Tests that need a host size or a root set
# their own; real-data tests read VBT_DL_REAL_DATA.
os.environ["VBT_HOST_MEMORY_MB"] = "16384"
for _var in ("VBT_LLM_BASE_URL", "VBT_LLM_API_KEY", "OPEN_TARGETS_DATA_PATH", "TAHOE_DATA_PATH", "DEPMAP_DATA_PATH",
             "GO_DATA_PATH", "MSIGDB_DATA_PATH", "VBT_CL_OBO", "VBT_ZENODO_DIR", "VBT_DATA_DIR", "VBT_HOME",
             "VBT_STATE_DIR", "VBT_PROJECTS_DIR", "VBT_LITERATURE_MAXDATE", "VBT_SECRETS_FILE", "SEARXNG_URL"):
    os.environ.pop(_var, None)

from netgate import network_enabled  # noqa: E402
from vbt.config import load_config  # noqa: E402

_LOOPBACK = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


@pytest.fixture(autouse=True)
def _no_live_requests(monkeypatch):
    """No in-process live-API request leaves the machine unless ``VBT_DL_NETWORK`` asks for it (RR-3): the live
    layout's transport refuses any non-loopback URL at once, so a test that forgets to stub it fails fast instead of
    calling the real API (or waiting out the descriptor's timeout behind a proxy that never answers). Loopback stubs
    keep working; a test that sets its own transport replaces this one. The failed-release caches start empty, so
    one test's refused release is not another's."""
    for module, attr in (("vbt.datalayer.plugins.layouts.live_api", "_RELEASE_FAILURES"),
                         ("vbt.datalayer.service.verbs.witness", "_STUDY_RELEASES")):
        if module in sys.modules:
            monkeypatch.setattr(sys.modules[module], attr, {}, raising=False)
    if network_enabled():
        yield
        return
    from urllib.parse import urlparse

    try:
        from vbt.datalayer.plugins.layouts import live_api
    except ImportError:              # pragma: no cover - the data layer's extras are missing
        yield
        return
    real = live_api.http_get

    def guarded(url, params=None, **kwargs):
        if (urlparse(str(url)).hostname or "") not in _LOOPBACK:
            raise live_api.RemoteError(f"request to {url} refused: the offline suite makes no live requests "
                                       f"(set VBT_DL_NETWORK=1)", url=str(url))
        return real(url, params, **kwargs)

    monkeypatch.setattr(live_api.LiveApiLayout, "transport", staticmethod(guarded))
    yield


# ---------------------------------------------------------------------------- opt-in real-data runs (DEP-4)
#: ``VBT_DL_REAL_DATA_STRICT=1`` (the real-data workflow sets it with ``VBT_DL_REAL_DATA``): the run fails when no
#: real-data test passed, so a data directory the tests cannot find is a red run, not 40 green skips.
_REAL = {"collected": 0, "passed": 0, "skipped": []}


def _real_data_test(item: pytest.Item) -> bool:
    return any("VBT_DL_REAL_DATA" in str(m.kwargs.get("reason", "")) for m in item.iter_markers("skipif"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_data: an opt-in test that reads the real releases (VBT_DL_REAL_DATA)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if _real_data_test(item):
            item.add_marker(pytest.mark.real_data)
            _REAL["collected"] += 1


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if "real_data" not in report.keywords:
        return
    if report.when == "call" and report.passed:
        _REAL["passed"] += 1
    elif report.skipped:
        _REAL["skipped"].append(f"{report.nodeid}: {report.longrepr[-1] if isinstance(report.longrepr, tuple) else ''}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not (os.environ.get("VBT_DL_REAL_DATA") and os.environ.get("VBT_DL_REAL_DATA_STRICT") == "1"):
        return
    if _REAL["passed"] == 0:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        lines = [f"VBT_DL_REAL_DATA={os.environ['VBT_DL_REAL_DATA']}: {_REAL['collected']} real-data test(s) selected, "
                 f"none passed"] + [f"  skipped {s}" for s in _REAL["skipped"][:20]]
        if reporter is not None:
            reporter.write_line("\n".join(lines), red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def config(tmp_path):
    return load_config(["mock"], overrides={"paths": {"runs_dir": str(tmp_path / "runs")}})


def scripted_provider(script: Any, **kwargs: Any):
    """A strict ScriptedProvider from rules (dict), a script callable, or a provider (returned as is)."""
    from vbt.providers.mock import ScriptedProvider

    if isinstance(script, ScriptedProvider):
        return script
    if isinstance(script, dict):
        return ScriptedProvider.from_rules(script, **kwargs)
    return ScriptedProvider(script, **kwargs)


async def open_scripted_session(config: dict, script: Any, *, enforce_review: bool | None = False,
                                on_event=None, **limits: Any):
    """open_session(config) with a scripted provider and no MCP servers.

    ``enforce_review`` (default False) sets ``orchestration.enforce_review``
    (None leaves the config alone); keyword ``limits`` update ``config['limits']``.
    """
    from vbt.orchestrator import open_session

    if enforce_review is not None:
        config.setdefault("orchestration", {})["enforce_review"] = enforce_review
    if limits:
        config.setdefault("limits", {}).update(limits)
    return await open_session(config, provider=scripted_provider(script), start_mcp=False, on_event=on_event)


@pytest.fixture
def scripted_session():
    return open_scripted_session


class _MockFactory:
    """``use(rules)``: every session gets this provider; ``use_each(make)``: each
    session gets a fresh provider built from ``make()`` (rules, script or provider)."""

    def __init__(self) -> None:
        self.provider = None
        self._make = None
        self.made: list[Any] = []

    def use(self, script: Any, **kwargs: Any):
        self._make = None
        self.provider = scripted_provider(script, **kwargs)
        return self.provider

    def use_each(self, make, **kwargs: Any) -> None:
        self.provider = None
        self._make = lambda: scripted_provider(make(), **kwargs)

    def __call__(self, **options: Any):
        provider = self._make() if self._make is not None else self.provider
        if provider is None:
            raise AssertionError("mock_provider.use(rules) was not called before a session started")
        self.made.append(provider)
        return provider


@pytest.fixture
def mock_provider(monkeypatch):
    """Route ``create_provider('mock', ...)`` to a scripted provider set with ``.use(rules)``."""
    from vbt import providers

    factory = _MockFactory()
    monkeypatch.setitem(providers._FACTORIES, "mock", factory)
    return factory

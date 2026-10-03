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

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vbt.config import load_config  # noqa: E402


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

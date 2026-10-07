"""Plugin kinds, registry and discovery (§9.1), protocol checks (§9.2) and conformance helpers (§9.3)."""

from __future__ import annotations

import os
import sys
import textwrap
from types import SimpleNamespace

import pytest

from vbt.datalayer.plugins import KIND_PACKAGES, KINDS, entry_point_group
from vbt.datalayer.plugins import conformance as conf
from vbt.datalayer.plugins import registry as reg_mod
from vbt.datalayer.plugins.base import (
    API_VERSION,
    CAPABILITIES,
    NORMALIZE_STEPS,
    FormatPlugin,
    IdentifierBase,
    IdentifierPlugin,
    LayoutPlugin,
    Normalized,
    PluginBase,
    PluginError,
    Rejected,
    Resolution,
    StatisticPlugin,
)
from vbt.datalayer.plugins.registry import PluginRegistry, discover, register, registered, validate_plugin
from vbt.datalayer.settings import DataSettings


class DlTestIdent(IdentifierBase):
    name = id_type = "dl_test_ident"
    canonical = r"T\d+"
    examples = ("T1", "T22")


class DlTestStat(PluginBase):
    kind = "statistic"
    name = "dl_test_stat"
    capabilities = frozenset({"test"})

    def sort_key(self, *a): ...
    def predicate(self, *a): ...
    def bounds(self, *a): ...
    def validate(self, *a): ...
    def aggregate(self, *a): ...
    def comparable(self, *a): ...
    def describe(self, *a): ...
    def family(self, *a): ...
    def test(self, *a): return 0.5


class DlTestLayout(PluginBase):
    kind = "layout"
    name = "dl_test_layout"
    capabilities = frozenset({"scan", "upstream_only"})
    requires = ("definitely_not_an_installed_module_xyz", "json")

    def fragments(self, *a): return []
    def partition_columns(self, *a): return {}
    def signature(self, *a): return ""
    def fingerprint(self, *a): return ""
    def partition_fingerprints(self, *a): return {}
    def probe(self, *a): return []
    def as_of(self, *a): return None


def _settings(**plugins) -> DataSettings:
    return DataSettings.from_config({"data": {"plugins": plugins}})


def test_kinds_are_exactly_five():
    assert set(KINDS) == {"format", "layout", "statistic", "identifier", "envelope"}
    assert KINDS["format"] is FormatPlugin and KINDS["layout"] is LayoutPlugin
    assert KINDS["statistic"] is StatisticPlugin and KINDS["identifier"] is IdentifierPlugin
    assert KIND_PACKAGES == {"format": "formats", "layout": "layouts", "statistic": "statistics",
                             "identifier": "identifiers", "envelope": "envelopes"}
    assert entry_point_group("format") == "vbt.datalayer.format"
    assert set(CAPABILITIES) == set(KINDS) and API_VERSION == 1
    assert {"canonical_prefix_case", "strip_suffix", "strip_version"} <= NORMALIZE_STEPS
    assert "extract_digits" not in NORMALIZE_STEPS


def test_registry_add_get_and_reports():
    reg = PluginRegistry()
    ident = reg.add(DlTestIdent)
    reg.add(DlTestStat())
    reg.add(DlTestLayout)
    assert reg.get("identifier", "dl_test_ident") is ident and reg.has("statistic", "dl_test_stat")
    assert reg.names("layout") == ["dl_test_layout"] and len(reg) == 3 and reg.all("format") == []
    assert reg.versions() == {"identifier/dl_test_ident": "1.0", "statistic/dl_test_stat": "1.0",
                              "layout/dl_test_layout": "1.0"}
    assert reg.requires_report() == {"layout/dl_test_layout": ["definitely_not_an_installed_module_xyz"]}
    assert reg.capability("statistic", "dl_test_stat", "test") and not reg.capability("layout", "dl_test_layout", "live")
    assert reg.find("format", "x") is None
    with pytest.raises(KeyError, match="registered: dl_test_layout"):
        reg.get("layout", "nope")
    with pytest.raises(PluginError, match="collision"):
        reg.add(DlTestIdent)
    reg.add(DlTestIdent, replace=True)
    assert "test_dl_registry" in reg.origins()["identifier/dl_test_ident"]


def test_identifier_base_defaults():
    p = DlTestIdent()
    assert p.normalize(" T12 ") == Normalized("T12", ("strip",)) and isinstance(p.normalize("x1"), Rejected)
    assert p.normalize_stored("T3") == Normalized("T3")
    assert p.looks_like("T9") == 1.0 and p.looks_like("nope") == 0.0
    assert p.label_key("ＰＣＳＫ9 ") == "pcsk9"                      # NFKC + casefold
    configured = p.configure({"canonical": r"X\d+"}, None)
    assert configured is not p and configured.pattern == r"X\d+" and p.pattern == r"T\d+"
    assert configured.normalize("X1") == Normalized("X1") and len(p.describe()) <= 200 and "T1" in p.describe()
    assert Resolution(status="resolved", canonical="T1", rule="exact").hops == ()


@pytest.mark.parametrize("attrs, match", [
    ({"api": 2}, "api"),
    ({"capabilities": frozenset({"teleport"})}, "unknown statistic capabilities"),
    ({"capabilities": frozenset({"paired"})}, "aggregate_pair"),
    ({"name": ""}, "lacks 'name'"),
    ({"kind": "teleporter"}, "unknown plugin kind"),
])
def test_validate_plugin_rejects(attrs, match):
    cls = type("Bad", (DlTestStat,), attrs)
    with pytest.raises(PluginError, match=match):
        validate_plugin(cls())


def test_validate_plugin_requires_protocol_methods():
    class NoMethods(PluginBase):
        kind = "format"
        name = "dl_bad_fmt"

    with pytest.raises(PluginError, match="logical_schema"):
        validate_plugin(NoMethods())


def test_register_decorator():
    @register
    class Decorated(DlTestIdent):
        name = id_type = "dl_decorated"

    assert Decorated in registered() and Decorated in registered([__name__])
    assert Decorated not in registered(["other.module"])
    with pytest.raises(PluginError):
        register(kind="format")(DlTestIdent)


def _write_plugin_module(tmp_path, name: str, body: str):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body))
    return path


PLUGIN_MODULE = """
    from vbt.datalayer.plugins.base import IdentifierBase
    from vbt.datalayer.plugins.registry import register

    @register
    class PathIdent(IdentifierBase):
        name = id_type = "{name}"
        canonical = r"P\\d+"
        examples = ("P1",)
        version = "{version}"
"""


def test_discover_from_settings_paths_and_extra(tmp_path):
    path = _write_plugin_module(tmp_path, "dl_path_plugins_a", PLUGIN_MODULE.format(name="dl_path_ident",
                                                                                    version="2.0"))
    reg = discover(_settings(paths=[str(path)], entry_points=False), extra=[DlTestStat])
    assert reg.get("identifier", "dl_path_ident").version == "2.0"
    assert reg.has("statistic", "dl_test_stat")
    # importing by module name works as well
    sys.path.insert(0, str(tmp_path))
    try:
        _write_plugin_module(tmp_path, "dl_path_plugins_b", PLUGIN_MODULE.format(name="dl_mod_ident", version="1.0"))
        reg2 = discover(_settings(paths=["dl_path_plugins_b"], entry_points=False))
        assert reg2.has("identifier", "dl_mod_ident")
    finally:
        sys.path.remove(str(tmp_path))
    with pytest.raises(PluginError, match="not a file"):
        discover(_settings(paths=[str(tmp_path / "missing.py")], entry_points=False))


def test_discover_disabled_and_collisions(tmp_path):
    a = _write_plugin_module(tmp_path, "dl_coll_a", PLUGIN_MODULE.format(name="dl_same", version="1.0"))
    b = _write_plugin_module(tmp_path, "dl_coll_b", PLUGIN_MODULE.format(name="dl_same", version="9.0"))
    with pytest.raises(PluginError, match="collision"):
        discover(_settings(paths=[str(a), str(b)], entry_points=False))
    reg = discover(_settings(paths=[str(a), str(b)], entry_points=False, override={"identifier/dl_same": str(b)}))
    assert reg.get("identifier", "dl_same").version == "9.0"
    assert not discover(_settings(paths=[str(a)], entry_points=False, disabled=["dl_same"])).has("identifier", "dl_same")
    assert not discover(_settings(paths=[str(a)], entry_points=False,
                                  disabled=["identifier/dl_same"])).has("identifier", "dl_same")
    with pytest.raises(PluginError, match="matches 0"):
        discover(_settings(paths=[str(a), str(b)], entry_points=False, override={"dl_same": "nothing.like.this"}))


def test_discover_entry_points_mocked(monkeypatch):
    class EP:
        def __init__(self, name, obj):
            self.name, self.value, self._obj = name, f"pkg:{name}", obj

        def load(self):
            return self._obj

    groups = {"vbt.datalayer.statistic": [EP("dl_test_stat", DlTestStat)],
              "vbt.datalayer.layout": [EP("dl_test_layout", DlTestLayout)]}
    monkeypatch.setattr(reg_mod, "_entry_points", lambda group: groups.get(group, []))
    reg = discover(_settings())
    assert reg.has("statistic", "dl_test_stat") and reg.has("layout", "dl_test_layout")
    assert reg.origins()["statistic/dl_test_stat"].startswith("entry_point:dl_test_stat")
    assert not discover(_settings(entry_points=False)).has("statistic", "dl_test_stat")

    class Broken(EP):
        def load(self):
            raise ImportError("boom")

    monkeypatch.setattr(reg_mod, "_entry_points", lambda group: [Broken("x", None)] if group.endswith("format") else [])
    with pytest.raises(PluginError, match="failed to load"):
        discover(_settings())
    # an entry point with the wrong API version is refused
    old = type("OldApi", (DlTestStat,), {"api": 0, "name": "dl_old"})
    monkeypatch.setattr(reg_mod, "_entry_points",
                        lambda group: [EP("dl_old", old)] if group.endswith("statistic") else [])
    with pytest.raises(PluginError, match="api"):
        discover(_settings())


def test_builtin_discovery_imports_existing_subpackages(tmp_path, monkeypatch):
    pkg = tmp_path / "dl_fake_builtins"
    (pkg / "identifiers").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "identifiers" / "__init__.py").write_text("")
    (pkg / "identifiers" / "one.py").write_text(textwrap.dedent(PLUGIN_MODULE.format(name="dl_builtin_ident",
                                                                                     version="1.0")))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(reg_mod, "_PKG", "dl_fake_builtins")            # formats/, layouts/, statistics/ missing
    reg = discover(_settings(entry_points=False))
    assert reg.names("identifier") == ["dl_builtin_ident"] and reg.names("format") == []
    (pkg / "statistics").mkdir()
    (pkg / "statistics" / "__init__.py").write_text("raise RuntimeError('broken builtin')\n")
    with pytest.raises(PluginError, match="broken builtin"):
        discover(_settings(entry_points=False))


def test_capability_helpers():
    stat = DlTestStat()
    assert conf.has_capability(stat, "test") and not conf.has_capability(stat, "paired")
    assert conf.applicable(stat, None) and conf.applicable(stat, "test") and not conf.applicable(stat, ["test", "paired"])
    cases = [{"name": "plain"}, {"name": "set-test", "requires": "test"}, {"name": "km", "requires": ["paired"]},
             SimpleNamespace(name="obj", capabilities=("test",))]
    assert [c["name"] if isinstance(c, dict) else c.name for c in conf.for_plugin(stat, cases)] == \
        ["plain", "set-test", "obj"]


def test_selected_plugins_and_run(monkeypatch):
    reg = PluginRegistry()
    reg.add(DlTestIdent)

    class Other(DlTestIdent):
        name = id_type = "dl_other"

    reg.add(Other)
    assert [p.name for p in conf.selected_plugins("identifier", reg)] == ["dl_other", "dl_test_ident"]
    monkeypatch.setenv(conf.PLUGIN_ENV, "dl_other")
    assert [p.name for p in conf.selected_plugins("identifier", reg)] == ["dl_other"]
    monkeypatch.delenv(conf.PLUGIN_ENV)
    assert conf.run("no_such_kind") == 4
    assert conf.suite_path("identifier").name == "identifier.py"


def test_conformance_stamps(tmp_path):
    p = DlTestIdent()
    assert conf.read_stamp(p, tmp_path) is None and not conf.stamp_valid(p, tmp_path)
    path = conf.write_stamp(p, tmp_path)
    assert path == tmp_path / "conformance" / "identifier.dl_test_ident.json" and path.is_file()
    stamp = conf.read_stamp(p, tmp_path)
    assert stamp["suite_version"] == conf.SUITE_VERSION and stamp["module_digest"] == conf.module_digest(p)
    assert conf.stamp_valid(p, tmp_path)

    class Bumped(DlTestIdent):
        version = "1.1"

    assert not conf.stamp_valid(Bumped(), tmp_path)
    assert not os.path.exists(str(path) + ".tmp")

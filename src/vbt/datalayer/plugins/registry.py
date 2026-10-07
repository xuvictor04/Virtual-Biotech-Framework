"""Plugin registry and discovery (§9.1). No pyarrow.

:func:`discover` loads, in order: in-tree builtins (every module of
``vbt.datalayer.plugins.<kind>s`` whose classes are decorated with :func:`register`; missing
subpackages are skipped), Python entry points in group ``vbt.datalayer.<kind>``, and the
modules or files listed in ``data.plugins.paths``. ``data.plugins.disabled`` drops plugins by
``kind/name`` or ``name``. Two plugins with one ``kind/name`` are an error unless
``data.plugins.override`` names the winner (matched against the plugin's ``module:Class``, its
entry-point origin or its source file). A plugin whose ``api`` differs from :data:`~.base.API_VERSION`, that lacks
a required attribute or method, or that declares an unknown capability is an error.

Every kind of :data:`~vbt.datalayer.plugins.KINDS` is discovered the same way, so the fifth kind
(``envelope``, phase 4) needed no change here beyond :meth:`PluginRegistry.envelope`, which falls back
to the builtin default decoder when a registry was built without the kind.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from . import KIND_PACKAGES, KINDS, entry_point_group
from .base import (
    API_VERSION,
    CAPABILITIES,
    CAPABILITY_METHODS,
    REQUIRED_ATTRS,
    REQUIRED_METHODS,
    PluginError,
    plugin_key,
)

__all__ = ["register", "registered", "PluginRegistry", "discover", "validate_plugin", "PluginError"]

_REGISTERED: list[type] = []
_PKG = "vbt.datalayer.plugins"


def register(cls: type | None = None, *, kind: str | None = None) -> Any:
    """Class decorator marking a plugin for discovery (``@register`` or ``@register(kind="format")``)."""

    def deco(c: type) -> type:
        if kind is not None and getattr(c, "kind", None) != kind:
            raise PluginError(f"{c.__qualname__}: registered as {kind!r} but declares kind={getattr(c, 'kind', None)!r}")
        if c not in _REGISTERED:
            _REGISTERED.append(c)
        return c

    return deco(cls) if cls is not None else deco


def registered(modules: Iterable[str] | None = None) -> list[type]:
    """Classes decorated with :func:`register`, optionally only those defined in ``modules``."""
    if modules is None:
        return list(_REGISTERED)
    mods = set(modules)
    return [c for c in _REGISTERED if c.__module__ in mods]


def _origin(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _source_file(obj: Any) -> str | None:
    cls = obj if isinstance(obj, type) else type(obj)
    try:
        return str(Path(inspect.getfile(cls)).resolve())
    except (TypeError, OSError):
        return None


def validate_plugin(plugin: Any, kinds: Mapping[str, type] | None = None) -> None:
    """Raise :class:`PluginError` unless ``plugin`` (an instance) satisfies its kind's contract."""
    kinds = kinds or KINDS
    kind = str(getattr(plugin, "kind", None))
    where = _origin(plugin)
    if kind not in kinds:
        raise PluginError(f"{where}: unknown plugin kind {kind!r} (known: {sorted(kinds)})")
    for attr in REQUIRED_ATTRS.get(kind, ("kind", "name", "version")):
        if getattr(plugin, attr, None) in (None, ""):
            raise PluginError(f"{where}: {kind} plugin lacks {attr!r}")
    api = getattr(plugin, "api", API_VERSION)
    if api != API_VERSION:
        raise PluginError(f"{where}: plugin api {api!r} != API_VERSION {API_VERSION}")
    for method in REQUIRED_METHODS.get(kind, ()):
        if not callable(getattr(plugin, method, None)):
            raise PluginError(f"{where}: {kind} plugin lacks method {method}()")
    caps = frozenset(getattr(plugin, "capabilities", frozenset()) or ())
    allowed = CAPABILITIES.get(kind)
    if allowed is not None:
        unknown = caps - allowed
        if unknown:
            raise PluginError(f"{where}: unknown {kind} capabilities {sorted(unknown)} (known: {sorted(allowed)})")
    for cap in caps:
        for method in CAPABILITY_METHODS.get(kind, {}).get(cap, ()):
            if not callable(getattr(plugin, method, None)):
                raise PluginError(f"{where}: declares capability {cap!r} but lacks {method}()")


class PluginRegistry:
    """Plugin instances by kind and name."""

    def __init__(self, kinds: Mapping[str, type] | None = None) -> None:
        self.kinds: dict[str, type] = dict(kinds or KINDS)
        self._plugins: dict[str, dict[str, Any]] = {k: {} for k in self.kinds}
        self._origins: dict[str, str] = {}

    def add(self, plugin: Any, *, origin: str | None = None, replace: bool = False) -> Any:
        """Register a plugin class (instantiated without arguments) or instance."""
        inst = plugin() if isinstance(plugin, type) else plugin
        validate_plugin(inst, self.kinds)
        kind, name = inst.kind, inst.name
        if name in self._plugins[kind] and not replace:
            raise PluginError(f"plugin name collision {kind}/{name}: {self._origins[f'{kind}/{name}']} and "
                              f"{origin or _origin(inst)}; set data.plugins.override to choose")
        self._plugins[kind][name] = inst
        self._origins[f"{kind}/{name}"] = origin or _origin(inst)
        return inst

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._plugins[kind][name]
        except KeyError:
            known = ", ".join(self.names(kind)) if kind in self._plugins else f"unknown kind {kind!r}"
            raise KeyError(f"no {kind} plugin {name!r} (registered: {known})") from None

    def find(self, kind: str, name: str) -> Any | None:
        return self._plugins.get(kind, {}).get(name)

    def has(self, kind: str, name: str) -> bool:
        return name in self._plugins.get(kind, {})

    def all(self, kind: str) -> list[Any]:
        return [self._plugins[kind][n] for n in sorted(self._plugins.get(kind, {}))]

    def names(self, kind: str) -> list[str]:
        return sorted(self._plugins.get(kind, {}))

    def versions(self) -> dict[str, str]:
        """``{"kind/name": version}`` (provenance and pinning)."""
        return {plugin_key(p): str(p.version) for k in self.kinds for p in self.all(k)}

    def origins(self) -> dict[str, str]:
        return dict(sorted(self._origins.items()))

    def envelope(self, name: str | None = None) -> Any:
        """The envelope plugin ``name`` (phase 4): the registered one, else the builtin default when
        ``name`` is None or names the default; an unknown name is a ``KeyError``."""
        from .envelopes import DEFAULT_ENVELOPE, default_envelope

        found = self.find("envelope", name or DEFAULT_ENVELOPE) if "envelope" in self.kinds else None
        if found is not None:
            return found
        if name in (None, DEFAULT_ENVELOPE):
            return default_envelope()
        return self.get("envelope", str(name))

    def capability(self, kind: str, name: str, cap: str) -> bool:
        p = self.find(kind, name)
        return p is not None and cap in (getattr(p, "capabilities", ()) or ())

    def requires_report(self) -> dict[str, list[str]]:
        """``{"kind/name": [missing modules]}`` for plugins whose ``requires`` cannot be imported
        (probed with ``find_spec``, nothing is imported). Plugins with nothing missing are omitted."""
        out: dict[str, list[str]] = {}
        for k in self.kinds:
            for p in self.all(k):
                missing = []
                for mod in getattr(p, "requires", ()) or ():
                    try:
                        found = importlib.util.find_spec(mod) is not None
                    except (ImportError, ValueError):
                        found = False
                    if not found:
                        missing.append(mod)
                if missing:
                    out[plugin_key(p)] = missing
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self._plugins.values())


def _entry_points(group: str) -> list[Any]:
    """Entry points of ``group`` (a seam tests replace)."""
    try:
        return list(importlib.metadata.entry_points(group=group))
    except Exception:  # noqa: BLE001 - broken distribution metadata never blocks discovery
        return []


def _import_builtins(kind: str) -> list[str]:
    """Import every module of the builtin package of ``kind``; returns the module names."""
    pkg_name = f"{_PKG}.{KIND_PACKAGES.get(kind, kind + 's')}"
    try:
        pkg = importlib.import_module(pkg_name)
    except ModuleNotFoundError as exc:
        if exc.name == pkg_name:
            return []                                  # the subpackage does not exist (yet)
        raise PluginError(f"builtin plugin package {pkg_name} failed to import: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - a broken builtin package is a bug, reported with its cause
        raise PluginError(f"builtin plugin package {pkg_name} failed to import: {exc}") from exc
    names = [pkg_name]
    for info in pkgutil.iter_modules(getattr(pkg, "__path__", [])):
        mod_name = f"{pkg_name}.{info.name}"
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - a broken builtin is a bug, reported with its cause
            raise PluginError(f"builtin plugin module {mod_name} failed to import: {exc}") from exc
        names.append(mod_name)
    return names


def _import_path(entry: str) -> str:
    """Import a module name or a ``.py`` file from ``data.plugins.paths``; returns its module name."""
    p = Path(entry)
    if entry.endswith(".py") or p.suffix == ".py" or ("/" in entry):
        if not p.is_file():
            raise PluginError(f"data.plugins.paths: {entry} is not a file")
        mod_name = "vbt_plugin_" + "_".join(p.resolve().with_suffix("").parts[-3:]).replace("-", "_")
        if mod_name in sys.modules:
            return mod_name
        spec = importlib.util.spec_from_file_location(mod_name, p)
        if spec is None or spec.loader is None:
            raise PluginError(f"data.plugins.paths: cannot load {entry}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(mod_name, None)
            raise PluginError(f"data.plugins.paths: {entry} failed to import: {exc}") from exc
        return mod_name
    try:
        importlib.import_module(entry)
    except Exception as exc:  # noqa: BLE001
        raise PluginError(f"data.plugins.paths: module {entry} failed to import: {exc}") from exc
    return entry


def _disabled(kind: str, name: str, disabled: Iterable[str]) -> bool:
    d = set(disabled)
    return f"{kind}/{name}" in d or name in d


def discover(settings: Any = None, *, kinds: Mapping[str, type] | None = None,
             entry_points: bool | None = None,
             extra: Iterable[Any] = (),
             entry_point_loader: Callable[[str], list[Any]] | None = None) -> PluginRegistry:
    """Build a registry from builtins, entry points, ``data.plugins.paths`` and ``extra`` classes.

    ``settings`` is a :class:`~vbt.datalayer.settings.DataSettings` (or None for the defaults)."""
    plugins_cfg = getattr(settings, "plugins", None)
    paths = tuple(getattr(plugins_cfg, "paths", ()) or ())
    disabled = tuple(getattr(plugins_cfg, "disabled", ()) or ())
    override = dict(getattr(plugins_cfg, "override", {}) or {})
    use_eps = getattr(plugins_cfg, "entry_points", True) if entry_points is None else entry_points
    loader = entry_point_loader or _entry_points
    reg = PluginRegistry(kinds)

    candidates: dict[tuple[str, str], list[tuple[str, Any]]] = {}

    def offer(obj: Any, origin: str) -> None:
        kind, name = getattr(obj, "kind", None), getattr(obj, "name", None)
        if kind not in reg.kinds:
            raise PluginError(f"{origin}: unknown plugin kind {kind!r}")
        if not name:
            raise PluginError(f"{origin}: plugin has no name")
        if _disabled(kind, name, disabled):
            return
        bucket = candidates.setdefault((kind, name), [])
        if any(o is obj or (isinstance(o, type) and o is obj) for _, o in bucket):
            return                                     # the same class reached through two routes
        bucket.append((origin, obj))

    for kind in reg.kinds:
        for cls in registered(_import_builtins(kind)):
            if getattr(cls, "kind", None) == kind:
                offer(cls, _origin(cls))
    if use_eps:
        for kind in reg.kinds:
            for ep in loader(entry_point_group(kind)):
                try:
                    obj = ep.load()
                except Exception as exc:  # noqa: BLE001
                    raise PluginError(f"entry point {ep.name} ({getattr(ep, 'value', '?')}) failed to load: "
                                      f"{exc}") from exc
                offer(obj, f"entry_point:{ep.name}={getattr(ep, 'value', _origin(obj))}")
    for entry in paths:
        mod = _import_path(str(entry))
        for cls in registered([mod]):
            offer(cls, _origin(cls))
    for obj in extra:
        offer(obj, _origin(obj))

    for (kind, name), bucket in sorted(candidates.items()):
        if len(bucket) > 1:
            choice = override.get(f"{kind}/{name}") or override.get(name)
            if not choice:
                origins = ", ".join(o for o, _ in bucket)
                raise PluginError(f"plugin name collision {kind}/{name}: {origins}; set data.plugins.override "
                                  f"to choose the winner")
            winners = [(o, obj) for o, obj in bucket if o == choice or o.startswith(choice)
                       or _origin(obj).startswith(choice) or _source_file(obj) == str(Path(choice).resolve())]
            if len(winners) != 1:
                raise PluginError(f"data.plugins.override[{kind}/{name}] = {choice!r} matches {len(winners)} of "
                                  f"{[o for o, _ in bucket]}")
            bucket = winners
        origin, obj = bucket[0]
        reg.add(obj, origin=origin)
    return reg

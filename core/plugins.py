"""Unified plugin system — decorator registration, metadata, lifecycle hooks.

Zero external dependencies (stdlib only). Category-scoped registries
with optional auto-discovery from packages.

Usage::

    registry = PluginRegistry("strategy")

    @registry.register(name="market_maker", version="1.0")
    class MarketMakerPlugin:
        def on_init(self, config): ...
        def on_start(self): ...
        def on_stop(self): ...

    # Programmatic registration:
    registry.register_instance(instance, name="mm", version="1.0")

    # Auto-discover from package:
    registry.discover_package("strategies.hft")
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)


# ── Plugin Metadata ──────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PluginMeta:
    """Immutable plugin metadata."""
    name: str
    version: str = "0.0.0"
    category: str = ""
    description: str = ""
    params_schema: Mapping[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()


# ── Plugin Entry ─────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PluginEntry:
    """A registered plugin with its metadata."""
    meta: PluginMeta
    plugin: Any
    plugin_class: Type


# ── Errors ───────────────────────────────────────────────

class PluginNotFoundError(KeyError):
    pass


# ── Plugin Registry ──────────────────────────────────────

class PluginRegistry:
    """Category-scoped plugin registry with decorator registration.

    Parameters
    ----------
    category : str
        The category this registry manages (e.g., "strategy", "alpha").
    """

    def __init__(self, category: str) -> None:
        self._category = category
        self._plugins: Dict[str, PluginEntry] = {}

    @property
    def category(self) -> str:
        return self._category

    # ── Decorator Registration ───────────────────────────

    def register(
        self,
        *,
        name: Optional[str] = None,
        version: str = "0.0.0",
        description: str = "",
        params_schema: Optional[Mapping[str, Any]] = None,
        tags: Tuple[str, ...] = (),
    ) -> Callable[[Type], Type]:
        """Decorator to register a plugin class."""
        def decorator(cls: Type) -> Type:
            plugin_name = name or getattr(cls, "name", None) or cls.__name__
            meta = PluginMeta(
                name=plugin_name,
                version=version,
                category=self._category,
                description=description,
                params_schema=dict(params_schema or {}),
                tags=tags,
            )
            self._plugins[plugin_name] = PluginEntry(
                meta=meta, plugin=cls, plugin_class=cls,
            )
            cls._plugin_meta = meta  # type: ignore[attr-defined]
            return cls
        return decorator

    # ── Programmatic Registration ────────────────────────

    def register_instance(
        self,
        instance: Any,
        *,
        name: Optional[str] = None,
        version: str = "0.0.0",
        description: str = "",
        params_schema: Optional[Mapping[str, Any]] = None,
        tags: Tuple[str, ...] = (),
    ) -> None:
        """Register an existing plugin instance."""
        plugin_name = name or getattr(instance, "name", None) or instance.__class__.__name__
        meta = PluginMeta(
            name=plugin_name,
            version=version,
            category=self._category,
            description=description,
            params_schema=dict(params_schema or {}),
            tags=tags,
        )
        self._plugins[plugin_name] = PluginEntry(
            meta=meta, plugin=instance, plugin_class=type(instance),
        )

    # ── Lookup ───────────────────────────────────────────

    def get(self, name: str) -> PluginEntry:
        entry = self._plugins.get(name)
        if entry is None:
            raise PluginNotFoundError(
                f"no plugin '{name}' in category '{self._category}', "
                f"available: {list(self._plugins.keys())}"
            )
        return entry

    def get_optional(self, name: str) -> Optional[PluginEntry]:
        return self._plugins.get(name)

    def list_names(self) -> Sequence[str]:
        return list(self._plugins.keys())

    def list_entries(self) -> Sequence[PluginEntry]:
        return list(self._plugins.values())

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)

    # ── Auto-Discovery ───────────────────────────────────

    def discover_package(self, package_name: str) -> int:
        """Import all modules in a package to trigger @register decorators.

        Returns the number of newly discovered plugins.
        """
        before = len(self._plugins)
        try:
            pkg = importlib.import_module(package_name)
        except ImportError:
            return 0

        if hasattr(pkg, "__path__"):
            for _importer, modname, _ispkg in pkgutil.walk_packages(
                pkg.__path__, prefix=package_name + ".",
            ):
                try:
                    importlib.import_module(modname)
                except ImportError:
                    pass
        return len(self._plugins) - before

    # ── Lifecycle Management ─────────────────────────────

    def init_all(self, config: Mapping[str, Any]) -> None:
        """Call on_init on all plugins that support it."""
        for entry in self._plugins.values():
            fn = getattr(entry.plugin, "on_init", None)
            if callable(fn):
                fn(config)

    def start_all(self) -> None:
        """Call on_start on all plugins that support it."""
        for entry in self._plugins.values():
            fn = getattr(entry.plugin, "on_start", None)
            if callable(fn):
                fn()

    def stop_all(self) -> None:
        """Call on_stop on all plugins that support it."""
        for entry in self._plugins.values():
            fn = getattr(entry.plugin, "on_stop", None)
            if callable(fn):
                fn()


# ── Global category registries ───────────────────────────

_GLOBAL_REGISTRIES: Dict[str, PluginRegistry] = {}


def get_registry(category: str) -> PluginRegistry:
    """Get or create the global registry for a category."""
    if category not in _GLOBAL_REGISTRIES:
        _GLOBAL_REGISTRIES[category] = PluginRegistry(category)
    return _GLOBAL_REGISTRIES[category]


def reset_global_registries() -> None:
    """Clear all global registries. For testing only."""
    _GLOBAL_REGISTRIES.clear()


# ── Convenience decorators ───────────────────────────────

def strategy_plugin(
    *, name: Optional[str] = None, version: str = "0.0.0", **kwargs: Any,
) -> Callable[[Type], Type]:
    return get_registry("strategy").register(name=name, version=version, **kwargs)


def venue_plugin(
    *, name: Optional[str] = None, version: str = "0.0.0", **kwargs: Any,
) -> Callable[[Type], Type]:
    return get_registry("venue").register(name=name, version=version, **kwargs)


def alpha_plugin(
    *, name: Optional[str] = None, version: str = "0.0.0", **kwargs: Any,
) -> Callable[[Type], Type]:
    return get_registry("alpha").register(name=name, version=version, **kwargs)


def indicator_plugin(
    *, name: Optional[str] = None, version: str = "0.0.0", **kwargs: Any,
) -> Callable[[Type], Type]:
    return get_registry("indicator").register(name=name, version=version, **kwargs)

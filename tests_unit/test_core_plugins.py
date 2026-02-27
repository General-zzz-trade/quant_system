"""Tests for core.plugins — PluginRegistry, decorators, lifecycle, discovery."""
from __future__ import annotations

import pytest

from core.plugins import (
    PluginEntry,
    PluginMeta,
    PluginNotFoundError,
    PluginRegistry,
    get_registry,
    reset_global_registries,
)


# ── PluginMeta ───────────────────────────────────────────


class TestPluginMeta:
    def test_frozen_immutable(self):
        meta = PluginMeta(name="test", version="1.0")
        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore[misc]

    def test_defaults(self):
        meta = PluginMeta(name="x")
        assert meta.version == "0.0.0"
        assert meta.category == ""
        assert meta.tags == ()


# ── PluginRegistry core ──────────────────────────────────


class TestPluginRegistry:
    def test_register_decorator(self):
        reg = PluginRegistry("test")

        @reg.register(name="my_plugin", version="1.0")
        class MyPlugin:
            pass

        entry = reg.get("my_plugin")
        assert entry.meta.name == "my_plugin"
        assert entry.meta.version == "1.0"
        assert entry.plugin_class is MyPlugin

    def test_register_uses_class_name_as_fallback(self):
        reg = PluginRegistry("test")

        @reg.register()
        class FancyStrategy:
            pass

        assert "FancyStrategy" in reg

    def test_register_instance(self):
        reg = PluginRegistry("test")

        class MyModel:
            name = "alpha_v1"

        instance = MyModel()
        reg.register_instance(instance, name="alpha_v1", version="2.0")

        entry = reg.get("alpha_v1")
        assert entry.plugin is instance
        assert entry.meta.version == "2.0"

    def test_get_not_found_raises(self):
        reg = PluginRegistry("test")
        with pytest.raises(PluginNotFoundError):
            reg.get("nonexistent")

    def test_get_optional_returns_none(self):
        reg = PluginRegistry("test")
        assert reg.get_optional("nonexistent") is None

    def test_list_names(self):
        reg = PluginRegistry("test")

        @reg.register(name="a")
        class A:
            pass

        @reg.register(name="b")
        class B:
            pass

        names = reg.list_names()
        assert "a" in names
        assert "b" in names

    def test_list_entries(self):
        reg = PluginRegistry("test")

        @reg.register(name="x")
        class X:
            pass

        entries = reg.list_entries()
        assert len(entries) == 1
        assert isinstance(entries[0], PluginEntry)

    def test_contains(self):
        reg = PluginRegistry("test")

        @reg.register(name="present")
        class P:
            pass

        assert "present" in reg
        assert "absent" not in reg

    def test_len(self):
        reg = PluginRegistry("test")
        assert len(reg) == 0

        @reg.register(name="one")
        class One:
            pass

        assert len(reg) == 1

    def test_duplicate_name_overwrites(self):
        reg = PluginRegistry("test")

        @reg.register(name="dup")
        class V1:
            pass

        @reg.register(name="dup", version="2.0")
        class V2:
            pass

        entry = reg.get("dup")
        assert entry.plugin_class is V2
        assert entry.meta.version == "2.0"

    def test_category_property(self):
        reg = PluginRegistry("strategy")
        assert reg.category == "strategy"

    def test_params_schema_and_tags(self):
        reg = PluginRegistry("test")

        @reg.register(
            name="configured",
            params_schema={"window": 20},
            tags=("momentum", "daily"),
        )
        class Configured:
            pass

        entry = reg.get("configured")
        assert entry.meta.params_schema == {"window": 20}
        assert entry.meta.tags == ("momentum", "daily")


# ── Lifecycle Hooks ──────────────────────────────────────


class TestLifecycleHooks:
    def test_init_all_calls_on_init(self):
        reg = PluginRegistry("test")
        calls = []

        class WithInit:
            def on_init(self, config):
                calls.append(("init", config))

        reg.register_instance(WithInit(), name="wi")
        reg.init_all({"key": "value"})
        assert len(calls) == 1
        assert calls[0] == ("init", {"key": "value"})

    def test_start_all_calls_on_start(self):
        reg = PluginRegistry("test")
        calls = []

        class WithStart:
            def on_start(self):
                calls.append("started")

        reg.register_instance(WithStart(), name="ws")
        reg.start_all()
        assert calls == ["started"]

    def test_stop_all_calls_on_stop(self):
        reg = PluginRegistry("test")
        calls = []

        class WithStop:
            def on_stop(self):
                calls.append("stopped")

        reg.register_instance(WithStop(), name="ws")
        reg.stop_all()
        assert calls == ["stopped"]

    def test_lifecycle_skips_plugins_without_hooks(self):
        reg = PluginRegistry("test")

        class NoHooks:
            pass

        reg.register_instance(NoHooks(), name="nh")
        # Should not raise
        reg.init_all({})
        reg.start_all()
        reg.stop_all()


# ── Auto-Discovery ───────────────────────────────────────


class TestAutoDiscovery:
    def test_discover_missing_package_returns_zero(self):
        reg = PluginRegistry("test")
        count = reg.discover_package("nonexistent.package.xyz")
        assert count == 0


# ── Convenience Decorators ───────────────────────────────


class TestConvenienceDecorators:
    def setup_method(self):
        reset_global_registries()

    def test_strategy_plugin_decorator(self):
        from core.plugins import strategy_plugin

        @strategy_plugin(name="test_strat", version="1.0")
        class TestStrat:
            pass

        reg = get_registry("strategy")
        assert "test_strat" in reg
        assert reg.get("test_strat").meta.category == "strategy"

    def test_venue_plugin_decorator(self):
        from core.plugins import venue_plugin

        @venue_plugin(name="test_venue")
        class TestVenue:
            pass

        reg = get_registry("venue")
        assert "test_venue" in reg

    def test_alpha_plugin_decorator(self):
        from core.plugins import alpha_plugin

        @alpha_plugin(name="test_alpha")
        class TestAlpha:
            pass

        reg = get_registry("alpha")
        assert "test_alpha" in reg

    def test_indicator_plugin_decorator(self):
        from core.plugins import indicator_plugin

        @indicator_plugin(name="test_ind")
        class TestInd:
            pass

        reg = get_registry("indicator")
        assert "test_ind" in reg

    def teardown_method(self):
        reset_global_registries()


# ── Global Registry ──────────────────────────────────────


class TestGlobalRegistry:
    def setup_method(self):
        reset_global_registries()

    def test_get_registry_creates_on_demand(self):
        reg = get_registry("custom")
        assert isinstance(reg, PluginRegistry)
        assert reg.category == "custom"

    def test_get_registry_returns_same_instance(self):
        r1 = get_registry("same")
        r2 = get_registry("same")
        assert r1 is r2

    def test_reset_clears_all(self):
        get_registry("a")
        get_registry("b")
        reset_global_registries()
        # New call creates fresh instance
        r = get_registry("a")
        assert len(r) == 0

    def teardown_method(self):
        reset_global_registries()

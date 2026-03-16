"""Tests for core/config — layered configuration with type casting and hot-reload."""
from __future__ import annotations

import json

import pytest

from core.config import (
    ConfigKeyError,
    ConfigService,
    ConfigTypeError,
    DefaultsLayer,
    EnvLayer,
    FileLayer,
    RuntimeLayer,
)


# ── DefaultsLayer ───────────────────────────────────────────

class TestDefaultsLayer:
    def test_has_and_get(self):
        layer = DefaultsLayer({"a": 1, "b": "hello"})
        assert layer.has("a")
        assert not layer.has("missing")
        assert layer.get_raw("a") == 1

    def test_empty_defaults(self):
        layer = DefaultsLayer()
        assert not layer.has("anything")


# ── FileLayer ────────────────────────────────────────────────

class TestFileLayer:
    def test_json_loading(self, tmp_path):
        config = {"risk.max_leverage": 5, "bus.capacity": 2000}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))
        layer = FileLayer(p)
        assert layer.has("risk.max_leverage")
        assert layer.get_raw("risk.max_leverage") == 5

    def test_missing_file(self, tmp_path):
        layer = FileLayer(tmp_path / "nonexistent.json")
        assert not layer.has("anything")

    def test_yaml_fallback(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("pyyaml not installed")
        p = tmp_path / "config.yaml"
        p.write_text("bus:\n  capacity: 3000\n")
        layer = FileLayer(p)
        # YAML loads as nested dict with top-level key "bus"
        assert layer.has("bus")


# ── EnvLayer ─────────────────────────────────────────────────

class TestEnvLayer:
    def test_key_transformation(self, monkeypatch):
        monkeypatch.setenv("QS_RISK__MAX_LEVERAGE", "3.0")
        layer = EnvLayer(prefix="QS_")
        assert layer.has("risk.max_leverage")
        assert layer.get_raw("risk.max_leverage") == "3.0"

    def test_missing_env(self):
        layer = EnvLayer(prefix="QS_NONEXIST_")
        assert not layer.has("some.key")

    def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MYAPP_FOO__BAR", "42")
        layer = EnvLayer(prefix="MYAPP_")
        assert layer.has("foo.bar")
        assert layer.get_raw("foo.bar") == "42"


# ── RuntimeLayer ─────────────────────────────────────────────

class TestRuntimeLayer:
    def test_set_get_delete(self):
        layer = RuntimeLayer()
        assert not layer.has("x")
        layer.set("x", 42)
        assert layer.has("x")
        assert layer.get_raw("x") == 42
        layer.delete("x")
        assert not layer.has("x")

    def test_delete_nonexistent(self):
        layer = RuntimeLayer()
        layer.delete("missing")  # should not raise

    def test_overwrite(self):
        layer = RuntimeLayer()
        layer.set("k", 1)
        layer.set("k", 2)
        assert layer.get_raw("k") == 2


# ── ConfigService layer priority ─────────────────────────────

class TestConfigServicePriority:
    def test_defaults_used(self):
        svc = ConfigService(defaults={"a": "default_val"})
        assert svc.get("a") == "default_val"

    def test_env_overrides_defaults(self, monkeypatch):
        monkeypatch.setenv("QS_A", "from_env")
        svc = ConfigService(defaults={"a": "default_val"})
        assert svc.get("a") == "from_env"

    def test_file_overrides_defaults(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"a": "from_file"}))
        svc = ConfigService(defaults={"a": "default_val"}, file_path=p)
        assert svc.get("a") == "from_file"

    def test_runtime_overrides_all(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QS_A", "from_env")
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"a": "from_file"}))
        svc = ConfigService(defaults={"a": "default"}, file_path=p)
        svc.hot_update("a", "from_runtime")
        assert svc.get("a") == "from_runtime"


# ── Type casting ─────────────────────────────────────────────

class TestTypeCasting:
    def test_int_cast(self):
        svc = ConfigService(defaults={"n": "42"})
        assert svc.get("n", int) == 42

    def test_float_cast(self):
        svc = ConfigService(defaults={"f": "3.14"})
        assert svc.get("f", float) == pytest.approx(3.14)

    def test_bool_from_bool(self):
        svc = ConfigService(defaults={"b": True})
        assert svc.get("b", bool) is True

    def test_bool_from_int(self):
        svc = ConfigService(defaults={"b": 0})
        assert svc.get("b", bool) is False

    def test_already_correct_type(self):
        svc = ConfigService(defaults={"n": 42})
        assert svc.get("n", int) == 42


# ── hot_update + watchers ────────────────────────────────────

class TestHotUpdate:
    def test_updates_value(self):
        svc = ConfigService(defaults={"x": 1})
        svc.hot_update("x", 99)
        assert svc.get("x", int) == 99

    def test_notifies_watchers(self):
        svc = ConfigService(defaults={"x": 1})
        values = []
        svc.watch("x", lambda v: values.append(v))
        svc.hot_update("x", 42)
        assert values == [42]

    def test_multiple_watchers(self):
        svc = ConfigService()
        results = []
        svc.watch("k", lambda v: results.append(("a", v)))
        svc.watch("k", lambda v: results.append(("b", v)))
        svc.hot_update("k", "new")
        assert len(results) == 2


# ── Errors ───────────────────────────────────────────────────

class TestErrors:
    def test_missing_key_raises(self):
        svc = ConfigService()
        with pytest.raises(ConfigKeyError):
            svc.get("nonexistent")

    def test_get_or_default(self):
        svc = ConfigService()
        assert svc.get_or("missing", "fallback") == "fallback"

    def test_invalid_cast_raises(self):
        svc = ConfigService(defaults={"x": "not_a_number"})
        with pytest.raises(ConfigTypeError):
            svc.get("x", int)

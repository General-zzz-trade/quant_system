"""Tests for core.config — ConfigService with layered resolution and hot-reload."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from core.config import ConfigKeyError, ConfigService, ConfigTypeError


class TestConfigServiceDefaults:
    def test_get_default_value(self) -> None:
        svc = ConfigService(defaults={"risk.max_leverage": 3.0})
        assert svc.get("risk.max_leverage", float) == 3.0

    def test_missing_key_raises(self) -> None:
        svc = ConfigService(defaults={})
        try:
            svc.get("nonexistent", str)
            assert False, "Should raise"
        except ConfigKeyError:
            pass

    def test_get_or_returns_default(self) -> None:
        svc = ConfigService(defaults={})
        assert svc.get_or("missing", 42, int) == 42


class TestConfigServiceFileLayer:
    def test_json_file_overrides_default(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"risk.max_leverage": 5.0}, f)
            f.flush()
            try:
                svc = ConfigService(
                    defaults={"risk.max_leverage": 3.0},
                    file_path=f.name,
                )
                assert svc.get("risk.max_leverage", float) == 5.0
            finally:
                os.unlink(f.name)

    def test_missing_file_ignored(self) -> None:
        svc = ConfigService(
            defaults={"key": "val"},
            file_path="/nonexistent/config.json",
        )
        assert svc.get("key", str) == "val"


class TestConfigServiceEnvLayer:
    def test_env_overrides_file(self) -> None:
        os.environ["QS_RISK__MAX_LEVERAGE"] = "10.0"
        try:
            svc = ConfigService(defaults={"risk.max_leverage": 3.0})
            assert svc.get("risk.max_leverage", float) == 10.0
        finally:
            del os.environ["QS_RISK__MAX_LEVERAGE"]


class TestConfigServiceHotReload:
    def test_hot_update_overrides_all(self) -> None:
        svc = ConfigService(defaults={"key": "original"})
        assert svc.get("key", str) == "original"

        svc.hot_update("key", "updated")
        assert svc.get("key", str) == "updated"

    def test_watch_callback_fires(self) -> None:
        svc = ConfigService(defaults={"key": "v1"})
        received: list = []
        svc.watch("key", received.append)

        svc.hot_update("key", "v2")
        assert received == ["v2"]

    def test_watch_multiple_callbacks(self) -> None:
        svc = ConfigService(defaults={"key": "v1"})
        a: list = []
        b: list = []
        svc.watch("key", a.append)
        svc.watch("key", b.append)

        svc.hot_update("key", "v2")
        assert a == ["v2"]
        assert b == ["v2"]


class TestConfigServiceTypeCasting:
    def test_string_to_int(self) -> None:
        svc = ConfigService(defaults={"port": "8080"})
        assert svc.get("port", int) == 8080

    def test_string_to_float(self) -> None:
        svc = ConfigService(defaults={"rate": "0.5"})
        assert svc.get("rate", float) == 0.5

    def test_bad_cast_raises(self) -> None:
        svc = ConfigService(defaults={"key": "not_a_number"})
        try:
            svc.get("key", int)
            assert False, "Should raise"
        except ConfigTypeError:
            pass

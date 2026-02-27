# tests/unit/runner/test_live_runner_wiring.py
"""Tests for LiveRunner production wiring — verifies all modules are connected."""
from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import pytest

from runner.live_runner import LiveRunner, LiveRunnerConfig
from risk.kill_switch_bridge import KillSwitchBridge
from risk.margin_monitor import MarginMonitor
from execution.latency.tracker import LatencyTracker
from monitoring.alerts.manager import AlertManager


class _FakeTransport:
    def __init__(self):
        pass

    def connect(self, url: str) -> None:
        pass

    def recv(self, timeout_s: float = 5.0) -> Optional[str]:
        time.sleep(0.01)
        return None

    def close(self) -> None:
        pass


class _FakeVenueClient:
    def __init__(self) -> None:
        self.orders: List[Any] = []

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        return []


class TestProductionWiring:
    def test_build_with_persistent_stores(self, tmp_path):
        config = LiveRunnerConfig(
            enable_persistent_stores=True,
            data_dir=str(tmp_path / "live_data"),
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.state_store is not None
        # Verify SQLite files were created
        assert os.path.exists(str(tmp_path / "live_data" / "state.db"))
        assert os.path.exists(str(tmp_path / "live_data" / "ack_store.db"))
        assert os.path.exists(str(tmp_path / "live_data" / "event_log.db"))

    def test_build_without_persistent_stores(self):
        config = LiveRunnerConfig(enable_persistent_stores=False)
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.state_store is None

    def test_uses_production_margin_monitor(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
            fetch_margin=lambda: {"margin_ratio": 0.5},
        )
        assert runner.margin_monitor is not None
        assert isinstance(runner.margin_monitor, MarginMonitor)

    def test_latency_tracker_wired(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.latency_tracker is not None
        assert isinstance(runner.latency_tracker, LatencyTracker)

    def test_alert_manager_wired(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.alert_manager is not None
        assert isinstance(runner.alert_manager, AlertManager)

    def test_no_internal_margin_monitor_class(self):
        """Verify the internal _MarginMonitor duplicate was removed."""
        import runner.live_runner as mod
        assert not hasattr(mod, "_MarginMonitor")

    def test_no_internal_kill_switch_adapter(self):
        """Verify the internal _KillSwitchAdapter duplicate was removed."""
        import runner.live_runner as mod
        assert not hasattr(mod, "_KillSwitchAdapter")

    def test_config_new_fields(self):
        config = LiveRunnerConfig(
            data_dir="custom/path",
            enable_persistent_stores=True,
            enable_structured_logging=False,
            log_level="DEBUG",
            log_file="custom.log",
        )
        assert config.data_dir == "custom/path"
        assert config.enable_persistent_stores is True
        assert config.enable_structured_logging is False
        assert config.log_level == "DEBUG"
        assert config.log_file == "custom.log"

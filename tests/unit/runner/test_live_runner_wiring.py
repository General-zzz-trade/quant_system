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

    def test_health_auth_token_env_missing_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_HEALTH_TOKEN", raising=False)
        config = LiveRunnerConfig(
            health_port=18080,
            health_auth_token_env="MISSING_HEALTH_TOKEN",
        )
        with pytest.raises(ValueError, match="health_auth_token_env"):
            LiveRunner.build(
                config,
                venue_clients={"binance": _FakeVenueClient()},
                transport=_FakeTransport(),
            )


def _make_binance_rest_client():
    from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
    return BinanceRestClient(
        cfg=BinanceRestConfig(
            base_url="https://testnet.binancefuture.com",
            api_key="test-key",
            api_secret="test-secret",
        )
    )


class TestUserStreamWiring:
    def test_user_stream_wired_for_binance(self):
        """BinanceRestClient + non-shadow -> user_stream is wired."""
        config = LiveRunnerConfig(
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _make_binance_rest_client()},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is not None

    def test_user_stream_not_wired_in_shadow(self):
        """Shadow mode -> no user stream."""
        config = LiveRunnerConfig(
            shadow_mode=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _make_binance_rest_client()},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is None

    def test_user_stream_not_wired_for_fake_client(self):
        """FakeVenueClient (not BinanceRestClient) -> no user stream."""
        config = LiveRunnerConfig(
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is None

    def test_user_stream_testnet_url(self):
        """testnet=True -> user stream ws_base_url is testnet."""
        config = LiveRunnerConfig(
            testnet=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _make_binance_rest_client()},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is not None
        assert runner.user_stream.cfg.ws_base_url == "wss://stream.binancefuture.com/ws"

    def test_user_stream_production_url(self):
        """testnet=False -> user stream ws_base_url is production."""
        config = LiveRunnerConfig(
            testnet=False,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _make_binance_rest_client()},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is not None
        assert runner.user_stream.cfg.ws_base_url == "wss://fstream.binance.com/ws"

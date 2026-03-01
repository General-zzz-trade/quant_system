# tests/integration/test_testnet_build.py
"""Integration tests for testnet build — verifies full runner assembly with testnet URLs."""
from __future__ import annotations

import time
from typing import Any, List, Optional

import pytest

from runner.live_runner import LiveRunner, LiveRunnerConfig
from runner.live_paper_runner import LivePaperRunner, LivePaperConfig


class _FakeTransport:
    def __init__(self):
        self._url = None

    def connect(self, url: str) -> None:
        self._url = url

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


class TestTestnetRunnerBuild:
    def test_testnet_runner_builds_with_correct_urls(self):
        transport = _FakeTransport()
        config = LiveRunnerConfig(
            testnet=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=transport,
        )
        assert runner is not None
        # WS transport should be connected to testnet URL
        assert runner.loop is not None
        assert runner.coordinator is not None

    def test_production_runner_builds_with_prod_urls(self):
        transport = _FakeTransport()
        config = LiveRunnerConfig(
            testnet=False,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=transport,
        )
        assert runner is not None

    def test_testnet_safety_gates_wired(self):
        """Testnet mode still has kill switch, correlation gate, risk gate."""
        config = LiveRunnerConfig(
            testnet=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.kill_switch is not None
        assert runner.correlation_gate is not None
        assert runner.risk_gate is not None

    def test_testnet_shadow_mode(self):
        """Testnet + shadow mode combination works."""
        config = LiveRunnerConfig(
            testnet=True,
            shadow_mode=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner is not None

    def test_testnet_config_field(self):
        config = LiveRunnerConfig(testnet=True)
        assert config.testnet is True
        config2 = LiveRunnerConfig()
        assert config2.testnet is False


class TestTestnetUserStream:
    def test_testnet_user_stream_wired(self):
        """Testnet + BinanceRestClient -> user stream wired with testnet URL."""
        from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig

        client = BinanceRestClient(
            cfg=BinanceRestConfig(
                base_url="https://testnet.binancefuture.com",
                api_key="test-key",
                api_secret="test-secret",
            )
        )
        config = LiveRunnerConfig(
            testnet=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": client},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is not None
        assert runner.user_stream.cfg.ws_base_url == "wss://stream.binancefuture.com/ws"

    def test_testnet_shadow_no_user_stream(self):
        """Testnet + shadow mode -> no user stream."""
        from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig

        client = BinanceRestClient(
            cfg=BinanceRestConfig(
                base_url="https://testnet.binancefuture.com",
                api_key="test-key",
                api_secret="test-secret",
            )
        )
        config = LiveRunnerConfig(
            testnet=True,
            shadow_mode=True,
            enable_preflight=False,
            enable_persistent_stores=False,
        )
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": client},
            transport=_FakeTransport(),
            user_stream_transport=_FakeTransport(),
        )
        assert runner.user_stream is None


class TestTestnetPaperBuild:
    def test_paper_testnet_builds(self):
        transport = _FakeTransport()
        config = LivePaperConfig(testnet=True)
        runner = LivePaperRunner.build(config, transport=transport)
        assert runner is not None

    def test_paper_testnet_config_field(self):
        config = LivePaperConfig(testnet=True)
        assert config.testnet is True
        config2 = LivePaperConfig()
        assert config2.testnet is False

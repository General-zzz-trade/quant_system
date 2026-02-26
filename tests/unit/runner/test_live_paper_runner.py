# tests/unit/runner/test_live_paper_runner.py
"""Tests for LivePaperRunner — E2E paper trading stack assembly."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from runner.live_paper_runner import LivePaperConfig, LivePaperRunner


# ── Fake WS transport ───────────────────────────────────────

class _FakeTransport:
    """Simulates WsTransport: connect/recv/close."""

    def __init__(self, messages: List[str] | None = None):
        self._messages = list(messages or [])
        self._idx = 0
        self._connected = False

    def connect(self, url: str) -> None:
        self._connected = True

    def recv(self, timeout_s: float = 5.0) -> Optional[str]:
        if self._idx >= len(self._messages):
            time.sleep(0.01)
            return None
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    def close(self) -> None:
        self._connected = False


def _kline_json(
    symbol: str = "BTCUSDT",
    close: str = "40000.00",
    ts_ms: int = 1704067200000,
    closed: bool = True,
) -> str:
    """Build a Binance kline WS JSON message."""
    import json
    return json.dumps({
        "stream": f"{symbol.lower()}@kline_1m",
        "data": {
            "e": "kline",
            "E": ts_ms,
            "s": symbol,
            "k": {
                "t": ts_ms - 60000,
                "T": ts_ms - 1,
                "s": symbol,
                "i": "1m",
                "o": "39900.00",
                "h": "40100.00",
                "l": "39800.00",
                "c": close,
                "v": "100.0",
                "n": 500,
                "x": closed,
                "q": "4000000.0",
            },
        },
    })


# ── Tests ────────────────────────────────────────────────────

class TestBuild:
    def test_build_creates_runner(self):
        config = LivePaperConfig(symbols=("BTCUSDT",))
        transport = _FakeTransport()
        runner = LivePaperRunner.build(config, transport=transport)
        assert runner.coordinator is not None
        assert runner.loop is not None
        assert runner.runtime is not None

    def test_build_without_decision_modules(self):
        config = LivePaperConfig(symbols=("BTCUSDT",))
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        # Should not have decision bridge attached
        assert runner.coordinator._decision_bridge is None

    def test_build_with_decision_module(self):
        config = LivePaperConfig(symbols=("BTCUSDT",), enable_regime_gate=False)

        class _DummyModule:
            def decide(self, snapshot):
                return []

        runner = LivePaperRunner.build(
            config,
            transport=_FakeTransport(),
            decision_modules=[_DummyModule()],
        )
        assert runner.coordinator._decision_bridge is not None

    def test_build_with_regime_gate(self):
        config = LivePaperConfig(symbols=("BTCUSDT",), enable_regime_gate=True)

        class _DummyModule:
            def decide(self, snapshot):
                return []

        from decision.regime_bridge import RegimeAwareDecisionModule
        runner = LivePaperRunner.build(
            config,
            transport=_FakeTransport(),
            decision_modules=[_DummyModule()],
        )
        bridge = runner.coordinator._decision_bridge
        assert bridge is not None
        # Verify module is regime-gated
        assert isinstance(bridge.modules[0], RegimeAwareDecisionModule)

    def test_build_multi_symbol(self):
        config = LivePaperConfig(symbols=("BTCUSDT", "ETHUSDT"))
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        view = runner.coordinator.get_state_view()
        assert "BTCUSDT" in view["markets"]
        assert "ETHUSDT" in view["markets"]

    def test_monitoring_enabled(self):
        config = LivePaperConfig(enable_monitoring=True)
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        assert runner.health is not None

    def test_monitoring_disabled(self):
        config = LivePaperConfig(enable_monitoring=False)
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        assert runner.health is None


class TestMarketDataFlow:
    def test_kline_reaches_pipeline(self):
        """Verify WS kline message flows through runtime → loop → coordinator."""
        messages = [
            _kline_json("BTCUSDT", "40000.00", ts_ms=1704067200000),
            _kline_json("BTCUSDT", "40100.00", ts_ms=1704067260000),
        ]
        transport = _FakeTransport(messages)
        config = LivePaperConfig(symbols=("BTCUSDT",), starting_balance=10000.0)
        runner = LivePaperRunner.build(config, transport=transport)

        runner.coordinator.start()
        runner.runtime.start()
        runner.loop.start_background()

        # Wait for events to flow through
        deadline = time.monotonic() + 5.0
        while runner.event_index < 2 and time.monotonic() < deadline:
            time.sleep(0.05)

        runner.runtime.stop()
        runner.loop.stop_background()
        runner.coordinator.stop()

        assert runner.event_index >= 2

    def test_non_closed_kline_ignored(self):
        """Non-closed klines (x=false) should not produce MarketEvent."""
        messages = [
            _kline_json("BTCUSDT", "40000.00", closed=False),
        ]
        transport = _FakeTransport(messages)
        config = LivePaperConfig(symbols=("BTCUSDT",))
        runner = LivePaperRunner.build(config, transport=transport)

        runner.coordinator.start()
        runner.runtime.start()
        runner.loop.start_background()

        time.sleep(0.5)  # give time for processing

        runner.runtime.stop()
        runner.loop.stop_background()
        runner.coordinator.stop()

        # Non-closed klines don't produce MarketEvent → event_index stays 0
        assert runner.event_index == 0


class TestFillTracking:
    def test_fills_recorded(self):
        config = LivePaperConfig(symbols=("BTCUSDT",))
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        # Initially no fills
        assert len(runner.fills) == 0


class TestLifecycle:
    def test_stop_idempotent(self):
        config = LivePaperConfig(symbols=("BTCUSDT",))
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        runner.stop()  # should not raise
        runner.stop()  # idempotent

    def test_status_logging(self):
        config = LivePaperConfig(symbols=("BTCUSDT",))
        runner = LivePaperRunner.build(config, transport=_FakeTransport())
        runner._log_status()  # should not raise

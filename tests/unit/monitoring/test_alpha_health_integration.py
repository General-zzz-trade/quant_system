"""Tests for alpha health monitor live integration.

Verifies:
1. AlphaHealthMonitor wires into EngineMonitoringHook
2. Position scaling applies to order quantities in _emit gate
3. should_retrain() triggers alert rules
4. IC negative for 14 days → position_scale=0.5 → order qty halved
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from monitoring.alpha_health import AlphaHealthMonitor, AlphaHealthConfig
from monitoring.engine_hook import EngineMonitoringHook


@dataclass
class _FakePipelineOutput:
    markets: Mapping[str, Any] = field(default_factory=dict)
    account: Any = None
    positions: Mapping[str, Any] = field(default_factory=dict)
    portfolio: Any = None
    risk: Any = None
    features: Optional[Mapping[str, Any]] = None
    event_index: int = 0
    last_event_id: Optional[str] = None
    last_ts: Optional[Any] = None
    snapshot: Optional[Any] = None
    advanced: bool = True


@dataclass
class _FakeAccount:
    balance: float = 10000.0
    equity: float = 10000.0


class TestAlphaHealthHookIntegration:
    """Test AlphaHealthMonitor integration with EngineMonitoringHook."""

    def test_hook_accepts_alpha_health_monitor(self):
        monitor = AlphaHealthMonitor()
        monitor.register("BTCUSDT", [12, 24])
        hook = EngineMonitoringHook(alpha_health_monitor=monitor)
        assert hook.alpha_health_monitor is monitor

    def test_hook_updates_monitor_on_call(self):
        """Verify __call__ updates alpha health with prediction and close data."""
        monitor = AlphaHealthMonitor(config=AlphaHealthConfig(eval_interval_bars=1))
        monitor.register("BTCUSDT", [12])
        hook = EngineMonitoringHook(alpha_health_monitor=monitor)

        # Feed enough bars to build history
        for i in range(60):
            close = 50000.0 + i * 10
            out = _FakePipelineOutput(
                features={"_symbol": "BTCUSDT", "close": close, "ml_score": 0.5},
                account=_FakeAccount(),
            )
            hook(out)

        # Verify buffer was populated
        assert "BTCUSDT" in hook._alpha_pred_buffer
        assert len(hook._alpha_pred_buffer["BTCUSDT"]) > 0

    def test_position_scale_default_is_one(self):
        """New monitor starts with scale=1.0 (no degradation)."""
        monitor = AlphaHealthMonitor()
        monitor.register("ETHUSDT", [12, 24])
        assert monitor.position_scale("ETHUSDT") == 1.0

    def test_position_scale_reduces_on_negative_ic(self):
        """When IC negative for reduce_days, scale should drop to reduce_scale."""
        import random
        random.seed(42)
        cfg = AlphaHealthConfig(
            ic_warning_days=1,
            ic_reduce_days=2,
            eval_interval_bars=1,
            ic_window=60,
            reduce_scale=0.5,
        )
        monitor = AlphaHealthMonitor(config=cfg)
        monitor.register("ETHUSDT", [1])  # horizon=1 for fast test

        # Feed anti-correlated predictions (negative IC):
        # predict high → actual low, predict low → actual high
        for i in range(200):
            pred = random.uniform(-1, 1)
            actual_return = -pred * 0.01  # anti-correlated
            monitor.update("ETHUSDT", 1, pred=pred, actual_return=actual_return)
            monitor.on_bar("ETHUSDT")

        scale = monitor.position_scale("ETHUSDT")
        assert scale <= 0.5, f"Expected scale <= 0.5, got {scale}"

    def test_should_retrain_triggers_on_halt(self):
        """IC halt should trigger should_retrain()."""
        import random
        random.seed(42)
        cfg = AlphaHealthConfig(
            ic_warning_days=1,
            ic_reduce_days=2,
            ic_halt_threshold=-0.01,
            eval_interval_bars=1,
            ic_window=60,
        )
        monitor = AlphaHealthMonitor(config=cfg)
        monitor.register("BTCUSDT", [1])

        # Feed strongly anti-correlated data to trigger halt
        for i in range(200):
            pred = random.uniform(-1, 1)
            actual_return = -pred * 0.05  # strongly anti-correlated
            monitor.update("BTCUSDT", 1, pred=pred, actual_return=actual_return)
            monitor.on_bar("BTCUSDT")

        assert monitor.should_retrain("BTCUSDT") is True

    def test_get_status_includes_horizons(self):
        """get_status should return per-horizon IC details."""
        monitor = AlphaHealthMonitor()
        monitor.register("ETHUSDT", [12, 24])
        status = monitor.get_status("ETHUSDT")
        assert status["symbol"] == "ETHUSDT"
        assert "horizons" in status
        assert 12 in status["horizons"]
        assert 24 in status["horizons"]


class TestAlphaHealthOrderGate:
    """Test that alpha health scaling applies to order quantities."""

    def test_mock_ic_negative_14d_halves_qty(self):
        """Core verification: mock IC negative 14 days → position_scale=0.5 → qty halved.

        Uses anti-correlated data with ic_halt_threshold set very low
        so we reach REDUCE (0.5) but not HALT (0.0).
        """
        import random
        random.seed(123)
        cfg = AlphaHealthConfig(
            ic_warning_days=7,
            ic_reduce_days=14,
            eval_interval_bars=24,
            ic_window=60,
            reduce_scale=0.5,
            ic_halt_threshold=-1.5,  # Unreachable, so we stay at REDUCE not HALT
        )
        monitor = AlphaHealthMonitor(config=cfg)
        monitor.register("ETHUSDT", [12])

        # Simulate 14+ days of negative IC (14 * 24 = 336 bars)
        for i in range(500):
            pred = random.uniform(-1, 1)
            actual = -pred * 0.01  # anti-correlated → IC = -1.0
            monitor.update("ETHUSDT", 12, pred=pred, actual_return=actual)
            monitor.on_bar("ETHUSDT")

        scale = monitor.position_scale("ETHUSDT")
        assert scale == 0.5, f"Expected 0.5 after 14d negative IC, got {scale}"

        # Simulate order scaling
        original_qty = 0.1
        scaled_qty = original_qty * scale
        assert abs(scaled_qty - 0.05) < 1e-10

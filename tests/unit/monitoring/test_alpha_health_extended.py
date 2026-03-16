"""Extended tests for AlphaHealthMonitor — registration, IC states, independence."""
from __future__ import annotations

import random

from monitoring.alpha_health import (
    AlphaHealthConfig,
    AlphaHealthMonitor,
)


def _make_monitor(**kwargs) -> AlphaHealthMonitor:
    cfg = AlphaHealthConfig(
        ic_warning_days=2,
        ic_reduce_days=4,
        ic_halt_threshold=-0.02,
        eval_interval_bars=1,
        ic_window=60,
        **kwargs,
    )
    return AlphaHealthMonitor(config=cfg)


def _feed_positive(monitor, symbol, horizons, n=60):
    """Feed positively correlated data to warm up."""
    for i in range(n):
        for h in horizons:
            pred = 0.1 + i * 0.001
            ret = 0.01 + i * 0.0001
            monitor.update(symbol, h, pred, ret)


def _feed_negative(monitor, symbol, horizons, n=60):
    """Feed anti-correlated data to push IC negative."""
    rng = random.Random(42)
    for _ in range(n):
        for h in horizons:
            pred = rng.uniform(-1, 1)
            ret = -pred * 0.01
            monitor.update(symbol, h, pred, ret)


class TestAlphaHealthRegistration:
    def test_register_creates_horizon_states(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12, 24, 48])
        status = monitor.get_status("ETHUSDT")
        assert 12 in status["horizons"]
        assert 24 in status["horizons"]
        assert 48 in status["horizons"]

    def test_register_multiple_symbols(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        monitor.register("BTCUSDT", [24])
        assert monitor.position_scale("ETHUSDT") == 1.0
        assert monitor.position_scale("BTCUSDT") == 1.0

    def test_update_unregistered_symbol_is_noop(self):
        monitor = _make_monitor()
        # Should not raise
        monitor.update("UNKNOWN", 12, 0.5, 0.01)

    def test_update_unregistered_horizon_is_noop(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        # Horizon 99 not registered — should not raise
        monitor.update("ETHUSDT", 99, 0.5, 0.01)


class TestAlphaHealthScaling:
    def test_scale_1_when_healthy(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        _feed_positive(monitor, "ETHUSDT", [12], n=60)
        monitor.check("ETHUSDT")
        assert monitor.position_scale("ETHUSDT") == 1.0

    def test_scale_reduces_after_prolonged_negative_ic(self):
        monitor = _make_monitor(reduce_scale=0.5)
        monitor.register("ETHUSDT", [12])
        _feed_negative(monitor, "ETHUSDT", [12], n=100)
        # Simulate enough bars to exceed reduce threshold (4 days x 24 bars)
        for _ in range(5 * 24):
            monitor.on_bar("ETHUSDT")
        scale = monitor.position_scale("ETHUSDT")
        assert scale <= 0.5

    def test_halt_gives_scale_zero(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        # Directly set halt state
        monitor._state["ETHUSDT"].horizon_states[12].alert_level = "halt"
        assert monitor.position_scale("ETHUSDT") == 0.0

    def test_warning_does_not_reduce_scale(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        monitor._state["ETHUSDT"].horizon_states[12].alert_level = "warning"
        assert monitor.position_scale("ETHUSDT") == 1.0


class TestAlphaHealthPerSymbolIndependence:
    def test_one_symbol_halt_does_not_affect_other(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        monitor.register("BTCUSDT", [24])
        monitor._state["ETHUSDT"].horizon_states[12].alert_level = "halt"
        assert monitor.position_scale("ETHUSDT") == 0.0
        assert monitor.position_scale("BTCUSDT") == 1.0

    def test_should_retrain_per_symbol(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        monitor.register("BTCUSDT", [24])
        monitor._state["ETHUSDT"].horizon_states[12].alert_level = "halt"
        assert monitor.should_retrain("ETHUSDT") is True
        assert monitor.should_retrain("BTCUSDT") is False


class TestAlphaHealthRetrain:
    def test_should_retrain_false_when_ok(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        assert monitor.should_retrain("ETHUSDT") is False

    def test_should_retrain_false_when_disabled(self):
        monitor = _make_monitor(auto_retrain_on_halt=False)
        monitor.register("ETHUSDT", [12])
        monitor._state["ETHUSDT"].horizon_states[12].alert_level = "halt"
        assert monitor.should_retrain("ETHUSDT") is False

    def test_get_horizon_ic_returns_zero_without_data(self):
        monitor = _make_monitor()
        monitor.register("ETHUSDT", [12])
        assert monitor.get_horizon_ic("ETHUSDT", 12) == 0.0

    def test_get_horizon_ic_unknown_symbol(self):
        monitor = _make_monitor()
        assert monitor.get_horizon_ic("UNKNOWN", 12) == 0.0

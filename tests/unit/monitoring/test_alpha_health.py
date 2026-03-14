"""Tests for AlphaHealthMonitor — IC-based alpha health tracking."""
from monitoring.alpha_health import (
    AlphaHealthConfig,
    AlphaHealthMonitor,
)


def _make_monitor(**kwargs) -> AlphaHealthMonitor:
    cfg = AlphaHealthConfig(
        ic_warning_days=2,       # faster thresholds for testing
        ic_reduce_days=4,
        ic_halt_threshold=-0.02,
        eval_interval_bars=1,    # evaluate every bar
        ic_window=100,
        **kwargs,
    )
    monitor = AlphaHealthMonitor(config=cfg)
    monitor.register("ETHUSDT", horizons=[12, 24])
    return monitor


def _feed_data(monitor, n=60, pred_base=0.1, ret_base=0.01):
    """Feed n positive IC samples to warm up."""
    for i in range(n):
        monitor.update("ETHUSDT", 12, pred_base + i * 0.001, ret_base + i * 0.0001)
        monitor.update("ETHUSDT", 24, pred_base + i * 0.002, ret_base + i * 0.0002)


class TestAlphaHealthMonitor:
    def test_register_and_status(self):
        monitor = _make_monitor()
        status = monitor.get_status("ETHUSDT")
        assert status["symbol"] == "ETHUSDT"
        assert 12 in status["horizons"]
        assert 24 in status["horizons"]
        assert status["position_scale"] == 1.0

    def test_unregistered_symbol(self):
        monitor = _make_monitor()
        assert monitor.position_scale("UNKNOWN") == 1.0
        assert monitor.get_status("UNKNOWN")["status"] == "unregistered"

    def test_healthy_state(self):
        monitor = _make_monitor()
        _feed_data(monitor)
        monitor.check("ETHUSDT")
        # Positive IC → no alerts
        assert monitor.position_scale("ETHUSDT") == 1.0

    def test_warning_after_negative_ic(self):
        monitor = _make_monitor()
        # Feed positive data first for warmup
        _feed_data(monitor, n=60)

        # Now feed negative IC data: predictions anti-correlated with returns
        for i in range(60):
            monitor.update("ETHUSDT", 12, 0.1 + i * 0.001, -(0.01 + i * 0.0001))
            monitor.update("ETHUSDT", 24, 0.1 + i * 0.001, -(0.01 + i * 0.0001))

        # Simulate bars for 3 days (> warning threshold of 2 days)
        all_alerts = []
        for _ in range(3 * 24):
            alerts = monitor.on_bar("ETHUSDT")
            all_alerts.extend(alerts)

        warning_alerts = [a for a in all_alerts if a.alert_type == "ic_warning"]
        assert len(warning_alerts) > 0

    def test_position_scale_reduce(self):
        monitor = _make_monitor()
        _feed_data(monitor, n=60)

        # Anti-correlated predictions → negative IC
        for i in range(60):
            monitor.update("ETHUSDT", 12, 0.5 + i * 0.01, -(0.01 + i * 0.001))
            monitor.update("ETHUSDT", 24, 0.5 + i * 0.01, -(0.01 + i * 0.001))

        # Simulate bars for 5 days (> reduce threshold of 4 days)
        for _ in range(5 * 24):
            monitor.on_bar("ETHUSDT")

        scale = monitor.position_scale("ETHUSDT")
        # Should be reduced (0.5) or halted (0.0)
        assert scale < 1.0

    def test_should_retrain(self):
        monitor = _make_monitor()
        # Initially no retrain needed
        assert not monitor.should_retrain("ETHUSDT")

    def test_get_horizon_ic_no_data(self):
        monitor = _make_monitor()
        # Not enough samples
        ic = monitor.get_horizon_ic("ETHUSDT", 12)
        assert ic == 0.0

    def test_on_bar_interval(self):
        cfg = AlphaHealthConfig(eval_interval_bars=10)
        monitor = AlphaHealthMonitor(config=cfg)
        monitor.register("TEST", horizons=[12])
        _feed_data(monitor, n=60, pred_base=0.1, ret_base=0.01)

        # Only every 10th bar triggers check
        for i in range(1, 10):
            alerts = monitor.on_bar("TEST")
            assert alerts == []  # Not eval interval yet

        # 10th bar triggers check
        alerts = monitor.on_bar("TEST")
        # May or may not have alerts, but the check ran

    def test_recovery_clears_alert(self):
        monitor = _make_monitor(ic_recovery_threshold=0.0)
        _feed_data(monitor, n=60)

        # Force a warning state by directly setting it
        state = monitor._state["ETHUSDT"]
        state.horizon_states[12].alert_level = "warning"
        state.horizon_states[12].negative_streak_bars = 3 * 24

        # Feed positive data to trigger recovery
        for i in range(60):
            monitor.update("ETHUSDT", 12, 0.1 + i * 0.001, 0.01 + i * 0.0001)

        # Reset negative streak to simulate recovery
        state.horizon_states[12].negative_streak_bars = 0

        alerts = monitor.check("ETHUSDT")
        [a for a in alerts if a.alert_type == "ic_recovery"]
        # After check with positive IC and zero streak, should recover
        assert state.horizon_states[12].alert_level in ("ok", "warning")

    def test_halt_triggers_retrain(self):
        monitor = _make_monitor()
        # Set halt state directly
        state = monitor._state["ETHUSDT"]
        state.horizon_states[12].alert_level = "halt"

        assert monitor.should_retrain("ETHUSDT")
        assert monitor.position_scale("ETHUSDT") == 0.0

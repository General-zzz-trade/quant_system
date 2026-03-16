"""Extended tests for SystemHealthMonitor — lifecycle, data freshness, drawdown."""
from __future__ import annotations

import time
from decimal import Decimal
from monitoring.alerts.base import Alert, Severity
from monitoring.health import HealthConfig, HealthStatus, SystemHealthMonitor


class _CollectorSink:
    """Collects emitted alerts for assertions."""

    def __init__(self):
        self.alerts: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)


class TestHealthStatusProperties:
    def test_data_age_none_when_no_market_ts(self):
        status = HealthStatus()
        assert status.data_age_sec is None

    def test_data_age_positive_after_market_ts(self):
        status = HealthStatus(last_market_ts=time.monotonic() - 5.0)
        age = status.data_age_sec
        assert age is not None
        assert age >= 4.5

    def test_drawdown_none_when_no_peak(self):
        status = HealthStatus(current_equity=Decimal("9000"))
        assert status.drawdown_pct is None

    def test_drawdown_none_when_peak_zero(self):
        status = HealthStatus(peak_equity=Decimal("0"), current_equity=Decimal("9000"))
        assert status.drawdown_pct is None

    def test_drawdown_calculation(self):
        status = HealthStatus(
            peak_equity=Decimal("10000"),
            current_equity=Decimal("8500"),
        )
        dd = status.drawdown_pct
        assert dd is not None
        assert abs(dd - 15.0) < 0.01

    def test_drawdown_clamps_to_zero(self):
        status = HealthStatus(
            peak_equity=Decimal("10000"),
            current_equity=Decimal("10500"),
        )
        dd = status.drawdown_pct
        assert dd == 0.0


class TestSystemHealthMonitorInitialState:
    def test_initial_status_defaults(self):
        monitor = SystemHealthMonitor()
        status = monitor.get_status()
        assert status.last_market_ts is None
        assert status.last_balance is None
        assert status.peak_equity is None
        assert status.current_equity is None
        assert status.is_connected is False

    def test_config_defaults(self):
        cfg = HealthConfig()
        assert cfg.stale_data_sec == 30.0
        assert cfg.min_balance_usdt == Decimal("100")
        assert cfg.drawdown_warning_pct == 10.0
        assert cfg.drawdown_critical_pct == 20.0


class TestSystemHealthMonitorUpdates:
    def test_on_market_data_updates_ts(self):
        monitor = SystemHealthMonitor()
        monitor.on_market_data()
        status = monitor.get_status()
        assert status.last_market_ts is not None

    def test_on_balance_update_balance_only(self):
        monitor = SystemHealthMonitor()
        monitor.on_balance_update(balance=Decimal("5000"))
        status = monitor.get_status()
        assert status.last_balance == Decimal("5000")
        assert status.current_equity is None

    def test_on_balance_update_equity_sets_peak(self):
        monitor = SystemHealthMonitor()
        monitor.on_balance_update(equity=Decimal("10000"))
        status = monitor.get_status()
        assert status.current_equity == Decimal("10000")
        assert status.peak_equity == Decimal("10000")

    def test_peak_equity_tracks_high_water_mark(self):
        monitor = SystemHealthMonitor()
        monitor.on_balance_update(equity=Decimal("10000"))
        monitor.on_balance_update(equity=Decimal("12000"))
        monitor.on_balance_update(equity=Decimal("11000"))
        status = monitor.get_status()
        assert status.peak_equity == Decimal("12000")
        assert status.current_equity == Decimal("11000")

    def test_on_connection_change_disconnect_emits_alert(self):
        sink = _CollectorSink()
        monitor = SystemHealthMonitor(sink=sink)
        # Must set connected first, then disconnect
        monitor.on_connection_change(connected=True)
        monitor.on_connection_change(connected=False)
        assert len(sink.alerts) >= 1
        disconnect_alerts = [a for a in sink.alerts if "disconnected" in a.title.lower()]
        assert len(disconnect_alerts) == 1
        assert disconnect_alerts[0].severity == Severity.ERROR

    def test_on_connection_change_reconnect_emits_info(self):
        sink = _CollectorSink()
        monitor = SystemHealthMonitor(sink=sink)
        # Starts disconnected, then connect
        monitor.on_connection_change(connected=True)
        reconnect_alerts = [a for a in sink.alerts if "reconnected" in a.title.lower()]
        assert len(reconnect_alerts) == 1
        assert reconnect_alerts[0].severity == Severity.INFO

    def test_connection_same_state_no_alert(self):
        sink = _CollectorSink()
        monitor = SystemHealthMonitor(sink=sink)
        # false -> false should not emit
        monitor.on_connection_change(connected=False)
        assert len(sink.alerts) == 0


class TestSystemHealthMonitorChecks:
    def _make_monitor(self, **cfg_kw):
        sink = _CollectorSink()
        cfg = HealthConfig(**cfg_kw)
        monitor = SystemHealthMonitor(config=cfg, sink=sink)
        return monitor, sink

    def test_stale_data_warning(self):
        monitor, sink = self._make_monitor(stale_data_sec=1.0, max_stale_exit_sec=0)
        # Set market_ts in the past
        monitor._status.last_market_ts = time.monotonic() - 5.0
        monitor._run_checks()
        stale = [a for a in sink.alerts if "stale" in a.title.lower()]
        assert len(stale) == 1
        assert stale[0].severity == Severity.WARNING

    def test_low_balance_warning(self):
        monitor, sink = self._make_monitor(min_balance_usdt=Decimal("1000"))
        monitor.on_balance_update(balance=Decimal("500"))
        monitor._run_checks()
        low = [a for a in sink.alerts if "low balance" in a.title.lower()]
        assert len(low) == 1

    def test_drawdown_warning(self):
        monitor, sink = self._make_monitor(
            drawdown_warning_pct=5.0, drawdown_critical_pct=15.0
        )
        monitor.on_balance_update(equity=Decimal("10000"))
        monitor.on_balance_update(equity=Decimal("9200"))  # 8% drawdown
        monitor._run_checks()
        dd = [a for a in sink.alerts if "drawdown" in a.title.lower()]
        assert len(dd) == 1
        assert dd[0].severity == Severity.WARNING

    def test_drawdown_critical(self):
        monitor, sink = self._make_monitor(
            drawdown_warning_pct=5.0, drawdown_critical_pct=15.0
        )
        monitor.on_balance_update(equity=Decimal("10000"))
        monitor.on_balance_update(equity=Decimal("8000"))  # 20% drawdown
        monitor._run_checks()
        dd = [a for a in sink.alerts if "critical drawdown" in a.title.lower()]
        assert len(dd) == 1
        assert dd[0].severity == Severity.CRITICAL

    def test_on_status_callback(self):
        statuses = []
        monitor = SystemHealthMonitor(on_status=lambda s: statuses.append(s))
        monitor.on_market_data()
        monitor._run_checks()
        assert len(statuses) == 1
        assert statuses[0].last_check_ts is not None

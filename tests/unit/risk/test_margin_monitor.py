# tests/unit/risk/test_margin_monitor.py
"""Tests for MarginMonitor — margin checks, funding alerts, kill switch integration."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from monitoring.alerts.base import Alert, Severity
from risk.kill_switch import KillMode, KillScope, KillSwitch
from risk.margin_monitor import MarginConfig, MarginMonitor


# ── Stubs ────────────────────────────────────────────────────


class _FakeAlertSink:
    def __init__(self) -> None:
        self.alerts: List[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)


def _margin_data(ratio: float = 0.50) -> Dict[str, Any]:
    return {"margin_ratio": ratio, "total_margin": "50000"}


# ── Tests: Warning threshold ─────────────────────────────────


class TestWarningThreshold:
    def test_margin_below_warning_triggers_alert(self):
        sink = _FakeAlertSink()
        cfg = MarginConfig(warning_margin_ratio=0.15, critical_margin_ratio=0.08)
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.12),
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["margin_ok"] is True  # warning, not critical
        assert len(sink.alerts) == 1
        assert sink.alerts[0].severity == Severity.WARNING
        assert "0.1200" in sink.alerts[0].message

    def test_margin_above_warning_no_alert(self):
        sink = _FakeAlertSink()
        cfg = MarginConfig(warning_margin_ratio=0.15, critical_margin_ratio=0.08)
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.25),
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["margin_ok"] is True
        assert len(sink.alerts) == 0


# ── Tests: Critical threshold ────────────────────────────────


class TestCriticalThreshold:
    def test_margin_below_critical_triggers_kill_switch(self):
        sink = _FakeAlertSink()
        ks = KillSwitch()
        cfg = MarginConfig(
            warning_margin_ratio=0.15,
            critical_margin_ratio=0.08,
            auto_reduce_on_critical=True,
        )
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.05),
            kill_switch=ks,
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["margin_ok"] is False
        assert len(sink.alerts) == 1
        assert sink.alerts[0].severity == Severity.CRITICAL

        # Kill switch should be triggered
        rec = ks.is_killed(symbol="BTCUSDT")
        assert rec is not None
        assert rec.mode == KillMode.REDUCE_ONLY
        assert rec.source == "margin_monitor"

    def test_critical_without_auto_reduce_no_kill_switch(self):
        ks = KillSwitch()
        cfg = MarginConfig(
            critical_margin_ratio=0.08,
            auto_reduce_on_critical=False,
        )
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.05),
            kill_switch=ks,
        )

        monitor.check_once()

        assert ks.is_killed(symbol="BTCUSDT") is None

    def test_critical_without_kill_switch_still_alerts(self):
        sink = _FakeAlertSink()
        cfg = MarginConfig(critical_margin_ratio=0.08)
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.05),
            kill_switch=None,
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["margin_ok"] is False
        assert len(sink.alerts) == 1
        assert sink.alerts[0].severity == Severity.CRITICAL


# ── Tests: Funding rate alerts ───────────────────────────────


class TestFundingRate:
    def test_extreme_funding_rate_alert(self):
        sink = _FakeAlertSink()
        cfg = MarginConfig(extreme_funding_threshold=0.001)
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.50),
            fetch_funding=lambda: {"BTCUSDT": 0.002, "ETHUSDT": 0.0005},
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["funding_ok"] is False
        assert "BTCUSDT" in status["extreme_funding"]
        assert "ETHUSDT" not in status["extreme_funding"]
        # One alert for funding (margin is fine)
        assert len(sink.alerts) == 1
        assert sink.alerts[0].severity == Severity.WARNING
        assert "funding" in sink.alerts[0].title.lower()

    def test_normal_funding_no_alert(self):
        sink = _FakeAlertSink()
        cfg = MarginConfig(extreme_funding_threshold=0.001)
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.50),
            fetch_funding=lambda: {"BTCUSDT": 0.0001, "ETHUSDT": -0.0002},
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["funding_ok"] is True
        assert len(sink.alerts) == 0

    def test_negative_extreme_funding_detected(self):
        sink = _FakeAlertSink()
        cfg = MarginConfig(extreme_funding_threshold=0.001)
        monitor = MarginMonitor(
            config=cfg,
            fetch_margin=lambda: _margin_data(ratio=0.50),
            fetch_funding=lambda: {"BTCUSDT": -0.005},
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["funding_ok"] is False
        assert "BTCUSDT" in status["extreme_funding"]


# ── Tests: Error handling ────────────────────────────────────


def _raise_connection_error() -> Dict[str, Any]:
    raise ConnectionError("timeout")


class TestErrorHandling:
    def test_fetch_margin_failure_handled(self):
        monitor = MarginMonitor(
            config=MarginConfig(),
            fetch_margin=_raise_connection_error,
        )

        status = monitor.check_once()

        assert status["margin_ok"] is False
        assert "fetch_margin_failed" in status.get("error", "")

    def test_no_fetch_funding_skips_funding_check(self):
        sink = _FakeAlertSink()
        monitor = MarginMonitor(
            config=MarginConfig(),
            fetch_margin=lambda: _margin_data(ratio=0.50),
            fetch_funding=None,
            alert_sink=sink,
        )

        status = monitor.check_once()

        assert status["funding_ok"] is True
        assert len(sink.alerts) == 0

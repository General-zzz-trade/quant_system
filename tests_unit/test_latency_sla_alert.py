"""Unit tests for latency SLA alert rule logic."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from execution.latency.report import LatencyReporter, LatencyStats
from monitoring.alerts.base import Severity
from monitoring.alerts.manager import AlertManager, AlertRule


def _make_sla_condition(reporter: LatencyReporter, thresh: float = 5000.0):
    """Replicate the SLA condition closure from LiveRunner.build()."""
    def condition(reporter=reporter, thresh=thresh) -> bool:
        stats = reporter.compute_stats()
        for s in stats:
            if s.metric == "signal_to_fill" and s.count >= 10 and s.p99_ms > thresh:
                return True
        return False
    return condition


class TestLatencySlaAlert:

    def test_triggers_when_p99_exceeds_threshold(self):
        reporter = MagicMock(spec=LatencyReporter)
        reporter.compute_stats.return_value = [
            LatencyStats(metric="signal_to_fill", p50_ms=100, p95_ms=200, p99_ms=6000, mean_ms=150, count=20),
        ]

        cond = _make_sla_condition(reporter, thresh=5000.0)
        assert cond() is True

    def test_no_trigger_when_p99_below_threshold(self):
        reporter = MagicMock(spec=LatencyReporter)
        reporter.compute_stats.return_value = [
            LatencyStats(metric="signal_to_fill", p50_ms=100, p95_ms=200, p99_ms=3000, mean_ms=150, count=20),
        ]

        cond = _make_sla_condition(reporter, thresh=5000.0)
        assert cond() is False

    def test_no_trigger_when_count_too_low(self):
        reporter = MagicMock(spec=LatencyReporter)
        reporter.compute_stats.return_value = [
            LatencyStats(metric="signal_to_fill", p50_ms=100, p95_ms=200, p99_ms=9000, mean_ms=150, count=5),
        ]

        cond = _make_sla_condition(reporter, thresh=5000.0)
        assert cond() is False

    def test_no_trigger_for_other_metrics(self):
        reporter = MagicMock(spec=LatencyReporter)
        reporter.compute_stats.return_value = [
            LatencyStats(metric="submit_to_ack", p50_ms=100, p95_ms=200, p99_ms=9000, mean_ms=150, count=20),
        ]

        cond = _make_sla_condition(reporter, thresh=5000.0)
        assert cond() is False

    def test_no_trigger_when_empty_stats(self):
        reporter = MagicMock(spec=LatencyReporter)
        reporter.compute_stats.return_value = []

        cond = _make_sla_condition(reporter, thresh=5000.0)
        assert cond() is False

    def test_alert_manager_fires_on_breach(self):
        reporter = MagicMock(spec=LatencyReporter)
        reporter.compute_stats.return_value = [
            LatencyStats(metric="signal_to_fill", p50_ms=100, p95_ms=200, p99_ms=7000, mean_ms=150, count=15),
        ]

        manager = AlertManager()
        manager.add_rule(AlertRule(
            name="latency_sla_breach",
            condition=_make_sla_condition(reporter, thresh=5000.0),
            severity=Severity.WARNING,
            message_template="P99 exceeds threshold",
            cooldown_sec=0.0,
        ))

        alerts = manager.check_all()
        assert len(alerts) == 1
        assert alerts[0].title == "latency_sla_breach"

"""Tests for the rule-based AlertManager."""
from __future__ import annotations

import time
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from monitoring.alerts.base import Alert, Severity
from monitoring.alerts.manager import AlertManager, AlertRule


# ── Helpers ──────────────────────────────────────────────────


class _RecordingSink:
    """In-memory sink that records emitted alerts."""

    def __init__(self) -> None:
        self.alerts: List[Alert] = []

    def emit(self, alert: Alert) -> None:
        self.alerts.append(alert)


def _make_rule(
    name: str = "test-rule",
    condition: Any = None,
    severity: Severity = Severity.WARNING,
    cooldown: float = 0.0,
) -> AlertRule:
    return AlertRule(
        name=name,
        condition=condition or (lambda: True),
        severity=severity,
        message_template=f"{name} triggered",
        cooldown_sec=cooldown,
    )


# ── Tests ────────────────────────────────────────────────────


class TestAlertManagerRules:
    def test_check_all_fires_matching_rules(self):
        sink = _RecordingSink()
        mgr = AlertManager(sink=sink, rules=[_make_rule("r1"), _make_rule("r2")])
        fired = mgr.check_all()
        assert len(fired) == 2
        assert len(sink.alerts) == 2

    def test_false_condition_does_not_fire(self):
        sink = _RecordingSink()
        rule = _make_rule("never", condition=lambda: False)
        mgr = AlertManager(sink=sink, rules=[rule])
        fired = mgr.check_all()
        assert len(fired) == 0
        assert len(sink.alerts) == 0

    def test_alert_fields(self):
        sink = _RecordingSink()
        rule = _make_rule("check-balance", severity=Severity.ERROR)
        mgr = AlertManager(sink=sink, rules=[rule])
        mgr.check_all()
        alert = sink.alerts[0]
        assert alert.title == "check-balance"
        assert alert.severity == Severity.ERROR
        assert alert.source == "alert_manager"
        assert alert.ts is not None

    def test_add_rule(self):
        mgr = AlertManager()
        assert len(mgr.rules) == 0
        mgr.add_rule(_make_rule("dynamic"))
        assert len(mgr.rules) == 1

    def test_remove_rule(self):
        mgr = AlertManager(rules=[_make_rule("a"), _make_rule("b")])
        mgr.remove_rule("a")
        assert len(mgr.rules) == 1
        assert mgr.rules[0].name == "b"


class TestAlertManagerCooldown:
    def test_cooldown_prevents_duplicate(self):
        sink = _RecordingSink()
        rule = _make_rule("cooldown-test", cooldown=1000.0)
        mgr = AlertManager(sink=sink, rules=[rule])

        mgr.check_all()
        mgr.check_all()
        mgr.check_all()

        assert len(sink.alerts) == 1

    def test_zero_cooldown_always_fires(self):
        sink = _RecordingSink()
        rule = _make_rule("no-cooldown", cooldown=0.0)
        mgr = AlertManager(sink=sink, rules=[rule])

        mgr.check_all()
        mgr.check_all()
        mgr.check_all()

        assert len(sink.alerts) == 3

    def test_independent_cooldowns(self):
        sink = _RecordingSink()
        r1 = _make_rule("r1", cooldown=1000.0)
        r2 = _make_rule("r2", cooldown=0.0)
        mgr = AlertManager(sink=sink, rules=[r1, r2])

        mgr.check_all()
        mgr.check_all()

        # r1 fires once (cooldown), r2 fires twice (no cooldown)
        assert len(sink.alerts) == 3


class TestAlertManagerErrorHandling:
    def test_condition_exception_does_not_crash(self):
        sink = _RecordingSink()
        bad_rule = _make_rule("bad", condition=lambda: 1 / 0)
        good_rule = _make_rule("good")
        mgr = AlertManager(sink=sink, rules=[bad_rule, good_rule])

        fired = mgr.check_all()
        assert len(fired) == 1
        assert fired[0].title == "good"

    def test_sink_exception_does_not_crash(self):
        bad_sink = MagicMock()
        bad_sink.emit.side_effect = RuntimeError("sink down")
        mgr = AlertManager(sink=bad_sink, rules=[_make_rule()])
        fired = mgr.check_all()
        assert len(fired) == 1

    def test_no_sink_still_returns_alerts(self):
        mgr = AlertManager(rules=[_make_rule()])
        fired = mgr.check_all()
        assert len(fired) == 1

    def test_emit_direct_forwards_to_sink(self):
        sink = _RecordingSink()
        mgr = AlertManager(sink=sink)
        alert = Alert(title="manual-halt", message="operator halted trading", severity=Severity.CRITICAL)

        returned = mgr.emit_direct(alert)

        assert returned is alert
        assert sink.alerts == [alert]

    def test_history_tracks_direct_and_rule_alerts(self):
        sink = _RecordingSink()
        mgr = AlertManager(sink=sink, rules=[_make_rule("rule-alert")], max_history=4)

        mgr.check_all()
        mgr.emit_direct(
            Alert(
                title="exec-timeout",
                message="timeout",
                severity=Severity.WARNING,
                source="execution:test",
                meta={"category": "execution_timeout"},
            )
        )

        history = mgr.history(limit=10)
        assert len(history) == 2
        assert history[0].title == "exec-timeout"
        assert history[1].title == "rule-alert"

    def test_history_can_filter_by_category(self):
        mgr = AlertManager(max_history=4)
        mgr.emit_direct(Alert(title="a", message="x", meta={"category": "execution_timeout"}))
        mgr.emit_direct(Alert(title="b", message="y", meta={"category": "execution_reconcile"}))

        history = mgr.history(limit=10, category="execution_timeout")

        assert len(history) == 1
        assert history[0].title == "a"


class TestAlertManagerPeriodic:
    def test_start_stop(self):
        mgr = AlertManager()
        mgr.start_periodic(interval_sec=0.05)
        time.sleep(0.02)
        mgr.stop()

    def test_periodic_fires_rules(self):
        sink = _RecordingSink()
        rule = _make_rule("periodic-test", cooldown=0.0)
        mgr = AlertManager(sink=sink, rules=[rule])
        mgr.start_periodic(interval_sec=0.02)
        time.sleep(0.08)
        mgr.stop()
        assert len(sink.alerts) >= 1

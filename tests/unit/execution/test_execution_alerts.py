from __future__ import annotations

from execution.observability.alerts import build_execution_alert
from execution.observability.incidents import (
    reconcile_report_to_alert,
    synthetic_fill_to_alert,
    timeout_to_alert,
)
from monitoring.alerts.base import Severity


def test_build_execution_alert_sets_taxonomy_fields() -> None:
    alert = build_execution_alert(
        title="execution-timeout",
        message="order timed out",
        severity=Severity.WARNING,
        category="execution_timeout",
        routing_key="binance:BTCUSDT:timeout",
        source="execution:test",
        meta={"order_id": "ord-1"},
    )

    assert alert.title == "execution-timeout"
    assert alert.severity == Severity.WARNING
    assert alert.meta is not None
    assert alert.meta["category"] == "execution_timeout"
    assert alert.meta["routing_key"] == "binance:BTCUSDT:timeout"
    assert alert.meta["order_id"] == "ord-1"


def test_timeout_to_alert_builds_timeout_taxonomy() -> None:
    alert = timeout_to_alert(venue="binance", symbol="BTCUSDT", order_id="ord-1", timeout_sec=30.0)
    assert alert.title == "execution-timeout"
    assert alert.meta["category"] == "execution_timeout"
    assert alert.meta["routing_key"] == "binance:BTCUSDT:timeout"
    assert alert.meta["timeout_sec"] == 30.0


def test_reconcile_report_to_alert_builds_drift_taxonomy() -> None:
    drift = type("D", (), {"symbol": "BTCUSDT"})()
    report = type("R", (), {"venue": "binance", "all_drifts": [drift], "should_halt": True})()
    alert = reconcile_report_to_alert(report)
    assert alert.title == "execution-reconcile-drift"
    assert alert.severity == Severity.ERROR
    assert alert.meta["category"] == "execution_reconcile"
    assert alert.meta["routing_key"] == "binance:BTCUSDT:critical"
    assert alert.meta["drift_count"] == 1


def test_synthetic_fill_to_alert_builds_fill_taxonomy() -> None:
    fill = type(
        "Fill",
        (),
        {"venue": "binance", "symbol": "ETHUSDT", "fill_id": "f-1", "order_id": "o-1", "qty": "1.5", "side": "buy"},
    )()
    alert = synthetic_fill_to_alert(fill)
    assert alert.title == "execution-synthetic-fill"
    assert alert.meta["category"] == "execution_fill"
    assert alert.meta["routing_key"] == "binance:ETHUSDT:synthetic_fill"
    assert alert.meta["synthetic"] is True

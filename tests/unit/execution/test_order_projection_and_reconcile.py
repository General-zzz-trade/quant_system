from __future__ import annotations

from decimal import Decimal

from execution.reconcile.drift import DriftSeverity, DriftType
from execution.reconcile.orders import reconcile_orders
from execution.state_machine.projection import project_order
from execution.state_machine.reconciliation_rules import reconcile_order
from execution.state_machine.transitions import OrderStatus


def test_project_order_tracks_latest_status_and_quantities() -> None:
    projection = project_order(
        [
            {"order_id": "ord-1", "status": "pending_new", "qty": "1.0", "filled_qty": "0", "ts_ms": 1000},
            {"order_id": "ord-1", "status": "new", "qty": "1.0", "filled_qty": "0", "ts_ms": 1100},
            {"order_id": "ord-1", "status": "partially_filled", "filled_qty": "0.4", "avg_price": "42500",
                "ts_ms": 1200},
            {"order_id": "ord-1", "status": "filled", "filled_qty": "1.0", "avg_price": "42600", "ts_ms": 1300},
        ]
    )

    assert projection is not None
    assert projection.order_id == "ord-1"
    assert projection.status is OrderStatus.FILLED
    assert projection.qty == Decimal("1.0")
    assert projection.filled_qty == Decimal("1.0")
    assert projection.avg_price == Decimal("42600")
    assert projection.last_ts_ms == 1300


def test_project_order_ignores_invalid_partial_fields_and_keeps_last_valid_values() -> None:
    projection = project_order(
        [
            {"order_id": "ord-1", "status": "new", "qty": "1.0", "filled_qty": "0.1", "avg_price": "42000",
                "ts_ms": 1000},
            {"order_id": "ord-1", "status": "bogus", "qty": "bad", "filled_qty": "still-bad", "avg_price": "nan-ish",
                "ts_ms": "bad-ts"},
        ]
    )

    assert projection is not None
    assert projection.status is OrderStatus.NEW
    assert projection.qty == Decimal("1.0")
    assert projection.filled_qty == Decimal("0.1")
    assert projection.avg_price == Decimal("42000")
    assert projection.last_ts_ms == 1000


def test_reconcile_order_marks_terminal_status_mismatch_as_critical() -> None:
    result = reconcile_order(
        order_id="ord-1",
        expected_status=OrderStatus.NEW,
        actual_status=OrderStatus.FILLED,
        expected_qty=Decimal("1"),
        actual_qty=Decimal("1"),
        expected_filled=Decimal("0"),
        actual_filled=Decimal("1"),
    )

    assert result.severity is DriftSeverity.CRITICAL
    assert result.status_match is False
    assert result.filled_qty_match is False


def test_reconcile_orders_reports_missing_active_local_order_as_critical_drift() -> None:
    result = reconcile_orders(
        venue="binance",
        local_orders={"ord-1": "new"},
        venue_orders={},
    )

    assert result.ok is True
    assert result.missing_venue == 1
    assert len(result.drifts) == 1
    drift = result.drifts[0]
    assert drift.drift_type is DriftType.ORDER_STATUS
    assert drift.severity is DriftSeverity.CRITICAL
    assert drift.expected == "new"
    assert drift.actual == "not found"


def test_reconcile_orders_reports_missing_terminal_local_order_as_warning_drift() -> None:
    result = reconcile_orders(
        venue="binance",
        local_orders={"ord-1": "canceled"},
        venue_orders={},
    )

    assert result.missing_venue == 1
    assert len(result.drifts) == 1
    assert result.drifts[0].severity is DriftSeverity.WARNING

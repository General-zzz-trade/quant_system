"""Tests for reconciliation backfill."""
from __future__ import annotations

from decimal import Decimal
from execution.reconcile.drift import detect_qty_drift


def test_qty_drift_zero():
    drift = detect_qty_drift(
        symbol="BTCUSDT", venue="binance",
        expected_qty=Decimal("1.0"), actual_qty=Decimal("1.0"),
    )
    assert drift is None


def test_qty_drift_detected():
    drift = detect_qty_drift(
        symbol="BTCUSDT", venue="binance",
        expected_qty=Decimal("1.0"), actual_qty=Decimal("1.5"),
    )
    assert drift is not None

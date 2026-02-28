"""Unit tests for reconciliation pure-logic functions.

Covers: drift detection, fill/position/balance/order reconciliation, policies.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.reconcile.drift import (
    Drift, DriftSeverity, DriftType,
    detect_qty_drift, detect_balance_drift,
)
from execution.reconcile.fills import FillRecord, reconcile_fills
from execution.reconcile.positions import reconcile_positions
from execution.reconcile.balances import reconcile_balances
from execution.reconcile.orders import reconcile_orders
from execution.reconcile.policies import ReconcileAction, ReconcilePolicy


# ── drift.py ──────────────────────────────────────────────


class TestDetectQtyDrift:

    def test_within_tolerance_returns_none(self):
        assert detect_qty_drift(
            symbol="BTC", venue="binance",
            expected_qty=Decimal("1.0"), actual_qty=Decimal("1.000005"),
        ) is None

    def test_info_severity(self):
        # diff > tolerance but <= tolerance*100 and <= 10% of expected
        d = detect_qty_drift(
            symbol="BTC", venue="binance",
            expected_qty=Decimal("10.0"), actual_qty=Decimal("10.0005"),
            tolerance=Decimal("0.0001"),
        )
        assert d is not None
        assert d.severity == DriftSeverity.INFO

    def test_warning_severity(self):
        # diff > tolerance*100 but <= 10% of expected
        d = detect_qty_drift(
            symbol="BTC", venue="binance",
            expected_qty=Decimal("100.0"), actual_qty=Decimal("100.5"),
            tolerance=Decimal("0.001"),
        )
        assert d is not None
        assert d.severity == DriftSeverity.WARNING

    def test_critical_severity(self):
        # diff > 10% of expected
        d = detect_qty_drift(
            symbol="BTC", venue="binance",
            expected_qty=Decimal("1.0"), actual_qty=Decimal("1.2"),
        )
        assert d is not None
        assert d.severity == DriftSeverity.CRITICAL

    def test_drift_type_is_position_qty(self):
        d = detect_qty_drift(
            symbol="BTC", venue="binance",
            expected_qty=Decimal("1.0"), actual_qty=Decimal("2.0"),
        )
        assert d.drift_type == DriftType.POSITION_QTY


class TestDetectBalanceDrift:

    def test_within_tolerance_returns_none(self):
        assert detect_balance_drift(
            venue="binance", asset="USDT",
            expected=Decimal("10000"), actual=Decimal("10000.005"),
        ) is None

    def test_warning_severity(self):
        # diff > tolerance but <= 5% of expected
        d = detect_balance_drift(
            venue="binance", asset="USDT",
            expected=Decimal("10000"), actual=Decimal("10100"),
        )
        assert d is not None
        assert d.severity == DriftSeverity.WARNING

    def test_critical_severity(self):
        # diff > 5% of expected
        d = detect_balance_drift(
            venue="binance", asset="USDT",
            expected=Decimal("10000"), actual=Decimal("10600"),
        )
        assert d is not None
        assert d.severity == DriftSeverity.CRITICAL


# ── fills.py ──────────────────────────────────────────────


class TestReconcileFills:

    def test_all_matched(self):
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1", "f2"}, venue_fill_ids={"f1", "f2"},
        )
        assert r.ok
        assert r.matched == 2

    def test_missing_local(self):
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1"}, venue_fill_ids={"f1", "f2"},
        )
        assert not r.ok
        assert r.missing_local == 1
        assert any(d.drift_type == DriftType.FILL_MISSING for d in r.drifts)

    def test_missing_venue(self):
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1", "f2"}, venue_fill_ids={"f1"},
        )
        assert not r.ok
        assert r.missing_venue == 1
        assert any(d.drift_type == DriftType.FILL_EXTRA for d in r.drifts)

    def test_price_mismatch_warning(self):
        local = {"f1": FillRecord("f1", "BTC", "buy", 1.0, 100.0)}
        venue = {"f1": FillRecord("f1", "BTC", "buy", 1.0, 100.5)}
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1"}, venue_fill_ids={"f1"},
            local_fills=local, venue_fills=venue,
            price_tolerance_pct=0.001,
        )
        assert r.price_mismatches == 1
        assert any(d.severity == DriftSeverity.WARNING for d in r.drifts)

    def test_price_mismatch_critical(self):
        local = {"f1": FillRecord("f1", "BTC", "buy", 1.0, 100.0)}
        venue = {"f1": FillRecord("f1", "BTC", "buy", 1.0, 102.0)}
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1"}, venue_fill_ids={"f1"},
            local_fills=local, venue_fills=venue,
            price_tolerance_pct=0.001,
        )
        assert any(d.severity == DriftSeverity.CRITICAL for d in r.drifts)

    def test_qty_mismatch(self):
        local = {"f1": FillRecord("f1", "BTC", "buy", 1.0, 100.0)}
        venue = {"f1": FillRecord("f1", "BTC", "buy", 1.5, 100.0)}
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1"}, venue_fill_ids={"f1"},
            local_fills=local, venue_fills=venue,
        )
        assert r.qty_mismatches == 1

    def test_id_only_mode(self):
        r = reconcile_fills(
            venue="binance", symbol="BTC",
            local_fill_ids={"f1", "f2"}, venue_fill_ids={"f1", "f3"},
        )
        assert r.missing_local == 1
        assert r.missing_venue == 1
        assert r.price_mismatches == 0


# ── positions.py ──────────────────────────────────────────


class TestReconcilePositions:

    def test_matched(self):
        r = reconcile_positions(
            venue="binance",
            local_positions={"BTC": Decimal("1.0")},
            venue_positions={"BTC": Decimal("1.0")},
        )
        assert r.ok
        assert r.matched == 1

    def test_missing_local(self):
        r = reconcile_positions(
            venue="binance",
            local_positions={},
            venue_positions={"BTC": Decimal("1.0")},
        )
        assert not r.ok
        assert r.missing_local == 1
        assert r.drifts[0].severity == DriftSeverity.CRITICAL

    def test_missing_venue(self):
        r = reconcile_positions(
            venue="binance",
            local_positions={"BTC": Decimal("1.0")},
            venue_positions={},
        )
        assert not r.ok
        assert r.missing_venue == 1
        assert r.drifts[0].severity == DriftSeverity.WARNING

    def test_qty_drift(self):
        r = reconcile_positions(
            venue="binance",
            local_positions={"BTC": Decimal("1.0")},
            venue_positions={"BTC": Decimal("1.2")},
        )
        assert r.mismatched == 1


# ── balances.py ───────────────────────────────────────────


class TestReconcileBalances:

    def test_matched(self):
        r = reconcile_balances(
            venue="binance",
            local_balances={"USDT": Decimal("10000")},
            venue_balances={"USDT": Decimal("10000")},
        )
        assert r.ok
        assert r.matched == 1

    def test_drift(self):
        r = reconcile_balances(
            venue="binance",
            local_balances={"USDT": Decimal("10000")},
            venue_balances={"USDT": Decimal("10200")},
        )
        assert not r.ok
        assert r.mismatched == 1

    def test_multi_asset(self):
        r = reconcile_balances(
            venue="binance",
            local_balances={"USDT": Decimal("10000"), "BTC": Decimal("0.5")},
            venue_balances={"USDT": Decimal("10000"), "BTC": Decimal("0.5")},
        )
        assert r.ok
        assert r.matched == 2


# ── orders.py ─────────────────────────────────────────────


class TestReconcileOrders:

    def test_matched(self):
        r = reconcile_orders(
            venue="binance",
            local_orders={"o1": "filled"}, venue_orders={"o1": "filled"},
        )
        assert r.ok
        assert r.matched == 1

    def test_terminal_mismatch_critical(self):
        r = reconcile_orders(
            venue="binance",
            local_orders={"o1": "open"}, venue_orders={"o1": "filled"},
        )
        assert not r.ok
        assert r.status_mismatch == 1
        assert r.drifts[0].severity == DriftSeverity.CRITICAL

    def test_non_terminal_mismatch_warning(self):
        r = reconcile_orders(
            venue="binance",
            local_orders={"o1": "open"}, venue_orders={"o1": "partial"},
        )
        assert r.drifts[0].severity == DriftSeverity.WARNING

    def test_missing_local(self):
        r = reconcile_orders(
            venue="binance",
            local_orders={}, venue_orders={"o1": "open"},
        )
        assert r.missing_local == 1
        assert r.drifts[0].severity == DriftSeverity.WARNING

    def test_missing_venue_no_drift(self):
        r = reconcile_orders(
            venue="binance",
            local_orders={"o1": "open"}, venue_orders={},
        )
        assert r.missing_venue == 1
        assert len(r.drifts) == 0  # missing_venue doesn't produce drift


# ── policies.py ───────────────────────────────────────────


def _drift(severity: DriftSeverity) -> Drift:
    return Drift(
        drift_type=DriftType.POSITION_QTY,
        severity=severity,
        venue="binance", symbol="BTC",
        expected="1.0", actual="1.1",
        detail="test",
    )


class TestReconcilePolicy:

    def test_none_drift_accepts(self):
        p = ReconcilePolicy()
        d = p.decide(_drift(DriftSeverity.NONE))
        assert d.action == ReconcileAction.ACCEPT

    def test_info_auto_accept(self):
        p = ReconcilePolicy(auto_accept_info=True)
        d = p.decide(_drift(DriftSeverity.INFO))
        assert d.action == ReconcileAction.ACCEPT

    def test_info_no_auto_accept(self):
        p = ReconcilePolicy(auto_accept_info=False)
        d = p.decide(_drift(DriftSeverity.INFO))
        assert d.action == ReconcileAction.ALERT

    def test_warning_auto_accept(self):
        p = ReconcilePolicy(auto_accept_warning=True)
        d = p.decide(_drift(DriftSeverity.WARNING))
        assert d.action == ReconcileAction.ACCEPT

    def test_warning_no_auto_accept(self):
        p = ReconcilePolicy(auto_accept_warning=False)
        d = p.decide(_drift(DriftSeverity.WARNING))
        assert d.action == ReconcileAction.ALERT

    def test_critical_halt(self):
        p = ReconcilePolicy(halt_on_critical=True)
        d = p.decide(_drift(DriftSeverity.CRITICAL))
        assert d.action == ReconcileAction.HALT

    def test_critical_manual_review(self):
        p = ReconcilePolicy(halt_on_critical=False)
        d = p.decide(_drift(DriftSeverity.CRITICAL))
        assert d.action == ReconcileAction.MANUAL_REVIEW

    def test_batch(self):
        p = ReconcilePolicy()
        drifts = [_drift(DriftSeverity.NONE), _drift(DriftSeverity.CRITICAL)]
        decisions = p.decide_batch(drifts)
        assert len(decisions) == 2
        assert decisions[0].action == ReconcileAction.ACCEPT
        assert decisions[1].action == ReconcileAction.HALT

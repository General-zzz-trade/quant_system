# tests/unit/execution/test_healer.py
"""Tests for reconciliation healer."""
from __future__ import annotations

from decimal import Decimal
from typing import List

import pytest

from execution.reconcile.healer import HealAction, ReconcileHealer
from execution.reconcile.controller import ReconcileReport
from execution.reconcile.drift import Drift, DriftType, DriftSeverity
from execution.reconcile.policies import PolicyDecision, ReconcileAction


def _make_report(
    decisions: list[PolicyDecision],
    venue: str = "binance",
) -> ReconcileReport:
    return ReconcileReport(venue=venue, decisions=tuple(decisions))


def _pos_drift(
    symbol: str = "BTCUSDT",
    expected: str = "1.0",
    actual: str = "1.1",
    severity: DriftSeverity = DriftSeverity.INFO,
) -> Drift:
    return Drift(
        drift_type=DriftType.POSITION_QTY,
        severity=severity,
        venue="binance",
        symbol=symbol,
        expected=expected,
        actual=actual,
        detail=f"diff={abs(Decimal(actual) - Decimal(expected))}",
    )


def _bal_drift(
    asset: str = "USDT",
    expected: str = "10000",
    actual: str = "10050",
    severity: DriftSeverity = DriftSeverity.WARNING,
) -> Drift:
    return Drift(
        drift_type=DriftType.BALANCE,
        severity=severity,
        venue="binance",
        symbol=asset,
        expected=expected,
        actual=actual,
        detail=f"diff={abs(Decimal(actual) - Decimal(expected))}",
    )


class TestReconcileHealer:
    def test_no_accepted_drifts(self):
        healer = ReconcileHealer()
        drift = _pos_drift()
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ALERT, reason="alert", drift=drift),
        ])
        actions = healer.heal(report)
        assert len(actions) == 0

    def test_heal_position_drift(self):
        adjustments: List[tuple] = []

        def on_adjust(symbol: str, qty: Decimal) -> None:
            adjustments.append((symbol, qty))

        healer = ReconcileHealer(emit_position_adjust=on_adjust)
        drift = _pos_drift("BTCUSDT", "1.0", "1.1")
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="auto", drift=drift),
        ])

        actions = healer.heal(report)
        assert len(actions) == 1
        assert actions[0].correction_type == "position_adjust"
        assert actions[0].symbol == "BTCUSDT"
        assert len(adjustments) == 1
        assert adjustments[0] == ("BTCUSDT", Decimal("0.1"))

    def test_heal_balance_drift(self):
        adjustments: List[tuple] = []

        def on_adjust(asset: str, amount: Decimal) -> None:
            adjustments.append((asset, amount))

        healer = ReconcileHealer(emit_balance_adjust=on_adjust)
        drift = _bal_drift("USDT", "10000", "10050")
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="auto", drift=drift),
        ])

        actions = healer.heal(report)
        assert len(actions) == 1
        assert actions[0].correction_type == "balance_adjust"
        assert adjustments[0] == ("USDT", Decimal("50"))

    def test_skip_none_severity(self):
        healer = ReconcileHealer(
            emit_position_adjust=lambda s, q: None,
        )
        drift = Drift(
            drift_type=DriftType.POSITION_QTY,
            severity=DriftSeverity.NONE,
            venue="binance", symbol="X", expected="0", actual="0",
        )
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="ok", drift=drift),
        ])
        actions = healer.heal(report)
        assert len(actions) == 0

    def test_audit_log(self):
        healer = ReconcileHealer(
            emit_position_adjust=lambda s, q: None,
        )
        drift = _pos_drift()
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="auto", drift=drift),
        ])

        healer.heal(report)
        assert len(healer.audit_log) == 1

        healer.heal(report)
        assert len(healer.audit_log) == 2

        healer.clear_audit_log()
        assert len(healer.audit_log) == 0

    def test_max_auto_heal_per_cycle(self):
        count = 0

        def on_adjust(s, q):
            nonlocal count
            count += 1

        healer = ReconcileHealer(
            emit_position_adjust=on_adjust,
            max_auto_heal_per_cycle=2,
        )

        decisions = []
        for i in range(5):
            d = _pos_drift(f"SYM{i}", "1.0", "1.1")
            decisions.append(PolicyDecision(action=ReconcileAction.ACCEPT, reason="auto", drift=d))

        report = _make_report(decisions)
        actions = healer.heal(report)
        assert len(actions) == 2
        assert count == 2

    def test_callback_error_handled(self):
        def failing_adjust(s, q):
            raise RuntimeError("oops")

        healer = ReconcileHealer(emit_position_adjust=failing_adjust)
        drift = _pos_drift()
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="auto", drift=drift),
        ])
        actions = healer.heal(report)
        assert len(actions) == 0

    def test_no_callback_no_action(self):
        healer = ReconcileHealer()  # no callbacks
        drift = _pos_drift()
        report = _make_report([
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="auto", drift=drift),
        ])
        actions = healer.heal(report)
        assert len(actions) == 0

    def test_halt_drifts_not_healed(self):
        healer = ReconcileHealer(
            emit_position_adjust=lambda s, q: None,
        )
        drift = _pos_drift(severity=DriftSeverity.CRITICAL)
        report = _make_report([
            PolicyDecision(action=ReconcileAction.HALT, reason="halt", drift=drift),
        ])
        actions = healer.heal(report)
        assert len(actions) == 0

    def test_mixed_drifts(self):
        pos_adjs: List[tuple] = []
        bal_adjs: List[tuple] = []

        healer = ReconcileHealer(
            emit_position_adjust=lambda s, q: pos_adjs.append((s, q)),
            emit_balance_adjust=lambda s, q: bal_adjs.append((s, q)),
        )

        decisions = [
            PolicyDecision(
                action=ReconcileAction.ACCEPT, reason="auto",
                drift=_pos_drift("BTCUSDT", "1.0", "1.05"),
            ),
            PolicyDecision(
                action=ReconcileAction.ACCEPT, reason="auto",
                drift=_bal_drift("USDT", "1000", "1010"),
            ),
            PolicyDecision(
                action=ReconcileAction.ALERT, reason="review",
                drift=_pos_drift("ETHUSDT", "5.0", "6.0"),
            ),
        ]

        report = _make_report(decisions)
        actions = healer.heal(report)
        assert len(actions) == 2
        assert len(pos_adjs) == 1
        assert len(bal_adjs) == 1

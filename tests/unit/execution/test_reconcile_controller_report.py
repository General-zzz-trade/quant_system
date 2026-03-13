from __future__ import annotations

from decimal import Decimal

from execution.reconcile.controller import ReconcileController, ReconcileReport
from execution.reconcile.drift import DriftSeverity, DriftType
from execution.reconcile.policies import PolicyDecision, ReconcileAction


def test_reconcile_controller_report_aggregates_drifts_and_decisions_in_stable_order() -> None:
    controller = ReconcileController()

    report = controller.reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal("1.0")},
        venue_positions={"BTCUSDT": Decimal("0.5")},
        local_balances={"USDT": Decimal("1000")},
        venue_balances={"USDT": Decimal("900")},
        local_fill_ids={"fill-1"},
        venue_fill_ids={"fill-1", "fill-2"},
        fill_symbol="BTCUSDT",
        local_orders={"ord-1": "new"},
        venue_orders={"ord-1": "filled"},
    )

    drifts = list(report.all_drifts)
    decisions = list(report.decisions)

    assert [d.drift_type for d in drifts] == [
        DriftType.POSITION_QTY,
        DriftType.BALANCE,
        DriftType.FILL_MISSING,
        DriftType.ORDER_STATUS,
    ]
    assert [d.drift for d in decisions] == drifts
    assert report.ok is False
    assert report.should_halt is True


def test_reconcile_report_ok_reflects_part_results_not_policy_actions() -> None:
    drift = report_drift = next(
        iter(
            ReconcileController().reconcile(
                venue="binance",
                local_positions={"BTCUSDT": Decimal("1.0")},
                venue_positions={"BTCUSDT": Decimal("0.5")},
            ).all_drifts
        )
    )
    report = ReconcileReport(
        venue="binance",
        decisions=(
            PolicyDecision(action=ReconcileAction.ACCEPT, reason="force accept", drift=report_drift),
        ),
    )

    assert report.ok is True
    assert list(report.all_drifts) == []
    assert report.should_halt is False
    assert drift.severity is DriftSeverity.CRITICAL


def test_reconcile_report_should_halt_depends_only_on_decisions() -> None:
    report = ReconcileReport(
        venue="binance",
        decisions=(
            PolicyDecision(
                action=ReconcileAction.ALERT,
                reason="warn",
                drift=ReconcileController().reconcile(
                    venue="binance",
                    local_balances={"USDT": Decimal("1000")},
                    venue_balances={"USDT": Decimal("900")},
                ).all_drifts[0],
            ),
            PolicyDecision(
                action=ReconcileAction.HALT,
                reason="critical",
                drift=ReconcileController().reconcile(
                    venue="binance",
                    local_orders={"ord-1": "new"},
                    venue_orders={"ord-1": "filled"},
                ).all_drifts[0],
            ),
        ),
    )

    assert report.should_halt is True


def test_reconcile_controller_empty_report_is_ok_and_has_no_drifts() -> None:
    report = ReconcileController().reconcile(venue="binance")

    assert report.ok is True
    assert report.should_halt is False
    assert list(report.all_drifts) == []
    assert report.decisions == ()

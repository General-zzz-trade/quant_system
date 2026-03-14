# execution/reconcile/balances.py
"""Balance reconciliation — compare internal balances with venue."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from execution.reconcile.drift import Drift, detect_balance_drift


@dataclass(frozen=True, slots=True)
class BalanceReconcileResult:
    """余额对账结果。"""
    venue: str
    matched: int
    mismatched: int
    drifts: tuple[Drift, ...]

    @property
    def ok(self) -> bool:
        return self.mismatched == 0


def reconcile_balances(
    *,
    venue: str,
    local_balances: Mapping[str, Decimal],    # asset → amount
    venue_balances: Mapping[str, Decimal],     # asset → amount
    tolerance: Decimal = Decimal("0.01"),
) -> BalanceReconcileResult:
    """对账本地余额 vs 交易所余额。"""
    all_assets = set(local_balances.keys()) | set(venue_balances.keys())
    drifts: list[Drift] = []
    matched = 0
    mismatched = 0

    for asset in sorted(all_assets):
        local_val = local_balances.get(asset, Decimal("0"))
        venue_val = venue_balances.get(asset, Decimal("0"))

        drift = detect_balance_drift(
            venue=venue, asset=asset,
            expected=local_val, actual=venue_val,
            tolerance=tolerance,
        )
        if drift:
            mismatched += 1
            drifts.append(drift)
        else:
            matched += 1

    return BalanceReconcileResult(
        venue=venue, matched=matched,
        mismatched=mismatched, drifts=tuple(drifts),
    )

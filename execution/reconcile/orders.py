# execution/reconcile/orders.py
"""Order reconciliation — compare local vs venue order state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from execution.reconcile.drift import Drift, DriftSeverity, DriftType


@dataclass(frozen=True, slots=True)
class OrderReconcileResult:
    """订单对账结果。"""
    venue: str
    matched: int
    status_mismatch: int
    missing_local: int
    missing_venue: int
    drifts: tuple[Drift, ...]

    @property
    def ok(self) -> bool:
        return self.status_mismatch == 0 and self.missing_local == 0


def reconcile_orders(
    *,
    venue: str,
    local_orders: Mapping[str, str],      # order_id → status
    venue_orders: Mapping[str, str],       # order_id → status
) -> OrderReconcileResult:
    """对账本地订单状态 vs 交易所订单状态。"""
    all_ids = set(local_orders.keys()) | set(venue_orders.keys())
    drifts: list[Drift] = []
    matched = 0
    status_mismatch = 0
    missing_local = 0
    missing_venue = 0

    for oid in sorted(all_ids):
        local_status = local_orders.get(oid)
        venue_status = venue_orders.get(oid)

        if local_status is None and venue_status:
            missing_local += 1
            drifts.append(Drift(
                drift_type=DriftType.ORDER_STATUS,
                severity=DriftSeverity.WARNING,
                venue=venue, symbol="",
                expected="not tracked", actual=venue_status,
                detail=f"order {oid} on venue but not local",
            ))
        elif venue_status is None and local_status:
            missing_venue += 1
        elif local_status != venue_status:
            status_mismatch += 1
            terminal = venue_status in ("filled", "canceled", "rejected", "expired")
            severity = DriftSeverity.CRITICAL if terminal else DriftSeverity.WARNING
            drifts.append(Drift(
                drift_type=DriftType.ORDER_STATUS,
                severity=severity,
                venue=venue, symbol="",
                expected=local_status or "", actual=venue_status or "",
                detail=f"order {oid} status mismatch",
            ))
        else:
            matched += 1

    return OrderReconcileResult(
        venue=venue, matched=matched,
        status_mismatch=status_mismatch,
        missing_local=missing_local,
        missing_venue=missing_venue,
        drifts=tuple(drifts),
    )

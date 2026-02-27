# execution/reconcile/fills.py
"""Fill reconciliation — detect missing, duplicate, and mismatched fills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set

from execution.reconcile.drift import Drift, DriftSeverity, DriftType


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Structured fill record for reconciliation."""
    fill_id: str
    symbol: str
    side: str
    qty: float
    price: float


@dataclass(frozen=True, slots=True)
class FillReconcileResult:
    """成交对账结果。"""
    venue: str
    matched: int
    missing_local: int     # 交易所有但本地没有
    missing_venue: int     # 本地有但交易所没有
    price_mismatches: int
    qty_mismatches: int
    drifts: tuple[Drift, ...]

    @property
    def ok(self) -> bool:
        return (
            self.missing_local == 0
            and self.missing_venue == 0
            and self.price_mismatches == 0
            and self.qty_mismatches == 0
        )


def reconcile_fills(
    *,
    venue: str,
    symbol: str,
    local_fill_ids: Set[str],
    venue_fill_ids: Set[str],
    local_fills: Optional[Dict[str, FillRecord]] = None,
    venue_fills: Optional[Dict[str, FillRecord]] = None,
    price_tolerance_pct: float = 0.001,
    qty_tolerance: float = 0.00001,
) -> FillReconcileResult:
    """Reconcile fills between local and venue.

    If structured FillRecords are provided, also validates price and quantity.
    Falls back to ID-only comparison if records are not provided.
    """
    missing_local = venue_fill_ids - local_fill_ids
    missing_venue = local_fill_ids - venue_fill_ids
    matched_ids = local_fill_ids & venue_fill_ids

    drifts: list[Drift] = []
    price_mismatches = 0
    qty_mismatches = 0

    # ID-based drift detection
    for fid in sorted(missing_local):
        drifts.append(Drift(
            drift_type=DriftType.FILL_MISSING,
            severity=DriftSeverity.CRITICAL,
            venue=venue, symbol=symbol,
            expected="present", actual="missing",
            detail=f"fill_id={fid} on venue but not local",
        ))
    for fid in sorted(missing_venue):
        drifts.append(Drift(
            drift_type=DriftType.FILL_EXTRA,
            severity=DriftSeverity.WARNING,
            venue=venue, symbol=symbol,
            expected="missing", actual="present",
            detail=f"fill_id={fid} local but not on venue",
        ))

    # Structured comparison for matched fills
    if local_fills is not None and venue_fills is not None:
        for fid in sorted(matched_ids):
            local = local_fills.get(fid)
            remote = venue_fills.get(fid)
            if local is None or remote is None:
                continue

            # Price comparison
            if remote.price > 0:
                price_diff_pct = abs(local.price - remote.price) / remote.price
                if price_diff_pct > price_tolerance_pct:
                    price_mismatches += 1
                    severity = (
                        DriftSeverity.CRITICAL if price_diff_pct > 0.01
                        else DriftSeverity.WARNING
                    )
                    drifts.append(Drift(
                        drift_type=DriftType.FILL_PRICE_MISMATCH,
                        severity=severity,
                        venue=venue, symbol=symbol,
                        expected=f"{remote.price}",
                        actual=f"{local.price}",
                        detail=f"fill_id={fid} price diff={price_diff_pct:.6f}",
                    ))

            # Quantity comparison
            qty_diff = abs(local.qty - remote.qty)
            if qty_diff > qty_tolerance:
                qty_mismatches += 1
                severity = (
                    DriftSeverity.CRITICAL if remote.qty > 0 and qty_diff / remote.qty > 0.1
                    else DriftSeverity.WARNING
                )
                drifts.append(Drift(
                    drift_type=DriftType.FILL_QTY_MISMATCH,
                    severity=severity,
                    venue=venue, symbol=symbol,
                    expected=f"{remote.qty}",
                    actual=f"{local.qty}",
                    detail=f"fill_id={fid} qty diff={qty_diff}",
                ))

    return FillReconcileResult(
        venue=venue,
        matched=len(matched_ids),
        missing_local=len(missing_local),
        missing_venue=len(missing_venue),
        price_mismatches=price_mismatches,
        qty_mismatches=qty_mismatches,
        drifts=tuple(drifts),
    )

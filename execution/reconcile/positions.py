# execution/reconcile/positions.py
"""Position reconciliation — compare internal positions with venue."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Mapping, Optional, Sequence

from execution.reconcile.drift import Drift, DriftSeverity, DriftType, detect_qty_drift


@dataclass(frozen=True, slots=True)
class PositionReconcileResult:
    """持仓对账结果。"""
    venue: str
    matched: int
    mismatched: int
    missing_local: int      # 本地没有但交易所有
    missing_venue: int      # 本地有但交易所没有
    drifts: tuple[Drift, ...]

    @property
    def ok(self) -> bool:
        return self.mismatched == 0 and self.missing_local == 0 and self.missing_venue == 0


def reconcile_positions(
    *,
    venue: str,
    local_positions: Mapping[str, Decimal],     # symbol → signed qty
    venue_positions: Mapping[str, Decimal],      # symbol → signed qty
    tolerance: Decimal = Decimal("0.00001"),
) -> PositionReconcileResult:
    """
    对账本地持仓 vs 交易所持仓。
    """
    all_symbols = set(local_positions.keys()) | set(venue_positions.keys())
    drifts: list[Drift] = []
    matched = 0
    mismatched = 0
    missing_local = 0
    missing_venue = 0

    for symbol in sorted(all_symbols):
        local_qty = local_positions.get(symbol, Decimal("0"))
        venue_qty = venue_positions.get(symbol, Decimal("0"))

        if symbol not in local_positions and venue_qty != 0:
            missing_local += 1
            drifts.append(Drift(
                drift_type=DriftType.POSITION_QTY,
                severity=DriftSeverity.CRITICAL,
                venue=venue, symbol=symbol,
                expected="0 (not tracked)", actual=str(venue_qty),
                detail="position exists on venue but not locally",
            ))
            continue

        if symbol not in venue_positions and local_qty != 0:
            missing_venue += 1
            drifts.append(Drift(
                drift_type=DriftType.POSITION_QTY,
                severity=DriftSeverity.WARNING,
                venue=venue, symbol=symbol,
                expected=str(local_qty), actual="0 (not on venue)",
                detail="position exists locally but not on venue",
            ))
            continue

        drift = detect_qty_drift(
            symbol=symbol, venue=venue,
            expected_qty=local_qty, actual_qty=venue_qty,
            tolerance=tolerance,
        )
        if drift:
            mismatched += 1
            drifts.append(drift)
        else:
            matched += 1

    return PositionReconcileResult(
        venue=venue,
        matched=matched,
        mismatched=mismatched,
        missing_local=missing_local,
        missing_venue=missing_venue,
        drifts=tuple(drifts),
    )

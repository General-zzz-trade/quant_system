# execution/reconcile/fills.py
"""Fill reconciliation — detect missing or duplicate fills."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set

from execution.reconcile.drift import Drift, DriftSeverity, DriftType


@dataclass(frozen=True, slots=True)
class FillReconcileResult:
    """成交对账结果。"""
    venue: str
    matched: int
    missing_local: int     # 交易所有但本地没有
    missing_venue: int     # 本地有但交易所没有
    drifts: tuple[Drift, ...]

    @property
    def ok(self) -> bool:
        return self.missing_local == 0 and self.missing_venue == 0


def reconcile_fills(
    *,
    venue: str,
    symbol: str,
    local_fill_ids: Set[str],
    venue_fill_ids: Set[str],
) -> FillReconcileResult:
    """
    对账成交记录 — 检查两端 fill_id 集合差异。
    """
    missing_local = venue_fill_ids - local_fill_ids
    missing_venue = local_fill_ids - venue_fill_ids
    matched = len(local_fill_ids & venue_fill_ids)

    drifts: list[Drift] = []
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

    return FillReconcileResult(
        venue=venue, matched=matched,
        missing_local=len(missing_local),
        missing_venue=len(missing_venue),
        drifts=tuple(drifts),
    )

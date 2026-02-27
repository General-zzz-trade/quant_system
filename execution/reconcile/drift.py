# execution/reconcile/drift.py
"""Drift detection — compare expected vs actual state."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence


class DriftSeverity(str, Enum):
    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(str, Enum):
    POSITION_QTY = "position_qty"
    BALANCE = "balance"
    ORDER_STATUS = "order_status"
    FILL_MISSING = "fill_missing"
    FILL_EXTRA = "fill_extra"
    FILL_PRICE_MISMATCH = "fill_price_mismatch"
    FILL_QTY_MISMATCH = "fill_qty_mismatch"


@dataclass(frozen=True, slots=True)
class Drift:
    """单项漂移记录。"""
    drift_type: DriftType
    severity: DriftSeverity
    venue: str
    symbol: str
    expected: str
    actual: str
    detail: str = ""


def detect_qty_drift(
    *,
    symbol: str,
    venue: str,
    expected_qty: Decimal,
    actual_qty: Decimal,
    tolerance: Decimal = Decimal("0.00001"),
) -> Optional[Drift]:
    """检测仓位数量漂移。"""
    diff = abs(expected_qty - actual_qty)
    if diff <= tolerance:
        return None

    if diff > abs(expected_qty) * Decimal("0.1"):
        severity = DriftSeverity.CRITICAL
    elif diff > tolerance * 100:
        severity = DriftSeverity.WARNING
    else:
        severity = DriftSeverity.INFO

    return Drift(
        drift_type=DriftType.POSITION_QTY,
        severity=severity,
        venue=venue,
        symbol=symbol,
        expected=str(expected_qty),
        actual=str(actual_qty),
        detail=f"diff={diff}",
    )


def detect_balance_drift(
    *,
    venue: str,
    asset: str,
    expected: Decimal,
    actual: Decimal,
    tolerance: Decimal = Decimal("0.01"),
) -> Optional[Drift]:
    """检测余额漂移。"""
    diff = abs(expected - actual)
    if diff <= tolerance:
        return None

    severity = DriftSeverity.CRITICAL if diff > expected * Decimal("0.05") else DriftSeverity.WARNING
    return Drift(
        drift_type=DriftType.BALANCE,
        severity=severity,
        venue=venue,
        symbol=asset,
        expected=str(expected),
        actual=str(actual),
        detail=f"diff={diff}",
    )

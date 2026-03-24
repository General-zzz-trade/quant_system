"""Helper classes for BacktestExecutionAdapter.

Extracted from adapter.py: TradingRules, ExecutionSummary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TradingRules:
    """Exchange trading rules for order validation."""
    min_qty: Decimal = Decimal("0.001")
    step_size: Decimal = Decimal("0.001")
    tick_size: Decimal = Decimal("0.01")
    min_notional: Decimal = Decimal("5")     # minimum order value in quote currency
    max_qty: Optional[Decimal] = None

    def round_qty(self, qty: Decimal) -> Decimal:
        """Round qty down to nearest step_size."""
        if self.step_size <= 0:
            return qty
        return (qty / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size

    def round_price(self, price: Decimal) -> Decimal:
        """Round price to nearest tick_size."""
        if self.tick_size <= 0:
            return price
        return (price / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size

    def validate(self, qty: Decimal, price: Decimal) -> Optional[str]:
        """Return rejection reason string, or None if valid."""
        if qty < self.min_qty:
            return f"qty {qty} < min_qty {self.min_qty}"
        if self.max_qty is not None and qty > self.max_qty:
            return f"qty {qty} > max_qty {self.max_qty}"
        notional = qty * price
        if notional < self.min_notional:
            return f"notional {notional} < min_notional {self.min_notional}"
        return None


@dataclass
class ExecutionSummary:
    """Accumulated execution statistics for backtest reporting."""
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    expired_orders: int = 0
    partial_fill_count: int = 0
    total_fills: int = 0
    gross_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")
    total_funding: Decimal = Decimal("0")
    liquidation_count: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "rejected_orders": self.rejected_orders,
            "expired_orders": self.expired_orders,
            "partial_fill_count": self.partial_fill_count,
            "total_fills": self.total_fills,
            "gross_pnl": float(self.gross_pnl),
            "net_pnl": float(self.net_pnl),
            "total_fees": float(self.total_fees),
            "total_slippage": float(self.total_slippage),
            "total_funding": float(self.total_funding),
            "liquidation_count": self.liquidation_count,
            "rejection_reasons": dict(self.rejection_reasons),
        }

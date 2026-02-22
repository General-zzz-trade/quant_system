# context/constraints/strategy_constraints.py
"""Strategy-level constraints — per-strategy limits."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True, slots=True)
class StrategyConstraints:
    """策略级约束。"""
    strategy_id: str
    max_order_qty: Optional[Decimal] = None
    max_order_notional: Optional[Decimal] = None
    max_daily_orders: Optional[int] = None
    allowed_symbols: tuple[str, ...] = ()
    allowed_sides: tuple[str, ...] = ("buy", "sell")
    enabled: bool = True

    def check_symbol(self, symbol: str) -> Optional[str]:
        if self.allowed_symbols and symbol.upper() not in self.allowed_symbols:
            return f"symbol {symbol} not in allowed list"
        return None

    def check_side(self, side: str) -> Optional[str]:
        if side not in self.allowed_sides:
            return f"side {side} not allowed for strategy {self.strategy_id}"
        return None

    def check_enabled(self) -> Optional[str]:
        if not self.enabled:
            return f"strategy {self.strategy_id} is disabled"
        return None

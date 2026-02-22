# context/constraints/portfolio_constraints.py
"""Portfolio-level constraints — max positions, exposure limits."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """组合级约束。"""
    max_positions: int = 10
    max_gross_exposure: Optional[Decimal] = None
    max_single_position_pct: Decimal = Decimal("0.2")  # 单品种最大占比 20%
    max_leverage: Decimal = Decimal("3.0")

    def check_position_count(self, current_count: int) -> Optional[str]:
        if current_count >= self.max_positions:
            return f"max positions reached: {current_count}/{self.max_positions}"
        return None

    def check_single_position_pct(
        self, position_value: Decimal, total_equity: Decimal,
    ) -> Optional[str]:
        if total_equity <= 0:
            return None
        pct = position_value / total_equity
        if pct > self.max_single_position_pct:
            return f"single position {pct:.2%} > max {self.max_single_position_pct:.2%}"
        return None

    def check_leverage(
        self, gross_exposure: Decimal, equity: Decimal,
    ) -> Optional[str]:
        if equity <= 0:
            return "equity <= 0"
        lev = gross_exposure / equity
        if lev > self.max_leverage:
            return f"leverage {lev:.2f} > max {self.max_leverage}"
        return None

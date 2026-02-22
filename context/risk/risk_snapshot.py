# context/risk/risk_snapshot.py
"""Risk snapshot — read-only view of risk metrics."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True, slots=True)
class ContextRiskSnapshot:
    """风险状态快照。"""
    current_drawdown: Decimal
    peak_equity: Decimal
    var_95: Optional[Decimal] = None
    daily_loss: Decimal = Decimal("0")
    kill_switch_active: bool = False

    @property
    def is_safe(self) -> bool:
        return not self.kill_switch_active and self.current_drawdown < Decimal("0.1")

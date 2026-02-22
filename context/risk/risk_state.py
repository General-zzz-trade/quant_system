# context/risk/risk_state.py
"""Risk state within context — aggregate risk metrics."""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, Optional

from context.risk.risk_snapshot import ContextRiskSnapshot


class ContextRiskState:
    """Context 内的风险状态追踪。"""

    def __init__(self) -> None:
        self._drawdown: Decimal = Decimal("0")
        self._peak_equity: Decimal = Decimal("0")
        self._var_95: Optional[Decimal] = None
        self._daily_loss: Decimal = Decimal("0")
        self._kill_switch_active: bool = False

    def update_equity(self, equity: Decimal) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            self._drawdown = (self._peak_equity - equity) / self._peak_equity

    def update_daily_loss(self, loss: Decimal) -> None:
        self._daily_loss = loss

    def set_kill_switch(self, active: bool) -> None:
        self._kill_switch_active = active

    def snapshot(self) -> ContextRiskSnapshot:
        return ContextRiskSnapshot(
            current_drawdown=self._drawdown,
            peak_equity=self._peak_equity,
            var_95=self._var_95,
            daily_loss=self._daily_loss,
            kill_switch_active=self._kill_switch_active,
        )

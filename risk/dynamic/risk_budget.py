"""Dynamic risk budget — adapts risk limits based on regime and volatility.

Adjusts position sizing and risk limits based on:
- Market regime (from regime detectors)
- Recent realized volatility
- Drawdown state
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskBudget:
    """Current risk budget allocation."""
    max_position_pct: float      # Max position size as % of equity
    max_portfolio_var: float     # Max portfolio VaR %
    max_leverage: float          # Max leverage
    target_volatility: float     # Target portfolio volatility
    scale_factor: float          # Overall risk scaling (0-1)
    regime: str                  # Current regime label
    reason: str                  # Why this budget was set


@dataclass
class DynamicRiskBudgetManager:
    """Manages dynamic risk budget based on market conditions."""

    base_position_pct: float = 10.0
    base_var_limit: float = 5.0
    base_leverage: float = 3.0
    target_vol: float = 0.15        # 15% annualized target vol
    vol_lookback: int = 20
    drawdown_reduce_threshold: float = 5.0   # Start reducing at 5% drawdown
    drawdown_halt_threshold: float = 15.0    # Halt trading at 15% drawdown

    _current_budget: Optional[RiskBudget] = field(default=None, init=False)

    def update(
        self,
        *,
        regime: str = "normal",
        recent_returns: Optional[Sequence[float]] = None,
        current_drawdown_pct: float = 0.0,
        equity: Optional[Decimal] = None,
    ) -> RiskBudget:
        """Recompute risk budget based on current conditions.

        Args:
            regime: Current market regime ("trending", "mean_reverting", "volatile", "normal").
            recent_returns: Recent return series for vol estimation.
            current_drawdown_pct: Current drawdown from peak (positive number).
            equity: Current portfolio equity.
        """
        scale = 1.0
        reason_parts = []

        # Regime adjustment
        regime_scales = {
            "trending": 1.0,
            "mean_reverting": 0.8,
            "volatile": 0.5,
            "crisis": 0.2,
            "normal": 0.8,
        }
        regime_scale = regime_scales.get(regime, 0.8)
        scale *= regime_scale
        reason_parts.append(f"regime={regime}({regime_scale:.1f})")

        # Volatility targeting
        if recent_returns and len(recent_returns) >= self.vol_lookback:
            window = recent_returns[-self.vol_lookback:]
            mean_r = sum(window) / len(window)
            var = sum((r - mean_r) ** 2 for r in window) / len(window)
            realized_vol = math.sqrt(var * 252)

            if realized_vol > 0:
                vol_scale = min(self.target_vol / realized_vol, 1.5)
                vol_scale = max(vol_scale, 0.2)  # Floor
                scale *= vol_scale
                reason_parts.append(f"vol={realized_vol:.2f}→scale={vol_scale:.2f}")

        # Drawdown reduction
        if current_drawdown_pct >= self.drawdown_halt_threshold:
            scale = 0.0
            reason_parts.append(f"dd={current_drawdown_pct:.1f}%→HALT")
        elif current_drawdown_pct >= self.drawdown_reduce_threshold:
            dd_range = self.drawdown_halt_threshold - self.drawdown_reduce_threshold
            dd_excess = current_drawdown_pct - self.drawdown_reduce_threshold
            dd_scale = 1.0 - (dd_excess / dd_range)
            dd_scale = max(dd_scale, 0.1)
            scale *= dd_scale
            reason_parts.append(f"dd={current_drawdown_pct:.1f}%→scale={dd_scale:.2f}")

        budget = RiskBudget(
            max_position_pct=self.base_position_pct * scale,
            max_portfolio_var=self.base_var_limit * scale,
            max_leverage=self.base_leverage * min(scale, 1.0),
            target_volatility=self.target_vol * scale,
            scale_factor=scale,
            regime=regime,
            reason=", ".join(reason_parts),
        )

        self._current_budget = budget
        logger.info("Risk budget updated: scale=%.2f reason=%s", scale, budget.reason)
        return budget

    @property
    def current(self) -> Optional[RiskBudget]:
        return self._current_budget

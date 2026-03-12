"""Staged Risk Manager — adapts risk parameters to current equity level.

Small accounts need aggressive sizing to clear minimum notional ($100),
but must have strict drawdown protection to survive. As equity grows,
risk automatically decreases for stable compounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RiskStage:
    """One stage of the risk ladder."""
    min_equity: float
    max_equity: float
    risk_fraction: float
    leverage: float
    max_drawdown_pct: float   # stop trading if DD exceeds this
    label: str

    def notional(self, equity: float) -> float:
        return equity * self.risk_fraction * self.leverage


# Default stages designed for crypto futures ($100 min notional)
DEFAULT_STAGES: List[RiskStage] = [
    RiskStage(0,     300,   0.50, 3.0, 0.25, "survival"),
    RiskStage(300,   800,   0.25, 3.0, 0.20, "growth"),
    RiskStage(800,  2000,   0.12, 3.0, 0.15, "stable"),
    RiskStage(2000, 5000,   0.05, 2.0, 0.10, "safe"),
    RiskStage(5000, float('inf'), 0.03, 2.0, 0.08, "institutional"),
]


class StagedRiskManager:
    """Automatically adjusts risk parameters based on current equity.

    Features:
    - Hysteresis: downgrade requires equity to drop 10% below stage boundary
      (prevents rapid stage oscillation at boundaries)
    - DrawdownController: scales position size inversely with drawdown depth
    - Minimum notional enforcement: auto-scales up if position too small

    Parameters
    ----------
    initial_equity : float
        Starting equity for stage determination.
    stages : list[RiskStage], optional
        Custom stage ladder. Defaults to DEFAULT_STAGES.
    min_notional : float
        Minimum order notional (exchange constraint).
    hysteresis_pct : float
        Downgrade buffer as fraction of stage boundary.
    """

    def __init__(
        self,
        initial_equity: float,
        stages: Optional[List[RiskStage]] = None,
        min_notional: float = 100.0,
        hysteresis_pct: float = 0.10,
    ):
        self._stages = stages or DEFAULT_STAGES
        self._min_notional = min_notional
        self._hysteresis_pct = hysteresis_pct
        self._peak_equity = initial_equity
        self._current_equity = initial_equity
        self._current_stage = self._find_stage(initial_equity)
        self._trading_halted = False
        self._halt_equity = 0.0

    def update_equity(self, equity: float) -> None:
        """Update equity and recalculate stage + drawdown state."""
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

        # Check if we should resume trading after halt
        if self._trading_halted:
            # Resume when equity recovers above halt level
            if equity > self._halt_equity * 1.05:
                self._trading_halted = False
                self._peak_equity = equity  # Reset peak

        # Check drawdown
        dd = self.current_drawdown
        if dd >= self._current_stage.max_drawdown_pct:
            self._trading_halted = True
            self._halt_equity = equity

        # Stage transition with hysteresis
        new_stage = self._find_stage(equity)
        if new_stage != self._current_stage:
            # Upgrading (more equity) — immediate
            if new_stage.min_equity > self._current_stage.min_equity:
                self._current_stage = new_stage
            else:
                # Downgrading — require hysteresis buffer
                boundary = self._current_stage.min_equity
                if equity < boundary * (1 - self._hysteresis_pct):
                    self._current_stage = new_stage

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak (0.0 to 1.0)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity)

    @property
    def can_trade(self) -> bool:
        """Whether trading is allowed (not halted by drawdown)."""
        return not self._trading_halted

    @property
    def stage(self) -> RiskStage:
        """Current risk stage."""
        return self._current_stage

    @property
    def risk_fraction(self) -> float:
        return self._current_stage.risk_fraction

    @property
    def leverage(self) -> float:
        return self._current_stage.leverage

    def position_scale(self) -> float:
        """Position scale factor based on drawdown depth (0.0 to 1.0).

        Gradually reduces position size as drawdown deepens,
        even before the hard halt threshold.
        """
        if self._trading_halted:
            return 0.0

        dd = self.current_drawdown
        max_dd = self._current_stage.max_drawdown_pct

        if dd < max_dd * 0.3:
            return 1.0
        elif dd < max_dd * 0.6:
            return 0.7
        elif dd < max_dd * 0.85:
            return 0.4
        elif dd < max_dd:
            return 0.2
        else:
            return 0.0

    def compute_notional(self, price: float) -> float:
        """Compute order notional, enforcing minimum.

        Returns 0.0 if position would exceed safe limits.
        """
        if not self.can_trade:
            return 0.0

        equity = self._current_equity
        scale = self.position_scale()
        notional = equity * self.risk_fraction * self.leverage * scale

        # Enforce minimum notional
        if notional < self._min_notional:
            # Can we safely scale up to minimum?
            max_safe_notional = equity * self.leverage * 0.8  # never use >80% of equity
            if self._min_notional <= max_safe_notional:
                notional = self._min_notional
            else:
                return 0.0  # Can't even open minimum position safely

        return notional

    def _find_stage(self, equity: float) -> RiskStage:
        for stage in self._stages:
            if stage.min_equity <= equity < stage.max_equity:
                return stage
        return self._stages[-1]

    def __repr__(self) -> str:
        dd = self.current_drawdown
        return (
            f"StagedRisk(eq=${self._current_equity:.0f}, "
            f"stage={self._current_stage.label}, "
            f"risk={self.risk_fraction:.0%}×{self.leverage:.0f}x, "
            f"dd={dd:.1%}, scale={self.position_scale():.0%}, "
            f"halted={self._trading_halted})"
        )

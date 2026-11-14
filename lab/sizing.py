"""Exposure for one position under 10x-isolated / 1-3x-actual sizing (user's decision, 2026-11-14).

    exposure = 0                                          when there is no signal
             = clamp(target_vol / realized_vol, min_open, max_exposure)
               then capped by max_trade_loss / stop_distance (this cap may go below min_open)
             = 0 (refused)                                when the stop is not well inside the
                                                          isolated liquidation distance

``exposure`` is gross notional / equity. ``max_exposure`` can never exceed the
account hard cap ``MAX_TOTAL_LEVERAGE`` (3x) nor the isolated leverage. The
liquidation distance of an isolated position is ``1/leverage - mmr`` whatever
its size, so the stop must sit within ``max_stop_to_liquidation`` of it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from shared.exposure_gate import MAX_TOTAL_LEVERAGE
from trading.domain.risk.liquidation import liquidation_distance  # moved in migration step 10


@dataclass(frozen=True)
class SizingRule:
    target_vol: float                       # annualized volatility the position aims for
    max_trade_loss: float                   # fraction of equity lost if the stop is hit
    min_open: float = 1.0                   # exposure floor when a position is opened (before the loss cap)
    max_exposure: float = 3.0
    isolated_leverage: float = 10.0
    maintenance_margin_rate: float = 0.005
    max_stop_to_liquidation: float = 0.5    # stop distance <= this x liquidation distance

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("sizing parameters must be finite and positive")
        if self.max_exposure > float(MAX_TOTAL_LEVERAGE) or self.max_exposure > self.isolated_leverage:
            raise ValueError(f"max_exposure must not exceed {MAX_TOTAL_LEVERAGE} or the isolated leverage")
        if self.min_open > self.max_exposure:
            raise ValueError("min_open must not exceed max_exposure")
        if self.max_trade_loss >= 1 or self.max_stop_to_liquidation >= 1:
            raise ValueError("max_trade_loss and max_stop_to_liquidation must be below 1")
        if liquidation_distance(self.isolated_leverage, self.maintenance_margin_rate) <= 0:
            raise ValueError("maintenance margin rate leaves no liquidation distance")

    def record(self) -> dict:
        return asdict(self)

    @property
    def max_stop_distance(self) -> float:
        return self.max_stop_to_liquidation * liquidation_distance(self.isolated_leverage,
                                                                   self.maintenance_margin_rate)

    def exposure(self, signal: int, realized_vol: float, stop_distance: float) -> tuple[float, str]:
        """(gross exposure in x equity, reason). Unknown or invalid inputs size to zero."""
        if signal == 0:
            return 0.0, "no signal"
        if not (math.isfinite(realized_vol) and realized_vol > 0 and math.isfinite(stop_distance)
                and stop_distance > 0):
            return 0.0, "volatility or stop distance unavailable"
        if stop_distance > self.max_stop_distance:
            return 0.0, (f"stop {stop_distance:.2%} beyond {self.max_stop_to_liquidation:.0%} of the "
                         f"liquidation distance ({self.max_stop_distance:.2%})")
        base = min(max(self.target_vol / realized_vol, self.min_open), self.max_exposure)
        cap = self.max_trade_loss / stop_distance
        if cap < base:
            return cap, f"capped by max trade loss {self.max_trade_loss:.1%} at stop {stop_distance:.2%}"
        return base, "volatility target within [min_open, max_exposure]"

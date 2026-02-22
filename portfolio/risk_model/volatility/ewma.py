# portfolio/risk_model/volatility/ewma.py
"""EWMA (Exponentially Weighted Moving Average) volatility."""
from __future__ import annotations

import math
from typing import Sequence


class EWMAVolatility:
    """EWMA 指数加权移动平均波动率。"""
    name: str = "ewma"

    def __init__(
        self, span: int = 30, annualization: float = 365.0
    ) -> None:
        self.alpha = 2.0 / (span + 1)
        self.annualization = annualization

    def estimate(self, returns: Sequence[float]) -> float:
        if len(returns) < 2:
            return 0.0
        var = returns[0] ** 2
        for r in returns[1:]:
            var = self.alpha * r ** 2 + (1 - self.alpha) * var
        return math.sqrt(var * self.annualization)

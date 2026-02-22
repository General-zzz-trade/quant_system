# portfolio/risk_model/volatility/historical.py
"""Historical (sample) volatility estimator."""
from __future__ import annotations

import math
from typing import Sequence


class HistoricalVolatility:
    """历史样本波动率。"""
    name: str = "historical"

    def __init__(self, annualization: float = 365.0) -> None:
        self.annualization = annualization

    def estimate(self, returns: Sequence[float]) -> float:
        n = len(returns)
        if n < 2:
            return 0.0
        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / (n - 1)
        return math.sqrt(var * self.annualization)

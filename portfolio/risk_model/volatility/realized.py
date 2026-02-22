# portfolio/risk_model/volatility/realized.py
"""Realized volatility from intraday returns."""
from __future__ import annotations

import math
from typing import Sequence


class RealizedVolatility:
    """已实现波动率（基于高频收益率）。"""
    name: str = "realized"

    def __init__(self, annualization: float = 365.0) -> None:
        self.annualization = annualization

    def estimate(self, returns: Sequence[float]) -> float:
        if len(returns) < 1:
            return 0.0
        rv = sum(r ** 2 for r in returns)
        return math.sqrt(rv * self.annualization)

    def estimate_from_prices(
        self, prices: Sequence[float], sampling_freq: int = 1
    ) -> float:
        """从价格序列计算已实现波动率。"""
        if len(prices) < 2:
            return 0.0
        log_returns = []
        for i in range(sampling_freq, len(prices), sampling_freq):
            if prices[i - sampling_freq] > 0 and prices[i] > 0:
                log_returns.append(math.log(prices[i] / prices[i - sampling_freq]))
        return self.estimate(log_returns)

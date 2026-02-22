# portfolio/risk_model/volatility/garch.py
"""Simplified GARCH(1,1) volatility estimator."""
from __future__ import annotations

import math
from typing import Sequence


class GARCHVolatility:
    """简化 GARCH(1,1) 波动率估计。"""
    name: str = "garch"

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.1,
        beta: float = 0.85,
        annualization: float = 365.0,
    ) -> None:
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.annualization = annualization

    def estimate(self, returns: Sequence[float]) -> float:
        if len(returns) < 2:
            return 0.0
        var = sum(r ** 2 for r in returns) / len(returns)
        for r in returns:
            var = self.omega + self.alpha * r ** 2 + self.beta * var
        return math.sqrt(var * self.annualization)

    def forecast(self, returns: Sequence[float], horizon: int = 1) -> float:
        """多步预测。"""
        if len(returns) < 2:
            return 0.0
        var = sum(r ** 2 for r in returns) / len(returns)
        for r in returns:
            var = self.omega + self.alpha * r ** 2 + self.beta * var
        # 长期均值
        long_run = self.omega / max(1e-12, 1.0 - self.alpha - self.beta)
        persistence = self.alpha + self.beta
        forecast_var = long_run + (var - long_run) * persistence ** horizon
        return math.sqrt(max(forecast_var, 0.0) * self.annualization)

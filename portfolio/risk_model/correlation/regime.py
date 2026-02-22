# portfolio/risk_model/correlation/regime.py
"""Regime-aware correlation: uses different correlation in different regimes."""
from __future__ import annotations

from typing import Mapping, Sequence

from portfolio.risk_model.correlation.rolling import RollingCorrelation


class RegimeCorrelation:
    """根据市场状态切换相关矩阵参数。"""
    name: str = "regime_correlation"

    def __init__(
        self,
        normal_window: int = 60,
        stress_window: int = 20,
        stress_multiplier: float = 1.3,
        vol_threshold: float = 0.03,
    ) -> None:
        self._normal = RollingCorrelation(window=normal_window)
        self._stress = RollingCorrelation(window=stress_window)
        self.stress_multiplier = stress_multiplier
        self.vol_threshold = vol_threshold

    def _detect_stress(self, returns: Mapping[str, Sequence[float]]) -> bool:
        """简单波动率阈值检测压力状态。"""
        import math
        for rets in returns.values():
            recent = list(rets)[-20:]
            if len(recent) < 5:
                continue
            vol = math.sqrt(sum(r ** 2 for r in recent) / len(recent))
            if vol > self.vol_threshold:
                return True
        return False

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        if self._detect_stress(returns):
            corr = self._stress.estimate(symbols, returns)
            # 压力期相关性放大
            for s1 in symbols:
                for s2 in symbols:
                    if s1 != s2:
                        v = corr[s1][s2] * self.stress_multiplier
                        corr[s1][s2] = max(-1.0, min(1.0, v))
            return corr
        return self._normal.estimate(symbols, returns)

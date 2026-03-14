# portfolio/risk_model/covariance/regime_aware.py
"""Regime-aware covariance blending."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from portfolio.risk_model.covariance.ewma import EWMACovariance


class RegimeAwareCovariance:
    """根据市场状态混合长/短窗口协方差。"""
    name: str = "regime_cov"

    def __init__(
        self,
        short_span: int = 15,
        long_span: int = 60,
        vol_threshold: float = 0.03,
        stress_weight: float = 0.7,
    ) -> None:
        self._short = EWMACovariance(span=short_span)
        self._long = EWMACovariance(span=long_span)
        self.vol_threshold = vol_threshold
        self.stress_weight = stress_weight

    def _detect_stress(self, returns: Mapping[str, Sequence[float]]) -> bool:
        for rets in returns.values():
            recent = list(rets)[-10:]
            if len(recent) < 3:
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
        short_cov = self._short.estimate(symbols, returns)
        long_cov = self._long.estimate(symbols, returns)

        w = self.stress_weight if self._detect_stress(returns) else 1 - self.stress_weight
        result: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            row: dict[str, float] = {}
            for s2 in symbols:
                row[s2] = w * short_cov[s1][s2] + (1 - w) * long_cov[s1][s2]
            result[s1] = row
        return result

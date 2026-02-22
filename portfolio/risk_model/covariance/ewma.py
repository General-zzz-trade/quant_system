# portfolio/risk_model/covariance/ewma.py
"""EWMA covariance estimator."""
from __future__ import annotations

from typing import Mapping, Sequence


class EWMACovariance:
    """EWMA 指数加权协方差矩阵。"""
    name: str = "ewma_cov"

    def __init__(self, span: int = 30) -> None:
        self.alpha = 2.0 / (span + 1)

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
        if n_obs < 2:
            return {s1: {s2: 0.0 for s2 in symbols} for s1 in symbols}

        # 初始化为第一期外积
        cov: dict[tuple[str, str], float] = {}
        for s1 in symbols:
            for s2 in symbols:
                cov[(s1, s2)] = returns[s1][0] * returns[s2][0]

        # EWMA 递推
        for t in range(1, n_obs):
            for s1 in symbols:
                for s2 in symbols:
                    key = (s1, s2)
                    cov[key] = (
                        self.alpha * returns[s1][t] * returns[s2][t]
                        + (1 - self.alpha) * cov[key]
                    )

        result: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            row: dict[str, float] = {}
            for s2 in symbols:
                row[s2] = cov[(s1, s2)]
            result[s1] = row
        return result

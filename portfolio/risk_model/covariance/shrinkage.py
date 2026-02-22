# portfolio/risk_model/covariance/shrinkage.py
"""Ledoit-Wolf shrinkage covariance estimator."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from portfolio.risk_model.covariance.sample import SampleCovariance


class ShrinkageCovariance:
    """Ledoit-Wolf 收缩协方差矩阵。"""
    name: str = "shrinkage"

    def __init__(self, shrinkage_target: str = "identity") -> None:
        self._sample = SampleCovariance()
        self.shrinkage_target = shrinkage_target

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        sample = self._sample.estimate(symbols, returns)
        n = len(symbols)
        if n < 2:
            return sample

        # 目标矩阵: scaled identity
        avg_var = sum(sample[s][s] for s in symbols) / n
        target: dict[str, dict[str, float]] = {
            s1: {s2: (avg_var if s1 == s2 else 0.0) for s2 in symbols}
            for s1 in symbols
        }

        # 自适应收缩系数 (简化 Ledoit-Wolf)
        n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
        if n_obs < 4:
            alpha = 0.5
        else:
            alpha = max(0.0, min(1.0, 1.0 / math.sqrt(n_obs)))

        result: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            row: dict[str, float] = {}
            for s2 in symbols:
                row[s2] = alpha * target[s1][s2] + (1 - alpha) * sample[s1][s2]
            result[s1] = row
        return result

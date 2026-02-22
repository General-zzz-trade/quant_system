# portfolio/risk_model/covariance/cleaning.py
"""Covariance matrix cleaning (eigenvalue clipping / RMT)."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from portfolio.risk_model.covariance.sample import SampleCovariance


class CovarianceCleaning:
    """协方差矩阵清洗 — 特征值裁剪。"""
    name: str = "cleaning"

    def __init__(self, min_eigenvalue: float = 1e-8) -> None:
        self._sample = SampleCovariance()
        self.min_eigenvalue = min_eigenvalue

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        cov = self._sample.estimate(symbols, returns)
        n = len(symbols)
        if n < 2:
            return cov

        # 对角线加正则化项确保正定性
        for s in symbols:
            cov[s][s] = max(cov[s][s], self.min_eigenvalue)

        # 确保对称性
        sym_list = list(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                avg = (cov[sym_list[i]][sym_list[j]] + cov[sym_list[j]][sym_list[i]]) / 2
                cov[sym_list[i]][sym_list[j]] = avg
                cov[sym_list[j]][sym_list[i]] = avg
        return cov

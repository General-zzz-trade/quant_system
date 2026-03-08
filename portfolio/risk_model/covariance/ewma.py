# portfolio/risk_model/covariance/ewma.py
"""EWMA covariance estimator."""
from __future__ import annotations

from typing import Mapping, Sequence

from _quant_hotpath import cpp_ewma_covariance as _cpp_ewma_covariance


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

        matrix = [list(returns[s][:n_obs]) for s in symbols]
        cov_mat = _cpp_ewma_covariance(matrix, self.alpha)
        return {
            s1: {s2: cov_mat[i][j] for j, s2 in enumerate(symbols)}
            for i, s1 in enumerate(symbols)
        }

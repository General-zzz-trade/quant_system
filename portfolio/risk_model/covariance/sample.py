# portfolio/risk_model/covariance/sample.py
"""Sample covariance matrix estimator."""
from __future__ import annotations

from typing import Mapping, Sequence

from _quant_hotpath import cpp_sample_covariance as _cpp_sample_covariance


class SampleCovariance:
    """样本协方差矩阵。"""
    name: str = "sample"

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        n_obs = min(len(returns.get(s, ())) for s in symbols) if symbols else 0
        if n_obs < 2:
            return {s1: {s2: 0.0 for s2 in symbols} for s1 in symbols}

        matrix = [list(returns[s][:n_obs]) for s in symbols]
        cov_mat = _cpp_sample_covariance(matrix)
        return {
            s1: {s2: cov_mat[i][j] for j, s2 in enumerate(symbols)}
            for i, s1 in enumerate(symbols)
        }

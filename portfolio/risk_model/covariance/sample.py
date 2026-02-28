# portfolio/risk_model/covariance/sample.py
"""Sample covariance matrix estimator."""
from __future__ import annotations

from typing import Mapping, Sequence

try:
    from features._quant_rolling import cpp_sample_covariance as _cpp_sample_covariance
    _USING_CPP = True
except ImportError:
    _USING_CPP = False


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

        if _USING_CPP:
            matrix = [list(returns[s][:n_obs]) for s in symbols]
            cov_mat = _cpp_sample_covariance(matrix)
            return {
                s1: {s2: cov_mat[i][j] for j, s2 in enumerate(symbols)}
                for i, s1 in enumerate(symbols)
            }

        means = {s: sum(returns[s][:n_obs]) / n_obs for s in symbols}
        result: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            row: dict[str, float] = {}
            r1 = returns[s1][:n_obs]
            m1 = means[s1]
            for s2 in symbols:
                r2 = returns[s2][:n_obs]
                m2 = means[s2]
                cov = sum((r1[i] - m1) * (r2[i] - m2) for i in range(n_obs)) / (n_obs - 1)
                row[s2] = cov
            result[s1] = row
        return result

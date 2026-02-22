# portfolio/risk_model/covariance/base.py
"""Covariance estimator protocol."""
from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class CovarianceEstimator(Protocol):
    """协方差估计器协议。"""
    name: str

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> Mapping[str, Mapping[str, float]]:
        """估计协方差矩阵。"""
        ...

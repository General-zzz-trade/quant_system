# portfolio/risk_model/correlation/base.py
"""Correlation estimator protocol."""
from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class CorrelationEstimator(Protocol):
    """相关性估计器协议。"""
    name: str

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> Mapping[str, Mapping[str, float]]:
        """估计相关矩阵。"""
        ...

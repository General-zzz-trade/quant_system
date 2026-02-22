# portfolio/risk_model/volatility/base.py
"""Volatility estimator protocol."""
from __future__ import annotations

from typing import Protocol, Sequence


class VolatilityEstimator(Protocol):
    """波动率估计器协议。"""
    name: str

    def estimate(self, returns: Sequence[float]) -> float:
        """估计年化波动率。"""
        ...

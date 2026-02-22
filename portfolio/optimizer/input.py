# portfolio/optimizer/input.py
"""Optimization input — standardized input for portfolio optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class OptimizationInput:
    """优化器标准化输入。"""
    symbols: tuple[str, ...]
    expected_returns: Mapping[str, float]          # symbol → E[r]
    covariance: Mapping[str, Mapping[str, float]]  # symbol × symbol → cov
    current_weights: Mapping[str, float]           # symbol → current weight
    risk_free_rate: float = 0.0
    total_equity: float = 0.0

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

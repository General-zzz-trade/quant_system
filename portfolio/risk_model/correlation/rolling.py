# portfolio/risk_model/correlation/rolling.py
"""Rolling window correlation estimator."""
from __future__ import annotations

from typing import Mapping, Sequence

from _quant_hotpath import cpp_rolling_correlation as _cpp_rolling_correlation


class RollingCorrelation:
    """滚动窗口相关性估计。"""
    name: str = "rolling"

    def __init__(self, window: int = 60) -> None:
        self.window = window

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        matrix = [list(returns.get(s, [])) for s in symbols]
        corr_mat = _cpp_rolling_correlation(matrix, self.window)
        return {
            s1: {s2: corr_mat[i][j] for j, s2 in enumerate(symbols)}
            for i, s1 in enumerate(symbols)
        }

# portfolio/risk_model/correlation/rolling.py
"""Rolling window correlation estimator."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


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
        result: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            row: dict[str, float] = {}
            r1 = list(returns.get(s1, []))[-self.window:]
            for s2 in symbols:
                if s1 == s2:
                    row[s2] = 1.0
                    continue
                r2 = list(returns.get(s2, []))[-self.window:]
                row[s2] = self._pearson(r1, r2)
            result[s1] = row
        return result

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        n = min(len(x), len(y))
        if n < 2:
            return 0.0
        mx = sum(x[:n]) / n
        my = sum(y[:n]) / n
        cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
        sx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)) / (n - 1))
        sy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)) / (n - 1))
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return max(-1.0, min(1.0, cov / (sx * sy)))

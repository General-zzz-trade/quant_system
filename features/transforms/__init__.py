# features/transforms
"""Feature transforms — normalization, scaling, windowing."""
from __future__ import annotations

import math
from typing import Sequence


def zscore_normalize(values: Sequence[float]) -> list[float]:
    """Z-score 标准化。"""
    n = len(values)
    if n < 2:
        return [0.0] * n
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def min_max_normalize(values: Sequence[float]) -> list[float]:
    """Min-max 归一化到 [0, 1]。"""
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    rng = mx - mn
    if rng < 1e-12:
        return [0.5] * len(values)
    return [(v - mn) / rng for v in values]


def rolling_apply(
    values: Sequence[float],
    window: int,
    fn: type | None = None,
) -> list[float]:
    """滚动窗口应用函数 (默认: 均值)。"""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        win = values[start:i + 1]
        if fn is None:
            result.append(sum(win) / len(win))
        else:
            result.append(fn(win))
    return result


def log_return(prices: Sequence[float]) -> list[float]:
    """对数收益率。"""
    result = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            result.append(math.log(prices[i] / prices[i - 1]))
        else:
            result.append(0.0)
    return result

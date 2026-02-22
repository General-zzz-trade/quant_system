# alpha/validation
"""Alpha model validation — backtesting and quality checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from alpha.base import Signal


@dataclass(frozen=True)
class ValidationMetrics:
    """验证指标。"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    n_signals: int = 0


def compute_hit_rate(signals: Sequence[Signal], actuals: Sequence[float]) -> float:
    """计算信号的命中率。"""
    if not signals or not actuals:
        return 0.0
    n = min(len(signals), len(actuals))
    hits = 0
    for i in range(n):
        s = signals[i]
        if s.side == "long" and actuals[i] > 0:
            hits += 1
        elif s.side == "short" and actuals[i] < 0:
            hits += 1
        elif s.side == "flat":
            pass  # neutral, don't count
    return hits / n if n > 0 else 0.0

# portfolio/risk_model/tail/drawdown.py
"""Drawdown analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class DrawdownStats:
    """回撤统计。"""
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # periods
    current_drawdown: float
    calmar_ratio: float  # annualized return / max drawdown


def compute_drawdowns(returns: Sequence[float]) -> list[float]:
    """从收益率序列计算每期回撤。"""
    if not returns:
        return []
    cumulative = 0.0
    peak = 0.0
    drawdowns = []
    for r in returns:
        cumulative += r
        peak = max(peak, cumulative)
        dd = cumulative - peak
        drawdowns.append(dd)
    return drawdowns


def analyze_drawdowns(
    returns: Sequence[float],
    annualization: float = 365.0,
) -> DrawdownStats:
    """完整的回撤分析。"""
    if not returns:
        return DrawdownStats(0.0, 0.0, 0, 0.0, 0.0)

    dds = compute_drawdowns(returns)
    max_dd = min(dds) if dds else 0.0

    # 平均回撤 (只看负值)
    neg_dds = [d for d in dds if d < 0]
    avg_dd = sum(neg_dds) / len(neg_dds) if neg_dds else 0.0

    # 最长回撤持续期
    max_duration = 0
    current_duration = 0
    for d in dds:
        if d < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    current_dd = dds[-1] if dds else 0.0

    # Calmar ratio
    total_return = sum(returns)
    ann_return = total_return * annualization / len(returns)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    return DrawdownStats(
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        max_drawdown_duration=max_duration,
        current_drawdown=current_dd,
        calmar_ratio=calmar,
    )

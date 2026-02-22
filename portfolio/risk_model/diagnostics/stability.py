# portfolio/risk_model/diagnostics/stability.py
"""Model parameter stability analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """稳定性检查结果。"""
    symbol: str
    metric: str
    current: float
    previous: float
    change_pct: float
    is_stable: bool


def check_volatility_stability(
    current_vols: Mapping[str, float],
    previous_vols: Mapping[str, float],
    max_change_pct: float = 50.0,
) -> list[StabilityResult]:
    """检查波动率估计的稳定性。"""
    results = []
    for s in current_vols:
        curr = current_vols[s]
        prev = previous_vols.get(s, curr)
        if prev > 0:
            change = abs(curr - prev) / prev * 100
        else:
            change = 0.0
        results.append(StabilityResult(
            symbol=s,
            metric="volatility",
            current=curr,
            previous=prev,
            change_pct=change,
            is_stable=change <= max_change_pct,
        ))
    return results


def check_correlation_stability(
    current_corr: Mapping[str, Mapping[str, float]],
    previous_corr: Mapping[str, Mapping[str, float]],
    max_abs_change: float = 0.3,
) -> list[StabilityResult]:
    """检查相关性估计的稳定性。"""
    results = []
    for s1 in current_corr:
        for s2 in current_corr.get(s1, {}):
            if s1 >= s2:
                continue
            curr = current_corr[s1].get(s2, 0.0)
            prev = previous_corr.get(s1, {}).get(s2, curr)
            change = abs(curr - prev)
            results.append(StabilityResult(
                symbol=f"{s1}/{s2}",
                metric="correlation",
                current=curr,
                previous=prev,
                change_pct=change * 100,
                is_stable=change <= max_abs_change,
            ))
    return results

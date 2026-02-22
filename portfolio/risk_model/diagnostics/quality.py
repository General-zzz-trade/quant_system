# portfolio/risk_model/diagnostics/quality.py
"""Risk model quality metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class QualityMetric:
    """模型质量指标。"""
    name: str
    value: float
    threshold: float
    passed: bool


def check_positive_definite(
    covariance: Mapping[str, Mapping[str, float]],
) -> QualityMetric:
    """检查协方差矩阵对角线是否为正（必要条件）。"""
    symbols = list(covariance.keys())
    min_diag = float("inf")
    for s in symbols:
        d = covariance.get(s, {}).get(s, 0.0)
        min_diag = min(min_diag, d)
    return QualityMetric(
        name="positive_diagonal",
        value=min_diag,
        threshold=0.0,
        passed=min_diag > 0,
    )


def check_correlation_bounds(
    correlation: Mapping[str, Mapping[str, float]],
) -> QualityMetric:
    """检查相关系数是否在 [-1, 1] 范围内。"""
    symbols = list(correlation.keys())
    max_abs = 0.0
    for s1 in symbols:
        for s2 in symbols:
            if s1 != s2:
                val = abs(correlation.get(s1, {}).get(s2, 0.0))
                max_abs = max(max_abs, val)
    return QualityMetric(
        name="correlation_bounds",
        value=max_abs,
        threshold=1.0,
        passed=max_abs <= 1.0 + 1e-6,
    )


def check_condition_number(
    covariance: Mapping[str, Mapping[str, float]],
    max_condition: float = 1000.0,
) -> QualityMetric:
    """简化条件数检查: max_diag / min_diag。"""
    symbols = list(covariance.keys())
    if not symbols:
        return QualityMetric("condition_number", 1.0, max_condition, True)
    diags = [covariance.get(s, {}).get(s, 0.0) for s in symbols]
    min_d = min(diags) if diags else 0.0
    max_d = max(diags) if diags else 0.0
    cond = max_d / max(min_d, 1e-12) if min_d > 0 else float("inf")
    return QualityMetric(
        name="condition_number",
        value=cond,
        threshold=max_condition,
        passed=cond <= max_condition,
    )

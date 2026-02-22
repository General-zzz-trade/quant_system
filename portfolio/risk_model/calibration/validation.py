# portfolio/risk_model/calibration/validation.py
"""Calibration validation — checks model calibration quality."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class CalibrationCheck:
    """标定质量检查结果。"""
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


def check_stationarity(
    returns: Sequence[float], window: int = 60
) -> CalibrationCheck:
    """简单平稳性检查: 前后半段均值是否差异过大。"""
    n = len(returns)
    if n < window:
        return CalibrationCheck("stationarity", True, 0.0, 0.0, "too few obs")
    mid = n // 2
    m1 = sum(returns[:mid]) / mid
    m2 = sum(returns[mid:]) / (n - mid)
    vol = math.sqrt(sum(r ** 2 for r in returns) / n) if n > 0 else 1.0
    ratio = abs(m1 - m2) / max(vol, 1e-12)
    passed = ratio < 2.0
    return CalibrationCheck("stationarity", passed, ratio, 2.0)


def check_sample_size(
    n_obs: int, min_required: int = 30
) -> CalibrationCheck:
    """检查样本量是否充足。"""
    return CalibrationCheck(
        "sample_size",
        n_obs >= min_required,
        float(n_obs),
        float(min_required),
    )


def validate_calibration(
    returns: Mapping[str, Sequence[float]],
    min_obs: int = 30,
) -> list[CalibrationCheck]:
    """批量标定质量检查。"""
    checks = []
    for symbol, rets in returns.items():
        checks.append(check_sample_size(len(rets), min_obs))
        checks.append(check_stationarity(rets))
    return checks

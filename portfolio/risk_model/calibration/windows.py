# portfolio/risk_model/calibration/wndows.py
"""Calibration window management."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class WindowType(Enum):
    ROLLING = "rolling"
    EXPANDING = "expanding"


@dataclass(frozen=True, slots=True)
class CalibrationWindow:
    """标定窗口配置。"""
    window_type: WindowType = WindowType.ROLLING
    length: int = 252
    min_observations: int = 30

    def select(self, data: Sequence[float]) -> list[float]:
        """从数据中选取窗口内的观测值。"""
        n = len(data)
        if n < self.min_observations:
            return list(data)
        if self.window_type == WindowType.EXPANDING:
            return list(data)
        # Rolling
        start = max(0, n - self.length)
        return list(data[start:])

    def is_valid(self, data: Sequence[float]) -> bool:
        """检查数据是否满足最小观测数要求。"""
        return len(data) >= self.min_observations

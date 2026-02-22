# context/regime/regime_state.py
"""Regime state — market regime detection and tracking."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RegimeType(str, Enum):
    """市场状态类型。"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class RegimeState:
    """市场状态追踪。"""

    def __init__(self) -> None:
        self._current: RegimeType = RegimeType.UNKNOWN
        self._confidence: float = 0.0
        self._last_change_bar: int = 0

    def update(self, regime: RegimeType, *, confidence: float = 1.0, bar_index: int = 0) -> None:
        if regime != self._current:
            self._last_change_bar = bar_index
        self._current = regime
        self._confidence = confidence

    @property
    def current(self) -> RegimeType:
        return self._current

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def bars_since_change(self) -> int:
        return 0  # 需要外部 bar_index 来计算

"""Compatibility shell: the position projection lives in trading.domain.position (migration step 7)."""
from __future__ import annotations

from trading.domain.position import StrategyPosition

__all__ = ["StrategyPosition"]

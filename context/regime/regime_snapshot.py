# context/regime/regime_snapshot.py
"""Regime snapshot — read-only view."""
from __future__ import annotations

from dataclasses import dataclass

from context.regime.regime_state import RegimeType


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """市场状态快照。"""
    regime: RegimeType
    confidence: float
    bars_since_change: int = 0

    @property
    def is_trending(self) -> bool:
        return self.regime in (RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN)

    @property
    def is_ranging(self) -> bool:
        return self.regime == RegimeType.RANGING

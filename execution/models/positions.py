# execution/models/positions.py
"""Venue-reported position snapshot (execution layer view)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
from trading.domain.venue import VenuePosition  # noqa: F401  re-exported


# VenuePosition: moved to trading.domain.venue (migration step 8)


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """一次完整的持仓快照（多个品种）。"""
    venue: str
    positions: tuple[VenuePosition, ...]
    ts_ms: int = 0

    def get(self, symbol: str) -> Optional[VenuePosition]:
        s = symbol.upper()
        for p in self.positions:
            if p.symbol == s:
                return p
        return None

    @property
    def symbols(self) -> Sequence[str]:
        return [p.symbol for p in self.positions]

    @property
    def active(self) -> Sequence[VenuePosition]:
        return [p for p in self.positions if not p.is_flat]


# 兼容别名
Position = VenuePosition

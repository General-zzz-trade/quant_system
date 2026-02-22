# context/market/account/position_state.py
"""Context-level position state — per-symbol position tracking."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional


@dataclass(frozen=True, slots=True)
class ContextPositionSnapshot:
    """上下文中的仓位快照。"""
    symbol: str
    venue: str
    qty: Decimal          # signed
    avg_price: Optional[Decimal]
    unrealized_pnl: Decimal = Decimal("0")


class ContextPositionState:
    """上下文中的仓位状态管理。"""

    def __init__(self) -> None:
        self._positions: Dict[str, dict] = {}  # key = "venue|symbol"

    def update(
        self, *, venue: str, symbol: str,
        qty: Decimal, avg_price: Optional[Decimal] = None,
    ) -> None:
        key = f"{venue}|{symbol}"
        self._positions[key] = {
            "symbol": symbol, "venue": venue,
            "qty": qty, "avg_price": avg_price,
            "unrealized_pnl": Decimal("0"),
        }

    def mark_to_market(self, *, venue: str, symbol: str, mark_price: Decimal) -> None:
        key = f"{venue}|{symbol}"
        pos = self._positions.get(key)
        if pos and pos["avg_price"] is not None and pos["qty"] != 0:
            diff = mark_price - pos["avg_price"]
            pos["unrealized_pnl"] = diff * pos["qty"]

    def get(self, venue: str, symbol: str) -> Optional[ContextPositionSnapshot]:
        key = f"{venue}|{symbol}"
        p = self._positions.get(key)
        if p is None:
            return None
        return ContextPositionSnapshot(
            symbol=p["symbol"], venue=p["venue"],
            qty=p["qty"], avg_price=p["avg_price"],
            unrealized_pnl=p["unrealized_pnl"],
        )

    @property
    def all_keys(self) -> list[str]:
        return list(self._positions.keys())

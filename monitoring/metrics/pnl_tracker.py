"""Real-time P&L tracker with equity curve history.

Tracks running P&L, equity, and drawdown metrics in real time.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PnLSnapshot:
    """Point-in-time P&L snapshot."""
    ts: float                  # monotonic time
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    peak_equity: Decimal
    drawdown_pct: float
    trade_count: int


class PnLTracker:
    """Tracks real-time P&L and equity curve."""

    def __init__(self, starting_equity: Decimal = Decimal("0")) -> None:
        self._starting_equity = starting_equity
        self._realized_pnl = Decimal("0")
        self._unrealized_pnl = Decimal("0")
        self._peak_equity = starting_equity
        self._trade_count = 0
        self._history: List[PnLSnapshot] = []
        self._max_history = 10000

    def on_fill(self, realized_pnl: Decimal) -> None:
        """Record a realized P&L from a fill."""
        self._realized_pnl += realized_pnl
        self._trade_count += 1

    def update_unrealized(self, unrealized_pnl: Decimal) -> None:
        """Update current unrealized P&L."""
        self._unrealized_pnl = unrealized_pnl

    def snapshot(self) -> PnLSnapshot:
        """Capture current state."""
        equity = self._starting_equity + self._realized_pnl + self._unrealized_pnl

        if equity > self._peak_equity:
            self._peak_equity = equity

        dd_pct = 0.0
        if self._peak_equity > 0:
            dd_pct = float((self._peak_equity - equity) / self._peak_equity * 100)

        snap = PnLSnapshot(
            ts=time.monotonic(),
            equity=equity,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=self._unrealized_pnl,
            peak_equity=self._peak_equity,
            drawdown_pct=dd_pct,
            trade_count=self._trade_count,
        )

        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snap

    @property
    def equity(self) -> Decimal:
        return self._starting_equity + self._realized_pnl + self._unrealized_pnl

    @property
    def history(self) -> List[PnLSnapshot]:
        return list(self._history)

    @property
    def max_drawdown_pct(self) -> float:
        if not self._history:
            return 0.0
        return max(s.drawdown_pct for s in self._history)

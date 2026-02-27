"""Execution embargo adapter for backtest look-ahead bias prevention.

Wraps BacktestExecutionAdapter to delay order execution by N bars,
preventing same-bar signal-to-fill look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple


@dataclass
class EmbargoExecutionAdapter:
    """Delays order execution by embargo_bars to prevent look-ahead bias.

    Implements ExecutionAdapter protocol: send_order returns [] (orders queued).
    Call on_bar() each bar to execute orders whose embargo has expired.
    """

    inner: Any  # BacktestExecutionAdapter
    embargo_bars: int = 1

    _pending: List[Tuple[int, Any]] = field(default_factory=list, repr=False)
    _current_bar: int = field(default=0, repr=False)

    def send_order(self, order_event: Any) -> List[Any]:
        """Queue order for deferred execution. Returns [] (fills via on_bar)."""
        if self.embargo_bars <= 0:
            return list(self.inner.send_order(order_event))
        self._pending.append((self._current_bar + self.embargo_bars, order_event))
        return []

    def set_bar(self, bar_index: int) -> None:
        """Set current bar index (called before coordinator.emit for this bar)."""
        self._current_bar = bar_index

    def on_bar(self, bar_index: int) -> List[Any]:
        """Execute all orders whose embargo has expired. Returns fill events."""
        ready = [order for (target, order) in self._pending if bar_index >= target]
        self._pending = [(t, o) for (t, o) in self._pending if bar_index < t]
        fills: List[Any] = []
        for order in ready:
            fills.extend(self.inner.send_order(order))
        return fills

    @property
    def pending_count(self) -> int:
        return len(self._pending)

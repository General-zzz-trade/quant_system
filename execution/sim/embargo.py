"""Execution embargo adapter for backtest look-ahead bias prevention.

Wraps BacktestExecutionAdapter to delay order execution by N bars,
preventing same-bar signal-to-fill look-ahead bias.

Fill price for embargoed orders uses the bar's **open** price (passed via
``on_bar(bar_index, open_price=...)``), which is the first tradeable price
after the embargo expires — matching realistic execution timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple


@dataclass
class EmbargoExecutionAdapter:
    """Delays order execution by embargo_bars to prevent look-ahead bias.

    Implements ExecutionAdapter protocol: send_order returns [] (orders queued).
    Call on_bar() each bar to execute orders whose embargo has expired.

    When ``open_price`` is provided to ``on_bar()``, embargoed fills use
    that price instead of the stale coordinator state — this is the bar's
    open price, the first price available after the embargo window.
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

    def on_bar(
        self,
        bar_index: int,
        open_price: Optional[Decimal] = None,
    ) -> List[Any]:
        """Execute all orders whose embargo has expired. Returns fill events.

        Parameters
        ----------
        bar_index : int
            Current bar index.
        open_price : Decimal, optional
            Open price of the current bar.  When supplied, embargoed orders
            are stamped with this price so the inner adapter fills at the
            realistic first-available price rather than the stale close of
            the previous bar.
        """
        ready = [order for (target, order) in self._pending if bar_index >= target]
        self._pending = [(t, o) for (t, o) in self._pending if bar_index < t]
        fills: List[Any] = []
        for order in ready:
            if open_price is not None:
                order = self._stamp_price(order, open_price)
            fills.extend(self.inner.send_order(order))
        return fills

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @staticmethod
    def _stamp_price(order: Any, price: Decimal) -> Any:
        """Return a copy of *order* with ``price`` set to *price*.

        If the order is a SimpleNamespace we mutate directly (cheap);
        otherwise we wrap it in a SimpleNamespace that delegates attribute
        lookups to the original.
        """
        if isinstance(order, SimpleNamespace):
            order.price = price
            return order
        # Wrap: copy all attrs, override price
        if is_dataclass(order):
            ns = SimpleNamespace(**{f.name: getattr(order, f.name) for f in fields(order)})
        else:
            ns = SimpleNamespace(**{k: getattr(order, k) for k in vars(order)})
        ns.price = price
        return ns

"""Tests for EmbargoExecutionAdapter."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from execution.sim.embargo import EmbargoExecutionAdapter


class FakeBacktestAdapter:
    """Minimal mock of BacktestExecutionAdapter."""

    def __init__(self) -> None:
        self.filled: List[Any] = []

    def send_order(self, order_event: Any) -> List[Any]:
        fill = SimpleNamespace(
            event_type="fill",
            symbol=getattr(order_event, "symbol", "TEST"),
            side=getattr(order_event, "side", "buy"),
            qty=getattr(order_event, "qty", 1),
            price=100.0,
        )
        self.filled.append(fill)
        return [fill]


def _order(symbol: str = "BTCUSDT", side: str = "buy", qty: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, side=side, qty=qty)


class TestEmbargoBasic:
    def test_embargo_1_delays_to_next_bar(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=1)

        adapter.set_bar(0)
        result = adapter.send_order(_order())
        assert result == []  # no immediate fill
        assert adapter.pending_count == 1

        # Bar 1: embargo expires
        fills = adapter.on_bar(1)
        assert len(fills) == 1
        assert adapter.pending_count == 0

    def test_embargo_0_fills_immediately(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=0)

        adapter.set_bar(0)
        result = adapter.send_order(_order())
        assert len(result) == 1  # immediate fill
        assert adapter.pending_count == 0

    def test_embargo_multi_bar(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=3)

        adapter.set_bar(0)
        adapter.send_order(_order())

        assert adapter.on_bar(1) == []
        assert adapter.on_bar(2) == []
        fills = adapter.on_bar(3)
        assert len(fills) == 1

    def test_multiple_orders_different_bars(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=1)

        adapter.set_bar(0)
        adapter.send_order(_order(side="buy"))
        adapter.set_bar(1)
        adapter.send_order(_order(side="sell"))

        fills_1 = adapter.on_bar(1)
        assert len(fills_1) == 1  # first order ready
        assert fills_1[0].side == "buy"

        fills_2 = adapter.on_bar(2)
        assert len(fills_2) == 1  # second order ready
        assert fills_2[0].side == "sell"

    def test_pending_cleared_after_execution(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=1)

        adapter.set_bar(0)
        adapter.send_order(_order())
        adapter.send_order(_order())
        assert adapter.pending_count == 2

        fills = adapter.on_bar(1)
        assert len(fills) == 2
        assert adapter.pending_count == 0

    def test_on_bar_idempotent(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=1)

        adapter.set_bar(0)
        adapter.send_order(_order())
        adapter.on_bar(1)

        # Calling again should produce no fills
        fills = adapter.on_bar(1)
        assert fills == []

    def test_late_on_bar_executes_all_overdue(self) -> None:
        inner = FakeBacktestAdapter()
        adapter = EmbargoExecutionAdapter(inner=inner, embargo_bars=1)

        adapter.set_bar(0)
        adapter.send_order(_order())
        adapter.set_bar(1)
        adapter.send_order(_order())

        # Skip bar 1, call bar 5 directly — both should execute
        fills = adapter.on_bar(5)
        assert len(fills) == 2

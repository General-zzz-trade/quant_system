"""Tests for MakerFirst execution algorithm."""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Optional

import pytest

from execution.algos.maker_first import MakerFirstAlgo, MakerFirstConfig, MakerFirstOrder


# ── Helpers ─────────────────────────────────────────────────


class _FakeSubmit:
    """Records calls and returns a configurable fill price."""

    def __init__(self, fill_price: Optional[Decimal] = Decimal("40000")) -> None:
        self.calls: list[tuple[str, str, Decimal]] = []
        self.fill_price = fill_price

    def __call__(self, symbol: str, side: str, qty: Decimal) -> Optional[Decimal]:
        self.calls.append((symbol, side, qty))
        return self.fill_price


# ── Tests ───────────────────────────────────────────────────


class TestMakerFirstCreate:
    def test_buy_limit_below_mid(self) -> None:
        submit = _FakeSubmit()
        algo = MakerFirstAlgo(submit, MakerFirstConfig(price_offset_bps=10.0))
        order = algo.create("BTCUSDT", "buy", Decimal("1"), mid_price=Decimal("40000"))
        assert order.limit_price is not None
        assert order.limit_price < Decimal("40000")
        # offset = 40000 * 10/10000 = 40
        assert order.limit_price == Decimal("40000") - Decimal("40")

    def test_sell_limit_above_mid(self) -> None:
        submit = _FakeSubmit()
        algo = MakerFirstAlgo(submit, MakerFirstConfig(price_offset_bps=10.0))
        order = algo.create("BTCUSDT", "sell", Decimal("1"), mid_price=Decimal("40000"))
        assert order.limit_price is not None
        assert order.limit_price > Decimal("40000")

    def test_order_fields(self) -> None:
        submit = _FakeSubmit()
        algo = MakerFirstAlgo(submit)
        order = algo.create("ETHUSDT", "buy", Decimal("5"), mid_price=Decimal("2000"))
        assert order.symbol == "ETHUSDT"
        assert order.side == "buy"
        assert order.total_qty == Decimal("5")
        assert order.filled_qty == Decimal("0")
        assert not order.is_complete


class TestMakerFirstExecution:
    def test_passive_fill(self) -> None:
        """Submit returns a price -> fill completes immediately."""
        submit = _FakeSubmit(Decimal("39990"))
        algo = MakerFirstAlgo(submit)
        order = algo.create("BTCUSDT", "buy", Decimal("1"), mid_price=Decimal("40000"))
        result = algo.tick(order)
        assert result is not None
        assert result.qty == Decimal("1")
        assert order.is_complete

    def test_no_fill_when_submit_returns_none(self) -> None:
        submit = _FakeSubmit(None)
        algo = MakerFirstAlgo(submit, MakerFirstConfig(maker_timeout_sec=100))
        order = algo.create("BTCUSDT", "buy", Decimal("1"), mid_price=Decimal("40000"))
        result = algo.tick(order)
        assert result is None
        assert not order.is_complete

    def test_timeout_triggers_taker(self) -> None:
        """After 90% of timeout, should force market order."""
        submit = _FakeSubmit(Decimal("40010"))
        config = MakerFirstConfig(maker_timeout_sec=1.0)
        algo = MakerFirstAlgo(submit, config)
        order = algo.create("BTCUSDT", "buy", Decimal("1"), mid_price=Decimal("40000"))
        # Simulate timeout by backdating created_at
        order.created_at = time.monotonic() - 2.0
        result = algo.tick(order)
        assert result is not None
        assert order.is_complete
        assert result.mode == "taker"

    def test_fill_ratio_property(self) -> None:
        submit = _FakeSubmit()
        algo = MakerFirstAlgo(submit)
        order = algo.create("BTCUSDT", "buy", Decimal("10"), mid_price=Decimal("100"))
        assert order.fill_ratio == 0.0
        order.filled_qty = Decimal("3")
        assert order.fill_ratio == pytest.approx(0.3)

    def test_already_complete_noop(self) -> None:
        submit = _FakeSubmit(Decimal("100"))
        algo = MakerFirstAlgo(submit)
        order = algo.create("BTCUSDT", "buy", Decimal("1"), mid_price=Decimal("100"))
        order.is_complete = True
        result = algo.tick(order)
        assert result is None
        assert len(submit.calls) == 0

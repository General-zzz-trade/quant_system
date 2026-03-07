# tests_unit/test_shadow_adapter.py
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from execution.sim.shadow_adapter import ShadowExecutionAdapter


def _make_order(symbol="BTCUSDT", side="BUY", qty=Decimal("0.1")):
    return SimpleNamespace(symbol=symbol, side=side, qty=qty)


class TestShadowExecutionAdapter:
    def test_send_order_records_entry(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: Decimal("50000"),
        )
        result = adapter.send_order(_make_order())
        assert result == []
        assert len(adapter.order_log) == 1

        entry = adapter.order_log[0]
        assert entry["symbol"] == "BTCUSDT"
        assert entry["side"] == "BUY"
        assert entry["simulated"] is True
        assert entry["fill_price"] is not None

    def test_buy_slippage_increases_price(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: Decimal("50000"),
            slippage_bps=Decimal("10"),  # 10 bps
        )
        adapter.send_order(_make_order(side="BUY"))
        fill_price = Decimal(adapter.order_log[0]["fill_price"])
        assert fill_price > Decimal("50000")

    def test_sell_slippage_decreases_price(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: Decimal("50000"),
            slippage_bps=Decimal("10"),
        )
        adapter.send_order(_make_order(side="SELL"))
        fill_price = Decimal(adapter.order_log[0]["fill_price"])
        assert fill_price < Decimal("50000")

    def test_fee_calculation(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: Decimal("50000"),
            fee_bps=Decimal("4"),
            slippage_bps=Decimal("0"),
        )
        adapter.send_order(_make_order(qty=Decimal("1")))
        fee = Decimal(adapter.order_log[0]["fee"])
        # fee = 50000 * 1 * 4/10000 = 20
        assert fee == Decimal("20")

    def test_no_price_source_returns_none(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: None,
        )
        adapter.send_order(_make_order())
        entry = adapter.order_log[0]
        assert entry["fill_price"] is None
        assert entry["fee"] is None

    def test_multiple_orders_accumulate(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: Decimal("100"),
        )
        for _ in range(5):
            adapter.send_order(_make_order())
        assert len(adapter.order_log) == 5

    def test_order_log_is_copy(self):
        adapter = ShadowExecutionAdapter(
            price_source=lambda sym: Decimal("100"),
        )
        adapter.send_order(_make_order())
        log1 = adapter.order_log
        log1.clear()
        assert len(adapter.order_log) == 1  # internal log unaffected

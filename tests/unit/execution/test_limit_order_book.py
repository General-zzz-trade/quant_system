# tests/unit/execution/test_limit_order_book.py
"""Tests for simulated limit order book."""
import pytest

from execution.sim.limit_order_book import (
    OrderType,
    OrderStatus,
    SimulatedOrderBook,
)


class TestMarketOrders:
    def test_market_buy_fills_at_open_plus_slippage(self):
        book = SimulatedOrderBook(fee_bps=4, slippage_bps=2)
        order = book.submit_order("ETHUSDT", side=1, qty=1.0, bar=0)
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1980, close=2020, volume=100)
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(2000 * 1.0002)  # 2bp slippage
        assert order.status == OrderStatus.FILLED

    def test_market_sell_fills_at_open_minus_slippage(self):
        book = SimulatedOrderBook(slippage_bps=2)
        order = book.submit_order("ETHUSDT", side=-1, qty=1.0, bar=0)
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1980, close=2020, volume=100)
        assert fills[0].fill_price == pytest.approx(2000 * 0.9998)


class TestLimitOrders:
    def test_buy_limit_fills_when_low_touches(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=1, qty=1.0,
                                  order_type=OrderType.LIMIT, price=1990, bar=0)
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1985, close=2020, volume=100)
        assert len(fills) == 1
        assert fills[0].fill_price == 1990  # fills at limit price
        assert order.status == OrderStatus.FILLED

    def test_buy_limit_no_fill_when_low_above(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=1, qty=1.0,
                                  order_type=OrderType.LIMIT, price=1990, bar=0)
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1995, close=2020, volume=100)
        assert len(fills) == 0
        assert order.status == OrderStatus.QUEUED

    def test_sell_limit_fills_when_high_touches(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=-1, qty=1.0,
                                  order_type=OrderType.LIMIT, price=2040, bar=0)
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1980, close=2020, volume=100)
        assert len(fills) == 1
        assert fills[0].fill_price == 2040


class TestPartialFills:
    def test_partial_fill_when_volume_insufficient(self):
        book = SimulatedOrderBook(max_participation=0.10)
        order = book.submit_order("ETHUSDT", side=1, qty=100.0, bar=0)
        # volume=50, max_participation=10% → max fill = 5
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1980, close=2020, volume=50)
        assert len(fills) == 1
        assert fills[0].fill_qty == 5.0
        assert fills[0].is_partial is True
        assert order.status == OrderStatus.PARTIALLY_FILLED


class TestStopOrders:
    def test_stop_buy_triggers_on_high(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=1, qty=1.0,
                                  order_type=OrderType.STOP, stop_price=2030, bar=0)
        assert order.status == OrderStatus.PENDING

        # Bar doesn't trigger
        fills = book.process_bar(bar=0, open_price=2000, high=2020, low=1990, close=2010, volume=100)
        assert len(fills) == 0

        # Bar triggers stop
        fills = book.process_bar(bar=1, open_price=2010, high=2040, low=2005, close=2035, volume=100)
        assert len(fills) == 1  # converted to market, filled


class TestOrderExpiry:
    def test_ttl_expires(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=1, qty=1.0,
                                  order_type=OrderType.LIMIT, price=1900,
                                  bar=0, ttl_bars=3)
        # 4 bars pass without fill (expire at bar >= created + ttl)
        for b in range(4):
            book.process_bar(bar=b, open_price=2000, high=2050, low=1950, close=2020, volume=100)

        assert order.status == OrderStatus.EXPIRED


class TestCancelOrder:
    def test_cancel_queued(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=1, qty=1.0,
                                  order_type=OrderType.LIMIT, price=1900, bar=0)
        assert book.cancel_order(order.order_id) is True
        assert order.status == OrderStatus.CANCELED

    def test_cancel_filled_fails(self):
        book = SimulatedOrderBook()
        order = book.submit_order("ETHUSDT", side=1, qty=1.0, bar=0)
        book.process_bar(bar=0, open_price=2000, high=2050, low=1980, close=2020, volume=100)
        assert order.status == OrderStatus.FILLED
        assert book.cancel_order(order.order_id) is False


class TestFees:
    def test_fee_calculated(self):
        book = SimulatedOrderBook(fee_bps=4)
        book.submit_order("ETHUSDT", side=1, qty=1.0, bar=0)
        fills = book.process_bar(bar=0, open_price=2000, high=2050, low=1980, close=2020, volume=100)
        # Fee = 1.0 × ~2000 × 4/10000 = ~0.8
        assert fills[0].fee == pytest.approx(1.0 * 2000.04 * 0.0004, rel=0.01)

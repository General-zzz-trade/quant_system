"""Tests for HFT strategies — market making and order book signals."""
from __future__ import annotations

from decimal import Decimal

import pytest

from strategies.hft.market_making import MarketMaker, Quote, QuotePair
from strategies.hft.order_book import (
    OrderBookSignal,
    analyze_book,
    compute_imbalance,
    compute_weighted_mid,
)
from execution.adapters.binance.depth_processor import OrderBookLevel, OrderBookSnapshot


# ── Helpers ──────────────────────────────────────────────────

def _make_book(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
    symbol: str = "BTCUSDT",
) -> OrderBookSnapshot:
    """Build an OrderBookSnapshot from (price, qty) string tuples."""
    return OrderBookSnapshot(
        symbol=symbol,
        bids=tuple(OrderBookLevel(Decimal(p), Decimal(q)) for p, q in bids),
        asks=tuple(OrderBookLevel(Decimal(p), Decimal(q)) for p, q in asks),
        ts_ms=1700000000000,
        last_update_id=12345,
    )


# ── MarketMaker ──────────────────────────────────────────────

class TestMarketMaker:
    def test_basic_quotes(self):
        mm = MarketMaker(symbol="BTCUSDT", base_spread_bps=10.0)
        quotes = mm.compute_quotes(Decimal("50000"))
        assert quotes is not None
        assert quotes.bid.price < Decimal("50000")
        assert quotes.ask.price > Decimal("50000")
        assert quotes.bid.side == "bid"
        assert quotes.ask.side == "ask"

    def test_spread_symmetric_no_inventory(self):
        mm = MarketMaker(symbol="BTC", base_spread_bps=10.0)
        quotes = mm.compute_quotes(Decimal("10000"))
        assert quotes is not None
        mid = Decimal("10000")
        half_spread = mid * Decimal(str(10.0 / 2 / 10000))
        assert quotes.bid.price == pytest.approx(float(mid - half_spread), rel=1e-6)
        assert quotes.ask.price == pytest.approx(float(mid + half_spread), rel=1e-6)

    def test_inventory_skew_long(self):
        mm = MarketMaker(symbol="BTC", base_spread_bps=10.0, inventory_skew_bps=2.0, max_inventory=Decimal("10"))
        mm.update_position(Decimal("5"))  # long inventory, below max
        quotes = mm.compute_quotes(Decimal("10000"))
        assert quotes is not None
        # With long inventory, inventory_adj is positive, shifting both quotes down
        mm2 = MarketMaker(symbol="BTC", base_spread_bps=10.0, inventory_skew_bps=2.0, max_inventory=Decimal("10"))
        neutral_quotes = mm2.compute_quotes(Decimal("10000"))
        assert quotes.ask.price < neutral_quotes.ask.price

    def test_inventory_skew_short(self):
        mm = MarketMaker(symbol="BTC", base_spread_bps=10.0, inventory_skew_bps=2.0, max_inventory=Decimal("10"))
        mm.update_position(Decimal("-5"))  # short inventory, below max
        quotes = mm.compute_quotes(Decimal("10000"))
        assert quotes is not None
        mm2 = MarketMaker(symbol="BTC", base_spread_bps=10.0, inventory_skew_bps=2.0, max_inventory=Decimal("10"))
        neutral_quotes = mm2.compute_quotes(Decimal("10000"))
        assert quotes.bid.price > neutral_quotes.bid.price

    def test_max_inventory_long_only_ask(self):
        mm = MarketMaker(symbol="BTC", max_inventory=Decimal("1.0"))
        mm.update_position(Decimal("1.0"))  # at max
        quotes = mm.compute_quotes(Decimal("10000"))
        assert quotes is not None
        assert quotes.bid.qty == Decimal("0")  # no bid
        assert quotes.ask.qty > 0  # only ask to reduce

    def test_max_inventory_short_only_bid(self):
        mm = MarketMaker(symbol="BTC", max_inventory=Decimal("1.0"))
        mm.update_position(Decimal("-1.0"))
        quotes = mm.compute_quotes(Decimal("10000"))
        assert quotes is not None
        assert quotes.ask.qty == Decimal("0")
        assert quotes.bid.qty > 0

    def test_zero_mid_price(self):
        mm = MarketMaker(symbol="BTC")
        quotes = mm.compute_quotes(Decimal("0"))
        assert quotes is None

    def test_negative_mid_price(self):
        mm = MarketMaker(symbol="BTC")
        quotes = mm.compute_quotes(Decimal("-100"))
        assert quotes is None

    def test_volatility_widens_spread(self):
        mm = MarketMaker(symbol="BTC", base_spread_bps=10.0)
        normal = mm.compute_quotes(Decimal("10000"))
        vol_quotes = mm.compute_quotes(Decimal("10000"), volatility=0.01)
        assert vol_quotes is not None
        assert normal is not None
        # High volatility should widen the spread
        normal_spread = normal.ask.price - normal.bid.price
        vol_spread = vol_quotes.ask.price - vol_quotes.bid.price
        assert vol_spread >= normal_spread


# ── Order book imbalance ─────────────────────────────────────

class TestImbalance:
    def test_balanced_book(self):
        book = _make_book(
            bids=[("100", "10"), ("99", "10")],
            asks=[("101", "10"), ("102", "10")],
        )
        imb = compute_imbalance(book, levels=2)
        assert imb == pytest.approx(0.0)

    def test_buy_pressure(self):
        book = _make_book(
            bids=[("100", "30")],
            asks=[("101", "10")],
        )
        imb = compute_imbalance(book, levels=1)
        assert imb == pytest.approx(0.5)  # (30-10)/(30+10)

    def test_sell_pressure(self):
        book = _make_book(
            bids=[("100", "5")],
            asks=[("101", "15")],
        )
        imb = compute_imbalance(book, levels=1)
        assert imb == pytest.approx(-0.5)  # (5-15)/(5+15)

    def test_empty_book(self):
        book = _make_book(bids=[], asks=[])
        imb = compute_imbalance(book, levels=5)
        assert imb == 0.0


# ── Weighted mid ─────────────────────────────────────────────

class TestWeightedMid:
    def test_symmetric_book(self):
        book = _make_book(
            bids=[("100", "10")],
            asks=[("102", "10")],
        )
        wmid = compute_weighted_mid(book)
        assert wmid is not None
        assert wmid == pytest.approx(101.0)  # equal weight => arithmetic mid

    def test_asymmetric_book(self):
        book = _make_book(
            bids=[("100", "10")],
            asks=[("102", "30")],
        )
        wmid = compute_weighted_mid(book)
        assert wmid is not None
        # Weight by opposite: bid_price * ask_qty + ask_price * bid_qty
        # (100 * 30 + 102 * 10) / 40 = 4020 / 40 = 100.5
        assert wmid == pytest.approx(Decimal("100.5"))

    def test_empty_book_returns_none(self):
        book = _make_book(bids=[], asks=[])
        assert compute_weighted_mid(book) is None

    def test_one_side_empty(self):
        book = _make_book(bids=[("100", "10")], asks=[])
        assert compute_weighted_mid(book) is None


# ── analyze_book ─────────────────────────────────────────────

class TestAnalyzeBook:
    def test_buy_pressure_signal(self):
        book = _make_book(
            bids=[("100", "50")],
            asks=[("101", "10")],
        )
        signal = analyze_book(book, levels=1, imbalance_threshold=0.3)
        assert signal.signal == "buy_pressure"
        assert signal.imbalance > 0.3

    def test_sell_pressure_signal(self):
        book = _make_book(
            bids=[("100", "10")],
            asks=[("101", "50")],
        )
        signal = analyze_book(book, levels=1, imbalance_threshold=0.3)
        assert signal.signal == "sell_pressure"
        assert signal.imbalance < -0.3

    def test_neutral_signal(self):
        book = _make_book(
            bids=[("100", "10")],
            asks=[("101", "12")],
        )
        signal = analyze_book(book, levels=1, imbalance_threshold=0.3)
        assert signal.signal == "neutral"

    def test_depth_ratio(self):
        book = _make_book(
            bids=[("100", "20"), ("99", "10")],
            asks=[("101", "10"), ("102", "5")],
        )
        signal = analyze_book(book, levels=2)
        assert signal.depth_ratio == pytest.approx(2.0)  # 30/15

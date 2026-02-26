"""Tests for Phase 3 HFT strategies and multi-exchange adapters."""
from __future__ import annotations

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch


# ── C3: HFT / Microstructure strategies ──

class TestMarketMaker:
    def test_compute_quotes_basic(self):
        from strategies.hft.market_making import MarketMaker

        mm = MarketMaker(
            symbol="BTCUSDT",
            base_spread_bps=10.0,
            qty_per_side=Decimal("0.1"),
        )
        result = mm.compute_quotes(Decimal("50000"))
        assert result is not None
        assert result.bid.price < Decimal("50000")
        assert result.ask.price > Decimal("50000")
        assert result.bid.qty == Decimal("0.1")
        assert result.spread_bps == 10.0

    def test_zero_mid_returns_none(self):
        from strategies.hft.market_making import MarketMaker

        mm = MarketMaker(symbol="TEST")
        assert mm.compute_quotes(Decimal("0")) is None
        assert mm.compute_quotes(Decimal("-1")) is None

    def test_inventory_skew(self):
        from strategies.hft.market_making import MarketMaker

        mm = MarketMaker(
            symbol="BTCUSDT",
            base_spread_bps=10.0,
            inventory_skew_bps=5.0,
        )
        # Long inventory: quotes shift down (cheaper bid, cheaper ask)
        mm.update_position(Decimal("0.5"))
        quotes = mm.compute_quotes(Decimal("50000"))
        assert quotes is not None

        mid = (quotes.bid.price + quotes.ask.price) / 2
        # Skew should push mid below 50000 when long
        assert mid < Decimal("50000")

    def test_max_inventory_one_sided(self):
        from strategies.hft.market_making import MarketMaker

        mm = MarketMaker(
            symbol="BTCUSDT",
            max_inventory=Decimal("1.0"),
        )
        mm.update_position(Decimal("1.5"))  # Over max
        quotes = mm.compute_quotes(Decimal("50000"))
        assert quotes is not None
        # Should only quote ask (to reduce long)
        assert quotes.bid.qty == Decimal("0")
        assert quotes.ask.qty > 0

    def test_volatility_widens_spread(self):
        from strategies.hft.market_making import MarketMaker

        mm = MarketMaker(
            symbol="BTCUSDT",
            base_spread_bps=10.0,
        )
        # High volatility should widen spread
        normal = mm.compute_quotes(Decimal("50000"))
        volatile = mm.compute_quotes(Decimal("50000"), volatility=0.05)

        assert normal is not None and volatile is not None
        normal_spread = normal.ask.price - normal.bid.price
        vol_spread = volatile.ask.price - volatile.bid.price
        assert vol_spread >= normal_spread


class TestOrderBookAnalysis:
    def _make_snapshot(self, bids, asks):
        from execution.adapters.binance.depth_processor import (
            OrderBookLevel, OrderBookSnapshot,
        )
        return OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=tuple(OrderBookLevel(price=Decimal(str(p)), qty=Decimal(str(q))) for p, q in bids),
            asks=tuple(OrderBookLevel(price=Decimal(str(p)), qty=Decimal(str(q))) for p, q in asks),
            ts_ms=1000,
            last_update_id=1,
        )

    def test_compute_imbalance_balanced(self):
        from strategies.hft.order_book import compute_imbalance

        snap = self._make_snapshot(
            [(100, 10), (99, 10)],
            [(101, 10), (102, 10)],
        )
        imb = compute_imbalance(snap, levels=2)
        assert imb == pytest.approx(0.0, abs=0.01)

    def test_compute_imbalance_buy_pressure(self):
        from strategies.hft.order_book import compute_imbalance

        snap = self._make_snapshot(
            [(100, 20), (99, 20)],
            [(101, 5), (102, 5)],
        )
        imb = compute_imbalance(snap, levels=2)
        assert imb > 0.5

    def test_weighted_mid(self):
        from strategies.hft.order_book import compute_weighted_mid

        snap = self._make_snapshot(
            [(100, 10)],
            [(102, 10)],
        )
        mid = compute_weighted_mid(snap)
        assert mid is not None
        assert mid == Decimal("101")  # Equal qty → simple mid

    def test_analyze_book_signal(self):
        from strategies.hft.order_book import analyze_book

        snap = self._make_snapshot(
            [(100, 30), (99, 25)],
            [(101, 5), (102, 5)],
        )
        signal = analyze_book(snap, levels=2)
        assert signal.signal == "buy_pressure"
        assert signal.imbalance > 0.3
        assert signal.spread_bps > 0

    def test_analyze_book_neutral(self):
        from strategies.hft.order_book import analyze_book

        snap = self._make_snapshot(
            [(100, 10), (99, 10)],
            [(101, 10), (102, 10)],
        )
        signal = analyze_book(snap, levels=2)
        assert signal.signal == "neutral"


# ── D3: CCXT venue adapter ──

class TestCcxtVenueAdapter:
    def test_import(self):
        from execution.adapters.generic.ccxt_venue import (
            CcxtVenueAdapter, VenueBalance, VenuePosition, VenueOrder,
        )
        # Just verify the classes exist and are importable
        assert VenueBalance is not None
        assert VenueOrder is not None

    def test_venue_balance_dataclass(self):
        from execution.adapters.generic.ccxt_venue import VenueBalance

        b = VenueBalance(
            currency="BTC",
            free=Decimal("1.5"),
            used=Decimal("0.5"),
            total=Decimal("2.0"),
        )
        assert b.currency == "BTC"
        assert b.total == Decimal("2.0")

    def test_venue_order_dataclass(self):
        from execution.adapters.generic.ccxt_venue import VenueOrder

        o = VenueOrder(
            order_id="123",
            symbol="BTC/USDT",
            side="buy",
            qty=Decimal("0.1"),
            price=Decimal("50000"),
            status="filled",
            filled_qty=Decimal("0.1"),
            avg_fill_price=Decimal("49950"),
        )
        assert o.order_id == "123"
        assert o.filled_qty == Decimal("0.1")

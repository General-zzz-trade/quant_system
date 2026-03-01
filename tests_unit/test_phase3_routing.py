"""Tests for Phase 3 smart routing and execution quality."""
from __future__ import annotations

import pytest
from decimal import Decimal

from execution.routing.smart_router import (
    SmartRouter, VenueQuote, RouteDecision, SplitRoute,
)
from execution.routing.execution_quality import (
    ExecutionQualityTracker, ExecutionRecord, QualityMetrics,
)


# ── D4: Smart routing ──

class _MockQuoteProvider:
    def __init__(self, quote: VenueQuote):
        self._quote = quote

    def get_quote(self, symbol: str):
        return self._quote


class TestSmartRouter:
    def test_register_venues(self):
        router = SmartRouter()
        router.register_venue("binance", _MockQuoteProvider(
            VenueQuote("binance", Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("10")),
        ))
        router.register_venue("sim", _MockQuoteProvider(
            VenueQuote("sim", Decimal("49990"), Decimal("50005"), Decimal("5"), Decimal("5")),
        ))
        assert len(router.venue_ids) == 2

    def test_route_buy_best_ask(self):
        router = SmartRouter()
        router.register_venue("binance", _MockQuoteProvider(
            VenueQuote("binance", Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("10")),
        ))
        router.register_venue("sim", _MockQuoteProvider(
            VenueQuote("sim", Decimal("49990"), Decimal("50005"), Decimal("5"), Decimal("5")),
        ))

        decision = router.route_order("BTCUSDT", "buy", Decimal("1"))
        assert decision is not None
        assert decision.venue_id == "sim"  # Lower ask
        assert decision.price == Decimal("50005")

    def test_route_sell_best_bid(self):
        router = SmartRouter()
        router.register_venue("binance", _MockQuoteProvider(
            VenueQuote("binance", Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("10")),
        ))
        router.register_venue("sim", _MockQuoteProvider(
            VenueQuote("sim", Decimal("49990"), Decimal("50005"), Decimal("5"), Decimal("5")),
        ))

        decision = router.route_order("BTCUSDT", "sell", Decimal("1"))
        assert decision is not None
        assert decision.venue_id == "binance"  # Higher bid
        assert decision.price == Decimal("50000")

    def test_route_no_venues_returns_none(self):
        router = SmartRouter()
        assert router.route_order("BTCUSDT", "buy", Decimal("1")) is None

    def test_split_order(self):
        router = SmartRouter()
        router.register_venue("binance", _MockQuoteProvider(
            VenueQuote("binance", Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("3")),
        ))
        router.register_venue("sim", _MockQuoteProvider(
            VenueQuote("sim", Decimal("49990"), Decimal("50005"), Decimal("5"), Decimal("5")),
        ))

        split = router.split_order("BTCUSDT", "buy", Decimal("7"))
        assert split is not None
        assert len(split.legs) == 2
        assert split.total_qty == Decimal("7")
        # First leg should be cheapest ask (sim at 50005)
        assert split.legs[0].venue_id == "sim"

    def test_unregister_venue(self):
        router = SmartRouter()
        router.register_venue("binance", _MockQuoteProvider(
            VenueQuote("binance", Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("10")),
        ))
        router.unregister_venue("binance")
        assert len(router.venue_ids) == 0

    def test_latency_filter(self):
        router = SmartRouter(max_latency_ms=100.0)
        router.register_venue("slow", _MockQuoteProvider(
            VenueQuote("slow", Decimal("50000"), Decimal("50005"), Decimal("10"), Decimal("10"), latency_ms=200),
        ))
        router.register_venue("fast", _MockQuoteProvider(
            VenueQuote("fast", Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("10"), latency_ms=50),
        ))

        decision = router.route_order("BTCUSDT", "buy", Decimal("1"))
        assert decision is not None
        assert decision.venue_id == "fast"  # Slow venue filtered out


# ── Execution quality ──

class TestExecutionQualityTracker:
    def _make_record(self, **kwargs):
        defaults = dict(
            order_id="1",
            symbol="BTCUSDT",
            side="buy",
            intended_qty=Decimal("1"),
            filled_qty=Decimal("1"),
            intended_price=Decimal("50000"),
            avg_fill_price=Decimal("50010"),
            venue_id="binance",
            submit_ts=1000.0,
            fill_ts=1000.05,
        )
        defaults.update(kwargs)
        return ExecutionRecord(**defaults)

    def test_record_and_count(self):
        tracker = ExecutionQualityTracker()
        tracker.record(self._make_record())
        assert tracker.record_count == 1

    def test_slippage_calculation(self):
        tracker = ExecutionQualityTracker()
        # Buy at 50000, filled at 50010 → 2 bps slippage
        tracker.record(self._make_record(
            intended_price=Decimal("50000"),
            avg_fill_price=Decimal("50010"),
        ))
        metrics = tracker.compute_metrics()
        assert metrics is not None
        assert metrics.avg_slippage_bps == pytest.approx(2.0, abs=0.1)

    def test_sell_slippage(self):
        tracker = ExecutionQualityTracker()
        # Sell at 50000, filled at 49990 → 2 bps slippage (unfavorable for sell)
        tracker.record(self._make_record(
            side="sell",
            intended_price=Decimal("50000"),
            avg_fill_price=Decimal("49990"),
        ))
        metrics = tracker.compute_metrics()
        assert metrics is not None
        assert metrics.avg_slippage_bps == pytest.approx(2.0, abs=0.1)

    def test_fill_rate(self):
        tracker = ExecutionQualityTracker()
        tracker.record(self._make_record(
            intended_qty=Decimal("2"),
            filled_qty=Decimal("1"),
        ))
        metrics = tracker.compute_metrics()
        assert metrics is not None
        assert metrics.fill_rate == pytest.approx(0.5)

    def test_venue_comparison(self):
        tracker = ExecutionQualityTracker()
        tracker.record(self._make_record(venue_id="binance"))
        tracker.record(self._make_record(venue_id="sim", order_id="2"))

        comparison = tracker.venue_comparison()
        assert "binance" in comparison
        assert "sim" in comparison

    def test_filter_by_symbol(self):
        tracker = ExecutionQualityTracker()
        tracker.record(self._make_record(symbol="BTCUSDT", order_id="1"))
        tracker.record(self._make_record(symbol="ETHUSDT", order_id="2"))

        btc_metrics = tracker.compute_metrics(symbol="BTCUSDT")
        assert btc_metrics is not None
        assert btc_metrics.total_orders == 1

    def test_latency(self):
        tracker = ExecutionQualityTracker()
        tracker.record(self._make_record(submit_ts=1000.0, fill_ts=1000.1))
        metrics = tracker.compute_metrics()
        assert metrics is not None
        assert metrics.avg_latency_ms == pytest.approx(100.0)

    def test_empty_returns_none(self):
        tracker = ExecutionQualityTracker()
        assert tracker.compute_metrics() is None

"""Tests for LiveOrderbookFeatureAggregator."""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.adapters.binance.depth_processor import OrderBookLevel, OrderBookSnapshot
from features.live_orderbook_features import LiveOrderbookFeatureAggregator, ORDERBOOK_FEATURE_NAMES


def _make_snapshot(
    symbol: str = "BTCUSDT",
    bid_price: float = 50000.0,
    ask_price: float = 50010.0,
    bid_qty: float = 10.0,
    ask_qty: float = 8.0,
    n_levels: int = 5,
    ts_ms: int = 1000000,
) -> OrderBookSnapshot:
    bids = tuple(
        OrderBookLevel(Decimal(str(bid_price - i * 10)), Decimal(str(bid_qty)))
        for i in range(n_levels)
    )
    asks = tuple(
        OrderBookLevel(Decimal(str(ask_price + i * 10)), Decimal(str(ask_qty)))
        for i in range(n_levels)
    )
    return OrderBookSnapshot(
        symbol=symbol, bids=bids, asks=asks,
        ts_ms=ts_ms, last_update_id=1,
    )


class TestLiveOrderbookFeatureAggregator:
    def test_empty_flush(self):
        agg = LiveOrderbookFeatureAggregator()
        feats = agg.flush_bar("BTCUSDT")
        for name in ORDERBOOK_FEATURE_NAMES:
            assert feats[name] is None

    def test_single_snapshot_insufficient(self):
        agg = LiveOrderbookFeatureAggregator()
        agg.on_depth(_make_snapshot())
        feats = agg.flush_bar("BTCUSDT")
        # Need at least 2 snapshots
        for name in ORDERBOOK_FEATURE_NAMES:
            assert feats[name] is None

    def test_imbalance_mean_balanced(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(10):
            agg.on_depth(_make_snapshot(bid_qty=10.0, ask_qty=10.0))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_imbalance_mean"] == pytest.approx(0.0)

    def test_imbalance_mean_bid_heavy(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(10):
            agg.on_depth(_make_snapshot(bid_qty=20.0, ask_qty=5.0))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_imbalance_mean"] > 0.5

    def test_imbalance_mean_ask_heavy(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(10):
            agg.on_depth(_make_snapshot(bid_qty=5.0, ask_qty=20.0))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_imbalance_mean"] < -0.5

    def test_imbalance_slope(self):
        agg = LiveOrderbookFeatureAggregator()
        # Transition from ask-heavy to bid-heavy
        for i in range(10):
            bid_qty = 5.0 + i * 2
            ask_qty = 20.0 - i * 2
            agg.on_depth(_make_snapshot(bid_qty=bid_qty, ask_qty=max(ask_qty, 1.0)))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_imbalance_slope"] is not None
        assert feats["ob_imbalance_slope"] > 0  # increasing imbalance

    def test_spread_stats(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(5):
            agg.on_depth(_make_snapshot(
                bid_price=50000.0,
                ask_price=50000.0 + 10 + i * 5,
            ))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_spread_mean_bps"] is not None
        assert feats["ob_spread_max_bps"] is not None
        assert feats["ob_spread_max_bps"] >= feats["ob_spread_mean_bps"]

    def test_depth_ratio(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(5):
            agg.on_depth(_make_snapshot(bid_qty=10.0, ask_qty=5.0))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_depth_ratio_mean"] is not None
        assert feats["ob_depth_ratio_mean"] > 1.0  # more bid depth

    def test_pressure_score(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(5):
            agg.on_depth(_make_snapshot(bid_qty=20.0, ask_qty=5.0))
        feats = agg.flush_bar("BTCUSDT")
        assert feats["ob_pressure_score"] is not None
        assert feats["ob_pressure_score"] > 0  # bid-heavy + tight spread

    def test_flush_resets(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(5):
            agg.on_depth(_make_snapshot())
        feats1 = agg.flush_bar("BTCUSDT")
        feats2 = agg.flush_bar("BTCUSDT")
        # Second flush should be empty
        for name in ORDERBOOK_FEATURE_NAMES:
            assert feats2[name] is None

    def test_multi_symbol(self):
        agg = LiveOrderbookFeatureAggregator()
        for i in range(5):
            agg.on_depth(_make_snapshot(symbol="BTCUSDT"))
            agg.on_depth(_make_snapshot(symbol="ETHUSDT", bid_price=3000.0, ask_price=3005.0))
        btc = agg.flush_bar("BTCUSDT")
        eth = agg.flush_bar("ETHUSDT")
        assert btc["ob_imbalance_mean"] is not None
        assert eth["ob_imbalance_mean"] is not None

    def test_feature_names_complete(self):
        assert len(ORDERBOOK_FEATURE_NAMES) == 6

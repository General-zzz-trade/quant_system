"""Tests for microstructure feature extractors: VPIN, Kyle's Lambda, orderbook."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from features.microstructure.vpin import VPINCalculator
from features.microstructure.kyle_lambda import KyleLambdaEstimator
from features.microstructure.orderbook import (
    OrderbookFeatureExtractor,
    OrderbookSnapshot,
)


# ── Helpers ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Tick:
    ts: datetime
    symbol: str
    price: Decimal
    qty: Decimal
    side: str
    trade_id: str = ""


def _make_tick(
    price: str,
    qty: str = "10",
    side: str = "buy",
    ts_offset: int = 0,
) -> _Tick:
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=ts_offset)
    return _Tick(
        ts=ts,
        symbol="BTCUSDT",
        price=Decimal(price),
        qty=Decimal(qty),
        side=side,
    )


def _make_snapshot(
    bids: list[tuple[str, str]],
    asks: list[tuple[str, str]],
) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        ts=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        symbol="BTCUSDT",
        bids=tuple((Decimal(p), Decimal(q)) for p, q in bids),
        asks=tuple((Decimal(p), Decimal(q)) for p, q in asks),
    )


# ── VPIN Tests ──────────────────────────────────────────────


class TestVPINCalculator:
    def test_empty_ticks_returns_zero(self) -> None:
        calc = VPINCalculator(bucket_volume=Decimal("10"), n_buckets=5)
        result = calc.calculate([])
        assert result.vpin == 0.0
        assert result.bucket_count == 0

    def test_balanced_flow_low_vpin(self) -> None:
        """Equal buy and sell volume should produce low VPIN."""
        calc = VPINCalculator(bucket_volume=Decimal("10"), n_buckets=5)
        ticks = []
        for i in range(100):
            side = "buy" if i % 2 == 0 else "sell"
            ticks.append(_make_tick("40000", "5", side=side, ts_offset=i))
        result = calc.calculate(ticks)
        assert result.vpin < 0.1
        assert result.bucket_count >= 5

    def test_imbalanced_flow_high_vpin(self) -> None:
        """All buys should produce VPIN close to 1.0."""
        calc = VPINCalculator(bucket_volume=Decimal("10"), n_buckets=5)
        ticks = [_make_tick("40000", "10", side="buy", ts_offset=i) for i in range(60)]
        result = calc.calculate(ticks)
        assert result.vpin > 0.9
        assert result.buy_volume > result.sell_volume

    def test_single_bucket(self) -> None:
        calc = VPINCalculator(bucket_volume=Decimal("20"), n_buckets=1)
        ticks = [
            _make_tick("100", "10", side="buy"),
            _make_tick("101", "10", side="sell"),
        ]
        result = calc.calculate(ticks)
        assert result.bucket_count == 1
        assert result.vpin == pytest.approx(0.0, abs=0.01)

    def test_classify_tick_uses_side_field(self) -> None:
        calc = VPINCalculator()
        tick = _make_tick("100", "1", side="sell")
        assert calc.classify_tick(tick) == "sell"

    def test_classify_tick_uses_tick_rule(self) -> None:
        calc = VPINCalculator()
        tick = _make_tick("101", "1", side="")
        assert calc.classify_tick(tick, prev_price=Decimal("100")) == "buy"
        assert calc.classify_tick(tick, prev_price=Decimal("102")) == "sell"

    def test_partial_bucket_not_counted(self) -> None:
        """Ticks that don't fill a complete bucket should not be counted."""
        calc = VPINCalculator(bucket_volume=Decimal("100"), n_buckets=5)
        ticks = [_make_tick("100", "5", side="buy")]
        result = calc.calculate(ticks)
        assert result.bucket_count == 0


# ── Kyle's Lambda Tests ─────────────────────────────────────


class TestKyleLambdaEstimator:
    def test_empty_ticks(self) -> None:
        est = KyleLambdaEstimator(window=50)
        result = est.estimate([])
        assert result.kyle_lambda == 0.0
        assert result.n_observations == 0

    def test_single_tick(self) -> None:
        est = KyleLambdaEstimator(window=50)
        result = est.estimate([_make_tick("100")])
        assert result.n_observations == 0

    def test_known_linear_relationship(self) -> None:
        """When price change = lambda * signed_volume exactly, lambda should be recovered."""
        # Construct ticks where price goes up by 0.01 per unit buy volume
        base_price = 100.0
        kyle_lambda_true = 0.01
        ticks = []
        price = base_price
        for i in range(101):
            side = "buy" if i % 2 == 0 else "sell"
            qty = 10.0
            if i > 0:
                signed = qty if side == "buy" else -qty
                price += kyle_lambda_true * signed
            ticks.append(_make_tick(f"{price:.6f}", f"{qty:.1f}", side=side, ts_offset=i))

        est = KyleLambdaEstimator(window=100)
        result = est.estimate(ticks)

        assert result.n_observations == 100
        assert result.kyle_lambda == pytest.approx(kyle_lambda_true, rel=0.01)
        assert result.r_squared > 0.95

    def test_no_price_impact_zero_lambda(self) -> None:
        """Constant price with varying volume should give lambda near zero."""
        ticks = []
        for i in range(51):
            side = "buy" if i % 2 == 0 else "sell"
            ticks.append(_make_tick("100.00", "5", side=side, ts_offset=i))
        est = KyleLambdaEstimator(window=50)
        result = est.estimate(ticks)
        assert abs(result.kyle_lambda) < 0.001


# ── Orderbook Feature Extractor Tests ───────────────────────


class TestOrderbookFeatureExtractor:
    def test_empty_book(self) -> None:
        ext = OrderbookFeatureExtractor(depth_levels=5)
        snap = OrderbookSnapshot(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            bids=(),
            asks=(),
        )
        feats = ext.extract(snap)
        assert feats.bid_ask_spread == Decimal("0")
        assert feats.mid_price == Decimal("0")

    def test_basic_spread_and_mid(self) -> None:
        snap = _make_snapshot(
            bids=[("40000", "1.0")],
            asks=[("40010", "1.0")],
        )
        ext = OrderbookFeatureExtractor(depth_levels=5)
        feats = ext.extract(snap)
        assert feats.bid_ask_spread == Decimal("10")
        assert feats.mid_price == Decimal("40005")

    def test_balanced_imbalance(self) -> None:
        snap = _make_snapshot(
            bids=[("100", "5"), ("99", "5")],
            asks=[("101", "5"), ("102", "5")],
        )
        ext = OrderbookFeatureExtractor(depth_levels=2)
        feats = ext.extract(snap)
        assert feats.bid_ask_imbalance == pytest.approx(0.0)

    def test_bid_heavy_imbalance(self) -> None:
        snap = _make_snapshot(
            bids=[("100", "10")],
            asks=[("101", "2")],
        )
        ext = OrderbookFeatureExtractor(depth_levels=1)
        feats = ext.extract(snap)
        assert feats.bid_ask_imbalance > 0  # More bid volume

    def test_weighted_mid_skewed(self) -> None:
        """Weighted mid should be closer to the side with less volume."""
        snap = _make_snapshot(
            bids=[("100", "1")],
            asks=[("102", "9")],
        )
        ext = OrderbookFeatureExtractor(depth_levels=1)
        feats = ext.extract(snap)
        # weighted_mid = (100*9 + 102*1) / (1+9) = 1002/10 = 100.2
        assert feats.weighted_mid == Decimal("1002") / Decimal("10")

    def test_extract_series(self) -> None:
        snaps = [
            _make_snapshot(bids=[("100", "1")], asks=[("101", "1")]),
            _make_snapshot(bids=[("200", "2")], asks=[("202", "2")]),
        ]
        ext = OrderbookFeatureExtractor(depth_levels=1)
        series = ext.extract_series(snaps)
        assert len(series["mid_price"]) == 2
        assert series["mid_price"][0] == pytest.approx(100.5)
        assert series["mid_price"][1] == pytest.approx(201.0)

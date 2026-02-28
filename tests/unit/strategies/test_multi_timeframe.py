"""Tests for multi-timeframe strategy framework."""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from strategies.multi_timeframe.aggregator import BarAggregator, AggregatedBar, _align_ts
from strategies.multi_timeframe.ensemble import (
    MultiTimeframeEnsemble,
    TimeframeConfig,
    TimeframeSignal,
    EnsembleSignal,
)


# ── BarAggregator tests ──────────────────────────────────────

class TestBarAggregator:
    def test_5m_aggregation(self):
        agg = BarAggregator(timeframes=["5m"], symbol="BTC")
        completed: list[AggregatedBar] = []
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

        for i in range(10):
            ts = base.replace(minute=i)
            result = agg.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 10.0)
            completed.extend(result.values())

        # After 10 bars of 1m, should have 1 completed 5m bar (0-4 minutes)
        assert len(completed) == 1
        assert completed[0].timeframe == "5m"
        assert completed[0].volume == 50.0  # 5 bars * 10

    def test_15m_aggregation(self):
        agg = BarAggregator(timeframes=["15m"], symbol="BTC")
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        completed_count = 0

        for i in range(45):  # 45 minutes = 3 completed 15m bars
            ts = base.replace(minute=i % 60, hour=i // 60)
            result = agg.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 1.0)
            completed_count += len(result)

        assert completed_count == 2  # First 15m completes at min 15, second at min 30

    def test_1h_aggregation(self):
        agg = BarAggregator(timeframes=["1h"], symbol="BTC")
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        completed_count = 0

        for i in range(120):  # 2 hours of 1m bars
            h = i // 60
            m = i % 60
            ts = base.replace(hour=h, minute=m)
            result = agg.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 1.0)
            completed_count += len(result)

        assert completed_count == 1  # Only first hour completes

    def test_ohlcv_correctness(self):
        agg = BarAggregator(timeframes=["5m"], symbol="BTC")
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

        prices = [(100, 110, 90, 105), (105, 115, 95, 108), (108, 120, 85, 112),
                   (112, 118, 100, 115), (115, 125, 105, 120)]

        for i, (o, h, l, c) in enumerate(prices):
            agg.on_bar(base.replace(minute=i), float(o), float(h), float(l), float(c), 10.0)

        # Trigger next period
        result = agg.on_bar(base.replace(minute=5), 120.0, 125.0, 118.0, 122.0, 10.0)
        assert "5m" in result
        bar = result["5m"]
        assert bar.open == 100.0
        assert bar.high == 125.0
        assert bar.low == 85.0
        assert bar.close == 120.0
        assert bar.volume == 50.0

    def test_multi_timeframe(self):
        agg = BarAggregator(timeframes=["5m", "15m"], symbol="BTC")
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        completed_5m = 0
        completed_15m = 0

        for i in range(30):
            ts = base.replace(minute=i)
            result = agg.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 1.0)
            if "5m" in result:
                completed_5m += 1
            if "15m" in result:
                completed_15m += 1

        assert completed_5m == 5  # 5m bars at min 5, 10, 15, 20, 25
        assert completed_15m == 1  # 15m bar at min 15

    def test_get_last(self):
        agg = BarAggregator(timeframes=["5m"])
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

        assert agg.get_last("5m") is None

        for i in range(6):
            agg.on_bar(base.replace(minute=i), 100.0, 105.0, 95.0, 102.0, 1.0)

        last = agg.get_last("5m")
        assert last is not None
        assert last.timeframe == "5m"


class TestAlignTs:
    def test_5m_alignment(self):
        ts = datetime(2024, 1, 1, 14, 37, tzinfo=timezone.utc)
        aligned = _align_ts(ts, 5)
        assert aligned.minute == 35

    def test_1h_alignment(self):
        ts = datetime(2024, 1, 1, 14, 37, tzinfo=timezone.utc)
        aligned = _align_ts(ts, 60)
        assert aligned.hour == 14
        assert aligned.minute == 0


# ── MultiTimeframeEnsemble tests ─────────────────────────────

def _bullish_signal(bar: AggregatedBar):
    return TimeframeSignal(timeframe=bar.timeframe, direction=1, strength=0.8)


def _bearish_signal(bar: AggregatedBar):
    return TimeframeSignal(timeframe=bar.timeframe, direction=-1, strength=0.6)


def _neutral_signal(bar: AggregatedBar):
    return TimeframeSignal(timeframe=bar.timeframe, direction=0, strength=0.0)


class TestMultiTimeframeEnsemble:
    def test_weighted_vote_bullish(self):
        ensemble = MultiTimeframeEnsemble(
            symbol="BTC",
            configs=[
                TimeframeConfig("5m", weight=0.3, signal_fn=_bullish_signal),
                TimeframeConfig("15m", weight=0.3, signal_fn=_bullish_signal),
                TimeframeConfig("1h", weight=0.4, signal_fn=_bullish_signal),
            ],
        )
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        signal = None

        # Feed enough bars to trigger all timeframes
        for i in range(65):
            ts = base.replace(hour=i // 60, minute=i % 60)
            result = ensemble.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 10.0)
            if result is not None:
                signal = result

        assert signal is not None
        assert signal.direction == 1
        assert signal.strength > 0
        assert signal.confidence > 0

    def test_weighted_vote_mixed(self):
        ensemble = MultiTimeframeEnsemble(
            symbol="BTC",
            configs=[
                TimeframeConfig("5m", weight=0.3, signal_fn=_bullish_signal),
                TimeframeConfig("15m", weight=0.7, signal_fn=_bearish_signal),
            ],
        )
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        signal = None

        for i in range(20):
            ts = base.replace(minute=i)
            result = ensemble.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 10.0)
            if result is not None:
                signal = result

        # 15m has higher weight and bearish → overall should be bearish
        assert signal is not None
        assert signal.direction == -1

    def test_majority_fusion(self):
        ensemble = MultiTimeframeEnsemble(
            symbol="BTC",
            configs=[
                TimeframeConfig("5m", weight=0.33, signal_fn=_bullish_signal),
                TimeframeConfig("15m", weight=0.33, signal_fn=_bullish_signal),
                TimeframeConfig("1h", weight=0.34, signal_fn=_bearish_signal),
            ],
            fusion_method="majority",
        )
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        signal = None

        for i in range(65):
            ts = base.replace(hour=i // 60, minute=i % 60)
            result = ensemble.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 10.0)
            if result is not None:
                signal = result

        assert signal is not None
        assert signal.direction == 1  # 2 bullish vs 1 bearish

    def test_cascade_fusion(self):
        ensemble = MultiTimeframeEnsemble(
            symbol="BTC",
            configs=[
                TimeframeConfig("5m", weight=0.5, signal_fn=_bullish_signal),
                TimeframeConfig("1h", weight=0.5, signal_fn=_bearish_signal),  # regime filter
            ],
            fusion_method="cascade",
        )
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        signal = None

        for i in range(65):
            ts = base.replace(hour=i // 60, minute=i % 60)
            result = ensemble.on_bar(ts, 100.0, 105.0, 95.0, 102.0, 10.0)
            if result is not None:
                signal = result

        # 1h is bearish (regime), 5m is bullish (disagrees) → flat
        assert signal is not None
        assert signal.direction in (0, -1)  # filtered or bearish

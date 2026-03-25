# tests/performance/test_regime_detection_latency.py
"""Benchmark: CompositeRegimeDetector.detect — target <0.5ms."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from strategy.regime.composite import CompositeRegimeDetector
from strategy.regime.param_router import RegimeParamRouter
from strategy.regime.volatility import VolatilityRegimeDetector
from strategy.regime.trend import TrendRegimeDetector

_TS = datetime(2026, 3, 17, tzinfo=timezone.utc)


def _make_features(i: int) -> dict:
    return {
        "parkinson_vol": 0.02 + (i % 20) * 0.001,
        "vol_of_vol": 0.005 + (i % 10) * 0.0005,
        "bb_width_20": 0.04 + (i % 15) * 0.002,
        "close_vs_ma20": 0.01 * ((i % 40) - 20),
        "close_vs_ma50": 0.008 * ((i % 60) - 30),
        "adx_14": 10.0 + (i % 50),
    }


@pytest.mark.benchmark
def test_composite_detect_latency():
    """CompositeRegimeDetector.detect must complete in <0.5ms."""
    det = CompositeRegimeDetector()

    # Warmup: build vol history for percentile calculation
    for i in range(50):
        det.detect(symbol="BTCUSDT", ts=_TS, features=_make_features(i))

    # Measure
    times = []
    for i in range(5000):
        feats = _make_features(50 + i)
        t0 = time.perf_counter()
        det.detect(symbol="BTCUSDT", ts=_TS, features=feats)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    p99_us = sorted(times)[int(0.99 * len(times))] * 1_000_000
    print(f"\nCompositeDetector: avg={avg_us:.1f}μs p99={p99_us:.1f}μs")
    assert avg_us < 500, f"Composite detect avg {avg_us:.1f}μs > 500μs (0.5ms)"


@pytest.mark.benchmark
def test_param_router_latency():
    """RegimeParamRouter.route must be <0.01ms (pure dict lookup)."""
    from strategy.regime.composite import CompositeRegimeLabel
    router = RegimeParamRouter()

    labels = [
        CompositeRegimeLabel(vol="low_vol", trend="strong_up"),
        CompositeRegimeLabel(vol="normal_vol", trend="ranging"),
        CompositeRegimeLabel(vol="high_vol", trend="weak_down"),
        CompositeRegimeLabel(vol="crisis", trend="strong_down"),
    ]

    times = []
    for i in range(10000):
        regime = labels[i % len(labels)]
        t0 = time.perf_counter()
        router.route(regime)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    print(f"\nParamRouter.route: avg={avg_us:.2f}μs")
    assert avg_us < 10, f"ParamRouter avg {avg_us:.1f}μs > 10μs"


@pytest.mark.benchmark
def test_volatility_detector_latency():
    """VolatilityRegimeDetector.detect must be <0.3ms."""
    det = VolatilityRegimeDetector()
    for i in range(50):
        det.detect(symbol="ETHUSDT", ts=_TS, features=_make_features(i))

    times = []
    for i in range(5000):
        t0 = time.perf_counter()
        det.detect(symbol="ETHUSDT", ts=_TS, features=_make_features(50 + i))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    print(f"\nVolDetector: avg={avg_us:.1f}μs")
    assert avg_us < 300, f"Vol detect avg {avg_us:.1f}μs > 300μs"


@pytest.mark.benchmark
def test_trend_detector_latency():
    """TrendRegimeDetector.detect must be <0.1ms."""
    det = TrendRegimeDetector()

    times = []
    for i in range(5000):
        t0 = time.perf_counter()
        det.detect(symbol="ETHUSDT", ts=_TS, features=_make_features(i))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    print(f"\nTrendDetector: avg={avg_us:.2f}μs")
    assert avg_us < 100, f"Trend detect avg {avg_us:.1f}μs > 100μs"

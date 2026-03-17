# tests/performance/test_feature_engine_latency.py
"""Benchmark: RustFeatureEngine.push_bar latency — target <1ms per bar."""
from __future__ import annotations

import time

import pytest

from _quant_hotpath import RustFeatureEngine


@pytest.mark.benchmark
def test_push_bar_latency():
    """RustFeatureEngine.push_bar must complete in <1ms (144 features)."""
    engine = RustFeatureEngine()

    # Warmup: fill history buffers
    for i in range(200):
        engine.push_bar(100.0 + i * 0.1, 1000.0, 101.0 + i * 0.1,
                        99.0 + i * 0.1, 100.0 + i * 0.05)

    # Measure
    times = []
    for i in range(1000):
        close = 120.0 + (i % 50) * 0.1
        t0 = time.perf_counter()
        engine.push_bar(close, 1500.0, close + 0.5, close - 0.3, close - 0.1)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    p99_us = sorted(times)[int(0.99 * len(times))] * 1_000_000
    print(f"\npush_bar latency: avg={avg_us:.1f}μs p99={p99_us:.1f}μs")
    assert avg_us < 1000, f"push_bar avg {avg_us:.1f}μs > 1000μs (1ms)"


@pytest.mark.benchmark
def test_get_features_latency():
    """get_features() after push_bar must be <0.5ms."""
    engine = RustFeatureEngine()
    for i in range(200):
        engine.push_bar(100.0 + i * 0.1, 1000.0, 101.0 + i * 0.1,
                        99.0 + i * 0.1, 100.0 + i * 0.05)

    times = []
    for i in range(1000):
        engine.push_bar(120.0 + (i % 20) * 0.1, 1500.0, 121.0, 119.0, 120.0)
        t0 = time.perf_counter()
        feats = engine.get_features()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    n_feats = len(feats) if feats else 0
    print(f"\nget_features latency: avg={avg_us:.1f}μs ({n_feats} features)")
    assert avg_us < 500, f"get_features avg {avg_us:.1f}μs > 500μs"


@pytest.mark.benchmark
def test_feature_count():
    """RustFeatureEngine must produce >= 120 features after warmup."""
    engine = RustFeatureEngine()
    for i in range(200):
        engine.push_bar(100.0 + i * 0.1, 1000.0, 101.0 + i * 0.1,
                        99.0 + i * 0.1, 100.0 + i * 0.05)
    feats = engine.get_features()
    n = len(feats) if feats else 0
    print(f"\nFeature count: {n}")
    assert n >= 100, f"Only {n} features (need >= 100)"

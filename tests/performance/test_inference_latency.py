# tests/performance/test_inference_latency.py
"""Benchmark: RustInferenceBridge.apply_constraints — target <0.1ms."""
from __future__ import annotations

import time

import pytest

from _quant_hotpath import RustInferenceBridge


@pytest.mark.benchmark
def test_apply_constraints_latency():
    """apply_constraints must complete in <0.1ms per call."""
    bridge = RustInferenceBridge(zscore_window=720, zscore_warmup=180)

    # Warmup: feed enough predictions for z-score to stabilize
    for i in range(300):
        bridge.zscore_normalize("BTCUSDT", 0.01 * (i % 50 - 25), i)

    # Measure apply_constraints
    times = []
    for i in range(5000):
        pred = 0.01 * ((i % 100) - 50)
        hour = 300 + i
        t0 = time.perf_counter()
        bridge.apply_constraints(
            "BTCUSDT", pred, hour,
            deadzone=0.5, min_hold=18, max_hold=96,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    p99_us = sorted(times)[int(0.99 * len(times))] * 1_000_000
    print(f"\napply_constraints: avg={avg_us:.2f}μs p99={p99_us:.2f}μs")
    assert avg_us < 100, f"apply_constraints avg {avg_us:.1f}μs > 100μs (0.1ms)"


@pytest.mark.benchmark
def test_zscore_normalize_latency():
    """zscore_normalize must complete in <0.05ms per call."""
    bridge = RustInferenceBridge(zscore_window=720, zscore_warmup=180)

    # Warmup
    for i in range(200):
        bridge.zscore_normalize("ETHUSDT", 0.01 * (i % 30 - 15), i)

    times = []
    for i in range(5000):
        pred = 0.005 * ((i % 80) - 40)
        t0 = time.perf_counter()
        bridge.zscore_normalize("ETHUSDT", pred, 200 + i)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    print(f"\nzscore_normalize: avg={avg_us:.2f}μs")
    assert avg_us < 50, f"zscore_normalize avg {avg_us:.1f}μs > 50μs"

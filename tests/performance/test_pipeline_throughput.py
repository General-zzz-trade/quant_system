# tests/performance/test_pipeline_throughput.py
"""Benchmark: rust_pipeline_apply throughput — target >10,000 events/sec."""
from __future__ import annotations

import time

import pytest

from _quant_hotpath import (
    RustAccountState,
    RustMarketEvent,
    RustMarketState,
    RustPositionState,
    rust_pipeline_apply,
)

_S = 100_000_000  # Fd8 scale


def _make_state():
    ms = RustMarketState("BTCUSDT", 100 * _S, 100 * _S, 101 * _S, 99 * _S, 1000 * _S)
    ps = RustPositionState("BTCUSDT", 0, 0)
    acct = RustAccountState("USDT", 10000 * _S, 0, 0, 0, 0)
    return ms, ps, acct


def _make_event(i: int):
    price = (100 + i % 10) * _S
    return RustMarketEvent("BTCUSDT", price, price, price + _S, price - _S, 1000 * _S)


@pytest.mark.benchmark
def test_pipeline_throughput_10k():
    """rust_pipeline_apply must handle >10,000 events/sec."""
    ms, ps, acct = _make_state()
    n = 50_000

    start = time.perf_counter()
    for i in range(n):
        result = rust_pipeline_apply(ms, ps, acct, _make_event(i))
        if result is not None:
            ms, ps, acct, _ = result
    elapsed = time.perf_counter() - start

    rate = n / elapsed
    print(f"\nPipeline throughput: {rate:,.0f} events/sec ({elapsed:.3f}s for {n:,})")
    assert rate > 10_000, f"Pipeline too slow: {rate:.0f} events/sec (need >10,000)"


@pytest.mark.benchmark
def test_pipeline_single_event_latency():
    """Single rust_pipeline_apply call must be <0.1ms."""
    ms, ps, acct = _make_state()
    evt = _make_event(0)

    # Warm up
    for _ in range(100):
        result = rust_pipeline_apply(ms, ps, acct, evt)
        if result is not None:
            ms, ps, acct, _ = result

    # Measure
    times = []
    for i in range(1000):
        e = _make_event(i)
        t0 = time.perf_counter()
        result = rust_pipeline_apply(ms, ps, acct, e)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if result is not None:
            ms, ps, acct, _ = result

    avg_us = sum(times) / len(times) * 1_000_000
    p99_us = sorted(times)[int(0.99 * len(times))] * 1_000_000
    print(f"\nPipeline latency: avg={avg_us:.1f}μs p99={p99_us:.1f}μs")
    assert avg_us < 100, f"Pipeline avg latency {avg_us:.1f}μs > 100μs"

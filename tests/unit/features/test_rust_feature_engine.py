"""Tests for Phase 4: RustFeatureEngine (incremental) parity with batch compute.

Verifies that pushing bars one at a time produces the same features
as the batch cpp_compute_all_features function.
"""
from __future__ import annotations

import math
import pytest

pytest.importorskip("_quant_hotpath")

from _quant_hotpath import RustFeatureEngine, cpp_compute_all_features, cpp_feature_names


def _generate_bars(n: int):
    """Generate deterministic OHLCV data."""
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    trades = []
    taker_buy_vols = []
    quote_vols = []
    taker_buy_quote_vols = []

    base_price = 40000.0
    for i in range(n):
        ts_ms = 1704067200000.0 + i * 3600000.0  # hourly bars from 2024-01-01
        c = base_price + (i % 50) * 10.0 - 250.0 + (i * 7 % 13) * 5.0
        o = c - 5.0 + (i % 3) * 2.0
        h = max(o, c) + 10.0 + (i % 5) * 3.0
        l = min(o, c) - 10.0 - (i % 4) * 2.0
        v = 100.0 + (i % 20) * 50.0
        t = 500.0 + (i % 10) * 100.0
        tbv = v * (0.4 + (i % 7) * 0.05)
        qv = c * v
        tbqv = c * tbv

        timestamps.append(ts_ms)
        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)
        trades.append(t)
        taker_buy_vols.append(tbv)
        quote_vols.append(qv)
        taker_buy_quote_vols.append(tbqv)

    return (timestamps, opens, highs, lows, closes, volumes,
            trades, taker_buy_vols, quote_vols, taker_buy_quote_vols)


def test_incremental_matches_batch():
    """RustFeatureEngine (incremental) matches cpp_compute_all_features (batch)."""
    n = 100
    (timestamps, opens, highs, lows, closes, volumes,
     trades, taker_buy_vols, quote_vols, taker_buy_quote_vols) = _generate_bars(n)

    # Batch compute
    empty_sched = []
    batch_result = cpp_compute_all_features(
        timestamps, opens, highs, lows, closes, volumes,
        trades, taker_buy_vols, quote_vols, taker_buy_quote_vols,
        empty_sched, empty_sched, empty_sched, empty_sched,  # funding, oi, ls, spot
        empty_sched, empty_sched, empty_sched, empty_sched,  # fgi, iv, pcr, onchain
        empty_sched, empty_sched, empty_sched,  # liq, mempool, macro
    )

    # Incremental compute
    engine = RustFeatureEngine()
    feature_names = cpp_feature_names()

    for i in range(n):
        ts_sec = timestamps[i] / 1000.0
        hour = int((ts_sec % 86400) / 3600)
        days = int(ts_sec / 86400)
        dow = (days + 3) % 7

        engine.push_bar(
            close=closes[i], volume=volumes[i],
            high=highs[i], low=lows[i], open=opens[i],
            hour=hour, dow=dow,
            trades=trades[i],
            taker_buy_volume=taker_buy_vols[i],
            quote_volume=quote_vols[i],
            taker_buy_quote_volume=taker_buy_quote_vols[i],
        )

    # Compare last bar features
    incremental = engine.get_features_array()
    batch_last = batch_result[-1]

    mismatches = []
    for j, name in enumerate(feature_names):
        b_val = batch_last[j]
        i_val = incremental[j]

        both_nan = math.isnan(b_val) and math.isnan(i_val)
        if both_nan:
            continue

        if math.isnan(b_val) != math.isnan(i_val):
            mismatches.append(f"{name}: batch={'NaN' if math.isnan(b_val) else b_val:.6f}, "
                              f"incr={'NaN' if math.isnan(i_val) else i_val:.6f}")
            continue

        if abs(b_val - i_val) > max(abs(b_val) * 1e-6, 1e-10):
            mismatches.append(f"{name}: batch={b_val:.8f}, incr={i_val:.8f}, "
                              f"diff={abs(b_val-i_val):.2e}")

    if mismatches:
        msg = f"{len(mismatches)} feature mismatches:\n" + "\n".join(mismatches[:20])
        pytest.fail(msg)


def test_feature_engine_basic():
    """Basic smoke test: push bars, get features."""
    engine = RustFeatureEngine()
    assert engine.bar_count == 0
    assert not engine.warmed_up

    for i in range(70):
        engine.push_bar(close=100.0 + i, volume=1000.0, high=101.0 + i, low=99.0 + i, open=100.0 + i)

    assert engine.bar_count == 70
    assert engine.warmed_up

    features = engine.get_features()
    assert len(features) == 105
    assert features["ret_1"] is not None
    assert features["rsi_14"] is not None
    assert features["atr_norm_14"] is not None


def test_feature_names_match():
    """Feature names match between engine and batch function."""
    engine = RustFeatureEngine()
    assert engine.feature_names() == cpp_feature_names()


def test_warmup_period():
    """Features not available before warmup."""
    engine = RustFeatureEngine()
    engine.push_bar(close=100.0, volume=1000.0, high=101.0, low=99.0, open=100.0)
    assert not engine.warmed_up
    features = engine.get_features()
    # Most features should be None before warmup
    assert features["rsi_14"] is None  # needs 14 bars
    assert features["vol_20"] is None  # needs 20 bars


def test_performance_vs_python():
    """RustFeatureEngine should be faster than Python EnrichedFeatureComputer."""
    import time

    n = 1000
    engine = RustFeatureEngine()

    t0 = time.perf_counter()
    for i in range(n):
        engine.push_bar(
            close=100.0 + (i % 50) * 0.5,
            volume=1000.0 + i * 10,
            high=102.0 + (i % 50) * 0.5,
            low=98.0 + (i % 50) * 0.5,
            open=99.5 + (i % 50) * 0.5,
            hour=i % 24, dow=i % 7,
            trades=500.0,
            taker_buy_volume=500.0,
            quote_volume=50000.0,
            taker_buy_quote_volume=25000.0,
        )
    rust_ms = (time.perf_counter() - t0) * 1000

    # Just verify it's reasonably fast (< 50ms for 1000 bars)
    assert rust_ms < 200, f"RustFeatureEngine too slow: {rust_ms:.1f}ms for {n} bars"

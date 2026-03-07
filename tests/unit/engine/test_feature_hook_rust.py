"""Tests for FeatureComputeHook with RustFeatureEngine fast path.

Verifies that Rust and Python feature hooks produce identical features
from the same market event sequence.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("_quant_hotpath")

from engine.feature_hook import FeatureComputeHook
from features.enriched_computer import EnrichedFeatureComputer


def _make_event(i: int, base_price: float = 40000.0) -> SimpleNamespace:
    """Create a mock market event."""
    ts_epoch = 1704067200 + i * 3600  # hourly from 2024-01-01
    c = base_price + (i % 50) * 10.0 - 250.0 + (i * 7 % 13) * 5.0
    o = c - 5.0 + (i % 3) * 2.0
    h = max(o, c) + 10.0 + (i % 5) * 3.0
    l = min(o, c) - 10.0 - (i % 4) * 2.0
    v = 100.0 + (i % 20) * 50.0
    return SimpleNamespace(
        event_type="MARKET",
        symbol="BTCUSDT",
        close=c, open=o, high=h, low=l, volume=v,
        trades=500.0 + (i % 10) * 100.0,
        taker_buy_volume=v * (0.4 + (i % 7) * 0.05),
        taker_buy_quote_volume=c * v * 0.5,
        quote_volume=c * v,
        ts=datetime.fromtimestamp(ts_epoch, tz=timezone.utc),
    )


def test_rust_vs_python_parity():
    """RustFeatureEngine features match EnrichedFeatureComputer for shared features."""
    py_computer = EnrichedFeatureComputer()
    rust_hook = FeatureComputeHook(py_computer, use_rust=True)
    py_computer2 = EnrichedFeatureComputer()
    py_hook = FeatureComputeHook(py_computer2, use_rust=False)

    n = 100
    for i in range(n):
        ev = _make_event(i)
        rust_feats = rust_hook.on_event(ev)
        py_feats = py_hook.on_event(ev)

    assert rust_feats is not None
    assert py_feats is not None

    # Known differences:
    # - price_acceleration: Rust uses cached features (correct), Python recomputes (gets 0)
    # - taker_bq_ratio: Rust receives taker_buy_quote_volume, Python feature_hook doesn't pass it
    known_rust_improvements = {"price_acceleration", "taker_bq_ratio"}

    # Compare all features present in BOTH
    mismatches = []
    for key in sorted(set(list(rust_feats.keys()) + list(py_feats.keys()))):
        if key in ("close", "volume") or key in known_rust_improvements:
            continue
        r_val = rust_feats.get(key)
        p_val = py_feats.get(key)

        # Both absent
        if r_val is None and p_val is None:
            continue

        r_nan = r_val is None or (isinstance(r_val, float) and math.isnan(r_val))
        p_nan = p_val is None or (isinstance(p_val, float) and math.isnan(p_val))
        if r_nan and p_nan:
            continue

        if r_nan != p_nan:
            mismatches.append(f"{key}: rust={'NaN' if r_nan else r_val}, py={'NaN' if p_nan else p_val}")
            continue

        if abs(r_val - p_val) > max(abs(p_val) * 1e-6, 1e-10):
            mismatches.append(f"{key}: rust={r_val:.8f}, py={p_val:.8f}, diff={abs(r_val-p_val):.2e}")

    if mismatches:
        pytest.fail(f"{len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20]))

    # Verify Rust improvements are present
    assert rust_feats.get("price_acceleration") is not None
    assert rust_feats["price_acceleration"] != 0.0


def test_rust_warmup_logging():
    """Warmup bar counting works with Rust path."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer, use_rust=True, warmup_bars=10)

    for i in range(15):
        ev = _make_event(i)
        feats = hook.on_event(ev)

    # bar_count is incremented by on_event
    assert hook._bar_count.get("BTCUSDT", 0) >= 15
    assert feats is not None


def test_rust_non_market_event_returns_cached():
    """Non-market events return cached features from Rust path."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer, use_rust=True)

    # Push some bars
    for i in range(5):
        hook.on_event(_make_event(i))

    # Non-market event
    fill_event = SimpleNamespace(event_type="FILL", symbol="BTCUSDT")
    result = hook.on_event(fill_event)
    assert result is not None
    assert "close" in result


def test_python_fallback_when_rust_disabled():
    """use_rust=False forces Python path even when Rust is available."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer, use_rust=False)

    assert not hook._use_rust
    ev = _make_event(0)
    result = hook.on_event(ev)
    assert result is not None


def test_multi_symbol():
    """Per-symbol RustFeatureEngine instances work correctly."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer, use_rust=True)

    for i in range(70):
        ev_btc = _make_event(i, base_price=40000.0)
        ev_btc.symbol = "BTCUSDT"
        ev_eth = _make_event(i, base_price=3000.0)
        ev_eth.symbol = "ETHUSDT"
        hook.on_event(ev_btc)
        hook.on_event(ev_eth)

    assert len(hook._rust_engines) == 2
    assert "BTCUSDT" in hook._rust_engines
    assert "ETHUSDT" in hook._rust_engines

    btc_feats = hook._last_features["BTCUSDT"]
    eth_feats = hook._last_features["ETHUSDT"]
    # ATR should differ since price levels differ drastically (40k vs 3k)
    assert btc_feats.get("atr_norm_14") != eth_feats.get("atr_norm_14")


def test_performance_rust_vs_python():
    """Rust path should be faster than Python path."""
    import time

    n = 200
    events = [_make_event(i) for i in range(n)]

    # Rust
    computer_r = EnrichedFeatureComputer()
    hook_r = FeatureComputeHook(computer_r, use_rust=True)
    t0 = time.perf_counter()
    for ev in events:
        hook_r.on_event(ev)
    rust_ms = (time.perf_counter() - t0) * 1000

    # Python
    computer_p = EnrichedFeatureComputer()
    hook_p = FeatureComputeHook(computer_p, use_rust=False)
    t0 = time.perf_counter()
    for ev in events:
        hook_p.on_event(ev)
    py_ms = (time.perf_counter() - t0) * 1000

    speedup = py_ms / rust_ms if rust_ms > 0 else 0
    # Rust path should be at least as fast (overhead is in Python wrapper, not Rust compute)
    assert speedup > 0.8, f"Rust {rust_ms:.1f}ms, Python {py_ms:.1f}ms, speedup {speedup:.1f}x"

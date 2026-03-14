"""Tests for FeatureComputeHook with RustFeatureEngine.

Verifies that RustFeatureEngine produces correct features from market event sequences.
"""
from __future__ import annotations

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


def test_rust_features_complete():
    """RustFeatureEngine produces complete feature set."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer)

    n = 100
    feats = None
    for i in range(n):
        ev = _make_event(i)
        feats = hook.on_event(ev)

    assert feats is not None
    # Core features should be present after warmup
    assert "close" in feats
    assert "volume" in feats
    assert feats.get("price_acceleration") is not None
    assert feats["price_acceleration"] != 0.0


def test_warmup_logging():
    """Warmup bar counting works."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer, warmup_bars=10)

    for i in range(15):
        ev = _make_event(i)
        feats = hook.on_event(ev)

    assert hook._bar_count.get("BTCUSDT", 0) >= 15
    assert feats is not None


def test_non_market_event_returns_cached():
    """Non-market events return cached features."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer)

    for i in range(5):
        hook.on_event(_make_event(i))

    fill_event = SimpleNamespace(event_type="FILL", symbol="BTCUSDT")
    result = hook.on_event(fill_event)
    assert result is not None
    assert "close" in result


def test_multi_symbol():
    """Per-symbol RustFeatureEngine instances work correctly."""
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer)

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
    assert btc_feats.get("atr_norm_14") != eth_feats.get("atr_norm_14")


@pytest.mark.benchmark
def test_performance():
    """Rust path performance benchmark."""
    import time

    n = 200
    events = [_make_event(i) for i in range(n)]

    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer)
    t0 = time.perf_counter()
    for ev in events:
        hook.on_event(ev)
    rust_ms = (time.perf_counter() - t0) * 1000

    # Should process 200 bars in under 500ms
    assert rust_ms < 500, f"Rust path too slow: {rust_ms:.1f}ms for {n} bars"

"""Test RustFeatureEngine checkpoint/restore — verifies rolling windows survive serialization."""
from __future__ import annotations

import json
import sys

import pytest

sys.path.insert(0, "/quant_system")

_hp = pytest.importorskip("_quant_hotpath")
RustFeatureEngine = _hp.RustFeatureEngine


def _push_bars(engine, n: int, start_price: float = 100.0):
    """Push n synthetic bars into an engine."""
    for i in range(n):
        price = start_price + i * 0.1
        engine.push_bar(
            close=price, volume=1000.0 + i, high=price + 0.5,
            low=price - 0.5, open=price - 0.1,
            hour=i % 24, dow=i % 7,
            funding_rate=0.0001, trades=100.0,
            taker_buy_volume=500.0, quote_volume=50000.0,
            taker_buy_quote_volume=25000.0,
        )


def test_checkpoint_restore_features_match():
    """Push 100 bars → checkpoint → restore → push 1 bar → features must match."""
    orig = RustFeatureEngine()
    _push_bars(orig, 100)
    assert orig.bar_count == 100

    cp = orig.checkpoint()
    assert isinstance(cp, str)
    data = json.loads(cp)
    assert data["version"] == 1
    assert len(data["bars"]) == 100

    # Restore into fresh engine
    restored = RustFeatureEngine()
    n_replayed = restored.restore_checkpoint(cp)
    assert n_replayed == 100
    assert restored.bar_count == 100

    # Push one more bar to both
    next_price = 100.0 + 100 * 0.1
    for eng in (orig, restored):
        eng.push_bar(
            close=next_price, volume=1100.0, high=next_price + 0.5,
            low=next_price - 0.5, open=next_price - 0.1,
            hour=4, dow=3,
            funding_rate=0.0001, trades=100.0,
            taker_buy_volume=500.0, quote_volume=50000.0,
            taker_buy_quote_volume=25000.0,
        )

    # Features must match exactly
    orig_f = orig.get_features()
    rest_f = restored.get_features()

    for name in orig_f:
        a = orig_f[name]
        b = rest_f[name]
        if a is None and b is None:
            continue
        assert a is not None and b is not None, f"Feature {name}: one is None"
        assert abs(a - b) < 1e-10, f"Feature {name}: {a} != {b}"


def test_checkpoint_restore_warmed_up():
    """Engine with >= 65 bars should be warmed up after restore."""
    eng = RustFeatureEngine()
    _push_bars(eng, 70)
    assert eng.warmed_up

    cp = eng.checkpoint()
    restored = RustFeatureEngine()
    restored.restore_checkpoint(cp)
    assert restored.warmed_up
    assert restored.bar_count == 70


def test_checkpoint_empty_engine():
    """Checkpoint of empty engine should produce valid JSON with 0 bars."""
    eng = RustFeatureEngine()
    cp = eng.checkpoint()
    data = json.loads(cp)
    assert len(data["bars"]) == 0

    restored = RustFeatureEngine()
    n = restored.restore_checkpoint(cp)
    assert n == 0
    assert restored.bar_count == 0


def test_checkpoint_round_trip_json():
    """Checkpoint JSON should be valid and contain expected keys."""
    eng = RustFeatureEngine()
    _push_bars(eng, 10)
    cp = eng.checkpoint()
    data = json.loads(cp)
    assert "version" in data
    assert "bars" in data
    assert "bar_count" in data
    bar = data["bars"][0]
    # Check compact key format matches TickProcessor
    assert "c" in bar  # close
    assert "v" in bar  # volume
    assert "h" in bar  # high
    assert "l" in bar  # low
    assert "o" in bar  # open

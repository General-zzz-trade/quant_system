# tests/unit/features/test_live_computer.py
"""Tests for LiveFeatureComputer."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from features.live_computer import LiveFeatureComputer


# ── LiveFeatureComputer tests ────────────────────────────────

class TestLiveFeatureComputer:
    def test_single_bar_partial_features(self):
        comp = LiveFeatureComputer(fast_ma=5, slow_ma=10)
        f = comp.on_bar("BTCUSDT", close=40000.0, volume=100.0)
        assert f.symbol == "BTCUSDT"
        assert f.close == 40000.0
        # Not enough bars for MA
        assert f.ma_fast is None
        assert f.ma_slow is None
        assert f.volatility is None

    def test_fast_ma_after_enough_bars(self):
        comp = LiveFeatureComputer(fast_ma=5, slow_ma=10)
        for i in range(5):
            f = comp.on_bar("BTCUSDT", close=100.0 + i)
        assert f.ma_fast is not None
        assert f.ma_fast == pytest.approx((100 + 101 + 102 + 103 + 104) / 5)

    def test_slow_ma_after_enough_bars(self):
        comp = LiveFeatureComputer(fast_ma=3, slow_ma=5)
        for i in range(5):
            f = comp.on_bar("BTCUSDT", close=100.0 + i)
        assert f.ma_slow is not None

    def test_volatility_computed(self):
        comp = LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=10)
        for i in range(15):
            f = comp.on_bar("BTCUSDT", close=100.0 + i * 0.1)
        assert f.volatility is not None
        assert f.volatility >= 0.0

    def test_momentum(self):
        comp = LiveFeatureComputer(fast_ma=3, slow_ma=5)
        # Rising prices → fast_ma > slow_ma → positive momentum
        for i in range(10):
            f = comp.on_bar("BTCUSDT", close=100.0 + i * 2)
        if f.momentum is not None:
            assert f.momentum > 0

    def test_vwap_ratio(self):
        comp = LiveFeatureComputer()
        for i in range(5):
            f = comp.on_bar("BTCUSDT", close=100.0, volume=10.0)
        # VWAP = 100.0, price = 100.0 → ratio = 1.0
        assert f.vwap_ratio == pytest.approx(1.0)

    def test_multi_symbol(self):
        comp = LiveFeatureComputer(fast_ma=3, slow_ma=5)
        for i in range(5):
            comp.on_bar("BTCUSDT", close=40000.0 + i)
            comp.on_bar("ETHUSDT", close=3000.0 + i)
        assert "BTCUSDT" in comp.symbols
        assert "ETHUSDT" in comp.symbols

    def test_get_features_dict(self):
        comp = LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=5)
        for i in range(10):
            comp.on_bar("BTCUSDT", close=100.0 + i)
        d = comp.get_features_dict("BTCUSDT")
        assert "ma_fast" in d
        assert "ma_slow" in d
        assert "vol" in d

    def test_reset(self):
        comp = LiveFeatureComputer()
        comp.on_bar("BTCUSDT", close=100.0)
        comp.reset("BTCUSDT")
        assert "BTCUSDT" not in comp.symbols

    def test_reset_all(self):
        comp = LiveFeatureComputer()
        comp.on_bar("BTCUSDT", close=100.0)
        comp.on_bar("ETHUSDT", close=100.0)
        comp.reset()
        assert len(comp.symbols) == 0

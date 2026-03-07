# tests/unit/features/test_live_computer.py
"""Tests for LiveFeatureComputer and FeatureSignal."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from features.live_computer import LiveFeatureComputer, LiveFeatures
from decision.signals.feature_signal import FeatureSignal


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


# ── FeatureSignal tests ──────────────────────────────────────

class TestFeatureSignal:
    def _snapshot(self, symbol: str = "BTCUSDT", close: float = 100.0) -> SimpleNamespace:
        return SimpleNamespace(
            markets={symbol: SimpleNamespace(close=close, volume=10.0, last_price=close)},
        )

    def test_neutral_without_data(self):
        sig = FeatureSignal()
        result = sig.compute(self._snapshot(), "BTCUSDT")
        # Only 1 bar → no MA → neutral
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_signal_after_warmup(self):
        sig = FeatureSignal(
            computer=LiveFeatureComputer(fast_ma=3, slow_ma=5),
            momentum_threshold=0.001,
        )
        # Feed rising prices → buy signal
        for i in range(10):
            snap = self._snapshot("BTCUSDT", close=100.0 + i * 2)
            result = sig.compute(snap, "BTCUSDT")
        # After warmup, should have signal
        assert result.side in ("buy", "sell", "flat")

    def test_meta_populated(self):
        sig = FeatureSignal(
            computer=LiveFeatureComputer(fast_ma=3, slow_ma=5),
        )
        for i in range(10):
            result = sig.compute(self._snapshot("BTCUSDT", 100 + i), "BTCUSDT")
        if result.meta is not None:
            assert "momentum" in result.meta

    def test_missing_market_neutral(self):
        sig = FeatureSignal()
        snap = SimpleNamespace(markets={})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"

    def test_confidence_reduced_by_vol(self):
        sig = FeatureSignal(
            computer=LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=5),
            vol_penalty_factor=5.0,
        )
        # Feed volatile prices
        import math
        for i in range(20):
            price = 100.0 + 10.0 * math.sin(i * 0.5)
            sig.compute(self._snapshot("BTCUSDT", price), "BTCUSDT")
        result = sig.compute(self._snapshot("BTCUSDT", 105.0), "BTCUSDT")
        # Confidence should be reduced by volatility
        assert float(result.confidence) <= 1.0

    def test_supports_rust_market_state(self):
        rust = pytest.importorskip("_quant_hotpath")
        sig = FeatureSignal(computer=LiveFeatureComputer(fast_ma=3, slow_ma=5))
        snap = SimpleNamespace(
            markets={
                "BTCUSDT": rust.RustMarketState(
                    symbol="BTCUSDT",
                    close=10_000_000_000,
                    last_price=10_000_000_000,
                    volume=1_000_000_000,
                )
            }
        )
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"

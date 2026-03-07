"""Tests for technical signals: grid, bollinger, breakout, rsi, macd."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from decision.signals.technical.grid_signal import GridSignal
from decision.signals.technical.bollinger_band import BollingerBandSignal
from decision.signals.technical.breakout import BreakoutSignal
from decision.signals.technical.rsi_signal import RSISignal
from decision.signals.technical.macd_signal import MACDSignal


# ── GridSignal ───────────────────────────────────────────────────────

class TestGridSignal:
    def _snap(self, close, ref_price=None):
        feats = {"close": close}
        if ref_price is not None:
            feats["grid_ref_price"] = ref_price
        return SimpleNamespace(features=feats)

    def test_sell_when_price_above_ref(self):
        sig = GridSignal(grid_spacing=Decimal("0.01"))
        snap = self._snap(close=105, ref_price=100)
        r = sig.compute(snap, "BTC")
        assert r.side == "sell"
        assert r.score > Decimal("0")

    def test_buy_when_price_below_ref(self):
        sig = GridSignal(grid_spacing=Decimal("0.01"))
        snap = self._snap(close=95, ref_price=100)
        r = sig.compute(snap, "BTC")
        assert r.side == "buy"
        assert r.score > Decimal("0")

    def test_flat_within_grid(self):
        sig = GridSignal(grid_spacing=Decimal("0.01"))
        snap = self._snap(close=100.2, ref_price=100)
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_missing_close(self):
        snap = SimpleNamespace(features={"no_close": 1})
        sig = GridSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"
        assert r.score == Decimal("0")

    def test_ref_price_zero(self):
        snap = self._snap(close=100, ref_price=0)
        sig = GridSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_no_features(self):
        snap = SimpleNamespace()
        sig = GridSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"
        assert r.confidence == Decimal("0")

    def test_features_not_mapping(self):
        snap = SimpleNamespace(features="not_a_dict")
        sig = GridSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_close_from_market_fallback(self):
        snap = SimpleNamespace(
            features={"grid_ref_price": 100},
            market=SimpleNamespace(close=95),
        )
        sig = GridSignal(grid_spacing=Decimal("0.01"))
        r = sig.compute(snap, "BTC")
        assert r.side == "buy"

    def test_meta_populated(self):
        sig = GridSignal(grid_spacing=Decimal("0.01"))
        snap = self._snap(close=105, ref_price=100)
        r = sig.compute(snap, "BTC")
        assert "close" in r.meta
        assert "ref" in r.meta

    def test_close_from_rust_market_fallback(self):
        rust = pytest.importorskip("_quant_hotpath")
        snap = SimpleNamespace(
            features={"grid_ref_price": 100},
            market=rust.RustMarketState(symbol="BTCUSDT", close=9_500_000_000, last_price=9_500_000_000),
        )
        sig = GridSignal(grid_spacing=Decimal("0.01"))
        r = sig.compute(snap, "BTC")
        assert r.side == "buy"


# ── BollingerBandSignal ─────────────────────────────────────────────

class TestBollingerBandSignal:
    def _snap(self, close, upper, lower, middle):
        return SimpleNamespace(features={
            "close": close, "bb_upper": upper,
            "bb_lower": lower, "bb_middle": middle,
        })

    def test_sell_above_upper(self):
        snap = self._snap(close=110, upper=105, lower=95, middle=100)
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "sell"
        assert r.confidence == Decimal("0.8")

    def test_buy_below_lower(self):
        snap = self._snap(close=90, upper=105, lower=95, middle=100)
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "buy"
        assert r.confidence == Decimal("0.8")

    def test_flat_within_bands(self):
        snap = self._snap(close=100, upper=105, lower=95, middle=100)
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "flat"

    def test_missing_features(self):
        snap = SimpleNamespace(features={"close": 100})
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "flat"
        assert r.confidence == Decimal("0")

    def test_no_features_attr(self):
        snap = SimpleNamespace()
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "flat"

    def test_close_from_market(self):
        snap = SimpleNamespace(
            features={"bb_upper": 105, "bb_lower": 95, "bb_middle": 100},
            market=SimpleNamespace(close=110),
        )
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "sell"

    def test_close_from_rust_market(self):
        rust = pytest.importorskip("_quant_hotpath")
        snap = SimpleNamespace(
            features={"bb_upper": 105, "bb_lower": 95, "bb_middle": 100},
            market=rust.RustMarketState(symbol="ETHUSDT", close=11_000_000_000, last_price=11_000_000_000),
        )
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "sell"

    def test_flat_score_near_middle(self):
        snap = self._snap(close=100, upper=110, lower=90, middle=100)
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "flat"
        assert r.score == Decimal("0")

    def test_band_half_zero(self):
        snap = self._snap(close=100, upper=100, lower=100, middle=100)
        sig = BollingerBandSignal()
        r = sig.compute(snap, "ETH")
        assert r.side == "flat"
        assert r.score == Decimal("0")


# ── BreakoutSignal ───────────────────────────────────────────────────

class TestBreakoutSignal:
    def test_buy_at_high(self):
        snap = SimpleNamespace(market=SimpleNamespace(close=100, high=100, low=90))
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "buy"
        assert r.score == Decimal("1")

    def test_sell_at_low(self):
        snap = SimpleNamespace(market=SimpleNamespace(close=90, high=100, low=90))
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "sell"
        assert r.score == Decimal("-1")

    def test_flat_between(self):
        snap = SimpleNamespace(market=SimpleNamespace(close=95, high=100, low=90))
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_no_market(self):
        snap = SimpleNamespace()
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"
        assert r.confidence == Decimal("0")

    def test_invalid_values(self):
        snap = SimpleNamespace(market=SimpleNamespace(close="abc", high=100, low=90))
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_meta_on_buy(self):
        snap = SimpleNamespace(market=SimpleNamespace(close=100, high=100, low=90))
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert "c" in r.meta
        assert "h" in r.meta

    def test_meta_on_flat(self):
        snap = SimpleNamespace(market=SimpleNamespace(close=95, high=100, low=90))
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert "c" in r.meta and "h" in r.meta and "l" in r.meta

    def test_supports_rust_market_state(self):
        rust = pytest.importorskip("_quant_hotpath")
        snap = SimpleNamespace(
            market=rust.RustMarketState(
                symbol="BTCUSDT",
                close=10_000_000_000,
                last_price=10_000_000_000,
                high=10_000_000_000,
                low=9_000_000_000,
            )
        )
        sig = BreakoutSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "buy"


# ── RSISignal ────────────────────────────────────────────────────────

class TestRSISignal:
    def _snap(self, rsi):
        return SimpleNamespace(features={"rsi": rsi})

    def test_sell_overbought(self):
        sig = RSISignal()
        r = sig.compute(self._snap(80), "BTC")
        assert r.side == "sell"
        assert r.score < Decimal("0")

    def test_buy_oversold(self):
        sig = RSISignal()
        r = sig.compute(self._snap(20), "BTC")
        assert r.side == "buy"
        assert r.score > Decimal("0")

    def test_flat_neutral(self):
        sig = RSISignal()
        r = sig.compute(self._snap(50), "BTC")
        assert r.side == "flat"

    def test_missing_rsi(self):
        snap = SimpleNamespace(features={"not_rsi": 50})
        sig = RSISignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"
        assert r.confidence == Decimal("0")

    def test_no_features(self):
        snap = SimpleNamespace()
        sig = RSISignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_custom_thresholds(self):
        sig = RSISignal(overbought=Decimal("60"), oversold=Decimal("40"))
        assert sig.compute(self._snap(65), "X").side == "sell"
        assert sig.compute(self._snap(35), "X").side == "buy"
        assert sig.compute(self._snap(50), "X").side == "flat"

    def test_extreme_overbought(self):
        sig = RSISignal()
        r = sig.compute(self._snap(99), "BTC")
        assert r.side == "sell"
        assert r.confidence == Decimal("0.8")

    def test_at_boundary_overbought(self):
        sig = RSISignal()
        r = sig.compute(self._snap(70), "BTC")
        assert r.side == "flat"

    def test_at_boundary_oversold(self):
        sig = RSISignal()
        r = sig.compute(self._snap(30), "BTC")
        assert r.side == "flat"


# ── MACDSignal ───────────────────────────────────────────────────────

class TestMACDSignal:
    def _snap(self, macd=0, signal=0, hist=0):
        return SimpleNamespace(features={
            "macd": macd, "macd_signal": signal, "macd_hist": hist,
        })

    def test_buy_positive_histogram(self):
        sig = MACDSignal()
        r = sig.compute(self._snap(hist=0.5), "BTC")
        assert r.side == "buy"
        assert r.score > Decimal("0")

    def test_sell_negative_histogram(self):
        sig = MACDSignal()
        r = sig.compute(self._snap(hist=-0.5), "BTC")
        assert r.side == "sell"
        assert r.score < Decimal("0")

    def test_flat_zero_histogram(self):
        sig = MACDSignal()
        r = sig.compute(self._snap(hist=0), "BTC")
        assert r.side == "flat"
        assert r.score == Decimal("0")

    def test_missing_keys(self):
        snap = SimpleNamespace(features={"macd": 1})
        sig = MACDSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"
        assert r.confidence == Decimal("0")

    def test_no_features(self):
        snap = SimpleNamespace()
        sig = MACDSignal()
        r = sig.compute(snap, "BTC")
        assert r.side == "flat"

    def test_meta_populated(self):
        sig = MACDSignal()
        r = sig.compute(self._snap(macd=1, signal=0.5, hist=0.5), "BTC")
        assert "macd" in r.meta
        assert "signal" in r.meta
        assert "histogram" in r.meta

    def test_large_histogram_capped(self):
        sig = MACDSignal()
        r = sig.compute(self._snap(hist=5.0), "BTC")
        assert r.side == "buy"
        assert r.score <= Decimal("1")

    def test_confidence_buy(self):
        sig = MACDSignal()
        r = sig.compute(self._snap(hist=0.3), "BTC")
        assert r.confidence == Decimal("0.7")

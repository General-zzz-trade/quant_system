"""Tests for strategy/signals/ — base signals, technical signals, ensemble."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from strategy.signals.base import NullSignal, Signal, SignalResult


# ===========================================================================
# SignalResult
# ===========================================================================

class TestSignalResult:
    def test_frozen(self):
        sr = SignalResult(symbol="BTC", side="flat", score=Decimal("0"))
        with pytest.raises(AttributeError):
            sr.symbol = "ETH"  # type: ignore[misc]

    def test_defaults(self):
        sr = SignalResult(symbol="BTC", side="flat", score=Decimal("0"))
        assert sr.confidence == Decimal("0")

    def test_custom_values(self):
        sr = SignalResult(symbol="BTC", side="buy", score=Decimal("0.8"), confidence=Decimal("0.9"))
        assert sr.side == "buy"
        assert sr.score == Decimal("0.8")
        assert sr.confidence == Decimal("0.9")


# ===========================================================================
# NullSignal
# ===========================================================================

class TestNullSignal:
    def test_name(self):
        ns = NullSignal()
        assert ns.name == "null"

    def test_compute_returns_flat(self):
        ns = NullSignal()
        result = ns.compute(None, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")
        assert result.confidence == Decimal("0")
        assert result.symbol == "BTCUSDT"

    def test_compute_different_symbols(self):
        ns = NullSignal()
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            result = ns.compute(None, sym)
            assert result.symbol == sym
            assert result.side == "flat"

    def test_frozen(self):
        ns = NullSignal()
        with pytest.raises(AttributeError):
            ns.name = "other"  # type: ignore[misc]


# ===========================================================================
# Signal
# ===========================================================================

class TestSignal:
    def test_signal_creation(self):
        s = Signal(symbol="BTC", ts=None, side="long", strength=0.8)
        assert s.symbol == "BTC"
        assert s.side == "long"
        assert s.strength == 0.8

    def test_default_strength(self):
        s = Signal(symbol="BTC", ts=None, side="flat")
        assert s.strength == 1.0

    def test_meta_defaults_to_empty_dict(self):
        s = Signal(symbol="BTC", ts=None, side="long")
        assert s.meta == {}

    def test_meta_set(self):
        s = Signal(symbol="BTC", ts=None, side="long", meta={"foo": 1})
        assert s.meta == {"foo": 1}

    def test_frozen(self):
        s = Signal(symbol="BTC", ts=None, side="long")
        with pytest.raises(AttributeError):
            s.symbol = "ETH"  # type: ignore[misc]


# ===========================================================================
# RSISignal
# ===========================================================================

class TestRSISignal:
    def test_oversold_buy(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        snap = SimpleNamespace(features={"rsi": 20.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score > 0
        assert result.confidence == Decimal("0.8")

    def test_overbought_sell(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        snap = SimpleNamespace(features={"rsi": 80.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score < 0
        assert result.confidence == Decimal("0.8")

    def test_neutral_zone(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        snap = SimpleNamespace(features={"rsi": 50.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_missing_features(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        snap = SimpleNamespace(features={})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_no_features_attr(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        result = sig.compute(object(), "BTCUSDT")
        assert result.side == "flat"

    def test_custom_thresholds(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal(overbought=Decimal("60"), oversold=Decimal("40"))
        snap = SimpleNamespace(features={"rsi": 35.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"

    def test_custom_rsi_key(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal(rsi_key="my_rsi")
        snap = SimpleNamespace(features={"my_rsi": 80.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"

    def test_boundary_exactly_overbought(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        snap = SimpleNamespace(features={"rsi": 70.0})
        result = sig.compute(snap, "BTCUSDT")
        # Exactly at threshold — not above
        assert result.side == "flat"

    def test_boundary_exactly_oversold(self):
        from strategy.signals.technical.rsi_signal import RSISignal
        sig = RSISignal()
        snap = SimpleNamespace(features={"rsi": 30.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"


# ===========================================================================
# MACrossSignal
# ===========================================================================

class TestMACrossSignal:
    def test_golden_cross_buy(self):
        from strategy.signals.technical.ma_cross import MACrossSignal
        sig = MACrossSignal()
        snap = SimpleNamespace(features={"ma_fast": 105.0, "ma_slow": 100.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score == Decimal("1")

    def test_death_cross_sell(self):
        from strategy.signals.technical.ma_cross import MACrossSignal
        sig = MACrossSignal()
        snap = SimpleNamespace(features={"ma_fast": 95.0, "ma_slow": 100.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score == Decimal("-1")

    def test_equal_flat(self):
        from strategy.signals.technical.ma_cross import MACrossSignal
        sig = MACrossSignal()
        snap = SimpleNamespace(features={"ma_fast": 100.0, "ma_slow": 100.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_missing_feature_flat(self):
        from strategy.signals.technical.ma_cross import MACrossSignal
        sig = MACrossSignal()
        snap = SimpleNamespace(features={"ma_fast": 100.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"

    def test_custom_keys(self):
        from strategy.signals.technical.ma_cross import MACrossSignal
        sig = MACrossSignal(fast_key="ema_10", slow_key="ema_50")
        snap = SimpleNamespace(features={"ema_10": 110.0, "ema_50": 100.0})
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"

    def test_meta_contains_values(self):
        from strategy.signals.technical.ma_cross import MACrossSignal
        sig = MACrossSignal()
        snap = SimpleNamespace(features={"ma_fast": 105.0, "ma_slow": 100.0})
        result = sig.compute(snap, "BTCUSDT")
        assert "fast" in result.meta
        assert "slow" in result.meta


# ===========================================================================
# BollingerBandSignal
# ===========================================================================

class TestBollingerBandSignal:
    def test_above_upper_sell(self):
        from strategy.signals.technical.bollinger_band import BollingerBandSignal
        sig = BollingerBandSignal()
        snap = SimpleNamespace(features={
            "close": 110.0, "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0
        }, market=None)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "sell"
        assert result.score == Decimal("-1")

    def test_below_lower_buy(self):
        from strategy.signals.technical.bollinger_band import BollingerBandSignal
        sig = BollingerBandSignal()
        snap = SimpleNamespace(features={
            "close": 90.0, "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0
        }, market=None)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "buy"
        assert result.score == Decimal("1")

    def test_within_bands_flat(self):
        from strategy.signals.technical.bollinger_band import BollingerBandSignal
        sig = BollingerBandSignal()
        snap = SimpleNamespace(features={
            "close": 100.0, "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0
        }, market=None)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"

    def test_within_bands_score_relative_to_middle(self):
        from strategy.signals.technical.bollinger_band import BollingerBandSignal
        sig = BollingerBandSignal()
        # close slightly above middle → small negative score (mean reversion)
        snap = SimpleNamespace(features={
            "close": 102.0, "bb_upper": 105.0, "bb_lower": 95.0, "bb_middle": 100.0
        }, market=None)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score < 0  # mean reversion: above middle → negative

    def test_missing_features_flat(self):
        from strategy.signals.technical.bollinger_band import BollingerBandSignal
        sig = BollingerBandSignal()
        snap = SimpleNamespace(features={}, market=None)
        result = sig.compute(snap, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")


# ===========================================================================
# WeightedEnsembleSignal
# ===========================================================================

class TestWeightedEnsembleSignal:
    def test_ensemble_single_model(self):
        from strategy.signals.ensemble import WeightedEnsembleSignal
        null = NullSignal()
        ens = WeightedEnsembleSignal(models=[(null, Decimal("1"))], name="test_ens")
        result = ens.compute(None, "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_ensemble_combines_scores(self):
        from strategy.signals.ensemble import WeightedEnsembleSignal

        class BuySignal:
            name = "buy_sig"
            def compute(self, snap, sym):
                from decision.types import SignalResult as SR
                return SR(symbol=sym, side="buy", score=Decimal("1"), confidence=Decimal("1"))

        class SellSignal:
            name = "sell_sig"
            def compute(self, snap, sym):
                from decision.types import SignalResult as SR
                return SR(symbol=sym, side="sell", score=Decimal("-1"), confidence=Decimal("1"))

        ens = WeightedEnsembleSignal(
            models=[(BuySignal(), Decimal("0.6")), (SellSignal(), Decimal("0.4"))],
        )
        result = ens.compute(None, "BTCUSDT")
        # 1*0.6 + (-1)*0.4 = 0.2 → side=buy
        assert result.score == Decimal("0.2")
        assert result.side == "buy"

    def test_ensemble_flat_when_scores_cancel(self):
        from strategy.signals.ensemble import WeightedEnsembleSignal

        class BuySignal:
            name = "buy_sig"
            def compute(self, snap, sym):
                from decision.types import SignalResult as SR
                return SR(symbol=sym, side="buy", score=Decimal("1"), confidence=Decimal("1"))

        class SellSignal:
            name = "sell_sig"
            def compute(self, snap, sym):
                from decision.types import SignalResult as SR
                return SR(symbol=sym, side="sell", score=Decimal("-1"), confidence=Decimal("1"))

        ens = WeightedEnsembleSignal(
            models=[(BuySignal(), Decimal("0.5")), (SellSignal(), Decimal("0.5"))],
        )
        result = ens.compute(None, "BTCUSDT")
        assert result.score == Decimal("0")
        assert result.side == "flat"

    def test_ensemble_meta_contains_sub_models(self):
        from strategy.signals.ensemble import WeightedEnsembleSignal
        null = NullSignal()
        ens = WeightedEnsembleSignal(models=[(null, Decimal("1"))])
        result = ens.compute(None, "BTCUSDT")
        assert "null" in result.meta

    def test_ensemble_name_default(self):
        from strategy.signals.ensemble import WeightedEnsembleSignal
        ens = WeightedEnsembleSignal(models=[])
        assert ens.name == "ensemble"

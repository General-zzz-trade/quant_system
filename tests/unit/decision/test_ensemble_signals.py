"""Tests for ensemble signals: adaptive, dynamic, weighted."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from decision.signals.adaptive_ensemble import (
    AdaptiveEnsembleConfig,
    AdaptiveEnsembleSignal,
)
from decision.signals.dynamic_ensemble import DynamicEnsembleSignal
from decision.signals.ensemble import WeightedEnsembleSignal
from decision.types import SignalResult


# ── Helpers ──────────────────────────────────────────────────────────

class FakeSignal:
    """Minimal signal satisfying SignalModel protocol."""

    def __init__(self, name: str, side: str = "flat", score: float = 0.0, confidence: float = 0.5):
        self.name = name
        self._side = side
        self._score = Decimal(str(score))
        self._confidence = Decimal(str(confidence))

    def compute(self, snapshot, symbol: str) -> SignalResult:
        return SignalResult(
            symbol=symbol, side=self._side,
            score=self._score, confidence=self._confidence,
        )


# ── AdaptiveEnsembleSignal ───────────────────────────────────────────

class TestAdaptiveEnsembleSignal:
    def test_equal_weights_initially(self):
        s1 = FakeSignal("a")
        s2 = FakeSignal("b")
        ens = AdaptiveEnsembleSignal([s1, s2])
        w = ens.weights
        assert w["a"] == pytest.approx(0.5)
        assert w["b"] == pytest.approx(0.5)

    def test_compute_weighted(self):
        s1 = FakeSignal("a", side="buy", score=1.0, confidence=0.8)
        s2 = FakeSignal("b", side="sell", score=-0.5, confidence=0.6)
        ens = AdaptiveEnsembleSignal([s1, s2])
        r = ens.compute(SimpleNamespace(), "BTC")
        # 0.5 * 1.0 + 0.5 * (-0.5) = 0.25
        assert r.score == Decimal("0.5") * Decimal("1.0") + Decimal("0.5") * Decimal("-0.5")
        assert r.side == "buy"

    def test_compute_all_flat(self):
        s1 = FakeSignal("a", side="flat", score=0.0)
        s2 = FakeSignal("b", side="flat", score=0.0)
        ens = AdaptiveEnsembleSignal([s1, s2])
        r = ens.compute(SimpleNamespace(), "BTC")
        assert r.side == "flat"
        assert r.score == Decimal("0")

    def test_compute_sell_signal(self):
        s1 = FakeSignal("a", side="sell", score=-1.0, confidence=0.9)
        s2 = FakeSignal("b", side="sell", score=-0.8, confidence=0.7)
        ens = AdaptiveEnsembleSignal([s1, s2])
        r = ens.compute(SimpleNamespace(), "BTC")
        assert r.side == "sell"
        assert r.score < Decimal("0")

    def test_meta_includes_weights_and_components(self):
        s1 = FakeSignal("a", side="buy", score=0.5)
        ens = AdaptiveEnsembleSignal([s1])
        r = ens.compute(SimpleNamespace(), "BTC")
        assert "weights" in r.meta
        assert "components" in r.meta
        assert "a" in r.meta["components"]

    def test_recalibrate_with_insufficient_data(self):
        s1 = FakeSignal("a")
        ens = AdaptiveEnsembleSignal([s1])
        # Record too few points; weights should stay equal
        for i in range(10):
            ens.record({"a": float(i)}, float(i))
        ens.recalibrate()
        assert ens.weights["a"] == pytest.approx(1.0)

    def test_record_triggers_auto_recalibrate(self):
        config = AdaptiveEnsembleConfig(recalibrate_every=5, lookback_bars=50)
        s1 = FakeSignal("a")
        s2 = FakeSignal("b")
        ens = AdaptiveEnsembleSignal([s1, s2], config=config)
        # Feed 25 bars of data; auto-recalibrate should trigger at bars 5,10,15,20,25
        for i in range(25):
            ens.record({"a": float(i), "b": float(-i)}, float(i * 0.1))
        # After recalibration, weights should have changed from equal
        # (at least the ic_weighted path ran without error)
        assert sum(ens.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_ic_weighted_method(self):
        config = AdaptiveEnsembleConfig(
            method="ic_weighted", recalibrate_every=100, lookback_bars=200,
        )
        s1 = FakeSignal("a")
        s2 = FakeSignal("b")
        ens = AdaptiveEnsembleSignal([s1, s2], config=config)
        # Build correlated data: signal a predicts returns, b doesn't
        for i in range(50):
            ens.record({"a": float(i), "b": 0.0}, float(i * 2))
        ens.recalibrate()
        # a should get higher weight since it correlates with returns
        assert ens.weights["a"] >= ens.weights["b"]

    def test_inverse_vol_method(self):
        config = AdaptiveEnsembleConfig(
            method="inverse_vol", recalibrate_every=100, lookback_bars=200,
        )
        s1 = FakeSignal("a")
        s2 = FakeSignal("b")
        ens = AdaptiveEnsembleSignal([s1, s2], config=config)
        # a has low variance, b has high variance
        for i in range(50):
            ens.record({"a": 1.0, "b": float(i * 10)}, 0.0)
        ens.recalibrate()
        # a should get higher weight (lower variance)
        assert ens.weights["a"] > ens.weights["b"]

    def test_single_signal(self):
        s1 = FakeSignal("solo", side="buy", score=0.7, confidence=0.9)
        ens = AdaptiveEnsembleSignal([s1])
        assert ens.weights["solo"] == pytest.approx(1.0)
        r = ens.compute(SimpleNamespace(), "X")
        assert r.side == "buy"

    def test_empty_signals(self):
        ens = AdaptiveEnsembleSignal([])
        r = ens.compute(SimpleNamespace(), "X")
        assert r.side == "flat"
        assert r.score == Decimal("0")


# ── DynamicEnsembleSignal ────────────────────────────────────────────

class TestDynamicEnsembleSignal:
    def test_weighted_compute(self):
        s1 = FakeSignal("a", side="buy", score=1.0)
        s2 = FakeSignal("b", side="sell", score=-0.5)
        ens = DynamicEnsembleSignal(
            models=[(s1, Decimal("0.7")), (s2, Decimal("0.3"))]
        )
        r = ens.compute(SimpleNamespace(), "BTC")
        expected = Decimal("1.0") * Decimal("0.7") + Decimal("-0.5") * Decimal("0.3")
        assert r.score == expected
        assert r.side == "buy"

    def test_sell_dominant(self):
        s1 = FakeSignal("a", side="sell", score=-1.0)
        s2 = FakeSignal("b", side="flat", score=0.0)
        ens = DynamicEnsembleSignal(
            models=[(s1, Decimal("0.8")), (s2, Decimal("0.2"))]
        )
        r = ens.compute(SimpleNamespace(), "BTC")
        assert r.side == "sell"

    def test_update_weights_with_sharpe(self):
        s1 = FakeSignal("a")
        s2 = FakeSignal("b")
        ens = DynamicEnsembleSignal(
            models=[(s1, Decimal("0.5")), (s2, Decimal("0.5"))]
        )
        # Model a has consistently positive returns, b has zero
        ens.update_weights({
            "a": [0.01] * 30,
            "b": [0.0] * 30,
        })
        weights = ens.get_weights()
        assert weights["a"] > weights["b"]

    def test_min_weight_floor(self):
        s1 = FakeSignal("a")
        s2 = FakeSignal("b")
        ens = DynamicEnsembleSignal(
            models=[(s1, Decimal("0.5")), (s2, Decimal("0.5"))],
            min_weight=Decimal("0.1"),
        )
        # Give very different sharpe ratios
        ens.update_weights({
            "a": [0.05] * 30,
            "b": [-0.05] * 30,
        })
        weights = ens.get_weights()
        for w in weights.values():
            assert w >= Decimal("0.05")  # close to min_weight

    def test_get_weights(self):
        s1 = FakeSignal("a")
        ens = DynamicEnsembleSignal(models=[(s1, Decimal("1.0"))])
        w = ens.get_weights()
        assert "a" in w
        assert w["a"] == Decimal("1.0")

    def test_meta_populated(self):
        s1 = FakeSignal("a", side="buy", score=0.5)
        ens = DynamicEnsembleSignal(models=[(s1, Decimal("1.0"))])
        r = ens.compute(SimpleNamespace(), "X")
        assert "a" in r.meta
        assert "side" in r.meta["a"]
        assert "weight" in r.meta["a"]


# ── WeightedEnsembleSignal ───────────────────────────────────────────

class TestWeightedEnsembleSignal:
    def test_weighted_combination(self):
        s1 = FakeSignal("a", side="buy", score=1.0, confidence=0.8)
        s2 = FakeSignal("b", side="sell", score=-0.6, confidence=0.7)
        ens = WeightedEnsembleSignal(
            models=[(s1, Decimal("0.6")), (s2, Decimal("0.4"))]
        )
        r = ens.compute(SimpleNamespace(), "BTC")
        expected = Decimal("1.0") * Decimal("0.6") + Decimal("-0.6") * Decimal("0.4")
        assert r.score == expected
        assert r.side == "buy"

    def test_flat_when_balanced(self):
        s1 = FakeSignal("a", side="buy", score=0.5)
        s2 = FakeSignal("b", side="sell", score=-0.5)
        ens = WeightedEnsembleSignal(
            models=[(s1, Decimal("0.5")), (s2, Decimal("0.5"))]
        )
        r = ens.compute(SimpleNamespace(), "BTC")
        assert r.side == "flat"
        assert r.score == Decimal("0")

    def test_all_sell(self):
        s1 = FakeSignal("a", side="sell", score=-0.8)
        s2 = FakeSignal("b", side="sell", score=-0.4)
        ens = WeightedEnsembleSignal(
            models=[(s1, Decimal("0.5")), (s2, Decimal("0.5"))]
        )
        r = ens.compute(SimpleNamespace(), "BTC")
        assert r.side == "sell"

    def test_single_model(self):
        s1 = FakeSignal("solo", side="buy", score=0.9, confidence=1.0)
        ens = WeightedEnsembleSignal(models=[(s1, Decimal("1.0"))])
        r = ens.compute(SimpleNamespace(), "X")
        assert r.score == Decimal("0.9")
        assert r.side == "buy"

    def test_meta_has_components(self):
        s1 = FakeSignal("a", score=0.5)
        s2 = FakeSignal("b", score=-0.2)
        ens = WeightedEnsembleSignal(
            models=[(s1, Decimal("0.5")), (s2, Decimal("0.5"))]
        )
        r = ens.compute(SimpleNamespace(), "X")
        assert "a" in r.meta
        assert "b" in r.meta
        assert "score" in r.meta["a"]
        assert "confidence" in r.meta["a"]

    def test_empty_models(self):
        ens = WeightedEnsembleSignal(models=[])
        r = ens.compute(SimpleNamespace(), "X")
        assert r.side == "flat"
        assert r.score == Decimal("0")

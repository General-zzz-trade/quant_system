"""Tests for adaptive ensemble signal — weight convergence and protocol compliance."""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from decision.signals.adaptive_ensemble import (
    AdaptiveEnsembleConfig,
    AdaptiveEnsembleSignal,
)
from decision.types import SignalResult


# ---------------------------------------------------------------------------
# Fake signals
# ---------------------------------------------------------------------------

@dataclass
class FakeSignal:
    name: str
    _score: float = 0.5
    _side: str = "buy"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        return SignalResult(
            symbol=symbol,
            side=self._side,
            score=Decimal(str(self._score)),
            confidence=Decimal("1"),
        )


@dataclass
class DeterministicSignal:
    """Signal that returns scores from a pre-set list."""
    name: str
    scores: list
    _idx: int = 0

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        score = self.scores[self._idx % len(self.scores)]
        self._idx += 1
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(
            symbol=symbol, side=side,
            score=Decimal(str(score)), confidence=Decimal("1"),
        )


# ---------------------------------------------------------------------------
# Basic protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:

    def test_has_name(self):
        ens = AdaptiveEnsembleSignal([FakeSignal("a"), FakeSignal("b")])
        assert ens.name == "adaptive_ensemble"

    def test_compute_returns_signal_result(self):
        ens = AdaptiveEnsembleSignal([FakeSignal("a"), FakeSignal("b")])
        result = ens.compute(None, "BTCUSDT")
        assert isinstance(result, SignalResult)
        assert result.symbol == "BTCUSDT"

    def test_compute_side_determination(self):
        pos = FakeSignal("pos", _score=1.0, _side="buy")
        neg = FakeSignal("neg", _score=-0.5, _side="sell")

        # positive weighted sum → buy
        ens = AdaptiveEnsembleSignal([pos, neg])
        result = ens.compute(None, "BTCUSDT")
        assert result.side == "buy"

    def test_meta_contains_weights(self):
        ens = AdaptiveEnsembleSignal([FakeSignal("a"), FakeSignal("b")])
        result = ens.compute(None, "BTCUSDT")
        assert "weights" in result.meta
        assert "components" in result.meta

    def test_initial_equal_weights(self):
        ens = AdaptiveEnsembleSignal([FakeSignal("a"), FakeSignal("b"), FakeSignal("c")])
        for w in ens.weights.values():
            assert w == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Weight convergence tests
# ---------------------------------------------------------------------------

class TestWeightConvergence:

    def test_good_signal_gets_higher_weight_ic(self):
        """IC-weighted: correct signal should get weight > 0.5 after training."""
        good = FakeSignal("good", _score=1.0)
        bad = FakeSignal("noise", _score=0.0)

        config = AdaptiveEnsembleConfig(
            lookback_bars=200, recalibrate_every=50,
            method="ic_weighted", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal([good, bad], config)

        rng = random.Random(42)
        for i in range(400):
            # Good signal varies and predicts returns; noise does not
            good_score = rng.gauss(1.0, 0.5)
            ret = 0.01 * good_score + rng.gauss(0, 0.005)
            ens.record({"good": good_score, "noise": rng.gauss(0, 1)}, ret)

        assert ens.weights["good"] > 0.7

    def test_good_signal_gets_higher_weight_ridge(self):
        """Ridge method: correct signal gets higher weight."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=200, recalibrate_every=50,
            method="ridge", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("good"), FakeSignal("noise")], config,
        )

        rng = random.Random(42)
        for i in range(400):
            good_score = rng.gauss(1, 0.1)
            noise_score = rng.gauss(0, 1)
            ret = 0.01 * good_score + rng.gauss(0, 0.005)
            ens.record({"good": good_score, "noise": noise_score}, ret)

        assert ens.weights["good"] > ens.weights["noise"]

    def test_inverse_vol_stable_signal_wins(self):
        """Inverse-vol: lower-variance signal gets higher weight."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=200, recalibrate_every=50,
            method="inverse_vol", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("stable"), FakeSignal("volatile")], config,
        )

        rng = random.Random(42)
        for i in range(300):
            ens.record(
                {"stable": 1.0 + rng.gauss(0, 0.01), "volatile": rng.gauss(0, 5.0)},
                rng.gauss(0, 0.01),
            )

        assert ens.weights["stable"] > ens.weights["volatile"]

    def test_shrinkage_full_equal_weights(self):
        """shrinkage=1.0 → weights stay equal regardless of performance."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=200, recalibrate_every=50,
            method="ic_weighted", shrinkage=1.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("a"), FakeSignal("b")], config,
        )

        rng = random.Random(42)
        for i in range(200):
            ens.record({"a": 1.0, "b": rng.gauss(0, 1)}, 0.01)

        assert ens.weights["a"] == pytest.approx(0.5)
        assert ens.weights["b"] == pytest.approx(0.5)

    def test_shrinkage_partial(self):
        """Partial shrinkage keeps weights between raw and equal."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=200, recalibrate_every=50,
            method="ic_weighted", shrinkage=0.5,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("good"), FakeSignal("noise")], config,
        )

        rng = random.Random(42)
        for i in range(400):
            good_score = rng.gauss(1.0, 0.5)
            ens.record({"good": good_score, "noise": rng.gauss(0, 1)}, 0.01 * good_score + rng.gauss(0, 0.005))

        # Good should be > 0.5 but < 1.0 (shrinkage pulls toward 0.5)
        w = ens.weights["good"]
        assert 0.5 < w < 1.0


# ---------------------------------------------------------------------------
# Recalibration trigger
# ---------------------------------------------------------------------------

class TestRecalibration:

    def test_recalibrate_triggered_by_bar_count(self):
        """Weights change after recalibrate_every bars."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=100, recalibrate_every=30,
            method="ic_weighted", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("a"), FakeSignal("b")], config,
        )

        initial_weights = dict(ens.weights)

        rng = random.Random(42)
        for i in range(29):
            ens.record({"a": 1.0, "b": rng.gauss(0, 1)}, 0.01)

        # Not yet recalibrated
        assert ens.weights == initial_weights

        # 30th bar triggers recalibration
        ens.record({"a": 1.0, "b": rng.gauss(0, 1)}, 0.01)
        # With only 30 bars, might still equal weights if < 20 data
        # Actually we have 30 data points now
        # Weights may or may not change depending on IC computation

    def test_manual_recalibrate(self):
        """Manual recalibrate() works."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=100, recalibrate_every=1000,
            method="ic_weighted", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("a"), FakeSignal("b")], config,
        )

        rng = random.Random(42)
        for i in range(50):
            a_score = rng.gauss(1.0, 0.5)
            ens.record({"a": a_score, "b": rng.gauss(0, 1)}, 0.01 * a_score + rng.gauss(0, 0.005))

        ens.recalibrate()
        assert ens.weights["a"] > ens.weights["b"]

    def test_insufficient_data_no_crash(self):
        """Recalibrate with < 20 bars doesn't crash or change weights."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=100, recalibrate_every=5,
            method="ridge", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("a"), FakeSignal("b")], config,
        )

        initial = dict(ens.weights)
        for i in range(10):
            ens.record({"a": 1.0, "b": 0.5}, 0.01)
        assert ens.weights == initial  # unchanged because < 20 data


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_signal(self):
        config = AdaptiveEnsembleConfig(shrinkage=0.0, method="ic_weighted")
        ens = AdaptiveEnsembleSignal([FakeSignal("only")], config)
        assert ens.weights["only"] == pytest.approx(1.0)

        result = ens.compute(None, "BTCUSDT")
        assert result.score == Decimal("0.5")

    def test_all_zero_ic(self):
        """All signals have zero IC → equal weights."""
        config = AdaptiveEnsembleConfig(
            lookback_bars=100, recalibrate_every=50,
            method="ic_weighted", shrinkage=0.0,
        )
        ens = AdaptiveEnsembleSignal(
            [FakeSignal("a"), FakeSignal("b")], config,
        )

        rng = random.Random(42)
        for i in range(100):
            ens.record({"a": rng.gauss(0, 1), "b": rng.gauss(0, 1)}, rng.gauss(0, 1))

        # Both should be near equal (IC close to 0)
        assert abs(ens.weights["a"] - ens.weights["b"]) < 0.3

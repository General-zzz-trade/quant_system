"""Tests for DynamicEnsembleSignal."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from decision.signals.dynamic_ensemble import DynamicEnsembleSignal
from decision.signals.base import NullSignal
from decision.types import SignalResult


class FixedSignal:
    """A signal that always returns a fixed score."""

    def __init__(self, name: str, score: float, side: str = "buy") -> None:
        self.name = name
        self._score = Decimal(str(score))
        self._side = side

    def compute(self, snapshot, symbol: str) -> SignalResult:
        return SignalResult(
            symbol=symbol, side=self._side,
            score=self._score, confidence=Decimal("1"),
        )


class TestDynamicEnsembleCompute:
    def test_initial_weights_used(self) -> None:
        m1 = FixedSignal("sig_a", 1.0, "buy")
        m2 = FixedSignal("sig_b", -0.5, "sell")
        ens = DynamicEnsembleSignal(
            models=[(m1, Decimal("0.6")), (m2, Decimal("0.4"))],
        )
        result = ens.compute(SimpleNamespace(), "BTCUSDT")
        # score = 1.0 * 0.6 + (-0.5) * 0.4 = 0.4
        assert result.score == Decimal("0.6") + Decimal("-0.5") * Decimal("0.4")
        assert result.side == "buy"

    def test_flat_when_scores_cancel(self) -> None:
        m1 = FixedSignal("sig_a", 1.0, "buy")
        m2 = FixedSignal("sig_b", -1.0, "sell")
        ens = DynamicEnsembleSignal(
            models=[(m1, Decimal("0.5")), (m2, Decimal("0.5"))],
        )
        result = ens.compute(SimpleNamespace(), "BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")

    def test_meta_contains_model_info(self) -> None:
        m1 = FixedSignal("sig_a", 1.0, "buy")
        ens = DynamicEnsembleSignal(models=[(m1, Decimal("1.0"))])
        result = ens.compute(SimpleNamespace(), "BTCUSDT")
        assert result.meta is not None
        assert "sig_a" in result.meta


class TestWeightUpdate:
    def test_weights_shift_toward_better_model(self) -> None:
        m1 = FixedSignal("sig_a", 1.0, "buy")
        m2 = FixedSignal("sig_b", -0.5, "sell")
        ens = DynamicEnsembleSignal(
            models=[(m1, Decimal("0.5")), (m2, Decimal("0.5"))],
            lookback=20,
        )

        # sig_a has consistently positive returns, sig_b has negative
        ens.update_weights({
            "sig_a": [0.02] * 20,
            "sig_b": [-0.01] * 20,
        })

        weights = ens.get_weights()
        assert weights["sig_a"] > weights["sig_b"]

    def test_min_weight_floor(self) -> None:
        m1 = FixedSignal("sig_a", 1.0, "buy")
        m2 = FixedSignal("sig_b", -0.5, "sell")
        ens = DynamicEnsembleSignal(
            models=[(m1, Decimal("0.5")), (m2, Decimal("0.5"))],
            min_weight=Decimal("0.10"),
            lookback=20,
        )

        # Give sig_a much better returns
        ens.update_weights({
            "sig_a": [0.05] * 20,
            "sig_b": [-0.03] * 20,
        })

        weights = ens.get_weights()
        assert weights["sig_b"] >= Decimal("0.10")

    def test_weights_sum_to_one(self) -> None:
        m1 = FixedSignal("sig_a", 1.0)
        m2 = FixedSignal("sig_b", 0.5)
        m3 = FixedSignal("sig_c", -0.3)
        ens = DynamicEnsembleSignal(
            models=[
                (m1, Decimal("0.33")),
                (m2, Decimal("0.33")),
                (m3, Decimal("0.34")),
            ],
            lookback=20,
        )

        ens.update_weights({
            "sig_a": [0.03] * 20,
            "sig_b": [0.01] * 20,
            "sig_c": [-0.02] * 20,
        })

        weights = ens.get_weights()
        total = sum(weights.values())
        assert abs(total - Decimal("1")) < Decimal("0.01")

    def test_compute_uses_updated_weights(self) -> None:
        m1 = FixedSignal("sig_a", 1.0, "buy")
        m2 = FixedSignal("sig_b", -1.0, "sell")
        ens = DynamicEnsembleSignal(
            models=[(m1, Decimal("0.5")), (m2, Decimal("0.5"))],
            lookback=20,
        )

        # Before update: score = 0 (cancels)
        r1 = ens.compute(SimpleNamespace(), "BTCUSDT")
        assert r1.score == Decimal("0")

        # After update: sig_a dominates
        ens.update_weights({
            "sig_a": [0.05] * 20,
            "sig_b": [-0.03] * 20,
        })

        r2 = ens.compute(SimpleNamespace(), "BTCUSDT")
        # sig_a has higher weight now, so score should be positive
        assert r2.score > Decimal("0")
        assert r2.side == "buy"

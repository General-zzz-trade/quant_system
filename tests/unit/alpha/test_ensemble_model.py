# tests/unit/alpha/test_ensemble_model.py
"""Tests for EnsembleAlphaModel — weighted average of sub-model predictions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from alpha.models.ensemble import EnsembleAlphaModel


@dataclass(frozen=True)
class _FakeSignal:
    symbol: str
    ts: datetime
    side: str
    strength: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


class FakeModel:
    """Stub model that returns a fixed prediction."""

    def __init__(self, name: str, side: str, strength: float):
        self.name = name
        self._side = side
        self._strength = strength

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[_FakeSignal]:
        return _FakeSignal(symbol=symbol, ts=ts, side=self._side, strength=self._strength)


class NoneModel:
    """Stub model that returns None."""

    def __init__(self, name: str = "none"):
        self.name = name

    def predict(self, **kwargs: Any) -> None:
        return None


TS = datetime(2024, 1, 1, tzinfo=timezone.utc)
FEAT: Dict[str, Any] = {"close": 100.0}


class TestEnsembleAlphaModel:

    def test_equal_weight_long(self):
        m1 = FakeModel("lgbm", "long", 0.6)
        m2 = FakeModel("xgb", "long", 0.4)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        assert sig.side == "long"
        assert abs(sig.strength - 0.5) < 1e-9  # (0.6+0.4)/2

    def test_weighted_average(self):
        m1 = FakeModel("lgbm", "long", 0.8)
        m2 = FakeModel("xgb", "long", 0.2)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.7, 0.3])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        # (0.7*0.8 + 0.3*0.2) / 1.0 = 0.62
        assert abs(sig.strength - 0.62) < 1e-9
        assert sig.side == "long"

    def test_mixed_long_short(self):
        m1 = FakeModel("lgbm", "long", 0.6)
        m2 = FakeModel("xgb", "short", 0.2)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        # (0.5*0.6 + 0.5*(-0.2)) / 1.0 = 0.2
        assert abs(sig.strength - 0.2) < 1e-9
        assert sig.side == "long"

    def test_cancellation_to_flat(self):
        m1 = FakeModel("lgbm", "long", 0.5)
        m2 = FakeModel("xgb", "short", 0.5)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        assert sig.side == "flat"
        assert sig.strength == 0.0

    def test_short_dominates(self):
        m1 = FakeModel("lgbm", "long", 0.1)
        m2 = FakeModel("xgb", "short", 0.9)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        # (0.5*0.1 + 0.5*(-0.9)) / 1.0 = -0.4
        assert sig.side == "short"
        assert abs(sig.strength - 0.4) < 1e-9

    def test_none_sub_model_skipped(self):
        m1 = FakeModel("lgbm", "long", 0.6)
        m2 = NoneModel("broken")
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        # Only m1 contributes: 0.6
        assert abs(sig.strength - 0.6) < 1e-9

    def test_all_none_returns_none(self):
        m1 = NoneModel("a")
        m2 = NoneModel("b")
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is None

    def test_empty_models_returns_none(self):
        ens = EnsembleAlphaModel(name="ens", sub_models=[], weights=[])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is None

    def test_flat_sub_model(self):
        m1 = FakeModel("lgbm", "flat", 0.0)
        m2 = FakeModel("xgb", "long", 0.4)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        # (0.5*0 + 0.5*0.4) / 1.0 = 0.2
        assert abs(sig.strength - 0.2) < 1e-9
        assert sig.side == "long"

    def test_strength_capped_at_1(self):
        m1 = FakeModel("lgbm", "long", 1.0)
        m2 = FakeModel("xgb", "long", 1.0)
        ens = EnsembleAlphaModel(name="ens", sub_models=[m1, m2], weights=[0.5, 0.5])
        sig = ens.predict(symbol="BTCUSDT", ts=TS, features=FEAT)
        assert sig is not None
        assert sig.strength <= 1.0

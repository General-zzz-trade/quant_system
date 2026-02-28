"""Tests for alpha/online — IncrementalUpdater and RegimeModelSwitcher."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from alpha.base import Signal
from alpha.online.incremental import IncrementalUpdater
from alpha.online.regime_switch import RegimeModelSwitcher, RegimeWeight


# ── Helpers ─────────────────────────────────────────────────


@dataclass
class MockPartialFitModel:
    name: str = "mock_partial"
    _fit_calls: int = 0

    def partial_fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._fit_calls += 1

    def predict(self, X: Any) -> Any:
        return [0.0] * len(X)


@dataclass
class MockFitModel:
    name: str = "mock_fit"
    _fit_calls: int = 0

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None:
        self._fit_calls += 1

    def predict(self, X: Any) -> Any:
        return [0.0] * len(X)


@dataclass
class StubAlpha:
    name: str = "stub"
    _side: str = "long"
    _strength: float = 0.8

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        return Signal(symbol=symbol, ts=ts, side=self._side, strength=self._strength)


@dataclass
class FailingAlpha:
    name: str = "failing"

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        raise RuntimeError("model error")


# ── IncrementalUpdater ─────────────────────────────────────


class TestIncrementalUpdater:
    def test_buffer_fills_and_triggers_update(self):
        model = MockPartialFitModel()
        updater = IncrementalUpdater(model=model, buffer_size=3)
        assert not updater.add_observation([1.0, 2.0], 0.5)
        assert not updater.add_observation([3.0, 4.0], 1.0)
        assert updater.add_observation([5.0, 6.0], 1.5)  # triggers update
        assert model._fit_calls == 1
        assert updater.update_count == 1
        assert updater._X_buffer == []
        assert updater._y_buffer == []

    def test_force_update_flushes_buffer(self):
        model = MockPartialFitModel()
        updater = IncrementalUpdater(model=model, buffer_size=100)
        updater.add_observation([1.0], 0.5)
        updater.add_observation([2.0], 1.0)
        assert updater.update_count == 0
        updater.force_update()
        assert updater.update_count == 1
        assert model._fit_calls == 1
        assert updater._X_buffer == []

    def test_force_update_empty_buffer_noop(self):
        model = MockPartialFitModel()
        updater = IncrementalUpdater(model=model, buffer_size=10)
        updater.force_update()
        assert updater.update_count == 0
        assert model._fit_calls == 0

    def test_fallback_to_fit_if_no_partial_fit(self):
        model = MockFitModel()
        updater = IncrementalUpdater(model=model, buffer_size=2)
        updater.add_observation([1.0], 0.5)
        updater.add_observation([2.0], 1.0)
        assert model._fit_calls == 1

    def test_multiple_updates(self):
        model = MockPartialFitModel()
        updater = IncrementalUpdater(model=model, buffer_size=2)
        for i in range(6):
            updater.add_observation([float(i)], float(i))
        assert updater.update_count == 3
        assert model._fit_calls == 3

    def test_update_count_property(self):
        model = MockPartialFitModel()
        updater = IncrementalUpdater(model=model, buffer_size=5)
        assert updater.update_count == 0


# ── RegimeModelSwitcher ────────────────────────────────────


class TestRegimeModelSwitcher:
    def test_register_model(self):
        switcher = RegimeModelSwitcher()
        m = StubAlpha(name="alpha1")
        switcher.register_model(m)
        assert "alpha1" in switcher.models

    def test_register_with_regime_weights(self):
        switcher = RegimeModelSwitcher()
        m = StubAlpha(name="alpha1")
        switcher.register_model(m, regime_weights={"bull": 0.8, "bear": 0.2})
        assert switcher.regime_weights["bull"]["alpha1"] == 0.8
        assert switcher.regime_weights["bear"]["alpha1"] == 0.2

    def test_set_regime(self):
        switcher = RegimeModelSwitcher()
        assert switcher.current_regime == "normal"
        switcher.set_regime("bull")
        assert switcher.current_regime == "bull"

    def test_get_active_weights_known_regime(self):
        switcher = RegimeModelSwitcher()
        m = StubAlpha(name="alpha1")
        switcher.register_model(m, regime_weights={"bull": 0.9})
        switcher.set_regime("bull")
        weights = switcher.get_active_weights()
        assert weights["alpha1"] == 0.9

    def test_get_active_weights_unknown_regime_fallback(self):
        switcher = RegimeModelSwitcher()
        m1 = StubAlpha(name="a1")
        m2 = StubAlpha(name="a2")
        switcher.register_model(m1)
        switcher.register_model(m2)
        switcher.set_regime("unknown_regime")
        weights = switcher.get_active_weights()
        # Equal weight fallback
        assert abs(weights["a1"] - 0.5) < 0.01
        assert abs(weights["a2"] - 0.5) < 0.01

    def test_get_active_weights_no_models(self):
        switcher = RegimeModelSwitcher()
        weights = switcher.get_active_weights()
        assert weights == {}

    def test_predict_ensemble_weighted(self):
        switcher = RegimeModelSwitcher()
        m1 = StubAlpha(name="m1", _side="long", _strength=0.8)
        m2 = StubAlpha(name="m2", _side="short", _strength=0.4)
        switcher.register_model(m1, regime_weights={"normal": 0.7})
        switcher.register_model(m2, regime_weights={"normal": 0.3})

        result = switcher.predict_ensemble(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={},
        )
        assert result is not None
        assert "m1" in result
        assert "m2" in result
        assert "_ensemble" in result
        assert result["m1"]["side"] == "long"
        assert result["m2"]["side"] == "short"

    def test_predict_ensemble_no_models(self):
        switcher = RegimeModelSwitcher()
        result = switcher.predict_ensemble(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={},
        )
        assert result is None

    def test_predict_ensemble_model_error_handled(self):
        switcher = RegimeModelSwitcher()
        m1 = StubAlpha(name="good", _side="long", _strength=0.5)
        m2 = FailingAlpha(name="bad")
        switcher.register_model(m1)
        switcher.register_model(m2)

        result = switcher.predict_ensemble(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={},
        )
        assert result is not None
        assert "good" in result
        assert "bad" not in result

    def test_ensemble_side_logic(self):
        switcher = RegimeModelSwitcher()
        # All long => ensemble long
        m1 = StubAlpha(name="m1", _side="long", _strength=0.8)
        switcher.register_model(m1)
        result = switcher.predict_ensemble(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={},
        )
        assert result["_ensemble"]["side"] == "long"

    def test_ensemble_flat_when_mixed(self):
        switcher = RegimeModelSwitcher()
        m1 = StubAlpha(name="m1", _side="long", _strength=0.3)
        m2 = StubAlpha(name="m2", _side="short", _strength=0.3)
        switcher.register_model(m1, regime_weights={"normal": 0.5})
        switcher.register_model(m2, regime_weights={"normal": 0.5})
        result = switcher.predict_ensemble(
            symbol="BTCUSDT",
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            features={},
        )
        # Weighted score: (1*0.3*0.5 + (-1)*0.3*0.5) / 1.0 = 0.0 => flat
        assert result["_ensemble"]["side"] == "flat"

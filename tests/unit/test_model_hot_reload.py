# tests/unit/test_model_hot_reload.py
"""Tests for model hot-reload (SIGHUP) pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from alpha.base import AlphaModel, Signal
from alpha.inference import InferenceEngine
from alpha.inference.bridge import LiveInferenceBridge


class _DummyModel(AlphaModel):
    def __init__(self, name: str = "dummy", strength: float = 0.5):
        self._name = name
        self._strength = strength

    @property
    def name(self) -> str:
        return self._name

    def predict(self, *, symbol: str, ts: datetime, features: dict) -> Signal:
        return Signal(symbol=symbol, ts=ts, side="long", strength=self._strength)


# ── InferenceEngine.set_models ──


def test_set_models_replaces_all():
    engine = InferenceEngine(models=[_DummyModel("a"), _DummyModel("b")])
    assert len(engine._models) == 2

    new = [_DummyModel("c")]
    engine.set_models(new)
    assert len(engine._models) == 1
    assert engine._models[0].name == "c"


def test_set_models_empty():
    engine = InferenceEngine(models=[_DummyModel("a")])
    engine.set_models([])
    assert engine._models == []


# ── LiveInferenceBridge.update_models ──


def test_update_models_clears_state():
    m1 = _DummyModel("m1", 0.8)
    bridge = LiveInferenceBridge(
        models=[m1],
        min_hold_bars={"BTCUSDT": 5},
        long_only_symbols={"BTCUSDT"},
        zscore_warmup=0,
    )
    # Warm up state
    bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
    cp = bridge.checkpoint()
    assert cp.get("position", {}) != {}

    m2 = _DummyModel("m2", 0.9)
    bridge.update_models([m2])

    cp = bridge.checkpoint()
    assert cp.get("position", {}) == {}
    assert cp.get("hold_counter", {}) == {}
    assert bridge._engine._models[0].name == "m2"


# ── _reload_models_pending flag ──


def test_reload_models_pending_flag():
    """Test the flag logic without a full LiveRunner."""
    from dataclasses import dataclass, field

    @dataclass
    class _FakeRunner:
        model_loader: MagicMock = field(default_factory=MagicMock)
        inference_bridge: MagicMock = field(default_factory=MagicMock)
        _reload_models_pending: bool = False
        _config: MagicMock = field(default_factory=MagicMock)

    runner = _FakeRunner()
    runner._config.model_names = ["alpha_btc"]
    runner.model_loader.reload_if_changed.return_value = [_DummyModel("new")]

    # Simulate SIGHUP
    runner._reload_models_pending = True
    assert runner._reload_models_pending is True

    # Simulate main loop reload logic
    if runner._reload_models_pending:
        runner._reload_models_pending = False
        names = tuple(runner._config.model_names)
        new_models = runner.model_loader.reload_if_changed(names)
        if new_models is not None and runner.inference_bridge is not None:
            runner.inference_bridge.update_models(new_models)

    assert runner._reload_models_pending is False
    runner.model_loader.reload_if_changed.assert_called_once_with(("alpha_btc",))
    runner.inference_bridge.update_models.assert_called_once()


def test_reload_models_pending_no_change():
    """When model_loader returns None, bridge.update_models is NOT called."""
    from dataclasses import dataclass, field

    @dataclass
    class _FakeRunner:
        model_loader: MagicMock = field(default_factory=MagicMock)
        inference_bridge: MagicMock = field(default_factory=MagicMock)
        _reload_models_pending: bool = False
        _config: MagicMock = field(default_factory=MagicMock)

    runner = _FakeRunner()
    runner._config.model_names = ["alpha_btc"]
    runner.model_loader.reload_if_changed.return_value = None

    runner._reload_models_pending = True
    if runner._reload_models_pending:
        runner._reload_models_pending = False
        names = tuple(runner._config.model_names)
        new_models = runner.model_loader.reload_if_changed(names)
        if new_models is not None and runner.inference_bridge is not None:
            runner.inference_bridge.update_models(new_models)

    runner.inference_bridge.update_models.assert_not_called()

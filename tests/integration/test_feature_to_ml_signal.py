"""Integration: feature computation -> ML inference -> signal full path."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.feature_hook import FeatureComputeHook
from features.live_computer import LiveFeatureComputer
from alpha.inference.bridge import LiveInferenceBridge
from alpha.base import Signal
from decision.signals.ml.model_runner import ModelRunnerSignal


class MockAlphaModel:
    """Mock model that returns momentum-based signals."""

    name = "mock_momentum"

    def predict(self, *, symbol, ts, features):
        momentum = features.get("momentum")
        if momentum is None:
            return None
        side = "long" if momentum > 0 else ("short" if momentum < 0 else "flat")
        return Signal(
            symbol=symbol,
            ts=ts,
            side=side,
            strength=min(abs(momentum) * 100, 1.0),
        )


def _market(close: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="MARKET",
        symbol="BTCUSDT",
        open=close, high=close + 1, low=close - 1, close=close,
        volume=50.0,
        ts=datetime(2024, 1, 1, 0, idx),
        header=SimpleNamespace(event_id=f"e{idx}", ts=datetime(2024, 1, 1, 0, idx)),
    )


def test_features_flow_to_ml_score():
    """Features computed from market data flow into ML inference and produce ml_score."""
    computer = LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=3)
    model = MockAlphaModel()
    bridge = LiveInferenceBridge(models=[model])
    hook = FeatureComputeHook(computer=computer, inference_bridge=bridge, warmup_bars=0)

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        feature_hook=hook,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    # Pump enough bars for all windows to fill (slow_ma=5)
    for i in range(10):
        coord.emit(_market(100.0 + i, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    assert snap.features is not None
    assert "ml_score" in snap.features
    assert snap.features["ml_score"] > 0  # uptrend → positive score


def test_model_runner_signal_consumes_ml_score():
    """ModelRunnerSignal.compute() reads ml_score from snapshot.features."""
    computer = LiveFeatureComputer(fast_ma=3, slow_ma=5, vol_window=3)
    model = MockAlphaModel()
    bridge = LiveInferenceBridge(models=[model])
    hook = FeatureComputeHook(computer=computer, inference_bridge=bridge, warmup_bars=0)

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        feature_hook=hook,
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    for i in range(10):
        coord.emit(_market(100.0 + i, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]

    signal = ModelRunnerSignal()
    result = signal.compute(snap, "BTCUSDT")
    assert result.side == "buy"  # positive ml_score → buy
    assert result.score > 0


def test_no_features_returns_flat():
    """Without feature hook, ModelRunnerSignal returns flat."""
    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
    )
    coord = EngineCoordinator(cfg=cfg)
    coord.start()

    for i in range(5):
        coord.emit(_market(100.0 + i, i), actor="test")

    snap = coord.get_state_view()["last_snapshot"]
    assert snap.features is None

    signal = ModelRunnerSignal()
    result = signal.compute(snap, "BTCUSDT")
    assert result.side == "flat"

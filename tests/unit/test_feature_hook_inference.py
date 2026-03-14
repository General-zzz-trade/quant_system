# tests/unit/test_feature_hook_inference.py
"""Integration test: FeatureComputeHook + LiveInferenceBridge."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from alpha.base import Signal
from alpha.inference.bridge import LiveInferenceBridge
from engine.feature_hook import FeatureComputeHook
from features.enriched_computer import EnrichedFeatureComputer


class _ConstModel:
    """Always predicts long with given strength."""
    def __init__(self, name: str = "test", strength: float = 0.8):
        self._name = name
        self._strength = strength

    @property
    def name(self) -> str:
        return self._name

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Signal:
        return Signal(symbol=symbol, ts=ts, side="long", strength=self._strength)


@dataclass
class _Event:
    event_type: str = "market_data"
    symbol: str = "BTCUSDT"
    close: float = 50000.0
    volume: float = 100.0
    high: float = 50100.0
    low: float = 49900.0
    open: float = 49950.0
    ts: datetime = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    trades: float = 500.0
    taker_buy_volume: float = 55.0
    quote_volume: float = 5_000_000.0
    taker_buy_quote_volume: float = 2_750_000.0


def test_hook_injects_ml_score():
    computer = EnrichedFeatureComputer()
    bridge = LiveInferenceBridge(models=[_ConstModel(strength=0.9)])
    hook = FeatureComputeHook(computer=computer, inference_bridge=bridge)

    # Feed enough bars to warm up
    for i in range(70):
        evt = _Event(close=50000.0 + i * 10, ts=datetime(2024, 1, 1, i % 24, tzinfo=timezone.utc))
        features = hook.on_event(evt)

    assert features is not None
    assert "ml_score" in features
    # With no min_hold, raw score should pass through
    assert isinstance(features["ml_score"], float)


def test_hook_without_bridge():
    computer = EnrichedFeatureComputer()
    hook = FeatureComputeHook(computer=computer, inference_bridge=None)

    evt = _Event()
    features = hook.on_event(evt)
    assert features is not None
    assert "ml_score" not in features


def test_hook_with_min_hold():
    computer = EnrichedFeatureComputer()
    bridge = LiveInferenceBridge(
        models=[_ConstModel(strength=0.9)],
        min_hold_bars={"BTCUSDT": 5},
        long_only_symbols={"BTCUSDT"},
    )
    hook = FeatureComputeHook(computer=computer, inference_bridge=bridge)

    # Feed enough bars for warmup + min_hold
    for i in range(80):
        evt = _Event(close=50000.0 + i * 10, ts=datetime(2024, 1, 1, i % 24, tzinfo=timezone.utc))
        features = hook.on_event(evt)

    assert features is not None
    assert "ml_score" in features
    # With min_hold and long_only, score should be 0 or 1
    assert features["ml_score"] in (0.0, 1.0)

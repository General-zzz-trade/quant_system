"""Tests for Alpha inference pipeline: InferenceEngine, LiveInferenceBridge, AlphaRegistry."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from alpha.base import AlphaModel, Signal
from alpha.inference import InferenceEngine, InferenceResult
from alpha.inference.bridge import LiveInferenceBridge
from alpha.registry import AlphaRegistry


# ── Stub model ────────────────────────────────────────────────

@dataclass
class StubAlphaModel:
    name: str = "stub_model"
    _side: str = "long"
    _strength: float = 0.75
    _should_fail: bool = False

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        if self._should_fail:
            raise RuntimeError("model error")
        return Signal(symbol=symbol, ts=ts, side=self._side, strength=self._strength)


# ── InferenceEngine ───────────────────────────────────────────

class TestInferenceEngine:
    def test_run_single_model(self):
        model = StubAlphaModel()
        engine = InferenceEngine(models=[model])
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={"sma": 40000})

        assert len(results) == 1
        r = results[0]
        assert r.model_name == "stub_model"
        assert r.symbol == "BTCUSDT"
        assert r.signal is not None
        assert r.signal.side == "long"
        assert r.signal.strength == 0.75
        assert r.error is None
        assert r.latency_ms >= 0

    def test_run_multiple_models(self):
        m1 = StubAlphaModel(name="m1", _side="long", _strength=0.8)
        m2 = StubAlphaModel(name="m2", _side="short", _strength=0.3)
        engine = InferenceEngine(models=[m1, m2])
        results = engine.run(symbol="ETHUSDT", ts=datetime(2024, 1, 1), features={})

        assert len(results) == 2
        assert results[0].model_name == "m1"
        assert results[1].model_name == "m2"
        assert results[0].signal.side == "long"
        assert results[1].signal.side == "short"

    def test_run_handles_model_error(self):
        model = StubAlphaModel(_should_fail=True)
        engine = InferenceEngine(models=[model])
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={})

        assert len(results) == 1
        assert results[0].signal is None
        assert results[0].error is not None
        assert "model error" in results[0].error

    def test_add_model(self):
        engine = InferenceEngine()
        assert len(engine._models) == 0
        engine.add_model(StubAlphaModel(name="added"))
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={})
        assert len(results) == 1
        assert results[0].model_name == "added"

    def test_empty_models(self):
        engine = InferenceEngine(models=[])
        results = engine.run(symbol="BTCUSDT", ts=datetime(2024, 1, 1), features={})
        assert results == []


# ── LiveInferenceBridge ───────────────────────────────────────

class TestLiveInferenceBridge:
    def test_enrich_adds_ml_score(self):
        model = StubAlphaModel(_side="long", _strength=0.9)
        bridge = LiveInferenceBridge(models=[model])
        features: Dict[str, Any] = {"sma": 40000}
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), features)

        assert "ml_score" in result
        assert result["ml_score"] == 0.9

    def test_enrich_short_signal_negative_score(self):
        model = StubAlphaModel(_side="short", _strength=0.6)
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})

        assert result["ml_score"] == -0.6

    def test_enrich_flat_signal_zero_score(self):
        model = StubAlphaModel(_side="flat", _strength=0.5)
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})

        assert result["ml_score"] == 0.0

    def test_enrich_custom_score_key(self):
        model = StubAlphaModel()
        bridge = LiveInferenceBridge(models=[model], score_key="alpha_signal")
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), {})

        assert "alpha_signal" in result
        assert "ml_score" not in result

    def test_enrich_model_error_no_crash(self):
        model = StubAlphaModel(_should_fail=True)
        bridge = LiveInferenceBridge(models=[model])
        features: Dict[str, Any] = {"sma": 40000}
        result = bridge.enrich("BTCUSDT", datetime(2024, 1, 1), features)

        # Should not crash, ml_score not added
        assert "ml_score" not in result
        assert result["sma"] == 40000

    def test_enrich_none_ts_uses_now(self):
        model = StubAlphaModel()
        bridge = LiveInferenceBridge(models=[model])
        result = bridge.enrich("BTCUSDT", None, {})

        assert "ml_score" in result


# ── AlphaRegistry ─────────────────────────────────────────────

class TestAlphaRegistry:
    def test_register_and_get(self):
        registry = AlphaRegistry()
        model = StubAlphaModel(name="test_model")
        registry.register(model)

        retrieved = registry.get("test_model")
        assert retrieved is model

    def test_get_missing_returns_none(self):
        registry = AlphaRegistry()
        assert registry.get("nonexistent") is None

    def test_list_names(self):
        registry = AlphaRegistry()
        registry.register(StubAlphaModel(name="m1"))
        registry.register(StubAlphaModel(name="m2"))

        names = list(registry.list_names())
        assert "m1" in names
        assert "m2" in names

    def test_register_overwrite(self):
        registry = AlphaRegistry()
        m1 = StubAlphaModel(name="dup", _strength=0.1)
        m2 = StubAlphaModel(name="dup", _strength=0.9)
        registry.register(m1)
        registry.register(m2)

        retrieved = registry.get("dup")
        assert retrieved._strength == 0.9

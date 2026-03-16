# tests/unit/decision/test_regime_bridge_composite.py
"""Tests for CompositeRegimeDetector + ParamRouter integration in regime_bridge."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List


from regime.base import RegimeLabel
from regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from decision.regime_policy import RegimePolicy
from decision.regime_bridge import RegimeAwareDecisionModule


# ── Stubs ────────────────────────────────────────────────────

_TS = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _market(close: float) -> SimpleNamespace:
    return SimpleNamespace(close=close, last_price=close)


def _snapshot(
    *,
    markets: Dict[str, Any],
    ts: datetime | None = None,
    features: Dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        markets=markets,
        ts=ts or _TS,
        features=features,
    )


class _MockDecisionModule:
    def __init__(self, intents: List[Any] | None = None):
        self.intents = intents or [SimpleNamespace(symbol="ETHUSDT", side="buy")]
        self.call_count = 0

    def decide(self, snapshot: Any) -> Iterable[Any]:
        self.call_count += 1
        return list(self.intents)


class _StubDetector:
    """Returns a fixed label for testing."""
    def __init__(self, label: RegimeLabel | None):
        self.name = label.name if label else "stub"
        self._label = label

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> RegimeLabel | None:
        return self._label


# ── CompositeRegimeDetector is default ─────────────────────

class TestCompositeDetectorDefault:
    def test_default_detector_is_composite(self):
        mod = RegimeAwareDecisionModule(inner=_MockDecisionModule())
        assert len(mod.detectors) == 1
        assert isinstance(mod.detectors[0], CompositeRegimeDetector)

    def test_composite_detector_called_with_snapshot_features(self):
        """When snapshot has features, they are passed to the detector."""
        features = {
            "parkinson_vol": 0.02,
            "vol_of_vol": 0.005,
            "close_vs_ma20": 0.01,
            "close_vs_ma50": 0.02,
            "adx_14": 30.0,
            "bb_width_20": 0.05,
        }
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(inner=inner)

        # Feed enough bars for the volatility detector's min_bars
        for i in range(35):
            snap = _snapshot(
                markets={"ETHUSDT": _market(100.0 + i * 0.1)},
                features=features,
            )
            list(mod.decide(snap))

        labels = mod.current_labels
        # Should have at least one composite label
        composite_labels = [ll for ll in labels if ll.name == "composite"]
        assert len(composite_labels) >= 1
        assert inner.call_count > 0


# ── Features from snapshot ──────────────────────────────────

class TestSnapshotFeatureExtraction:
    def test_snapshot_features_preferred_over_buffer(self):
        """When snapshot.features has data, it should be used."""
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(inner=inner)

        features = {
            "parkinson_vol": 0.03,
            "vol_of_vol": 0.01,
            "close_vs_ma20": 0.02,
            "close_vs_ma50": 0.03,
        }
        snap = _snapshot(
            markets={"ETHUSDT": _market(100.0)},
            features=features,
        )
        # Even on first call, features from snapshot should be used
        list(mod.decide(snap))
        # Inner gets called since no labels block (insufficient vol history)
        assert inner.call_count == 1

    def test_fallback_to_buffer_when_no_features(self):
        """When snapshot.features is None, fall back to buffer computation."""
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(inner=inner, buffer_maxlen=200)

        snap = _snapshot(
            markets={"ETHUSDT": _market(100.0)},
            features=None,
        )
        list(mod.decide(snap))
        # Should still work (delegates when no labels)
        assert inner.call_count == 1
        # Buffer should be populated
        assert "ETHUSDT" in mod._buffers

    def test_empty_features_dict_falls_back(self):
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(inner=inner)
        snap = _snapshot(
            markets={"ETHUSDT": _market(100.0)},
            features={},
        )
        list(mod.decide(snap))
        assert inner.call_count == 1


# ── ParamRouter integration ─────────────────────────────────

class TestParamRouting:
    def test_param_routing_disabled_by_default(self):
        mod = RegimeAwareDecisionModule(inner=_MockDecisionModule())
        assert mod.enable_param_routing is False
        assert mod.current_regime_params is None

    def test_param_routing_when_enabled(self):
        """When param routing is enabled and composite label is available,
        params should be routed."""
        composite = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        label = RegimeLabel(
            name="composite", ts=_TS,
            value="strong_up|low_vol", score=0.8,
            meta={"composite": composite, "is_crisis": False, "is_favorable": True},
        )
        detector = _StubDetector(label)
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(
            inner=inner,
            detectors=[detector],
            enable_param_routing=True,
        )

        # Provide features so detection proceeds (not skipped due to empty features)
        snap = _snapshot(
            markets={"ETHUSDT": _market(100.0)},
            features={"parkinson_vol": 0.02, "close_vs_ma20": 0.01},
        )
        list(mod.decide(snap))

        params = mod.current_regime_params
        assert params is not None
        # strong_up + low_vol → aggressive params
        assert params.deadzone == 0.3
        assert params.min_hold == 18
        assert params.position_scale == 1.0

    def test_param_routing_disabled_no_params(self):
        """When param routing is disabled, no params are set even with labels."""
        composite = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        label = RegimeLabel(
            name="composite", ts=_TS,
            value="strong_up|low_vol", score=0.8,
            meta={"composite": composite},
        )
        detector = _StubDetector(label)
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(
            inner=inner,
            detectors=[detector],
            enable_param_routing=False,
        )

        snap = _snapshot(
            markets={"ETHUSDT": _market(100.0)},
            features={"parkinson_vol": 0.02},
        )
        list(mod.decide(snap))
        assert mod.current_regime_params is None

    def test_param_routing_crisis_regime(self):
        """Crisis regime should produce minimal trading params."""
        composite = CompositeRegimeLabel(vol="crisis", trend="ranging")
        label = RegimeLabel(
            name="composite", ts=_TS,
            value="ranging|crisis", score=0.95,
            meta={"composite": composite, "is_crisis": True, "is_favorable": False},
        )
        detector = _StubDetector(label)
        inner = _MockDecisionModule()
        mod = RegimeAwareDecisionModule(
            inner=inner,
            detectors=[detector],
            enable_param_routing=True,
        )

        snap = _snapshot(
            markets={"ETHUSDT": _market(100.0)},
            features={"parkinson_vol": 0.1},
        )
        # Crisis will be blocked by policy, but params should still be routed
        list(mod.decide(snap))

        params = mod.current_regime_params
        assert params is not None
        assert params.deadzone == 2.5
        assert params.position_scale == 0.1


# ── Policy: composite crisis blocking ───────────────────────

class TestCompositeCrisisPolicy:
    def test_crisis_in_composite_blocked(self):
        """RegimePolicy should block when composite label contains crisis."""
        policy = RegimePolicy()
        labels = [
            RegimeLabel(
                name="composite", ts=_TS,
                value="ranging|crisis", score=0.95,
            ),
        ]
        ok, reason = policy.allow(labels)
        assert ok is False
        assert "crisis" in reason

    def test_non_crisis_composite_allowed(self):
        policy = RegimePolicy()
        labels = [
            RegimeLabel(
                name="composite", ts=_TS,
                value="strong_up|normal_vol", score=0.5,
            ),
        ]
        ok, reason = policy.allow(labels)
        assert ok is True

    def test_high_vol_composite_allowed(self):
        """High vol without crisis should not trigger composite crisis block."""
        policy = RegimePolicy()
        labels = [
            RegimeLabel(
                name="composite", ts=_TS,
                value="ranging|high_vol", score=0.7,
            ),
        ]
        ok, reason = policy.allow(labels)
        assert ok is True

    def test_crisis_blocked_even_with_strong_trend(self):
        policy = RegimePolicy()
        labels = [
            RegimeLabel(
                name="composite", ts=_TS,
                value="strong_up|crisis", score=0.9,
            ),
        ]
        ok, reason = policy.allow(labels)
        assert ok is False


# ── Decision builder wiring ──────────────────────────────────

class TestDecisionBuilderWiring:
    def test_regime_sizing_enables_param_routing(self):
        """When config.enable_regime_sizing=True, param_routing should be set."""
        inner = _MockDecisionModule()
        config = SimpleNamespace(
            enable_regime_gate=True,
            enable_regime_sizing=True,
        )
        # Simulate what decision_builder does
        modules = [inner]
        if config.enable_regime_gate and modules:
            gated_modules = []
            for mod in modules:
                gated = RegimeAwareDecisionModule(
                    inner=mod,
                    policy=RegimePolicy(),
                )
                if getattr(config, "enable_regime_sizing", False):
                    gated.enable_param_routing = True
                gated_modules.append(gated)
            modules = gated_modules

        assert isinstance(modules[0], RegimeAwareDecisionModule)
        assert modules[0].enable_param_routing is True

    def test_no_regime_sizing_no_param_routing(self):
        inner = _MockDecisionModule()
        config = SimpleNamespace(
            enable_regime_gate=True,
            enable_regime_sizing=False,
        )
        modules = [inner]
        if config.enable_regime_gate and modules:
            gated_modules = []
            for mod in modules:
                gated = RegimeAwareDecisionModule(
                    inner=mod,
                    policy=RegimePolicy(),
                )
                if getattr(config, "enable_regime_sizing", False):
                    gated.enable_param_routing = True
                gated_modules.append(gated)
            modules = gated_modules

        assert modules[0].enable_param_routing is False

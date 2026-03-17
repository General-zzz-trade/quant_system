# tests/contract/test_regime_contract.py
"""Contract: regime detection — composite, param router, policy gate."""
from __future__ import annotations

from datetime import datetime, timezone


from decision.regime_policy import RegimePolicy
from regime.base import RegimeLabel
from regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from regime.param_router import (
    RegimeParamRouter,
    RegimeParams,
    _FALLBACK,
)

_TS = datetime(2026, 3, 17, tzinfo=timezone.utc)


# ── CompositeRegimeDetector ──────────────────────────────────

class TestCompositeDetectorContract:
    def test_crisis_vol_produces_crisis_label(self):
        """Extreme vol_of_vol → composite labels crisis."""
        det = CompositeRegimeDetector()
        # Feed enough bars to build history for percentile calculation
        for i in range(40):
            det.detect(
                symbol="BTCUSDT", ts=_TS,
                features={
                    "parkinson_vol": 0.01 + i * 0.0001,
                    "vol_of_vol": 0.002 + i * 0.0001,
                    "close_vs_ma20": 0.01,
                    "close_vs_ma50": 0.01,
                },
            )
        # Spike vol_of_vol to extreme
        label = det.detect(
            symbol="BTCUSDT", ts=_TS,
            features={
                "parkinson_vol": 0.10,
                "vol_of_vol": 0.50,
                "close_vs_ma20": -0.05,
                "close_vs_ma50": -0.03,
            },
        )
        assert label is not None
        assert label.name == "composite"
        composite = label.meta["composite"]
        assert composite.is_crisis

    def test_composite_value_format(self):
        """Composite value must be 'trend|vol' format."""
        det = CompositeRegimeDetector()
        for i in range(35):
            det.detect(
                symbol="ETHUSDT", ts=_TS,
                features={
                    "parkinson_vol": 0.02,
                    "vol_of_vol": 0.005,
                    "close_vs_ma20": 0.03,
                    "close_vs_ma50": 0.04,
                    "adx_14": 30.0,
                },
            )
        label = det.detect(
            symbol="ETHUSDT", ts=_TS,
            features={
                "parkinson_vol": 0.02,
                "vol_of_vol": 0.005,
                "close_vs_ma20": 0.03,
                "close_vs_ma50": 0.04,
                "adx_14": 30.0,
            },
        )
        if label is not None:
            assert "|" in label.value
            parts = label.value.split("|")
            assert len(parts) == 2

    def test_favorable_when_strong_trend_low_vol(self):
        """Strong trend + low vol = favorable."""
        comp = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        assert comp.is_favorable is True
        assert comp.is_crisis is False

    def test_not_favorable_when_ranging_high_vol(self):
        comp = CompositeRegimeLabel(vol="high_vol", trend="ranging")
        assert comp.is_favorable is False


# ── RegimeParamRouter ────────────────────────────────────────

class TestParamRouterContract:
    def test_crisis_always_mapped(self):
        """Crisis regime must always have a mapping (wildcard)."""
        router = RegimeParamRouter()
        for trend in ["strong_up", "weak_up", "ranging", "weak_down", "strong_down"]:
            crisis = CompositeRegimeLabel(vol="crisis", trend=trend)
            params = router.route(crisis)
            assert params.position_scale <= 0.2, (
                f"Crisis {trend}|crisis should have low scale, got {params.position_scale}"
            )
            assert params.deadzone >= 2.0, (
                f"Crisis should have high deadzone, got {params.deadzone}"
            )

    def test_all_vol_levels_have_mapping(self):
        """Every vol level should produce valid params (no KeyError)."""
        router = RegimeParamRouter()
        for vol in ["low_vol", "normal_vol", "high_vol", "crisis"]:
            for trend in ["strong_up", "weak_up", "ranging", "weak_down", "strong_down"]:
                regime = CompositeRegimeLabel(vol=vol, trend=trend)
                params = router.route(regime)
                assert isinstance(params, RegimeParams)
                assert 0.0 <= params.position_scale <= 1.0
                assert params.deadzone > 0
                assert params.min_hold > 0
                assert params.max_hold >= params.min_hold

    def test_fallback_for_unknown_regime(self):
        """Unknown combination must produce fallback params, not error."""
        router = RegimeParamRouter()
        unknown = CompositeRegimeLabel(vol="unknown", trend="unknown")
        params = router.route(unknown)
        assert params == _FALLBACK

    def test_favorable_more_aggressive_than_unfavorable(self):
        """Favorable regime should have lower deadzone than unfavorable."""
        router = RegimeParamRouter()
        favorable = router.route(CompositeRegimeLabel(vol="low_vol", trend="strong_up"))
        unfavorable = router.route(CompositeRegimeLabel(vol="high_vol", trend="ranging"))
        assert favorable.deadzone < unfavorable.deadzone
        assert favorable.position_scale > unfavorable.position_scale


# ── RegimePolicy ─────────────────────────────────────────────

class TestRegimePolicyContract:
    def test_crisis_composite_blocked(self):
        """Composite label with 'crisis' in value must be blocked."""
        policy = RegimePolicy()
        labels = [
            RegimeLabel(name="composite", ts=_TS,
                        value="ranging|crisis", score=0.95),
        ]
        allowed, reason = policy.allow(labels)
        assert allowed is False
        assert "crisis" in reason

    def test_non_crisis_allowed(self):
        """Normal composite labels must be allowed."""
        policy = RegimePolicy()
        labels = [
            RegimeLabel(name="composite", ts=_TS,
                        value="strong_up|normal_vol", score=0.5),
        ]
        allowed, _ = policy.allow(labels)
        assert allowed is True

    def test_empty_labels_allowed(self):
        """No labels → trading allowed."""
        policy = RegimePolicy()
        allowed, reason = policy.allow([])
        assert allowed is True
        assert reason == "ok"

    def test_high_vol_flat_trend_blocked(self):
        """High vol + flat trend combination must be blocked (default policy)."""
        policy = RegimePolicy()
        labels = [
            RegimeLabel(name="volatility", ts=_TS, value="high"),
            RegimeLabel(name="trend", ts=_TS, value="flat"),
        ]
        allowed, reason = policy.allow(labels)
        assert allowed is False
        assert "high_vol_flat_trend" in reason

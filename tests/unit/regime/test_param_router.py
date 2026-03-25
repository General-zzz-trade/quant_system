"""Tests for RegimeParamRouter."""

import pytest

from regime.composite import CompositeRegimeLabel
from regime.param_router import DEFAULT_PARAMS, RegimeParamRouter, RegimeParams


@pytest.fixture
def router():
    return RegimeParamRouter()


class TestRegimeParams:
    def test_frozen(self):
        p = RegimeParams(0.5, 18, 60, 0.8)
        with pytest.raises(AttributeError):
            p.deadzone = 1.0

    def test_values(self):
        p = RegimeParams(0.5, 18, 60, 0.8)
        assert p.deadzone == 0.5
        assert p.min_hold == 18
        assert p.max_hold == 60
        assert p.position_scale == 0.8


class TestRegimeParamRouter:
    def test_exact_match_strong_up_low_vol(self, router):
        regime = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        params = router.route(regime)
        assert params.deadzone == 0.3
        assert params.position_scale == 1.0

    def test_exact_match_ranging_normal_vol(self, router):
        regime = CompositeRegimeLabel(vol="normal_vol", trend="ranging")
        params = router.route(regime)
        assert params.deadzone == 1.2
        assert params.position_scale == 0.4

    def test_wildcard_crisis(self, router):
        """Any trend + crisis should match the wildcard crisis entry."""
        for trend in ["strong_up", "weak_down", "ranging"]:
            regime = CompositeRegimeLabel(vol="crisis", trend=trend)
            params = router.route(regime)
            assert params.deadzone == 2.5
            assert params.position_scale == 0.1
            assert params.min_hold == 48

    def test_wildcard_high_vol_fallback(self, router):
        """Trend values without explicit high_vol entry use wildcard."""
        regime = CompositeRegimeLabel(vol="high_vol", trend="strong_up")
        params = router.route(regime)
        assert params.deadzone == 1.5
        assert params.position_scale == 0.3

    def test_ranging_high_vol_wildcard_match(self, router):
        """Wildcard (*, high_vol) matches before exact (ranging, high_vol) due to lookup order."""
        regime = CompositeRegimeLabel(vol="high_vol", trend="ranging")
        params = router.route(regime)
        # Wildcard (*, high_vol) takes precedence in current implementation
        assert params.deadzone == 1.5
        assert params.position_scale == 0.3

    def test_fallback_on_unknown(self, router):
        """Unknown regime combinations use fallback."""
        regime = CompositeRegimeLabel(vol="unknown_regime", trend="unknown_trend")
        params = router.route(regime)
        assert params.deadzone == 1.0
        assert params.position_scale == 0.5

    def test_custom_params(self):
        custom = {
            ("strong_up", "low_vol"): RegimeParams(0.1, 10, 30, 1.0),
        }
        router = RegimeParamRouter(params=custom)
        regime = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        assert router.route(regime).deadzone == 0.1

        # Unknown falls back
        regime2 = CompositeRegimeLabel(vol="crisis", trend="ranging")
        assert router.route(regime2).deadzone == 1.0

    def test_custom_fallback(self):
        fallback = RegimeParams(2.0, 36, 120, 0.2)
        router = RegimeParamRouter(params={}, fallback=fallback)
        regime = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        params = router.route(regime)
        assert params == fallback

    def test_all_default_params_have_valid_fields(self):
        for key, params in DEFAULT_PARAMS.items():
            assert 0.0 <= params.position_scale <= 1.0, f"Bad scale for {key}"
            assert params.min_hold > 0, f"Bad min_hold for {key}"
            assert params.max_hold >= params.min_hold, f"max_hold < min_hold for {key}"
            assert params.deadzone > 0, f"Bad deadzone for {key}"

"""Tests for EnsemblePredictor and SignalDiscretizer."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer


# ── EnsemblePredictor tests ──────────────────────────────────────────


class TestEnsemblePredictor:
    """Tests for Ridge+LGBM IC-weighted ensemble prediction."""

    def _make_model(self, return_val: float) -> MagicMock:
        m = MagicMock()
        m.predict.return_value = [return_val]
        return m

    def test_ridge_lgbm_weighted_ensemble(self):
        """Ridge 60% + LGBM 40% weighting."""
        ridge = self._make_model(1.0)
        lgbm = self._make_model(2.0)
        hm = [
            {
                "ridge": ridge,
                "lgbm": lgbm,
                "features": ["feat_a", "feat_b"],
                "ridge_features": ["feat_a", "feat_b"],
                "ic": 1.0,
            }
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"feat_a": 0.5, "feat_b": 0.3})
        # Ridge 1.0 * 0.6 + LGBM 2.0 * 0.4 = 1.4
        assert result == pytest.approx(1.4)

    def test_4h_ridge_only(self):
        """4h models skip LGBM, use Ridge only."""
        ridge = self._make_model(1.0)
        lgbm = self._make_model(99.0)  # should be ignored
        hm = [
            {
                "ridge": ridge,
                "lgbm": lgbm,
                "features": ["feat_a"],
                "ridge_features": ["feat_a"],
                "ic": 1.0,
            }
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={"version": "BTCUSDT_4h_v2"})
        result = ep.predict({"feat_a": 0.5})
        assert result == pytest.approx(1.0)
        lgbm.predict.assert_not_called()

    def test_nan_features_use_neutral_defaults(self):
        """NaN/None features replaced with neutral defaults."""
        ridge = self._make_model(1.0)
        lgbm = self._make_model(2.0)
        hm = [
            {
                "ridge": ridge,
                "lgbm": lgbm,
                "features": ["rsi_14", "ls_ratio"],
                "ridge_features": ["rsi_14", "ls_ratio"],
                "ic": 1.0,
            }
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"rsi_14": float("nan"), "ls_ratio": None})
        assert result is not None
        # Check that ridge was called with neutral defaults
        ridge_call_args = ridge.predict.call_args[0][0]
        assert ridge_call_args[0] == [50.0, 1.0]

    def test_missing_features_use_neutral_defaults(self):
        """Features not in feat_dict use neutral defaults."""
        ridge = self._make_model(1.0)
        lgbm = self._make_model(2.0)
        hm = [
            {
                "ridge": ridge,
                "lgbm": lgbm,
                "features": ["rsi_14", "unknown_feat"],
                "ridge_features": ["rsi_14", "unknown_feat"],
                "ic": 1.0,
            }
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        ep.predict({})
        ridge_call_args = ridge.predict.call_args[0][0]
        # rsi_14 -> 50.0 (neutral), unknown_feat -> 0.0 (default)
        assert ridge_call_args[0] == [50.0, 0.0]

    def test_ic_weighting_across_horizons(self):
        """Multiple horizons weighted by IC."""
        ridge1 = self._make_model(1.0)
        lgbm1 = self._make_model(1.0)
        ridge2 = self._make_model(3.0)
        lgbm2 = self._make_model(3.0)
        hm = [
            {"ridge": ridge1, "lgbm": lgbm1, "features": ["f"], "ridge_features": ["f"], "ic": 0.1},
            {"ridge": ridge2, "lgbm": lgbm2, "features": ["f"], "ridge_features": ["f"], "ic": 0.3},
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"f": 0.5})
        # h1: pred = 1*0.6 + 1*0.4 = 1.0, weighted by ic=0.1 -> 0.1
        # h2: pred = 3*0.6 + 3*0.4 = 3.0, weighted by ic=0.3 -> 0.9
        # total = (0.1 + 0.9) / (0.1 + 0.3) = 1.0 / 0.4 = 2.5
        assert result == pytest.approx(2.5)

    def test_no_ridge_uses_lgbm_only(self):
        """When ridge is None, use LGBM only."""
        lgbm = self._make_model(5.0)
        hm = [
            {"ridge": None, "lgbm": lgbm, "features": ["f"], "ridge_features": None, "ic": 1.0}
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"f": 0.5})
        assert result == pytest.approx(5.0)

    def test_empty_horizon_models_returns_none(self):
        """No horizon models returns None."""
        ep = EnsemblePredictor(horizon_models=[], config={})
        assert ep.predict({"f": 1.0}) is None


# ── SignalDiscretizer tests ──────────────────────────────────────────


class TestSignalDiscretizer:
    """Tests for z-score normalization + deadzone + min-hold discretization."""

    def _make_bridge(self, z_val=1.5, signal=1):
        bridge = MagicMock()
        bridge.zscore_normalize.return_value = z_val
        bridge.apply_constraints.return_value = signal
        return bridge

    def test_z_above_deadzone_returns_long(self):
        """Z above deadzone produces long signal."""
        bridge = self._make_bridge(z_val=2.0, signal=1)
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=True)
        assert signal == 1
        assert z == pytest.approx(2.0)
        bridge.apply_constraints.assert_called_once()

    def test_regime_filtered_forces_flat(self):
        """When regime_ok=False, deadzone=999 forces flat."""
        bridge = self._make_bridge(z_val=2.0, signal=0)
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=False)
        assert signal == 0
        # Verify deadzone=999.0 was passed
        call_kwargs = bridge.apply_constraints.call_args
        assert call_kwargs[1]["deadzone"] == 999.0

    def test_z_clamp_extreme_no_position(self):
        """Z > 3.5 with no position clamped to 3.0."""
        bridge = self._make_bridge(z_val=4.5, signal=1)
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=True, current_signal=0)
        assert z == pytest.approx(3.0)

    def test_z_clamp_negative(self):
        """Z < -3.5 with no position clamped to -3.0."""
        bridge = self._make_bridge(z_val=-4.5, signal=-1)
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=True, current_signal=0)
        assert z == pytest.approx(-3.0)

    def test_z_clamp_not_applied_when_in_position(self):
        """Z > 3.5 NOT clamped when already in position."""
        bridge = self._make_bridge(z_val=4.5, signal=1)
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=True, current_signal=1)
        assert z == pytest.approx(4.5)

    def test_warmup_returns_zero(self):
        """During warmup (bridge returns None), return (0, 0.0)."""
        bridge = MagicMock()
        bridge.zscore_normalize.return_value = None
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=True)
        assert signal == 0
        assert z == 0.0
        bridge.apply_constraints.assert_not_called()

    def test_z_clipped_to_5(self):
        """Z-scores clipped to [-5, 5] range."""
        bridge = self._make_bridge(z_val=8.0, signal=1)
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        # With current_signal=1, no clamp, but clip to 5.0
        signal, z = sd.discretize(pred=0.01, hour_key=100, regime_ok=True, current_signal=1)
        assert z == pytest.approx(5.0)

    def test_deadzone_is_settable(self):
        """Deadzone is a public settable attribute for vol-adaptive use."""
        bridge = self._make_bridge()
        sd = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=0.5, min_hold=5, max_hold=60, long_only=False,
        )
        assert sd.deadzone == 0.5
        sd.deadzone = 1.2
        assert sd.deadzone == 1.2

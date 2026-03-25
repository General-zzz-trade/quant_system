"""Exception and boundary-condition tests for EnsemblePredictor and SignalDiscretizer.

Covers: all horizons failing, partial failures, NaN/inf model outputs,
missing features, warmup, min-hold.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from decision.signals.alpha_signal import (
    EnsemblePredictor,
    SignalDiscretizer,
    _safe_val,
    _NEUTRAL_DEFAULTS,
)


# ── helpers ────────────────────────────────────────────────────────


def _make_model(return_val: float) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = [return_val]
    return m


def _make_failing_model() -> MagicMock:
    m = MagicMock()
    m.predict.side_effect = RuntimeError("model error")
    return m


def _make_horizon(
    ridge_val: float | None = 0.5,
    lgbm_val: float = 0.5,
    ic: float = 1.0,
    features: list[str] | None = None,
    ridge_features: list[str] | None = None,
    ridge_failing: bool = False,
    lgbm_failing: bool = False,
) -> dict:
    feats = features or ["feat_a", "feat_b"]
    hm: dict = {
        "features": feats,
        "ic": ic,
    }
    if ridge_val is not None:
        hm["ridge"] = _make_failing_model() if ridge_failing else _make_model(ridge_val)
        hm["ridge_features"] = ridge_features or feats
    if lgbm_failing:
        hm["lgbm"] = _make_failing_model()
    else:
        hm["lgbm"] = _make_model(lgbm_val)
    return hm


def _make_bridge(z_val: float | None = 1.0, constraint_val: int = 1) -> MagicMock:
    bridge = MagicMock()
    bridge.zscore_normalize.return_value = z_val
    bridge.apply_constraints.return_value = constraint_val
    return bridge


# ── _safe_val tests ────────────────────────────────────────────────


class TestSafeVal:
    """Tests for the _safe_val helper."""

    def test_none_returns_neutral_default(self):
        assert _safe_val(None, "rsi_14") == 50.0

    def test_none_unknown_feature_returns_zero(self):
        assert _safe_val(None, "unknown_feat") == 0.0

    def test_nan_returns_neutral_default(self):
        assert _safe_val(float("nan"), "ls_ratio") == 1.0

    def test_normal_float_passes_through(self):
        assert _safe_val(0.75, "rsi_14") == 0.75

    def test_inf_passes_through(self):
        # inf is not NaN, so it passes through
        assert _safe_val(float("inf"), "feat_a") == float("inf")

    def test_string_returns_neutral(self):
        assert _safe_val("bad", "rsi_14") == 50.0


# ── EnsemblePredictor tests ───────────────────────────────────────


class TestEnsemblePredictorExceptions:
    """Edge cases for EnsemblePredictor.predict()."""

    def test_empty_horizon_models_returns_none(self):
        ep = EnsemblePredictor(horizon_models=[], config={})
        assert ep.predict({"feat_a": 0.5}) is None

    def test_all_horizons_ridge_fail_propagates_exception(self):
        """If all ridge models raise, predict() propagates the exception."""
        hm = [_make_horizon(ridge_val=0.5, ridge_failing=True)]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        with pytest.raises(RuntimeError):
            ep.predict({"feat_a": 0.5, "feat_b": 0.3})

    def test_lgbm_only_fail_propagates(self):
        """If no ridge and LGBM fails, exception propagates."""
        hm = [{
            "features": ["feat_a"],
            "lgbm": _make_failing_model(),
            "ic": 1.0,
        }]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        with pytest.raises(RuntimeError):
            ep.predict({"feat_a": 0.5})

    def test_model_returns_nan_passes_through(self):
        """Model returning NaN still produces a result (NaN propagates)."""
        hm = [_make_horizon(ridge_val=float("nan"), lgbm_val=0.5)]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"feat_a": 0.5, "feat_b": 0.3})
        # NaN * 0.6 + 0.5 * 0.4 → NaN (NaN poisons arithmetic)
        assert result is not None
        assert math.isnan(result)

    def test_model_returns_inf_passes_through(self):
        """Model returning inf produces inf result."""
        hm = [_make_horizon(ridge_val=float("inf"), lgbm_val=0.5)]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"feat_a": 0.5, "feat_b": 0.3})
        assert result is not None
        assert math.isinf(result)

    def test_missing_features_use_neutral_defaults(self):
        """Missing keys in feat_dict → _safe_val fills neutrals."""
        ridge = _make_model(1.0)
        lgbm = _make_model(2.0)
        hm = [{
            "ridge": ridge,
            "lgbm": lgbm,
            "features": ["rsi_14", "ls_ratio", "unknown_x"],
            "ridge_features": ["rsi_14", "ls_ratio", "unknown_x"],
            "ic": 1.0,
        }]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        # Empty dict → all features will be neutral defaults
        result = ep.predict({})
        assert result is not None
        # Ridge called with [50.0, 1.0, 0.0], LGBM called with same
        ridge.predict.assert_called_once()
        # predict([rx]) where rx=[50.0, 1.0, 0.0] → arg is [[50.0, 1.0, 0.0]]
        call_args = ridge.predict.call_args[0][0]
        assert call_args == [[50.0, 1.0, 0.0]]

    def test_ic_zero_clamped_to_minimum(self):
        """IC = 0 is clamped to 0.001 so it doesn't zero out the weight."""
        hm = [_make_horizon(ridge_val=None, lgbm_val=1.0, ic=0.0)]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"feat_a": 0.5, "feat_b": 0.3})
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_negative_ic_clamped_to_minimum(self):
        """Negative IC is clamped to 0.001."""
        hm = [_make_horizon(ridge_val=None, lgbm_val=2.0, ic=-0.5)]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"feat_a": 0.5, "feat_b": 0.3})
        assert result is not None
        assert result == pytest.approx(2.0)

    def test_multiple_horizons_partial_ic(self):
        """Multiple horizons with different ICs are properly weighted."""
        hm = [
            _make_horizon(ridge_val=None, lgbm_val=1.0, ic=2.0),
            _make_horizon(ridge_val=None, lgbm_val=3.0, ic=1.0),
        ]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        result = ep.predict({"feat_a": 0.5, "feat_b": 0.3})
        # (1.0*2.0 + 3.0*1.0) / (2.0+1.0) = 5.0/3.0 ≈ 1.667
        assert result == pytest.approx(5.0 / 3.0)

    def test_ridge_features_differ_from_lgbm_features(self):
        """Ridge uses its own feature list when ridge_features is set."""
        ridge = _make_model(1.0)
        lgbm = _make_model(2.0)
        hm = [{
            "ridge": ridge,
            "lgbm": lgbm,
            "features": ["feat_a", "feat_b", "feat_c"],
            "ridge_features": ["feat_x", "feat_y"],
            "ic": 1.0,
        }]
        ep = EnsemblePredictor(horizon_models=hm, config={})
        ep.predict({"feat_a": 1, "feat_b": 2, "feat_c": 3, "feat_x": 10, "feat_y": 20})
        # Ridge called with [10, 20], LGBM called with [1, 2, 3]
        # predict([rx]) wraps the feature list in another list
        ridge_args = ridge.predict.call_args[0][0]
        lgbm_args = lgbm.predict.call_args[0][0]
        assert ridge_args == [[10.0, 20.0]]
        assert lgbm_args == [[1.0, 2.0, 3.0]]


# ── SignalDiscretizer tests ────────────────────────────────────────


class TestSignalDiscretizerExceptions:
    """Edge cases for SignalDiscretizer.discretize()."""

    def test_zscore_none_during_warmup_returns_zero(self):
        """Bridge returns None during warmup → signal=0, z=0.0."""
        bridge = _make_bridge(z_val=None)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        signal, z = disc.discretize(0.5, hour_key=1, regime_ok=True)
        assert signal == 0
        assert z == 0.0

    def test_zscore_nan_from_bridge(self):
        """Bridge returns NaN → clips to [-5, 5] range (NaN handling)."""
        bridge = _make_bridge(z_val=float("nan"))
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        # NaN comparison: max(-5, min(5, NaN)) → NaN in Python
        # The bridge.apply_constraints still gets called
        signal, z = disc.discretize(0.5, hour_key=5, regime_ok=True)
        # constraints bridge returns 1 (mocked)
        bridge.apply_constraints.assert_called_once()

    def test_extreme_z_positive_clamped(self):
        """Z > 3.5 with no position → clamped to +3.0."""
        bridge = _make_bridge(z_val=4.0, constraint_val=1)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        signal, z = disc.discretize(0.5, hour_key=5, regime_ok=True, current_signal=0)
        assert z == 3.0

    def test_extreme_z_negative_clamped(self):
        """Z < -3.5 with no position → clamped to -3.0."""
        bridge = _make_bridge(z_val=-4.0, constraint_val=-1)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        signal, z = disc.discretize(0.5, hour_key=5, regime_ok=True, current_signal=0)
        assert z == -3.0

    def test_extreme_z_with_position_no_clamp(self):
        """Z > 3.5 with existing position → no clamping."""
        bridge = _make_bridge(z_val=4.0, constraint_val=1)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        signal, z = disc.discretize(0.5, hour_key=5, regime_ok=True, current_signal=1)
        assert z == 4.0  # not clamped

    def test_regime_not_ok_forces_high_deadzone(self):
        """Regime not OK → effective deadzone = 999 (forces flat)."""
        bridge = _make_bridge(z_val=1.5, constraint_val=0)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        signal, z = disc.discretize(0.5, hour_key=5, regime_ok=False)
        # apply_constraints called with deadzone=999.0
        call_kwargs = bridge.apply_constraints.call_args
        assert call_kwargs[1]["deadzone"] == 999.0 or call_kwargs[0][3] == 999.0

    def test_long_only_passed_to_bridge(self):
        """long_only flag is forwarded to the bridge."""
        bridge = _make_bridge(z_val=1.0, constraint_val=0)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="ETHUSDT",
            deadzone=0.9, min_hold=18, max_hold=120, long_only=True,
        )
        disc.discretize(0.5, hour_key=5, regime_ok=True)
        call_kwargs = bridge.apply_constraints.call_args[1]
        assert call_kwargs["long_only"] is True

    def test_z_clip_at_5(self):
        """Z values > 5 are clipped to 5."""
        bridge = _make_bridge(z_val=10.0, constraint_val=1)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        # current_signal=1 to avoid z_clamp (only active when signal=0)
        _, z = disc.discretize(0.5, hour_key=5, regime_ok=True, current_signal=1)
        assert z == 5.0

    def test_z_clip_at_neg_5(self):
        """Z values < -5 are clipped to -5."""
        bridge = _make_bridge(z_val=-10.0, constraint_val=-1)
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        _, z = disc.discretize(0.5, hour_key=5, regime_ok=True, current_signal=-1)
        assert z == -5.0

    def test_deadzone_property_setter(self):
        """Deadzone can be updated at runtime (vol-adaptive)."""
        bridge = _make_bridge()
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        assert disc.deadzone == 1.0
        disc.deadzone = 2.5
        assert disc.deadzone == 2.5

    def test_min_hold_property_setter(self):
        """min_hold can be updated at runtime."""
        bridge = _make_bridge()
        disc = SignalDiscretizer(
            bridge=bridge, symbol="BTCUSDT",
            deadzone=1.0, min_hold=18, max_hold=120,
        )
        disc.min_hold = 30
        assert disc.min_hold == 30

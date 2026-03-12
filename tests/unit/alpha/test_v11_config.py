"""Tests for V11Config — config loading, defaults, validation."""
from __future__ import annotations

import pytest
from alpha.v11_config import ExitConfig, RegimeGateConfig, TimeFilterConfig, V11Config


class TestV11ConfigDefaults:
    """Default values must match v10 behavior."""

    def test_default_horizons(self):
        cfg = V11Config()
        assert cfg.horizons == [12, 24, 48]

    def test_default_ensemble_method(self):
        cfg = V11Config()
        assert cfg.ensemble_method == "mean_zscore"

    def test_default_lgbm_xgb_weight(self):
        cfg = V11Config()
        assert cfg.lgbm_xgb_weight == 0.5

    def test_default_zscore_params(self):
        cfg = V11Config()
        assert cfg.zscore_window == 720
        assert cfg.zscore_warmup == 180

    def test_default_entry_params(self):
        cfg = V11Config()
        assert cfg.deadzone == 0.3
        assert cfg.min_hold == 12
        assert cfg.max_hold == 96
        assert cfg.long_only is False

    def test_default_exit_config(self):
        cfg = V11Config()
        assert cfg.exit.trailing_stop_pct == 0.0
        assert cfg.exit.zscore_cap == 0.0
        assert cfg.exit.reversal_threshold == -0.3
        assert cfg.exit.deadzone_fade == 0.2

    def test_default_regime_gate_disabled(self):
        cfg = V11Config()
        assert cfg.regime_gate.enabled is False

    def test_default_time_filter_disabled(self):
        cfg = V11Config()
        assert cfg.time_filter.enabled is False


class TestV11ConfigFromJson:
    """Loading from v10 and v11 config.json formats."""

    def test_v10_config_loads(self):
        """A v10 config.json (no exit/regime/time_filter) loads with defaults."""
        v10_cfg = {
            "symbol": "ETHUSDT",
            "version": "v10",
            "multi_horizon": True,
            "horizons": [12, 24, 48],
            "ensemble_method": "mean_zscore",
            "deadzone": 0.3,
            "min_hold": 12,
            "max_hold": 96,
            "long_only": False,
        }
        cfg = V11Config.from_config_json(v10_cfg)
        assert cfg.horizons == [12, 24, 48]
        assert cfg.deadzone == 0.3
        assert cfg.min_hold == 12
        assert cfg.max_hold == 96
        assert cfg.ensemble_method == "mean_zscore"
        # New modules default to disabled
        assert cfg.exit.trailing_stop_pct == 0.0
        assert cfg.regime_gate.enabled is False
        assert cfg.time_filter.enabled is False

    def test_v10_btc_config_loads(self):
        """BTC v10 config with different deadzone and max_hold."""
        v10_cfg = {
            "symbol": "BTCUSDT",
            "horizons": [12, 24, 48],
            "deadzone": 2.5,
            "min_hold": 12,
            "max_hold": 60,
        }
        cfg = V11Config.from_config_json(v10_cfg)
        assert cfg.deadzone == 2.5
        assert cfg.max_hold == 60

    def test_v11_config_with_exit(self):
        v11_cfg = {
            "horizons": [6, 12, 24],
            "ensemble_method": "ic_weighted",
            "exit": {
                "trailing_stop_pct": 0.02,
                "zscore_cap": 4.0,
                "reversal_threshold": -0.4,
                "deadzone_fade": 0.15,
            },
            "regime_gate": {
                "enabled": True,
                "reduce_factor": 0.3,
            },
            "time_filter": {
                "enabled": True,
                "skip_hours_utc": [0, 1, 2],
            },
        }
        cfg = V11Config.from_config_json(v11_cfg)
        assert cfg.horizons == [6, 12, 24]
        assert cfg.ensemble_method == "ic_weighted"
        assert cfg.exit.trailing_stop_pct == 0.02
        assert cfg.exit.zscore_cap == 4.0
        assert cfg.regime_gate.enabled is True
        assert cfg.regime_gate.reduce_factor == 0.3
        assert cfg.time_filter.enabled is True
        assert cfg.time_filter.skip_hours_utc == [0, 1, 2]

    def test_missing_horizons_uses_default(self):
        cfg = V11Config.from_config_json({})
        assert cfg.horizons == [12, 24, 48]

    def test_lgbm_xgb_weight_custom(self):
        cfg = V11Config.from_config_json({"lgbm_xgb_weight": 0.7})
        assert cfg.lgbm_xgb_weight == 0.7


class TestV11ConfigValidation:
    """Invalid values should raise."""

    def test_invalid_ensemble_method(self):
        with pytest.raises(ValueError, match="ensemble_method"):
            V11Config(ensemble_method="invalid")

    def test_invalid_lgbm_xgb_weight_high(self):
        with pytest.raises(ValueError, match="lgbm_xgb_weight"):
            V11Config(lgbm_xgb_weight=1.5)

    def test_invalid_lgbm_xgb_weight_low(self):
        with pytest.raises(ValueError, match="lgbm_xgb_weight"):
            V11Config(lgbm_xgb_weight=-0.1)

    def test_invalid_zscore_warmup_zero(self):
        with pytest.raises(ValueError, match="zscore_warmup"):
            V11Config(zscore_warmup=0)

    def test_invalid_zscore_window_lt_warmup(self):
        with pytest.raises(ValueError, match="zscore_window"):
            V11Config(zscore_window=100, zscore_warmup=200)

    def test_empty_horizons(self):
        with pytest.raises(ValueError, match="horizons"):
            V11Config(horizons=[])

    def test_invalid_regime_gate_action(self):
        with pytest.raises(ValueError, match="ranging_high_vol_action"):
            RegimeGateConfig(ranging_high_vol_action="invalid")

    def test_invalid_reduce_factor(self):
        with pytest.raises(ValueError, match="reduce_factor"):
            RegimeGateConfig(reduce_factor=1.5)

    def test_invalid_skip_hours(self):
        with pytest.raises(ValueError, match="skip_hours_utc"):
            TimeFilterConfig(skip_hours_utc=[25])


class TestExitConfigTimeFilterSync:
    """exit.time_filter should be synced with top-level time_filter."""

    def test_time_filter_synced(self):
        tf = TimeFilterConfig(enabled=True, skip_hours_utc=[3, 4])
        cfg = V11Config(time_filter=tf)
        assert cfg.exit.time_filter is cfg.time_filter
        assert cfg.exit.time_filter.enabled is True

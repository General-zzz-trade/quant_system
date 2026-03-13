"""Tests for runner.model_discovery — model auto-discovery and loading."""
from __future__ import annotations

import json
import logging
import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from runner.model_discovery import (
    discover_active_models,
    load_symbol_models,
    build_inference_bridge,
    build_feature_computer,
)


@pytest.fixture
def models_dir(tmp_path):
    """Create a temporary models_v8 directory structure."""
    # Active BTCUSDT model
    btc_dir = tmp_path / "BTCUSDT_gate_v2"
    btc_dir.mkdir()
    btc_config = {
        "symbol": "BTCUSDT",
        "version": "v11",
        "horizon_models": [
            {
                "horizon": 24,
                "lgbm": "lgbm_h24.pkl",
                "xgb": "xgb_h24.pkl",
                "features": ["ret_1", "rsi_14"],
                "ic": 0.04,
            }
        ],
        "deadzone": 2.0,
        "min_hold": 12,
        "max_hold": 96,
        "long_only": False,
    }
    (btc_dir / "config.json").write_text(json.dumps(btc_config))

    # Active ETHUSDT model
    eth_dir = tmp_path / "ETHUSDT_gate_v2"
    eth_dir.mkdir()
    eth_config = {
        "symbol": "ETHUSDT",
        "version": "v11",
        "horizon_models": [
            {
                "horizon": 12,
                "lgbm": "lgbm_h12.pkl",
                "features": ["ret_1"],
                "ic": 0.05,
            },
            {
                "horizon": 24,
                "lgbm": "lgbm_h24.pkl",
                "features": ["ret_1", "vol_20"],
                "ic": 0.07,
            },
        ],
        "deadzone": 0.3,
        "min_hold": 12,
        "max_hold": 60,
    }
    (eth_dir / "config.json").write_text(json.dumps(eth_config))

    # Backup — should be skipped
    backup_dir = tmp_path / "BTCUSDT_gate_v2_backup_20260312"
    backup_dir.mkdir()
    (backup_dir / "config.json").write_text(json.dumps({"symbol": "BTCUSDT"}))

    # Non-gate model — should be skipped
    other_dir = tmp_path / "BTCUSDT_15m"
    other_dir.mkdir()
    (other_dir / "config.json").write_text(json.dumps({"symbol": "BTCUSDT"}))

    # Directory without config.json — should be skipped
    no_cfg = tmp_path / "SOLUSDT_gate_v2"
    no_cfg.mkdir()

    return tmp_path


def test_discover_finds_active_models(models_dir):
    result = discover_active_models(str(models_dir))
    assert "BTCUSDT" in result
    assert "ETHUSDT" in result
    assert len(result) == 2


def test_discover_skips_backup(models_dir):
    result = discover_active_models(str(models_dir))
    # Only 2 active models, backup is excluded
    assert len(result) == 2


def test_discover_skips_non_gate(models_dir):
    result = discover_active_models(str(models_dir))
    # BTCUSDT_15m is not _gate_v2, so only gate_v2 entries
    dirs = [str(v["dir"].name) for v in result.values()]
    assert "BTCUSDT_15m" not in dirs


def test_discover_skips_missing_config(models_dir):
    result = discover_active_models(str(models_dir))
    # SOLUSDT_gate_v2 has no config.json
    assert "SOLUSDT" not in result


def test_discover_nonexistent_dir():
    result = discover_active_models("/nonexistent/path")
    assert result == {}


def test_load_symbol_models_config_only(models_dir):
    """Only loads models specified in config.json, not all pkl files."""
    btc_dir = models_dir / "BTCUSDT_gate_v2"
    config = json.loads((btc_dir / "config.json").read_text())

    # Create dummy pkl files (content doesn't matter — we mock load())
    for fname in ["lgbm_h24.pkl", "xgb_h24.pkl", "legacy_h12.pkl"]:
        (btc_dir / fname).touch()

    with patch("alpha.models.lgbm_alpha.LGBMAlphaModel") as MockLGBM, \
         patch("alpha.models.xgb_alpha.XGBAlphaModel") as MockXGB:
        mock_lgbm = MagicMock()
        mock_xgb = MagicMock()
        MockLGBM.return_value = mock_lgbm
        MockXGB.return_value = mock_xgb

        models, weights = load_symbol_models("BTCUSDT", btc_dir, config)

        # Should load exactly 2 models (1 lgbm + 1 xgb), not the legacy pkl
        assert len(models) == 2
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-6


def test_load_symbol_models_missing_pkl(models_dir):
    """Handles missing pkl files gracefully."""
    btc_dir = models_dir / "BTCUSDT_gate_v2"
    config = json.loads((btc_dir / "config.json").read_text())
    # Don't create any pkl files — paths in config won't exist

    models, weights = load_symbol_models("BTCUSDT", btc_dir, config)
    assert len(models) == 0
    assert len(weights) == 0


def test_load_multi_horizon_weights(models_dir):
    """Multi-horizon models get IC-weighted ensemble weights."""
    eth_dir = models_dir / "ETHUSDT_gate_v2"
    config = json.loads((eth_dir / "config.json").read_text())

    # Create dummy pkl files for both horizons (lgbm only)
    for fname in ["lgbm_h12.pkl", "lgbm_h24.pkl"]:
        (eth_dir / fname).touch()

    with patch("alpha.models.lgbm_alpha.LGBMAlphaModel") as MockLGBM:
        mock_m = MagicMock()
        MockLGBM.return_value = mock_m

        models, weights = load_symbol_models("ETHUSDT", eth_dir, config)
        # h12 has lgbm only, h24 has lgbm only → 2 models
        assert len(models) == 2
        assert len(weights) == 2
        # h24 IC (0.07) > h12 IC (0.05) → h24 weight should be higher
        assert weights[1] > weights[0]


def test_build_inference_bridge_logs_override(models_dir, caplog):
    """config.json deadzone overrides YAML and is logged."""
    caplog.set_level(logging.INFO)
    btc_dir = models_dir / "BTCUSDT_gate_v2"
    config = json.loads((btc_dir / "config.json").read_text())

    mock_model = MagicMock()
    mock_model.name = "test_model"

    # Runner config with deadzone=0.5 (YAML default)
    runner_cfg = MagicMock()
    runner_cfg.deadzone = 0.5
    runner_cfg.min_hold_bars = {"BTCUSDT": 12}
    runner_cfg.max_hold = 120
    runner_cfg.long_only_symbols = None
    runner_cfg.monthly_gate = False
    runner_cfg.monthly_gate_window = 480
    runner_cfg.vol_target = None
    runner_cfg.vol_feature = "atr_norm_14"

    with patch("alpha.inference.bridge.LiveInferenceBridge") as MockBridge:
        MockBridge.return_value = MagicMock()
        bridge = build_inference_bridge(
            "BTCUSDT", [mock_model], config, runner_cfg,
            ensemble_weights=[1.0],
        )
        # config.json has deadzone=2.0, YAML has 0.5 → should be called with 2.0
        call_kwargs = MockBridge.call_args[1]
        assert call_kwargs["deadzone"] == {"BTCUSDT": 2.0}
        assert "overrides" in caplog.text


def test_build_feature_computer():
    """build_feature_computer returns an EnrichedFeatureComputer."""
    with patch("features.enriched_computer.EnrichedFeatureComputer") as MockFC:
        MockFC.return_value = MagicMock()
        fc = build_feature_computer()
        MockFC.assert_called_once()
        assert fc is not None

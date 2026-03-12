# tests/unit/runner/test_build_ml_stack_v8.py
"""Tests for _build_ml_stack V8 ensemble loading and signal constraint extraction."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from runner.testnet_validation import _build_ml_stack


@pytest.fixture()
def v8_model_dir(tmp_path: Path) -> Path:
    """Create a fake V8 model directory with config.json and stub pkl files."""
    model_dir = tmp_path / "models_v8" / "BTCUSDT_gate_v2"
    model_dir.mkdir(parents=True)

    # Stub pkl files (just need model + features keys)
    for fname, prefix in [("lgbm_v8.pkl", "lgbm"), ("xgb_v8.pkl", "xgb")]:
        with (model_dir / fname).open("wb") as f:
            pickle.dump({
                "model": None,  # no actual model needed for loading test
                "features": ["feat_a", "feat_b"],
            }, f)

    config = {
        "version": "v8",
        "symbol": "BTCUSDT",
        "ensemble": True,
        "ensemble_weights": [0.5, 0.5],
        "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
        "features": ["feat_a", "feat_b"],
        "long_only": True,
        "deadzone": 0.5,
        "min_hold": 24,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


@pytest.fixture()
def legacy_model_dir(tmp_path: Path) -> Path:
    """Create a fake legacy model directory (sym/config.pkl)."""
    model_dir = tmp_path / "legacy_models"
    sym_dir = model_dir / "BTCUSDT"
    sym_dir.mkdir(parents=True)

    with (sym_dir / "mod_reg_1h.pkl").open("wb") as f:
        pickle.dump({
            "model": None,
            "features": ["feat_x"],
        }, f)
    return model_dir


class TestBuildMlStackV8:

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_ensemble_loading(self, _mock_verify, v8_model_dir: Path):
        raw = {
            "strategy": {"model_path": str(v8_model_dir), "threshold": 0.001},
            "trading": {"symbols": ["BTCUSDT"]},
        }
        fc, models, dms, signal_kwargs = _build_ml_stack(raw)

        assert fc is not None
        assert len(models) == 1  # single ensemble wrapping 2 sub-models
        assert models[0].name == "ensemble_BTCUSDT"
        assert len(models[0].sub_models) == 2
        assert len(dms) == 1

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_signal_kwargs_extracted(self, _mock_verify, v8_model_dir: Path):
        raw = {
            "strategy": {"model_path": str(v8_model_dir)},
            "trading": {"symbols": ["BTCUSDT"]},
        }
        _, _, _, signal_kwargs = _build_ml_stack(raw)

        assert signal_kwargs["long_only_symbols"] == {"BTCUSDT"}
        assert signal_kwargs["deadzone"] == 0.5
        assert signal_kwargs["min_hold_bars"] == {"BTCUSDT": 24}

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_monthly_gate_from_strategy(self, _mock_verify, v8_model_dir: Path):
        raw = {
            "strategy": {
                "model_path": str(v8_model_dir),
                "monthly_gate": True,
                "monthly_gate_window": 480,
            },
            "trading": {"symbols": ["BTCUSDT"]},
        }
        _, _, _, signal_kwargs = _build_ml_stack(raw)

        assert signal_kwargs["monthly_gate"] is True
        assert signal_kwargs["monthly_gate_window"] == 480

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_trend_and_vol_overrides_from_strategy(self, _mock_verify, v8_model_dir: Path):
        raw = {
            "strategy": {
                "model_path": str(v8_model_dir),
                "trend_follow": True,
                "trend_indicator": "trend",
                "trend_threshold": 0.1,
                "max_hold": 72,
                "position_management": {
                    "vol_target": 0.2,
                    "vol_feature": "atr_norm_14",
                },
            },
            "trading": {"symbols": ["BTCUSDT"]},
        }
        _, _, _, signal_kwargs = _build_ml_stack(raw)

        assert signal_kwargs["trend_follow"] is True
        assert signal_kwargs["trend_indicator"] == "trend"
        assert signal_kwargs["trend_threshold"] == 0.1
        assert signal_kwargs["max_hold"] == 72
        assert signal_kwargs["vol_target"] == 0.2
        assert signal_kwargs["vol_feature"] == "atr_norm_14"

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_legacy_fallback(self, _mock_verify, legacy_model_dir: Path):
        raw = {
            "strategy": {
                "model_path": str(legacy_model_dir),
                "config_name": "mod_reg_1h",
            },
            "trading": {"symbols": ["BTCUSDT"]},
        }
        fc, models, dms, signal_kwargs = _build_ml_stack(raw)

        assert fc is not None
        assert len(models) == 1
        assert models[0].name == "mod_reg_1h_BTCUSDT"
        assert signal_kwargs == {}

    def test_no_model_path_returns_empty(self):
        raw = {"strategy": {}, "trading": {"symbols": ["BTCUSDT"]}}
        fc, models, dms, signal_kwargs = _build_ml_stack(raw)

        assert fc is None
        assert models == []
        assert dms == []
        assert signal_kwargs == {}

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_ensemble_weights_match(self, _mock_verify, v8_model_dir: Path):
        raw = {
            "strategy": {"model_path": str(v8_model_dir)},
            "trading": {"symbols": ["BTCUSDT"]},
        }
        _, models, _, _ = _build_ml_stack(raw)

        ens = models[0]
        assert list(ens.weights) == [0.5, 0.5]

    @patch("infra.model_signing.verify_file", return_value=True)
    def test_missing_pkl_skipped(self, _mock_verify, v8_model_dir: Path):
        # Remove one pkl file
        (v8_model_dir / "xgb_v8.pkl").unlink()

        raw = {
            "strategy": {"model_path": str(v8_model_dir)},
            "trading": {"symbols": ["BTCUSDT"]},
        }
        _, models, _, _ = _build_ml_stack(raw)

        ens = models[0]
        assert len(ens.sub_models) == 1

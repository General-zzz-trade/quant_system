"""Tests for alpha/model_loader_prod.py — model loading and adapter creation.

NOTE: Uses pickle for test fixtures because the production code (model_loader_prod.py)
deserializes HMAC-verified pickle model artifacts. This is intentional.
"""
from __future__ import annotations

import json
import os
import pickle  # noqa: S403 — required to create test model fixtures matching production format
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha.model_loader_prod import load_model, create_adapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _DummyModel:
    """A picklable stand-in for an ML model."""
    n_features_in_ = 3
    def predict(self, X):
        return [0.5] * len(X)


@pytest.fixture
def model_dir(tmp_path):
    """Create a minimal model directory with config + dummy model files."""
    lgbm_model = _DummyModel()

    lgbm_path = tmp_path / "model_h1.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump({"model": lgbm_model}, f)  # noqa: S301

    # Create config.json
    config = {
        "horizon_models": [
            {
                "horizon": "1h",
                "lgbm": "model_h1.pkl",
                "features": ["rsi_14", "macd_signal", "atr_14"],
                "ic": 0.05,
            },
        ],
        "deadzone": 1.5,
        "min_hold": 18,
        "max_hold": 96,
        "zscore_window": 720,
        "zscore_warmup": 180,
        "long_only": False,
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    return tmp_path


@pytest.fixture
def model_dir_with_ridge(model_dir):
    """Model directory that also includes a Ridge model."""
    ridge_model = _DummyModel()

    ridge_path = model_dir / "ridge_h1.pkl"
    with open(ridge_path, "wb") as f:
        pickle.dump({"model": ridge_model, "features": ["rsi_14", "macd_signal", "atr_14"]}, f)  # noqa: S301

    # Update config to include ridge
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["horizon_models"][0]["ridge"] = "ridge_h1.pkl"
    with open(config_path, "w") as f:
        json.dump(config, f)

    return model_dir


# ---------------------------------------------------------------------------
# load_model — happy path
# ---------------------------------------------------------------------------

class TestLoadModel:
    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_returns_dict_with_expected_keys(self, mock_verify, model_dir):
        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        result = load_model(model_dir)
        expected_keys = {"config", "model", "features", "horizon_models",
                         "deadzone", "min_hold", "max_hold",
                         "zscore_window", "zscore_warmup", "long_only"}
        assert expected_keys.issubset(set(result.keys()))

    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_horizon_models_is_list(self, mock_verify, model_dir):
        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        result = load_model(model_dir)
        assert isinstance(result["horizon_models"], list)
        assert len(result["horizon_models"]) == 1

    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_deadzone_from_config(self, mock_verify, model_dir):
        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        result = load_model(model_dir)
        assert result["deadzone"] == 1.5

    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_features_from_primary_model(self, mock_verify, model_dir):
        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        result = load_model(model_dir)
        assert result["features"] == ["rsi_14", "macd_signal", "atr_14"]

    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_config_defaults_when_missing(self, mock_verify, model_dir):
        """Config values fall back to defaults when not specified."""
        config_path = model_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        del config["deadzone"]
        del config["min_hold"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        result = load_model(model_dir)
        assert result["deadzone"] == 2.0  # default
        assert result["min_hold"] == 18   # default

    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_accepts_file_path_to_config(self, mock_verify, model_dir):
        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        result = load_model(model_dir / "config.json")
        assert "horizon_models" in result


# ---------------------------------------------------------------------------
# load_model — error paths
# ---------------------------------------------------------------------------

class TestLoadModelErrors:
    def test_raises_file_not_found_for_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/model/dir"))

    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_raises_runtime_error_for_no_models(self, mock_verify, tmp_path):
        """If config has no horizon_models or all model files missing, raises."""
        config = {"horizon_models": [{"horizon": "1h", "lgbm": "missing.pkl", "features": ["a"]}]}
        (tmp_path / "config.json").write_text(json.dumps(config))
        # load_verified_pickle won't be called since lgbm_path won't exist
        with pytest.raises(RuntimeError, match="No models found"):
            load_model(tmp_path)


# ---------------------------------------------------------------------------
# Model signature verification
# ---------------------------------------------------------------------------

class TestModelVerification:
    @patch("alpha.model_loader_prod.load_verified_pickle")
    def test_verify_called_for_each_model(self, mock_verify, model_dir):
        mock_verify.side_effect = lambda p: pickle.loads(Path(p).read_bytes())  # noqa: S301
        load_model(model_dir)
        assert mock_verify.call_count >= 1
        # Verify it was called with the lgbm path
        called_paths = [str(c[0][0]) for c in mock_verify.call_args_list]
        assert any("model_h1.pkl" in p for p in called_paths)


# ---------------------------------------------------------------------------
# create_adapter
# ---------------------------------------------------------------------------

class TestCreateAdapter:
    def test_raises_without_api_key(self):
        env = {"BYBIT_API_KEY": "", "BYBIT_API_SECRET": "secret"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="BYBIT_API_KEY"):
                create_adapter()

    def test_raises_without_api_secret(self):
        env = {"BYBIT_API_KEY": "key", "BYBIT_API_SECRET": ""}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError, match="BYBIT_API_KEY"):
                create_adapter()

    @patch("execution.adapters.bybit.BybitAdapter")
    @patch("execution.adapters.bybit.BybitConfig")
    def test_creates_adapter_with_env_vars(self, mock_config_cls, mock_adapter_cls):
        """With valid env vars, adapter is created and connected."""
        env = {
            "BYBIT_API_KEY": "test-key",
            "BYBIT_API_SECRET": "test-secret",
            "BYBIT_BASE_URL": "https://api-demo.bybit.com",
        }
        mock_adapter = MagicMock()
        mock_adapter.connect.return_value = True
        mock_adapter_cls.return_value = mock_adapter

        with patch.dict(os.environ, env, clear=False):
            result = create_adapter()
        mock_adapter.connect.assert_called_once()
        assert result is mock_adapter

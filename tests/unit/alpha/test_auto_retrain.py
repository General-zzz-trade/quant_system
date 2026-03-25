"""Tests for auto_retrain and auto_retrain_helpers modules."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch



# ---------------------------------------------------------------------------
# calibrate_ensemble_weights
# ---------------------------------------------------------------------------

class TestCalibrateEnsembleWeights:
    """Tests for calibrate_ensemble_weights helper."""

    def test_returns_none_when_no_config(self, tmp_path: Path):
        """Should return None gracefully when config.json is missing."""
        from alpha.retrain.helpers import calibrate_ensemble_weights

        result = calibrate_ensemble_weights(tmp_path, shrinkage=0.3)
        assert result is None

    def test_returns_none_when_no_score_data(self, tmp_path: Path):
        """Should return None when config exists but no score/return history."""
        from alpha.retrain.helpers import calibrate_ensemble_weights

        cfg = {"metrics": {}}  # no per_horizon_ic
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        result = calibrate_ensemble_weights(tmp_path, shrinkage=0.3)
        assert result is None

    def test_returns_weights_with_valid_data(self, tmp_path: Path):
        """Should return normalized weights dict when Rust calibration succeeds."""
        from alpha.retrain.helpers import calibrate_ensemble_weights

        cfg = {
            "metrics": {"per_horizon_ic": {"6": 0.05, "12": 0.04, "24": 0.03}},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        mock_rust = MagicMock(return_value={"ridge": 0.7, "lgbm": 0.3})

        with patch.dict("sys.modules", {"_quant_hotpath": MagicMock(
            rust_adaptive_ensemble_calibrate=mock_rust,
        )}):
            result = calibrate_ensemble_weights(tmp_path, shrinkage=0.3)

        assert result is not None
        assert "ridge" in result
        assert "lgbm" in result
        assert abs(result["ridge"] + result["lgbm"] - 1.0) < 1e-6

    def test_never_raises_on_error(self, tmp_path: Path):
        """Should return None on any internal error, never raise."""
        from alpha.retrain.helpers import calibrate_ensemble_weights

        # Write invalid JSON
        (tmp_path / "config.json").write_text("{invalid json")

        result = calibrate_ensemble_weights(tmp_path, shrinkage=0.3)
        assert result is None


# ---------------------------------------------------------------------------
# save_experiment_metadata
# ---------------------------------------------------------------------------

class TestSaveExperimentMetadata:
    """Tests for save_experiment_metadata helper."""

    def test_writes_json_with_expected_fields(self, tmp_path: Path):
        """Should write experiment_meta.json with all required fields."""
        from alpha.retrain.helpers import save_experiment_metadata

        cfg = {
            "metrics": {"avg_ic": 0.05, "sharpe": 2.0, "train_rows": 5000},
            "features": ["feat_a", "feat_b"],
            "ridge_weight": 0.6,
            "lgbm_weight": 0.4,
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        old_config = {"train_date": "2025-01-01"}

        ok = save_experiment_metadata(
            tmp_path, "BTCUSDT", [6, 12, 24], old_config,
            retrain_trigger="scheduled",
        )
        assert ok is True

        meta_path = tmp_path / "experiment_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())

        assert meta["symbol"] == "BTCUSDT"
        assert meta["horizons"] == [6, 12, 24]
        assert meta["n_features"] == 2
        assert meta["retrain_trigger"] == "scheduled"
        assert "trained_at" in meta
        assert "ensemble_weights" in meta

    def test_returns_false_on_missing_config(self, tmp_path: Path):
        """Should return False when config.json does not exist."""
        from alpha.retrain.helpers import save_experiment_metadata

        ok = save_experiment_metadata(tmp_path, "BTCUSDT", [6], None)
        assert ok is False

    def test_returns_false_on_error_never_raises(self, tmp_path: Path):
        """Should return False on any error, never raise."""
        from alpha.retrain.helpers import save_experiment_metadata

        (tmp_path / "config.json").write_text("{bad json")
        ok = save_experiment_metadata(tmp_path, "BTCUSDT", [6], None)
        assert ok is False


# ---------------------------------------------------------------------------
# _check_model_age_hours
# ---------------------------------------------------------------------------

class TestCheckModelAgeHours:
    """Tests for _check_model_age_hours."""

    def test_returns_hours_for_existing_model(self, tmp_path: Path):
        """Should return age in hours when config has valid train_date."""
        from alpha.retrain.pipeline import _check_model_age_hours

        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        cfg = {"train_date": yesterday}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        age = _check_model_age_hours("BTCUSDT", model_dir=tmp_path)
        assert age is not None
        # Should be roughly 24 hours (within a few hours tolerance)
        assert 20.0 < age < 50.0

    def test_returns_none_for_missing_model(self, tmp_path: Path):
        """Should return None when config.json does not exist."""
        from alpha.retrain.pipeline import _check_model_age_hours

        age = _check_model_age_hours("BTCUSDT", model_dir=tmp_path)
        assert age is None

    def test_returns_none_for_missing_train_date(self, tmp_path: Path):
        """Should return None when config exists but train_date is missing."""
        from alpha.retrain.pipeline import _check_model_age_hours

        cfg = {"metrics": {"avg_ic": 0.05}}
        (tmp_path / "config.json").write_text(json.dumps(cfg))

        age = _check_model_age_hours("BTCUSDT", model_dir=tmp_path)
        assert age is None

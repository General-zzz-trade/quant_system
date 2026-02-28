"""Tests for scripts/train_lgbm.py — utility functions and config loading."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.train_lgbm import (
    FEATURE_NAMES,
    TARGET_HORIZON,
    compute_target,
    load_config,
)


# ── compute_target ──────────────────────────────────────────


def test_compute_target_basic():
    closes = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    result = compute_target(closes, horizon=1)
    # (close[t+1] - close[t]) / close[t]
    assert result.iloc[0] == pytest.approx(0.1)  # (110-100)/100
    assert result.iloc[1] == pytest.approx(120.0 / 110.0 - 1.0)
    assert pd.isna(result.iloc[-1])  # last has no forward data


def test_compute_target_default_horizon():
    closes = pd.Series([100.0] * 10 + [200.0])
    result = compute_target(closes)
    # horizon=5 by default
    assert result.iloc[0] == pytest.approx(0.0)  # 100/100 - 1
    assert pd.isna(result.iloc[-1])  # last 5 are NaN


def test_compute_target_all_same():
    closes = pd.Series([50.0] * 20)
    result = compute_target(closes, horizon=3)
    # 50/50 - 1 = 0 for all non-NaN
    valid = result.dropna()
    assert (valid == 0.0).all()


def test_compute_target_nan_count():
    n = 100
    closes = pd.Series(np.random.uniform(90, 110, n))
    horizon = 7
    result = compute_target(closes, horizon=horizon)
    # Exactly `horizon` NaN values at the tail
    assert result.isna().sum() == horizon


def test_compute_target_negative_return():
    closes = pd.Series([200.0, 100.0])
    result = compute_target(closes, horizon=1)
    assert result.iloc[0] == pytest.approx(-0.5)


# ── load_config ─────────────────────────────────────────────


def test_load_config_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "features:\n  fast_ma: 5\n  slow_ma: 20\n"
        "training:\n  n_splits: 3\n  horizon: 10\n  out_dir: models/test\n"
        "data_path: data/test.csv\n"
    )
    cfg = load_config(cfg_path)
    assert cfg["features"]["fast_ma"] == 5
    assert cfg["features"]["slow_ma"] == 20
    assert cfg["training"]["n_splits"] == 3
    assert cfg["training"]["horizon"] == 10
    assert cfg["data_path"] == "data/test.csv"


def test_load_config_minimal(tmp_path):
    cfg_path = tmp_path / "min.yaml"
    cfg_path.write_text("features: {}\ntraining: {}\n")
    cfg = load_config(cfg_path)
    assert cfg["features"] == {}
    assert cfg["training"] == {}


# ── Constants ───────────────────────────────────────────────


def test_feature_names():
    assert isinstance(FEATURE_NAMES, tuple)
    assert len(FEATURE_NAMES) == 4
    assert "ma_fast" in FEATURE_NAMES
    assert "momentum" in FEATURE_NAMES


def test_target_horizon():
    assert TARGET_HORIZON == 5


# ── compute_features_from_ohlcv (mocked) ───────────────────


@patch("scripts.train_lgbm.LiveFeatureComputer")
def test_compute_features_from_ohlcv(mock_computer_cls):
    from scripts.train_lgbm import compute_features_from_ohlcv

    mock_computer = MagicMock()
    mock_computer_cls.return_value = mock_computer
    mock_computer.get_features_dict.return_value = {
        "ma_fast": 100.0,
        "ma_slow": 99.0,
        "vol": 0.02,
        "momentum": 0.01,
    }

    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0],
        "volume": [1000, 1100, 1200],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
    })

    result = compute_features_from_ohlcv(df)

    assert len(result) == 3
    assert "ma_fast" in result.columns
    assert "close" in result.columns
    assert mock_computer.on_bar.call_count == 3


@patch("scripts.train_lgbm.LiveFeatureComputer")
def test_compute_features_from_ohlcv_missing_optional_cols(mock_computer_cls):
    from scripts.train_lgbm import compute_features_from_ohlcv

    mock_computer = MagicMock()
    mock_computer_cls.return_value = mock_computer
    mock_computer.get_features_dict.return_value = {"ma_fast": 1.0}

    df = pd.DataFrame({"close": [100.0, 101.0]})

    result = compute_features_from_ohlcv(df)

    assert len(result) == 2
    # When volume/high/low are missing, defaults are used
    calls = mock_computer.on_bar.call_args_list
    for call in calls:
        _, kwargs = call
        assert kwargs["volume"] == 0  # default volume
        assert kwargs["high"] == kwargs["close"]  # high defaults to close
        assert kwargs["low"] == kwargs["close"]   # low defaults to close


@patch("scripts.train_lgbm.LiveFeatureComputer")
def test_compute_features_custom_params(mock_computer_cls):
    from scripts.train_lgbm import compute_features_from_ohlcv

    mock_computer = MagicMock()
    mock_computer_cls.return_value = mock_computer
    mock_computer.get_features_dict.return_value = {}

    df = pd.DataFrame({"close": [100.0]})
    compute_features_from_ohlcv(df, fast_ma=5, slow_ma=15, vol_window=10)

    mock_computer_cls.assert_called_once_with(fast_ma=5, slow_ma=15, vol_window=10)


# ── main (CLI arg parsing, mocked) ─────────────────────────


def test_main_requires_data_or_config():
    with patch("sys.argv", ["train_lgbm.py"]):
        with pytest.raises(SystemExit):
            from scripts.train_lgbm import main
            main()


@patch("scripts.train_lgbm.load_config")
def test_main_config_overrides(mock_load_config, tmp_path):
    mock_load_config.return_value = {
        "features": {"fast_ma": 7, "slow_ma": 25, "vol_window": 15},
        "training": {"n_splits": 4, "horizon": 8, "out_dir": "out/"},
        "data_path": str(tmp_path / "data.csv"),
    }

    csv_path = tmp_path / "data.csv"
    # Create CSV with too few rows to actually train — will exit with error
    df = pd.DataFrame({"close": [100.0] * 50})
    df.to_csv(csv_path, index=False)

    with patch("sys.argv", ["train_lgbm.py", "--config", str(tmp_path / "cfg.yaml")]):
        with pytest.raises(SystemExit):
            from scripts.train_lgbm import main
            main()

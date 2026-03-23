"""Integration tests for AlphaRunner gate chain — end-to-end signal→gate→size→order."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pytest


def _make_runner(symbol="ETHUSDT", dry_run=True):
    """Create a minimal AlphaRunner for gate integration testing."""
    from runner.alpha_runner import AlphaRunner

    adapter = MagicMock()
    adapter.get_balances.return_value = {
        "USDT": type("B", (), {"total": 1000.0, "available": 1000.0})()
    }
    adapter.get_ticker.return_value = {"fundingRate": "0.0001"}

    model = MagicMock()
    model.predict.return_value = np.array([0.001])
    model.coef_ = np.zeros(5)
    model.intercept_ = 0.0

    lgbm = MagicMock()
    lgbm.predict.return_value = np.array([0.001])

    model_info = {
        "model": model,
        "features": ["rsi_14", "macd_hist", "vol_20", "ret_24", "cvd_20"],
        "config": {"version": "v11"},
        "deadzone": 0.5,
        "long_only": True,
        "min_hold": 18,
        "max_hold": 60,
        "zscore_window": 720,
        "zscore_warmup": 180,
        "horizon_models": [{
            "horizon": 24,
            "lgbm": lgbm,
            "ridge": model,
            "ridge_features": ["rsi_14", "macd_hist", "vol_20", "ret_24", "cvd_20"],
            "features": ["rsi_14", "macd_hist", "vol_20", "ret_24", "cvd_20"],
            "ic": 0.15,
        }],
    }

    oi_cache = MagicMock()
    oi_cache.get.return_value = {
        "open_interest": 50000.0,
        "ls_ratio": 1.0,
        "taker_buy_vol": 100.0,
        "top_trader_ls_ratio": 1.0,
    }

    runner = AlphaRunner(
        adapter, model_info, symbol,
        dry_run=dry_run,
        oi_cache=oi_cache,
        start_oi_cache=False,
    )
    return runner, adapter


class TestGateIntegration:
    def test_gates_initialized(self):
        """Gates should be initialized in __init__."""
        runner, _ = _make_runner()
        assert hasattr(runner, "_mtf_gate")
        assert hasattr(runner, "_liq_gate")
        assert hasattr(runner, "_carry_gate")

    def test_evaluate_gates_no_signal(self):
        """Signal=0 should return scale=1.0."""
        runner, _ = _make_runner()
        scale = runner._evaluate_gates(0, {"rsi_14": 50.0})
        assert scale == 1.0

    def test_evaluate_gates_normal(self):
        """Normal conditions → scale close to 1.0."""
        runner, _ = _make_runner()
        scale = runner._evaluate_gates(1, {
            "rsi_14": 50.0,
            "funding_rate": 0.0001,
            "basis": 0.0,
        })
        assert 0.5 <= scale <= 1.5

    def test_evaluate_gates_liq_block(self):
        """High liquidation zscore → scale=0.0 (blocked)."""
        runner, _ = _make_runner()
        scale = runner._evaluate_gates(1, {
            "liquidation_volume_zscore_24": 5.0,
        })
        assert scale == 0.0

    def test_evaluate_gates_empty_features(self):
        """Empty features → all gates pass through."""
        runner, _ = _make_runner()
        scale = runner._evaluate_gates(1, {})
        assert scale == 1.0

    def test_evaluate_gates_nan_features(self):
        """NaN features → treated as missing, gates pass through."""
        runner, _ = _make_runner()
        scale = runner._evaluate_gates(1, {
            "tf4h_close_vs_ma20": float("nan"),
            "liquidation_volume_zscore_24": float("nan"),
            "funding_rate": float("nan"),
        })
        assert scale == 1.0

    def test_gate_scale_logged(self):
        """gate_scale should be stored for logging."""
        runner, _ = _make_runner()
        runner._evaluate_gates(1, {"funding_rate": 0.0001})
        assert hasattr(runner, "_gate_scale")
        assert runner._gate_scale >= 0.0


class TestOnlineRidgeIntegration:
    def test_enable_online_ridge(self):
        """enable_online_ridge should initialize without crash."""
        runner, _ = _make_runner()
        runner.enable_online_ridge()
        assert runner._online_ridge is not None

    def test_update_online_ridge_first_bar(self):
        """First bar should set last_close without updating."""
        runner, _ = _make_runner()
        runner.enable_online_ridge()
        runner._update_online_ridge({"rsi_14": 50.0}, 2000.0)
        assert runner._last_close == 2000.0
        assert runner._online_ridge.n_updates == 0

    def test_update_online_ridge_subsequent_bar(self):
        """Second bar should trigger update."""
        runner, _ = _make_runner()
        runner.enable_online_ridge()
        runner._update_online_ridge({"rsi_14": 50.0}, 2000.0)
        runner._update_online_ridge({"rsi_14": 51.0}, 2010.0)
        assert runner._online_ridge.n_updates == 1

    def test_online_ridge_not_enabled(self):
        """Without enable_online_ridge, _update should be no-op."""
        runner, _ = _make_runner()
        runner._update_online_ridge({"rsi_14": 50.0}, 2000.0)
        assert runner._online_ridge is None

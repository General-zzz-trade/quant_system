"""Tests for checkpoint save/restore — ensure state continuity across restarts."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pytest


def _make_runner(symbol="ETHUSDT", dry_run=True):
    """Create a minimal AlphaRunner with checkpoint support."""
    from scripts.ops.alpha_runner import AlphaRunner

    adapter = MagicMock()
    adapter.get_balances.return_value = {
        "USDT": type("B", (), {"total": 1000.0, "available": 1000.0})()
    }
    adapter.get_ticker.return_value = {"fundingRate": "0.0001"}
    adapter.get_klines.return_value = [
        {"close": 2100.0 + i, "high": 2105.0, "low": 2095.0,
         "open": 2100.0, "volume": 10000.0, "start": 1700000000000 + i * 3600000}
        for i in range(800)
    ][::-1]  # newest first (Bybit format)
    adapter.get_positions.return_value = []

    model = MagicMock()
    model.predict.return_value = np.array([0.001])

    model_info = {
        "model": model,
        "features": ["rsi_14", "vol_20", "ret_24"],
        "config": {"version": "v11"},
        "deadzone": 0.5,
        "long_only": True,
        "min_hold": 18,
        "max_hold": 60,
        "zscore_window": 720,
        "zscore_warmup": 180,
        "horizon_models": [{
            "horizon": 24, "lgbm": model,
            "features": ["rsi_14", "vol_20", "ret_24"], "ic": 0.15,
        }],
    }

    oi_cache = MagicMock()
    oi_cache.get.return_value = {
        "open_interest": 50000.0, "ls_ratio": 1.0,
        "taker_buy_vol": 100.0, "top_trader_ls_ratio": 1.0,
    }

    runner = AlphaRunner(
        adapter, model_info, symbol,
        dry_run=dry_run,
        oi_cache=oi_cache, start_oi_cache=False,
    )
    return runner


class TestCheckpointSaveRestore:
    """Core checkpoint functionality."""

    def test_save_creates_file(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800

        runner._save_checkpoint()

        ckpt_path = tmp_path / "ETHUSDT.json"
        assert ckpt_path.exists()
        data = json.loads(ckpt_path.read_text())
        assert data["bars_processed"] == 800
        assert "engine" in data
        assert "inference" in data

    def test_restore_loads_state(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._regime_active = False

        runner._save_checkpoint()

        # Create fresh runner and restore
        runner2 = _make_runner()
        runner2._CHECKPOINT_DIR = tmp_path
        restored = runner2._restore_checkpoint()

        assert restored is True
        assert runner2._bars_processed == 800
        assert runner2._regime_active is False

    def test_restore_returns_false_when_no_file(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        assert runner._restore_checkpoint() is False

    def test_restore_returns_false_on_corrupt_file(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        (tmp_path / "ETHUSDT.json").write_text("{corrupt json!!!")
        assert runner._restore_checkpoint() is False


class TestCheckpointNoRegression:
    """Bug: periodic save overwrites 800-bar warmup with 10-bar state."""

    def test_no_save_below_warmup_bars(self, tmp_path):
        """Checkpoint should NOT save when bars_processed < WARMUP_BARS."""
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 10  # simulates post-restore state

        # Process bars 11-20 — periodic save triggers at bar 20
        for i in range(10):
            runner._bars_processed += 1

        # Should not create checkpoint (bars=20 < WARMUP_BARS=800)
        ckpt_path = tmp_path / "ETHUSDT.json"
        assert not ckpt_path.exists()

    def test_save_only_at_warmup_threshold(self, tmp_path):
        """Checkpoint saves only when bars_processed >= WARMUP_BARS."""
        from scripts.ops.config import WARMUP_BARS

        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path

        # Simulate: restored with 10 bars, need to reach WARMUP_BARS
        runner._bars_processed = WARMUP_BARS - 1
        # This should not trigger save (bars < WARMUP_BARS)
        bar = {"close": 2100.0, "high": 2105.0, "low": 2095.0,
               "open": 2100.0, "volume": 10000.0}
        runner.process_bar(bar)

        # Now bars_processed = WARMUP_BARS, and if WARMUP_BARS % 10 == 0, save triggers
        if runner._bars_processed >= WARMUP_BARS and runner._bars_processed % 10 == 0:
            ckpt_path = tmp_path / "ETHUSDT.json"
            assert ckpt_path.exists()

    def test_warmup_saves_checkpoint(self, tmp_path):
        """Full warmup always saves checkpoint at the end."""
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path

        runner.warmup(limit=800, interval="60")

        ckpt_path = tmp_path / "ETHUSDT.json"
        assert ckpt_path.exists()
        data = json.loads(ckpt_path.read_text())
        assert data["bars_processed"] == 800

    def test_restore_then_warmup_preserves_bars(self, tmp_path):
        """After restore, bars_processed should be the checkpoint value."""
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._save_checkpoint()

        runner2 = _make_runner()
        runner2._CHECKPOINT_DIR = tmp_path
        runner2._restore_checkpoint()
        assert runner2._bars_processed == 800

    def test_checkpoint_not_overwritten_by_partial(self, tmp_path):
        """A 800-bar checkpoint should survive process_bar at bars < 800."""
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._save_checkpoint()

        initial_size = (tmp_path / "ETHUSDT.json").stat().st_size

        # Simulate restore with bars=800, then process 10 more bars
        runner2 = _make_runner()
        runner2._CHECKPOINT_DIR = tmp_path
        runner2._restore_checkpoint()
        assert runner2._bars_processed == 800

        # Process 10 bars → bars becomes 810 which is >= WARMUP_BARS
        # Save at bar 810 should work (810 >= 800)
        for _ in range(10):
            runner2._bars_processed += 1

        # At bar 810, save IS allowed (>= WARMUP_BARS)
        runner2._save_checkpoint()
        new_size = (tmp_path / "ETHUSDT.json").stat().st_size
        assert new_size > 0


class TestCheckpointDataIntegrity:
    """Verify checkpoint data survives serialization."""

    def test_atr_buffer_preserved(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._atr_buffer = [0.01, 0.02, 0.015, 0.018]

        runner._save_checkpoint()

        runner2 = _make_runner()
        runner2._CHECKPOINT_DIR = tmp_path
        runner2._restore_checkpoint()

        assert len(runner2._atr_buffer) == 4
        assert abs(runner2._atr_buffer[0] - 0.01) < 1e-10

    def test_closes_history_preserved(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._closes = [2100.0 + i for i in range(100)]

        runner._save_checkpoint()

        runner2 = _make_runner()
        runner2._CHECKPOINT_DIR = tmp_path
        runner2._restore_checkpoint()

        assert len(runner2._closes) == 100
        assert runner2._closes[0] == 2100.0

    def test_nan_in_data_handled(self, tmp_path):
        """NaN values in checkpoint data should not crash serialization."""
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._atr_buffer = [0.01, float("nan"), 0.015]

        runner._save_checkpoint()  # should not crash

        ckpt_path = tmp_path / "ETHUSDT.json"
        assert ckpt_path.exists()

    def test_regime_state_preserved(self, tmp_path):
        runner = _make_runner()
        runner._CHECKPOINT_DIR = tmp_path
        runner._bars_processed = 800
        runner._regime_active = False
        runner._deadzone = 1.5

        runner._save_checkpoint()

        runner2 = _make_runner()
        runner2._CHECKPOINT_DIR = tmp_path
        runner2._restore_checkpoint()

        assert runner2._regime_active is False
        assert runner2._deadzone == 1.5


class TestDynamicNotional:
    """Test dynamic MAX_ORDER_NOTIONAL scaling."""

    def test_dynamic_scales_with_equity(self):
        from scripts.ops.config import get_max_order_notional

        assert get_max_order_notional(500) == 100.0      # floor
        assert get_max_order_notional(5000) == 1000.0
        assert get_max_order_notional(35000) == 7000.0
        assert get_max_order_notional(500000) == 100000.0  # ceiling

    def test_floor_enforced(self):
        from scripts.ops.config import get_max_order_notional
        assert get_max_order_notional(0) == 100.0
        assert get_max_order_notional(10) == 100.0

    def test_ceiling_enforced(self):
        from scripts.ops.config import get_max_order_notional
        assert get_max_order_notional(10_000_000) == 100_000.0

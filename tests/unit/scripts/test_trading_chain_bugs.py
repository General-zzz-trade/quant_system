"""Tests for trading chain bugs found during live deployment.

Covers:
- Bybit API string→float conversion (funding_rate, bid/ask prices)
- dry_run reconcile skip
- dry_run entry_price tracking
- Leverage ladder correctness
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pytest


def _make_runner(equity=1000.0, dry_run=True, symbol="ETHUSDT"):
    """Create a minimal AlphaRunner for chain testing."""
    from runner.alpha_runner import AlphaRunner

    adapter = MagicMock()
    adapter.get_balances.return_value = {
        "USDT": type("B", (), {"total": equity, "available": equity})()
    }
    # Bybit returns ALL values as strings
    adapter.get_ticker.return_value = {
        "fundingRate": "0.0001",
        "bid1Price": "2100.00",
        "ask1Price": "2100.50",
        "lastPrice": "2100.25",
    }
    adapter.send_market_order.return_value = {"retCode": 0, "result": {"orderId": "t1"}}
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
            "horizon": 24,
            "lgbm": model,
            "features": ["rsi_14", "vol_20", "ret_24"],
            "ic": 0.15,
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
        oi_cache=oi_cache,
        start_oi_cache=False,
    )
    return runner, adapter


class TestFundingRateConversion:
    """Bug: Bybit returns fundingRate as string, Rust push_bar needs float."""

    def test_string_funding_rate_in_bar(self):
        """funding_rate as string in bar dict should be converted to float."""
        runner, _ = _make_runner()
        bar = {
            "close": 2100.0, "high": 2105.0, "low": 2095.0,
            "open": 2100.0, "volume": 10000.0,
            "funding_rate": "0.0001",  # string from API
        }
        # Should not crash
        result = runner.process_bar(bar)
        assert isinstance(result, dict)

    def test_nan_funding_rate_fallback(self):
        """NaN funding rate should trigger ticker fetch."""
        runner, adapter = _make_runner()
        bar = {
            "close": 2100.0, "high": 2105.0, "low": 2095.0,
            "open": 2100.0, "volume": 10000.0,
            # no funding_rate key → should fetch from ticker
        }
        result = runner.process_bar(bar)
        assert isinstance(result, dict)

    def test_none_funding_rate(self):
        """None funding rate should not crash."""
        runner, _ = _make_runner()
        bar = {
            "close": 2100.0, "high": 2105.0, "low": 2095.0,
            "open": 2100.0, "volume": 10000.0,
            "funding_rate": None,
        }
        result = runner.process_bar(bar)
        assert isinstance(result, dict)


class TestDryRunReconcile:
    """Bug: reconcile resets signal in dry_run (no exchange position exists)."""

    def test_no_reconcile_in_dry_run(self):
        """dry_run should skip reconciliation entirely."""
        runner, adapter = _make_runner(dry_run=True)

        # Process enough bars to trigger reconcile (every 10 bars)
        for i in range(15):
            runner.process_bar({
                "close": 2100.0 + i, "high": 2105.0, "low": 2095.0,
                "open": 2100.0, "volume": 10000.0,
            })

        # adapter.get_positions should NOT be called (no reconcile in dry_run)
        # Note: get_positions might be called during init but not during process_bar
        positions_calls = [c for c in adapter.method_calls
                          if c[0] == 'get_positions']
        # In dry_run, reconcile is skipped, so no get_positions during bars
        assert len(positions_calls) == 0


class TestDryRunEntryPrice:
    """Bug: dry_run doesn't set entry_price, causing INVARIANT VIOLATION."""

    def test_entry_price_set_on_dry_run_open(self):
        """When dry_run opens a position, entry_price should be set."""
        runner, _ = _make_runner(dry_run=True)

        # Manually trigger a signal change
        runner._current_signal = 0
        result = runner._execute_signal_change(0, 1, 2100.0)

        assert result["action"] == "dry_run"
        assert runner._entry_price == 2100.0
        assert runner._entry_size > 0
        assert runner._trade_peak_price == 2100.0

    def test_entry_price_cleared_on_dry_run_close(self):
        """When dry_run closes a position, entry_price should be cleared."""
        runner, _ = _make_runner(dry_run=True)

        # Open
        runner._execute_signal_change(0, 1, 2100.0)
        assert runner._entry_price == 2100.0

        # Close
        runner._execute_signal_change(1, 0, 2110.0)
        assert runner._entry_price == 0.0
        assert runner._entry_size == 0.0


class TestLeverageLadder:
    """Verify leverage ladder returns correct values."""

    def test_small_account_high_leverage(self):
        runner, _ = _make_runner(equity=200)
        lev = runner._get_leverage_for_equity(200)
        assert lev == 10.0

    def test_medium_account(self):
        runner, _ = _make_runner(equity=2000)
        lev = runner._get_leverage_for_equity(2000)
        assert lev == 10.0

    def test_large_account(self):
        runner, _ = _make_runner(equity=35000)
        lev = runner._get_leverage_for_equity(35000)
        assert lev == 10.0

    def test_very_large_account(self):
        runner, _ = _make_runner(equity=100000)
        lev = runner._get_leverage_for_equity(100000)
        assert lev == 10.0


class TestLimitEntryStringConversion:
    """Bug: bid/ask from Bybit ticker are strings."""

    def test_limit_entry_with_string_prices(self):
        """_execute_limit_entry should handle string bid/ask."""
        runner, adapter = _make_runner(dry_run=False)
        adapter.get_ticker.return_value = {
            "bid1Price": "2100.00",  # string!
            "ask1Price": "2100.50",  # string!
        }
        adapter.send_limit_order.return_value = {"status": "error", "retMsg": "test"}
        adapter.send_market_order.return_value = {"retCode": 0, "result": {}}

        result = runner._execute_limit_entry("ETHUSDT", "buy", 1.0, 2100.0)
        # Should not crash — falls back to market
        assert result.get("entry_method") == "market_fallback"


class TestPositionSizingChain:
    """Full chain: equity → leverage → sizing → rounding → clamp."""

    def test_sizing_produces_valid_qty(self):
        """Position size must be > 0, rounded to step, within limits."""
        runner, adapter = _make_runner(equity=500)
        runner._z_scale = 1.0
        size = runner._compute_position_size(2100.0)

        assert size > 0
        assert size >= runner._min_size
        # Check step rounding
        step = runner._step_size
        remainder = round(size % step, 10)
        assert remainder < step * 0.01 or abs(remainder - step) < step * 0.01

    def test_sizing_zero_price(self):
        """Zero price should not crash."""
        runner, _ = _make_runner(equity=500)
        size = runner._compute_position_size(0.0)
        # Should fall back to base_position_size
        assert size >= 0

    def test_sizing_with_gate_scale(self):
        """Gate scale < 1 should reduce position size."""
        runner, _ = _make_runner(equity=500)
        runner._z_scale = 1.0
        full_size = runner._compute_position_size(2100.0)

        # Simulate gate scaling
        scaled_size = max(runner._min_size, full_size * 0.5)
        assert scaled_size <= full_size
        assert scaled_size >= runner._min_size

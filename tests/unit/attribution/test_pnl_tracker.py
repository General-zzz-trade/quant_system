"""Tests for attribution/pnl_tracker.py — RustPnLTracker wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Build a mock RustPnLTracker so tests run without the Rust binary
# ---------------------------------------------------------------------------

def _make_mock_rust_tracker():
    """Create a mock that behaves like RustPnLTracker."""
    mock = MagicMock()
    mock.total_pnl = 0.0
    mock.peak_equity = 10_000.0
    mock.trade_count = 0
    mock.win_count = 0
    mock.win_rate = 0.0
    mock.drawdown_pct = 0.0

    trades_recorded: list[dict] = []

    def _record_close(symbol, side, entry_price, exit_price, size, reason=""):
        pnl = (exit_price - entry_price) * size * side
        mock.total_pnl += pnl
        mock.trade_count += 1
        if pnl > 0:
            mock.win_count += 1
        mock.win_rate = mock.win_count / mock.trade_count if mock.trade_count else 0.0
        trade = {
            "symbol": symbol, "side": side,
            "entry": entry_price, "exit": exit_price,
            "size": size, "pnl_usd": pnl, "reason": reason,
            "total_pnl": mock.total_pnl, "trade_count": mock.trade_count,
        }
        trades_recorded.append(trade)
        return trade

    mock.record_close = _record_close

    def _summary():
        return {
            "total_pnl": mock.total_pnl,
            "trade_count": mock.trade_count,
            "win_count": mock.win_count,
            "win_rate": mock.win_rate,
            "drawdown_pct": mock.drawdown_pct,
        }

    mock.summary = _summary
    return mock


@pytest.fixture(autouse=True)
def _patch_rust(monkeypatch):
    """Patch _quant_hotpath.RustPnLTracker before importing PnLTracker."""
    import sys
    mock_mod = MagicMock()
    mock_mod.RustPnLTracker = _make_mock_rust_tracker
    monkeypatch.setitem(sys.modules, "_quant_hotpath", mock_mod)
    # Force reimport
    if "attribution.pnl_tracker" in sys.modules:
        del sys.modules["attribution.pnl_tracker"]


def _get_tracker():
    from attribution.pnl_tracker import PnLTracker
    return PnLTracker()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPnLTrackerRecordClose:
    def test_single_winning_trade(self):
        t = _get_tracker()
        result = t.record_close("BTCUSDT", 1, 50_000.0, 51_000.0, 0.1)
        assert result["pnl_usd"] == pytest.approx(100.0)
        assert t.total_pnl == pytest.approx(100.0)
        assert t.trade_count == 1

    def test_single_losing_trade(self):
        t = _get_tracker()
        result = t.record_close("ETHUSDT", 1, 2_000.0, 1_900.0, 1.0)
        assert result["pnl_usd"] == pytest.approx(-100.0)
        assert t.total_pnl == pytest.approx(-100.0)

    def test_invalid_entry_price_skipped(self):
        t = _get_tracker()
        result = t.record_close("BTCUSDT", 1, 0.0, 51_000.0, 0.1)
        assert result.get("error") == "invalid_entry_price"
        assert t.trade_count == 0

    def test_nan_exit_price_skipped(self):
        t = _get_tracker()
        result = t.record_close("BTCUSDT", 1, 50_000.0, float("nan"), 0.1)
        assert result.get("error") == "invalid_exit_price"
        assert t.trade_count == 0


class TestPnLTrackerWinRate:
    def test_win_rate_all_winners(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 110.0, 1.0)
        t.record_close("BTCUSDT", 1, 100.0, 120.0, 1.0)
        assert t.win_rate == pytest.approx(1.0)

    def test_win_rate_mixed(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 110.0, 1.0)  # win
        t.record_close("BTCUSDT", 1, 100.0, 90.0, 1.0)   # loss
        assert t.win_rate == pytest.approx(0.5)


class TestPnLTrackerSummary:
    def test_summary_keys(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 110.0, 1.0)
        s = t.summary()
        required_keys = {"total_pnl", "trade_count", "win_count", "win_rate",
                         "drawdown_pct", "pnl_by_symbol", "pnl_by_horizon",
                         "best_symbol", "worst_symbol"}
        assert required_keys.issubset(s.keys())

    def test_summary_pnl_by_symbol(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 110.0, 1.0)
        t.record_close("ETHUSDT", 1, 200.0, 190.0, 1.0)
        s = t.summary()
        assert "BTCUSDT" in s["pnl_by_symbol"]
        assert "ETHUSDT" in s["pnl_by_symbol"]


class TestPnLTrackerAccumulation:
    def test_multiple_trades_accumulate(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 110.0, 1.0)   # +10
        t.record_close("BTCUSDT", 1, 100.0, 105.0, 2.0)   # +10
        t.record_close("ETHUSDT", 1, 200.0, 190.0, 1.0)    # -10
        assert t.total_pnl == pytest.approx(10.0)
        assert t.trade_count == 3

    def test_horizon_attribution(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 110.0, 1.0, horizon=60)
        t.record_close("BTCUSDT", 1, 100.0, 105.0, 1.0, horizon=240)
        pnl_by_h = t.pnl_by_horizon
        assert 60 in pnl_by_h
        assert 240 in pnl_by_h

    def test_best_worst_symbol(self):
        t = _get_tracker()
        t.record_close("BTCUSDT", 1, 100.0, 120.0, 1.0)   # +20
        t.record_close("ETHUSDT", 1, 200.0, 180.0, 1.0)    # -20
        assert t.best_symbol == "BTCUSDT"
        assert t.worst_symbol == "ETHUSDT"

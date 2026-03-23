"""Tests for PnLTracker — unified P&L tracking."""
from __future__ import annotations

import pytest

from attribution.pnl_tracker import PnLTracker


class TestPnLTracker:

    def test_empty_state(self):
        t = PnLTracker()
        assert t.total_pnl == 0.0
        assert t.peak_equity == 0.0
        assert t.trade_count == 0
        assert t.win_count == 0
        assert t.trades == []

    def test_record_long_win(self):
        t = PnLTracker()
        trade = t.record_close("ETHUSDT", side=1, entry_price=2000.0,
                               exit_price=2200.0, size=1.0)
        # pnl_pct = (2200-2000)/2000 * 100 = 10%
        # pnl_usd = 10/100 * 2000 * 1.0 = 200
        assert trade["pnl_pct"] == pytest.approx(10.0)
        assert trade["pnl_usd"] == pytest.approx(200.0)
        assert t.total_pnl == pytest.approx(200.0)
        assert t.win_count == 1

    def test_record_long_loss(self):
        t = PnLTracker()
        trade = t.record_close("ETHUSDT", side=1, entry_price=2000.0,
                               exit_price=1800.0, size=1.0)
        # pnl_pct = (1800-2000)/2000 * 100 = -10%
        # pnl_usd = -10/100 * 2000 * 1.0 = -200
        assert trade["pnl_pct"] == pytest.approx(-10.0)
        assert trade["pnl_usd"] == pytest.approx(-200.0)
        assert t.win_count == 0

    def test_record_short_win(self):
        t = PnLTracker()
        trade = t.record_close("ETHUSDT", side=-1, entry_price=2000.0,
                               exit_price=1800.0, size=1.0)
        # pnl_pct = (2000-1800)/2000 * 100 = 10%
        # pnl_usd = 10/100 * 2000 * 1.0 = 200
        assert trade["pnl_pct"] == pytest.approx(10.0)
        assert trade["pnl_usd"] == pytest.approx(200.0)
        assert t.win_count == 1

    def test_record_short_loss(self):
        t = PnLTracker()
        trade = t.record_close("ETHUSDT", side=-1, entry_price=2000.0,
                               exit_price=2200.0, size=1.0)
        # pnl_pct = (2000-2200)/2000 * 100 = -10%
        assert trade["pnl_pct"] == pytest.approx(-10.0)
        assert trade["pnl_usd"] == pytest.approx(-200.0)
        assert t.win_count == 0

    def test_multiple_trades_total(self):
        t = PnLTracker()
        t.record_close("ETHUSDT", 1, 2000.0, 2200.0, 1.0)   # +200
        # short win: pnl_pct=4%, pnl_usd=200
        t.record_close("BTCUSDT", -1, 50000.0, 48000.0, 0.1)
        t.record_close("ETHUSDT", 1, 2000.0, 1900.0, 1.0)   # -100
        assert t.total_pnl == pytest.approx(300.0)
        assert t.trade_count == 3
        assert t.win_count == 2

    def test_win_rate_calculation(self):
        t = PnLTracker()
        t.record_close("ETHUSDT", 1, 2000.0, 2200.0, 1.0)  # win
        t.record_close("ETHUSDT", 1, 2000.0, 1800.0, 1.0)  # loss
        t.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)  # win
        assert t.win_rate == pytest.approx(200 / 3)  # 66.67%

    def test_win_rate_empty(self):
        t = PnLTracker()
        assert t.win_rate == 0

    def test_drawdown_from_peak(self):
        t = PnLTracker()
        t.record_close("ETHUSDT", 1, 2000.0, 2200.0, 1.0)  # +200, peak=200
        t.record_close("ETHUSDT", 1, 2000.0, 1800.0, 1.0)  # -200, total=0
        # drawdown = (200 - 0) / 200 * 100 = 100%
        assert t.drawdown_pct == pytest.approx(100.0)

    def test_peak_only_increases(self):
        t = PnLTracker()
        t.record_close("ETHUSDT", 1, 2000.0, 2200.0, 1.0)  # +200, peak=200
        assert t.peak_equity == pytest.approx(200.0)
        t.record_close("ETHUSDT", 1, 2000.0, 1800.0, 1.0)  # -200, total=0
        assert t.peak_equity == pytest.approx(200.0)  # peak unchanged
        t.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)  # +100, total=100
        assert t.peak_equity == pytest.approx(200.0)  # still 200

    def test_peak_increases_on_new_high(self):
        t = PnLTracker()
        t.record_close("ETHUSDT", 1, 2000.0, 2200.0, 1.0)  # +200
        t.record_close("ETHUSDT", 1, 2000.0, 2400.0, 1.0)  # +400, total=600
        assert t.peak_equity == pytest.approx(600.0)

    def test_trade_log_capped(self):
        t = PnLTracker()
        for i in range(110):
            t.record_close("ETHUSDT", 1, 2000.0, 2001.0, 0.01)
        assert len(t.trades) == 100

    def test_summary_structure(self):
        t = PnLTracker()
        t.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        s = t.summary()
        assert "total_pnl" in s
        assert "trades" in s
        assert "wins" in s
        assert "win_rate" in s
        assert "peak" in s
        assert "drawdown" in s

    def test_reason_stored(self):
        t = PnLTracker()
        trade = t.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0, reason="stop_loss")
        assert trade["reason"] == "stop_loss"

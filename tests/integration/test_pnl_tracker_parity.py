"""Parity test: RustPnLTracker vs Python PnLTracker.

Requires _quant_hotpath to be built. Skipped if Rust not available.
"""
import pytest

try:
    from _quant_hotpath import RustPnLTracker
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not built")


def make_py_tracker():
    """Create a pure-Python PnLTracker (no Rust delegation)."""
    from scripts.ops.pnl_tracker import PnLTracker
    t = PnLTracker.__new__(PnLTracker)
    t._use_rust = False
    t._total_pnl = 0.0
    t._peak_equity = 0.0
    t._trade_count = 0
    t._win_count = 0
    t._trades = []
    return t


class TestRustPnLTrackerBasic:
    def test_long_win(self):
        rust = RustPnLTracker()
        r = rust.record_close("ETHUSDT", 1, 100.0, 110.0, 1.0, "signal")
        assert r["pnl_usd"] == pytest.approx(10.0, rel=1e-9)
        assert r["pnl_pct"] == pytest.approx(10.0, rel=1e-9)
        assert rust.win_rate == pytest.approx(100.0)
        assert rust.total_pnl == pytest.approx(10.0)
        assert rust.trade_count == 1
        assert rust.win_count == 1

    def test_short_win(self):
        rust = RustPnLTracker()
        r = rust.record_close("ETHUSDT", -1, 100.0, 90.0, 1.0, "stop")
        assert r["pnl_pct"] == pytest.approx(10.0, rel=1e-9)
        assert r["pnl_usd"] == pytest.approx(10.0, rel=1e-9)
        assert rust.win_rate == pytest.approx(100.0)

    def test_short_loss(self):
        rust = RustPnLTracker()
        r = rust.record_close("ETHUSDT", -1, 100.0, 110.0, 1.0, "stop")
        assert r["pnl_pct"] == pytest.approx(-10.0, rel=1e-9)
        assert r["pnl_usd"] == pytest.approx(-10.0, rel=1e-9)
        assert rust.win_rate == pytest.approx(0.0)

    def test_no_trades_win_rate_zero(self):
        rust = RustPnLTracker()
        assert rust.win_rate == pytest.approx(0.0)

    def test_peak_equity_tracks_max(self):
        rust = RustPnLTracker()
        rust.record_close("ETH", 1, 100.0, 110.0, 1.0, "win")  # pnl=10
        rust.record_close("ETH", 1, 110.0, 100.0, 1.0, "loss")  # pnl≈-9.09
        # peak should be ~10.0, current should be ~0.91
        assert rust.peak_equity == pytest.approx(10.0, rel=1e-9)
        assert rust.drawdown_pct > 0.0

    def test_drawdown_zero_before_any_profit(self):
        rust = RustPnLTracker()
        # Only losses — peak stays 0, drawdown stays 0
        rust.record_close("ETH", 1, 100.0, 90.0, 1.0, "loss")
        assert rust.drawdown_pct == pytest.approx(0.0)

    def test_summary_keys(self):
        rust = RustPnLTracker()
        rust.record_close("ETH", 1, 100.0, 110.0, 1.0, "win")
        s = rust.summary()
        assert set(s.keys()) == {"total_pnl", "trades", "wins", "win_rate", "peak", "drawdown"}

    def test_summary_values(self):
        rust = RustPnLTracker()
        rust.record_close("ETH", 1, 100.0, 110.0, 1.0, "win")
        s = rust.summary()
        assert s["total_pnl"] == pytest.approx(10.0)
        assert s["trades"] == 1
        assert s["wins"] == 1
        assert s["win_rate"] == pytest.approx(100.0)
        assert s["peak"] == pytest.approx(10.0)
        assert s["drawdown"] == pytest.approx(0.0)

    def test_nan_entry_rejected(self):
        rust = RustPnLTracker()
        with pytest.raises(Exception):
            rust.record_close("ETH", 1, float("nan"), 100.0, 1.0, "test")

    def test_nan_exit_rejected(self):
        rust = RustPnLTracker()
        with pytest.raises(Exception):
            rust.record_close("ETH", 1, 100.0, float("nan"), 1.0, "test")

    def test_inf_entry_rejected(self):
        rust = RustPnLTracker()
        with pytest.raises(Exception):
            rust.record_close("ETH", 1, float("inf"), 100.0, 1.0, "test")

    def test_zero_size_rejected(self):
        rust = RustPnLTracker()
        with pytest.raises(Exception):
            rust.record_close("ETH", 1, 100.0, 110.0, 0.0, "test")

    def test_negative_size_rejected(self):
        rust = RustPnLTracker()
        with pytest.raises(Exception):
            rust.record_close("ETH", 1, 100.0, 110.0, -1.0, "test")

    def test_nan_size_rejected(self):
        rust = RustPnLTracker()
        with pytest.raises(Exception):
            rust.record_close("ETH", 1, 100.0, 110.0, float("nan"), "test")

    def test_return_dict_keys(self):
        rust = RustPnLTracker()
        r = rust.record_close("ETHUSDT", 1, 100.0, 110.0, 2.0, "test")
        expected_keys = {
            "symbol", "side", "entry", "exit", "size",
            "pnl_usd", "pnl_pct", "reason", "total_pnl", "trade_count",
        }
        assert set(r.keys()) == expected_keys


class TestPnLTrackerParity:
    """Numeric parity between RustPnLTracker and pure-Python path."""

    def test_long_win_parity(self):
        rust = RustPnLTracker()
        py = make_py_tracker()
        r_rust = rust.record_close("ETHUSDT", 1, 100.0, 110.0, 1.0, "signal")
        r_py = py.record_close("ETHUSDT", 1, 100.0, 110.0, 1.0, "signal")
        assert r_rust["pnl_usd"] == pytest.approx(r_py["pnl_usd"], rel=1e-9)
        assert r_rust["pnl_pct"] == pytest.approx(r_py["pnl_pct"], rel=1e-9)

    def test_short_win_parity(self):
        rust = RustPnLTracker()
        py = make_py_tracker()
        r_rust = rust.record_close("ETHUSDT", -1, 100.0, 90.0, 1.0, "stop")
        r_py = py.record_close("ETHUSDT", -1, 100.0, 90.0, 1.0, "stop")
        assert r_rust["pnl_pct"] == pytest.approx(r_py["pnl_pct"], rel=1e-9)
        assert r_rust["pnl_usd"] == pytest.approx(r_py["pnl_usd"], rel=1e-9)

    def test_sequence_parity(self):
        """Run same trade sequence through Rust and Python, compare results."""
        rust = RustPnLTracker()
        py = make_py_tracker()

        trades = [
            ("ETHUSDT", 1, 100.0, 110.0, 1.0, "win"),
            ("BTCUSDT", -1, 50000.0, 48000.0, 0.1, "win"),
            ("ETHUSDT", 1, 110.0, 105.0, 1.0, "loss"),
        ]
        for sym, side, entry, exit_p, size, reason in trades:
            r_rust = rust.record_close(sym, side, entry, exit_p, size, reason)
            r_py = py.record_close(sym, side, entry, exit_p, size, reason)
            assert r_rust["pnl_usd"] == pytest.approx(r_py["pnl_usd"], rel=1e-9), \
                f"pnl_usd mismatch on trade {reason}"
            assert r_rust["pnl_pct"] == pytest.approx(r_py["pnl_pct"], rel=1e-9), \
                f"pnl_pct mismatch on trade {reason}"
            assert r_rust["total_pnl"] == pytest.approx(r_py["total_pnl"], rel=1e-9), \
                f"total_pnl mismatch on trade {reason}"

        assert rust.win_rate == pytest.approx(py.win_rate, rel=1e-9)
        assert rust.drawdown_pct == pytest.approx(py.drawdown_pct, rel=1e-9)
        assert rust.total_pnl == pytest.approx(py.total_pnl, rel=1e-9)
        assert rust.peak_equity == pytest.approx(py.peak_equity, rel=1e-9)

    def test_summary_parity(self):
        rust = RustPnLTracker()
        py = make_py_tracker()

        trades = [
            ("ETH", 1, 2000.0, 2100.0, 0.5, "win"),
            ("ETH", -1, 2100.0, 2050.0, 0.5, "win"),
            ("ETH", 1, 2050.0, 1900.0, 0.5, "loss"),
        ]
        for args in trades:
            rust.record_close(*args)
            py.record_close(*args)

        sr = rust.summary()
        sp = py.summary()
        assert sr["total_pnl"] == pytest.approx(sp["total_pnl"], rel=1e-9)
        assert sr["trades"] == sp["trades"]
        assert sr["wins"] == sp["wins"]
        assert sr["win_rate"] == pytest.approx(sp["win_rate"], rel=1e-9)
        assert sr["peak"] == pytest.approx(sp["peak"], rel=1e-9)
        assert sr["drawdown"] == pytest.approx(sp["drawdown"], rel=1e-9)

    def test_btc_large_notional_parity(self):
        rust = RustPnLTracker()
        py = make_py_tracker()
        # Large notional: BTC at $50k, size=0.5
        r_rust = rust.record_close("BTCUSDT", 1, 50000.0, 55000.0, 0.5, "tp")
        r_py = py.record_close("BTCUSDT", 1, 50000.0, 55000.0, 0.5, "tp")
        assert r_rust["pnl_usd"] == pytest.approx(r_py["pnl_usd"], rel=1e-9)
        assert r_rust["pnl_pct"] == pytest.approx(r_py["pnl_pct"], rel=1e-9)


class TestPnLTrackerWrapper:
    """Test the PnLTracker Python wrapper with Rust backend."""

    def test_wrapper_uses_rust_when_available(self):
        from scripts.ops.pnl_tracker import PnLTracker, _HAS_RUST
        t = PnLTracker()
        if _HAS_RUST:
            assert t._use_rust is True

    def test_wrapper_record_close_returns_dict(self):
        from scripts.ops.pnl_tracker import PnLTracker
        t = PnLTracker()
        r = t.record_close("ETHUSDT", 1, 100.0, 110.0, 1.0, "test")
        assert isinstance(r, dict)
        assert "pnl_usd" in r

    def test_wrapper_properties_accessible(self):
        from scripts.ops.pnl_tracker import PnLTracker
        t = PnLTracker()
        t.record_close("ETH", 1, 100.0, 110.0, 1.0, "win")
        assert t.total_pnl == pytest.approx(10.0)
        assert t.trade_count == 1
        assert t.win_count == 1
        assert t.win_rate == pytest.approx(100.0)
        assert t.peak_equity == pytest.approx(10.0)
        assert t.drawdown_pct == pytest.approx(0.0)

    def test_wrapper_summary(self):
        from scripts.ops.pnl_tracker import PnLTracker
        t = PnLTracker()
        t.record_close("ETH", 1, 100.0, 110.0, 1.0, "win")
        s = t.summary()
        assert set(s.keys()) == {"total_pnl", "trades", "wins", "win_rate", "peak", "drawdown"}

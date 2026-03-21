"""Tests for PnLTracker per-symbol and per-horizon attribution."""
from __future__ import annotations

import math
import sys
import unittest
from unittest.mock import patch


# Force pure-Python path so tests don't depend on Rust build
with patch.dict(sys.modules, {"_quant_hotpath": None}):
    # Reload to pick up _HAS_RUST = False
    import importlib
    from scripts.ops import pnl_tracker as _mod
    _orig_has_rust = _mod._HAS_RUST
    _mod._HAS_RUST = False
    from scripts.ops.pnl_tracker import PnLTracker


class TestPnLTrackerAttribution(unittest.TestCase):
    """Per-symbol and per-horizon attribution tests."""

    def setUp(self):
        # Ensure pure-Python path
        _mod._HAS_RUST = False
        self.tracker = PnLTracker()

    def tearDown(self):
        _mod._HAS_RUST = _orig_has_rust

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def test_basic_record_close_unchanged(self):
        """Existing callers without horizon kwarg still work."""
        trade = self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        self.assertIn("pnl_usd", trade)
        self.assertGreater(trade["pnl_usd"], 0)
        self.assertEqual(self.tracker.trade_count, 1)
        # No horizon key when not supplied
        self.assertNotIn("horizon", trade)

    def test_summary_has_original_keys(self):
        """summary() still contains the original keys."""
        self.tracker.record_close("BTCUSDT", 1, 50000, 51000, 0.1)
        s = self.tracker.summary()
        for key in ("total_pnl", "trades", "wins", "win_rate", "peak", "drawdown"):
            self.assertIn(key, s, f"Missing original key: {key}")

    # ------------------------------------------------------------------
    # Per-symbol PnL
    # ------------------------------------------------------------------

    def test_pnl_by_symbol_single(self):
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        by_sym = self.tracker.pnl_by_symbol
        self.assertIn("ETHUSDT", by_sym)
        self.assertAlmostEqual(by_sym["ETHUSDT"], 100.0, places=2)

    def test_pnl_by_symbol_multiple(self):
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)   # +100
        self.tracker.record_close("BTCUSDT", -1, 50000, 49000, 0.1)    # +100
        self.tracker.record_close("ETHUSDT", -1, 2000.0, 2050.0, 1.0)  # -50
        by_sym = self.tracker.pnl_by_symbol
        self.assertAlmostEqual(by_sym["ETHUSDT"], 50.0, places=2)
        self.assertAlmostEqual(by_sym["BTCUSDT"], 100.0, places=2)

    def test_pnl_by_symbol_empty(self):
        self.assertEqual(self.tracker.pnl_by_symbol, {})

    # ------------------------------------------------------------------
    # Best / worst symbol
    # ------------------------------------------------------------------

    def test_best_worst_symbol(self):
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)   # +100
        self.tracker.record_close("BTCUSDT", 1, 50000, 49000, 0.1)     # -100 (long, price dropped)
        self.assertEqual(self.tracker.best_symbol, "ETHUSDT")
        self.assertEqual(self.tracker.worst_symbol, "BTCUSDT")

    def test_best_worst_empty(self):
        self.assertEqual(self.tracker.best_symbol, "")
        self.assertEqual(self.tracker.worst_symbol, "")

    def test_best_worst_single_symbol(self):
        self.tracker.record_close("SOLUSDT", 1, 100.0, 110.0, 10.0)
        self.assertEqual(self.tracker.best_symbol, "SOLUSDT")
        self.assertEqual(self.tracker.worst_symbol, "SOLUSDT")

    # ------------------------------------------------------------------
    # Per-horizon PnL
    # ------------------------------------------------------------------

    def test_pnl_by_horizon(self):
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0, horizon=24)
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2050.0, 1.0, horizon=96)
        by_h = self.tracker.pnl_by_horizon
        self.assertAlmostEqual(by_h[24], 100.0, places=2)
        self.assertAlmostEqual(by_h[96], 50.0, places=2)

    def test_pnl_by_horizon_not_set(self):
        """When no horizon is given, pnl_by_horizon stays empty."""
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        self.assertEqual(self.tracker.pnl_by_horizon, {})

    def test_horizon_in_trade_dict(self):
        trade = self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0, horizon=24)
        self.assertEqual(trade["horizon"], 24)

    def test_horizon_absent_when_none(self):
        trade = self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        self.assertNotIn("horizon", trade)

    # ------------------------------------------------------------------
    # Per-symbol Sharpe
    # ------------------------------------------------------------------

    def test_per_symbol_sharpe_insufficient_trades(self):
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        self.assertEqual(self.tracker.per_symbol_sharpe("ETHUSDT"), 0.0)

    def test_per_symbol_sharpe_no_trades(self):
        self.assertEqual(self.tracker.per_symbol_sharpe("NONEXIST"), 0.0)

    def test_per_symbol_sharpe_positive(self):
        # Record several winning trades with some variance
        for exit_p in [2100, 2050, 2080, 2120, 2060]:
            self.tracker.record_close("ETHUSDT", 1, 2000.0, exit_p, 1.0)
        sharpe = self.tracker.per_symbol_sharpe("ETHUSDT")
        self.assertGreater(sharpe, 0.0)

    def test_per_symbol_sharpe_all_equal(self):
        """If all trades have identical PnL, std=0 -> Sharpe=0."""
        for _ in range(5):
            self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0)
        self.assertEqual(self.tracker.per_symbol_sharpe("ETHUSDT"), 0.0)

    def test_per_symbol_sharpe_mixed(self):
        """Mixed wins and losses should produce finite Sharpe."""
        self.tracker.record_close("BTCUSDT", 1, 50000, 51000, 0.1)   # +100
        self.tracker.record_close("BTCUSDT", 1, 50000, 49000, 0.1)   # -100
        self.tracker.record_close("BTCUSDT", 1, 50000, 50500, 0.1)   # +50
        sharpe = self.tracker.per_symbol_sharpe("BTCUSDT")
        self.assertTrue(math.isfinite(sharpe))

    # ------------------------------------------------------------------
    # Summary includes attribution
    # ------------------------------------------------------------------

    def test_summary_includes_attribution(self):
        self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0, horizon=24)
        self.tracker.record_close("BTCUSDT", -1, 50000, 49000, 0.1, horizon=96)
        s = self.tracker.summary()
        self.assertIn("pnl_by_symbol", s)
        self.assertIn("pnl_by_horizon", s)
        self.assertIn("best_symbol", s)
        self.assertIn("worst_symbol", s)
        self.assertIn("ETHUSDT", s["pnl_by_symbol"])
        self.assertIn(24, s["pnl_by_horizon"])

    # ------------------------------------------------------------------
    # Invalid price edge cases still work
    # ------------------------------------------------------------------

    def test_invalid_price_no_attribution(self):
        """Invalid prices should not pollute attribution dicts."""
        trade = self.tracker.record_close("ETHUSDT", 1, 0.0, 2100.0, 1.0)
        self.assertIn("error", trade)
        self.assertEqual(self.tracker.pnl_by_symbol, {})

    def test_nan_price_no_attribution(self):
        trade = self.tracker.record_close("ETHUSDT", 1, float("nan"), 2100.0, 1.0)
        self.assertIn("error", trade)
        self.assertEqual(self.tracker.pnl_by_symbol, {})

    # ------------------------------------------------------------------
    # Backward-compatible keyword-only horizon
    # ------------------------------------------------------------------

    def test_horizon_is_keyword_only(self):
        """horizon must be passed as keyword arg (positional raises TypeError)."""
        with self.assertRaises(TypeError):
            # 7 positional args: symbol, side, entry, exit, size, reason, horizon
            self.tracker.record_close("ETHUSDT", 1, 2000.0, 2100.0, 1.0, "", 24)


if __name__ == "__main__":
    unittest.main()

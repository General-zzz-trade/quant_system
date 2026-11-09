from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace as NS

from runner.gates.portfolio_risk_file import PortfolioRiskFileReader
from shared.drawdown_guard import DrawdownGuard
from shared.entry_risk import CompositeRiskGate


def timestamp(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()


class DrawdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "risk.sqlite"
        self.now = timestamp("2026-10-26T12:00:00")  # Monday
        self.guard = DrawdownGuard(self.path, "okx-demo-a")

    def observe(self, equity, delta=0):
        return self.guard.observe(equity, now=self.now + delta)[0]

    def test_no_observation_blocks(self):
        self.assertFalse(self.guard.check(now=self.now)[0])

    def test_exact_daily_boundary_latches_even_after_recovery(self):
        self.assertTrue(self.observe(100))
        self.assertFalse(self.observe(97, 1))
        self.assertFalse(self.observe(110, 2))

    def test_restart_keeps_original_daily_equity(self):
        self.observe(100)
        self.guard = DrawdownGuard(self.path, "okx-demo-a")
        self.assertFalse(self.guard.check(now=self.now)[0])  # fresh exchange read required
        self.assertFalse(self.observe(96, 1))

    def test_restart_cannot_relax_latched_limit(self):
        self.observe(100)
        self.observe(96, 1)
        self.guard = DrawdownGuard(self.path, "okx-demo-a", daily_pct=20)
        self.assertFalse(self.observe(100, 2))

    def test_daily_reset_does_not_reset_weekly_baseline(self):
        self.observe(100)
        self.assertTrue(self.observe(98, 86400))
        self.assertFalse(self.observe(89, 86401))
        self.assertFalse(self.observe(95, 86400 * 2))

    def test_new_day_clears_only_daily_latch(self):
        self.observe(100)
        self.assertFalse(self.observe(96, 1))
        self.assertTrue(self.observe(96, 86400))

    def test_next_iso_week_reinitializes_week_and_day(self):
        self.observe(100)
        self.assertFalse(self.observe(89, 1))
        self.assertTrue(self.observe(89, 7 * 86400))

    def test_iso_year_boundary_uses_iso_week_year(self):
        self.now = timestamp("2026-12-31T12:00:00")
        self.observe(100)
        self.assertFalse(self.observe(89, 86400))
        self.assertFalse(self.observe(95, 2 * 86400))

    def test_stale_or_backwards_time_blocks(self):
        self.observe(100)
        self.assertFalse(self.guard.check(now=self.now + 121)[0])
        self.assertFalse(self.guard.check(now=self.now - 1)[0])
        self.assertFalse(self.observe(100, -1))

    def test_invalid_equity_or_refresh_failure_blocks(self):
        for equity in (float("nan"), float("inf"), 0, -1):
            self.observe(100)
            self.assertFalse(self.observe(equity, 1))
        self.guard.invalidate("network down")
        self.assertFalse(self.guard.check(now=self.now)[0])

    def test_account_environments_are_isolated(self):
        self.observe(100)
        self.observe(90, 1)
        other = DrawdownGuard(self.path, "okx-live-b")
        self.assertTrue(other.observe(90, now=self.now + 1)[0])

    def test_corrupt_state_blocks(self):
        self.observe(100)
        self.path.write_text("not SQLite")
        self.assertFalse(self.guard.check(now=self.now)[0])
        self.assertFalse(self.observe(100, 1))


class PortfolioAndGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "portfolio.json"
        self.reader = PortfolioRiskFileReader(self.path)
        self.limiter = NS(check_order=lambda **kw: NS(allowed=True, detail=None, violated_rule=None))
        self.gate = CompositeRiskGate(self.limiter, portfolio_reader=self.reader)
        self.event = NS(symbol="BTCUSDT", side="buy", qty=D(1), price=D(100), reduce_only=False)

    def write(self, **overrides):
        snap = {"ts": time.time(), "symbols": {"BTCUSDT": {
            "max_total_notional_usd": 1000, "total_long_notional_usd": 0, "total_short_notional_usd": 0}}}
        snap.update(overrides)
        self.path.write_text(json.dumps(snap))
        self.reader._cache = None

    def test_missing_stale_invalid_snapshot_blocks(self):
        self.assertFalse(self.gate.check(self.event).allowed)
        for ts in (time.time() - 301, time.time() + 10, float("nan")):
            self.write(ts=ts)
            self.assertFalse(self.gate.check(self.event).allowed)
        self.path.write_text("bad json")
        self.reader._cache = None
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_complete_snapshot_and_valid_order_allowed(self):
        self.write()
        self.assertTrue(self.gate.check(self.event).allowed)

    def test_unconfigured_symbol_and_absent_exposure_block(self):
        self.write(symbols={})
        self.assertFalse(self.gate.check(self.event).allowed)
        self.write(symbols={"BTCUSDT": {"max_total_notional_usd": 1000}})
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_risk_module_exception_blocks_entry_but_allows_exit(self):
        def fail(**kwargs):
            raise RuntimeError("risk module unavailable")
        self.gate._kill_switch = NS(allow_order=fail)
        self.assertFalse(self.gate.check(self.event).allowed)
        self.event.qty = D(0)
        self.assertTrue(self.gate.check(self.event).allowed)
        self.event.qty, self.event.reduce_only = D(1), True
        self.assertTrue(self.gate.check(self.event).allowed)

    def test_portfolio_exception_blocks(self):
        self.gate._portfolio_reader = NS(check_would_exceed=lambda **kw: 1 / 0)
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_drawdown_and_runner_pause_block_only_entries(self):
        self.write()
        self.gate._drawdown_guard = NS(check=lambda: (False, "day drawdown"))
        self.assertFalse(self.gate.check(self.event).allowed)
        self.gate._drawdown_guard = None
        self.gate._paused = lambda symbol: True
        self.assertFalse(self.gate.check(self.event).allowed)
        self.event.qty = D(0)
        self.assertTrue(self.gate.check(self.event).allowed)

    def test_wallet_reconciliation_blocks_only_entries(self):
        self.write()
        self.gate._account_ledger = NS(check=lambda: (False, "wallet_balance_mismatch"))
        self.assertFalse(self.gate.check(self.event).allowed)
        self.event.qty = D(0)
        self.assertTrue(self.gate.check(self.event).allowed)

    def test_nan_negative_and_unpriced_entries_block(self):
        self.write()
        for qty in (D("NaN"), D("Infinity"), D(-1)):
            self.event.qty = qty
            self.assertFalse(self.gate.check(self.event).allowed)
        self.event.qty = D(1)
        for price in (None, D(0), D("NaN"), D("Infinity")):
            self.event.price = price
            self.assertFalse(self.gate.check(self.event).allowed)

    def test_cap_and_nonfinite_exposure_block(self):
        for exposure in (950, float("nan"), float("inf"), -1):
            self.write(symbols={"BTCUSDT": {"max_total_notional_usd": 1000,
                                            "total_long_notional_usd": exposure}})
            self.assertFalse(self.gate.check(self.event).allowed)

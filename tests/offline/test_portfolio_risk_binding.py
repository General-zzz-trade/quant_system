"""Actual gate and strict account collector integration, temporary state only."""
from decimal import Decimal as D
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from runner.portfolio_risk import (
    PortfolioRiskBinding, collect_accounts, execution_risk_preflight,
    refresh_portfolio, runtime_registration,
)
from shared.entry_risk import CompositeRiskGate
from shared.portfolio_reservations import PortfolioReservationStore
from shared.runtime_state import initialize_runtime_state


class PortfolioBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "account"
        self.journal = initialize_runtime_state(self.state, "scope-a")
        self.registration = runtime_registration("scope-a", self.state, "demo")
        self.store = PortfolioReservationStore.initialize(self.root / "risk.sqlite", group="test",
            registrations=[self.registration], caps={"BTCUSDT": D(100)})
        self.binding = PortfolioRiskBinding(self.store, self.registration)
        self.positions, self.open_orders = [], []
        self.adapter = NS(venue="bybit", _config=NS(base_url="https://api-demo.bybit.com",
                         category="linear", account_type="UNIFIED"),
                         get_positions=lambda **kw: tuple(self.positions),
                         get_open_orders=lambda **kw: tuple(self.open_orders))
        self.accounts = {"scope-a": (self.adapter, self.state)}
        self.limiter = NS(check_order=lambda **kw: NS(allowed=True, detail=None, violated_rule=None))
        self.gate = CompositeRiskGate(self.limiter, reservation=self.binding)
        self.event = NS(order_id="intent", symbol="BTCUSDT", side="buy", qty=D(1), price=D(60))

    def refresh(self):
        return refresh_portfolio(self.store, self.accounts)

    def test_gate_requires_verified_baseline_then_reserves_atomically(self):
        self.assertFalse(self.gate.check(self.event).allowed)
        self.refresh()
        self.assertTrue(self.gate.check(self.event).allowed)
        self.event.order_id = "second"
        self.assertFalse(self.gate.check(self.event).allowed)
        self.event.qty = D(0)
        self.assertTrue(self.gate.check(self.event).allowed)

    def test_limiter_or_data_pause_never_consumes_quota(self):
        self.refresh()
        self.limiter.check_order = lambda **kw: NS(allowed=False, detail="limited", violated_rule=None)
        self.assertFalse(self.gate.check(self.event).allowed)
        self.limiter.check_order = lambda **kw: NS(allowed=True, detail=None, violated_rule=None)
        self.assertTrue(self.gate.check(self.event).allowed)

    def test_changed_account_identity_blocks_before_requests(self):
        self.adapter._config.base_url = "https://api.bybit.com"
        with self.assertRaises(ValueError):
            self.refresh()
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_unknown_exposure_and_open_orders_block(self):
        self.positions.append(NS(venue="bybit", symbol="BTCUSDT", qty=D(1), mark_price=D(60)))
        with self.assertRaises(ValueError):
            self.refresh()
        self.positions.clear()
        self.open_orders.append(NS(order_id="manual"))
        with self.assertRaises(ValueError):
            self.refresh()

    def test_query_failure_immediately_invalidates_previous_success(self):
        self.refresh()
        def unavailable(**kw):
            raise TimeoutError("offline fixture")
        self.adapter.get_positions = unavailable
        with self.assertRaises(ValueError):
            self.refresh()
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_orphan_reservation_cannot_be_released_by_flat_snapshot(self):
        self.refresh()
        self.assertTrue(self.gate.check(self.event).allowed)
        with self.assertRaises(ValueError):
            self.refresh()
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_verified_terminal_fill_replaces_reservation_without_double_count(self):
        self.refresh()
        self.assertTrue(self.gate.check(self.event).allowed)
        with sqlite3.connect(self.journal) as db:
            db.execute("INSERT INTO submissions "
                "(scope,intent_id,symbol,client_id,order_id,settled,side,qty,reducing,owner,parent,status,order_type) "
                "VALUES ('scope-a','intent','BTCUSDT','client','venue',1,'buy','1',0,"
                "'alpha.BTCUSDT','{}','filled','market')")
            checkpoint = {"revision": "fill-1", "position": {"qty": "1"}}
            db.execute("INSERT INTO execution_consumers VALUES ('scope-a','alpha.BTCUSDT',?)",
                       (json.dumps(checkpoint),))
        self.positions.append(NS(venue="bybit", symbol="BTCUSDT", qty=D(1), mark_price=D(60)))
        self.refresh()
        self.event.order_id, self.event.price = "remaining", D(40)
        self.assertTrue(self.gate.check(self.event).allowed)
        self.event.order_id, self.event.price = "excess", D(1)
        self.assertFalse(self.gate.check(self.event).allowed)

    def test_partial_registry_and_slow_collection_are_not_proof(self):
        with self.assertRaises(ValueError):
            collect_accounts(self.store, {})
        with self.assertRaises(ValueError):
            self.store.reconcile(lambda: collect_accounts(self.store, self.accounts,
                                                         clock=lambda: time.time() - 301))

    def test_missing_risk_config_and_unsupported_venue_fail_before_adapter(self):
        for venue in ("bybit", "okx", "binance"):
            with self.assertRaises(ValueError):
                execution_risk_preflight(venue, {})
        env = {"RISK_SCOPE": "scope-a", "RUNTIME_STATE_DIR": str(self.state),
               "PORTFOLIO_RISK_PATH": str(self.store.path), "PORTFOLIO_RISK_GROUP": "test"}
        with patch("strategy.config.PORTFOLIO_NOTIONAL_CAPS_USD", {"BTCUSDT": 100}):
            self.assertEqual(execution_risk_preflight("bybit", env).registration, self.registration)
            env["PORTFOLIO_RISK_PATH"] = str(self.root / "lost.sqlite")
            with self.assertRaises((ValueError, FileNotFoundError)):
                execution_risk_preflight("bybit", env)
            self.assertFalse(Path(env["PORTFOLIO_RISK_PATH"]).exists())

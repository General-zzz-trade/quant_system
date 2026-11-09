from __future__ import annotations

from decimal import Decimal
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

from runner.account_reconciliation import account_ledger_preflight, reconcile_bybit_wallet
from shared.account_ledger import AccountLedger
from shared.runtime_state import MANIFEST_NAME, initialize_runtime_state, _json


class AccountReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "runtime"
        self.journal = initialize_runtime_state(self.state, "account-a")
        self.manifest = _json((self.state / MANIFEST_NAME).read_text())
        self.now = 2_000_000_000.0
        self.start = int((self.now - 60) * 1000)
        self.ledger = AccountLedger.initialize(
            self.journal, scope="account-a", state_id=self.manifest["state_id"],
            balance=Decimal("100"), start_ms=self.start, source="offline-fixture",
            clock=lambda: self.now,
        )
        self.balance = Decimal("100")
        self.transactions = []
        self.windows = []
        self.adapter = SimpleNamespace(
            venue="bybit", get_wallet_snapshot=self.wallet, get_transaction_log=self.transactions_for,
        )

    def wallet(self):
        return {"observed_at": self.now, "balances": {"USDT": str(self.balance)}}

    def transactions_for(self, *, start_ms, end_ms):
        self.windows.append((start_ms, end_ms))
        return tuple(row for row in self.transactions if start_ms <= int(row["transactionTime"]) <= end_ms)

    def row(self, identity="tx-1", change="10", stamp=None):
        return {"id": identity, "currency": "USDT", "type": "TRANSFER_IN",
                "transactionTime": str(stamp or self.start + 1), "change": change,
                "cashFlow": change, "funding": "0", "fee": "0", "cashBalance": "110",
                "bonusChange": "0", "category": "", "transSubType": "", "extraFees": "",
                "symbol": "", "tradeId": "", "orderId": "", "orderLinkId": "", "side": ""}

    def test_preflight_requires_existing_initialized_wallet(self):
        env = {"RISK_SCOPE": "account-a", "RUNTIME_STATE_DIR": str(self.state)}
        self.assertEqual(account_ledger_preflight(env).state_id, self.manifest["state_id"])
        other = Path(self.temp.name) / "uninitialized"
        initialize_runtime_state(other, "account-b")
        with self.assertRaises(ValueError):
            account_ledger_preflight({"RISK_SCOPE": "account-b", "RUNTIME_STATE_DIR": str(other)})

    def test_stable_wallet_and_complete_window_become_ready(self):
        result = reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        self.assertTrue(result["ready"])
        self.assertEqual(result["expected_balance"], "100")
        self.assertEqual(self.windows, [(self.start, int(self.now * 1000))])

    def test_cash_flow_is_counted_once(self):
        self.transactions.append(self.row())
        self.balance = Decimal("110")
        first = reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        second = reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        self.assertTrue(first["ready"] and second["ready"])
        self.assertEqual(second["expected_balance"], "110")
        self.assertEqual(second["transaction_count"], 1)

    def test_wallet_change_during_collection_invalidates_readiness(self):
        calls = 0

        def changing_wallet():
            nonlocal calls
            calls += 1
            return {"observed_at": self.now, "balances": {"USDT": str(100 + calls)}}

        self.adapter.get_wallet_snapshot = changing_wallet
        with self.assertRaises(ValueError):
            reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        self.assertFalse(self.ledger.check()[0])

    def test_page_failure_invalidates_previous_success_and_does_not_advance(self):
        reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        previous = self.ledger.snapshot()["verified_through_ms"]
        self.now += 1

        def unavailable(**_):
            raise TimeoutError("offline fixture")

        self.adapter.get_transaction_log = unavailable
        with self.assertRaises(TimeoutError):
            reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        status = self.ledger.snapshot()
        self.assertFalse(status["ready"])
        self.assertEqual(status["verified_through_ms"], previous)

    def test_mismatch_forces_full_replay_and_late_fact_can_recover(self):
        self.balance = Decimal("110")
        mismatch = reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        self.assertFalse(mismatch["ready"])
        self.transactions.append(self.row())
        recovered = reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        self.assertTrue(recovered["ready"])

    def test_wrong_venue_and_malformed_wallet_fail_closed(self):
        self.adapter.venue = "okx"
        with self.assertRaises(ValueError):
            reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)
        self.adapter.venue = "bybit"
        self.adapter.get_wallet_snapshot = lambda: {"observed_at": time.time(), "balances": {}}
        with self.assertRaises(ValueError):
            reconcile_bybit_wallet(self.ledger, self.adapter, clock=lambda: self.now)


if __name__ == "__main__":
    unittest.main()

"""Cash-ledger tests use identified temporary SQLite state and no network."""
from __future__ import annotations

from decimal import Decimal, localcontext
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from shared.account_ledger import AccountLedger
from shared.runtime_state import MANIFEST_NAME, initialize_runtime_state


class AccountLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state_dir = Path(self.temp.name) / "runtime"
        self.scope = "ledger-test"
        self.path = initialize_runtime_state(self.state_dir, self.scope)
        manifest = json.loads((self.state_dir / MANIFEST_NAME).read_text())
        self.state_id = manifest["state_id"]
        self.now = 2_000_000.0
        self.start = int(self.now * 1000) - 1000
        self.end = int(self.now * 1000)
        self.ledger = AccountLedger.initialize(
            self.path, scope=self.scope, state_id=self.state_id,
            balance=Decimal("100"), start_ms=self.start, source="verified statement fixture",
            clock=lambda: self.now,
        )

    def tx(self, identity: str, *, stamp: int | None = None, kind: str = "TRADE",
           cash: str = "0", funding: str = "0", fee: str = "0", **overrides):
        with localcontext() as context:
            context.prec = 512
            change = Decimal(cash) + Decimal(funding) - Decimal(fee)
        result = {
            "id": identity, "currency": "USDT", "type": kind,
            "transactionTime": stamp if stamp is not None else self.start + 100,
            "change": str(change), "cashFlow": cash, "funding": funding, "fee": fee,
            "cashBalance": "100", "bonusChange": "0",
            "category": "linear" if kind in {"TRADE", "SETTLEMENT"} else "",
            "transSubType": "", "extraFees": "", "symbol": "BTCUSDT",
            "tradeId": "trade-" + identity, "orderId": "order-" + identity,
            "orderLinkId": "link-" + identity, "side": "Buy",
        }
        result.update(overrides)
        return result

    def reconcile(self, transactions=(), *, balance="100", start: int | None = None,
                  end: int | None = None, observed_at: float | None = None):
        return self.ledger.ingest_and_reconcile(
            list(transactions), start_ms=self.start if start is None else start,
            end_ms=self.end if end is None else end, wallet_balance=Decimal(balance),
            observed_at=observed_at,
        )

    def test_initialization_is_explicit_immutable_and_restartable(self):
        self.assertEqual(self.ledger.check(), (False, "baseline_unreconciled"))
        reopened = AccountLedger(self.path, scope=self.scope, state_id=self.state_id,
                                 clock=lambda: self.now)
        self.assertEqual(reopened.snapshot()["baseline_balance"], "100")
        with self.assertRaisesRegex(ValueError, "reset is forbidden"):
            AccountLedger.initialize(
                self.path, scope=self.scope, state_id=self.state_id, balance=Decimal("0"),
                start_ms=self.start, source="attempted reset", clock=lambda: self.now,
            )

    def test_missing_journal_and_wrong_runtime_identity_are_rejected(self):
        missing = Path(self.temp.name) / "missing.sqlite"
        with self.assertRaisesRegex(ValueError, "existing order journal"):
            AccountLedger.initialize(
                missing, scope=self.scope, state_id=self.state_id, balance=Decimal("100"),
                start_ms=self.start, source="fixture", clock=lambda: self.now,
            )
        self.assertFalse(missing.exists())
        with self.assertRaisesRegex(ValueError, "identity"):
            AccountLedger(self.path, scope=self.scope, state_id="0" * 32, clock=lambda: self.now)

    def test_fee_rebate_funding_and_transfer_are_counted_once(self):
        rows = [
            self.tx("fee", fee="2"),
            self.tx("rebate", stamp=self.start + 200, fee="-1"),
            self.tx("funding", stamp=self.start + 300, kind="SETTLEMENT", funding="3"),
            self.tx("transfer", stamp=self.start + 400, kind="TRANSFER_IN", cash="10"),
        ]
        status = self.reconcile(rows, balance="112")
        self.assertTrue(status["ready"])
        self.assertEqual(status["expected_balance"], "112")
        self.assertEqual(status["transaction_count"], 4)
        self.assertEqual(status["verified_through_ms"], self.end)

    def test_duplicate_is_idempotent_but_conflict_blocks_without_new_facts(self):
        row = self.tx("same", fee="2")
        first = self.reconcile([row, dict(row)], balance="98")
        self.assertEqual(first["transaction_count"], 1)
        second = self.reconcile([row], balance="98")
        self.assertTrue(second["ready"])
        self.assertEqual(second["transaction_count"], 1)
        conflict = dict(row, fee="3", change="-3")
        with self.assertRaisesRegex(ValueError, "Conflicting durable"):
            self.reconcile([conflict], balance="97")
        status = self.ledger.snapshot()
        self.assertFalse(status["ready"])
        self.assertEqual(status["transaction_count"], 1)
        self.assertEqual(status["expected_balance"], "98")

    def test_conflict_inside_batch_rolls_back_every_fact(self):
        first = self.tx("duplicate", fee="1")
        second = dict(first, fee="2", change="-2")
        with self.assertRaisesRegex(ValueError, "within batch"):
            self.reconcile([first, self.tx("would-be-new", fee="1"), second], balance="97")
        status = self.ledger.snapshot()
        self.assertEqual(status["transaction_count"], 0)
        self.assertEqual(status["expected_balance"], "100")
        self.assertFalse(status["ready"])

    def test_mismatch_retains_valid_facts_without_advancing_watermark(self):
        first = self.tx("first", fee="2")
        status = self.reconcile([first], balance="97")
        self.assertFalse(status["ready"])
        self.assertEqual(status["reason"], "wallet_balance_mismatch")
        self.assertEqual(status["verified_through_ms"], self.start)
        self.assertEqual(status["transaction_count"], 1)
        late = self.tx("late", stamp=self.start + 50, fee="1")
        recovered = self.reconcile([late, first], balance="97")
        self.assertTrue(recovered["ready"])
        self.assertEqual(recovered["verified_through_ms"], self.end)
        self.assertEqual(recovered["transaction_count"], 2)
        reopened = AccountLedger(self.path, scope=self.scope, state_id=self.state_id,
                                 clock=lambda: self.now)
        self.assertTrue(reopened.check()[0])

    def test_delayed_fact_can_be_added_after_a_successful_watermark(self):
        earlier_end = self.end - 100
        status = self.reconcile([], balance="100", end=earlier_end)
        self.assertTrue(status["ready"])
        delayed = self.tx("delayed", stamp=earlier_end - 50, fee="1")
        status = self.reconcile([delayed], balance="99", start=self.start, end=self.end)
        self.assertTrue(status["ready"])
        self.assertEqual(status["transaction_count"], 1)
        self.assertEqual(status["expected_balance"], "99")

    def test_baseline_cutoff_is_not_double_counted_but_still_validated(self):
        baseline_fact = self.tx("included-in-baseline", stamp=self.start, cash="50", kind="TRANSFER_IN")
        status = self.reconcile([baseline_fact], balance="100")
        self.assertTrue(status["ready"])
        self.assertEqual(status["transaction_count"], 0)
        invalid = dict(baseline_fact, change="49")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.reconcile([invalid], balance="100")

    def test_unsupported_and_malformed_facts_block_without_partial_ingest(self):
        cases = [
            self.tx("currency", currency="BTC"),
            self.tx("type", type="BONUS"),
            self.tx("category", category="spot"),
            self.tx("bonus", bonusChange="1"),
            self.tx("borrow", borrowAmount="1"),
            self.tx("subtype", transSubType="MOVE_POSITION"),
            self.tx("equation", change="999"),
            self.tx("outside", transactionTime=self.end + 1),
            self.tx("not-finite", fee="NaN", change="NaN"),
        ]
        for row in cases:
            with self.subTest(row=row["id"]), self.assertRaises(ValueError):
                self.reconcile([self.tx("valid-before-" + row["id"], fee="1"), row], balance="99")
            self.assertEqual(self.ledger.snapshot()["transaction_count"], 0)

    def test_gap_stale_and_future_observations_fail_closed(self):
        self.reconcile([], balance="100")
        with self.assertRaisesRegex(ValueError, "gap"):
            self.reconcile([], balance="100", start=self.end + 1, end=self.end + 1)
        self.now += 200
        with self.assertRaisesRegex(ValueError, "stale"):
            self.reconcile([], balance="100", start=self.start, end=self.end)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.reconcile([], balance="100", start=self.start, end=int(self.now * 1000) + 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.reconcile([], balance="100", start=self.start, end=int(self.now * 1000),
                           observed_at=self.now + 1)
        self.assertFalse(self.ledger.check()[0])

    def test_invalidation_and_staleness_survive_restart(self):
        self.reconcile([], balance="100")
        self.ledger.invalidate("collector pagination failed")
        reopened = AccountLedger(self.path, scope=self.scope, state_id=self.state_id,
                                 clock=lambda: self.now)
        self.assertEqual(reopened.check(), (False, "collector pagination failed"))
        self.reconcile([], balance="100")
        self.now += 121
        self.assertEqual(self.ledger.check(), (False, "wallet_reconciliation_stale"))

    def test_failed_invalidation_write_latches_failure_in_memory(self):
        self.reconcile([], balance="100")
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TRIGGER fail_wallet_invalidation BEFORE UPDATE ON wallet_reconciliation "
                       "BEGIN SELECT RAISE(ABORT, 'injected disk failure'); END")
        with self.assertRaises(sqlite3.DatabaseError):
            self.ledger.invalidate("collector failed")
        self.assertEqual(self.ledger.check(), (False, "collector failed"))
        with sqlite3.connect(self.path) as db:
            db.execute("DROP TRIGGER fail_wallet_invalidation")
        self.reconcile([], balance="100")
        self.assertTrue(self.ledger.check()[0])

    def test_exact_decimal_precision_is_not_rounded_away(self):
        tiny = "0.00000000000000000000000000001"
        row = self.tx("tiny", kind="TRANSFER_IN", cash=tiny)
        expected = "100.00000000000000000000000000001"
        status = self.reconcile([row], balance=expected)
        self.assertTrue(status["ready"])
        self.assertEqual(status["expected_balance"], expected)

    def test_logical_payload_corruption_is_detected_on_read_and_restart(self):
        self.reconcile([self.tx("fact", fee="1")], balance="99")
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE wallet_transactions SET payload='{}' WHERE transaction_id='fact'")
        self.assertFalse(self.ledger.check()[0])
        with self.assertRaises(ValueError):
            AccountLedger(self.path, scope=self.scope, state_id=self.state_id,
                          clock=lambda: self.now)

    def test_foreign_scope_rows_are_rejected(self):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO wallet_transactions VALUES ('foreign','id',?, '{}')",
                       (self.start + 1,))
        with self.assertRaisesRegex(ValueError, "foreign"):
            AccountLedger(self.path, scope=self.scope, state_id=self.state_id,
                          clock=lambda: self.now)

    def test_symlink_journal_is_rejected(self):
        alias = Path(self.temp.name) / "orders-alias.sqlite"
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            AccountLedger(alias, scope=self.scope, state_id=self.state_id,
                          clock=lambda: self.now)


if __name__ == "__main__":
    unittest.main()

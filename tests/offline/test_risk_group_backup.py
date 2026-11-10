"""Complete portfolio backups use temporary accounts and never contact a venue."""
from __future__ import annotations

from decimal import Decimal as D
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from shared.account_ledger import AccountLedger
from shared.execution_journal import ExecutionJournal
from shared.portfolio_reservations import AccountEvidence, PortfolioReservationStore, ScopeRegistration
from shared.risk_group_backup import backup_risk_group, restore_risk_group
from shared.runtime_state import RuntimeStateLease, initialize_runtime_state, validate_runtime_state


class RiskGroupBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.accounts = {name: self.root / name for name in ("account-a", "account-b")}
        self.registrations = []
        self.wallets = {}
        now = time.time()
        for scope, directory in self.accounts.items():
            journal = initialize_runtime_state(directory, scope)
            identity = json.loads((directory / "runtime-state.json").read_text())
            self.registrations.append(ScopeRegistration(scope, "bybit", "demo", identity["state_id"]))
            ledger = AccountLedger.initialize(journal, scope=scope, state_id=identity["state_id"],
                balance=D(100), start_ms=int(now * 1000) - 1000, source="offline account statement")
            ledger.ingest_and_reconcile([], start_ms=ledger.baseline_start_ms,
                end_ms=int(time.time() * 1000), wallet_balance=D(100))
            self.wallets[scope] = ledger
        self.path = self.root / "risk.sqlite"
        self.store = PortfolioReservationStore.initialize(self.path, group="group",
            registrations=self.registrations, caps={"BTCUSDT": D(100)})
        self.store.reconcile(lambda: tuple(AccountEvidence(r.scope, r.venue, r.environment, r.state_id,
                                                          time.time(), (), (), ()) for r in self.registrations))
        self.snapshot = self.root / "snapshot"
        self.restored = self.root / "restored"

    def backup(self):
        return backup_risk_group(self.path, self.snapshot, group="group", accounts=self.accounts)

    def restore(self):
        return restore_risk_group(self.snapshot, self.restored, group="group")

    def metadata(self):
        return json.loads((self.snapshot / "risk-group.json").read_text())

    def write_metadata(self, value):
        (self.snapshot / "risk-group.json").write_text(json.dumps(value))

    def seed_unknown(self):
        r = self.registrations[0]
        self.assertTrue(self.store.reserve(scope=r.scope, venue=r.venue, environment=r.environment,
            state_id=r.state_id, intent_id="intent", symbol="BTCUSDT", side="buy",
            quantity=D(1), notional=D(60)).allowed)
        journal = ExecutionJournal(self.accounts[r.scope] / "orders.sqlite", r.scope)
        with journal.connect() as db:
            db.execute("INSERT INTO submissions(scope,intent_id,symbol,client_id,order_id,settled,"
                "side,qty,reducing,owner,parent,status,order_type) VALUES "
                "(?,'intent','BTCUSDT','client','',0,'buy','1',0,'alpha.BTCUSDT','{}','unknown','market')", (r.scope,))
            db.execute("INSERT INTO execution_fills VALUES(?,?,?,?,?,0)",
                (r.scope, "partial-fill", "intent", "alpha.BTCUSDT", '{"qty":"0.4","fill_id":"partial-fill"}'))
            db.execute("INSERT INTO execution_consumers VALUES(?,?,?)",
                (r.scope, "alpha.BTCUSDT", '{"revision":"before-fill","position":{"qty":"0"}}'))
        return journal

    def test_complete_restore_preserves_unknowns_and_requires_new_reconciliation(self):
        original = self.seed_unknown()
        before = self.store.snapshot()
        self.backup()
        self.restore()
        risk = PortfolioReservationStore(self.restored / "risk.sqlite", group="group")
        status = risk.snapshot()
        self.assertFalse(status["ready"])
        self.assertEqual(status["reason"], "restored_group_requires_reconciliation")
        self.assertEqual(status["baseline"], before["baseline"])
        self.assertEqual(status["reservations"], before["reservations"])
        self.assertEqual(status["used"], before["used"])
        for row in self.metadata()["accounts"]:
            scope = row["scope"]
            target = self.restored / "accounts" / row["slot"]
            validate_runtime_state(target, scope)
            self.assertEqual((target / "runtime-state.json").read_bytes(),
                             (self.accounts[scope] / "runtime-state.json").read_bytes())
            ledger = AccountLedger(target / "orders.sqlite", scope=scope, state_id=row["state_id"])
            self.assertEqual(ledger.check(), (False, "restored_group_requires_reconciliation"))
            self.assertEqual(ledger.baseline_balance, D(100))
            self.assertEqual(ledger.baseline_start_ms, self.wallets[scope].baseline_start_ms)
            if scope == "account-a":
                journal = ExecutionJournal(target / "orders.sqlite", scope)
                self.assertEqual(journal.orders(), original.orders())
                self.assertEqual(journal.fills("alpha.BTCUSDT"), original.fills("alpha.BTCUSDT"))
                self.assertEqual(journal.checkpoint("alpha.BTCUSDT"), original.checkpoint("alpha.BTCUSDT"))
        self.assertTrue(self.store.snapshot()["ready"])
        self.assertTrue(all(ledger.check()[0] for ledger in self.wallets.values()))

    def test_committed_wal_and_extra_files_are_included_for_all_accounts(self):
        writers = []
        try:
            for scope, directory in self.accounts.items():
                (directory / "empty").mkdir()
                (directory / "observer.json").write_text('{"counter":42}')
                db = sqlite3.connect(directory / "extra.db")
                writers.append(db)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA wal_autocheckpoint=0")
                db.execute("CREATE TABLE facts(value TEXT)")
                db.execute("INSERT INTO facts VALUES (?)", (scope,))
                db.commit()
                self.assertGreater((directory / "extra.db-wal").stat().st_size, 0)
            self.backup()
            self.restore()
            for row in self.metadata()["accounts"]:
                target = self.restored / "accounts" / row["slot"]
                with sqlite3.connect(target / "extra.db") as db:
                    self.assertEqual(db.execute("SELECT value FROM facts").fetchall(), [(row["scope"],)])
                self.assertEqual((target / "observer.json").read_text(), '{"counter":42}')
                self.assertTrue((target / "empty").is_dir())
        finally:
            for db in writers:
                db.close()

    def test_partial_or_wrong_group_and_state_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "every"):
            backup_risk_group(
                self.path, self.snapshot, group="group", accounts={"account-a": self.accounts["account-a"]}
            )
        with self.assertRaises(ValueError):
            backup_risk_group(self.path, self.snapshot, group="wrong", accounts=self.accounts)
        replacement = self.root / "replacement"
        initialize_runtime_state(replacement, "account-a")
        with self.assertRaisesRegex(ValueError, "identity"):
            backup_risk_group(self.path, self.snapshot, group="group",
                              accounts={**self.accounts, "account-a": replacement})
        self.assertFalse(self.snapshot.exists())

    def test_active_account_writer_blocks_without_releasing_reservations(self):
        self.seed_unknown()
        before = self.store.snapshot()["reservations"]
        with RuntimeStateLease(self.accounts["account-a"], "account-a"):
            with self.assertRaises(RuntimeError):
                self.backup()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(self.store.snapshot()["reservations"], before)

    def test_shared_risk_and_nested_account_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            backup_risk_group(self.accounts["account-a"] / "orders.sqlite", self.snapshot,
                              group="group", accounts=self.accounts)
        with self.assertRaisesRegex(ValueError, "overlap"):
            backup_risk_group(self.path, self.snapshot, group="group",
                accounts={"account-a": self.accounts["account-a"],
                          "account-b": self.accounts["account-a"] / "nested"})

    def test_existing_destination_is_never_overwritten(self):
        self.backup()
        original = (self.snapshot / "risk-group.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.backup()
        self.assertEqual((self.snapshot / "risk-group.json").read_bytes(), original)
        self.restored.mkdir()
        (self.restored / "keep").write_text("original")
        with self.assertRaises(FileExistsError):
            self.restore()
        self.assertEqual((self.restored / "keep").read_text(), "original")

    def test_backup_failure_publishes_nothing_and_cleans_staging(self):
        with patch("shared.risk_group_backup._sqlite_snapshot", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.backup()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(list(self.root.glob(".snapshot*")), [])

    def test_restore_invalidation_failure_publishes_nothing(self):
        self.backup()
        with patch("shared.risk_group_backup._invalidate_wallet", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.restore()
        self.assertFalse(self.restored.exists())
        self.assertEqual(list(self.root.glob(".restored*")), [])

    def test_missing_account_tampered_bytes_and_boolean_proof_are_rejected(self):
        self.backup()
        meta = self.metadata()
        meta["accounts"].pop()
        self.write_metadata(meta)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.restore()
        self.assertFalse(self.restored.exists())

    def test_checksum_change_in_risk_file_is_detected(self):
        self.backup()
        db = sqlite3.connect(self.snapshot / "payload" / "risk.sqlite")
        try:
            db.execute("UPDATE portfolio_metadata SET value='wrong' WHERE key='group'")
            db.commit()
        finally:
            db.close()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.restore()

    def test_duplicate_account_manifest_cannot_substitute_coverage(self):
        self.backup()
        meta = self.metadata()
        meta["accounts"][1]["scope"] = meta["accounts"][0]["scope"]
        self.write_metadata(meta)
        with self.assertRaises(ValueError):
            self.restore()

    def test_restore_rechecks_risk_bytes_after_copy(self):
        self.backup()
        from shared.risk_group_backup import shutil
        real_copy = shutil.copyfile

        def altered(source, target):
            result = real_copy(source, target)
            if Path(target).name == "risk.sqlite":
                db = sqlite3.connect(target)
                try:
                    db.execute("UPDATE portfolio_metadata SET value='wrong' WHERE key='group'")
                    db.commit()
                finally:
                    db.close()
            return result

        with patch("shared.risk_group_backup.shutil.copyfile", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.restore()
        self.assertFalse(self.restored.exists())

    def test_sensitive_symlink_and_orphan_sidecar_are_rejected(self):
        state = self.accounts["account-a"]
        secret = state / ".env"
        secret.write_text("DUMMY_SETTING=fixture")
        with self.assertRaisesRegex(ValueError, "credential"):
            self.backup()
        secret.unlink()
        link = state / "outside"
        link.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.backup()
        link.unlink()
        (state / "lost.sqlite-wal").write_bytes(b"orphan")
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.backup()


if __name__ == "__main__":
    unittest.main()

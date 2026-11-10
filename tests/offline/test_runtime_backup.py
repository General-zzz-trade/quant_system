"""Local runtime snapshots retain WAL facts and reject incomplete recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import time
from decimal import Decimal
from unittest.mock import patch

from shared.execution_journal import ExecutionJournal
from shared.runtime_backup import backup_state, restore_state
from shared.runtime_state import RuntimeStateLease, initialize_runtime_state, validate_runtime_state


class RuntimeBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "runtime"
        self.snapshot = self.root / "snapshot"
        self.restored = self.root / "restored"
        self.scope = "paper-bybit"
        initialize_runtime_state(self.state, self.scope)
        self.journal = ExecutionJournal(self.state / "orders.sqlite", self.scope)
        with self.journal.connect() as db:
            db.execute("INSERT INTO submissions(scope,intent_id,symbol,client_id,order_id,settled,"
                       "side,qty,reducing,owner,parent,status,order_type) VALUES "
                       "(?,?,?,'fixed-client','',0,'buy','1',0,'alpha.BTCUSDT','{}','unknown','market')",
                       (self.scope, "intent-1", "BTCUSDT"))
            db.execute("INSERT INTO execution_fills VALUES(?,?,?,?,?,0)",
                       (self.scope, "fill-1", "intent-1", "alpha.BTCUSDT",
                        json.dumps({"fill_id": "fill-1", "qty": "0.4", "price": "100"})))
            db.execute("INSERT INTO execution_consumers VALUES(?,?,?)",
                       (self.scope, "alpha.BTCUSDT", '{"revision":"before-fill","qty":"0"}'))
        (self.state / "checkpoints").mkdir()
        (self.state / "checkpoints" / "model.json").write_text('{"observations":42}')
        (self.state / "empty-checkpoints").mkdir()

    def backup(self):
        return backup_state(self.state, self.snapshot, self.scope)

    def metadata(self):
        return json.loads((self.snapshot / "snapshot.json").read_text())

    def write_metadata(self, value):
        (self.snapshot / "snapshot.json").write_text(json.dumps(value))

    def test_restore_retains_identity_unknown_order_outbox_and_checkpoint(self):
        self.backup()
        restore_state(self.snapshot, self.restored, self.scope)
        validate_runtime_state(self.restored, self.scope)
        self.assertEqual((self.state / "runtime-state.json").read_bytes(),
                         (self.restored / "runtime-state.json").read_bytes())
        journal = ExecutionJournal(self.restored / "orders.sqlite", self.scope)
        self.assertEqual(journal.orders(unresolved=True), self.journal.orders(unresolved=True))
        self.assertEqual(journal.fills("alpha.BTCUSDT"), self.journal.fills("alpha.BTCUSDT"))
        self.assertEqual(journal.checkpoint("alpha.BTCUSDT"), self.journal.checkpoint("alpha.BTCUSDT"))
        applied = []

        def apply():
            applied.append("fill-1")
            return {"revision": "after-fill", "qty": "0.4"}

        self.assertTrue(journal.consume("alpha.BTCUSDT", "fill-1", apply, applied.clear,
                                        expected_revision="before-fill"))
        self.assertFalse(journal.consume("alpha.BTCUSDT", "fill-1", apply, applied.clear,
                                         expected_revision="after-fill"))
        self.assertEqual(applied, ["fill-1"])
        self.assertEqual((self.restored / "checkpoints/model.json").read_text(), '{"observations":42}')
        self.assertTrue((self.restored / "empty-checkpoints").is_dir())
        names = [entry["path"] for entry in self.metadata()["files"]]
        self.assertFalse(any(name.endswith((".lock", "-wal", "-shm")) for name in names))

    def test_committed_wal_records_in_every_database_survive(self):
        writers = []
        try:
            for name in ("drawdown.sqlite", "state.db", "checkpoint_without_extension"):
                db = sqlite3.connect(self.state / name)
                writers.append(db)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA wal_autocheckpoint=0")
                db.execute("CREATE TABLE facts(value TEXT)")
                db.execute("INSERT INTO facts VALUES('committed-in-wal')")
                db.commit()
                self.assertGreater((self.state / (name + "-wal")).stat().st_size, 0)
            self.backup()
            restore_state(self.snapshot, self.restored, self.scope)
            for name in ("drawdown.sqlite", "state.db", "checkpoint_without_extension"):
                db = sqlite3.connect(self.restored / name)
                try:
                    self.assertEqual(db.execute("SELECT value FROM facts").fetchall(), [("committed-in-wal",)])
                    self.assertEqual(db.execute("PRAGMA integrity_check").fetchall(), [("ok",)])
                finally:
                    db.close()
        finally:
            for db in writers:
                db.close()

    def test_backup_refuses_active_execution_writer(self):
        with RuntimeStateLease(self.state, self.scope):
            with self.assertRaises(RuntimeError):
                self.backup()
        self.assertFalse(self.snapshot.exists())

    def test_restore_preserves_wallet_facts_but_revokes_snapshot_readiness(self):
        from shared.account_ledger import AccountLedger
        identity = json.loads((self.state / "runtime-state.json").read_text())["state_id"]
        now = time.time()
        stamp = int(now * 1000)
        original = AccountLedger.initialize(self.state / "orders.sqlite", scope=self.scope, state_id=identity,
                                            balance=Decimal(100), start_ms=stamp - 1000, source="fixture")
        original.ingest_and_reconcile([], start_ms=stamp - 1000, end_ms=stamp, wallet_balance=Decimal(100))
        self.assertTrue(original.check()[0])
        self.backup()
        restore_state(self.snapshot, self.restored, self.scope)
        restored = AccountLedger(self.restored / "orders.sqlite", scope=self.scope, state_id=identity)
        self.assertEqual(restored.baseline_balance, Decimal(100))
        self.assertEqual(restored.check(), (False, "restored_account_requires_reconciliation"))
        self.assertTrue(original.check()[0])

    def test_backup_refuses_wrong_scope_and_missing_journal(self):
        with self.assertRaises(ValueError):
            backup_state(self.state, self.snapshot, "another-account")
        (self.state / "orders.sqlite").unlink()
        with self.assertRaises(ValueError):
            self.backup()
        self.assertFalse((self.state / "orders.sqlite").exists())

    def test_backup_refuses_corrupt_auxiliary_database_and_cleans_partial_output(self):
        (self.state / "drawdown.sqlite").write_bytes(b"broken database")
        with self.assertRaises(sqlite3.DatabaseError):
            self.backup()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(list(self.root.glob(".snapshot*")), [])

    def test_backup_refuses_credential_files(self):
        (self.state / ".env").write_text("DUMMY_SETTING=not-a-credential")
        with self.assertRaisesRegex(ValueError, "credential"):
            self.backup()
        self.assertFalse(self.snapshot.exists())

    def test_backup_refuses_symlink_and_orphan_wal(self):
        link = self.state / "outside"
        link.symlink_to(self.root)
        with self.assertRaises(ValueError):
            self.backup()
        link.unlink()
        (self.state / "missing.sqlite-wal").write_bytes(b"orphan")
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.backup()

    def test_destination_cannot_overwrite_or_be_inside_state(self):
        self.backup()
        sentinel = (self.snapshot / "snapshot.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.backup()
        self.assertEqual((self.snapshot / "snapshot.json").read_bytes(), sentinel)
        with self.assertRaises(ValueError):
            backup_state(self.state, self.state / "snapshot", self.scope)
        self.restored.mkdir()
        (self.restored / "keep").write_text("original")
        with self.assertRaises(FileExistsError):
            restore_state(self.snapshot, self.restored, self.scope)
        self.assertEqual((self.restored / "keep").read_text(), "original")

    def test_restore_refuses_wrong_scope(self):
        self.backup()
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, "other-account")
        self.assertFalse(self.restored.exists())

    def test_restore_refuses_missing_database_even_if_removed_from_inventory(self):
        self.backup()
        (self.snapshot / "state/orders.sqlite").unlink()
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, self.scope)
        meta = self.metadata()
        meta["files"] = [entry for entry in meta["files"] if entry["path"] != "orders.sqlite"]
        self.write_metadata(meta)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            restore_state(self.snapshot, self.restored, self.scope)

    def test_restore_refuses_corrupt_and_rehashed_database(self):
        self.backup()
        database = self.snapshot / "state/orders.sqlite"
        database.write_bytes(b"corrupted snapshot")
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, self.scope)
        meta = self.metadata()
        for entry in meta["files"]:
            if entry["path"] == "orders.sqlite":
                entry["size"] = database.stat().st_size
                entry["sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
        self.write_metadata(meta)
        with self.assertRaises(sqlite3.DatabaseError):
            restore_state(self.snapshot, self.restored, self.scope)
        self.assertFalse(self.restored.exists())

    def test_restore_refuses_missing_drawdown_even_if_removed_from_inventory(self):
        self.backup()
        (self.snapshot / "state/drawdown.sqlite").unlink()
        meta = self.metadata()
        meta["files"] = [entry for entry in meta["files"] if entry["path"] != "drawdown.sqlite"]
        self.write_metadata(meta)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            restore_state(self.snapshot, self.restored, self.scope)
        self.assertFalse(self.restored.exists())

    def test_restore_rejects_path_traversal_absolute_and_duplicate_members(self):
        self.backup()
        original = self.metadata()
        for name in ("../escape", "/absolute", "state/../escape", "nested\\escape"):
            with self.subTest(name=name):
                meta = json.loads(json.dumps(original))
                meta["files"][0]["path"] = name
                self.write_metadata(meta)
                with self.assertRaises(ValueError):
                    restore_state(self.snapshot, self.restored, self.scope)
        original["files"].append(original["files"][0])
        self.write_metadata(original)
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, self.scope)
        self.assertFalse(self.restored.exists())

    def test_restore_rejects_symlinks_and_unlisted_files_and_empty_directories(self):
        self.backup()
        extra = self.snapshot / "state" / "extra"
        extra.symlink_to(self.state / "orders.sqlite")
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, self.scope)
        extra.unlink()
        extra.write_text("unexpected")
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, self.scope)
        extra.unlink()
        (self.snapshot / "extra-directory").mkdir()
        with self.assertRaises(ValueError):
            restore_state(self.snapshot, self.restored, self.scope)

    def test_restore_rejects_changed_database_identity_after_rehash(self):
        self.backup()
        database = self.snapshot / "state/orders.sqlite"
        db = sqlite3.connect(database)
        db.execute("UPDATE runtime_identity SET state_id=?", ("0" * 32,))
        db.commit()
        db.close()
        meta = self.metadata()
        for entry in meta["files"]:
            if entry["path"] == "orders.sqlite":
                entry["size"] = database.stat().st_size
                entry["sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
        self.write_metadata(meta)
        with self.assertRaisesRegex(ValueError, "identity"):
            restore_state(self.snapshot, self.restored, self.scope)
        self.assertFalse(self.restored.exists())
        self.assertEqual(list(self.root.glob(".restored*")), [])

    def test_failed_copy_does_not_publish_partial_snapshot(self):
        with patch("shared.runtime_backup.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.backup()
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(list(self.root.glob(".snapshot*")), [])
        self.backup()  # Failed operation released its lease and destination lock.

    def test_cli_returns_nonzero_for_incomplete_snapshot(self):
        self.backup()
        (self.snapshot / "state/orders.sqlite").unlink()
        result = subprocess.run(
            [sys.executable, "-m", "scripts.runtime_state_backup", "restore", "--snapshot",
             str(self.snapshot), "--destination", str(self.restored), "--scope", self.scope],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], timeout=10,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("blocked", result.stderr)
        self.assertFalse(self.restored.exists())


if __name__ == "__main__":
    unittest.main()

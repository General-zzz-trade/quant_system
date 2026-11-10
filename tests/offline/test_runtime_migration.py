"""Migration safety against disposable account fixtures, with no account I/O."""
from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from shared.account_ledger import AccountLedger
from shared.drawdown_guard import DrawdownGuard
from shared.execution_journal import ExecutionJournal
from shared.runtime_migration import MIGRATION_NAME, inspect_migration, migrate_runtime_state
from shared.runtime_state import MANIFEST_NAME, RuntimeStateLease, initialize_runtime_state, validate_runtime_state


class RuntimeMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.target = self.root / "target"
        self.scope = "migration-demo"
        self.journal_path = initialize_runtime_state(self.source, self.scope)
        self.identity = json.loads((self.source / MANIFEST_NAME).read_text())
        self.journal = ExecutionJournal(self.journal_path, self.scope)
        with self.journal.connect() as db:
            db.execute("INSERT INTO submissions(scope,intent_id,symbol,client_id,order_id,settled,"
                       "side,qty,reducing,owner,parent,status,order_type) VALUES "
                       "(?, 'unknown-1','BTCUSDT','fixed-client','',0,'buy','1',0,"
                       "'alpha.BTCUSDT','{}','unknown','market')", (self.scope,))
            db.execute("INSERT INTO execution_fills VALUES(?,?,?,?,?,0)",
                       (self.scope, "fill-1", "unknown-1", "alpha.BTCUSDT", '{"qty":"0.4","price":"100"}'))
            db.execute("INSERT INTO execution_consumers VALUES(?,?,?)",
                       (self.scope, "alpha.BTCUSDT", '{"revision":"prior","position":{"qty":"0"}}'))
        self.now = time.time()
        self.ledger = AccountLedger.initialize(
            self.journal_path, scope=self.scope, state_id=self.identity["state_id"], balance=Decimal("100"),
            start_ms=int((self.now - 10) * 1000), source="offline fixture", clock=lambda: self.now,
        )
        self.ledger.ingest_and_reconcile([], start_ms=int((self.now - 10) * 1000),
                                         end_ms=int(self.now * 1000), wallet_balance=Decimal("100"))
        guard = DrawdownGuard(self.source / "drawdown.sqlite", self.scope)
        guard.observe(100, now=self.now)
        guard.observe(90, now=self.now + 1)
        (self.source / "checkpoints").mkdir()
        (self.source / "checkpoints" / "model.json").write_text('{"count":42}')
        (self.source / "empty").mkdir()

    @staticmethod
    def dump(path):
        db = sqlite3.connect(path)
        try:
            return tuple(db.iterdump())
        finally:
            db.close()

    def migrate(self):
        return migrate_runtime_state(self.source, self.target, self.scope)

    def test_managed_state_preserves_identity_facts_and_source(self):
        before_orders = self.dump(self.journal_path)
        before_drawdown = self.dump(self.source / "drawdown.sqlite")
        result = self.migrate()
        self.assertFalse(result.execution_ready)
        self.assertEqual(result.state_id, self.identity["state_id"])
        self.assertEqual(result.source_kind, "managed")
        validate_runtime_state(self.target, self.scope)
        restored = ExecutionJournal(self.target / "orders.sqlite", self.scope)
        self.assertEqual(restored.orders(), self.journal.orders())
        self.assertEqual(restored.fills("alpha.BTCUSDT"), self.journal.fills("alpha.BTCUSDT"))
        self.assertEqual(restored.checkpoint("alpha.BTCUSDT"), self.journal.checkpoint("alpha.BTCUSDT"))
        self.assertEqual(self.dump(self.source / "drawdown.sqlite"), before_drawdown)
        self.assertEqual(self.dump(self.target / "drawdown.sqlite"), before_drawdown)
        self.assertEqual(self.dump(self.journal_path), before_orders)
        self.assertTrue(self.ledger.check()[0])
        moved = AccountLedger(self.target / "orders.sqlite", scope=self.scope, state_id=result.state_id)
        self.assertFalse(moved.check()[0])
        self.assertEqual(moved.snapshot()["baseline_balance"], "100")
        self.assertEqual((self.target / "checkpoints/model.json").read_text(), '{"count":42}')
        self.assertTrue((self.target / "empty").is_dir())
        metadata = inspect_migration(self.target, self.scope)
        self.assertEqual(metadata["facts"]["unresolved_submissions"], 1)
        self.assertEqual(metadata["facts"]["unconsumed_fills"], 1)
        self.assertFalse(metadata["risk_group_verified"])

    def test_wal_only_commits_survive_for_order_and_auxiliary_databases(self):
        connections = []
        try:
            for name in ("orders.sqlite", "checkpoint_without_extension"):
                db = sqlite3.connect(self.source / name)
                connections.append(db)
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA wal_autocheckpoint=0")
                db.execute("CREATE TABLE extra_facts(value TEXT)")
                db.execute("INSERT INTO extra_facts VALUES('committed-in-wal')")
                db.commit()
                self.assertGreater((self.source / (name + "-wal")).stat().st_size, 0)
            self.migrate()
            for name in ("orders.sqlite", "checkpoint_without_extension"):
                with sqlite3.connect(self.target / name) as db:
                    self.assertEqual(db.execute("SELECT * FROM extra_facts").fetchall(), [("committed-in-wal",)])
            inspect_migration(self.target, self.scope)
        finally:
            for db in connections:
                db.close()

    def test_inspection_rejects_committed_wal_update_with_unchanged_row_count(self):
        source = sqlite3.connect(self.source / "auxiliary.sqlite")
        try:
            source.execute("PRAGMA journal_mode=WAL")
            source.execute("CREATE TABLE facts(value TEXT)")
            source.execute("INSERT INTO facts VALUES('original')")
            source.commit()
        finally:
            source.close()
        self.migrate()
        inspect_migration(self.target, self.scope)
        db = sqlite3.connect(self.target / "auxiliary.sqlite")
        try:
            db.execute("PRAGMA wal_autocheckpoint=0")
            db.execute("UPDATE facts SET value='changed'")
            db.commit()
            self.assertGreater((self.target / "auxiliary.sqlite-wal").stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "uncheckpointed"):
                inspect_migration(self.target, self.scope)
        finally:
            db.close()

    def legacy(self):
        source = self.root / "legacy"
        source.mkdir()
        with sqlite3.connect(source / "orders.sqlite") as db:
            db.execute("CREATE TABLE submissions(scope TEXT,intent_id TEXT,symbol TEXT,client_id TEXT,"
                       "order_id TEXT,settled INTEGER,PRIMARY KEY(scope,intent_id))")
            db.execute("INSERT INTO submissions VALUES(?,'legacy-unknown','BTCUSDT','fixed','',0)", (self.scope,))
        DrawdownGuard(source / "drawdown.sqlite", self.scope).observe(100, now=self.now)
        return source

    def test_legacy_adoption_changes_target_only_and_keeps_unknown_blocking(self):
        source = self.legacy()
        before = self.dump(source / "orders.sqlite")
        result = migrate_runtime_state(source, self.target, self.scope)
        self.assertEqual(result.source_kind, "legacy")
        self.assertFalse((source / MANIFEST_NAME).exists())
        self.assertEqual(self.dump(source / "orders.sqlite"), before)
        row = ExecutionJournal(self.target / "orders.sqlite", self.scope).orders()[0]
        self.assertEqual((row["client_id"], row["settled"], row["status"], row["owner"]),
                         ("fixed", 0, "unknown", ""))
        self.assertIsNone(row["side"])
        metadata = inspect_migration(self.target, self.scope)
        self.assertEqual(metadata["facts"]["unknown_submissions"], 1)
        self.assertEqual(metadata["facts"]["wallet_tables"], [])

    def test_legacy_missing_drawdown_tables_are_not_recreated(self):
        source = self.legacy()
        with sqlite3.connect(source / "drawdown.sqlite") as db:
            db.execute("DROP TABLE drawdown_periods")
        with self.assertRaisesRegex(ValueError, "missing drawdown_periods"):
            migrate_runtime_state(source, self.target, self.scope)
        self.assertFalse(self.target.exists())

    def test_wrong_scope_and_partial_identity_are_rejected(self):
        with self.assertRaises(ValueError):
            migrate_runtime_state(self.source, self.target, "foreign")
        source = self.legacy()
        with sqlite3.connect(source / "orders.sqlite") as db:
            db.execute("CREATE TABLE runtime_identity(singleton INTEGER,state_id TEXT,scope TEXT)")
        with self.assertRaisesRegex(ValueError, "Partial runtime identity"):
            migrate_runtime_state(source, self.target, self.scope)
        self.assertFalse(self.target.exists())

    def test_partial_wallet_or_corrupt_checkpoint_is_rejected(self):
        with sqlite3.connect(self.journal_path) as db:
            db.execute("DROP TABLE wallet_transactions")
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.target.exists())

    def test_corrupt_checkpoint_is_rejected(self):
        with sqlite3.connect(self.journal_path) as db:
            db.execute("UPDATE execution_consumers SET checkpoint='broken JSON'")
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse(self.target.exists())

    def test_foreign_scope_in_auxiliary_database_is_rejected(self):
        with sqlite3.connect(self.source / "extra.sqlite") as db:
            db.execute("CREATE TABLE extra(scope TEXT)")
            db.execute("INSERT INTO extra VALUES('foreign')")
        with self.assertRaisesRegex(ValueError, "foreign"):
            self.migrate()

    def test_embedded_shared_risk_database_requires_group_backup(self):
        with sqlite3.connect(self.source / "risk.sqlite") as db:
            db.execute("CREATE TABLE portfolio_metadata(key TEXT,value TEXT)")
        with self.assertRaisesRegex(ValueError, "coordinated group backup"):
            self.migrate()

    def test_active_writer_is_rejected(self):
        with RuntimeStateLease(self.source, self.scope):
            with self.assertRaises(RuntimeError):
                self.migrate()
        self.assertFalse(self.target.exists())

    def test_existing_or_nested_target_is_rejected(self):
        self.target.mkdir()
        sentinel = self.target / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.migrate()
        self.assertEqual(sentinel.read_text(), "keep")
        with self.assertRaises(ValueError):
            migrate_runtime_state(self.source, self.source / "nested", self.scope)

    def test_symlink_sensitive_and_special_files_are_rejected(self):
        for name in (".env", "secret.pem"):
            path = self.source / name
            path.write_text("fixture")
            with self.assertRaises(ValueError):
                self.migrate()
            path.unlink()
        alias = self.source / "alias"
        alias.symlink_to(self.root)
        with self.assertRaises(ValueError):
            self.migrate()
        alias.unlink()
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.source / "pipe")
            with self.assertRaises(ValueError):
                self.migrate()
        self.assertFalse(self.target.exists())

    def test_failure_before_publish_leaves_no_target_or_stage(self):
        with patch("shared.runtime_migration._metadata", side_effect=RuntimeError("injected crash")):
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                self.migrate()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob(".target.*")), [])
        self.assertEqual(self.journal.orders()[0]["status"], "unknown")

    def test_inspect_detects_changed_file_and_rejects_inventory_traversal(self):
        self.migrate()
        path = self.target / "checkpoints/model.json"
        path.write_text('{"count":43}')
        with self.assertRaisesRegex(ValueError, "checksum"):
            inspect_migration(self.target, self.scope)
        path.write_text('{"count":42}')
        provenance = self.target / MIGRATION_NAME
        metadata = json.loads(provenance.read_text())
        metadata["files"][0]["path"] = "../escape"
        provenance.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            inspect_migration(self.target, self.scope)

    def test_cli_migrates_and_inspects_only_temporary_state(self):
        result = subprocess.run([sys.executable, "-m", "scripts.migrate_runtime_state", "migrate",
                                 "--source-dir", str(self.source), "--target-dir", str(self.target),
                                 "--scope", self.scope], capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["execution_ready"])
        inspected = subprocess.run([sys.executable, "-m", "scripts.migrate_runtime_state", "inspect",
                                    "--target-dir", str(self.target), "--scope", self.scope],
                                   capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(inspected.stdout)["source_kind"], "managed")


if __name__ == "__main__":
    unittest.main()

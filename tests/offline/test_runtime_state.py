"""Runtime identity and process exclusion tests using only temporary local state."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shared.execution_journal import ExecutionJournal
from shared.runtime_state import (
    JOURNAL_NAME,
    MANIFEST_NAME,
    RuntimeStateLease,
    initialize_runtime_state,
    validate_runtime_state,
)

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "paper-recovery-test"


class RuntimeStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "runtime"
        self.path = initialize_runtime_state(self.directory, SCOPE)
        self.drawdown = self.directory / "drawdown.sqlite"

    def connect(self):
        db = sqlite3.connect(self.path)
        self.addCleanup(db.close)
        return db

    def test_initialization_binds_database_identity_and_passes_check(self):
        manifest = json.loads((self.directory / MANIFEST_NAME).read_text())
        with self.connect() as db:
            identity = db.execute("SELECT state_id,scope FROM runtime_identity").fetchone()
        self.assertEqual(identity, (manifest["state_id"], SCOPE))
        self.assertEqual(manifest["databases"], [JOURNAL_NAME, "drawdown.sqlite"])
        db = sqlite3.connect(self.drawdown)
        try:
            self.assertEqual(db.execute("SELECT state_id,scope FROM runtime_identity").fetchone(), identity)
        finally:
            db.close()
        with RuntimeStateLease(self.directory, SCOPE) as lease:
            self.assertEqual(lease.journal_path, self.path)

    def test_reinitialization_never_overwrites_existing_state(self):
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "empty directory"):
            initialize_runtime_state(self.directory, SCOPE)
        self.assertEqual(self.path.read_bytes(), before)
        foreign = Path(self.temp.name) / "foreign"
        foreign.mkdir()
        sentinel = foreign / "unknown.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty directory"):
            initialize_runtime_state(foreign, SCOPE)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse((foreign / JOURNAL_NAME).exists())

    def test_deleted_order_database_is_rejected_without_recreation(self):
        journal = ExecutionJournal(self.path, SCOPE)
        with journal.connect() as db:
            db.execute("INSERT INTO submissions (scope,intent_id,symbol,client_id,order_id,settled) "
                       "VALUES (?, 'intent', 'BTCUSDT', 'fixed', '', 0)", (SCOPE,))
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "refusing an empty replacement"):
            RuntimeStateLease(self.directory, SCOPE)
        self.assertFalse(self.path.exists())

    def test_missing_manifest_or_entire_directory_is_rejected(self):
        (self.directory / MANIFEST_NAME).unlink()
        with self.assertRaises(ValueError):
            validate_runtime_state(self.directory, SCOPE)
        missing = Path(self.temp.name) / "missing"
        with self.assertRaises(ValueError):
            RuntimeStateLease(missing, SCOPE)
        self.assertFalse(missing.exists())

    def test_deleted_drawdown_database_is_rejected_without_recreation(self):
        self.drawdown.unlink()
        with self.assertRaisesRegex(ValueError, "drawdown.sqlite"):
            RuntimeStateLease(self.directory, SCOPE)
        self.assertFalse(self.drawdown.exists())
        self.assertTrue(self.path.is_file())

    def test_corrupt_drawdown_database_blocks_execution(self):
        self.drawdown.write_bytes(b"corrupt drawdown state")
        with self.assertRaises(sqlite3.DatabaseError):
            RuntimeStateLease(self.directory, SCOPE)
        self.assertEqual(self.drawdown.read_bytes(), b"corrupt drawdown state")

    def test_drawdown_foreign_scope_in_identity_or_period_blocks_execution(self):
        for target in ("identity", "period"):
            with self.subTest(target=target):
                db = sqlite3.connect(self.drawdown)
                try:
                    if target == "identity":
                        db.execute("UPDATE runtime_identity SET scope='foreign'")
                    else:
                        db.execute("INSERT INTO drawdown_periods VALUES "
                                   "('foreign','day','2026-11-07',1000,5,1)")
                    db.commit()
                    with self.assertRaisesRegex(ValueError, "identity differs|different account scope"):
                        RuntimeStateLease(self.directory, SCOPE)
                finally:
                    db.execute("UPDATE runtime_identity SET scope=?", (SCOPE,))
                    db.execute("DELETE FROM drawdown_periods WHERE scope='foreign'")
                    db.commit()
                    db.close()

    def test_drawdown_from_another_state_with_same_scope_is_rejected(self):
        second = Path(self.temp.name) / "second"
        initialize_runtime_state(second, SCOPE)
        self.drawdown.write_bytes((second / "drawdown.sqlite").read_bytes())
        with self.assertRaisesRegex(ValueError, "drawdown.sqlite identity differs"):
            RuntimeStateLease(self.directory, SCOPE)

    def test_read_only_drawdown_database_blocks_execution(self):
        mode = self.drawdown.stat().st_mode & 0o777
        self.drawdown.chmod(0o400)
        try:
            with self.assertRaises((ValueError, PermissionError)):
                RuntimeStateLease(self.directory, SCOPE)
            self.assertEqual(validate_runtime_state(self.directory, SCOPE, require_writable=False), self.path)
        finally:
            self.drawdown.chmod(mode)

    def test_scope_and_manifest_journal_must_match(self):
        for scope in ("another-account", "", " paper-recovery-test", "paper-recovery-test "):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                validate_runtime_state(self.directory, scope)
        manifest_path = self.directory / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["journal"] = "../orders.sqlite"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            validate_runtime_state(self.directory, SCOPE)

    def test_journal_override_cannot_escape_or_select_a_second_database(self):
        for path in (self.directory / "other.sqlite", self.directory / ".." / "orders.sqlite"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                RuntimeStateLease(self.directory, SCOPE, journal_path=path)

    def test_symlink_directory_database_manifest_and_wal_are_rejected(self):
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            validate_runtime_state(alias, SCOPE)
        for name in (JOURNAL_NAME, MANIFEST_NAME, JOURNAL_NAME + "-wal"):
            with self.subTest(name=name):
                target = self.directory / name
                saved = Path(self.temp.name) / (name + ".saved")
                existed = target.exists()
                if existed:
                    target.rename(saved)
                else:
                    saved.write_bytes(b"")
                target.symlink_to(saved)
                try:
                    with self.assertRaisesRegex(ValueError, "symbolic links"):
                        RuntimeStateLease(self.directory, SCOPE)
                finally:
                    target.unlink()
                    if existed:
                        saved.rename(target)

    def test_corrupt_database_and_incomplete_schema_block_startup(self):
        self.path.write_bytes(b"not a SQLite database")
        with self.assertRaises(sqlite3.DatabaseError):
            RuntimeStateLease(self.directory, SCOPE)
        self.path.unlink()
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE unrelated (value TEXT)")
            db.execute("PRAGMA journal_mode=WAL")
        with self.assertRaises((sqlite3.DatabaseError, ValueError)):
            RuntimeStateLease(self.directory, SCOPE)

    def test_database_identity_cannot_be_replaced_with_another_empty_journal(self):
        second = initialize_runtime_state(Path(self.temp.name) / "second", SCOPE)
        self.path.write_bytes(second.read_bytes())
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_runtime_state(self.directory, SCOPE)

    def test_foreign_scope_in_any_execution_table_blocks_startup(self):
        statements = (
            "INSERT INTO submissions (scope,intent_id) VALUES ('foreign','intent')",
            "INSERT INTO execution_fills VALUES ('foreign','fill','intent','owner','{}',0)",
            "INSERT INTO execution_consumers VALUES ('foreign','owner','{}')",
        )
        for index, statement in enumerate(statements):
            with self.subTest(table=index):
                with self.connect() as db:
                    db.execute(statement)
                with self.assertRaisesRegex(ValueError, "different account scope"):
                    validate_runtime_state(self.directory, SCOPE)
                with self.connect() as db:
                    for table in ("submissions", "execution_fills", "execution_consumers"):
                        db.execute(f"DELETE FROM {table} WHERE scope='foreign'")

    def test_validation_reads_committed_wal_records(self):
        db = self.connect()
        db.execute("PRAGMA wal_autocheckpoint=0")
        db.execute("INSERT INTO submissions (scope,intent_id) VALUES ('foreign','wal-only')")
        db.commit()
        self.assertGreater(Path(str(self.path) + "-wal").stat().st_size, 0)
        with self.assertRaisesRegex(ValueError, "different account scope"):
            validate_runtime_state(self.directory, SCOPE, require_writable=False)

    def test_invalid_checkpoint_json_blocks_startup(self):
        with self.connect() as db:
            db.execute("INSERT INTO execution_consumers VALUES (?, 'owner', ?)",
                       (SCOPE, '{"revision": NaN}'))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            RuntimeStateLease(self.directory, SCOPE)

    def test_read_only_directory_and_database_block_execution(self):
        for path, mode in ((self.directory, 0o500), (self.path, 0o400)):
            with self.subTest(path=path):
                before = path.stat().st_mode & 0o777
                path.chmod(mode)
                try:
                    with self.assertRaises((ValueError, PermissionError)):
                        RuntimeStateLease(self.directory, SCOPE)
                finally:
                    path.chmod(before)

    def test_read_only_snapshot_validation_is_available_without_write_probe(self):
        self.path.chmod(0o400)
        try:
            self.assertEqual(validate_runtime_state(self.directory, SCOPE, require_writable=False), self.path)
        finally:
            self.path.chmod(0o600)

    def test_second_process_is_blocked_until_lease_is_released(self):
        command = [sys.executable, "-m", "scripts.check_runtime_state", "check",
                   "--state-dir", str(self.directory), "--scope", SCOPE]
        with RuntimeStateLease(self.directory, SCOPE):
            blocked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
            self.assertEqual(blocked.returncode, 1, blocked.stdout + blocked.stderr)
            self.assertIn("already has an execution process", blocked.stderr)
        recovered = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        self.assertIn("validation passed", recovered.stdout)

    def test_failed_validation_releases_the_lease(self):
        manifest_path = self.directory / MANIFEST_NAME
        manifest = manifest_path.read_text()
        with self.connect() as db:
            db.execute("UPDATE runtime_identity SET state_id='wrong'")
        with self.assertRaises(ValueError):
            RuntimeStateLease(self.directory, SCOPE)
        with self.connect() as db:
            db.execute("UPDATE runtime_identity SET state_id=?", (json.loads(manifest)["state_id"],))
        with RuntimeStateLease(self.directory, SCOPE):
            pass


if __name__ == "__main__":
    unittest.main()

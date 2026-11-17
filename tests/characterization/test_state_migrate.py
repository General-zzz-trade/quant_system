"""``state_dir migrate`` (ADR-0009 5; state database version 2, tech lead 2026-11-17: the funding rows kept).

A person brings an initialized directory's version-1 database to version 2 with the runtime stopped: only tables are
added, a copy of the old file is kept beside it, the migration is recorded in ``manual_actions`` and the manifest,
and the directory passes the start-up check after it. A database with any other problem, a running runtime or
another scope is refused. A restore of a version-1 backup migrates its copy.
"""
from __future__ import annotations

import io
import json
import sqlite3

import pytest

from tests.characterization.test_state_restore import ENV, NOW, SCOPE, _restore
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.sqlite import schema
from trading.infrastructure import state_backup
from trading.infrastructure import state_dir as managed


def _to_version_1(path):
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE funding_rows")
        db.execute("DROP TABLE funding_coverage")
        db.execute("PRAGMA user_version=1")


@pytest.fixture
def old(tmp_path):
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(), env=ENV) == 0
    _to_version_1(state / schema.STATE_DB)
    return state


def _migrate(state, env=ENV):
    out = io.StringIO()
    return managed.main(["--state-dir", str(state), "migrate"], stdout=out, env=env, clock=lambda: NOW), out.getvalue()


def test_a_version_1_directory_is_refused_at_start_up_and_told_to_migrate(old):
    with pytest.raises(RuntimeError, match="migrate"):
        managed.verify(old, SCOPE)


def test_migrate_adds_the_tables_keeps_the_old_file_and_records_it(old):
    code, out = _migrate(old)
    assert code == 0, out
    managed.verify(old, SCOPE)
    kept = [p for p in old.iterdir() if p.name.startswith(f"{schema.STATE_DB}.v1-")]
    assert len(kept) == 1 and "version 1 to 2" in out
    with sqlite3.connect(kept[0]) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    with sqlite3.connect(old / schema.STATE_DB) as db:
        rows = db.execute("SELECT kind, subject FROM manual_actions").fetchall()
    assert rows == [("migrate", "state database version 1 to 2")]
    assert json.loads((old / managed.MANIFEST).read_text())["state_db_version"] == 2
    assert _migrate(old)[0] == 0 and "nothing to migrate" in _migrate(old)[1]


def test_migrate_is_refused_while_a_runtime_holds_the_directory(old):
    lease = ExecutionLease(old / schema.STATE_DB, SCOPE)
    try:
        code, out = _migrate(old)
    finally:
        lease.close()
    assert code == 3 and "stop the runtime" in out
    with pytest.raises(RuntimeError, match="migrate"):
        managed.verify(old, SCOPE)                                  # nothing changed


def test_migrate_refuses_a_database_with_another_problem_or_another_scope(old, tmp_path):
    with sqlite3.connect(old / schema.STATE_DB) as db:
        db.execute("DROP TABLE entry_pauses")                       # lost a table as well
    code, out = _migrate(old)
    assert code == 2 and "cannot be migrated" in out
    other = {**ENV, "RISK_SCOPE": "binance-testnet-other"}
    assert _migrate(old, env=other)[0] == 2


def test_a_restore_of_a_version_1_backup_migrates_its_copy(tmp_path):
    from datetime import datetime, timezone
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(), env=ENV) == 0
    snapshot = state_backup.backup(state, tmp_path / "backups", now=datetime.fromtimestamp(NOW - 3600, timezone.utc))
    _to_version_1(snapshot)                                          # a backup made before version 2
    (state / schema.STATE_DB).write_bytes(b"")
    code, out = _restore(state, snapshot)
    assert code == 0, out
    assert "migrated to 2" in out
    managed.verify(state, SCOPE)

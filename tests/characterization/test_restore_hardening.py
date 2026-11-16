"""S6-S8 audit (2026-11-16): the restore and the database checks, hardened.

- The common case: a snapshot that already has this day's and week's drawdown rows, unlatched, is latched after
  the restore (audit M2). The restore says that the next start flattens every position and stays flat until the
  week ends (M1, tech lead 2026-11-16: keep the latch and say so).
- Files left beside the broken database (a WAL with committed frames, a hot rollback journal) are never applied
  to the restored one (M3, L1); the broken database is kept whole, its files together (L7).
- A corrupt page is found by the start-up check, by the backup and by the restore (M4).
- Today's count is used up whatever limits the service runs with (L2); a symbol only in the order journal is
  paused (A6); a temporary file left by an interrupted restore does not get in the way (A9).
"""
from __future__ import annotations

import io
import os
import random
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest

from tests.characterization.test_state_restore import ENV, NOW, SCOPE, _restore, broken  # noqa: F401 (fixture)
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.domain.risk.drawdown import periods
from trading.domain.risk.order_limits import OrderLimitsConfig, check_limits
from trading.infrastructure import state_backup
from trading.infrastructure import state_dir as managed

ROOT = Path(__file__).resolve().parents[2]


def _at(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc)


def _init(tmp_path):
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(), env=ENV) == 0
    return state, state / schema.STATE_DB


def _tripped(db):
    conn = sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return {(k, p): t for k, p, t in conn.execute("SELECT kind, period, tripped FROM drawdown_periods")}
    finally:
        conn.close()


def _died_in_the_middle(db, statement, *, journal_mode):
    """Another process commits (WAL) or starts (rollback) a write and dies without a checkpoint or a rollback."""
    code = ("import os, sqlite3, sys\n"
            "db = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
            f"db.execute('PRAGMA journal_mode={journal_mode}')\n"
            "db.execute('PRAGMA wal_autocheckpoint=0')\n"
            + ("db.execute('BEGIN IMMEDIATE')\n" if journal_mode == "DELETE" else "")
            + f"db.execute({statement!r})\n"
            "os._exit(0)\n")
    subprocess.run([sys.executable, "-c", code, str(db)], check=True, cwd=ROOT)


def _spoil_header(db):
    with open(db, "r+b") as handle:                      # "file is not a database": the start-up check refuses
        handle.write(b"not a database" * 8)


# -- M2 and M1: the common case, and what the restore says -----------------------------------------------------
def test_a_snapshot_with_this_day_and_week_unlatched_is_latched_and_the_flatten_is_announced(tmp_path):
    state, db = _init(tmp_path)
    SqliteDrawdownGuard(db, SCOPE, daily_pct=5, weekly_pct=12).observe(10_000.0, now=NOW - 120)
    keys = periods(NOW)
    assert _tripped(db) == {("day", keys["day"]): 0, ("week", keys["week"]): 0}
    snapshot = state_backup.backup(state, tmp_path / "backups", now=_at(NOW - 60))
    db.write_bytes(b"")
    code, out = _restore(state, snapshot)
    assert code == 0, out
    assert _tripped(db) == {("day", keys["day"]): 1, ("week", keys["week"]): 1}
    assert "flatten" in out and "week" in out                           # the next start closes every position


# -- M3, L1, L7: nothing left beside the broken database reaches the restored one ------------------------------
def test_a_wal_left_beside_the_broken_database_is_never_applied_to_the_restored_one(tmp_path):
    state, db = _init(tmp_path)
    SqliteDrawdownGuard(db, SCOPE, daily_pct=5, weekly_pct=12).observe(10_000.0, now=NOW - 120)
    snapshot = state_backup.backup(state, tmp_path / "backups", now=_at(NOW - 60))
    _died_in_the_middle(db, "UPDATE drawdown_periods SET baseline = baseline + 1", journal_mode="WAL")
    assert Path(f"{db}-wal").stat().st_size > 0                         # committed frames, never checkpointed
    _spoil_header(db)
    code, out = _restore(state, snapshot)
    assert code == 0, out
    keys = periods(NOW)
    assert _tripped(db) == {("day", keys["day"]): 1, ("week", keys["week"]): 1}
    managed.verify(state, SCOPE)


def test_a_hot_rollback_journal_beside_the_broken_database_is_never_rolled_into_the_restored_one(tmp_path):
    # The broken database is in rollback mode (for example a snapshot copied in by hand) and a write died half way:
    # its hot journal must not be rolled into the restored database.
    state, db = _init(tmp_path)
    snapshot = state_backup.backup(state, tmp_path / "backups", now=_at(NOW - 60))
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    sqlite3.connect(db).close()                                         # version 0, no tables: broken
    code = ("import os, sqlite3, sys\n"
            "db = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
            "db.execute('PRAGMA journal_mode=DELETE')\n"
            "db.execute('PRAGMA cache_size=1')\n"                         # spill: the journal gets the pages
            "db.execute('BEGIN')\n"
            "db.execute('CREATE TABLE junk(x)')\n"
            "for i in range(3000):\n"
            "    db.execute('INSERT INTO junk VALUES (?)', ('x' * 500,))\n"
            "os._exit(0)\n")
    subprocess.run([sys.executable, "-c", code, str(db)], check=True, cwd=ROOT)
    assert Path(f"{db}-journal").stat().st_size > 0                     # a hot journal: the write never ended
    code_, out = _restore(state, snapshot)
    assert code_ == 0, out
    managed.verify(state, SCOPE)
    assert SqliteDrawdownGuard(db, SCOPE).latched(now=NOW)


def test_the_broken_database_is_kept_whole_with_its_files_together(tmp_path):
    state, db = _init(tmp_path)
    snapshot = state_backup.backup(state, tmp_path / "backups", now=_at(NOW - 60))
    _died_in_the_middle(db, "UPDATE drawdown_periods SET baseline = 1", journal_mode="WAL")
    _spoil_header(db)
    assert _restore(state, snapshot)[0] == 0
    (kept,) = [p for p in state.iterdir() if p.name.startswith("state.sqlite.broken-")]
    assert kept.is_dir() and (kept / schema.STATE_DB).is_file() and (kept / f"{schema.STATE_DB}-wal").is_file()


# -- M4: a corrupt page --------------------------------------------------------------------------------------
def _corrupt_a_page(db):
    conn = sqlite3.connect(db)
    with conn:
        for i in range(400):
            conn.execute("INSERT INTO submissions (scope, intent_id, symbol, qty) VALUES (?, ?, ?, ?)",
                         (SCOPE, f"i{i}", "BTCUSDT", "0.1" * 20))
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    size = db.stat().st_size
    with open(db, "r+b") as handle:                      # garbage over pages past the schema page
        handle.seek(size // 2)
        handle.write(random.Random(7).randbytes(8192))


def test_a_corrupt_page_is_found_by_the_start_up_check_the_backup_and_the_restore(tmp_path):
    state, db = _init(tmp_path)
    good = state_backup.backup(state, tmp_path / "backups", now=_at(NOW - 3600))
    _corrupt_a_page(db)
    assert schema.problems(db)                                          # was [] before the audit
    with pytest.raises(RuntimeError, match="not usable"):
        managed.verify(state, SCOPE)
    out = io.StringIO()
    assert state_backup.main(["--state-dir", str(state), "--backup-dir", str(tmp_path / "backups")], stdout=out,
                             now=lambda: _at(NOW)) != 0
    assert sorted(p.name for p in (tmp_path / "backups").glob("state-*.sqlite")) == [good.name]
    code, text = _restore(state, good)                                  # not refused as "intact" any more
    assert code == 0, text
    managed.verify(state, SCOPE)


# -- L2, A6, A9 ----------------------------------------------------------------------------------------------
def test_today_s_count_is_used_up_whatever_limits_the_service_runs_with(broken):  # noqa: F811
    state, snapshot, kill = broken
    assert _restore(state, snapshot)[0] == 0                            # the shell's limit: 40 orders
    orders, notional = SqliteLimitStore(state / schema.STATE_DB, SCOPE).load(periods(NOW)["day"])
    service = OrderLimitsConfig(max_daily_orders=200, max_daily_notional=D("10000000"))   # the service's .env
    result = check_limits(service, qty=D("0.001"), price=D("100"), position_notional=D("0"), orders_last_second=0,
                          orders_today=orders, notional_today=notional)
    assert not result.allowed


def test_a_symbol_only_in_the_order_journal_is_paused(tmp_path):
    state, db = _init(tmp_path)
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("INSERT INTO submissions (scope, intent_id, symbol, qty) VALUES (?, 'i1', 'DOGEUSDT', '5')",
                     (SCOPE,))
    conn.close()
    snapshot = state_backup.backup(state, tmp_path / "backups", now=_at(NOW - 60))
    db.write_bytes(b"")
    assert _restore(state, snapshot)[0] == 0
    assert "DOGEUSDT" in {row["symbol"] for row in SqliteEntryPauses(db).active()}


def test_a_temporary_file_left_by_an_interrupted_restore_does_not_get_in_the_way(broken):  # noqa: F811
    state, snapshot, kill = broken
    for suffix in ("", "-journal", "-wal"):
        Path(f"{state}/.{schema.STATE_DB}.restoring{suffix}").write_bytes(b"left by a restore that was killed")
    code, out = _restore(state, snapshot)
    assert code == 0, out
    managed.verify(state, SCOPE)
    assert not os.path.exists(state / f".{schema.STATE_DB}.restoring")

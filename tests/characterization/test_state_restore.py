"""ADR-0016 point 6 (S8): ``state_dir restore`` puts a snapshot back only when the state database is lost or broken,
only while no runtime of the account runs (the host lock and the directory lease), with a reason and the scope
typed back, and leaves an audit record. A restore only tightens: the current day's and week's drawdown are
latched, today's order count is used up, every symbol's entries are paused for a person to release, and the kill
file is not touched. The broken database is kept beside it.
"""
from __future__ import annotations

import io
import json
import sqlite3
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading.adapters.files.host_lock import HostLock
from trading.adapters.sqlite import schema
from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.adapters.sqlite.limit_store import SqliteLimitStore
from trading.domain.risk.drawdown import periods
from trading.infrastructure import state_backup
from trading.infrastructure import state_dir as managed
from trading.infrastructure.config import account_identity

SCOPE = "binance-testnet-ut"
ENV = {"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1", "BINANCE_TESTNET_API_KEY": "k", "BINANCE_TESTNET_API_SECRET": "s",
       "BINANCE_MAX_DAILY_ORDERS": "40", "BINANCE_MAX_DAILY_NOTIONAL": "30000"}
NOW = 1_795_000_000.0                                     # a fixed instant; its UTC day and ISO week are latched


@pytest.fixture
def broken(tmp_path):
    """An initialized directory with some state, a snapshot of it, then the database emptied."""
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(), env=ENV) == 0
    SqliteEntryPauses(state / schema.STATE_DB).pause("SOLUSDT", "stop_triggered", ["t1"])
    from datetime import datetime, timezone
    snapshot = state_backup.backup(state, tmp_path / "backups", now=datetime.fromtimestamp(NOW - 3600, timezone.utc))
    kill = (state / managed.KILL_FILE).read_bytes()
    (state / schema.STATE_DB).write_bytes(b"")
    return state, snapshot, kill


def _restore(state, snapshot, *, typed=SCOPE, reason="state.sqlite emptied by a disk fault", env=ENV):
    out = io.StringIO()
    argv = ["--state-dir", str(state), "restore", "--snapshot", str(snapshot)]
    if reason is not None:
        argv += ["--reason", reason]
    code = managed.main(argv, stdout=out, env=env, stdin=io.StringIO(typed + "\n"), clock=lambda: NOW)
    return code, out.getvalue()


def test_a_restore_puts_the_snapshot_back_and_the_directory_passes_the_start_up_check(broken):
    state, snapshot, kill = broken
    code, out = _restore(state, snapshot)
    assert code == 0, out
    managed.verify(state, SCOPE)
    db = sqlite3.connect(state / schema.STATE_DB)
    try:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        (action,) = db.execute("SELECT kind, subject, reason FROM manual_actions").fetchall()
    finally:
        db.close()
    assert action == ("restore", snapshot.name, "state.sqlite emptied by a disk fault")
    assert [p.name for p in state.glob("state.sqlite.broken-*")]          # the broken database is kept


def test_a_restore_only_tightens(broken):
    state, snapshot, kill = broken
    assert _restore(state, snapshot)[0] == 0
    db = state / schema.STATE_DB
    assert SqliteDrawdownGuard(db, SCOPE).latched(now=NOW)               # the current day and week
    day = periods(NOW)["day"]
    orders, notional = SqliteLimitStore(db, SCOPE).load(day)
    assert orders >= 40 and notional >= D("30000")                       # today's count is used up
    paused = {row["symbol"] for row in SqliteEntryPauses(db).active()}
    assert {"BTCUSDT", "ETHUSDT", "SOLUSDT"} <= paused                   # every symbol, for a person to release
    assert (state / managed.KILL_FILE).read_bytes() == kill              # the kill file is not touched


def test_the_day_and_week_latch_both_hold(broken):
    state, snapshot, kill = broken
    assert _restore(state, snapshot)[0] == 0
    db = sqlite3.connect(state / schema.STATE_DB)
    try:
        rows = db.execute("SELECT kind, period, tripped FROM drawdown_periods WHERE scope=?", (SCOPE,)).fetchall()
    finally:
        db.close()
    keys = periods(NOW)
    assert {(kind, period) for kind, period, tripped in rows if tripped} == {(k, keys[k]) for k in ("day", "week")}


def test_an_intact_database_is_never_restored(tmp_path):
    state = tmp_path / "state"
    assert managed.main(["--state-dir", str(state), "init"], stdout=io.StringIO(), env=ENV) == 0
    from datetime import datetime, timezone
    snapshot = state_backup.backup(state, tmp_path / "backups", now=datetime.fromtimestamp(NOW, timezone.utc))
    before = (state / schema.STATE_DB).read_bytes()
    code, out = _restore(state, snapshot)
    assert code != 0 and "intact" in out
    assert (state / schema.STATE_DB).read_bytes() == before and not list(state.glob("*.broken-*"))


def test_nothing_is_restored_while_a_runtime_of_the_account_runs(broken):
    state, snapshot, kill = broken
    with HostLock(account_identity(testnet=True, api_key="k"), holder="the runtime"):
        code, out = _restore(state, snapshot)
    assert code == 3 and "stop the runtime" in out
    assert (state / schema.STATE_DB).read_bytes() == b""


@pytest.mark.parametrize("typed,reason", [("binance-testnet-other", "why"), (SCOPE, "  "), (SCOPE, None)])
def test_a_restore_needs_a_reason_and_the_scope_typed_back(broken, typed, reason):
    state, snapshot, kill = broken
    if reason is None:
        with pytest.raises(SystemExit):
            _restore(state, snapshot, typed=typed, reason=None)
    else:
        code, out = _restore(state, snapshot, typed=typed, reason=reason)
        assert code != 0
    assert (state / schema.STATE_DB).read_bytes() == b""


def test_a_snapshot_that_fails_its_check_is_refused(broken, tmp_path):
    state, snapshot, kill = broken
    bad = tmp_path / "backups" / "state-20261231T000000Z.sqlite"
    bad.write_bytes(b"not a database")
    code, out = _restore(state, bad)
    assert code != 0 and "snapshot" in out
    assert (state / schema.STATE_DB).read_bytes() == b""


def test_a_directory_that_was_never_initialized_is_not_restored_into(tmp_path, broken):
    state, snapshot, kill = broken
    other = tmp_path / "other"
    other.mkdir()
    code, out = _restore(other, snapshot)
    assert code != 0 and not (other / schema.STATE_DB).exists()


def test_the_snapshot_s_own_rows_come_back(broken):
    state, snapshot, kill = broken
    assert _restore(state, snapshot)[0] == 0
    causes = {row["symbol"]: row["cause"] for row in SqliteEntryPauses(state / schema.STATE_DB).active()}
    assert causes["SOLUSDT"] == "stop_triggered"                         # the pause from before keeps its cause
    assert json.loads((state / managed.MANIFEST).read_text())["scope"] == SCOPE
    assert Path(snapshot).is_file()                                      # the snapshot itself is left as it was

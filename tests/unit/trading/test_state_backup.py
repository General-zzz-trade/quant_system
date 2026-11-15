"""ADR-0016 point 6 (S8): an hourly snapshot of ``state.sqlite`` with SQLite's online backup, checked after it is
written (opens, right version, every table), kept 14 days. The kill file is never backed up. A failure exits
non-zero (the unit's OnFailure= sends the alert) and leaves the existing snapshots alone.
"""
from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading.adapters.sqlite import schema
from trading.adapters.sqlite.entry_pause import SqliteEntryPauses
from trading.infrastructure import state_backup
from trading.infrastructure import state_dir as managed

NOW = datetime(2026, 11, 20, 12, 0, 5, tzinfo=timezone.utc)
ENV = {"RISK_SCOPE": "binance-testnet-ut", "BINANCE_TESTNET": "1"}


@pytest.fixture
def state(tmp_path):
    directory = tmp_path / "state"
    assert managed.main(["--state-dir", str(directory), "init"], stdout=io.StringIO(), env=ENV) == 0
    SqliteEntryPauses(directory / schema.STATE_DB).pause("BTCUSDT", "stop_triggered", ["t1"])
    return directory


def _run(state, backups, now=NOW):
    out = io.StringIO()
    code = state_backup.main(["--state-dir", str(state), "--backup-dir", str(backups)], stdout=out,
                             now=lambda: now)
    return code, out.getvalue()


def _snapshots(backups):
    return sorted(p.name for p in Path(backups).glob("state-*.sqlite"))


def test_a_snapshot_is_written_checked_and_holds_the_state(state, tmp_path):
    code, out = _run(state, tmp_path / "backups")
    assert code == 0
    assert _snapshots(tmp_path / "backups") == ["state-20261120T120005Z.sqlite"]
    snapshot = tmp_path / "backups" / "state-20261120T120005Z.sqlite"
    assert schema.problems(snapshot) == []
    db = sqlite3.connect(f"{snapshot.resolve().as_uri()}?mode=ro", uri=True)
    try:
        assert db.execute("SELECT symbol, cause FROM entry_pauses").fetchall() == [("BTCUSDT", "stop_triggered")]
    finally:
        db.close()


def test_the_kill_file_is_never_backed_up(state, tmp_path):
    _run(state, tmp_path / "backups")
    assert [p.name for p in (tmp_path / "backups").iterdir()] == ["state-20261120T120005Z.sqlite"]


def test_a_broken_state_database_is_not_backed_up_and_fails(state, tmp_path):
    (state / schema.STATE_DB).write_bytes(b"")                          # emptied
    code, out = _run(state, tmp_path / "backups")
    assert code != 0 and "state.sqlite" in out
    assert _snapshots(tmp_path / "backups") == []


def test_a_missing_state_directory_fails_and_creates_nothing(tmp_path):
    code, out = _run(tmp_path / "nowhere", tmp_path / "backups")
    assert code != 0 and not (tmp_path / "nowhere").exists()


def test_snapshots_older_than_14_days_are_deleted_after_a_good_backup(state, tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    old = NOW - timedelta(days=14, seconds=1)
    young = NOW - timedelta(days=13, hours=23)
    for when in (old, young):
        (backups / f"state-{when:%Y%m%dT%H%M%SZ}.sqlite").write_bytes(b"x")
    (backups / "notes.txt").write_text("not a snapshot")
    (backups / "state-latest.sqlite").write_bytes(b"x")                 # not a snapshot name: left alone
    assert _run(state, backups)[0] == 0
    assert sorted(p.name for p in backups.iterdir()) == sorted([
        f"state-{young:%Y%m%dT%H%M%SZ}.sqlite", "state-20261120T120005Z.sqlite", "notes.txt",
        "state-latest.sqlite"])


def test_a_failed_backup_deletes_no_snapshot(state, tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    old = backups / f"state-{NOW - timedelta(days=30):%Y%m%dT%H%M%SZ}.sqlite"
    old.write_bytes(b"x")
    (state / schema.STATE_DB).write_bytes(b"")
    assert _run(state, backups)[0] != 0
    assert old.exists()


def test_a_snapshot_that_fails_its_check_is_not_kept_and_fails(state, tmp_path, monkeypatch):
    real = schema.problems
    calls = []

    def second_call_fails(path):
        calls.append(Path(path))
        return real(path) if len(calls) == 1 else ["state.sqlite has lost its tables (orders)"]
    monkeypatch.setattr(state_backup.schema, "problems", second_call_fails)
    code, out = _run(state, tmp_path / "backups")
    assert code != 0 and "lost its tables" in out
    assert len(calls) == 2 and calls[1] != state / schema.STATE_DB     # the snapshot itself was checked
    assert list((tmp_path / "backups").glob("state-*")) == []


def test_a_backup_directory_inside_the_state_directory_is_refused(state):
    code, out = _run(state, state / "backups")
    assert code != 0 and not (state / "backups").exists()


def test_two_backups_in_the_same_second_do_not_overwrite_each_other(state, tmp_path):
    assert _run(state, tmp_path / "backups")[0] == 0
    code, out = _run(state, tmp_path / "backups")
    assert code != 0 and _snapshots(tmp_path / "backups") == ["state-20261120T120005Z.sqlite"]

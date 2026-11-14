"""Paper state in SQLite with a JSON mirror (migration step 10, ADR-0007 sub-decision 3b)."""
from __future__ import annotations

import json
import sqlite3
from decimal import Decimal as D

from trading.adapters.files.state_files import JsonPaperStore
from trading.adapters.sqlite.paper_store import SqlitePaperStore
from trading.application import ports
from trading.infrastructure.paper_store import MirroredPaperStore

DOC = {"schema": 1, "spec": {"band": 0.1}, "symbols": {"BTCUSDT": {"cursor": 3_600_000, "equity": D("5000.25")}}}


def _store(tmp_path, clock=lambda: 1000.0):
    return MirroredPaperStore(SqlitePaperStore(tmp_path / "paper_state.sqlite", clock=clock),
                              JsonPaperStore(tmp_path / "paper_state.json", clock=clock))


def _later(cursor):
    return {**DOC, "symbols": {"BTCUSDT": {"cursor": cursor, "equity": D("5000.25")}}}


def test_a_saved_document_round_trips_through_the_database_and_is_mirrored_to_json(tmp_path):
    store = _store(tmp_path)
    assert isinstance(store, ports.PaperStore) and store.load() is None
    store.save(DOC)
    assert _store(tmp_path).load() == DOC                         # Decimals survive
    assert JsonPaperStore(tmp_path / "paper_state.json").load() == DOC
    with sqlite3.connect(tmp_path / "paper_state.sqlite") as db:
        assert db.execute("SELECT version, saved_at FROM paper_state").fetchall() == [(1, 1000.0)]


def test_an_existing_json_file_is_imported_once_and_kept(tmp_path):
    JsonPaperStore(tmp_path / "paper_state.json").save(DOC)       # written by the version before step 10
    assert _store(tmp_path).load() == DOC
    assert (tmp_path / "paper_state.json").exists()
    assert SqlitePaperStore(tmp_path / "paper_state.sqlite").load() == DOC


def test_a_json_file_changed_from_outside_wins_over_the_database(tmp_path):
    store = _store(tmp_path)
    store.save(DOC)
    JsonPaperStore(tmp_path / "paper_state.json").save(_later(7_200_000))   # e.g. a rolled-back version ran
    assert _store(tmp_path).load() == _later(7_200_000)
    assert SqlitePaperStore(tmp_path / "paper_state.sqlite").load() == _later(7_200_000)


def test_the_database_is_enough_when_the_json_file_is_gone(tmp_path):
    store = _store(tmp_path)
    store.save(DOC)
    (tmp_path / "paper_state.json").unlink()
    assert _store(tmp_path).load() == DOC


def test_a_crash_between_the_two_writes_loses_nothing(tmp_path):
    store = _store(tmp_path)
    store.save(DOC)
    SqlitePaperStore(tmp_path / "paper_state.sqlite").save(_later(7_200_000))   # database written, mirror not yet
    assert _store(tmp_path).load() == _later(7_200_000)                          # the stale mirror does not win
    store = _store(tmp_path)
    store.save(_later(10_800_000))
    with sqlite3.connect(tmp_path / "paper_state.sqlite") as db:                 # mirror written, digest not yet
        db.execute("UPDATE paper_state SET mirror_digest = 'stale'")
    assert _store(tmp_path).load() == _later(10_800_000)


def test_archiving_sets_both_copies_aside_and_reports_the_name(tmp_path):
    store = _store(tmp_path, clock=lambda: 12.5)
    assert store.archive() is None
    store.save(DOC)
    assert store.archive() == "paper_state.12500.json"
    assert store.load() is None and not (tmp_path / "paper_state.json").exists()
    assert json.loads((tmp_path / "paper_state.12500.json").read_text())["schema"] == 1
    with sqlite3.connect(tmp_path / "paper_state.sqlite") as db:
        assert db.execute("SELECT count(*) FROM paper_state").fetchone() == (0,)
        assert db.execute("SELECT archived_at FROM paper_state_archive").fetchall() == [(12.5,)]

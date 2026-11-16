"""S10 decision 4: the paper state (the simulated exchange's snapshot and the cursors) is saved atomically with the
order journal - in the same SQLite transaction, whenever either changed - so after a crash they are always from the
same instant.

The unit tests drive ``PaperOrderJournal`` with a fake document; the crash tests in
tests/unit/lab/test_paper_crash.py restart a paper run from the database as it was after a commit.
"""
from __future__ import annotations

import sqlite3

import pytest

from trading.adapters.sqlite import schema
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.adapters.sqlite.paper_state import PaperOrderJournal, SqlitePaperState

SCOPE = "paper-target"


class Source:
    """The runtime's side: a revision that changes with the document, None until the saved state is loaded."""

    def __init__(self):
        self.revision, self.document, self.built, self.broken = None, {"n": 0}, 0, False

    def build(self):
        if self.broken:
            raise OSError("disk full")
        self.built += 1
        return dict(self.document)


def _journal(tmp_path, source=None):
    path = tmp_path / schema.STATE_DB
    schema.create(path, paper=True)
    state = SqlitePaperState(path, clock=lambda: 1000.0)
    if source is not None:
        state.follow(lambda: source.revision, source.build)
    return PaperOrderJournal(path, SCOPE, state=state), state


def _reserve(journal, intent_id):
    return journal.reserve(intent_id=intent_id, symbol="BTCUSDT", client_id=f"c-{intent_id}", side="buy", qty="1",
                           reducing=False, owner="o", parent=None, order_type="market", clock=lambda: 1000.0)[0]


def test_a_new_paper_record_has_every_state_table_and_no_saved_document(tmp_path):
    journal, state = _journal(tmp_path)
    assert schema.problems(tmp_path / schema.STATE_DB) == []
    assert state.load() is None
    with pytest.raises(FileExistsError):
        schema.create(tmp_path / schema.STATE_DB, paper=True)


def test_a_journal_transaction_carries_the_document_when_it_changed(tmp_path):
    source = Source()
    journal, state = _journal(tmp_path, source)
    _reserve(journal, "i1")
    assert state.load() is None and source.built == 0         # nothing to save before the state is loaded
    source.revision, source.document = 1, {"n": 1}
    journal.orders()                                          # a read is a transaction too
    assert state.load() == {"n": 1} and source.built == 1
    _reserve(journal, "i2")
    journal.update_status("i1", "submitted", settle=False)
    assert source.built == 1                                  # unchanged: not written again
    source.revision, source.document = 2, {"n": 2}
    journal.set_order_id("i1", "BTCUSDT-1")
    assert state.load() == {"n": 2} and source.built == 2


def test_the_document_and_the_journal_change_commit_together_or_not_at_all(tmp_path):
    source = Source()
    journal, state = _journal(tmp_path, source)
    source.revision = 1
    journal.orders()
    source.revision, source.document = 2, {"n": 2}
    source.broken = True                                      # the document cannot be written ...
    with pytest.raises(OSError):
        _reserve(journal, "i1")
    assert SqliteOrderJournal(journal.path, SCOPE).orders() == []   # ... so the reservation did not happen either
    assert state.load() == {"n": 0}


def test_a_journal_operation_that_fails_writes_no_document(tmp_path):
    source = Source()
    journal, state = _journal(tmp_path, source)
    source.revision, source.document = 1, {"n": 1}
    with pytest.raises(sqlite3.Error):
        with journal.connect() as db:
            db.execute("INSERT INTO no_such_table VALUES (1)")
    assert state.load() is None
    journal.orders()                                          # still owed: written by the next transaction
    assert state.load() == {"n": 1}


def test_every_committed_document_is_also_handed_on_once(tmp_path):
    # Migration plan step 10: the JSON copy is written at every save until the audit after step 15 lets it stop.
    source, copies = Source(), []
    path = tmp_path / schema.STATE_DB
    schema.create(path, paper=True)
    state = SqlitePaperState(path, clock=lambda: 1000.0, saved=copies.append)
    state.follow(lambda: source.revision, source.build)
    journal = PaperOrderJournal(path, SCOPE, state=state)
    source.revision, source.document = 1, {"n": 1}
    journal.orders()
    journal.orders()
    state.save({"n": 2})
    source.revision, source.document = 3, {"n": 3}
    with pytest.raises(sqlite3.Error):
        with journal.connect() as db:
            db.execute("INSERT INTO no_such_table VALUES (1)")
    assert copies == [{"n": 1}, {"n": 2}]                     # not the one whose transaction rolled back


def test_saving_writes_the_given_document_and_marks_the_revision_written(tmp_path):
    source = Source()
    journal, state = _journal(tmp_path, source)
    source.revision = 5
    state.save({"n": 5})
    journal.orders()
    assert state.load() == {"n": 5} and source.built == 0


def test_a_commit_that_fails_hands_nothing_on_and_the_next_transaction_writes_again(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from trading.adapters.sqlite import paper_state as module
    source, copies = Source(), []
    path = tmp_path / schema.STATE_DB
    schema.create(path, paper=True)
    state = SqlitePaperState(path, clock=lambda: 1000.0, saved=copies.append)
    state.follow(lambda: source.revision, source.build)
    journal = PaperOrderJournal(path, SCOPE, state=state)
    real = module.connect

    @contextmanager
    def failing_commit(p):
        with real(p) as db:
            yield db
            db.rollback()                                     # the commit fails: nothing of it is kept
            raise sqlite3.OperationalError("disk I/O error")
    source.revision, source.document = 1, {"n": 1}
    monkeypatch.setattr(module, "connect", failing_commit)
    with pytest.raises(sqlite3.OperationalError):
        journal.orders()
    monkeypatch.setattr(module, "connect", real)
    assert copies == [] and state.load() is None
    journal.orders()                                        # still owed: written by the next transaction
    assert copies == [{"n": 1}] and state.load() == {"n": 1}

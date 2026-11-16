"""The paper state in the paper directory's state database, saved atomically with the order journal (S10 decision 4).

Paper state is the simulated exchange's snapshot plus, per symbol, the last bar decided on (ADR-0013 consequences);
the order journal holds the rest. Both live in ``state.sqlite``, and the snapshot is written in the same SQLite
transaction as the journal: ``PaperOrderJournal`` writes the current document just before each journal transaction
commits whenever the exchange or the cursors changed since the document was last written. Every committed state of
the database is therefore one instant of the running process - the exchange and the journal can never be from
different moments - and a crash goes back to the last commit as a whole:

- a change of the exchange that no commit has followed yet is lost with the crash, as if it had not happened (an
  order the journal reserved but whose placement was not committed is then not at the venue; the lifecycle settles
  it as ``not_found`` once the grace period has passed, exactly as for a real venue);
- a bar the exchange has matched but no decision followed yet is decided again at the restart; the exchange says
  which bars it has matched (``has_matched``) so none is matched twice.

The table is created with the database when a paper record starts (``schema.create(path, paper=True)``); this
module never creates one.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from trading.adapters.codec import decode, encode
from trading.adapters.sqlite.connection import connect
from trading.adapters.sqlite.order_journal import SqliteOrderJournal

class SqlitePaperState:
    """One document: the simulated exchange's snapshot and the cursors. ``follow`` names where the current document
    and its revision come from; until then nothing is written inside a journal transaction (a runtime that has not
    restored its saved state yet must not overwrite it). ``saved`` is told every document once it is committed."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time,
                 saved: Callable[[dict[str, Any]], None] = lambda document: None) -> None:
        self.path, self.clock, self.saved = Path(path), clock, saved
        self._revision: Callable[[], object | None] = lambda: None
        self._document: Callable[[], dict[str, Any]] | None = None
        self._written: object | None = None
        self._pending: tuple[object, dict[str, Any]] | None = None

    def follow(self, revision: Callable[[], object | None], document: Callable[[], dict[str, Any]]) -> None:
        """``revision()`` changes whenever the document would; None means "nothing to save yet"."""
        self._revision, self._document = revision, document

    def load(self) -> dict[str, Any] | None:
        with connect(self.path) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_state'").fetchone() is None:
                raise RuntimeError(f"{self.path} holds no paper state (made before S10 part 3); start a new paper "
                                   "record with --reset-paper-state")
            row = db.execute("SELECT document FROM paper_state WHERE id = 1").fetchone()
        if row is None:
            return None
        document: dict[str, Any] = json.loads(row[0], object_hook=decode)
        return document

    def save(self, document: dict[str, Any]) -> None:
        """Write ``document`` (the current one) in a transaction of its own."""
        revision = self._revision()
        with connect(self.path) as db:
            self._write(db, document)
        self._written = revision
        self.saved(document)

    def write_within(self, db: sqlite3.Connection) -> bool:
        """Inside a journal transaction, as its last statement: write the current document if it changed since it
        was last written. True when it did; ``committed`` follows once the transaction has committed."""
        revision = self._revision()
        if revision is None or revision == self._written or self._document is None:
            return False
        document = self._document()
        self._write(db, document)
        self._pending = (revision, document)
        return True

    def committed(self) -> None:
        assert self._pending is not None
        (revision, document), self._pending = self._pending, None
        self._written = revision
        self.saved(document)

    def _write(self, db: sqlite3.Connection, document: dict[str, Any]) -> None:
        db.execute("INSERT INTO paper_state (id, saved_at, document) VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE "
                   "SET saved_at=excluded.saved_at, document=excluded.document",
                   (float(self.clock()), json.dumps(document, default=encode)))


class PaperOrderJournal(SqliteOrderJournal):
    """The order journal of a paper run: every transaction also carries the paper state when it changed."""

    def __init__(self, path: Path, scope: str, *, state: SqlitePaperState,
                 entry_beside_working_reduction: bool = False) -> None:
        super().__init__(path, scope, entry_beside_working_reduction=entry_beside_working_reduction)
        self.state = state

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:    # type: ignore[override]
        with connect(self.path) as db:
            yield db
            wrote = self.state.write_within(db)            # the last statement of the transaction
        if wrote:                                          # committed
            self.state.committed()


__all__ = ["PaperOrderJournal", "SqlitePaperState"]

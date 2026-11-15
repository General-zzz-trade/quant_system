"""Entry pauses in SQLite (implements the EntryPauses port; migration step 12).

One row per pause; a pause is active until ``released_at`` is set. The runtime only adds pauses; a person
releases them with ``python -m trading.infrastructure.entry_pause``. Rows are never deleted, so the history
of every pause and its release reason stays. A file that cannot be read raises; it never reads as "no pause".

``pause_triggers`` lists every stop trigger (a stop row id of the order journal) a pause has been set for.
The runtime pauses for each trigger not listed yet, so a lost or emptied store pauses again rather than
releasing (audit after step 12b, findings 2 and 8).
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class SqliteEntryPauses:
    TABLES = ("entry_pauses", "pause_triggers")

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = Path(path), clock

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS entry_pauses (
                    symbol TEXT NOT NULL, paused_at REAL NOT NULL, cause TEXT NOT NULL,
                    released_at REAL, release_reason TEXT)""")
                db.execute("""CREATE TABLE IF NOT EXISTS pause_triggers (
                    symbol TEXT NOT NULL, trigger TEXT NOT NULL, PRIMARY KEY(symbol, trigger))""")
                yield db
        finally:
            db.close()

    def pause(self, symbol: str, cause: str, triggers: Iterable[str] = ()) -> bool:
        """Pause ``symbol``'s entries for ``triggers``; False if they already are (the first cause is kept).
        The triggers are recorded either way, in the same transaction."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.executemany("INSERT OR IGNORE INTO pause_triggers (symbol, trigger) VALUES (?, ?)",
                           [(symbol, trigger) for trigger in triggers])
            if db.execute("SELECT 1 FROM entry_pauses WHERE symbol=? AND released_at IS NULL", (symbol,)).fetchone():
                return False
            db.execute("INSERT INTO entry_pauses (symbol, paused_at, cause) VALUES (?, ?, ?)",
                       (symbol, float(self.clock()), cause))
            return True

    def reason(self, symbol: str) -> str | None:
        """Why ``symbol``'s entries are paused, or None."""
        with self._connect() as db:
            row = db.execute("SELECT cause FROM entry_pauses WHERE symbol=? AND released_at IS NULL "
                             "ORDER BY paused_at LIMIT 1", (symbol,)).fetchone()
        return str(row["cause"]) if row else None

    def covered(self, symbol: str) -> set[str]:
        """The stop triggers of ``symbol`` a pause has been set for."""
        with self._connect() as db:
            return {str(r["trigger"]) for r in db.execute("SELECT trigger FROM pause_triggers WHERE symbol=?",
                                                          (symbol,))}

    def active(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [dict(r) for r in db.execute("SELECT symbol, paused_at, cause FROM entry_pauses "
                                                "WHERE released_at IS NULL ORDER BY paused_at")]

    def release(self, symbol: str, reason: str) -> int:
        """End ``symbol``'s pause, recording why; returns how many pauses ended. For a person only."""
        if not reason.strip():
            raise ValueError("a release needs a reason")
        with self._connect() as db:
            return db.execute("UPDATE entry_pauses SET released_at=?, release_reason=? "
                              "WHERE symbol=? AND released_at IS NULL",
                              (float(self.clock()), reason.strip(), symbol)).rowcount


__all__ = ["SqliteEntryPauses"]

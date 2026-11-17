"""Entry pauses in SQLite (implements the EntryPauses port; migration step 12).

One row per pause; a pause is active until ``released_at`` (unset: until a person releases it). The runtime adds
pauses: a stop's pause is written with its end already set (it ends by itself, tech lead 2026-11-16), other pauses
wait for a person, who releases with ``python -m trading.infrastructure.entry_pause``, which also ends a scheduled
pause early. Rows are never deleted, so the history of every pause and how it ended stays. A file that cannot be
read raises; it never reads as "no pause".

``pause_triggers`` lists every stop trigger (a stop row id of the order journal) a pause has been set for.
The runtime pauses for each trigger not listed yet, so a lost or emptied store pauses again rather than
releasing (audit after step 12b, findings 2 and 8).
"""
from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from trading.adapters.sqlite.connection import connect

SCHEDULED_END = "ended by itself at its scheduled time"
RESET_WAIT_RELEASED = "a person ended the wait for the strategy's reset"
_ACTIVE = "(released_at IS NULL OR released_at > ?)"


class SqliteEntryPauses:
    TABLES = ("entry_pauses", "pause_triggers")

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = Path(path), clock

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """The state database as ``state_dir init`` created it; never creates a file or a table (ADR-0009)."""
        return connect(self.path)

    def pause(self, symbol: str, cause: str, triggers: Iterable[str] = (), *, ends_at: float | None = None) -> bool:
        """Pause ``symbol``'s entries for ``triggers`` until ``ends_at`` (None: until a person releases them). False
        if an active pause already lasts as long (a pause for a person outlasts any scheduled one; the earlier cause
        is kept). The triggers are recorded either way, in the same transaction."""
        now = float(self.clock())
        if ends_at is not None and not (math.isfinite(ends_at) and ends_at > now):
            raise ValueError("a scheduled pause must end in the future")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.executemany("INSERT OR IGNORE INTO pause_triggers (symbol, trigger) VALUES (?, ?)",
                           [(symbol, trigger) for trigger in triggers])
            ends = [r["released_at"] for r in
                    db.execute(f"SELECT released_at FROM entry_pauses WHERE symbol=? AND {_ACTIVE}", (symbol, now))]
            if any(end is None for end in ends) or (ends_at is not None and any(end >= ends_at for end in ends)):
                return False
            db.execute("INSERT INTO entry_pauses (symbol, paused_at, cause, released_at, release_reason) "
                       "VALUES (?, ?, ?, ?, ?)", (symbol, now, cause, None if ends_at is None else float(ends_at),
                                                  None if ends_at is None else SCHEDULED_END))
            return True

    def reason(self, symbol: str) -> str | None:
        """Why ``symbol``'s entries are paused, or None."""
        with self._connect() as db:
            row = db.execute(f"SELECT cause FROM entry_pauses WHERE symbol=? AND {_ACTIVE} "
                             "ORDER BY paused_at LIMIT 1", (symbol, float(self.clock()))).fetchone()
        return str(row["cause"]) if row else None

    def ended_by_itself(self, symbol: str) -> float | None:
        """When ``symbol``'s latest pause began, if that pause has ended by itself at its scheduled time; None when
        there is none, it is still running or a person released it (tech lead 2026-11-17: a spec's
        ``after_stop: pause_then_reset`` waits for the strategy's reset after such a pause)."""
        with self._connect() as db:
            row = db.execute("SELECT paused_at, released_at, release_reason FROM entry_pauses WHERE symbol=? "
                             "ORDER BY paused_at DESC, rowid DESC LIMIT 1", (symbol,)).fetchone()
        if row is None or row["release_reason"] != SCHEDULED_END or row["released_at"] is None:
            return None
        return float(row["paused_at"]) if float(row["released_at"]) <= float(self.clock()) else None

    def covered(self, symbol: str) -> set[str]:
        """The stop triggers of ``symbol`` a pause has been set for."""
        with self._connect() as db:
            return {str(r["trigger"]) for r in db.execute("SELECT trigger FROM pause_triggers WHERE symbol=?",
                                                          (symbol,))}

    def active(self) -> list[dict[str, Any]]:
        """The active pauses; ``ends_at`` is when a scheduled one ends by itself, None when a person releases it."""
        with self._connect() as db:
            return [dict(r) for r in db.execute(f"SELECT symbol, paused_at, cause, released_at AS ends_at "
                                                f"FROM entry_pauses WHERE {_ACTIVE} ORDER BY paused_at",
                                                (float(self.clock()),))]

    def release_wait(self, symbol: str, reason: str) -> bool:
        """End the wait for the strategy's reset after ``symbol``'s pause ended by itself (a spec's ``after_stop:
        pause_then_reset``): a release row, so the latest pause is no longer one that ended by itself. False when
        there is no such pause. For a person only (``entry_pause release``, audit of PR #142, finding 3)."""
        if not reason.strip():
            raise ValueError("a release needs a reason")
        if self.ended_by_itself(symbol) is None:
            return False
        now = float(self.clock())
        with self._connect() as db:
            db.execute("INSERT INTO entry_pauses (symbol, paused_at, cause, released_at, release_reason) "
                       "VALUES (?, ?, ?, ?, ?)", (symbol, now, RESET_WAIT_RELEASED, now, reason.strip()))
        return True

    def release(self, symbol: str, reason: str) -> int:
        """End ``symbol``'s active pauses now (a scheduled one too), recording why; returns how many ended. For a
        person only."""
        if not reason.strip():
            raise ValueError("a release needs a reason")
        now = float(self.clock())
        with self._connect() as db:
            return db.execute(f"UPDATE entry_pauses SET released_at=?, release_reason=? WHERE symbol=? AND {_ACTIVE}",
                              (now, reason.strip(), symbol, now)).rowcount


__all__ = ["SqliteEntryPauses"]

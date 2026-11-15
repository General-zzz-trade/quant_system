"""The one connection setting of the state database (ADR-0009 2).

Every store opens the existing file read-write (``mode=rw``: a missing file is an error, never a new empty
database), with full synchronous writes and the same busy timeout. WAL is set once, by ``schema.create``.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BUSY_TIMEOUT_MS = 10_000


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """A connection in a transaction (committed on success, rolled back on error), closed afterwards."""
    db = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=rw", uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA synchronous=FULL")      # the busy timeout is set by ``timeout`` above
        with db:
            yield db
    finally:
        db.close()


__all__ = ["BUSY_TIMEOUT_MS", "connect"]

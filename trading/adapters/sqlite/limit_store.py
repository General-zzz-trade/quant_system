"""Today's order count and notional per scope and UTC day, in SQLite (the LimitStore port; step 13b).

The table is created when the store is constructed (by ``state_dir init``, or before the runtime starts). Later
reads and writes open the existing file only: a store deleted or emptied while running raises, and the limiter
then refuses orders (ADR-0007 2b), instead of a re-created store counting the day from zero (audit after M3).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path


class SqliteLimitStore:
    TABLES = ("order_limits",)

    def __init__(self, path: Path, scope: str) -> None:
        if not scope.strip():
            raise ValueError("risk scope is required")
        self.path, self.scope = Path(path), scope
        db = sqlite3.connect(self.path, timeout=10)          # never creates the directory
        try:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS order_limits (
                    scope TEXT NOT NULL, day TEXT NOT NULL, orders INTEGER NOT NULL, notional TEXT NOT NULL,
                    PRIMARY KEY(scope, day))""")
        finally:
            db.close()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=rw", uri=True, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def load(self, day: str) -> tuple[int, Decimal]:
        with self._connect() as db:
            row = db.execute("SELECT orders, notional FROM order_limits WHERE scope=? AND day=?",
                             (self.scope, day)).fetchone()
        return (int(row[0]), Decimal(row[1])) if row else (0, Decimal(0))

    def add(self, day: str, notional: Decimal) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT orders, notional FROM order_limits WHERE scope=? AND day=?",
                             (self.scope, day)).fetchone()
            orders, total = (int(row[0]), Decimal(row[1])) if row else (0, Decimal(0))
            db.execute("INSERT OR REPLACE INTO order_limits VALUES (?, ?, ?, ?)",
                       (self.scope, day, orders + 1, str(total + notional)))


__all__ = ["SqliteLimitStore"]

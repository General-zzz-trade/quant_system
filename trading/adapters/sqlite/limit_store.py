"""Today's order count and notional per scope and UTC day, in SQLite (the LimitStore port; step 13b)."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path


class SqliteLimitStore:
    def __init__(self, path: Path, scope: str) -> None:
        if not scope.strip():
            raise ValueError("risk scope is required")
        self.path, self.scope = Path(path), scope

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS order_limits (
                    scope TEXT NOT NULL, day TEXT NOT NULL, orders INTEGER NOT NULL, notional TEXT NOT NULL,
                    PRIMARY KEY(scope, day))""")
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

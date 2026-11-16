"""Today's order count and notional per scope and UTC day, in SQLite (the LimitStore port; step 13b).

The table is created when the store is constructed (by ``state_dir init``, or before the runtime starts). Later
reads and writes open the existing file only: a store deleted or emptied while running raises, and the limiter
then refuses orders (ADR-0007 2b), instead of a re-created store counting the day from zero (audit after M3).
"""
from __future__ import annotations

import sqlite3
from decimal import Decimal
from contextlib import AbstractContextManager
from pathlib import Path

from trading.adapters.sqlite.connection import connect


class SqliteLimitStore:
    TABLES = ("order_limits",)

    def __init__(self, path: Path, scope: str) -> None:
        if not scope.strip():
            raise ValueError("risk scope is required")
        self.path, self.scope = Path(path), scope

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """The state database as ``state_dir init`` created it; never creates a file or a table (ADR-0009). An
        unreadable count is treated as used up by the limiter (ADR-0007 2b)."""
        return connect(self.path)

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


    def use_up(self, day: str, *, orders: int, notional: Decimal) -> None:
        """Count ``day`` as used up to at least these limits (a restore, ADR-0016 point 6); never lowers a count."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT orders, notional FROM order_limits WHERE scope=? AND day=?",
                             (self.scope, day)).fetchone()
            have_orders, have_notional = (int(row[0]), Decimal(row[1])) if row else (0, Decimal(0))
            db.execute("INSERT OR REPLACE INTO order_limits VALUES (?, ?, ?, ?)",
                       (self.scope, day, max(have_orders, orders), str(max(have_notional, notional))))


__all__ = ["SqliteLimitStore"]

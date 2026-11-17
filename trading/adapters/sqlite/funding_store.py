"""The strategy's funding rows in the state database (version 2; tech lead 2026-11-17).

Binance keeps the income history for three months only. The runtime keeps every funding row it reads (once, by the
venue's transaction id) and, per symbol, the span ``[from_ms, through_ms]`` its rows are known whole for. A refresh
then reads from shortly before ``through_ms`` on, so the total since the strategy's first fill stays known however
old the first fill is. A read that would leave a gap in the span is refused: the total across a gap is unknown.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import AbstractContextManager
from decimal import Decimal
from pathlib import Path

from trading.adapters.sqlite.connection import connect


class SqliteFundingStore:
    TABLES = ("funding_rows", "funding_coverage")

    def __init__(self, path: Path, scope: str) -> None:
        self.path, self.scope = Path(path), scope

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """The state database as ``state_dir init`` (or ``migrate``) left it; never creates anything (ADR-0009)."""
        return connect(self.path)

    def coverage(self, symbol: str) -> tuple[int, int] | None:
        """The span (ms) the symbol's rows are known whole for, or None."""
        with self._connect() as db:
            row = db.execute("SELECT from_ms, through_ms FROM funding_coverage WHERE scope=? AND symbol=?",
                             (self.scope, symbol)).fetchone()
        return None if row is None else (int(row["from_ms"]), int(row["through_ms"]))

    def add(self, symbol: str, rows: Iterable[tuple[str, int, Decimal]], *, from_ms: int, through_ms: int) -> None:
        """Keep the rows of a read that covered ``[from_ms, through_ms]`` whole, in one transaction. A read that
        starts after the known span ends would leave a gap: refused, nothing kept."""
        if through_ms < from_ms:
            raise ValueError("a read ends before it starts")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT from_ms, through_ms FROM funding_coverage WHERE scope=? AND symbol=?",
                             (self.scope, symbol)).fetchone()
            if old is None or from_ms <= int(old["from_ms"]):
                span = (from_ms, max(through_ms, int(old["through_ms"]) if old is not None else through_ms))
            elif from_ms <= int(old["through_ms"]):
                span = (int(old["from_ms"]), max(through_ms, int(old["through_ms"])))
            else:
                raise ValueError(f"{symbol}: a read from {from_ms} after the known span ends ({old['through_ms']}) "
                                 "would leave a gap")
            db.executemany("INSERT OR IGNORE INTO funding_rows (scope, symbol, tran_id, time_ms, amount) "
                           "VALUES (?, ?, ?, ?, ?)",
                           [(self.scope, symbol, str(tran), int(at), str(amount)) for tran, at, amount in rows])
            db.execute("INSERT INTO funding_coverage (scope, symbol, from_ms, through_ms) VALUES (?, ?, ?, ?) "
                       "ON CONFLICT(scope, symbol) DO UPDATE SET from_ms=excluded.from_ms, "
                       "through_ms=excluded.through_ms", (self.scope, symbol, *span))

    def total(self, symbol: str, *, since_ms: int) -> Decimal:
        """The sum of the symbol's kept rows from ``since_ms`` on."""
        with self._connect() as db:
            amounts = [str(r["amount"]) for r in db.execute(
                "SELECT amount FROM funding_rows WHERE scope=? AND symbol=? AND time_ms>=?",
                (self.scope, symbol, int(since_ms)))]
        return sum((Decimal(a) for a in amounts), Decimal(0))


__all__ = ["SqliteFundingStore"]

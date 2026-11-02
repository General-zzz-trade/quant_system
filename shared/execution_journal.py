"""SQLite order/fill journal and transactional consumer checkpoints.

No network or event framework dependency. A fill row is also its outbox row;
it cannot be lost between committing the exchange fact and publishing it.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ExecutionJournal:
    def __init__(self, path: Path, scope: str) -> None:
        self.path, self.scope = Path(path), scope
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS submissions (
                scope TEXT, intent_id TEXT, symbol TEXT, client_id TEXT, order_id TEXT,
                settled INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(scope, intent_id))""")
            columns = {r[1] for r in db.execute("PRAGMA table_info(submissions)")}
            for name, definition in {
                "side": "TEXT", "qty": "TEXT", "reducing": "INTEGER",
                "owner": "TEXT NOT NULL DEFAULT ''", "parent": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'unknown'",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE submissions ADD COLUMN {name} {definition}")
            db.execute("""CREATE TABLE IF NOT EXISTS execution_fills (
                scope TEXT NOT NULL, fill_id TEXT NOT NULL, intent_id TEXT NOT NULL,
                owner TEXT NOT NULL, payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(scope, fill_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS execution_consumers (
                scope TEXT NOT NULL, owner TEXT NOT NULL, checkpoint TEXT NOT NULL,
                PRIMARY KEY(scope, owner))""")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def orders(self, owner: str | None = None, *, unresolved: bool = False) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM submissions WHERE scope=?", [self.scope]
        if owner is not None:
            query += " AND owner=?"
            params.append(owner)
        if unresolved:
            query += " AND settled=0"
        with self.connect() as db:
            return [dict(r) for r in db.execute(query, params)]

    def fills(self, owner: str, *, undelivered: bool = True) -> list[dict[str, Any]]:
        query = "SELECT payload FROM execution_fills WHERE scope=? AND owner=?"
        if undelivered:
            query += " AND delivered=0"
        query += " ORDER BY rowid"
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute(query, (self.scope, owner))]

    def checkpoint(self, owner: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                             (self.scope, owner)).fetchone()
            return json.loads(row[0]) if row else None

    def consume(self, owner: str, fill_id: str, apply: Any, rollback: Any,
                *, expected_revision: str | None = None) -> bool:
        """Commit consumer state + delivery receipt together; restore memory on failure.

        The caller serializes access to its in-memory consumer. `apply` returns
        a JSON-compatible checkpoint and must not send orders or notifications.
        """
        applied = False
        try:
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT delivered, owner FROM execution_fills WHERE scope=? AND fill_id=?",
                                 (self.scope, fill_id)).fetchone()
                if row is None or row["owner"] != owner:
                    raise ValueError("Fill does not belong to this durable consumer")
                if row["delivered"]:
                    return False
                previous = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                                      (self.scope, owner)).fetchone()
                revision = json.loads(previous[0]).get("revision") if previous else None
                if revision != expected_revision:
                    raise RuntimeError("Consumer checkpoint changed in another process; restart required")
                applied = True
                checkpoint = apply()
                db.execute("INSERT OR REPLACE INTO execution_consumers VALUES (?, ?, ?)",
                           (self.scope, owner, json.dumps(checkpoint, allow_nan=False)))
                db.execute("UPDATE execution_fills SET delivered=1 WHERE scope=? AND fill_id=?",
                           (self.scope, fill_id))
            return True
        except BaseException:
            if applied:
                rollback()
            raise

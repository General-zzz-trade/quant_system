"""SQLite order journal: intents, exchange fills, consumer checkpoints and protective stops.

On the Binance target-runtime path this class is the only writer of these tables (migration step 6); the
ML runtime's DurableFillConsumer and the frozen Bybit/OKX adapters still write through ``connect()``.
A fill row is also its outbox row, so it cannot be lost between committing the exchange fact and
publishing it. Reservation rules that must be atomic with the insert run inside one ``BEGIN IMMEDIATE``
transaction here.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Why a reservation was refused (empty string: reserved).
DUPLICATE = "duplicate intent"
ENTRY_UNRESOLVED = "unresolved submission for the symbol"
REDUCTION_OVERLAP = "another reduction is unresolved"
UNCONSUMED_OWN_FILL = "an own fill is not consumed yet"
STALE_CHECKPOINT = "position checkpoint changed since the reduction was sized"
UNCONSUMED_FILL = "a fill of the symbol is not consumed yet"


class SqliteOrderJournal:
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
                "order_type": "TEXT",
                # Wall-clock reservation time; NULL for legacy rows (never auto-resolved).
                "created_at": "REAL",
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
        """Raw access, kept for the frozen Bybit/OKX adapters, the ML runtime and tests."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    # -- reads ------------------------------------------------------------
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

    def intent_fills(self, intent_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM execution_fills WHERE scope=? AND intent_id=?",
                              (self.scope, intent_id)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def fill(self, fill_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        query, params = "SELECT payload FROM execution_fills WHERE scope=? AND fill_id=?", [self.scope, fill_id]
        if owner is not None:
            query += " AND owner=?"
            params.append(owner)
        with self.connect() as db:
            row = db.execute(query, params).fetchone()
        return json.loads(row[0]) if row else None

    def checkpoint(self, owner: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                             (self.scope, owner)).fetchone()
            return json.loads(row[0]) if row else None

    # -- intents ----------------------------------------------------------
    def reserve(self, *, intent_id: str, symbol: str, client_id: str, side: str, qty: str, reducing: bool,
                owner: str, parent: dict[str, Any] | None, order_type: str,
                clock: Callable[[], float]) -> tuple[str, float]:
        """Atomically check the reservation rules and insert a ``reserved`` row: (refusal or "", created_at)."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM submissions WHERE scope=? AND intent_id=?",
                          (self.scope, intent_id)).fetchone():
                return DUPLICATE, 0.0
            if db.execute("SELECT 1 FROM submissions WHERE scope=? AND symbol=? AND settled=0",
                          (self.scope, symbol)).fetchone() and not reducing:
                return ENTRY_UNRESOLVED, 0.0
            # Reductions may proceed while an entry is ambiguous, but may not
            # overlap another unresolved reduction for the same account/symbol.
            if reducing and db.execute(
                "SELECT 1 FROM submissions WHERE scope=? AND symbol=? AND settled=0 AND reducing=1",
                (self.scope, symbol),
            ).fetchone():
                return REDUCTION_OVERLAP, 0.0
            if reducing and owner:
                # An unconsumed trade makes the owner's checkpoint stale. Never
                # size another reduction from it, even if that order is terminal.
                if db.execute(
                    "SELECT 1 FROM execution_fills f JOIN submissions s "
                    "ON f.scope=s.scope AND f.intent_id=s.intent_id "
                    "WHERE f.scope=? AND f.owner=? AND s.symbol=? AND f.delivered=0",
                    (self.scope, owner, symbol),
                ).fetchone():
                    return UNCONSUMED_OWN_FILL, 0.0
                if parent is not None and "consumer_revision" in parent:
                    row = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                                     (self.scope, owner)).fetchone()
                    current = json.loads(row[0]).get("revision") if row else None
                    if current != parent["consumer_revision"]:
                        return STALE_CHECKPOINT, 0.0
            if not reducing and db.execute(
                "SELECT 1 FROM execution_fills f JOIN submissions s "
                "ON f.scope=s.scope AND f.intent_id=s.intent_id "
                "WHERE f.scope=? AND s.symbol=? AND f.delivered=0 AND s.parent IS NOT NULL",
                (self.scope, symbol),
            ).fetchone():
                return UNCONSUMED_FILL, 0.0
            created_at = float(clock())
            db.execute("""INSERT INTO submissions
                (scope,intent_id,symbol,client_id,order_id,settled,side,qty,reducing,owner,parent,status,order_type,
                 created_at)
                VALUES (?, ?, ?, ?, '', 0, ?, ?, ?, ?, ?, 'reserved', ?, ?)""",
                (self.scope, intent_id, symbol, client_id, side, qty, int(reducing), owner,
                 json.dumps(parent) if parent is not None else None, order_type, created_at))
        return "", created_at

    def adopt(self, *, intent_id: str, symbol: str, client_id: str, order_id: str, side: str, owner: str,
              parent: dict[str, Any], created_at: float) -> None:
        """Insert a venue-originated reducing order (a triggered stop) as ``submitted``."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT INTO submissions
                (scope,intent_id,symbol,client_id,order_id,settled,side,qty,reducing,owner,parent,status,
                 order_type,created_at) VALUES (?,?,?,?,?,0,?,?,1,?,?,'submitted','market',?)""",
                (self.scope, intent_id, symbol, client_id, order_id, side, "0", owner,
                 json.dumps(parent), created_at))

    def set_order_id(self, intent_id: str, order_id: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE submissions SET order_id=? WHERE scope=? AND intent_id=?",
                       (order_id, self.scope, intent_id))

    def update_status(self, intent_id: str, status: str, *, settle: bool) -> str | None:
        """Write a status (and settle). Returns the previous status when the row was still unsettled."""
        with self.connect() as db:
            row = db.execute("SELECT status, settled FROM submissions WHERE scope=? AND intent_id=?",
                             (self.scope, intent_id)).fetchone()
            if settle:
                db.execute("UPDATE submissions SET settled=1, status=? WHERE scope=? AND intent_id=?",
                           (status, self.scope, intent_id))
            else:
                db.execute("UPDATE submissions SET status=? WHERE scope=? AND intent_id=? AND settled=0",
                           (status, self.scope, intent_id))
        return str(row[0]) if row is not None and not row[1] else None

    # -- fills and consumers ----------------------------------------------
    def insert_fill(self, *, fill_id: str, intent_id: str, owner: str, payload: dict[str, Any]) -> bool:
        """Durable dedup record and outbox in one row; False when the fill was already stored."""
        with self.connect() as db:
            inserted = db.execute("INSERT OR IGNORE INTO execution_fills "
                                  "(scope,fill_id,intent_id,owner,payload) VALUES (?,?,?,?,?)",
                                  (self.scope, fill_id, intent_id, owner,
                                   json.dumps(payload, allow_nan=False))).rowcount
        return bool(inserted)

    def consume(self, owner: str, fill_id: str, apply: Callable[[], dict[str, Any]], rollback: Callable[[], None],
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

    # -- protective stops -------------------------------------------------
    def ensure_stop_table(self) -> None:
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS protective_stops (
                scope TEXT NOT NULL, owner TEXT NOT NULL, client_algo_id TEXT NOT NULL,
                symbol TEXT NOT NULL, revision TEXT NOT NULL, attempt INTEGER NOT NULL,
                side TEXT NOT NULL, trigger_price TEXT NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, PRIMARY KEY(scope, client_algo_id))""")

    def stop_rows(self, owner: str, symbol: str, *, live_only: bool) -> list[dict[str, Any]]:
        query = "SELECT * FROM protective_stops WHERE scope=? AND owner=? AND symbol=?"
        if live_only:
            query += " AND status IN ('active','unknown','canceling')"
        with self.connect() as db:
            return [dict(r) for r in db.execute(query + " ORDER BY created_at", (self.scope, owner, symbol))]

    def add_stop(self, *, owner: str, client_algo_id: str, symbol: str, revision: str, attempt: int, side: str,
                 trigger_price: str, created_at: float) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO protective_stops VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (self.scope, owner, client_algo_id, symbol, revision, attempt, side, trigger_price,
                        "unknown", created_at))

    def set_stop_status(self, client_algo_id: str, status: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE protective_stops SET status=? WHERE scope=? AND client_algo_id=?",
                       (status, self.scope, client_algo_id))

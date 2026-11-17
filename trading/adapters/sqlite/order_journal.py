"""SQLite order journal: intents, exchange fills, consumer checkpoints and protective stops.

On the Binance target-runtime path this class is the only writer of these tables (migration step 6); the
ML runtime's DurableFillConsumer and the frozen Bybit/OKX adapters still write through ``connect()``.
A fill row is also its outbox row, so it cannot be lost between committing the exchange fact and
publishing it. Reservation rules that must be atomic with the insert run inside one ``BEGIN IMMEDIATE``
transaction here.
"""
from __future__ import annotations

import sqlite3
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from trading.adapters.sqlite.connection import connect

from trading.domain.orders import (DUPLICATE, ENTRY_UNRESOLVED, REDUCTION_OVERLAP, SETTLED, STALE_CHECKPOINT,
                                   UNCONSUMED_FILL, UNCONSUMED_OWN_FILL)


class SqliteOrderJournal:
    TABLES = ("submissions", "execution_fills", "execution_consumers", "manual_actions", "protective_stops")

    def __init__(self, path: Path, scope: str, *, entry_beside_working_reduction: bool = False) -> None:
        self.path, self.scope = Path(path), scope
        # ModePolicy.entry_beside_working_reduction (paper, S10 decision 1): an entry may be reserved while the
        # symbol's only unsettled orders are reductions the venue acknowledged ("submitted"). The simulated venue
        # fills a market reduction at the next bar's open, so the entry of a flip is reserved while its reduction
        # rests, as research places both on one bar. Execution keeps the stricter rule.
        self.entry_beside_working_reduction = entry_beside_working_reduction

    def connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """The state database as ``state_dir init`` created it (WAL, synchronous=FULL); never creates a file or a
        table (ADR-0009). Raw access is kept for tests."""
        return connect(self.path)

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

    def entry_filled_since(self, owner: str, since: float) -> bool:
        """Whether an entry order (not reducing) of the owner created after ``since`` (seconds) has an exchange fill.
        An order is created just before it is sent, so its fills came after ``since`` (tech lead 2026-11-17: an
        entry after a stop ends the wait for the strategy's reset, also after a restart)."""
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM submissions s WHERE s.scope=? AND s.owner=? AND s.reducing=0 AND "
                             "s.created_at > ? AND EXISTS (SELECT 1 FROM execution_fills f WHERE f.scope=s.scope "
                             "AND f.intent_id=s.intent_id) LIMIT 1", (self.scope, owner, float(since))).fetchone()
        return row is not None

    def first_fill_at(self, owner: str) -> float | None:
        """When the owner's ledger first held something (seconds): the creation time of its earliest order with an
        exchange fill, or of a position a person put into its ledger, whichever is earlier; None if neither exists.
        The fill rows carry no time; an order is created just before it is sent, so this is at or before the fill.
        A fill whose order has no creation time raises: its time is unknown."""
        with self.connect() as db:
            filled, untimed = db.execute(
                "SELECT MIN(s.created_at), SUM(s.created_at IS NULL) FROM execution_fills f LEFT JOIN submissions s "
                "ON f.scope=s.scope AND f.intent_id=s.intent_id WHERE f.scope=? AND f.owner=?",
                (self.scope, owner)).fetchone()
            manual = db.execute("SELECT MIN(at) FROM manual_actions WHERE scope=? AND kind='set_position' "
                                "AND subject=?", (self.scope, owner)).fetchone()[0]
        if untimed:
            raise RuntimeError(f"{owner}: a fill whose order has no creation time")
        times = [float(t) for t in (filled, manual) if t is not None]
        return min(times) if times else None

    def owners(self) -> list[str]:
        """Every owner the scope's ledger knows: one with a checkpoint (a consumed fill, or a position a person put
        into its ledger, which writes a checkpoint too) or with a fill not consumed yet. The strategy capital base
        and the loss stop count each of the strategy's (audit 2026-11-16, finding 7), also one whose symbol the
        current spec no longer trades."""
        with self.connect() as db:
            rows = db.execute("SELECT owner FROM execution_consumers WHERE scope=? "
                              "UNION SELECT owner FROM execution_fills WHERE scope=?",
                              (self.scope, self.scope)).fetchall()
        return sorted(str(row[0]) for row in rows)

    def checkpoint(self, owner: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                             (self.scope, owner)).fetchone()
            return json.loads(row[0]) if row else None

    # -- intents ----------------------------------------------------------
    def reserve(self, *, intent_id: str, symbol: str, client_id: str, side: str, qty: str, reducing: bool,
                owner: str, parent: dict[str, Any] | None, order_type: str,
                clock: Callable[[], float], venue_close: bool = False) -> tuple[str, float]:
        """Atomically check the reservation rules and insert a ``reserved`` row: (refusal or "", created_at).

        ``venue_close`` (a kill switch or drawdown close sized from the venue, ADR-0005 3b and 7b) does not
        depend on the ledger, so the two ledger rules - an unconsumed own fill, a changed checkpoint - do not
        apply to it. The rule against overlapping reductions still does."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM submissions WHERE scope=? AND intent_id=?",
                          (self.scope, intent_id)).fetchone():
                return DUPLICATE, 0.0
            unsettled = "SELECT 1 FROM submissions WHERE scope=? AND symbol=? AND settled=0"
            if self.entry_beside_working_reduction:
                unsettled += " AND NOT (reducing=1 AND status='submitted')"
            if db.execute(unsettled, (self.scope, symbol)).fetchone() and not reducing:
                return ENTRY_UNRESOLVED, 0.0
            # Reductions may proceed while an entry is ambiguous, but may not
            # overlap another unresolved reduction for the same account/symbol.
            if reducing and db.execute(
                "SELECT 1 FROM submissions WHERE scope=? AND symbol=? AND settled=0 AND reducing=1",
                (self.scope, symbol),
            ).fetchone():
                return REDUCTION_OVERLAP, 0.0
            if reducing and owner and not venue_close:
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
    def stop_rows(self, owner: str, symbol: str, *, live_only: bool) -> list[dict[str, Any]]:
        query = "SELECT * FROM protective_stops WHERE scope=? AND owner=? AND symbol=?"
        if live_only:
            query += " AND status IN ('active','unknown','canceling','awaiting_order')"
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

    # -- manual actions: called only by trading.infrastructure.manual, for a person, never by the runtime ----
    def settle_manually(self, intent_id: str, status: str, *, venue: dict[str, Any], reason: str,
                        at: float) -> str | None:
        """Settle an unsettled order as ``status``; returns its previous status, None if it was not open."""
        if status not in SETTLED or not reason.strip():
            raise ValueError("a manual settlement needs a settled status and a reason")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status, settled FROM submissions WHERE scope=? AND intent_id=?",
                             (self.scope, intent_id)).fetchone()
            if row is None or row["settled"]:
                return None
            db.execute("UPDATE submissions SET settled=1, status=? WHERE scope=? AND intent_id=?",
                       (status, self.scope, intent_id))
            db.execute("INSERT INTO manual_actions VALUES (?, ?, 'settle_order', ?, ?, ?, ?, ?)",
                       (self.scope, at, intent_id, row["status"], status, json.dumps(venue), reason.strip()))
            return str(row["status"])

    def set_position_manually(self, owner: str, position: dict[str, Any], *, venue: dict[str, Any], reason: str,
                              at: float) -> dict[str, Any]:
        """Replace the owner's ledger position (not a fill) and give the checkpoint a new revision. Refused
        while the owner has fills that are not consumed yet: they would apply on top of the new position."""
        if not reason.strip():
            raise ValueError("a manual position needs a reason")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM execution_fills WHERE scope=? AND owner=? AND delivered=0",
                          (self.scope, owner)).fetchone():
                raise ValueError("the owner has fills that are not consumed yet")
            row = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                             (self.scope, owner)).fetchone()
            before = json.loads(row[0]) if row else None
            checkpoint = {"position": position, "revision": f"manual-{int(at * 1000)}",
                          "strategy": (before or {}).get("strategy", {})}
            db.execute("INSERT OR REPLACE INTO execution_consumers VALUES (?, ?, ?)",
                       (self.scope, owner, json.dumps(checkpoint, allow_nan=False)))
            db.execute("INSERT INTO manual_actions VALUES (?, ?, 'set_position', ?, ?, ?, ?, ?)",
                       (self.scope, at, owner, json.dumps(before["position"]) if before else None,
                        json.dumps(position), json.dumps(venue), reason.strip()))
            return checkpoint

    def manual_actions(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM manual_actions WHERE scope=? ORDER BY at, rowid",
                                                (self.scope,))]

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from execution.store.interfaces import EventLog


@dataclass(slots=True)
class InMemoryEventLog(EventLog):
    _rows: list[Dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1

    def append(self, *, event_type: str, payload: Mapping[str, Any], correlation_id: Optional[str] = None) -> int:
        rid = self._next_id
        self._next_id += 1
        self._rows.append(
            {
                "id": rid,
                "ts": time.time(),
                "event_type": str(event_type),
                "correlation_id": correlation_id,
                "payload": dict(payload),
            }
        )
        return rid

    def iter(self, *, after_id: int = 0) -> Iterable[Mapping[str, Any]]:
        for r in self._rows:
            if int(r["id"]) > int(after_id):
                yield dict(r)

    def list_recent(self, *, event_type: str | None = None, limit: int = 20) -> list[Dict[str, Any]]:
        rows = list(self._rows)
        if event_type is not None:
            rows = [r for r in rows if str(r["event_type"]) == str(event_type)]
        return [dict(r) for r in rows[-int(limit):]][::-1]


@dataclass(slots=True)
class SQLiteEventLog(EventLog):
    """Append-only event log for execution layer.

    This is the building block for:
    - restart-safe auditing
    - replay-driven recovery of execution-side truth
    - reconcile drift reports
    """

    path: str
    timeout_s: float = 10.0

    _conn: sqlite3.Connection = field(init=False, repr=False)
    _closed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=self.timeout_s, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                event_type TEXT NOT NULL,
                correlation_id TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_log_ts ON event_log(ts);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_log_corr ON event_log(correlation_id);")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_log_type ON event_log(event_type);")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._conn.close()
        finally:
            self._closed = True

    def __enter__(self) -> "SQLiteEventLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def append(self, *, event_type: str, payload: Mapping[str, Any], correlation_id: Optional[str] = None) -> int:
        now = time.time()
        payload_json = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        cur = self._conn.execute(
            "INSERT INTO event_log(ts, event_type, correlation_id, payload) VALUES(?,?,?,?)",
            (now, str(event_type), correlation_id, payload_json),
        )
        return int(cur.lastrowid)

    def iter(self, *, after_id: int = 0) -> Iterable[Mapping[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, ts, event_type, correlation_id, payload FROM event_log WHERE id > ? ORDER BY id ASC",
            (int(after_id),),
        ).fetchall()
        for rid, ts, event_type, correlation_id, payload_json in rows:
            yield {
                "id": int(rid),
                "ts": float(ts),
                "event_type": str(event_type),
                "correlation_id": correlation_id,
                "payload": json.loads(payload_json),
            }

    def list_recent(self, *, event_type: str | None = None, limit: int = 20) -> list[Dict[str, Any]]:
        if event_type is None:
            rows = self._conn.execute(
                "SELECT id, ts, event_type, correlation_id, payload FROM event_log ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT id, ts, event_type, correlation_id, payload
                   FROM event_log WHERE event_type = ? ORDER BY id DESC LIMIT ?""",
                (str(event_type), int(limit)),
            ).fetchall()
        return [
            {
                "id": int(rid),
                "ts": float(ts),
                "event_type": str(kind),
                "correlation_id": correlation_id,
                "payload": json.loads(payload_json),
            }
            for rid, ts, kind, correlation_id, payload_json in rows
        ]

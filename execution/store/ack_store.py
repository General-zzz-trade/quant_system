from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from execution.store.interfaces import AckStore

from _quant_hotpath import RustAckStore as _RustAckStore


@dataclass(slots=True)
class InMemoryAckStore(AckStore):
    """In-memory idempotency store backed by Rust. Suitable for backtests and short-lived runs."""
    ttl_sec: Optional[float] = None
    _rust: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rust = _RustAckStore(ttl_sec=self.ttl_sec)

    def get(self, key: str) -> Optional[Mapping[str, Any]]:
        payload_json = self._rust.get_json(key, time.time())
        if payload_json is None:
            return None
        result: Mapping[str, Any] = json.loads(payload_json)
        return result

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        payload_json = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        self._rust.put_json(key, payload_json, time.time())

    def prune(self) -> int:
        return int(self._rust.prune(time.time()))


@dataclass(slots=True)
class SQLiteAckStore(AckStore):
    """SQLite-backed idempotency store.

    Design goals:
    - restart-safe deduplication
    - deterministic behavior (no background threads)
    - safe defaults (WAL mode)

    This is enough for single-node institutional-grade execution.
    For multi-node, swap out for Redis/Postgres with the same interface.
    """

    path: str
    ttl_sec: Optional[float] = None
    timeout_s: float = 10.0

    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=self.timeout_s, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS acks (
                idem_key TEXT PRIMARY KEY,
                payload  TEXT NOT NULL,
                ts       REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_acks_ts ON acks(ts);")
        # Prune eagerly on startup if ttl set
        self.prune()

    def get(self, key: str) -> Optional[Mapping[str, Any]]:
        row = self._conn.execute("SELECT payload, ts FROM acks WHERE idem_key=?", (key,)).fetchone()
        if row is None:
            return None
        payload_json, ts = row
        if self.ttl_sec is not None and (time.time() - float(ts)) > float(self.ttl_sec):
            self._conn.execute("DELETE FROM acks WHERE idem_key=?", (key,))
            return None
        result: Mapping[str, Any] = json.loads(payload_json)
        return result

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        payload_json = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO acks(idem_key, payload, ts) VALUES(?,?,?)",
            (key, payload_json, now),
        )

    def prune(self) -> int:
        if self.ttl_sec is None:
            return 0
        cutoff = time.time() - float(self.ttl_sec)
        cur = self._conn.execute("DELETE FROM acks WHERE ts < ?", (cutoff,))
        return int(cur.rowcount or 0)

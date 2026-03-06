from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from execution.store.interfaces import DedupStore


@dataclass(slots=True)
class InMemoryDedupStore(DedupStore):
    """In-memory dedup store keyed by string -> digest.

    Notes:
    - Suitable for backtests / short-lived runs.
    - Not restart-safe.
    """
    ttl_sec: Optional[float] = None
    _m: Dict[str, str] = field(default_factory=dict, init=False)
    _ts: Dict[str, float] = field(default_factory=dict, init=False)

    def get(self, key: str) -> Optional[str]:
        v = self._m.get(key)
        if v is None:
            return None
        if self.ttl_sec is not None:
            ts = self._ts.get(key)
            if ts is not None and (time.time() - ts) > float(self.ttl_sec):
                self._m.pop(key, None)
                self._ts.pop(key, None)
                return None
        return v

    def put(self, key: str, digest: str) -> None:
        self._m[key] = str(digest)
        self._ts[key] = time.time()

    def prune(self) -> int:
        if self.ttl_sec is None:
            return 0
        now = time.time()
        dead = [k for k, ts in self._ts.items() if (now - ts) > float(self.ttl_sec)]
        for k in dead:
            self._m.pop(k, None)
            self._ts.pop(k, None)
        return len(dead)


@dataclass(slots=True)
class SQLiteDedupStore(DedupStore):
    """SQLite-backed dedup store.

    This is intended to provide restart-safe deduplication for critical ingress
    signals (fills, order updates) where duplicates can corrupt positions/cash.

    Table schema:
      key TEXT PRIMARY KEY
      digest TEXT NOT NULL
      ts REAL NOT NULL
    """
    path: str
    ttl_sec: Optional[float] = None
    _conn: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS dedup (key TEXT PRIMARY KEY, digest TEXT NOT NULL, ts REAL NOT NULL)"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_dedup_ts ON dedup(ts)")
            self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT digest, ts FROM dedup WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        digest, ts = row[0], float(row[1])
        if self.ttl_sec is not None and (time.time() - ts) > float(self.ttl_sec):
            with self._lock:
                self._conn.execute("DELETE FROM dedup WHERE key=?", (key,))
                self._conn.commit()
            return None
        return str(digest)

    def put(self, key: str, digest: str) -> None:
        now = time.time()
        # Prefer insert; if exists, update ts but keep digest unchanged.
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO dedup(key, digest, ts) VALUES(?,?,?)",
                (key, str(digest), now),
            )
            self._conn.execute("UPDATE dedup SET ts=? WHERE key=?", (now, key))
            self._conn.commit()

    def prune(self) -> int:
        if self.ttl_sec is None:
            return 0
        cutoff = time.time() - float(self.ttl_sec)
        with self._lock:
            cur = self._conn.execute("DELETE FROM dedup WHERE ts < ?", (cutoff,))
            self._conn.commit()
        return int(cur.rowcount)

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

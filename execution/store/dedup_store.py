from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

from execution.store.interfaces import DedupStore  # noqa: E402

from _quant_hotpath import RustDedupStore as _RustDedupStore  # noqa: E402


@dataclass(slots=True)
class InMemoryDedupStore(DedupStore):
    """In-memory dedup store backed by Rust. Not restart-safe."""
    ttl_sec: Optional[float] = None
    _rust: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rust = _RustDedupStore(ttl_sec=self.ttl_sec)

    def get(self, key: str) -> Optional[str]:
        return self._rust.get(key, time.time())

    def put(self, key: str, digest: str) -> None:
        self._rust.put(key, str(digest), time.time())

    def prune(self) -> int:
        return int(self._rust.prune(time.time()))


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
    _closed: bool = field(init=False, repr=False, default=False)

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
        with self._lock:
            self._conn.execute(
                "INSERT INTO dedup(key, digest, ts) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET ts=excluded.ts",
                (key, str(digest), now),
            )
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
        if self._closed:
            return
        try:
            with self._lock:
                self._conn.close()
                self._closed = True
        except Exception as e:
            logger.error("Failed to close dedup store connection: %s", e, exc_info=True)

    def __enter__(self) -> "SQLiteDedupStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

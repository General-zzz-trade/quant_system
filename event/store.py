from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from typing import Iterable

from event.types import BaseEvent
from event.errors import EventFatalError

from _quant_hotpath import RustInMemoryEventStore as _RustInMemoryEventStore


# ============================================================
# Event Store Interface
# ============================================================

class EventStore(ABC):
    """
    EventStore —— 事件事实仓库（BaseEvent-only）

    铁律：
    - 只存 BaseEvent（事实）
    - 不编码、不解码
    - 不接触 dict / payload
    """

    @abstractmethod
    def append(self, event: BaseEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def iter_events(self) -> Iterable[BaseEvent]:
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError


# ============================================================
# In-Memory Store (冻结参考实现)
# ============================================================

class InMemoryEventStore(EventStore):
    """In-memory event store backed by Rust."""

    def __init__(self) -> None:
        self._rust = _RustInMemoryEventStore()

    def append(self, event: BaseEvent) -> None:
        if not isinstance(event, BaseEvent):
            raise EventFatalError(
                f"EventStore.append only accepts BaseEvent, got {type(event)}"
            )
        self._rust.append(event)

    def iter_events(self) -> Iterable[BaseEvent]:
        return tuple(self._rust.iter_events())

    def size(self) -> int:
        return int(self._rust.size())


# ============================================================
# SQLite Store (持久化实现)
# ============================================================

class SQLiteEventStore(EventStore):
    """SQLite-backed event store for production use.

    Uses WAL mode + synchronous=NORMAL for safe concurrent access.
    Events are serialized via EventCodecRegistry for durable storage.
    """

    def __init__(self, path: str) -> None:
        import os
        import sqlite3
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id   TEXT,
                    event_type TEXT NOT NULL,
                    version    INTEGER NOT NULL,
                    ts_iso     TEXT,
                    payload    TEXT NOT NULL,
                    saved_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            self._conn.commit()

    def append(self, event: BaseEvent) -> None:
        if not isinstance(event, BaseEvent):
            raise EventFatalError(
                f"EventStore.append only accepts BaseEvent, got {type(event)}"
            )
        from event.codec import encode_event_json
        payload_json = encode_event_json(event)
        header = event.header
        event_id = getattr(header, "event_id", None)
        ts = getattr(header, "ts", None)
        ts_iso = ts.isoformat() if ts is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(event_id, event_type, version, ts_iso, payload) VALUES(?,?,?,?,?)",
                (event_id, event.event_type.value, event.version, ts_iso, payload_json),
            )
            self._conn.commit()

    def iter_events(self) -> Iterable[BaseEvent]:
        from event.codec import decode_event_json
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM events ORDER BY id ASC"
            ).fetchall()
        return tuple(decode_event_json(row[0]) for row in rows)

    def size(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteEventStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ============================================================
# Utilities（可选）
# ============================================================

def assert_store_integrity(store: EventStore) -> None:
    """
    开发期校验：确认 store 内部只存 BaseEvent
    """
    for e in store.iter_events():
        if not isinstance(e, BaseEvent):
            raise EventFatalError(
                f"Store integrity violation: {type(e)} is not BaseEvent"
            )

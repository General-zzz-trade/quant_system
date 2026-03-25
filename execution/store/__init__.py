"""execution.store — Durable persistence for execution layer (Domain 3: core plumbing).

Provides restart-safe idempotency (AckStore) and append-only event logging
(EventLog) that enable exactly-once execution semantics across restarts.

Sub-modules:
  interfaces   Protocol definitions (AckStore, DedupStore, EventLog)
  ack_store    InMemoryAckStore (Rust-backed) + SQLiteAckStore
  event_log    InMemoryEventLog + SQLiteEventLog
"""
from execution.store.interfaces import AckStore, DedupStore, EventLog
from execution.store.ack_store import InMemoryAckStore, SQLiteAckStore
from execution.store.event_log import InMemoryEventLog, SQLiteEventLog

__all__ = [
    # Protocols
    "AckStore",
    "DedupStore",
    "EventLog",
    # Implementations
    "InMemoryAckStore",
    "SQLiteAckStore",
    "InMemoryEventLog",
    "SQLiteEventLog",
]

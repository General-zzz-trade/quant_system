from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol


class AckStore(Protocol):
    """Durable (or in-memory) idempotency store for execution acks."""

    def get(self, key: str) -> Optional[Mapping[str, Any]]: ...
    def put(self, key: str, value: Mapping[str, Any]) -> None: ...
    def prune(self) -> int: ...




class DedupStore(Protocol):
    """Deduplicate by key -> payload digest.

    Used for restart-safe ingestion of fills / order updates.
    """

    def get(self, key: str) -> Optional[str]: ...
    def put(self, key: str, digest: str) -> None: ...
    def prune(self) -> int: ...


class EventLog(Protocol):
    """Append-only log for execution-level events (acks, ingress snapshots, reconcile reports)."""

    def append(self, *, event_type: str, payload: Mapping[str, Any], correlation_id: Optional[str] = None) -> int: ...
    def iter(self, *, after_id: int = 0) -> Iterable[Mapping[str, Any]]: ...

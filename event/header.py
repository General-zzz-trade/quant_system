# event/header.py
"""EventHeader — Python header with Rust conversion support."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import EventType

from _quant_hotpath import (
    rust_event_id as _rust_event_id,
    rust_now_ns as _rust_now_ns,
    RustEventHeader as _RustEventHeader,
)


def _now_ns() -> int:
    return int(_rust_now_ns())


@dataclass(frozen=True)
class EventHeader:
    event_id: str
    event_type: EventType
    version: int

    ts_ns: int

    source: str

    parent_event_id: Optional[str] = None
    root_event_id: Optional[str] = None

    run_id: Optional[str] = None
    seq: Optional[int] = None
    correlation_id: Optional[str] = None

    # ---------- constructors ----------

    @staticmethod
    def new_root(
        *,
        event_type: EventType,
        version: int,
        source: str,
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> EventHeader:
        eid = _rust_event_id()
        return EventHeader(
            event_id=eid,
            event_type=event_type,
            version=version,
            ts_ns=_now_ns(),
            source=source,
            parent_event_id=None,
            root_event_id=eid,
            run_id=run_id,
            correlation_id=correlation_id,
        )

    @staticmethod
    def from_parent(
        *,
        parent: EventHeader,
        event_type: EventType,
        version: int,
        source: str,
    ) -> EventHeader:
        return EventHeader(
            event_id=_rust_event_id(),
            event_type=event_type,
            version=version,
            ts_ns=_now_ns(),
            source=source,
            parent_event_id=parent.event_id,
            root_event_id=parent.root_event_id,
            run_id=parent.run_id,
            correlation_id=parent.correlation_id,
        )

    # ---------- Rust conversion ----------

    def to_rust(self) -> _RustEventHeader:
        """Convert to RustEventHeader for FFI calls."""
        return _RustEventHeader(
            event_id=self.event_id,
            event_type=self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            version=self.version,
            ts_ns=self.ts_ns,
            source=self.source,
            parent_event_id=self.parent_event_id,
            root_event_id=self.root_event_id,
            run_id=self.run_id,
            seq=self.seq,
            correlation_id=self.correlation_id,
        )

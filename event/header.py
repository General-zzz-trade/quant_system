# event/header.py
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .types import EventType
from .errors import EventValidationError


def _now_ns() -> int:
    return int(time.time() * 1e9)


@dataclass(frozen=True)
class EventHeader:
    event_id: str
    event_type: EventType
    version: int

    ts_ns: int

    source: str

    parent_event_id: Optional[str] = None
    root_event_id: Optional[str] = None

    # 机构级扩展字段（runtime 注入）
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
        eid = str(uuid.uuid4())
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
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            version=version,
            ts_ns=_now_ns(),
            source=source,
            parent_event_id=parent.event_id,
            root_event_id=parent.root_event_id,
            run_id=parent.run_id,
            correlation_id=parent.correlation_id,
        )

    # ---------- validation ----------

    def validate(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise EventValidationError("header.event_type must be EventType")

        if not isinstance(self.version, int) or self.version < 1:
            raise EventValidationError("header.version must be int >= 1")

        if not isinstance(self.ts_ns, int):
            raise EventValidationError("header.ts_ns must be int (ns)")

        if not self.event_id:
            raise EventValidationError("header.event_id missing")

        if not self.root_event_id:
            raise EventValidationError("header.root_event_id missing")

    # ---------- serialization ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "version": self.version,
            "ts_ns": self.ts_ns,
            "source": self.source,
            "parent_event_id": self.parent_event_id,
            "root_event_id": self.root_event_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "correlation_id": self.correlation_id,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> EventHeader:
        return EventHeader(
            event_id=d["event_id"],
            event_type=EventType(d["event_type"]),
            version=d["version"],
            ts_ns=d["ts_ns"],
            source=d["source"],
            parent_event_id=d.get("parent_event_id"),
            root_event_id=d.get("root_event_id"),
            run_id=d.get("run_id"),
            seq=d.get("seq"),
            correlation_id=d.get("correlation_id"),
        )

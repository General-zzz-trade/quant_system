# context/audit/audit_log.py
"""Context audit log — track all state mutations."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """审计条目。"""
    seq: int
    ts: float
    event_type: str
    context_id: str
    snapshot_id: str
    detail: str = ""


class ContextAuditLog:
    """Context 级审计日志。"""

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: List[AuditEntry] = []
        self._seq = 0
        self._max = max_entries

    def record(
        self,
        *,
        event_type: str,
        context_id: str,
        snapshot_id: str = "",
        detail: str = "",
    ) -> AuditEntry:
        self._seq += 1
        entry = AuditEntry(
            seq=self._seq, ts=time.time(),
            event_type=event_type,
            context_id=context_id,
            snapshot_id=snapshot_id,
            detail=detail,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]
        return entry

    def query(self, *, after_seq: int = 0, limit: int = 100) -> Sequence[AuditEntry]:
        return [e for e in self._entries if e.seq > after_seq][:limit]

    @property
    def count(self) -> int:
        return len(self._entries)

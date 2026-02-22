# execution/observability/audit_log.py
"""Execution audit log — append-only record of all execution events."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """审计日志条目。"""
    entry_id: int
    ts: float
    event_type: str
    venue: str
    symbol: str
    actor: str
    payload: Mapping[str, Any]
    correlation_id: str = ""


class AuditLog:
    """
    执行审计日志 — 不可变、追加写入。

    记录所有订单操作、成交、错误，用于事后审计和问题追踪。
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []
        self._seq = 0

    def append(
        self,
        *,
        event_type: str,
        venue: str = "",
        symbol: str = "",
        actor: str = "",
        payload: Optional[Mapping[str, Any]] = None,
        correlation_id: str = "",
    ) -> AuditEntry:
        self._seq += 1
        entry = AuditEntry(
            entry_id=self._seq, ts=time.time(),
            event_type=event_type, venue=venue, symbol=symbol,
            actor=actor, payload=payload or {},
            correlation_id=correlation_id,
        )
        self._entries.append(entry)
        return entry

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        venue: Optional[str] = None,
        symbol: Optional[str] = None,
        after_id: int = 0,
        limit: int = 100,
    ) -> Sequence[AuditEntry]:
        results: list[AuditEntry] = []
        for e in self._entries:
            if e.entry_id <= after_id:
                continue
            if event_type and e.event_type != event_type:
                continue
            if venue and e.venue != venue:
                continue
            if symbol and e.symbol != symbol:
                continue
            results.append(e)
            if len(results) >= limit:
                break
        return results

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def last_entry(self) -> Optional[AuditEntry]:
        return self._entries[-1] if self._entries else None

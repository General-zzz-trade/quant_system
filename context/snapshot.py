# context/snapshot.py
"""Snapshot utilities — helpers for context snapshot management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from context.context import ContextSnapshot


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """快照存储记录。"""
    snapshot: ContextSnapshot
    seq: int


class SnapshotHistory:
    """
    快照历史 — 保留最近 N 个快照用于回放/调试。
    """

    def __init__(self, max_size: int = 100) -> None:
        self._max = max_size
        self._records: list[SnapshotRecord] = []
        self._seq = 0

    def append(self, snapshot: ContextSnapshot) -> SnapshotRecord:
        self._seq += 1
        record = SnapshotRecord(snapshot=snapshot, seq=self._seq)
        self._records.append(record)
        if len(self._records) > self._max:
            self._records = self._records[-self._max:]
        return record

    @property
    def latest(self) -> Optional[ContextSnapshot]:
        if not self._records:
            return None
        return self._records[-1].snapshot

    @property
    def count(self) -> int:
        return len(self._records)

    def get_by_id(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        for r in self._records:
            if r.snapshot.snapshot_id == snapshot_id:
                return r.snapshot
        return None

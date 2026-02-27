"""Latency tracker — record timestamps across pipeline stages for each order."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LatencyRecord:
    """Timestamp record for a single order across pipeline stages."""

    order_id: str
    signal_ts: Optional[float] = None
    decision_ts: Optional[float] = None
    submit_ts: Optional[float] = None
    ack_ts: Optional[float] = None
    fill_ts: Optional[float] = None

    @property
    def signal_to_fill_ms(self) -> Optional[float]:
        if self.signal_ts is not None and self.fill_ts is not None:
            return (self.fill_ts - self.signal_ts) * 1000
        return None

    @property
    def submit_to_ack_ms(self) -> Optional[float]:
        if self.submit_ts is not None and self.ack_ts is not None:
            return (self.ack_ts - self.submit_ts) * 1000
        return None

    @property
    def signal_to_decision_ms(self) -> Optional[float]:
        if self.signal_ts is not None and self.decision_ts is not None:
            return (self.decision_ts - self.signal_ts) * 1000
        return None

    @property
    def decision_to_submit_ms(self) -> Optional[float]:
        if self.decision_ts is not None and self.submit_ts is not None:
            return (self.submit_ts - self.decision_ts) * 1000
        return None

    @property
    def ack_to_fill_ms(self) -> Optional[float]:
        if self.ack_ts is not None and self.fill_ts is not None:
            return (self.fill_ts - self.ack_ts) * 1000
        return None


class LatencyTracker:
    """Track execution latency across pipeline stages.

    Thread-safe. Evicts oldest records when capacity is reached.
    """

    def __init__(self, *, max_records: int = 10000) -> None:
        self._records: OrderedDict[str, LatencyRecord] = OrderedDict()
        self._max_records = max_records
        self._lock = threading.Lock()

    def _ensure_record(self, order_id: str) -> LatencyRecord:
        """Get or create a record for order_id."""
        if order_id not in self._records:
            if len(self._records) >= self._max_records:
                self._records.popitem(last=False)
            self._records[order_id] = LatencyRecord(order_id=order_id)
        return self._records[order_id]

    def record_signal(self, order_id: str) -> None:
        with self._lock:
            rec = self._ensure_record(order_id)
            self._records[order_id] = replace(rec, signal_ts=time.monotonic())

    def record_decision(self, order_id: str) -> None:
        with self._lock:
            rec = self._ensure_record(order_id)
            self._records[order_id] = replace(rec, decision_ts=time.monotonic())

    def record_submit(self, order_id: str) -> None:
        with self._lock:
            rec = self._ensure_record(order_id)
            self._records[order_id] = replace(rec, submit_ts=time.monotonic())

    def record_ack(self, order_id: str) -> None:
        with self._lock:
            rec = self._ensure_record(order_id)
            self._records[order_id] = replace(rec, ack_ts=time.monotonic())

    def record_fill(self, order_id: str) -> None:
        with self._lock:
            rec = self._ensure_record(order_id)
            self._records[order_id] = replace(rec, fill_ts=time.monotonic())

    def get(self, order_id: str) -> Optional[LatencyRecord]:
        with self._lock:
            return self._records.get(order_id)

    def all_records(self) -> list[LatencyRecord]:
        with self._lock:
            return list(self._records.values())

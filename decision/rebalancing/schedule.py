"""Rebalance scheduling — time-based and bar-count-based triggers.

Controls when the decision engine should produce new targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Protocol

from state.snapshot import StateSnapshot


class RebalanceSchedule(Protocol):
    """Protocol for rebalance timing decisions."""
    def should_rebalance(self, snapshot: StateSnapshot) -> bool: ...


@dataclass(frozen=True, slots=True)
class AlwaysRebalance:
    """Rebalance on every snapshot (useful for testing)."""
    def should_rebalance(self, snapshot: StateSnapshot) -> bool:
        return True


@dataclass
class BarCountSchedule:
    """Rebalance every N bars."""
    interval: int = 1
    _count: int = 0

    def should_rebalance(self, snapshot: StateSnapshot) -> bool:
        self._count += 1
        if self._count >= self.interval:
            self._count = 0
            return True
        return False


@dataclass
class TimeIntervalSchedule:
    """Rebalance at fixed time intervals (e.g., every 4 hours).

    Uses the snapshot's ``ts`` field to determine elapsed time.
    """
    interval: timedelta = timedelta(hours=4)
    _last_ts: Optional[datetime] = None

    def should_rebalance(self, snapshot: StateSnapshot) -> bool:
        ts = getattr(snapshot, "ts", None)
        if ts is None:
            return True

        if self._last_ts is None:
            self._last_ts = ts
            return True

        if ts - self._last_ts >= self.interval:
            self._last_ts = ts
            return True

        return False

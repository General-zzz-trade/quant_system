"""Unified clock abstraction — the sole time source in the system.

Every module that needs ``now()`` or ``sleep()`` receives a ``Clock``
instance via dependency injection.  This enables:

  * Deterministic backtesting (``SimulatedClock``)
  * Replay from event streams (``ReplayClock``)
  * NTP-synced production (``SystemClock``)
  * Fully controllable unit tests (``FakeClock``)
"""
from __future__ import annotations

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

_log = logging.getLogger(__name__)


# ── Protocol ─────────────────────────────────────────────

class Clock(Protocol):
    """Minimal clock contract used throughout the system."""

    def now(self) -> datetime:
        """Current wall-clock time (timezone-aware UTC)."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds — safe for latency measurement."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block for *seconds* (real or simulated)."""
        ...


# ── System clock (production) ────────────────────────────

class SystemClock:
    """Real wall clock — for live trading."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return _time.monotonic()

    def sleep(self, seconds: float) -> None:
        _time.sleep(seconds)


# ── Simulated clock (backtest / unit test) ───────────────

@dataclass
class SimulatedClock:
    """Manually-controlled clock — ``advance()`` or ``set()`` to move time.

    ``sleep()`` is a no-op by default (instant); set ``real_sleep=True``
    if you want to block for debugging.
    """

    _current: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    _mono: float = field(default=0.0)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    real_sleep: bool = False

    def now(self) -> datetime:
        with self._lock:
            return self._current

    def monotonic(self) -> float:
        with self._lock:
            return self._mono

    def sleep(self, seconds: float) -> None:
        if self.real_sleep:
            _time.sleep(seconds)
        else:
            self.advance(timedelta(seconds=seconds))

    def advance(self, delta: timedelta) -> None:
        """Move time forward by *delta*."""
        with self._lock:
            self._current += delta
            self._mono += delta.total_seconds()

    def set(self, t: datetime) -> None:
        """Jump to an absolute time.  Monotonic clock advances accordingly."""
        with self._lock:
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            delta = (t - self._current).total_seconds()
            self._mono += max(0.0, delta)
            self._current = t


# ── Replay clock (event-driven) ──────────────────────────

@dataclass
class ReplayClock:
    """Clock driven by event timestamps — for deterministic replay.

    Call ``feed(ts)`` with each event's timestamp; ``now()`` returns
    the most recently fed value.  ``sleep()`` is a no-op.
    """

    _current: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    _mono: float = field(default=0.0)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def now(self) -> datetime:
        with self._lock:
            return self._current

    def monotonic(self) -> float:
        with self._lock:
            return self._mono

    def sleep(self, seconds: float) -> None:
        pass  # no-op in replay

    def feed(self, ts: datetime) -> None:
        """Advance clock to *ts* (must be >= current)."""
        with self._lock:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta = (ts - self._current).total_seconds()
            if delta < 0:
                _log.warning(
                    "ReplayClock: non-monotonic feed — ts=%s is %.1fs before current=%s, ignoring",
                    ts.isoformat(), abs(delta), self._current.isoformat(),
                )
                return
            if delta > 0:
                self._mono += delta
                self._current = ts

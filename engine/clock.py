# engine/clock.py
"""Unified clock module — merges engine clock (ClockMode/ReplayClock/LiveClock)
with core clock (CoreClock protocol, CoreSystemClock, CoreSimulatedClock, CoreReplayClock).

CoreSystemClock and CoreSimulatedClock delegate to Rust equivalents.
Engine clocks (ReplayClock, LiveClock) remain pure Python.
"""
from __future__ import annotations

import logging
import time
import time as _time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from _quant_hotpath import RustSimulatedClock, RustSystemClock

_log = logging.getLogger(__name__)


# ============================================================
# Engine Clock API (advance_to / advance_by)
# ============================================================

class ClockMode(str, Enum):
    LIVE = "live"
    REPLAY = "replay"


@runtime_checkable
class Clock(Protocol):
    @property
    def mode(self) -> ClockMode: ...
    def now(self) -> Any: ...
    def set(self, ts: Any) -> None: ...
    def advance_to(self, ts: Any) -> None: ...
    def advance_by(self, seconds: float) -> None: ...


class ClockError(RuntimeError):
    pass

class ClockMonotonicError(ClockError):
    pass

class ClockImmutableError(ClockError):
    pass


def _to_float_seconds(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp") and callable(getattr(ts, "timestamp")):
        try:
            return float(ts.timestamp())
        except Exception:
            return None
    for attr in ("ts", "timestamp"):
        if hasattr(ts, attr):
            v = getattr(ts, attr)
            if isinstance(v, (int, float)):
                return float(v)
            if hasattr(v, "timestamp") and callable(getattr(v, "timestamp")):
                try:
                    return float(v.timestamp())
                except Exception:
                    return None
    return None


@dataclass(slots=True)
class ReplayClock:
    """Replay mode clock: controllable, monotonic."""
    _ts: Any = None

    @property
    def mode(self) -> ClockMode:
        return ClockMode.REPLAY

    def now(self) -> Any:
        return self._ts

    def set(self, ts: Any) -> None:
        self.advance_to(ts)

    def advance_to(self, ts: Any) -> None:
        if ts is None:
            return
        if self._ts is None:
            self._ts = ts
            return
        old = _to_float_seconds(self._ts)
        new = _to_float_seconds(ts)
        if old is None or new is None:
            if str(ts) < str(self._ts):
                raise ClockMonotonicError(f"ReplayClock time moved backwards: {self._ts} -> {ts}")
            self._ts = ts
            return
        if new < old:
            raise ClockMonotonicError(f"ReplayClock time moved backwards: {self._ts} -> {ts}")
        self._ts = ts

    def advance_by(self, seconds: float) -> None:
        if seconds < 0:
            raise ClockMonotonicError(f"ReplayClock cannot advance by negative seconds: {seconds}")
        if self._ts is None:
            self._ts = float(seconds)
            return
        old = _to_float_seconds(self._ts)
        if old is None:
            self._ts = f"{self._ts}+{seconds}s"
            return
        self._ts = old + float(seconds)


@dataclass(slots=True)
class LiveClock:
    """Live mode clock: immutable, read-only."""
    time_fn: Any = time.time

    @property
    def mode(self) -> ClockMode:
        return ClockMode.LIVE

    def now(self) -> float:
        return float(self.time_fn())

    def set(self, ts: Any) -> None:
        raise ClockImmutableError("LiveClock is immutable; use ReplayClock for controllable time.")

    def advance_to(self, ts: Any) -> None:
        raise ClockImmutableError("LiveClock is immutable; use ReplayClock for controllable time.")

    def advance_by(self, seconds: float) -> None:
        raise ClockImmutableError("LiveClock is immutable; use ReplayClock for controllable time.")


# ============================================================
# Core Clock API (now/monotonic/sleep — datetime-based)
# CoreSystemClock and CoreSimulatedClock delegate to Rust.
# ============================================================

class CoreClock(Protocol):
    """Minimal clock contract used by Effects and infrastructure."""
    def now(self) -> datetime: ...
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class CoreSystemClock:
    """Real wall clock — delegates to RustSystemClock for now/monotonic."""

    __slots__ = ("_rust",)

    def __init__(self) -> None:
        self._rust = RustSystemClock()

    def now(self) -> datetime:
        return datetime.fromisoformat(self._rust.now().replace("Z", "+00:00"))

    def monotonic(self) -> float:
        return self._rust.monotonic()

    def sleep(self, seconds: float) -> None:
        _time.sleep(seconds)


class CoreSimulatedClock:
    """Manually-controlled clock — delegates to RustSimulatedClock.

    ``sleep()`` is a no-op by default (instant); set ``real_sleep=True``
    if you want to block for debugging.
    """

    __slots__ = ("_rust", "real_sleep")

    def __init__(self, *, real_sleep: bool = False,
                 start: Optional[datetime] = None) -> None:
        self._rust = RustSimulatedClock()
        self.real_sleep = real_sleep
        if start is not None:
            self.set(start)

    def now(self) -> datetime:
        return datetime.fromisoformat(self._rust.now().replace("Z", "+00:00"))

    def monotonic(self) -> float:
        return self._rust.monotonic()

    def sleep(self, seconds: float) -> None:
        if self.real_sleep:
            _time.sleep(seconds)
        else:
            self.advance(timedelta(seconds=seconds))

    def advance(self, delta: timedelta) -> None:
        """Move time forward by *delta*."""
        self._rust.advance(delta.total_seconds())

    def set(self, t: datetime) -> None:
        """Jump to an absolute time."""
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        self._rust.set(t.timestamp())


@dataclass
class CoreReplayClock:
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
                    "CoreReplayClock: non-monotonic feed — ts=%s is %.1fs before current=%s, ignoring",
                    ts.isoformat(), abs(delta), self._current.isoformat(),
                )
                return
            if delta > 0:
                self._mono += delta
                self._current = ts

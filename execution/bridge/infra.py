"""Infrastructure classes for ExecutionBridge: token bucket, circuit breaker, configs.

Extracted from execution_bridge.py to keep it under 500 lines.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


# -----------------------------
# Configs
# -----------------------------
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_sec: float = 0.10
    max_delay_sec: float = 2.00
    jitter_sec: float = 0.05


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    rate_per_sec: float = 10.0
    burst: float = 10.0


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 8
    window_sec: float = 10.0
    cooldown_sec: float = 5.0
    max_consecutive_trips: int = 5  # permanent halt after N consecutive trips


# -----------------------------
# Small infrastructure
# -----------------------------
class Clock(Protocol):
    def now(self) -> float: ...


class Sleeper(Protocol):
    def sleep(self, sec: float) -> None: ...


@dataclass(slots=True)
class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class RealSleeper:
    def sleep(self, sec: float) -> None:
        time.sleep(sec)


@dataclass(slots=True)
class TokenBucket:
    rate_per_sec: float
    burst: float
    clock: Clock

    tokens: float = field(init=False)
    last_ts: float = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.burst)
        self.last_ts = self.clock.now()
        self._lock = threading.Lock()

    def allow(self, n: float = 1.0) -> bool:
        with self._lock:
            now = self.clock.now()
            dt = max(0.0, now - self.last_ts)
            self.last_ts = now

            self.tokens = min(self.burst, self.tokens + dt * self.rate_per_sec)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


@dataclass(slots=True)
class CircuitBreaker:
    cfg: CircuitBreakerConfig
    clock: Clock
    _fail_ts: list[float] = field(default_factory=list)
    _open_until: float = 0.0
    _consecutive_trips: int = field(default=0, init=False)
    _permanently_open: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def is_open(self) -> bool:
        with self._lock:
            if self._permanently_open:
                return True
            return self.clock.now() < self._open_until

    @property
    def permanently_halted(self) -> bool:
        with self._lock:
            return self._permanently_open

    def record_success(self) -> None:
        with self._lock:
            self._prune()
            self._consecutive_trips = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._permanently_open:
                return
            now = self.clock.now()
            self._fail_ts.append(now)
            self._prune()
            if len(self._fail_ts) >= self.cfg.failure_threshold:
                self._consecutive_trips += 1
                if self._consecutive_trips >= self.cfg.max_consecutive_trips:
                    self._permanently_open = True
                    logger.critical(
                        "CircuitBreaker permanently halted after %d consecutive trips",
                        self._consecutive_trips,
                    )
                else:
                    self._open_until = now + self.cfg.cooldown_sec

    def _prune(self) -> None:
        """Must be called with self._lock held."""
        now = self.clock.now()
        w = self.cfg.window_sec
        if not self._fail_ts:
            return
        self._fail_ts = [t for t in self._fail_ts if (now - t) <= w]

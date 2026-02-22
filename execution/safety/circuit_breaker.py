# execution/safety/circuit_breaker.py
"""Reusable circuit breaker for venue-level failure isolation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from time import monotonic
from typing import Optional, Tuple


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    window_seconds: float = 60.0
    cooldown_seconds: float = 30.0
    half_open_max: int = 1


class CircuitBreaker:
    """
    机构级熔断器

    CLOSED → (failures >= threshold) → OPEN → (cooldown) → HALF_OPEN → success → CLOSED
                                                                      → failure → OPEN
    """

    def __init__(self, cfg: Optional[CircuitBreakerConfig] = None) -> None:
        self._cfg = cfg or CircuitBreakerConfig()
        self._lock = RLock()
        self._state = BreakerState.CLOSED
        self._failures: list[float] = []
        self._open_since: float = 0.0
        self._half_open_count: int = 0

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_transition()
            return self._state

    def allow_request(self) -> bool:
        with self._lock:
            self._maybe_transition()
            if self._state == BreakerState.CLOSED:
                return True
            if self._state == BreakerState.HALF_OPEN:
                if self._half_open_count < self._cfg.half_open_max:
                    self._half_open_count += 1
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.CLOSED
            self._failures.clear()
            self._half_open_count = 0

    def record_failure(self) -> None:
        now = monotonic()
        with self._lock:
            self._failures.append(now)
            cutoff = now - self._cfg.window_seconds
            self._failures = [t for t in self._failures if t > cutoff]

            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.OPEN
                self._open_since = now
                self._half_open_count = 0
            elif self._state == BreakerState.CLOSED:
                if len(self._failures) >= self._cfg.failure_threshold:
                    self._state = BreakerState.OPEN
                    self._open_since = now

    def reset(self) -> None:
        with self._lock:
            self._state = BreakerState.CLOSED
            self._failures.clear()
            self._open_since = 0.0
            self._half_open_count = 0

    def snapshot(self) -> Tuple[BreakerState, int, float]:
        with self._lock:
            self._maybe_transition()
            return self._state, len(self._failures), self._open_since

    def _maybe_transition(self) -> None:
        if self._state == BreakerState.OPEN:
            elapsed = monotonic() - self._open_since
            if elapsed >= self._cfg.cooldown_seconds:
                self._state = BreakerState.HALF_OPEN
                self._half_open_count = 0

# execution/safety/circuit_breaker.py
"""Reusable circuit breaker for venue-level failure isolation.

Delegates to RustCircuitBreaker for lock-free, high-performance state management.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from _quant_hotpath import RustCircuitBreaker as _RustCircuitBreaker


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
    Rust-backed circuit breaker.

    CLOSED -> (failures >= threshold) -> OPEN -> (cooldown) -> HALF_OPEN -> success -> CLOSED
                                                                         -> failure -> OPEN
    """

    def __init__(self, cfg: Optional[CircuitBreakerConfig] = None) -> None:
        self._cfg = cfg or CircuitBreakerConfig()
        self._rust = _RustCircuitBreaker(
            failure_threshold=self._cfg.failure_threshold,
            window_s=self._cfg.window_seconds,
            recovery_timeout_s=self._cfg.cooldown_seconds,
        )

    @property
    def state(self) -> BreakerState:
        return BreakerState(self._rust.state)

    def allow_request(self) -> bool:
        return bool(self._rust.allow_request())

    def record_success(self) -> None:
        self._rust.record_success()

    def record_failure(self) -> None:
        self._rust.record_failure()

    def reset(self) -> None:
        self._rust.reset()

    def snapshot(self) -> Tuple[BreakerState, int, float]:
        state_str, failures, open_since = self._rust.snapshot()
        return BreakerState(state_str), failures, open_since

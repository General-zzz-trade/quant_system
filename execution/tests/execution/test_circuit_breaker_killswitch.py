"""Tests for circuit breaker killswitch."""
from __future__ import annotations

from execution.safety.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    BreakerState,
)


def test_circuit_breaker_initial_closed():
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    assert cb.state == BreakerState.CLOSED


def test_circuit_breaker_opens_on_failures():
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, window_seconds=60))
    cb.record_failure()
    cb.record_failure()
    assert cb.state == BreakerState.OPEN


def test_circuit_breaker_success_resets():
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    cb.record_failure()
    cb.record_success()
    assert cb.state == BreakerState.CLOSED

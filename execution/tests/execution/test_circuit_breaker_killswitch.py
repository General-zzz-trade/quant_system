"""Tests for circuit breaker killswitch."""
from __future__ import annotations

import time
from unittest.mock import patch

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


def test_circuit_breaker_recovers_after_cooldown():
    """After cooldown expires, circuit breaker allows traffic again."""
    # Use a very short cooldown so we can test transition
    cfg = CircuitBreakerConfig(
        failure_threshold=2,
        window_seconds=60.0,
        cooldown_seconds=0.05,  # 50ms cooldown
    )
    cb = CircuitBreaker(cfg)

    # Record enough failures to trip the breaker
    cb.record_failure()
    cb.record_failure()
    assert cb.state == BreakerState.OPEN

    # Verify breaker blocks requests while open
    assert cb.allow_request() is False

    # Wait past cooldown
    time.sleep(0.1)

    # Breaker should transition to HALF_OPEN
    assert cb.state == BreakerState.HALF_OPEN

    # Should allow a probe request in half-open state
    assert cb.allow_request() is True

    # Record success to fully close
    cb.record_success()
    assert cb.state == BreakerState.CLOSED

    # Verify fully closed -- requests are allowed
    assert cb.allow_request() is True


def test_circuit_breaker_reopens_on_immediate_failure():
    """If request fails immediately after cooldown, breaker reopens."""
    cfg = CircuitBreakerConfig(
        failure_threshold=2,
        window_seconds=60.0,
        cooldown_seconds=0.05,  # 50ms cooldown
    )
    cb = CircuitBreaker(cfg)

    # Trip the breaker
    cb.record_failure()
    cb.record_failure()
    assert cb.state == BreakerState.OPEN

    # Wait past cooldown to reach HALF_OPEN
    time.sleep(0.1)
    assert cb.state == BreakerState.HALF_OPEN

    # Failure during half-open should reopen the breaker
    cb.record_failure()
    assert cb.state == BreakerState.OPEN

    # Verify breaker blocks again
    assert cb.allow_request() is False

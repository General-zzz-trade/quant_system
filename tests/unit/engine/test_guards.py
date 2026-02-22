# tests/unit/engine/test_guards.py
"""BasicGuard unit tests — error classification, threshold enforcement, counter management."""
from __future__ import annotations

import pytest

from engine.errors import (
    ClassifiedError,
    EngineErrorContext,
    ErrorDomain,
    ErrorSeverity,
    ExecutionError,
    FatalError,
    InvariantViolation,
    RetryableError,
)
from engine.guards import BasicGuard, GuardAction, GuardConfig, GuardDecision, build_basic_guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CTX = EngineErrorContext(actor="test")


# ---------------------------------------------------------------------------
# Tests: basic decisions
# ---------------------------------------------------------------------------

class TestBeforeAfter:
    def test_before_event_allows(self) -> None:
        guard = build_basic_guard()
        d = guard.before_event(object(), actor="live", ctx=_CTX)
        assert d.action == GuardAction.ALLOW

    def test_after_event_allows(self) -> None:
        guard = build_basic_guard()
        d = guard.after_event(object(), actor="live", ctx=_CTX)
        assert d.action == GuardAction.ALLOW

    def test_after_event_clears_counters(self) -> None:
        guard = build_basic_guard()
        # Trigger some errors first
        guard.on_error(ValueError("x"), actor="test", ctx=_CTX)
        guard.on_error(ValueError("y"), actor="test", ctx=_CTX)
        assert guard._consecutive_errors == 2
        # after_event should clear
        guard.after_event(object(), actor="test", ctx=_CTX)
        assert guard._consecutive_errors == 0
        assert guard._domain_consecutive == {}
        assert guard._execution_consecutive == 0


# ---------------------------------------------------------------------------
# Tests: fatal / invariant → STOP
# ---------------------------------------------------------------------------

class TestFatalStop:
    def test_fatal_error_stops(self) -> None:
        guard = build_basic_guard(GuardConfig(stop_on_fatal=True))
        d = guard.on_error(FatalError("crash"), actor="test", ctx=_CTX)
        assert d.action == GuardAction.STOP
        assert "fatal" in d.reason

    def test_fatal_no_stop_when_disabled(self) -> None:
        guard = build_basic_guard(GuardConfig(stop_on_fatal=False))
        d = guard.on_error(FatalError("crash"), actor="test", ctx=_CTX)
        # With stop_on_fatal=False, FatalError doesn't auto-stop,
        # but it still increments counters. Since max_consecutive_errors=5,
        # a single error won't trigger STOP. Should DROP.
        assert d.action == GuardAction.DROP

    def test_invariant_violation_always_stops(self) -> None:
        guard = build_basic_guard(GuardConfig(stop_on_fatal=False))
        d = guard.on_error(InvariantViolation("invariant broken"), actor="test", ctx=_CTX)
        assert d.action == GuardAction.STOP
        assert "invariant" in d.reason


# ---------------------------------------------------------------------------
# Tests: threshold enforcement
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_execution_error_threshold(self) -> None:
        guard = build_basic_guard(GuardConfig(max_consecutive_execution_errors=2))
        # First execution error: below threshold → DROP
        d1 = guard.on_error(ExecutionError("e1"), actor="test", ctx=_CTX)
        assert d1.action == GuardAction.DROP
        # Second execution error: at threshold → STOP
        d2 = guard.on_error(ExecutionError("e2"), actor="test", ctx=_CTX)
        assert d2.action == GuardAction.STOP

    def test_domain_consecutive_threshold(self) -> None:
        guard = build_basic_guard(GuardConfig(max_consecutive_domain_errors=3))
        for i in range(2):
            d = guard.on_error(ValueError(f"v{i}"), actor="test", ctx=_CTX)
            assert d.action == GuardAction.DROP
        # Third domain error hits threshold
        d3 = guard.on_error(ValueError("v3"), actor="test", ctx=_CTX)
        assert d3.action == GuardAction.STOP

    def test_global_consecutive_threshold(self) -> None:
        guard = build_basic_guard(GuardConfig(
            max_consecutive_errors=3,
            max_consecutive_domain_errors=100,  # high to avoid domain trigger
        ))
        for i in range(2):
            guard.on_error(ValueError(f"e{i}"), actor="test", ctx=_CTX)
        d = guard.on_error(ValueError("e3"), actor="test", ctx=_CTX)
        assert d.action == GuardAction.STOP
        assert "consecutive errors" in d.reason


# ---------------------------------------------------------------------------
# Tests: retryable
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retryable_error_suggests_retry(self) -> None:
        guard = build_basic_guard(GuardConfig(default_retry_after_s=0.5))
        d = guard.on_error(RetryableError("timeout"), actor="test", ctx=_CTX)
        assert d.action == GuardAction.RETRY
        assert d.retry_after_s == 0.5

    def test_timeout_error_suggests_retry(self) -> None:
        guard = build_basic_guard()
        d = guard.on_error(TimeoutError("timed out"), actor="test", ctx=_CTX)
        assert d.action == GuardAction.RETRY


# ---------------------------------------------------------------------------
# Tests: default DROP
# ---------------------------------------------------------------------------

class TestDefaultDrop:
    def test_regular_exception_drops(self) -> None:
        guard = build_basic_guard()
        d = guard.on_error(ValueError("minor"), actor="test", ctx=_CTX)
        assert d.action == GuardAction.DROP
        assert d.classified is not None

    def test_classified_error_attached(self) -> None:
        guard = build_basic_guard()
        d = guard.on_error(ValueError("v"), actor="test", ctx=_CTX)
        assert isinstance(d.classified, ClassifiedError)
        assert d.classified.domain == ErrorDomain.DATA


# ---------------------------------------------------------------------------
# Tests: counter tracking
# ---------------------------------------------------------------------------

class TestCounterTracking:
    def test_consecutive_errors_increment(self) -> None:
        guard = build_basic_guard(GuardConfig(max_consecutive_errors=100))
        for i in range(5):
            guard.on_error(ValueError(f"e{i}"), actor="test", ctx=_CTX)
        assert guard._consecutive_errors == 5

    def test_execution_consecutive_increments_separately(self) -> None:
        guard = build_basic_guard(GuardConfig(
            max_consecutive_errors=100,
            max_consecutive_execution_errors=100,
            max_consecutive_domain_errors=100,
        ))
        guard.on_error(ExecutionError("e1"), actor="test", ctx=_CTX)
        guard.on_error(ValueError("v1"), actor="test", ctx=_CTX)
        guard.on_error(ExecutionError("e2"), actor="test", ctx=_CTX)
        assert guard._execution_consecutive == 2
        assert guard._consecutive_errors == 3

    def test_success_resets_all_counters(self) -> None:
        guard = build_basic_guard(GuardConfig(
            max_consecutive_errors=100,
            max_consecutive_execution_errors=100,
        ))
        guard.on_error(ExecutionError("e1"), actor="test", ctx=_CTX)
        guard.on_error(ValueError("v1"), actor="test", ctx=_CTX)
        assert guard._consecutive_errors == 2
        # Successful event clears everything
        guard.after_event(object(), actor="test", ctx=_CTX)
        assert guard._consecutive_errors == 0
        assert guard._execution_consecutive == 0
        assert len(guard._domain_consecutive) == 0


# ---------------------------------------------------------------------------
# Tests: build_basic_guard helper
# ---------------------------------------------------------------------------

class TestBuildHelper:
    def test_default_config(self) -> None:
        guard = build_basic_guard()
        assert isinstance(guard, BasicGuard)

    def test_custom_config(self) -> None:
        cfg = GuardConfig(max_consecutive_errors=10)
        guard = build_basic_guard(cfg)
        assert guard._cfg.max_consecutive_errors == 10

"""Canonical error hierarchy for the entire system.

Every module raises a subclass of ``QuantError`` so that callers can
catch at the granularity they want.
"""
from __future__ import annotations


class QuantError(Exception):
    """Root exception for all quant-system errors."""


# ── Domain ───────────────────────────────────────────────

class StateError(QuantError):
    """State mutation or consistency violation."""


class RiskError(QuantError):
    """Risk limit breach or risk-engine failure."""


class DecisionError(QuantError):
    """Signal/strategy/allocation failure."""


# ── Infrastructure ───────────────────────────────────────

class ExecutionError(QuantError):
    """Order submission, cancellation, or reconciliation failure."""


class VenueError(ExecutionError):
    """Exchange/broker-specific failure."""


class RetryableVenueError(VenueError):
    """Transient venue error — caller should retry."""


class NonRetryableVenueError(VenueError):
    """Permanent venue error — do not retry."""


class ConfigError(QuantError):
    """Configuration loading, validation, or hot-reload error."""


# ── Pipeline ─────────────────────────────────────────────

class PipelineError(QuantError):
    """Event normalization or reducer-chain failure."""


class InterceptedError(PipelineError):
    """Pipeline interceptor rejected the event."""

    def __init__(self, interceptor: str, reason: str) -> None:
        self.interceptor = interceptor
        self.reason = reason
        super().__init__(f"Intercepted by {interceptor}: {reason}")

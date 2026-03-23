"""Canonical error hierarchy for the entire system.

Merged from core/errors.py + core/exceptions.py.

Every module raises a subclass of ``QuantError`` so that callers can
catch at the granularity they want.

Exception type -> response strategy:
- VenueError -> retry with backoff
- InsufficientMargin -> stop ordering, alert
- OrderRejected -> check parameters, alert
- ModelError -> fallback to last signal
- DataError -> skip bar, continue
- ReconcileError -> reconcile and alert
"""
from __future__ import annotations


# -- Root -----------------------------------------------------

class QuantError(Exception):
    """Root exception for all quant-system errors."""


# -- Domain ---------------------------------------------------

class StateError(QuantError):
    """State mutation or consistency violation."""


class RiskError(QuantError):
    """Risk limit breach or risk-engine failure."""


class DecisionError(QuantError):
    """Signal/strategy/allocation failure."""


# -- Trading --------------------------------------------------

class TradingError(QuantError):
    """Base class for all trading system errors."""


class ModelError(TradingError):
    """Model loading, inference, or prediction error."""


class DataError(TradingError):
    """Data fetching, feature computation, or data quality error."""


class ReconcileError(TradingError):
    """Position reconciliation mismatch between local and exchange state."""
    def __init__(self, message: str = "", local_pos: float = 0.0, exchange_pos: float = 0.0):
        self.local_pos = local_pos
        self.exchange_pos = exchange_pos
        super().__init__(message or f"Position mismatch: local={local_pos}, exchange={exchange_pos}")


# -- Infrastructure -------------------------------------------

class ExecutionError(QuantError):
    """Order submission, cancellation, or reconciliation failure."""


class VenueError(ExecutionError, TradingError):
    """Exchange/broker-specific failure."""


class RetryableVenueError(VenueError):
    """Transient venue error — caller should retry."""


class NonRetryableVenueError(VenueError):
    """Permanent venue error — do not retry."""


class InsufficientMargin(VenueError):
    """Insufficient margin/balance for order.

    Attributes:
        needed: Required margin amount.
        available: Available margin amount.
    """
    def __init__(self, message: str = "", needed: float = 0.0, available: float = 0.0):
        self.needed = needed
        self.available = available
        super().__init__(message or f"Insufficient margin: needed={needed}, available={available}")


class OrderRejected(VenueError):
    """Order rejected by exchange.

    Attributes:
        reason: Exchange rejection reason code/message.
        order_params: Original order parameters.
    """
    def __init__(self, message: str = "", reason: str = "", order_params: dict | None = None):
        self.reason = reason
        self.order_params = order_params or {}
        super().__init__(message or f"Order rejected: {reason}")


class ConfigError(QuantError):
    """Configuration loading, validation, or hot-reload error."""


# -- Pipeline -------------------------------------------------

class PipelineError(QuantError):
    """Event normalization or reducer-chain failure."""


class InterceptedError(PipelineError):
    """Pipeline interceptor rejected the event."""

    def __init__(self, interceptor: str, reason: str) -> None:
        self.interceptor = interceptor
        self.reason = reason
        super().__init__(f"Intercepted by {interceptor}: {reason}")

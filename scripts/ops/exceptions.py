"""Trading system exception hierarchy.

Defines structured exceptions for the AlphaRunner and related components.
Each exception type maps to a specific response strategy:

- VenueError → retry with backoff
- InsufficientMargin → stop ordering, alert
- OrderRejected → check parameters, alert
- ModelError → fallback to last signal
- DataError → skip bar, continue
- ReconcileError → reconcile and alert
"""
from __future__ import annotations


class TradingError(Exception):
    """Base class for all trading system errors."""
    pass


class VenueError(TradingError):
    """Exchange/venue communication error (timeout, network, rate limit)."""
    pass


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


class ModelError(TradingError):
    """Model loading, inference, or prediction error."""
    pass


class DataError(TradingError):
    """Data fetching, feature computation, or data quality error."""
    pass


class ReconcileError(TradingError):
    """Position reconciliation mismatch between local and exchange state."""
    def __init__(self, message: str = "", local_pos: float = 0.0, exchange_pos: float = 0.0):
        self.local_pos = local_pos
        self.exchange_pos = exchange_pos
        super().__init__(message or f"Position mismatch: local={local_pos}, exchange={exchange_pos}")

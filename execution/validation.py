"""Input validation utilities for production safety.

Migrated from core/validation.py.

Validates prices, quantities, signals, and feature dictionaries to prevent
NaN/Inf propagation through the trading pipeline.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def validate_price(price: float, context: str = "") -> float:
    """Validate a price value — must be finite and non-negative.

    Raises:
        ValueError: if price is NaN, Inf, or negative.
    """
    if not isinstance(price, (int, float)):
        raise ValueError(f"price must be numeric, got {type(price).__name__} [{context}]")
    price = float(price)
    if math.isnan(price):
        raise ValueError(f"price is NaN [{context}]")
    if math.isinf(price):
        raise ValueError(f"price is Inf [{context}]")
    if price < 0:
        raise ValueError(f"price is negative: {price} [{context}]")
    return price


def validate_qty(qty: float, context: str = "") -> float:
    """Validate an order quantity — must be finite and non-negative.

    Raises:
        ValueError: if qty is NaN, Inf, or negative.
    """
    if not isinstance(qty, (int, float)):
        raise ValueError(f"qty must be numeric, got {type(qty).__name__} [{context}]")
    qty = float(qty)
    if math.isnan(qty):
        raise ValueError(f"qty is NaN [{context}]")
    if math.isinf(qty):
        raise ValueError(f"qty is Inf [{context}]")
    if qty < 0:
        raise ValueError(f"qty is negative: {qty} [{context}]")
    return qty


def validate_signal(signal: int) -> int:
    """Validate a discrete trading signal — must be -1, 0, or +1.

    Raises:
        ValueError: if signal is not in {-1, 0, 1}.
    """
    if signal not in (-1, 0, 1):
        raise ValueError(f"signal must be -1, 0, or +1, got {signal!r}")
    return int(signal)


def sanitize_features(feat_dict: dict) -> dict:
    """Replace NaN/Inf values with 0.0, logging warnings for each replacement.

    Returns a new dict — the original is not mutated.
    """
    result = {}
    for key, value in feat_dict.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            logger.warning("sanitize_features: %s=%s replaced with 0.0", key, value)
            result[key] = 0.0
        else:
            result[key] = value
    return result

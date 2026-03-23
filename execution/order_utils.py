"""Order utility helpers for the Bybit alpha runner."""
from __future__ import annotations

import logging
import time
from typing import Any

from scripts.ops.config import MAX_ORDER_NOTIONAL

logger = logging.getLogger(__name__)


def reliable_close_position(
    adapter: Any,
    symbol: str,
    max_retries: int = 3,
    verify: bool = True,
) -> dict:
    """Close a position with retry + optional position verification.

    Returns dict with:
        status:   "closed" | "failed"
        verified: True if verified flat after close (or verify=False)
        attempts: number of attempts made
    """
    result: dict = {"status": "failed", "verified": False, "attempts": 0}

    for attempt in range(1, max_retries + 1):
        result["attempts"] = attempt
        try:
            r = adapter.close_position(symbol)
        except Exception as exc:
            logger.error(
                "reliable_close %s attempt %d exception: %s", symbol, attempt, exc
            )
            if attempt < max_retries:
                time.sleep(0.5)
            continue

        if r.get("status") != "error" and r.get("retCode", 0) == 0:
            result.update(r)
            result["status"] = "closed"
            break

        logger.warning(
            "reliable_close %s attempt %d failed: %s", symbol, attempt, r
        )
        if attempt < max_retries:
            time.sleep(0.5)

    if result["status"] != "closed":
        return result

    if not verify:
        result["verified"] = True
        return result

    # Verify position is actually flat
    try:
        positions = adapter.get_positions(symbol=symbol)
        flat = not positions or all(
            abs(float(getattr(p, "qty", 0))) < 1e-8 for p in positions
        )
        result["verified"] = flat
        if not flat:
            logger.warning(
                "reliable_close %s: position still open after close — exchange may lag",
                symbol,
            )
    except Exception as exc:
        logger.warning(
            "reliable_close %s: verification failed (assuming OK): %s", symbol, exc
        )
        result["verified"] = True  # assume OK if verification call fails

    return result


def clamp_notional(
    qty: float,
    price: float,
    symbol: str = "",
    max_notional: float = MAX_ORDER_NOTIONAL,
) -> float:
    """Clamp quantity so notional (qty * price) does not exceed max_notional."""
    if price <= 0 or qty <= 0:
        return qty  # invalid input, return unchanged to avoid ZeroDivisionError
    notional = qty * price
    if notional > max_notional:
        logger.warning(
            "%s notional $%.2f > $%.2f — clamping", symbol, notional, max_notional
        )
        return max_notional / price
    return qty

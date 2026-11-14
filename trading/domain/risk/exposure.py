"""Account gross exposure cap: total notional <= MAX_TOTAL_LEVERAGE x equity.

The cap is a code constant. Configuration may lower it but never raise it.
"""
from __future__ import annotations

from decimal import Decimal

MAX_TOTAL_LEVERAGE = Decimal("3")


def validated_leverage(value: Decimal | str | float) -> Decimal:
    """A configured cap must lie in (0, MAX_TOTAL_LEVERAGE]; anything else is a configuration error."""
    leverage = Decimal(str(value))
    if not leverage.is_finite() or not 0 < leverage <= MAX_TOTAL_LEVERAGE:
        raise ValueError(f"max_leverage must be in (0, {MAX_TOTAL_LEVERAGE}]")
    return leverage


def exposure_breach(equity: Decimal, gross: Decimal, proposed: Decimal, max_leverage: Decimal) -> str:
    """Why ``proposed`` notional may not be added, or "" when it may.

    Non-positive or non-finite equity and a non-positive proposal are refusals, never passes.
    """
    if not equity.is_finite() or equity <= 0:
        return "account equity unavailable or non-positive"
    if not proposed.is_finite() or proposed <= 0:
        return "invalid proposed notional"
    limit = equity * max_leverage
    if gross + proposed > limit:
        return f"gross exposure {gross + proposed:.2f} would exceed {max_leverage}x equity ({limit:.2f})"
    return ""

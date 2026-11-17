"""Account gross exposure cap: total notional <= the configured cap x equity, never above MAX_TOTAL_LEVERAGE.

The ceiling is a code constant. Configuration may set any cap up to it, never above; without a setting the cap is
DEFAULT_ACCOUNT_LEVERAGE.
"""
from __future__ import annotations

from decimal import Decimal

# Owner 2026-11-17: "改为资金的10倍，我要上调", then "放宽到 10 倍" after the 2-10x study
# (research/results/overlay_pilot_leverage_2026-11-17/, where every 10x setting lost 74-95 %). It was 3. It bounds
# STRATEGY_LEVERAGE and BINANCE_MAX_LEVERAGE; a person sets both, and nothing in the program raises them.
MAX_TOTAL_LEVERAGE = Decimal("10")
DEFAULT_ACCOUNT_LEVERAGE = Decimal("3")    # the account cap when no person set BINANCE_MAX_LEVERAGE (unchanged)


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

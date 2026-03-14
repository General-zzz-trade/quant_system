"""Kelly criterion position sizing for binary outcome markets."""
from __future__ import annotations


def kelly_size(
    estimated_prob: float,
    market_price: float,
    bankroll: float,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.10,
) -> float:
    """Compute position size using Kelly criterion for binary outcomes.

    estimated_prob: our estimate of true probability (0-1)
    market_price: current YES share price (0-1)
    bankroll: total available capital
    kelly_fraction: conservative multiplier (0.5 = half-Kelly)
    max_position_pct: maximum fraction of bankroll per market

    Returns: dollar amount to invest (0 if no edge)
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if estimated_prob <= 0 or estimated_prob >= 1:
        return 0.0

    # Kelly formula for binary outcomes
    # f = (p * b - q) / b where b = odds = (1 - price) / price
    b = (1.0 - market_price) / market_price  # odds
    p = estimated_prob
    q = 1.0 - p

    f = (p * b - q) / b

    if f <= 0:
        return 0.0  # No edge

    # Apply Kelly fraction (conservative)
    f *= kelly_fraction

    # Cap at max position
    f = min(f, max_position_pct)

    return round(f * bankroll, 2)
